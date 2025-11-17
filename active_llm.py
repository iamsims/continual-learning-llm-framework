"""
Active Learning with LLM (Gemini) for Sample Selection
Uses Gemini to identify the most informative samples from unlabeled pool.
"""

import os
import pandas as pd
import numpy as np
from datasets import load_dataset
import google.generativeai as genai
from typing import List, Dict, Tuple
import json
import time
from tqdm import tqdm
import argparse

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# --- Configuration ---
LABELS_TO_KEEP = ['POLITICS', 'ENTERTAINMENT', 'TRAVEL', 'BUSINESS', 'SPORTS']
DATASET_NAME = 'heegyu/news-category-dataset'

# Gemini API configuration
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', None)


def setup_gemini(api_key: str = None):
    """
    Setup Gemini API client.

    Args:
        api_key: Gemini API key (if not provided, reads from environment)
    """
    if api_key is None:
        api_key = GEMINI_API_KEY

    if api_key is None:
        raise ValueError(
            "Gemini API key not found. Please set GEMINI_API_KEY environment variable "
            "or pass api_key parameter."
        )

    genai.configure(api_key=api_key)
    # Use gemini-1.5-flash-latest which is available in the API
    return genai.GenerativeModel('gemini-2.5-flash')


def load_and_filter_data(year: int = 2016):
    """
    Load and filter the news dataset for a specific year.

    Args:
        year: Year to filter data for

    Returns:
        DataFrame with filtered data
    """
    print(f"Loading dataset: '{DATASET_NAME}'...")
    full_dataset = load_dataset(DATASET_NAME, split='train')
    df_full = full_dataset.to_pandas()

    print(f"Original dataset size: {len(df_full)} rows")

    # Filter for target labels
    print(f"Filtering for {len(LABELS_TO_KEEP)} target labels...")
    df_filtered = df_full[df_full['category'].isin(LABELS_TO_KEEP)].copy()

    # Convert date and extract year
    df_filtered['date'] = pd.to_datetime(df_filtered['date'])
    df_filtered['year'] = df_filtered['date'].dt.year

    # Filter for specific year
    df_year = df_filtered[df_filtered['year'] == year].copy()
    df_year.reset_index(drop=True, inplace=True)

    print(f"\nData for year {year}: {len(df_year)} rows")
    print("\nCategory distribution:")
    print(df_year['category'].value_counts())

    return df_year


def create_sample_prompt(
    samples: List[Dict],
    current_train_distribution: Dict[str, int],
    num_to_select: int,
    pool_distribution: Dict[str, int] = None
) -> str:
    """
    Create a prompt for Gemini to select most informative samples.

    Args:
        samples: List of sample dictionaries with 'headline', 'description', 'index', 'category'
        current_train_distribution: Current distribution of categories in training set
        num_to_select: Number of samples to select
        pool_distribution: Overall distribution in the pool (for context)

    Returns:
        Formatted prompt string
    """
    prompt = f"""You are an expert in active learning for IMBALANCED text classification. Your task is to select the {num_to_select} most informative news article samples from the candidates below.

The classifier needs to distinguish between these categories:
- POLITICS
- ENTERTAINMENT
- TRAVEL
- BUSINESS
- SPORTS

**Current Training Set Distribution:**
{json.dumps(current_train_distribution, indent=2)}
"""

    if pool_distribution:
        prompt += f"""
**Overall Pool Distribution :**
{json.dumps(pool_distribution, indent=2)}
"""

    prompt += f"""
**Your Goal:** Select the MOST INFORMATIVE samples for training a robust news classifier.

**Selection Strategy (PRIORITY ORDER):**

1. **Informativeness & Uncertainty** (HIGHEST PRIORITY):
   - Select samples that are challenging, ambiguous, or information-rich
   - Prefer samples with nuanced content that will help the model learn better
   - Examples: articles that could fit multiple categories, complex stories, edge cases

2. **Diversity Within Categories**:
   - For each category, select diverse samples covering different topics, perspectives, writing styles
   - Avoid selecting multiple similar articles from the same category
   - Example POLITICS: elections, international affairs, policy, scandals, local vs national
   - Example SPORTS: different sports, business aspects, athlete profiles, championships

3. **Boundary/Hard Cases**:
   - Prioritize samples that blur category lines (e.g., political entertainment, sports business, travel politics)
   - These boundary examples help the classifier learn decision boundaries

4. **Quality**:
   - Prefer samples with detailed, descriptive content over vague headlines
   - Avoid very short or uninformative descriptions

**Distribution Awareness:**
- Be mindful of maintaining realistic category representation
- Don't force artificial balance - real-world data is naturally skewed
- Ensure minority classes get sufficient representation to be learnable

**Candidate Samples:**
"""

    for i, sample in enumerate(samples):
        prompt += f"\n[Sample {i}]\n"
        prompt += f"Headline: {sample['headline']}\n"
        prompt += f"Description: {sample['description']}\n"
        prompt += "---\n"

    prompt += f"""

Analyze these samples and select the {num_to_select} MOST INFORMATIVE ones following the strategy above.

Return your response as a JSON object with this exact format:
{{
    "selected_indices": [list of {num_to_select} selected sample indices (0 to {len(samples)-1})],
    "reasoning": "Explain why these samples are informative (focus on uncertainty, diversity, boundary cases)",
    "category_breakdown": {{"POLITICS": X, "ENTERTAINMENT": Y, "TRAVEL": Z, "BUSINESS": W, "SPORTS": V}}
}}

IMPORTANT: Return ONLY the JSON object, nothing else."""

    return prompt


def query_gemini_for_selection(
    model,
    samples: List[Dict],
    current_train_distribution: Dict[str, int],
    num_to_select: int,
    pool_distribution: Dict[str, int] = None,
    max_retries: int = 3
) -> Tuple[List[int], str, Dict[str, int]]:
    """
    Query Gemini to select most informative samples.

    Args:
        model: Gemini model instance
        samples: List of sample dictionaries
        current_train_distribution: Current category distribution
        num_to_select: Number of samples to select
        pool_distribution: Overall pool distribution
        max_retries: Maximum number of retries on failure

    Returns:
        Tuple of (selected_indices, reasoning, category_breakdown)
    """
    prompt = create_sample_prompt(samples, current_train_distribution, num_to_select, pool_distribution)

    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            response_text = response.text.strip()

            # Try to extract JSON from response
            # Sometimes the model wraps JSON in markdown code blocks
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_str = response_text.split("```")[1].split("```")[0].strip()
            else:
                json_str = response_text

            result = json.loads(json_str)

            selected_indices = result.get('selected_indices', [])
            reasoning = result.get('reasoning', 'No reasoning provided')
            category_breakdown = result.get('category_breakdown', {})

            # Validate indices
            if len(selected_indices) != num_to_select:
                print(f"Warning: Expected {num_to_select} indices, got {len(selected_indices)}")

            if max(selected_indices) >= len(samples) or min(selected_indices) < 0:
                raise ValueError(f"Invalid indices: {selected_indices}")

            return selected_indices, reasoning, category_breakdown

        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                # Fallback to random selection
                print("All attempts failed. Falling back to random selection.")
                selected_indices = np.random.choice(len(samples), num_to_select, replace=False).tolist()
                reasoning = "Random selection (LLM query failed)"
                category_breakdown = {}
                return selected_indices, reasoning, category_breakdown


def active_learning_with_llm(
    pool_data: pd.DataFrame,
    model,
    initial_samples_per_class: int = 10,
    samples_per_iteration: int = 20,
    num_iterations: int = 10,
    batch_size: int = 100, 
    random_state: int = 42
) -> Dict:
    """
    Perform active learning using LLM for sample selection.

    Args:
        pool_data: Pool of unlabeled data
        model: Gemini model instance
        initial_samples_per_class: Number of initial samples per class
        samples_per_iteration: Number of samples to select in each iteration
        num_iterations: Number of active learning iterations
        batch_size: Number of candidates to present to LLM at once
        random_state: Random seed

    Returns:
        Dictionary containing training data and selection history
    """
    np.random.seed(random_state)

    # Create a copy of the pool
    pool = pool_data.copy()
    pool['selected'] = False
    pool['iteration'] = -1

    # Initial random sampling (stratified)
    print("\n" + "="*60)
    print("Initial Sampling")
    print("="*60)

    initial_indices = []
    for category in LABELS_TO_KEEP:
        category_pool = pool[pool['category'] == category].index.tolist()
        if len(category_pool) >= initial_samples_per_class:
            selected = np.random.choice(category_pool, initial_samples_per_class, replace=False)
            initial_indices.extend(selected)

    pool.loc[initial_indices, 'selected'] = True
    pool.loc[initial_indices, 'iteration'] = 0

    print(f"Selected {len(initial_indices)} initial samples")
    print("\nInitial distribution:")
    print(pool[pool['selected']]['category'].value_counts())

    # Get overall pool distribution for context
    pool_distribution = pool['category'].value_counts().to_dict()

    # Active learning iterations
    selection_history = []

    for iteration in range(1, num_iterations + 1):
        print("\n" + "="*60)
        print(f"Active Learning Iteration {iteration}/{num_iterations}")
        print("="*60)

        # Get current training distribution
        current_train = pool[pool['selected']]
        current_distribution = current_train['category'].value_counts().to_dict()

        print(f"\nCurrent training set size: {len(current_train)}")
        print("Current distribution:")
        print(current_train['category'].value_counts())

        # Get remaining pool
        remaining_pool = pool[~pool['selected']]

        if len(remaining_pool) == 0:
            print("No more samples in pool!")
            break

        # Sample candidates from pool using RANDOM sampling (naturally reflects distribution)
        candidate_size = min(batch_size, len(remaining_pool))
        candidate_indices = np.random.choice(
            remaining_pool.index,
            candidate_size,
            replace=False
        )
        candidates = pool.loc[candidate_indices]

        # Show candidate distribution
        print(f"\nCandidate pool distribution ({len(candidates)} samples):")
        print(candidates['category'].value_counts())

        # Prepare samples for LLM (NOW INCLUDING CATEGORY LABELS)
        samples = []
        for idx, row in candidates.iterrows():
            samples.append({
                'index': idx,
                'headline': row['headline'],
                'description': row.get('short_description', ''),
                'category': row['category']  # NOW SHOWN TO LLM
            })

        # Query LLM for selection
        print(f"\nQuerying Gemini to select {samples_per_iteration} samples from {len(samples)} candidates...")

        num_to_select = min(samples_per_iteration, len(samples))
        selected_local_indices, reasoning, category_breakdown = query_gemini_for_selection(
            model,
            samples,
            current_distribution,
            num_to_select,
            pool_distribution
        )

        # Map local indices to global indices
        selected_global_indices = [samples[i]['index'] for i in selected_local_indices]

        # Update pool
        pool.loc[selected_global_indices, 'selected'] = True
        pool.loc[selected_global_indices, 'iteration'] = iteration

        # Store selection history
        selection_info = {
            'iteration': iteration,
            'selected_indices': selected_global_indices,
            'reasoning': reasoning,
            'selected_categories': pool.loc[selected_global_indices, 'category'].tolist(),
            'category_breakdown': category_breakdown
        }
        selection_history.append(selection_info)

        print(f"\nSelected {len(selected_global_indices)} samples")
        print(f"Reasoning: {reasoning}")
        if category_breakdown:
            print(f"LLM's planned breakdown: {category_breakdown}")
        print("\nActual selected categories:")
        print(pool.loc[selected_global_indices, 'category'].value_counts())

        # Small delay to avoid rate limiting
        time.sleep(1)

    # Final results
    print("\n" + "="*60)
    print("Active Learning Complete")
    print("="*60)

    final_train = pool[pool['selected']]
    print(f"\nFinal training set size: {len(final_train)}")
    print("\nFinal category distribution:")
    print(final_train['category'].value_counts())

    print("\nDistribution by iteration:")
    for i in range(num_iterations + 1):
        iter_data = pool[pool['iteration'] == i]
        if len(iter_data) > 0:
            print(f"\nIteration {i}: {len(iter_data)} samples")
            print(iter_data['category'].value_counts().to_dict())

    return {
        'pool_with_selections': pool,
        'training_data': final_train,
        'selection_history': selection_history,
        'final_distribution': final_train['category'].value_counts().to_dict()
    }


def save_results(results: Dict, output_dir: str = './active_learning_results'):
    """
    Save active learning results to files.

    Args:
        results: Results dictionary from active_learning_with_llm
        output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save training data
    train_file = f"{output_dir}/training_data.csv"
    results['training_data'].to_csv(train_file, index=False)
    print(f"\nTraining data saved to {train_file}")

    # Save full pool with selections
    pool_file = f"{output_dir}/pool_with_selections.csv"
    results['pool_with_selections'].to_csv(pool_file, index=False)
    print(f"Pool data with selections saved to {pool_file}")

    # Save selection history
    history_file = f"{output_dir}/selection_history.json"
    with open(history_file, 'w') as f:
        json.dump(results['selection_history'], f, indent=2)
    print(f"Selection history saved to {history_file}")

    # Save summary
    summary_file = f"{output_dir}/summary.json"
    summary = {
        'total_training_samples': len(results['training_data']),
        'final_distribution': results['final_distribution'],
        'num_iterations': len(results['selection_history'])
    }
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_file}")


def main():
    parser = argparse.ArgumentParser(description='Active Learning with LLM (Gemini)')
    parser.add_argument('--api_key', type=str, default=None,
                        help='Gemini API key (or set GEMINI_API_KEY env var)')
    parser.add_argument('--year', type=int, default=2016,
                        help='Year to filter data for')
    parser.add_argument('--initial_samples', type=int, default=10,
                        help='Initial samples per class')
    parser.add_argument('--samples_per_iteration', type=int, default=20,
                        help='Samples to select per iteration')
    parser.add_argument('--num_iterations', type=int, default=20,
                        help='Number of active learning iterations')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Number of candidates to show LLM at once')
    parser.add_argument('--output_dir', type=str, default='./active_learning_results',
                        help='Output directory for results')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    # Setup Gemini
    print("Setting up Gemini API...")
    model = setup_gemini(args.api_key)

    # Load data
    pool_data = load_and_filter_data(year=args.year)

    # Run active learning
    results = active_learning_with_llm(
        pool_data=pool_data,
        model=model,
        initial_samples_per_class=args.initial_samples,
        samples_per_iteration=args.samples_per_iteration,
        num_iterations=args.num_iterations,
        batch_size=args.batch_size,
        random_state=args.random_state
    )

    # Save results
    save_results(results, args.output_dir)

    print("\n" + "="*60)
    print("All done!")
    print("="*60)


if __name__ == '__main__':
    main()

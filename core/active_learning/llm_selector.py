"""
LLM-based Sample Selector

Uses Gemini or other LLMs to select the most informative samples for active learning.
"""

import os
import json
import time
import numpy as np
from typing import List, Dict, Tuple, Optional
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()


class LLMSelector:
    """
    LLM-based intelligent sample selector for active learning.

    Uses an LLM (Gemini) to analyze candidate samples and select the most
    informative ones based on uncertainty, diversity, and boundary cases.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = 'gemini-2.5-flash',
        max_retries: int = 3
    ):
        """
        Initialize the LLM selector.

        Args:
            api_key: Gemini API key (reads from GEMINI_API_KEY env var if None)
            model_name: Name of the Gemini model to use
            max_retries: Maximum number of retries on API failure
        """
        self.api_key = api_key or os.environ.get('GEMINI_API_KEY')
        self.model_name = model_name
        self.max_retries = max_retries
        self.model = None

        self._setup_model()

    def _setup_model(self):
        """Setup the Gemini model"""
        if self.api_key is None:
            raise ValueError(
                "Gemini API key not found. Please set GEMINI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)

    def create_selection_prompt(
        self,
        samples: List[Dict],
        current_train_distribution: Dict[str, int],
        num_to_select: int,
        pool_distribution: Optional[Dict[str, int]] = None,
        categories: Optional[List[str]] = None,
        text_fields: Optional[List[str]] = None
    ) -> str:
        """
        Create a prompt for the LLM to select informative samples.

        Args:
            samples: List of sample dictionaries with text content, 'index', and optionally 'category'
            current_train_distribution: Current distribution of categories in training set
            num_to_select: Number of samples to select
            pool_distribution: Overall distribution in the pool
            categories: List of category names
            text_fields: Names of text fields to display (e.g., ['text'], ['headline', 'description'])

        Returns:
            Formatted prompt string
        """
        if categories is None:
            categories = list(current_train_distribution.keys())

        if text_fields is None:
            # Auto-detect text fields from samples
            text_fields = [k for k in samples[0].keys() if k not in ['index', 'category']]

        prompt = f"""You are an expert in active learning for IMBALANCED text classification. Your task is to select the {num_to_select} most informative samples from the candidates below.

The classifier needs to distinguish between these categories:
{chr(10).join(f'- {cat}' for cat in categories)}

**Current Training Set Distribution:**
{json.dumps(current_train_distribution, indent=2)}
"""

        if pool_distribution:
            prompt += f"""
**Overall Pool Distribution:**
{json.dumps(pool_distribution, indent=2)}
"""

        prompt += f"""
**Your Goal:** Select the MOST INFORMATIVE samples for training a robust classifier.

**Selection Strategy (PRIORITY ORDER):**

1. **Informativeness & Uncertainty** (HIGHEST PRIORITY):
   - Select samples that are challenging, ambiguous, or information-rich
   - Prefer samples with nuanced content that will help the model learn better
   - Examples: samples that could fit multiple categories, complex content, edge cases

2. **Diversity Within Categories**:
   - For each category, select diverse samples covering different topics, perspectives, writing styles
   - Avoid selecting multiple similar samples from the same category

3. **Boundary/Hard Cases**:
   - Prioritize samples that blur category lines
   - These boundary examples help the classifier learn decision boundaries

4. **Quality**:
   - Prefer samples with detailed, descriptive content
   - Avoid very short or uninformative text

**Distribution Awareness:**
- Be mindful of maintaining realistic category representation
- Don't force artificial balance - real-world data is naturally skewed
- Ensure minority classes get sufficient representation to be learnable

**Candidate Samples:**
"""

        for i, sample in enumerate(samples):
            prompt += f"\n[Sample {i}]\n"
            # Display all text fields dynamically
            for field in text_fields:
                if field in sample and sample[field]:
                    # Capitalize field name for display
                    field_display = field.replace('_', ' ').title()
                    prompt += f"{field_display}: {sample[field]}\n"
            if 'category' in sample:
                prompt += f"Category: {sample['category']}\n"
            prompt += "---\n"

        prompt += f"""

Analyze these samples and select the {num_to_select} MOST INFORMATIVE ones following the strategy above.

Return your response as a JSON object with this exact format:
{{
    "selected_indices": [list of {num_to_select} selected sample indices (0 to {len(samples)-1})],
    "reasoning": "Explain why these samples are informative (focus on uncertainty, diversity, boundary cases)",
    "category_breakdown": {{{', '.join(f'"{cat}": X' for cat in categories)}}}
}}

IMPORTANT: Return ONLY the JSON object, nothing else."""

        return prompt

    def select_samples(
        self,
        samples: List[Dict],
        current_train_distribution: Dict[str, int],
        num_to_select: int,
        pool_distribution: Optional[Dict[str, int]] = None,
        categories: Optional[List[str]] = None,
        text_fields: Optional[List[str]] = None
    ) -> Tuple[List[int], str, Dict[str, int]]:
        """
        Query the LLM to select the most informative samples.

        Args:
            samples: List of sample dictionaries
            current_train_distribution: Current category distribution
            num_to_select: Number of samples to select
            pool_distribution: Overall pool distribution
            categories: List of category names
            text_fields: Names of text fields to display

        Returns:
            Tuple of (selected_indices, reasoning, category_breakdown)
        """
        prompt = self.create_selection_prompt(
            samples,
            current_train_distribution,
            num_to_select,
            pool_distribution,
            categories,
            text_fields
        )

        for attempt in range(self.max_retries):
            try:
                response = self.model.generate_content(prompt)
                response_text = response.text.strip()

                # Extract JSON from response
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
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    # Fallback to random selection
                    print("All attempts failed. Falling back to random selection.")
                    selected_indices = np.random.choice(
                        len(samples), num_to_select, replace=False
                    ).tolist()
                    reasoning = "Random selection (LLM query failed)"
                    category_breakdown = {}
                    return selected_indices, reasoning, category_breakdown

    def __repr__(self) -> str:
        return f"LLMSelector(model_name={self.model_name}, max_retries={self.max_retries})"

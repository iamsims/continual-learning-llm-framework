"""
Active Learning orchestrator.

Manages the active learning loop with different selection strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import json
import time
from pathlib import Path

from .llm_selector import LLMSelector


class ActiveLearner:
    """
    Main active learning orchestrator using LLM-based sample selection.

    Manages the complete active learning pipeline including:
    - Initial sampling
    - Iterative LLM-based sample selection
    - Selection history tracking
    - Results saving
    """

    def __init__(
        self,
        llm_selector: LLMSelector,
        random_state: int = 42
    ):
        """
        Initialize the active learner.

        Args:
            llm_selector: LLM selector instance for intelligent sample selection
            random_state: Random seed for initial sampling
        """
        self.llm_selector = llm_selector
        self.random_state = random_state

        self.selection_history = []
        self.pool = None

    def initial_sampling(
        self,
        pool_data: pd.DataFrame,
        samples_per_class: int,
        category_column: str = 'category',
        categories: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Perform initial stratified sampling.

        Args:
            pool_data: Pool of data to sample from
            samples_per_class: Number of samples per class
            category_column: Name of the category column
            categories: List of categories to sample from

        Returns:
            DataFrame with selection markers added
        """
        np.random.seed(self.random_state)

        self.pool = pool_data.copy()
        self.pool['selected'] = False
        self.pool['iteration'] = -1

        if categories is None:
            categories = self.pool[category_column].unique().tolist()

        print("="*60)
        print("Initial Sampling")
        print("="*60)

        initial_indices = []
        for category in categories:
            category_pool = self.pool[self.pool[category_column] == category].index.tolist()
            if len(category_pool) >= samples_per_class:
                selected = np.random.choice(category_pool, samples_per_class, replace=False)
                initial_indices.extend(selected)

        self.pool.loc[initial_indices, 'selected'] = True
        self.pool.loc[initial_indices, 'iteration'] = 0

        print(f"Selected {len(initial_indices)} initial samples")
        print("\nInitial distribution:")
        print(self.pool[self.pool['selected']][category_column].value_counts())

        return self.pool

    def run_iteration(
        self,
        iteration: int,
        samples_per_iteration: int,
        batch_size: int,
        category_column: str = 'category',
        text_columns: List[str] = None
    ) -> Dict:
        """
        Run a single active learning iteration.

        Args:
            iteration: Iteration number
            samples_per_iteration: Number of samples to select
            batch_size: Number of candidates to consider
            category_column: Name of the category column
            text_columns: Names of text columns to use (e.g., ['text'] or ['headline', 'description'])

        Returns:
            Dictionary with selection information
        """
        if text_columns is None:
            text_columns = ['text']

        print("\n" + "="*60)
        print(f"Active Learning Iteration {iteration}")
        print("="*60)

        # Get current training distribution
        current_train = self.pool[self.pool['selected']]
        current_distribution = current_train[category_column].value_counts().to_dict()

        print(f"\nCurrent training set size: {len(current_train)}")
        print("Current distribution:")
        print(current_train[category_column].value_counts())

        # Get remaining pool
        remaining_pool = self.pool[~self.pool['selected']]

        if len(remaining_pool) == 0:
            print("No more samples in pool!")
            return None

        # Sample candidates
        candidate_size = min(batch_size, len(remaining_pool))
        candidate_indices = np.random.choice(
            remaining_pool.index,
            candidate_size,
            replace=False
        )
        candidates = self.pool.loc[candidate_indices]

        print(f"\nCandidate pool distribution ({len(candidates)} samples):")
        print(candidates[category_column].value_counts())

        # Prepare samples for selection
        samples = []
        for idx, row in candidates.iterrows():
            sample = {'index': idx}
            # Add all text columns to sample
            for col in text_columns:
                if col in row:
                    sample[col] = row[col]
            # Add category if present
            if category_column in row:
                sample['category'] = row[category_column]
            samples.append(sample)

        # Select samples using LLM
        num_to_select = min(samples_per_iteration, len(samples))
        pool_distribution = self.pool[category_column].value_counts().to_dict()

        print(f"\nQuerying LLM to select {num_to_select} samples from {len(samples)} candidates...")

        selected_local_indices, reasoning, category_breakdown = self.llm_selector.select_samples(
            samples,
            current_distribution,
            num_to_select,
            pool_distribution,
            text_fields=text_columns
        )

        # Map local indices to global indices
        selected_global_indices = [samples[i]['index'] for i in selected_local_indices]

        # Update pool
        self.pool.loc[selected_global_indices, 'selected'] = True
        self.pool.loc[selected_global_indices, 'iteration'] = iteration

        # Store selection history
        selection_info = {
            'iteration': iteration,
            'selected_indices': selected_global_indices,
            'reasoning': reasoning,
            'selected_categories': self.pool.loc[selected_global_indices, category_column].tolist(),
            'category_breakdown': category_breakdown
        }
        self.selection_history.append(selection_info)

        print(f"\nSelected {len(selected_global_indices)} samples")
        print(f"Reasoning: {reasoning}")
        if category_breakdown:
            print(f"Category breakdown: {category_breakdown}")
        print("\nActual selected categories:")
        print(self.pool.loc[selected_global_indices, category_column].value_counts())

        # Small delay for API rate limiting
        time.sleep(1)

        return selection_info

    def run(
        self,
        pool_data: pd.DataFrame,
        initial_samples_per_class: int = 10,
        samples_per_iteration: int = 20,
        num_iterations: int = 10,
        batch_size: int = 100,
        category_column: str = 'category',
        text_columns: List[str] = None
    ) -> Dict:
        """
        Run the complete active learning pipeline.

        Args:
            pool_data: Pool of unlabeled data
            initial_samples_per_class: Number of initial samples per class
            samples_per_iteration: Number of samples to select in each iteration
            num_iterations: Number of active learning iterations
            batch_size: Number of candidates to consider at once
            category_column: Name of the category column
            text_columns: Names of text columns to use (e.g., ['text'] or ['headline', 'description'])
                         If None, defaults to ['text']

        Returns:
            Dictionary containing training data and selection history
        """
        if text_columns is None:
            text_columns = ['text']
        # Initial sampling
        self.initial_sampling(pool_data, initial_samples_per_class, category_column)

        # Active learning iterations
        for iteration in range(1, num_iterations + 1):
            result = self.run_iteration(
                iteration,
                samples_per_iteration,
                batch_size,
                category_column,
                text_columns
            )

            if result is None:
                break

        # Final results
        print("\n" + "="*60)
        print("Active Learning Complete")
        print("="*60)

        final_train = self.pool[self.pool['selected']]
        print(f"\nFinal training set size: {len(final_train)}")
        print("\nFinal category distribution:")
        print(final_train[category_column].value_counts())

        print("\nDistribution by iteration:")
        for i in range(num_iterations + 1):
            iter_data = self.pool[self.pool['iteration'] == i]
            if len(iter_data) > 0:
                print(f"\nIteration {i}: {len(iter_data)} samples")
                print(iter_data[category_column].value_counts().to_dict())

        return {
            'pool_with_selections': self.pool,
            'training_data': final_train,
            'selection_history': self.selection_history,
            'final_distribution': final_train[category_column].value_counts().to_dict()
        }

    def save_results(self, output_dir: Union[str, Path]):
        """
        Save active learning results to files.

        Args:
            output_dir: Directory to save results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.pool is None:
            raise ValueError("No results to save. Run active learning first.")

        # Save training data
        train_file = output_dir / "training_data.csv"
        training_data = self.pool[self.pool['selected']]
        training_data.to_csv(train_file, index=False)
        print(f"\nTraining data saved to {train_file}")

        # Save full pool with selections
        pool_file = output_dir / "pool_with_selections.csv"
        self.pool.to_csv(pool_file, index=False)
        print(f"Pool data with selections saved to {pool_file}")

        # Save selection history
        history_file = output_dir / "selection_history.json"
        with open(history_file, 'w') as f:
            json.dump(self.selection_history, f, indent=2)
        print(f"Selection history saved to {history_file}")

        # Save summary
        summary_file = output_dir / "summary.json"
        summary = {
            'total_training_samples': len(training_data),
            'final_distribution': training_data['category'].value_counts().to_dict() if 'category' in training_data.columns else {},
            'num_iterations': len(self.selection_history),
            'strategy': 'llm_based'
        }
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to {summary_file}")

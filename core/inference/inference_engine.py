"""
Inference Engine for BERT-based Text Classification

This module provides a high-level inference engine that takes a pandas DataFrame
and returns predictions with probabilities, confidence scores, and predicted labels.
"""

import torch
import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Union
from pathlib import Path
from transformers import BertTokenizer, BertForSequenceClassification
from tqdm import tqdm
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InferenceEngine:
    """
    High-level inference engine for BERT-based text classification.

    Features:
    - Load trained models from checkpoint
    - Batch prediction with progress tracking
    - Return probabilities, confidence scores, and predicted labels
    - Seamless integration with pandas DataFrames
    - GPU acceleration support

    Example:
        >>> engine = InferenceEngine(model_path='path/to/model')
        >>> df = pd.DataFrame({'text': ['Sample text 1', 'Sample text 2']})
        >>> result_df = engine.predict(df, text_column='text')
        >>> print(result_df[['text', 'predicted_label', 'confidence']])
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        device: Optional[str] = None,
        max_length: int = 512,
        batch_size: int = 16
    ):
        """
        Initialize the inference engine.

        Args:
            model_path: Path to the trained model checkpoint directory
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            max_length: Maximum sequence length for tokenization
            batch_size: Batch size for inference
        """
        self.model_path = Path(model_path)
        self.device = device or self._get_device()
        self.max_length = max_length
        self.batch_size = batch_size

        logger.info(f"Initializing InferenceEngine with model: {self.model_path}")
        logger.info(f"Device: {self.device}")

        # Load model, tokenizer, and label mappings
        self.model = None
        self.tokenizer = None
        self.label2id = None
        self.id2label = None

        self._load_model()

    def _get_device(self) -> str:
        """Auto-detect the best available device"""
        if torch.cuda.is_available():
            return 'cuda'
        else:
            return 'cpu'

    def _load_model(self):
        """Load the model, tokenizer, and label mappings"""
        try:
            logger.info(f"Loading model from {self.model_path}...")

            # Load tokenizer
            self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
            logger.info(f"Loaded tokenizer")

            # Load model
            self.model = BertForSequenceClassification.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()

            # Load label mappings
            label_mappings_path = self.model_path / "label_mappings.json"
            if label_mappings_path.exists():
                with open(label_mappings_path, 'r') as f:
                    mappings = json.load(f)
                    self.label2id = mappings['label2id']
                    self.id2label = {int(k): v for k, v in mappings['id2label'].items()}
            else:
                # Fallback to model config
                self.label2id = self.model.config.label2id
                self.id2label = self.model.config.id2label

            logger.info(f"Model loaded successfully on {self.device}")
            logger.info(f"Number of classes: {len(self.id2label)}")
            logger.info(f"Classes: {list(self.id2label.values())}")

        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

    def predict(
        self,
        df: pd.DataFrame,
        text_column: str = 'text',
        include_probabilities: bool = False,
        show_progress: bool = True
    ) -> pd.DataFrame:
        """
        Perform inference on a DataFrame and add prediction columns.

        Args:
            df: Input DataFrame containing text data
            text_column: Name of the column containing text
            include_probabilities: Whether to include probability columns for all classes
            show_progress: Whether to show progress bar

        Returns:
            DataFrame with prediction columns added:
            - predicted_label_id: The predicted class ID
            - predicted_label: The predicted class label
            - confidence: Confidence score (max probability)
            - prob_{class_name}: Probability for each class (if include_probabilities=True)
        """
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame. Available columns: {df.columns.tolist()}")

        # Make a copy to avoid modifying the original
        result_df = df.copy()

        # Drop rows with missing text
        initial_size = len(result_df)
        result_df = result_df.dropna(subset=[text_column])
        if len(result_df) < initial_size:
            logger.warning(f"Dropped {initial_size - len(result_df)} rows with missing text")

        # Extract texts
        texts = result_df[text_column].astype(str).tolist()
        logger.info(f"Running inference on {len(texts)} samples...")

        # Run batch inference
        predictions, probabilities = self._run_batch_inference(
            texts=texts,
            show_progress=show_progress
        )

        # Add predictions to dataframe
        result_df['predicted_label_id'] = predictions
        result_df['predicted_label'] = [self.id2label[int(pred)] for pred in predictions]
        result_df['confidence'] = np.max(probabilities, axis=1)

        # Add all class probabilities if requested
        if include_probabilities:
            for label_id, label_name in self.id2label.items():
                # Sanitize column name
                col_name = f'prob_{label_name.replace(" ", "_").replace("&", "and")}'
                result_df[col_name] = probabilities[:, label_id]

        logger.info("Inference completed successfully")

        return result_df

    def _run_batch_inference(
        self,
        texts: List[str],
        show_progress: bool = True
    ) -> tuple:
        """
        Run batch inference on a list of texts.

        Args:
            texts: List of text strings to classify
            show_progress: Whether to show progress bar

        Returns:
            predictions, probabilities: Arrays of predicted labels and probabilities
        """
        all_predictions = []
        all_probabilities = []

        # Process in batches
        num_batches = (len(texts) + self.batch_size - 1) // self.batch_size

        with torch.no_grad():
            iterator = range(0, len(texts), self.batch_size)
            if show_progress:
                iterator = tqdm(iterator, total=num_batches, desc="Running inference")

            for i in iterator:
                batch_texts = texts[i:i + self.batch_size]

                # Tokenize batch
                encodings = self.tokenizer(
                    batch_texts,
                    padding='max_length',
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                )

                # Move to device
                input_ids = encodings['input_ids'].to(self.device)
                attention_mask = encodings['attention_mask'].to(self.device)

                # Get predictions
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                # Get probabilities and predictions
                probs = torch.softmax(logits, dim=-1)
                predictions = torch.argmax(logits, dim=-1)

                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probs.cpu().numpy())

        return np.array(all_predictions), np.array(all_probabilities)

    def predict_single(
        self,
        text: str,
        return_probabilities: bool = True
    ) -> Dict:
        """
        Predict a single text sample.

        Args:
            text: Text string to classify
            return_probabilities: Whether to include probability distribution

        Returns:
            Dictionary with prediction results:
            - predicted_label: The predicted class label
            - predicted_label_id: The predicted class ID
            - confidence: Confidence score
            - probabilities: Dict of class probabilities (if return_probabilities=True)
        """
        predictions, probabilities = self._run_batch_inference(
            texts=[text],
            show_progress=False
        )

        result = {
            'predicted_label_id': int(predictions[0]),
            'predicted_label': self.id2label[int(predictions[0])],
            'confidence': float(np.max(probabilities[0]))
        }

        if return_probabilities:
            result['probabilities'] = {
                label: float(probabilities[0][label_id])
                for label_id, label in self.id2label.items()
            }

        return result

    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model.

        Returns:
            Dictionary containing model metadata
        """
        return {
            'model_path': str(self.model_path),
            'device': self.device,
            'num_classes': len(self.id2label),
            'classes': list(self.id2label.values()),
            'id2label': self.id2label,
            'label2id': self.label2id,
            'max_length': self.max_length,
            'batch_size': self.batch_size,
            'model_type': self.model.config.model_type if hasattr(self.model.config, 'model_type') else 'bert'
        }

    def __repr__(self) -> str:
        return (
            f"InferenceEngine(\n"
            f"  model_path={self.model_path},\n"
            f"  device={self.device},\n"
            f"  num_classes={len(self.id2label)},\n"
            f"  classes={list(self.id2label.values())}\n"
            f")"
        )

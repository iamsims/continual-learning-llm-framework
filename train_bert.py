"""
BERT Training Script for News Category Classification
Trains a BERT model on the filtered news dataset with 5 categories.
"""

import torch
import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import argparse
from datetime import datetime
import random


# set cuda visible devices
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# --- Configuration ---
LABELS_TO_KEEP = ['POLITICS', 'ENTERTAINMENT', 'TRAVEL', 'BUSINESS', 'SPORTS']
DATASET_NAME = 'heegyu/news-category-dataset'

# Create label to id mapping
LABEL2ID = {label: idx for idx, label in enumerate(LABELS_TO_KEEP)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}


def load_and_prepare_data(test_size=0.2, val_size=0.1, random_state=42, filter_year=None, csv_path=None):
    """
    Load and prepare the dataset for training.

    Args:
        test_size: Proportion of data for test set
        val_size: Proportion of training data for validation set
        random_state: Random seed for reproducibility
        filter_year: If specified, only keep data from this year (e.g., 2022)
        csv_path: If specified, load data from CSV instead of HuggingFace dataset

    Returns:
        train_df, val_df, test_df: DataFrames for training, validation, and testing
    """
    # Load data from CSV or HuggingFace
    if csv_path is not None:
        print(f"Loading dataset from CSV: '{csv_path}'...")
        df_filtered = pd.read_csv(csv_path)
        print(f"Loaded dataset size: {len(df_filtered)} rows")
    else:
        print(f"Loading dataset from HuggingFace: '{DATASET_NAME}'...")
        full_dataset = load_dataset(DATASET_NAME, split='train')
        df_full = full_dataset.to_pandas()

        print(f"Original dataset size: {len(df_full)} rows")

        # Filter for target labels
        print(f"Filtering for {len(LABELS_TO_KEEP)} target labels...")
        df_filtered = df_full[df_full['category'].isin(LABELS_TO_KEEP)].copy()
        df_filtered.reset_index(drop=True, inplace=True)

        print(f"After label filtering: {len(df_filtered)} rows")

        # Filter by year if specified
        if filter_year is not None:
            print(f"\nFiltering for year {filter_year}...")
            # Convert date column to datetime if not already
            df_filtered['date'] = pd.to_datetime(df_filtered['date'])
            df_filtered['year'] = df_filtered['date'].dt.year

            # Show year distribution before filtering
            print(f"Year distribution before filtering:")
            print(df_filtered['year'].value_counts().sort_index())

            # Filter for the specified year
            df_filtered = df_filtered[df_filtered['year'] == filter_year].copy()
            df_filtered.reset_index(drop=True, inplace=True)

            print(f"After year filtering: {len(df_filtered)} rows")

    print(f"\nFinal dataset size: {len(df_filtered)} rows")
    print("\nCategory distribution:")
    category_counts = df_filtered['category'].value_counts()
    print(category_counts)

    # Check if all categories are present
    missing_categories = set(LABELS_TO_KEEP) - set(category_counts.index)
    if missing_categories:
        print(f"\nWARNING: The following categories are missing from the filtered data: {missing_categories}")
        print("Consider using a different year or removing the year filter.")

    # Check if any category has too few samples for stratified split
    min_samples_needed = int(1 / test_size) + int(1 / val_size) + 1
    categories_with_few_samples = category_counts[category_counts < min_samples_needed]
    if len(categories_with_few_samples) > 0:
        print(f"\nWARNING: Some categories have very few samples and may cause issues with stratified splitting:")
        print(categories_with_few_samples)

    # Create text column (combine headline and short_description)
    df_filtered['text'] = df_filtered['headline'] + " " + df_filtered['short_description'].fillna("")

    # Map labels to integers
    df_filtered['label'] = df_filtered['category'].map(LABEL2ID)

    # Try stratified split, fall back to regular split if it fails
        # Split into train and test
    train_val_df, test_df = train_test_split(
        df_filtered,
        test_size=test_size,
        random_state=random_state,
        stratify=df_filtered['label']
    )
    
    # Split train into train and validation
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size,
        random_state=random_state,
        stratify=train_val_df['label']
    )
    print("\nUsing stratified splitting")


    print(f"\nDataset splits:")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Validation: {len(val_df)} samples")
    print(f"  Test: {len(test_df)} samples")

    # Show category distribution in each split
    print("\nTrain set category distribution:")
    print(train_df['category'].value_counts())
    print("\nValidation set category distribution:")
    print(val_df['category'].value_counts())
    print("\nTest set category distribution:")
    print(test_df['category'].value_counts())

    return train_df, val_df, test_df


def tokenize_data(df, tokenizer, max_length=512):
    """
    Tokenize the text data.

    Args:
        df: DataFrame with 'text' and 'label' columns
        tokenizer: BERT tokenizer
        max_length: Maximum sequence length

    Returns:
        Hugging Face Dataset object
    """
    # Create dataset from dataframe
    dataset = Dataset.from_pandas(df[['text', 'label']])

    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=max_length
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(['text'])
    tokenized_dataset.set_format('torch')

    return tokenized_dataset


def compute_metrics(pred):
    """
    Compute evaluation metrics.

    Args:
        pred: Predictions from the model

    Returns:
        Dictionary of metrics
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted', zero_division=0
    )
    acc = accuracy_score(labels, preds)

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def train_bert_model(
    model_name='bert-base-uncased',
    output_dir='./results',
    num_epochs=10,
    batch_size=8,
    learning_rate=2e-5,
    max_length=512,
    warmup_steps=500,
    weight_decay=0.01,
    early_stopping_patience=3,
    test_size=0.2,
    val_size=0.1,
    random_state=42,
    save_model=True,
    filter_year=None,
    csv_path=None,
    dropout_rate=0.3,
    use_class_weights=False,
    gradient_accumulation_steps=2
):
    """
    Main training function for BERT model.

    Args:
        model_name: Pretrained BERT model name
        output_dir: Directory to save model and results
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        max_length: Maximum sequence length
        warmup_steps: Number of warmup steps
        weight_decay: Weight decay for optimizer
        early_stopping_patience: Patience for early stopping
        test_size: Proportion of data for test set
        val_size: Proportion of training data for validation
        random_state: Random seed
        save_model: Whether to save the final model
        filter_year: If specified, only use data from this year (e.g., 2022)
        csv_path: If specified, load data from CSV instead of HuggingFace dataset
        dropout_rate: Dropout probability for regularization (default: 0.3)
        use_class_weights: Whether to use class weights to handle imbalance
        gradient_accumulation_steps: Number of steps to accumulate gradients
    """
    # Set random seeds for reproducibility
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
    print(f"\nUsing device: {device}")
    print(f"Number of devices available: {num_devices}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        for i in range(num_devices):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")

    # Load and prepare data
    train_df, val_df, test_df = load_and_prepare_data(
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
        filter_year=filter_year,
        csv_path=csv_path
    )

    # Initialize tokenizer and model
    print(f"\nLoading tokenizer and model: {model_name}")
    tokenizer = BertTokenizer.from_pretrained(model_name)

    # Load model with custom dropout
    from transformers import BertConfig
    config = BertConfig.from_pretrained(model_name)
    config.num_labels = len(LABELS_TO_KEEP)
    config.id2label = ID2LABEL
    config.label2id = LABEL2ID
    config.hidden_dropout_prob = dropout_rate
    config.attention_probs_dropout_prob = dropout_rate
    config.classifier_dropout = dropout_rate

    model = BertForSequenceClassification.from_pretrained(
        model_name,
        config=config
    )

    # Calculate class weights if requested
    class_weights = None
    if use_class_weights:
        from sklearn.utils.class_weight import compute_class_weight
        class_weights_array = compute_class_weight(
            'balanced',
            classes=np.unique(train_df['label']),
            y=train_df['label']
        )
        class_weights = torch.tensor(class_weights_array, dtype=torch.float).to(device)
        print(f"\nUsing class weights: {class_weights}")

        # Create custom trainer with weighted loss
        from torch import nn

        class WeightedLossTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False):
                labels = inputs.pop("labels")
                outputs = model(**inputs)
                logits = outputs.logits
                loss_fct = nn.CrossEntropyLoss(weight=class_weights)
                loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
                return (loss, outputs) if return_outputs else loss

        TrainerClass = WeightedLossTrainer
    else:
        TrainerClass = Trainer

    # Tokenize datasets
    print("\nTokenizing datasets...")
    train_dataset = tokenize_data(train_df, tokenizer, max_length)
    val_dataset = tokenize_data(val_df, tokenizer, max_length)
    test_dataset = tokenize_data(test_df, tokenizer, max_length)

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{output_dir}/bert_{timestamp}"

    # Define training arguments with regularization
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        learning_rate=learning_rate,
        logging_dir=f'{output_dir}/logs',
        logging_steps=50,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        greater_is_better=True,
        save_total_limit=2,
        push_to_hub=False,
        report_to='none',  # Disable wandb/tensorboard
        seed=random_state,
        # Regularization techniques
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_grad_norm=1.0,  # Gradient clipping
        lr_scheduler_type='cosine',  # Cosine learning rate schedule
        warmup_ratio=0.1,  # 10% warmup
        # Data efficiency
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        # Evaluation
        eval_accumulation_steps=1,
    )

    # Calculate and display actual batch size
    actual_batch_size = batch_size * num_devices * gradient_accumulation_steps
    print(f"\n{'='*50}")
    print("Batch Size Configuration:")
    print(f"{'='*50}")
    print(f"Per-device batch size: {batch_size}")
    print(f"Number of devices: {num_devices}")
    print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"Actual total batch size: {actual_batch_size}")
    print(f"{'='*50}")

    # Initialize Trainer
    trainer = TrainerClass(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)]
    )

    # Train the model
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50)
    train_result = trainer.train()

    # Evaluate on validation set
    print("\n" + "="*50)
    print("Evaluating on validation set...")
    print("="*50)
    val_metrics = trainer.evaluate()
    print("\nValidation Metrics:")
    for key, value in val_metrics.items():
        print(f"  {key}: {value:.4f}")

    # Evaluate on test set
    print("\n" + "="*50)
    print("Evaluating on test set...")
    print("="*50)
    test_metrics = trainer.evaluate(test_dataset)
    print("\nTest Metrics:")
    for key, value in test_metrics.items():
        print(f"  {key}: {value:.4f}")

    # Get predictions for detailed classification report
    predictions = trainer.predict(test_dataset)
    preds = predictions.predictions.argmax(-1)
    labels = predictions.label_ids

    # Get unique labels present in test set
    unique_labels = sorted(np.unique(labels))
    labels_present = [ID2LABEL[label_id] for label_id in unique_labels]

    print("\n" + "="*50)
    print("Classification Report (Test Set):")
    print("="*50)

    # Use labels parameter to specify which labels are actually present
    print(classification_report(
        labels,
        preds,
        labels=unique_labels,
        target_names=labels_present,
        digits=4,
        zero_division=0
    ))

    # Show which labels are missing from test set if any
    missing_from_test = set(LABELS_TO_KEEP) - set(labels_present)
    if missing_from_test:
        print(f"\nNote: The following categories were not present in the test set: {missing_from_test}")

    # Save the model and tokenizer
    if save_model:
        final_model_dir = f"{output_dir}/final_model"
        print(f"\nSaving model to {final_model_dir}")
        trainer.save_model(final_model_dir)
        tokenizer.save_pretrained(final_model_dir)

        # Save label mappings
        import json
        with open(f"{final_model_dir}/label_mappings.json", 'w') as f:
            json.dump({
                'label2id': LABEL2ID,
                'id2label': ID2LABEL,
                'labels': LABELS_TO_KEEP
            }, f, indent=2)

        print(f"Model, tokenizer, and label mappings saved to {final_model_dir}")

    # Save test predictions
    results_df = test_df.copy()
    results_df['predicted_label_id'] = preds
    results_df['predicted_category'] = [ID2LABEL[pred] for pred in preds]
    results_df['correct'] = results_df['label'] == results_df['predicted_label_id']

    results_file = f"{output_dir}/test_predictions.csv"
    results_df.to_csv(results_file, index=False)
    print(f"\nTest predictions saved to {results_file}")

    return trainer, train_result, test_metrics


def main():
    parser = argparse.ArgumentParser(description='Train BERT model for news classification')
    parser.add_argument('--model_name', type=str, default='bert-base-uncased',
                        help='Pretrained BERT model name')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Output directory for model and results')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--max_length', type=int, default=512,
                        help='Maximum sequence length')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Proportion of data for test set')
    parser.add_argument('--val_size', type=float, default=0.1,
                        help='Proportion of training data for validation')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--early_stopping_patience', type=int, default=3,
                        help='Early stopping patience')
    parser.add_argument('--filter_year', type=int, default=None,
                        help='Filter data by year (e.g., 2022). If not specified, use all years.')
    parser.add_argument('--csv_path', type=str, default=None,
                        help='Path to CSV file with training data. If specified, data will be loaded from CSV instead of HuggingFace.')
    parser.add_argument('--no_save', action='store_true',
                        help='Do not save the trained model')
    parser.add_argument('--dropout_rate', type=float, default=0.3,
                        help='Dropout rate for regularization (default: 0.3)')
    parser.add_argument('--use_class_weights', action='store_true',
                        help='Use class weights to handle imbalanced data')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2,
                        help='Number of gradient accumulation steps (default: 2)')

    args = parser.parse_args()

    # Train the model
    train_bert_model(
        model_name=args.model_name,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state,
        early_stopping_patience=args.early_stopping_patience,
        filter_year=args.filter_year,
        csv_path=args.csv_path,
        save_model=not args.no_save,
        dropout_rate=args.dropout_rate,
        use_class_weights=args.use_class_weights,
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )


if __name__ == '__main__':
    main()

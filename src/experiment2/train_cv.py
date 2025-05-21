#!/usr/bin/env python
# train_cv.py - Cross-validation training for TREC CAR and TREC ROBUST datasets
import torch
import torch.optim as optim
import torch.nn as nn
import os
import json
from tqdm import tqdm
import time
import random
import numpy as np
import argparse

import config
from models import get_model
from data_loader import load_embeddings_and_mappings
from evaluate import evaluate_model_on_dev
from torch.utils.data import Dataset, DataLoader
from utils import load_folds, setup_logging

def ensure_tensor_dtype(tensor, dtype=torch.float32):
    """Ensure tensor is of the specified data type."""
    if isinstance(tensor, torch.Tensor) and tensor.dtype != dtype:
        return tensor.to(dtype)
    return tensor


def load_training_triples(dataset_name, fold_idx, triples_dir=None):
    """
    Load pre-created training triples for a specific fold.
    """
    if triples_dir is None:
        triples_dir = 'data/cv_triples'

    # Fix the path construction to avoid duplicate dataset names
    # Check if dataset name is already the last component of the path
    path_components = os.path.normpath(triples_dir).split(os.sep)
    if path_components and path_components[-1] == dataset_name:
        # Dataset name already in path, don't add it again
        triples_path = os.path.join(triples_dir, f"fold_{fold_idx}_triples.pt")
    else:
        # Dataset name not in path, add it
        triples_path = os.path.join(triples_dir, dataset_name, f"fold_{fold_idx}_triples.pt")

    if not os.path.exists(triples_path):
        raise FileNotFoundError(f"Training triples file not found: {triples_path}. "
                                f"Please run create_cv_triples.py first.")

    print(f"Loading training triples from {triples_path}")
    training_triples = torch.load(triples_path)
    print(f"Loaded {len(training_triples)} training triples")

    return training_triples


def load_test_data(dataset_name, fold_idx, triples_dir=None):
    """
    Load pre-created test data for a specific fold.

    Args:
        dataset_name: Name of the dataset (car or robust)
        fold_idx: Fold index to use
        triples_dir: Directory containing triples files (default: data/cv_triples)

    Returns:
        dict: Mapping from query IDs to lists of candidate passage IDs
    """
    if triples_dir is None:
        triples_dir = 'data/cv_triples'

    # Fix the path construction to avoid duplicate dataset names
    # Check if dataset name is already the last component of the path
    path_components = os.path.normpath(triples_dir).split(os.sep)
    if path_components and path_components[-1] == dataset_name:
        # Dataset name already in path, don't add it again
        test_data_path = os.path.join(triples_dir, f"fold_{fold_idx}_test_data.json")
    else:
        # Dataset name not in path, add it
        test_data_path = os.path.join(triples_dir, dataset_name, f"fold_{fold_idx}_test_data.json")

    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"Test data file not found: {test_data_path}. "
                                f"Please run create_cv_triples.py first.")

    print(f"Loading test data from {test_data_path}")
    with open(test_data_path, 'r') as f:
        test_data = json.load(f)

    print(f"Loaded test data for {len(test_data)} queries")

    return test_data


def create_dataloader_from_triples(training_triples, query_embeddings, passage_embeddings, batch_size=32):
    """
    Create a DataLoader from pre-created training triples.

    Args:
        training_triples: List of training triples (query_idx, pos_idx, neg_idx)
        query_embeddings: Query embeddings tensor
        passage_embeddings: Passage embeddings tensor
        batch_size: Batch size for dataloader

    Returns:
        torch.utils.data.DataLoader: DataLoader for training
    """
    # First, ensure embeddings are float32
    query_embeddings = ensure_tensor_dtype(query_embeddings)
    passage_embeddings = ensure_tensor_dtype(passage_embeddings)

    class TrainingTriplesDataset(Dataset):
        def __init__(self, triples, query_embeddings, passage_embeddings):
            self.triples = triples
            self.query_embeddings = query_embeddings
            self.passage_embeddings = passage_embeddings

        def __len__(self):
            return len(self.triples)

        def __getitem__(self, idx):
            q_idx, pos_idx, neg_idx = self.triples[idx]
            # Ensure we return float32 tensors
            return (
                ensure_tensor_dtype(self.query_embeddings[q_idx]),
                ensure_tensor_dtype(self.passage_embeddings[pos_idx]),
                ensure_tensor_dtype(self.passage_embeddings[neg_idx])
            )

    dataset = TrainingTriplesDataset(training_triples, query_embeddings, passage_embeddings)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    return dataloader


def train_model_cv(model_name_key, dataset_name, fold_idx, folds_data,
                   query_embeddings, passage_embeddings, qid_to_idx, pid_to_idx, main_metric_name,
                   triples_dir=None):
    """
    Train a model using cross-validation for a specific fold.

    Args:
        model_name_key: Key in config.MODEL_CONFIGS
        dataset_name: Name of the dataset
        fold_idx: Fold index to use
        folds_data: Fold data with training and testing splits
        query_embeddings: Loaded query embeddings
        passage_embeddings: Loaded passage embeddings
        qid_to_idx: Query ID to index mapping
        pid_to_idx: Passage ID to index mapping
        triples_dir: Directory containing triples files (default: data/cv_triples)

    Returns:
        best_metric: Best evaluation metric achieved
        final_results: Dictionary with training results
    """

    model_config_params = config.MODEL_CONFIGS[model_name_key]

    # Create save directory for this model and fold
    fold_dir = f"fold_{fold_idx}"
    current_model_save_dir = os.path.join(config.MODEL_SAVE_DIR, dataset_name, model_name_key, fold_dir)
    os.makedirs(current_model_save_dir, exist_ok=True)

    # Setup logging
    logger = setup_logging(current_model_save_dir)
    logger.info(f"Starting training for model: {model_name_key}")
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Fold: {fold_idx}")
    logger.info(f"Model config: {model_config_params}")
    logger.info(f"Device: {config.DEVICE}")

    # Get train queries and test queries for this fold
    train_qids = folds_data[str(fold_idx)]['training']
    test_qids = folds_data[str(fold_idx)]['testing']

    logger.info(f"Training queries: {len(train_qids)}")
    logger.info(f"Testing queries: {len(test_qids)}")

    # Load pre-created training triples
    training_triples = load_training_triples(dataset_name, fold_idx, triples_dir)

    # Create training dataloader
    logger.info("Creating training dataloader...")
    train_dataloader = create_dataloader_from_triples(
        training_triples,
        query_embeddings,
        passage_embeddings,
        batch_size=config.CV_BATCH_SIZE
    )

    # Load pre-created test data
    test_query_to_candidates = load_test_data(dataset_name, fold_idx, triples_dir)

    # Initialize model
    model = get_model(model_name_key, model_config_params).to(config.DEVICE)
    logger.info(f"Model architecture:\n{model}")
    model.apply(lambda m: m.to(torch.float32) if isinstance(m, nn.Module) else None)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Determine the correct qrels path based on dataset
    if dataset_name == "car":
        qrels_path = config.CAR_QRELS_FILE
    elif dataset_name == "robust":
        qrels_path = config.ROBUST_QRELS_FILE
    else:
        qrels_path = None  # For MS MARCO, can use ir_datasets

    # Handle dot product model (no training needed)
    if model_config_params["type"] == "dot_product":
        logger.info("Dot product model requires no training. Evaluating directly.")
        run_file_path = os.path.join(current_model_save_dir, f"run.test.{model_name_key}.txt")
        # Updated to pass dataset_name and appropriate qrels path
        metric_at_k, all_metrics = evaluate_model_on_dev(
            model, query_embeddings, passage_embeddings, qid_to_idx, pid_to_idx,
            test_query_to_candidates, run_file_path=run_file_path,
            use_ir_datasets=(dataset_name == "msmarco-passage" or dataset_name == "msmarco"),
            qrels_path=qrels_path,
            dataset_name=dataset_name
        )

        # Use appropriate main metric based on dataset
        main_metric = all_metrics.get(main_metric_name, metric_at_k)

        logger.info(f"Dot Product Test {main_metric_name.upper()}: {main_metric:.4f}")

        # Log additional metrics
        logger.info(f"Additional metrics:")
        for metric_name, value in all_metrics.items():
            logger.info(f"  {metric_name}: {value:.4f}")

        # Save results with all metrics
        with open(os.path.join(current_model_save_dir, "test_results.txt"), "w") as f_out:
            f_out.write(f"Model: {model_name_key}\n")
            f_out.write(f"Dataset: {dataset_name}\n")
            f_out.write(f"Fold: {fold_idx}\n")
            f_out.write(f"Test {main_metric_name.upper()}: {main_metric:.4f}\n")
            f_out.write(f"Additional Metrics:\n")
            for metric_name, value in all_metrics.items():
                f_out.write(f"  {metric_name}: {value:.4f}\n")

        # Save detailed results as JSON
        results = {
            'model_name': model_name_key,
            'dataset': dataset_name,
            'fold': fold_idx,
            main_metric_name: main_metric,
            'all_metrics': all_metrics,
            'num_parameters': total_params
        }
        with open(os.path.join(current_model_save_dir, "results.json"), "w") as f_out:
            json.dump(results, f_out, indent=2)

        return main_metric, results

    # Setup optimizer and loss function for trainable models
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    loss_fn = nn.MarginRankingLoss(margin=config.MARGIN).to(config.DEVICE)
    logger.info(f"Optimizer: AdamW (lr={config.LEARNING_RATE}, weight_decay={config.WEIGHT_DECAY})")
    logger.info(f"Loss function: MarginRankingLoss (margin={config.MARGIN})")

    best_metric = 0.0
    best_epoch = 0
    best_all_metrics = {}
    training_start_time = time.time()

    # Training loop
    for epoch in range(config.NUM_EPOCHS):
        epoch_start_time = time.time()
        model.train()
        total_loss = 0
        num_batches = 0

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{config.NUM_EPOCHS}")

        for i, (q_embeds, pos_p_embeds, neg_p_embeds) in enumerate(progress_bar):
            # Move data to device and ensure float32
            q_embeds = ensure_tensor_dtype(q_embeds.to(config.DEVICE, non_blocking=True))
            pos_p_embeds = ensure_tensor_dtype(pos_p_embeds.to(config.DEVICE, non_blocking=True))
            neg_p_embeds = ensure_tensor_dtype(neg_p_embeds.to(config.DEVICE, non_blocking=True))

            optimizer.zero_grad()

            # Forward pass
            scores_pos = model(q_embeds, pos_p_embeds)
            scores_neg = model(q_embeds, neg_p_embeds)

            # Compute loss
            targets = torch.ones_like(scores_pos).to(config.DEVICE)
            loss = loss_fn(scores_pos, scores_neg, targets)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # Log progress
            if (i + 1) % config.LOG_INTERVAL == 0:
                avg_loss = total_loss / num_batches
                progress_bar.set_postfix({'loss': avg_loss})
                logger.info(f"Epoch {epoch + 1}, Batch {i + 1}: Average loss = {avg_loss:.4f}")

        epoch_time = time.time() - epoch_start_time
        avg_epoch_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        logger.info(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s")
        logger.info(f"Epoch {epoch + 1} Average Loss: {avg_epoch_loss:.4f}")

        # Evaluate on test set
        logger.info(f"Evaluating on test set after epoch {epoch + 1}...")
        run_file_path = os.path.join(current_model_save_dir, f"run.test.epoch_{epoch + 1}.txt")
        # Pass dataset name and appropriate qrels path
        metric_at_k, all_metrics = evaluate_model_on_dev(
            model, query_embeddings, passage_embeddings, qid_to_idx, pid_to_idx,
            test_query_to_candidates, run_file_path=run_file_path,
            use_ir_datasets=(dataset_name == "msmarco-passage" or dataset_name == "msmarco"),
            qrels_path=qrels_path,
            dataset_name=dataset_name
        )

        # Get the main metric for this dataset
        current_metric = all_metrics.get(main_metric_name, metric_at_k)

        logger.info(f"Epoch {epoch + 1} Test {main_metric_name.upper()}: {current_metric:.4f}")

        # Log additional metrics
        for metric_name, value in all_metrics.items():
            if metric_name != main_metric_name:
                logger.info(f"  {metric_name}: {value:.4f}")

        # Save best model
        if current_metric > best_metric:
            best_metric = current_metric
            best_epoch = epoch + 1
            best_all_metrics = all_metrics.copy()
            logger.info(f"New best test {main_metric_name.upper()}: {best_metric:.4f}. Saving model...")

            # Save model state dict
            torch.save(model.state_dict(), os.path.join(current_model_save_dir, f"best_model.pth"))

            # Save the best run file
            best_run_file_path = os.path.join(current_model_save_dir, f"run.test.best_model.txt")
            # Copy the current run file to the best run file
            import shutil
            shutil.copy2(run_file_path, best_run_file_path)
            logger.info(f"Saved best model run file to {best_run_file_path}")

            # Save model config and metrics
            save_dict = {
                'model_state_dict': model.state_dict(),
                'epoch': epoch + 1,
                main_metric_name: current_metric,
                'all_metrics': all_metrics,
                'loss': avg_epoch_loss,
                'model_config': model_config_params,
                'optimizer_state_dict': optimizer.state_dict()
            }
            torch.save(save_dict, os.path.join(current_model_save_dir, "best_checkpoint.pth"))

        # Save current epoch model
        torch.save({
            'model_state_dict': model.state_dict(),
            'epoch': epoch + 1,
            main_metric_name: current_metric,
            'all_metrics': all_metrics,
            'loss': avg_epoch_loss,
            'model_config': model_config_params,
            'optimizer_state_dict': optimizer.state_dict()
        }, os.path.join(current_model_save_dir, f"model_epoch_{epoch + 1}.pth"))

    training_time = time.time() - training_start_time
    logger.info(f"Training complete for {model_name_key}")
    logger.info(f"Total training time: {training_time:.2f}s")
    logger.info(f"Best Test {main_metric_name.upper()}: {best_metric:.4f} (achieved at epoch {best_epoch})")

    # Save final results with all metrics
    final_results = {
        'model_name': model_name_key,
        'dataset': dataset_name,
        'fold': fold_idx,
        f'best_{main_metric_name}': best_metric,
        'best_epoch': best_epoch,
        'final_loss': avg_epoch_loss,
        'training_time': training_time,
        'total_epochs': config.NUM_EPOCHS,
        'best_all_metrics': best_all_metrics,
        'num_parameters': total_params,
        'trainable_parameters': trainable_params
    }

    with open(os.path.join(current_model_save_dir, "test_results.txt"), "w") as f_out:
        f_out.write(f"Model: {model_name_key}\n")
        f_out.write(f"Dataset: {dataset_name}\n")
        f_out.write(f"Fold: {fold_idx}\n")
        f_out.write(f"Best Test {main_metric_name.upper()}: {best_metric:.4f}\n")
        f_out.write(f"Best Epoch: {best_epoch}\n")
        f_out.write(f"Final Epoch Average Loss: {avg_epoch_loss:.4f}\n")
        f_out.write(f"Total Training Time: {training_time:.2f}s\n")
        f_out.write(f"Total Parameters: {total_params:,}\n")
        f_out.write(f"Trainable Parameters: {trainable_params:,}\n")
        f_out.write(f"\nBest Model Metrics:\n")
        for metric_name, value in best_all_metrics.items():
            f_out.write(f"  {metric_name}: {value:.4f}\n")

    # Save results as JSON for easy parsing
    with open(os.path.join(current_model_save_dir, "results.json"), "w") as f_out:
        json.dump(final_results, f_out, indent=2)

    return best_metric, final_results


def main():
    """Main function to run cross-validation training"""
    # Set random seeds for reproducibility
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train with cross-validation for TREC CAR and TREC ROBUST')
    parser.add_argument('--dataset', type=str, required=True, choices=['car', 'robust'],
                        help='Dataset to use (car or robust)')
    parser.add_argument('--models', nargs='+', help='Specific model keys to train (default: all models)')
    parser.add_argument('--folds', nargs='+', type=int, default=[0],
                        help='Specific folds to run (default: only fold 0)')
    parser.add_argument('--folds-file', type=str, default=None,
                        help='Path to folds JSON file (default: from config)')
    parser.add_argument('--triples-dir', type=str, default='data/cv_triples',
                        help='Directory containing pre-created training triples')
    # Add embedding-dir argument
    parser.add_argument('--embedding-dir', type=str, default=None,
                        help='Directory containing embeddings (overrides config.EMBEDDING_DIR)')
    args = parser.parse_args()

    dataset_name = args.dataset
    triples_dir = args.triples_dir

    # Update embedding directory if specified
    if args.embedding_dir:
        original_embedding_dir = config.EMBEDDING_DIR
        config.EMBEDDING_DIR = args.embedding_dir
        print(f"Overriding embedding directory to: {config.EMBEDDING_DIR}")

        # Update all dependent paths
        for dataset_prefix in ['', 'CAR_', 'ROBUST_']:
            # Skip if the original dataset prefix doesn't match our current dataset
            config_prefix = dataset_name.upper()
            if dataset_prefix and not dataset_prefix.startswith(config_prefix):
                continue

            # Update Query Embeddings Path
            path_attr = f"{dataset_prefix}QUERY_EMBEDDINGS_PATH"
            if hasattr(config, path_attr):
                original_path = getattr(config, path_attr)
                new_path = original_path.replace(original_embedding_dir, args.embedding_dir)
                setattr(config, path_attr, new_path)
                print(f"Updated {path_attr}: {new_path}")

            # Update Passage Embeddings Path
            path_attr = f"{dataset_prefix}PASSAGE_EMBEDDINGS_PATH"
            if hasattr(config, path_attr):
                original_path = getattr(config, path_attr)
                new_path = original_path.replace(original_embedding_dir, args.embedding_dir)
                setattr(config, path_attr, new_path)
                print(f"Updated {path_attr}: {new_path}")

            # Update Query ID to Index Path
            path_attr = f"{dataset_prefix}QUERY_ID_TO_IDX_PATH"
            if hasattr(config, path_attr):
                original_path = getattr(config, path_attr)
                new_path = original_path.replace(original_embedding_dir, args.embedding_dir)
                setattr(config, path_attr, new_path)
                print(f"Updated {path_attr}: {new_path}")

            # Update Passage ID to Index Path
            path_attr = f"{dataset_prefix}PASSAGE_ID_TO_IDX_PATH"
            if hasattr(config, path_attr):
                original_path = getattr(config, path_attr)
                new_path = original_path.replace(original_embedding_dir, args.embedding_dir)
                setattr(config, path_attr, new_path)
                print(f"Updated {path_attr}: {new_path}")

    # Get appropriate folds file
    if args.folds_file:
        folds_file_path = args.folds_file
    else:
        if dataset_name == "car":
            folds_file_path = config.CAR_FOLDS_FILE
        elif dataset_name == "robust":
            folds_file_path = config.ROBUST_FOLDS_FILE
        else:
            raise ValueError(f"Unsupported dataset for cross-validation: {dataset_name}")

    # Ensure model save directory exists
    os.makedirs(os.path.join(config.MODEL_SAVE_DIR, dataset_name), exist_ok=True)

    # Log system information
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"Device: {config.DEVICE}")
    print(f"Dataset: {dataset_name}")
    print(f"Folds file: {folds_file_path}")
    print(f"Triples directory: {triples_dir}")
    print(f"Embedding directory: {config.EMBEDDING_DIR}")

    # Load folds
    folds = load_folds(folds_file_path)
    print(f"Loaded {len(folds)} folds")

    # Process only specified folds
    folds_to_process = args.folds
    print(f"Processing folds: {folds_to_process}")

    # Load embeddings once for all folds and models
    print("Loading embeddings and mappings...")
    query_embeddings, passage_embeddings, qid_to_idx, pid_to_idx = load_embeddings_and_mappings(dataset_name)

    # Models to run
    if args.models:
        models_to_run = args.models
        print(f"Training specific models: {models_to_run}")
    else:
        # To run all defined models:
        models_to_run = list(config.MODEL_CONFIGS.keys())
        print(f"Training all {len(models_to_run)} models")

    # Store results for all folds and models
    all_results = {model_key: {} for model_key in models_to_run}

    # Define the main metric based on dataset before any potential exceptions
    if dataset_name == "car":
        main_metric_name = 'map'
    elif dataset_name == "robust":
        main_metric_name = "ndcg_cut_10"
    else:
        main_metric_name = "mrr_10"

    # Run training for each model and fold
    for model_key in models_to_run:
        if model_key not in config.MODEL_CONFIGS:
            print(f"Warning: Model key '{model_key}' not found in MODEL_CONFIGS. Skipping.")
            continue

        model_results = {}

        for fold_idx in folds_to_process:
            if str(fold_idx) not in folds:
                print(f"Warning: Fold {fold_idx} not found in folds file. Skipping.")
                continue

            print(f"\n{'=' * 50}")
            print(f"Training {model_key} on {dataset_name}, Fold {fold_idx}")
            print(f"{'=' * 50}")

            try:
                best_metric, fold_results = train_model_cv(
                    model_key,
                    dataset_name,
                    fold_idx,
                    folds,
                    query_embeddings,
                    passage_embeddings,
                    qid_to_idx,
                    pid_to_idx,
                    main_metric_name=main_metric_name,
                    triples_dir=triples_dir
                )

                model_results[fold_idx] = {
                    main_metric_name: best_metric,
                    'all_metrics': fold_results.get('best_all_metrics', {})
                }

                print(
                    f"Completed training {model_key} on fold {fold_idx}: {main_metric_name.upper()} = {best_metric:.4f}")

            except Exception as e:
                print(f"Error training {model_key} on fold {fold_idx}: {e}")
                import traceback
                traceback.print_exc()
                model_results[fold_idx] = {main_metric_name: 0.0}

        if model_results:
            metrics_sum = sum(results.get(main_metric_name, 0.0) for results in model_results.values())
            avg_metric = metrics_sum / len(model_results)

            # Store average in results
            all_results[model_key] = {
                'fold_results': model_results,
                f'avg_{main_metric_name}': avg_metric
            }

            print(
                f"\nModel {model_key} average {main_metric_name.upper()} across {len(model_results)} folds: {avg_metric:.4f}")

    # Print final summary
    print(f"\n{'=' * 50}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'=' * 50}")
    print(f"Dataset: {dataset_name}")

    metric_display = main_metric_name.upper()

    print(f"{'Model':<25} {'Avg ' + metric_display:<15} {'Folds':<10}")
    print(f"{'-' * 50}")

    for model_name, results in all_results.items():
        avg_metric = results.get(f'avg_{main_metric_name}', 0.0)
        fold_count = len(results.get('fold_results', {}))
        print(f"{model_name:<25} {avg_metric:<15.4f} {fold_count:<10}")

    # Save overall summary
    summary_path = os.path.join(config.MODEL_SAVE_DIR, f"{dataset_name}_cv_summary_results.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nAll results saved to {summary_path}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python
# train_ms_marco.py - Training script for MS MARCO passage ranking
import torch
import torch.optim as optim
import torch.nn as nn
import os
import json
from tqdm import tqdm
import time
import logging
import random
import numpy as np
import argparse

import config
from models import get_model
from data_loader import (
    load_embeddings_and_mappings,
    load_dev_data_for_eval,
    create_msmarco_train_dataloader
)
from evaluate import evaluate_model_on_dev
from utils import setup_logging


def train_model(model_name_key, use_ir_datasets=True):
    """
    Train a model specified by model_name_key on MS MARCO.

    Args:
        model_name_key: Key in config.MODEL_CONFIGS
        use_ir_datasets: Whether to use ir_datasets for data loading

    Returns:
        best_mrr: Best MRR@10 achieved on dev set
        final_results: Dictionary with training results
    """
    print(f"Starting training for model: {model_name_key}")
    model_config_params = config.MODEL_CONFIGS[model_name_key]

    # Use MS MARCO dataset
    dataset_name = "msmarco-passage"

    # Create save directory for this model
    current_model_save_dir = os.path.join(config.MODEL_SAVE_DIR, dataset_name, model_name_key)
    os.makedirs(current_model_save_dir, exist_ok=True)

    # Setup logging
    logger = setup_logging(current_model_save_dir)
    logger.info(f"Starting training for model: {model_name_key}")
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Model config: {model_config_params}")
    logger.info(f"Device: {config.DEVICE}")
    logger.info(f"Using ir_datasets: {use_ir_datasets}")

    # Load data - load once and reuse
    logger.info("Loading embeddings and mappings...")
    query_embeddings, passage_embeddings, qid_to_idx, pid_to_idx = load_embeddings_and_mappings(dataset_name)

    # Create train dataloader - pass embeddings to avoid reloading
    train_dataset_limit = None  # Set to a small number like 10000 for quick tests
    logger.info(f"Creating training dataloader (limit_size={train_dataset_limit})...")

    train_dataloader, _, _, _, _ = create_msmarco_train_dataloader(
        query_embeddings=query_embeddings,
        passage_embeddings=passage_embeddings,
        qid_to_idx=qid_to_idx,
        pid_to_idx=pid_to_idx,
        limit_size=train_dataset_limit,
        use_ir_datasets=use_ir_datasets,
        dataset_name=dataset_name
    )

    # Load dev data for evaluation (load once)
    logger.info("Loading dev data for evaluation...")
    dev_query_to_candidates = load_dev_data_for_eval(
        qid_to_idx,
        pid_to_idx,
        use_ir_datasets=use_ir_datasets,
        dataset_name=dataset_name
    )

    # Initialize model
    model = get_model(model_name_key, model_config_params).to(config.DEVICE)
    logger.info(f"Model architecture:\n{model}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Handle dot product model (no training needed)
    if model_config_params["type"] == "dot_product":
        logger.info("Dot product model requires no training. Evaluating directly.")
        run_file_path = os.path.join(current_model_save_dir, f"run.dev.{model_name_key}.txt")
        # Updated to handle new return format from evaluate_model_on_dev
        mrr_at_10, all_metrics = evaluate_model_on_dev(
            model, query_embeddings, passage_embeddings, qid_to_idx, pid_to_idx,
            dev_query_to_candidates, run_file_path=run_file_path
        )
        logger.info(f"Dot Product Dev MRR@10: {mrr_at_10:.4f}")

        # Log additional metrics
        logger.info(f"Additional metrics:")
        logger.info(f"  nDCG@10: {all_metrics.get('ndcg_cut_10', 0):.4f}")
        logger.info(f"  Recall@100: {all_metrics.get('recall_100', 0):.4f}")
        logger.info(f"  Recall@1000: {all_metrics.get('recall_1000', 0):.4f}")

        # Save results with all metrics
        with open(os.path.join(current_model_save_dir, "eval_results.txt"), "w") as f_out:
            f_out.write(f"Model: {model_name_key}\n")
            f_out.write(f"Dataset: {dataset_name}\n")
            f_out.write(f"Dev MRR@10: {mrr_at_10:.4f}\n")
            f_out.write(f"Additional Metrics:\n")
            for metric, value in all_metrics.items():
                f_out.write(f"  {metric}: {value:.4f}\n")

        # Save detailed results as JSON
        results = {
            'model_name': model_name_key,
            'dataset': dataset_name,
            'mrr_10': mrr_at_10,
            'all_metrics': all_metrics,
            'num_parameters': total_params
        }
        with open(os.path.join(current_model_save_dir, "results.json"), "w") as f_out:
            json.dump(results, f_out, indent=2)

        return mrr_at_10, results

    # Setup optimizer and loss function for trainable models
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    loss_fn = nn.MarginRankingLoss(margin=config.MARGIN).to(config.DEVICE)
    logger.info(f"Optimizer: AdamW (lr={config.LEARNING_RATE}, weight_decay={config.WEIGHT_DECAY})")
    logger.info(f"Loss function: MarginRankingLoss (margin={config.MARGIN})")

    best_dev_mrr = 0.0
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
            # Move data to device
            q_embeds = q_embeds.to(config.DEVICE, non_blocking=True)
            pos_p_embeds = pos_p_embeds.to(config.DEVICE, non_blocking=True)
            neg_p_embeds = neg_p_embeds.to(config.DEVICE, non_blocking=True)

            optimizer.zero_grad()

            # Forward pass
            scores_pos = model(q_embeds, pos_p_embeds)
            scores_neg = model(q_embeds, neg_p_embeds)

            # Compute loss
            targets = torch.ones_like(scores_pos).to(config.DEVICE)
            loss = loss_fn(scores_pos, scores_neg, targets)

            # Backward pass
            loss.backward()

            # Gradient clipping (optional but often helpful)
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

        # Evaluate on dev set
        logger.info(f"Evaluating on dev set after epoch {epoch + 1}...")
        run_file_path = os.path.join(current_model_save_dir, f"run.dev.epoch_{epoch + 1}.txt")
        # Updated to handle new return format
        current_dev_mrr, current_all_metrics = evaluate_model_on_dev(
            model, query_embeddings, passage_embeddings, qid_to_idx, pid_to_idx,
            dev_query_to_candidates, run_file_path=run_file_path
        )
        logger.info(f"Epoch {epoch + 1} Dev MRR@10: {current_dev_mrr:.4f}")

        # Log additional metrics
        logger.info(f"  nDCG@10: {current_all_metrics.get('ndcg_cut_10', 0):.4f}")
        logger.info(f"  Recall@100: {current_all_metrics.get('recall_100', 0):.4f}")

        # Save best model
        if current_dev_mrr > best_dev_mrr:
            best_dev_mrr = current_dev_mrr
            best_epoch = epoch + 1
            best_all_metrics = current_all_metrics.copy()
            logger.info(f"New best dev MRR@10: {best_dev_mrr:.4f}. Saving model...")

            # Save model state dict
            torch.save(model.state_dict(), os.path.join(current_model_save_dir, f"best_model.pth"))

            # Save model config and metrics
            save_dict = {
                'model_state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'mrr_10': current_dev_mrr,
                'all_metrics': current_all_metrics,
                'loss': avg_epoch_loss,
                'model_config': model_config_params,
                'optimizer_state_dict': optimizer.state_dict()
            }
            torch.save(save_dict, os.path.join(current_model_save_dir, "best_checkpoint.pth"))

        # Save current epoch model
        torch.save({
            'model_state_dict': model.state_dict(),
            'epoch': epoch + 1,
            'mrr_10': current_dev_mrr,
            'all_metrics': current_all_metrics,
            'loss': avg_epoch_loss,
            'model_config': model_config_params,
            'optimizer_state_dict': optimizer.state_dict()
        }, os.path.join(current_model_save_dir, f"model_epoch_{epoch + 1}.pth"))

    training_time = time.time() - training_start_time
    logger.info(f"Training complete for {model_name_key}")
    logger.info(f"Total training time: {training_time:.2f}s")
    logger.info(f"Best Dev MRR@10: {best_dev_mrr:.4f} (achieved at epoch {best_epoch})")

    # Save final results with all metrics
    final_results = {
        'model_name': model_name_key,
        'dataset': dataset_name,
        'best_dev_mrr': best_dev_mrr,
        'best_epoch': best_epoch,
        'final_loss': avg_epoch_loss,
        'training_time': training_time,
        'total_epochs': config.NUM_EPOCHS,
        'best_all_metrics': best_all_metrics,
        'num_parameters': total_params,
        'trainable_parameters': trainable_params
    }

    with open(os.path.join(current_model_save_dir, "eval_results.txt"), "w") as f_out:
        f_out.write(f"Model: {model_name_key}\n")
        f_out.write(f"Dataset: {dataset_name}\n")
        f_out.write(f"Best Dev MRR@10: {best_dev_mrr:.4f}\n")
        f_out.write(f"Best Epoch: {best_epoch}\n")
        f_out.write(f"Final Epoch Average Loss: {avg_epoch_loss:.4f}\n")
        f_out.write(f"Total Training Time: {training_time:.2f}s\n")
        f_out.write(f"Total Parameters: {total_params:,}\n")
        f_out.write(f"Trainable Parameters: {trainable_params:,}\n")
        f_out.write(f"\nBest Model Metrics:\n")
        for metric, value in best_all_metrics.items():
            f_out.write(f"  {metric}: {value:.4f}\n")

    # Save results as JSON for easy parsing
    with open(os.path.join(current_model_save_dir, "results.json"), "w") as f_out:
        json.dump(final_results, f_out, indent=2)

    return best_dev_mrr, final_results


def main():
    """Main function to run training for MS MARCO models"""
    # Set random seeds for reproducibility
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Ensure model save directory exists
    dataset_name = "msmarco-passage"
    os.makedirs(os.path.join(config.MODEL_SAVE_DIR, dataset_name), exist_ok=True)

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train MS MARCO passage ranking models')
    parser.add_argument('--use-files', action='store_true', help='Use file-based loading instead of ir_datasets')
    parser.add_argument('--models', nargs='+', help='Specific model keys to train (default: all models)')
    args = parser.parse_args()

    # Use ir_datasets by default
    use_ir_datasets = not args.use_files

    # Log system information
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"Device: {config.DEVICE}")
    print(f"Dataset: MS MARCO Passage")
    print(f"Using ir_datasets: {use_ir_datasets}")

    # Models to run
    if args.models:
        models_to_run = args.models
        print(f"Training specific models: {models_to_run}")
    else:
        # To run all defined models:
        models_to_run = list(config.MODEL_CONFIGS.keys())
        print(f"Training all {len(models_to_run)} models")

    # Store results for comparison
    all_results = {}

    for model_key in models_to_run:
        if model_key in config.MODEL_CONFIGS:
            print(f"\n{'=' * 50}")
            print(f"Training {model_key} on MS MARCO")
            print(f"{'=' * 50}")

            try:
                best_mrr, model_results = train_model(
                    model_key,
                    use_ir_datasets=use_ir_datasets
                )

                # Store results
                all_results[model_key] = {
                    'dev_mrr': best_mrr,
                    'dataset': 'msmarco-passage'
                }

                print(f"Completed training {model_key}: Dev MRR@10 = {best_mrr:.4f}")

            except Exception as e:
                print(f"Error training {model_key}: {e}")
                import traceback
                traceback.print_exc()
                all_results[model_key] = {'dev_mrr': 0.0, 'dataset': 'msmarco-passage'}
        else:
            print(f"Warning: Model key '{model_key}' not found in MODEL_CONFIGS. Skipping.")

    # Print final summary
    print(f"\n{'=' * 50}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'=' * 50}")
    print(f"Dataset: MS MARCO Passage")
    print(f"{'Model':<25} {'Dev MRR@10':<15}")
    print(f"{'-' * 40}")
    for model_name, results in all_results.items():
        dev_mrr = results.get('dev_mrr', 0.0)
        print(f"{model_name:<25} {dev_mrr:<15.4f}")

    # Save overall summary
    summary_path = os.path.join(config.MODEL_SAVE_DIR, "msmarco_passage_summary_results.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nAll results saved to {summary_path}")


if __name__ == "__main__":
    main()
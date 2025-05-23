#!/usr/bin/env python
# run_experiment_pipeline.py
# FIXED VERSION - Handles all the issues that were causing failures

import os
import sys
import argparse
import subprocess
import time
import json
import shutil
from datetime import datetime
from pathlib import Path


class ExperimentPipeline:
    def __init__(self, args):
        self.args = args
        self.base_dir = Path(args.base_dir)
        self.embedding_models = args.embedding_models
        self.datasets = args.datasets
        self.pipeline_components = args.pipeline_components
        self.device = args.device
        self.experiment_dir = self.base_dir / "experiment2"
        self.logs_dir = self.base_dir / "logs"
        self.logs_dir.mkdir(exist_ok=True, parents=True)
        self.verbose = args.verbose
        self.timeout = args.timeout

        # Set up path to scripts
        self.code_dir = Path(args.code_dir)
        if not self.code_dir.exists():
            print(f"Code directory {self.code_dir} does not exist!")
            sys.exit(1)

        # Configure paths
        if args.embedding_dir:
            self.embedding_dir = Path(args.embedding_dir)
        else:
            self.embedding_dir = self.base_dir / "embeddings"
        self.embedding_dir.mkdir(exist_ok=True, parents=True)

        if args.cv_triples_dir:
            self.cv_triples_dir = Path(args.cv_triples_dir)
        else:
            self.cv_triples_dir = self.base_dir / "cv_triples"
        self.cv_triples_dir.mkdir(exist_ok=True, parents=True)

        if args.model_save_dir:
            self.model_save_dir = Path(args.model_save_dir)
        else:
            self.model_save_dir = self.base_dir / "saved_models"
        self.model_save_dir.mkdir(exist_ok=True, parents=True)

        # Configure data paths (if provided)
        self.car_data_dir = Path(args.car_data_dir) if args.car_data_dir else None
        self.robust_data_dir = Path(args.robust_data_dir) if args.robust_data_dir else None

        # Configure cross-validation options
        self.folds = args.folds if args.folds else list(range(5))
        self.num_negatives = args.num_negatives

        # Ranks to test for low-rank bilinear models
        self.lrb_ranks = args.lrb_ranks

        # Log start
        self.start_time = datetime.now()
        self.log_file = self.logs_dir / f"pipeline_{self.start_time.strftime('%Y%m%d_%H%M%S')}.log"
        self.write_log(f"Starting experiment pipeline at {self.start_time}")
        self.write_log(f"Configuration:")
        self.write_log(f"  Base dir: {self.base_dir}")
        self.write_log(f"  Code dir: {self.code_dir}")
        self.write_log(f"  Embedding dir: {self.embedding_dir}")
        self.write_log(f"  CV triples dir: {self.cv_triples_dir}")
        self.write_log(f"  Model save dir: {self.model_save_dir}")
        self.write_log(f"  Datasets: {self.datasets}")
        self.write_log(f"  Embedding models: {self.embedding_models}")
        self.write_log(f"  Pipeline components: {self.pipeline_components}")
        self.write_log(f"  Device: {self.device}")
        self.write_log(f"  Folds: {self.folds}")
        self.write_log(f"  LRB Ranks: {self.lrb_ranks}")
        self.write_log(f"  Verbose: {self.verbose}")
        self.write_log(f"  Timeout: {self.timeout} seconds")

    def write_log(self, message):
        """Write a message to the log file and print to console"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        try:
            with open(self.log_file, "a") as f:
                f.write(log_message + "\n")
        except Exception as e:
            print(f"Warning: Could not write to log file: {e}")

    def run_command(self, command, description=None):
        """Run a shell command and log the output with proper error handling"""
        if description:
            self.write_log(f"Running: {description}")
        self.write_log(f"Command: {' '.join(str(c) for c in command)}")

        # Create a unique log file for this command
        timestamp = int(time.time() * 1000)
        cmd_log = self.logs_dir / f"cmd_{timestamp}.log"

        # Ensure the log directory exists
        cmd_log.parent.mkdir(exist_ok=True, parents=True)

        try:
            # Create the log file immediately so it exists
            with open(cmd_log, "w") as f:
                f.write(f"Command: {' '.join(str(c) for c in command)}\n")
                f.write(f"Started at: {datetime.now()}\n")
                f.write("-" * 50 + "\n")

            # Run the command and capture output
            start_time = time.time()

            # Convert all command parts to strings
            str_command = [str(c) for c in command]

            # Set up process with appropriate output redirection
            process = subprocess.Popen(
                str_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=self.code_dir  # Run from the code directory
            )

            # Open log file to write output
            with open(cmd_log, "a") as log_file:
                # Read output line by line and both print and log it
                while True:
                    line = process.stdout.readline()
                    if not line and process.poll() is not None:
                        break
                    if line:
                        log_file.write(line)
                        log_file.flush()
                        if self.verbose:
                            print(f"  > {line.rstrip()}")

            # Wait for process to complete with timeout
            try:
                returncode = process.wait(timeout=self.timeout)
                if returncode != 0:
                    raise subprocess.CalledProcessError(returncode, str_command)
            except subprocess.TimeoutExpired:
                process.kill()
                raise subprocess.TimeoutExpired(str_command, self.timeout)

            elapsed = time.time() - start_time

            # Log success
            self.write_log(f"Command completed successfully in {elapsed:.2f} seconds")
            self.write_log(f"Log file: {cmd_log}")
            return True

        except subprocess.TimeoutExpired:
            self.write_log(f"Command timed out after {self.timeout} seconds")
            self.write_log(f"Check log file for details: {cmd_log}")
            return False
        except subprocess.CalledProcessError as e:
            self.write_log(f"Command failed with exit code {e.returncode}")
            self.write_log(f"Check log file for details: {cmd_log}")
            return False
        except FileNotFoundError as e:
            self.write_log(f"Command not found: {e}")
            self.write_log(f"Make sure Python is installed and scripts exist")
            return False
        except Exception as e:
            self.write_log(f"Command failed with exception: {str(e)}")
            self.write_log(f"Check log file for details: {cmd_log}")
            return False

    def preprocess_embeddings(self, model_name, dataset):
        """Generate embeddings for a specific model and dataset"""
        self.write_log(f"Generating {model_name} embeddings for {dataset}...")

        # Create model-specific directory with proper naming
        model_dir_name = model_name.replace('/', '-')
        model_dir = self.embedding_dir / model_dir_name
        model_dir.mkdir(exist_ok=True, parents=True)

        # Check if embeddings already exist
        embedding_files_to_check = [
            model_dir / f"{dataset}_query_embeddings.npy",
            model_dir / f"{dataset}_passage_embeddings.npy",
            model_dir / f"{dataset}_query_id_to_idx.json",
            model_dir / f"{dataset}_passage_id_to_idx.json"
        ]

        all_exist = all(f.exists() for f in embedding_files_to_check)

        if all_exist and not self.args.force_rebuild:
            self.write_log(f"Embeddings already exist for {model_name}/{dataset}. Skipping...")
            return True

        # Build command
        cmd = [
            sys.executable,
            "preprocess_embeddings.py",
            "--dataset", dataset,
            "--model-name", model_name,
            "--embedding-dir", str(model_dir)
        ]

        # Add dataset-specific options
        if dataset == "robust":
            cmd.extend([
                "--use-chunking",
                "--chunk-size", "512",
                "--chunk-stride", "256",
                "--chunk-aggregation", "hybrid"
            ])

        # Add data directory if provided
        if dataset == "car" and self.car_data_dir:
            cmd.extend([
                "--queries-file", str(self.car_data_dir / "queries.tsv"),
                "--qrels-file", str(self.car_data_dir / "qrels.txt"),
                "--run-file", str(self.car_data_dir / "run.txt")
            ])
        elif dataset == "robust" and self.robust_data_dir:
            cmd.extend([
                "--queries-file", str(self.robust_data_dir / "queries.tsv"),
                "--qrels-file", str(self.robust_data_dir / "qrels.txt"),
                "--run-file", str(self.robust_data_dir / "run.txt")
            ])

        # Execute command
        desc = f"Preprocessing {model_name} embeddings for {dataset}"
        return self.run_command(cmd, desc)

    def create_cv_triples(self, model_name, dataset):
        """Create cross-validation triples for a specific model and dataset"""
        if dataset == "msmarco":
            self.write_log(f"Skipping CV triples creation for MS MARCO (not needed)")
            return True

        self.write_log(f"Creating CV triples for {dataset} with {model_name} embeddings...")

        # Create dataset-specific directory in CV triples
        dataset_dir = self.cv_triples_dir / dataset
        dataset_dir.mkdir(exist_ok=True, parents=True)

        # Check if triples already exist
        all_triples_exist = True
        for fold in self.folds:
            triples_file = dataset_dir / f"fold_{fold}_triples.pt"
            test_file = dataset_dir / f"fold_{fold}_test_data.json"
            if not triples_file.exists() or not test_file.exists() or self.args.force_rebuild:
                all_triples_exist = False
                break

        if all_triples_exist and not self.args.force_rebuild:
            self.write_log(f"CV triples already exist for all folds. Skipping...")
            return True

        # Build the model embedding path
        model_dir_name = model_name.replace('/', '-')
        model_embedding_dir = self.embedding_dir / model_dir_name

        # Build command
        cmd = [
            sys.executable,
            "create_cv_triples.py",
            "--dataset", dataset,
            "--embedding-dir", str(model_embedding_dir),
            "--output-dir", str(dataset_dir),
            "--negatives", str(self.num_negatives)
        ]

        # Add specific folds if provided
        if self.folds:
            cmd.extend(["--folds"] + [str(fold) for fold in self.folds])

        # Add data directory if provided
        if dataset == "car" and self.car_data_dir:
            cmd.extend([
                "--qrels-file", str(self.car_data_dir / "qrels.txt"),
                "--run-file", str(self.car_data_dir / "run.txt"),
                "--folds-file", str(self.car_data_dir / "folds.json")
            ])
        elif dataset == "robust" and self.robust_data_dir:
            cmd.extend([
                "--qrels-file", str(self.robust_data_dir / "qrels.txt"),
                "--run-file", str(self.robust_data_dir / "run.txt"),
                "--folds-file", str(self.robust_data_dir / "folds.json")
            ])

        # Execute command
        desc = f"Creating CV triples for {dataset} with {model_name}"
        return self.run_command(cmd, desc)

    def run_training(self, model_name, dataset):
        """Run training for a specific model and dataset"""
        self.write_log(f"Training models on {dataset} with {model_name} embeddings...")

        # Build the model embedding path
        model_dir_name = model_name.replace('/', '-')
        model_embedding_dir = self.embedding_dir / model_dir_name

        # Create model-specific save directory
        save_dir = self.model_save_dir / model_dir_name / dataset
        save_dir.mkdir(exist_ok=True, parents=True)

        # Determine which script to use based on dataset
        if dataset == "msmarco":
            script = "train_ms_marco.py"
        else:  # car or robust
            script = "train_cv.py"

        # Generate model list based on lrb_ranks
        model_list = ["dot_product", "weighted_dot_product"]

        # Add low-rank bilinear models
        for rank in self.lrb_ranks:
            model_list.append(f"low_rank_bilinear_{rank}")

        # Add full-rank bilinear if enabled
        if self.args.include_full_rank:
            model_list.append("full_rank_bilinear")

        # Build command
        cmd = [
                  sys.executable,
                  script,
                  "--embedding-dir", str(model_embedding_dir),
                  "--models"
              ] + model_list

        # Add dataset-specific options
        if dataset != "msmarco":
            cmd.extend([
                "--dataset", dataset,
                "--triples-dir", str(self.cv_triples_dir / dataset)
            ])

            # Add folds if provided
            if self.folds:
                cmd.extend(["--folds"] + [str(fold) for fold in self.folds])

            # Add model save directory
            cmd.extend(["--model-save-dir", str(save_dir)])

        # Device setting
        if self.device:
            cmd.extend(["--device", self.device])

        # Execute command
        desc = f"Training on {dataset} with {model_name} embeddings"
        return self.run_command(cmd, desc)

    def run_pipeline(self):
        """Run the full pipeline for all models and datasets"""
        self.write_log("Starting pipeline execution...")
        results = {}

        for model_name in self.embedding_models:
            model_key = model_name.replace('/', '-')
            results[model_key] = {}

            for dataset in self.datasets:
                self.write_log(f"\n{'=' * 60}")
                self.write_log(f"Processing {model_name} on {dataset}")
                self.write_log(f"{'=' * 60}")

                results[model_key][dataset] = {
                    "embedding_generation": False,
                    "triples_creation": False,
                    "training": False
                }

                # Step 1: Preprocess embeddings
                if "embeddings" in self.pipeline_components:
                    success = self.preprocess_embeddings(model_name, dataset)
                    results[model_key][dataset]["embedding_generation"] = success
                    if not success and not self.args.continue_on_failure:
                        self.write_log(
                            f"Failed to generate embeddings for {model_name} on {dataset}. Stopping pipeline for this combination.")
                        continue
                else:
                    self.write_log("Skipping embedding generation (not in components)")
                    results[model_key][dataset]["embedding_generation"] = True

                # Step 2: Create CV triples (only for CAR/ROBUST)
                if "triples" in self.pipeline_components and dataset != "msmarco":
                    success = self.create_cv_triples(model_name, dataset)
                    results[model_key][dataset]["triples_creation"] = success
                    if not success and not self.args.continue_on_failure:
                        self.write_log(
                            f"Failed to create CV triples for {model_name} on {dataset}. Stopping pipeline for this combination.")
                        continue
                else:
                    if dataset == "msmarco":
                        self.write_log("Skipping CV triples for MS MARCO (not needed)")
                    else:
                        self.write_log("Skipping CV triples creation (not in components)")
                    results[model_key][dataset]["triples_creation"] = True

                # Step 3: Run training
                if "training" in self.pipeline_components:
                    success = self.run_training(model_name, dataset)
                    results[model_key][dataset]["training"] = success
                    if not success:
                        self.write_log(f"Failed to run training for {model_name} on {dataset}.")
                else:
                    self.write_log("Skipping training (not in components)")
                    results[model_key][dataset]["training"] = True

        # Save and report overall results
        end_time = datetime.now()
        duration = end_time - self.start_time
        results_summary = {
            "start_time": self.start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": duration.total_seconds(),
            "configuration": {
                "embedding_models": self.embedding_models,
                "datasets": self.datasets,
                "pipeline_components": self.pipeline_components,
                "device": self.device,
                "folds": self.folds,
                "lrb_ranks": self.lrb_ranks
            },
            "results": results
        }

        # Save results to JSON
        results_path = self.logs_dir / f"pipeline_results_{self.start_time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_path, "w") as f:
            json.dump(results_summary, f, indent=2)

        self.write_log(f"Pipeline execution completed in {duration}")
        self.write_log(f"Results saved to {results_path}")

        # Print summary
        total_combinations = len(self.embedding_models) * len(self.datasets)
        successful_combinations = sum(
            all(results[model_key][dataset].values())
            for model_key in results
            for dataset in results[model_key]
        )

        self.write_log(f"Summary: {successful_combinations}/{total_combinations} combinations completed successfully")

        return results_summary


def main():
    parser = argparse.ArgumentParser(
        description="Run the complete experiment pipeline across multiple datasets and models")

    # Basic configuration
    parser.add_argument("--base-dir", type=str, default="/home/user/sisap2025",
                        help="Base directory for all experiment files")
    parser.add_argument("--code-dir", type=str, default="/home/user/bilinear-projection-theory/src/experiment2",
                        help="Directory containing experiment code")
    parser.add_argument("--embedding-dir", type=str, help="Directory to store/read embeddings")
    parser.add_argument("--cv-triples-dir", type=str, help="Directory to store/read CV triples")
    parser.add_argument("--model-save-dir", type=str, help="Directory to store trained models")

    # Dataset configurations
    parser.add_argument("--datasets", nargs="+", default=["car", "robust"],
                        help="Datasets to process (default: car, robust)")
    parser.add_argument("--car-data-dir", type=str, default="/home/user/sisap2025/data/car",
                        help="Directory containing TREC CAR data files")
    parser.add_argument("--robust-data-dir", type=str, default="/home/user/sisap2025/data/robust",
                        help="Directory containing TREC ROBUST data files")

    # Embedding model configuration
    parser.add_argument("--embedding-models", nargs="+",
                        default=["facebook/contriever"],
                        help="Embedding models to use")

    # Pipeline components
    parser.add_argument("--pipeline-components", nargs="+", default=["embeddings", "triples", "training"],
                        help="Pipeline components to run (default: all)")

    # Cross-validation configuration
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4],
                        help="Specific folds to process")
    parser.add_argument("--num-negatives", type=int, default=3, help="Number of negatives per positive")

    # Model configuration
    parser.add_argument("--lrb-ranks", nargs="+", type=int, default=[32, 64, 128],
                        help="Ranks for low-rank bilinear models")
    parser.add_argument("--include-full-rank", action="store_true",
                        help="Include full-rank bilinear model in training")

    # Execution options
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                        help="Device to use for training")
    parser.add_argument("--continue-on-failure", action="store_true",
                        help="Continue pipeline even if a step fails")
    parser.add_argument("--force-rebuild", action="store_true",
                        help="Force rebuild all artifacts even if they already exist")
    parser.add_argument("--verbose", action="store_true",
                        help="Show output from commands in real-time")
    parser.add_argument("--timeout", type=int, default=7200,
                        help="Timeout for each command in seconds (default: 2 hours)")

    args = parser.parse_args()

    # Create and run the pipeline
    pipeline = ExperimentPipeline(args)
    pipeline.run_pipeline()


if __name__ == "__main__":
    main()
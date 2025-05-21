# Experiment Pipeline Automation

This document explains how to use the automated pipeline script (`run_experiment_pipeline.py`) to systematically execute the entire experiment workflow across multiple embedding models and datasets.

## Overview

The pipeline script automates these key steps:
1. Generate embeddings for multiple models and datasets
2. Create cross-validation triples for TREC CAR and ROBUST datasets
3. Train and evaluate all model configurations

## Quick Start

```bash
# Clone the repository (if not already done)
git clone https://github.com/your-username/bilinear-proj-theory.git
cd bilinear-proj-theory

# Run the full pipeline with default settings
python run_experiment_pipeline.py

# Run with specific embedding models
python run_experiment_pipeline.py --embedding-models microsoft/mpnet-base facebook/contriever

# Run for specific datasets
python run_experiment_pipeline.py --datasets msmarco car

# Run only specific pipeline components
python run_experiment_pipeline.py --pipeline-components embeddings training
```

## Directory Structure

The pipeline maintains the following structure:
```
base_dir/
├── embeddings/                    # Embedding storage
│   ├── microsoft-mpnet-base/      # Model-specific directory
│   ├── google-electra-base/
│   └── facebook-contriever/
├── cv_triples/                    # Cross-validation triples
│   ├── car/                       # Dataset-specific subdirectory
│   └── robust/
├── saved_models/                  # Trained models
│   ├── microsoft-mpnet-base/      # Organization by model
│   │   ├── msmarco/
│   │   ├── car/
│   │   └── robust/
│   ├── google-electra-base/
│   └── facebook-contriever/
└── logs/                          # Execution logs
    ├── pipeline_20240521_123456.log  # Main pipeline log
    ├── cmd_1621506789.log            # Individual command logs
    └── pipeline_results_20240521_123456.json  # Results summary
```

## Command Line Options

### Basic Configuration
```
--base-dir PATH           Base directory for all experiment files
--code-dir PATH           Directory containing experiment code (default: src/experiment2)
--embedding-dir PATH      Directory to store/read embeddings
--cv-triples-dir PATH     Directory to store/read CV triples
--model-save-dir PATH     Directory to store trained models
```

### Dataset Selection
```
--datasets [DATASET ...]  Datasets to process (default: msmarco car robust)
--car-data-dir PATH       Directory containing TREC CAR data files
--robust-data-dir PATH    Directory containing TREC ROBUST data files
```

### Embedding Models
```
--embedding-models [MODEL ...]  Embedding models to use
                                Default: microsoft/mpnet-base google/electra-base facebook/contriever
```

### Pipeline Components
```
--pipeline-components [COMPONENT ...]  Components to run (default: embeddings triples training)
```

### Cross-Validation Configuration
```
--folds [FOLD ...]       Specific folds to process (default: 0 1 2 3 4)
--num-negatives INT      Number of negatives per positive (default: 3)
```

### Model Configuration
```
--lrb-ranks [RANK ...]   Ranks for low-rank bilinear models (default: 32 64 128)
--include-full-rank      Include full-rank bilinear model in training
```

### ROBUST Chunking Options
```
--use-chunking           Use chunking for ROBUST documents
--chunk-size INT         Maximum chunk size for ROBUST documents
--chunk-stride INT       Stride between chunks for ROBUST documents
--chunk-aggregation {mean,max,hybrid}  Chunk aggregation method
```

### Execution Options
```
--device {cuda,cpu}      Device to use for training (default: cuda)
--continue-on-failure    Continue pipeline even if a step fails
--use-config-save-dir    Use the save directory from config.py instead of command-line args
```

## Examples

### Run Full Pipeline with All Default Options
```bash
python run_experiment_pipeline.py
```

### Run Only for One Model and Dataset
```bash
python run_experiment_pipeline.py --embedding-models microsoft/mpnet-base --datasets msmarco
```

### Only Generate Embeddings (Skip Training)
```bash
python run_experiment_pipeline.py --pipeline-components embeddings
```

### Run Only Training with Existing Embeddings
```bash
python run_experiment_pipeline.py --pipeline-components training
```

### Use Different Low-Rank Bilinear Ranks
```bash
python run_experiment_pipeline.py --lrb-ranks 16 32 48 64
```

### Process Specific Folds for Cross-Validation
```bash
python run_experiment_pipeline.py --datasets car robust --folds 0 1
```

### Use Chunking for ROBUST Documents
```bash
python run_experiment_pipeline.py --datasets robust --use-chunking --chunk-size 512 --chunk-stride 256 --chunk-aggregation hybrid
```

### Use Custom Directories
```bash
python run_experiment_pipeline.py \
  --base-dir /path/to/experiment \
  --embedding-dir /path/to/embeddings \
  --cv-triples-dir /path/to/triples \
  --model-save-dir /path/to/models
```

## Results and Logs

After completion, the pipeline generates:

1. **Pipeline Log File**: Detailed log of all operations
2. **Command Logs**: Individual logs for each command execution
3. **Results JSON**: Summary of all execution results with success/failure status

Example results file:
```json
{
  "start_time": "2025-05-21T12:34:56.789012",
  "end_time": "2025-05-21T18:45:23.456789",
  "duration_seconds": 21866.667777,
  "configuration": {
    "embedding_models": ["microsoft/mpnet-base", "facebook/contriever"],
    "datasets": ["msmarco", "car", "robust"],
    "pipeline_components": ["embeddings", "triples", "training"],
    "device": "cuda",
    "folds": [0, 1, 2, 3, 4],
    "lrb_ranks": [32, 64, 128]
  },
  "results": {
    "microsoft-mpnet-base": {
      "msmarco": {
        "embedding_generation": true,
        "triples_creation": true,
        "training": true
      },
      "car": {
        "embedding_generation": true,
        "triples_creation": true,
        "training": true
      },
      "robust": {
        "embedding_generation": true,
        "triples_creation": true,
        "training": true
      }
    },
    "facebook-contriever": {
      "msmarco": {
        "embedding_generation": true,
        "triples_creation": true,
        "training": true
      },
      "car": {
        "embedding_generation": true,
        "triples_creation": true,
        "training": false
      },
      "robust": {
        "embedding_generation": true,
        "triples_creation": true,
        "training": true
      }
    }
  }
}
```

## Troubleshooting

If parts of the pipeline fail:

1. Check the individual command logs in the `logs/` directory
2. Use the `--continue-on-failure` flag to proceed past failures
3. Run only specific components with `--pipeline-components`
4. For memory issues, try:
   - Using a smaller model
   - Running on fewer datasets or folds
   - Adjusting batch sizes in the config files

## Extending the Pipeline

To support additional models or datasets:

1. Add the new model to the `--embedding-models` parameter
2. If adding a new dataset, implement the necessary data loading and evaluation functions in the experiment2 code
3. Modify the pipeline script to handle dataset-specific configurations

## Important Notes

- The pipeline uses command-line arguments to override values in `config.py` wherever possible
- For large datasets like TREC ROBUST, the embedding generation may take several hours
- Training all models on all datasets may take a day or more depending on hardware
- Consider running specific subsets initially to ensure everything works as expected
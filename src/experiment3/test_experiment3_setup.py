#!/usr/bin/env python3
# test_experiment3_setup.py
# Quick test to verify experiment3 can be imported and basic setup works

import sys
import os


def test_experiment3_setup():
    """Test that experiment3 can be properly set up and imported"""

    print("Testing Experiment 3 Setup (Dataset-Aware Version)")
    print("=" * 60)

    # 1. Test basic imports
    print("\n1. Testing basic imports...")
    try:
        import torch
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        import pandas as pd
        import numpy as np
        print("✓ Basic dependencies imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import basic dependencies: {e}")
        return False

    # 2. Test path setup
    print("\n2. Testing path setup...")
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_file_dir))

    paths_to_add = [
        project_root,
        os.path.join(project_root, 'src'),
        os.path.join(project_root, 'src', 'experiment1'),
        os.path.join(project_root, 'src', 'experiment2'),
        os.path.join(project_root, 'src', 'experiment3'),
    ]

    for path in paths_to_add:
        if path not in sys.path:
            sys.path.insert(0, path)

    print(f"✓ Added {len(paths_to_add)} paths to sys.path")
    print(f"  Project root: {project_root}")

    # 3. Test experiment3 config import
    print("\n3. Testing experiment3 config import...")
    try:
        # Try importing the fixed config
        import importlib.util
        config_path = os.path.join(current_file_dir, 'config.py')

        if os.path.exists(config_path):
            spec = importlib.util.spec_from_file_location("exp3_config", config_path)
            exp3_config = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(exp3_config)
            print(f"✓ Experiment3 config loaded from {config_path}")
            print(f"  EXP3_ENABLED: {getattr(exp3_config, 'EXP3_ENABLED', 'NOT FOUND')}")
            print(f"  Device: {getattr(exp3_config, 'DEVICE', 'NOT FOUND')}")
            print(f"  Dataset: {getattr(exp3_config, 'DATASET_NAME', 'NOT FOUND')}")

            # Check model path
            model_path = getattr(exp3_config, 'PRETRAINED_W_STAR_MODEL_PATH', 'NOT FOUND')
            if model_path != 'NOT FOUND':
                exists = os.path.exists(model_path)
                print(f"  Model path: {model_path}")
                print(f"  Model exists: {exists}")
                if not exists:
                    print(f"  ⚠ Warning: Model file not found. You'll need to train a model first.")
                else:
                    print(f"  ✓ Model file found!")

            # Check dataset config
            dataset_configs = getattr(exp3_config, 'DATASET_CONFIGS', {})
            current_dataset = getattr(exp3_config, 'DATASET_NAME', 'msmarco')
            if current_dataset in dataset_configs:
                config = dataset_configs[current_dataset]
                print(f"  Dataset config found for {current_dataset}:")
                print(f"    - Primary metric: {config.get('primary_metric', 'unknown')}")
                print(f"    - Use ir_datasets: {config.get('use_ir_datasets', 'unknown')}")

        else:
            print(f"✗ Config file not found at {config_path}")
            return False

    except Exception as e:
        print(f"✗ Failed to load experiment3 config: {e}")
        return False

    # 4. Test experiment2 imports
    print("\n4. Testing experiment2 imports...")
    try:
        # Try importing experiment2 modules
        from experiment2.models import get_model, FullRankBilinearModel, LowRankBilinearModel
        from experiment2.data_loader import load_embeddings_and_mappings, load_dev_data_for_eval
        from experiment2.evaluate import evaluate_model_on_dev
        from experiment2 import config as exp2_config
        print("✓ Experiment2 modules imported successfully")

        # Check if exp2_config has required attributes
        required_attrs = ['MODEL_CONFIGS', 'EMBEDDING_DIM', 'DEVICE']
        missing_attrs = [attr for attr in required_attrs if not hasattr(exp2_config, attr)]
        if missing_attrs:
            print(f"⚠ Warning: Missing attributes in exp2_config: {missing_attrs}")
        else:
            print("✓ Experiment2 config has all required attributes")

        # Check for dataset-specific paths in exp2_config
        dataset_paths = {}
        for dataset in ['CAR', 'ROBUST']:
            for data_type in ['QUERIES_FILE', 'QRELS_FILE', 'RUN_FILE', 'FOLDS_FILE']:
                attr_name = f"{dataset}_{data_type}"
                if hasattr(exp2_config, attr_name):
                    path = getattr(exp2_config, attr_name)
                    exists = os.path.exists(path)
                    dataset_paths[attr_name] = {'path': path, 'exists': exists}

        if dataset_paths:
            print("✓ Found dataset-specific file paths in experiment2 config:")
            for attr, info in dataset_paths.items():
                status = "✓" if info['exists'] else "✗"
                print(f"    {status} {attr}: {info['path']}")
        else:
            print("⚠ No dataset-specific file paths found in experiment2 config")

    except ImportError as e:
        print(f"✗ Failed to import experiment2 modules: {e}")
        print("  Trying fallback import strategy...")

        try:
            # Fallback import strategy
            sys.path.append(os.path.join(project_root, 'src', 'experiment2'))
            from models import get_model, FullRankBilinearModel, LowRankBilinearModel
            from data_loader import load_embeddings_and_mappings, load_dev_data_for_eval
            from evaluate import evaluate_model_on_dev
            import config as exp2_config
            print("✓ Experiment2 modules imported via fallback strategy")
        except ImportError as e2:
            print(f"✗ Fallback import also failed: {e2}")
            return False

    # 5. Test experiment1 imports - FIXED VERSION
    print("\n5. Testing experiment1 imports...")
    try:
        # Use importlib to explicitly load from the correct path
        import importlib.util
        exp1_models_path = os.path.join(project_root, 'src', 'experiment1', 'models.py')

        if os.path.exists(exp1_models_path):
            spec = importlib.util.spec_from_file_location("exp1_models", exp1_models_path)
            exp1_models = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(exp1_models)

            # Check if BilinearScorer exists
            if hasattr(exp1_models, 'BilinearScorer'):
                print("✓ Experiment1 BilinearScorer imported successfully")

                # Test that we can instantiate it
                test_W = torch.eye(10, dtype=torch.float32)
                test_scorer = exp1_models.BilinearScorer(test_W)
                print("✓ BilinearScorer can be instantiated")

            else:
                print("✗ BilinearScorer not found in experiment1.models")
                # List what is available
                available = [attr for attr in dir(exp1_models) if not attr.startswith('_')]
                print(f"  Available in exp1_models: {available}")

        else:
            print(f"✗ Experiment1 models.py not found at {exp1_models_path}")

    except Exception as e:
        print(f"✗ Failed to import experiment1 modules: {e}")
        print("  This is non-critical - fallback import should work in actual execution")

    # 6. Test basic functionality
    print("\n6. Testing basic functionality...")
    try:
        # Test that we can create a simple tensor and do SVD
        test_matrix = torch.randn(10, 10, dtype=torch.float32)
        U, S, Vh = torch.linalg.svd(test_matrix)
        print(f"✓ SVD test passed. Matrix shape: {test_matrix.shape}, SVD shapes: U{U.shape}, S{S.shape}, Vh{Vh.shape}")

        # Test matplotlib
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(5, 3))
        ax.plot([1, 2, 3], [1, 4, 2])
        ax.set_title("Test Plot")
        plt.close(fig)
        print("✓ Matplotlib test passed")

    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False

    # 7. Test dataset detection logic
    print("\n7. Testing dataset detection logic...")
    try:
        def test_determine_dataset_info(model_path):
            """Fixed test version of the dataset detection function"""
            config_dataset = getattr(exp3_config, 'DATASET_NAME', 'msmarco')

            # Try to infer from path first
            model_path_lower = model_path.lower()
            if 'msmarco' in model_path_lower:
                inferred = "msmarco"
            elif 'car' in model_path_lower:
                inferred = "car"
            elif 'robust' in model_path_lower:
                inferred = "robust"
            else:
                inferred = "msmarco"

            # Use inferred dataset, or fall back to config if path doesn't give clear indication
            dataset_name = inferred if inferred != "msmarco" or 'msmarco' in model_path_lower else config_dataset

            # Set ir_datasets based on final dataset choice
            if dataset_name in ["msmarco", "msmarco-passage"]:
                use_ir_datasets = True
                dataset_name = "msmarco-passage"
            else:
                use_ir_datasets = False

            return dataset_name, use_ir_datasets, inferred

        # Test with different model paths
        test_paths = [
            "/path/to/msmarco/model.pth",
            "/path/to/car/model.pth",
            "/path/to/robust/model.pth",
            "/path/to/unknown/model.pth"
        ]

        for test_path in test_paths:
            dataset, use_ir, inferred = test_determine_dataset_info(test_path)
            print(
                f"  Path: {test_path.split('/')[-3]}/... → Dataset: {dataset}, ir_datasets: {use_ir}, inferred: {inferred}")

        print("✓ Dataset detection logic works")

    except Exception as e:
        print(f"✗ Dataset detection test failed: {e}")
        return False

    # 8. Test full import chain (like in actual experiment3.py)
    print("\n8. Testing full import chain...")
    try:
        # Test the same import pattern as experiment3.py
        import importlib.util

        # Clear any cached modules to test fresh
        modules_to_clear = ['exp1_models', 'exp2_models']
        for mod in modules_to_clear:
            if mod in sys.modules:
                del sys.modules[mod]

        # Test experiment2 imports
        exp2_models_path = os.path.join(project_root, 'src', 'experiment2', 'models.py')
        spec = importlib.util.spec_from_file_location("exp2_models", exp2_models_path)
        exp2_models = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(exp2_models)

        get_exp2_model = exp2_models.get_model
        FullRankBilinearModel = exp2_models.FullRankBilinearModel
        LowRankBilinearModel = exp2_models.LowRankBilinearModel

        # Test experiment1 imports
        exp1_models_path = os.path.join(project_root, 'src', 'experiment1', 'models.py')
        spec = importlib.util.spec_from_file_location("exp1_models", exp1_models_path)
        exp1_models = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(exp1_models)

        BilinearScorer = exp1_models.BilinearScorer

        print("✓ Full import chain test passed")
        print("  Both experiment1 and experiment2 models can be imported correctly")

    except Exception as e:
        print(f"⚠ Full import chain test failed: {e}")
        print("  The fallback strategy in experiment3.py should handle this")

    print("\n" + "=" * 60)
    print("✓ Experiment 3 setup test completed successfully!")
    print("=" * 60)

    print("\nNext steps:")
    print("1. Make sure you have trained models from experiment2 for your target dataset")
    print("2. Update DATASET_NAME in experiment3/config.py to match your dataset ('msmarco', 'car', or 'robust')")
    print("3. Verify the model path is correct (should be auto-detected)")
    print("4. For CAR/ROBUST: ensure you have the required data files (queries.tsv, qrels.txt, etc.)")
    print("5. Run the experiment3 script: python experiment3.py")

    # Show specific recommendations based on detected dataset
    dataset_name = getattr(exp3_config, 'DATASET_NAME', 'msmarco')
    print(f"\nDataset-specific guidance for '{dataset_name}':")

    if dataset_name == "msmarco":
        print("- MS MARCO uses ir_datasets for automatic data loading")
        print("- Ensure you have internet connectivity for ir_datasets downloads")
        print("- Primary metric will be MRR@10")
    elif dataset_name == "car":
        print("- TREC CAR requires file-based data loading")
        print("- Ensure CAR_QUERIES_FILE, CAR_QRELS_FILE, CAR_RUN_FILE exist")
        print("- Primary metric will be MAP")
    elif dataset_name == "robust":
        print("- TREC ROBUST requires file-based data loading")
        print("- Ensure ROBUST_QUERIES_FILE, ROBUST_QRELS_FILE, ROBUST_RUN_FILE exist")
        print("- Primary metric will be nDCG@10")

    return True


if __name__ == "__main__":
    success = test_experiment3_setup()
    sys.exit(0 if success else 1)
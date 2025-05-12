# test_setup.py
import os
import sys
import torch
import numpy as np
import json


def test_imports():
    """Test if all required packages can be imported"""
    print("Testing imports...")

    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError:
        print("✗ PyTorch not installed")
        return False

    try:
        from sentence_transformers import SentenceTransformer
        print("✓ sentence-transformers")
    except ImportError:
        print("✗ sentence-transformers not installed")
        return False

    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
    except ImportError:
        print("✗ NumPy not installed")
        return False

    try:
        from tqdm import tqdm
        print("✓ tqdm")
    except ImportError:
        print("✗ tqdm not installed")
        return False

    return True


def test_config():
    """Test if config.py can be imported and has required variables"""
    print("\nTesting config.py...")

    try:
        import config
        print("✓ config.py imported successfully")
    except ImportError:
        print("✗ config.py not found")
        return False

    required_vars = [
        'MSMARCO_V1_DIR', 'EMBEDDING_DIR', 'MODEL_CONFIGS',
        'DEVICE', 'LEARNING_RATE', 'BATCH_SIZE'
    ]

    for var in required_vars:
        if hasattr(config, var):
            print(f"✓ {var} defined")
        else:
            print(f"✗ {var} not defined")
            return False

    return True


def test_data_paths():
    """Test if MS MARCO data paths exist"""
    print("\nTesting data paths...")

    import config

    # Check if directories exist
    if os.path.exists(config.MSMARCO_V1_DIR):
        print(f"✓ MS MARCO directory exists: {config.MSMARCO_V1_DIR}")
    else:
        print(f"⚠ MS MARCO directory not found: {config.MSMARCO_V1_DIR}")
        print("  You need to download MS MARCO data")

    # Check specific files
    files_to_check = [
        (config.TRAIN_TRIPLES_PATH, "Training triples"),
        (config.DEV_QUERIES_PATH, "Dev queries"),
        (config.DEV_QRELS_PATH, "Dev qrels"),
        (config.CORPUS_PATH, "Passage collection"),
    ]

    for file_path, description in files_to_check:
        if os.path.exists(file_path):
            print(f"✓ {description}: {file_path}")
        else:
            print(f"⚠ {description} not found: {file_path}")

    return True


def test_embedding_paths():
    """Test if embedding paths exist"""
    print("\nTesting embedding paths...")

    import config

    if os.path.exists(config.EMBEDDING_DIR):
        print(f"✓ Embedding directory exists: {config.EMBEDDING_DIR}")
    else:
        print(f"⚠ Embedding directory not found: {config.EMBEDDING_DIR}")
        print("  You need to run preprocess_embeddings.py")
        return False

    # Check embedding files
    files_to_check = [
        (config.QUERY_EMBEDDINGS_PATH, "Query embeddings"),
        (config.PASSAGE_EMBEDDINGS_PATH, "Passage embeddings"),
        (config.QUERY_ID_TO_IDX_PATH, "Query ID mapping"),
        (config.PASSAGE_ID_TO_IDX_PATH, "Passage ID mapping"),
    ]

    all_exist = True
    for file_path, description in files_to_check:
        if os.path.exists(file_path):
            print(f"✓ {description}: {file_path}")
            # Check file size
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"  Size: {size_mb:.1f} MB")
        else:
            print(f"⚠ {description} not found: {file_path}")
            all_exist = False

    return all_exist


def test_models():
    """Test if models can be imported and instantiated"""
    print("\nTesting models...")

    try:
        from models import get_model
        import config

        # Test each model type
        for model_name, model_config in config.MODEL_CONFIGS.items():
            try:
                model = get_model(model_name, model_config)
                print(f"✓ {model_name} model created successfully")

                # Test with dummy input
                dummy_q = torch.randn(2, config.EMBEDDING_DIM)
                dummy_p = torch.randn(2, config.EMBEDDING_DIM)
                with torch.no_grad():
                    scores = model(dummy_q, dummy_p)
                    assert scores.shape == (2,), f"Expected shape (2,), got {scores.shape}"
                print(f"  Forward pass test passed")
            except Exception as e:
                print(f"✗ {model_name} model error: {e}")
                return False
    except ImportError as e:
        print(f"✗ Cannot import models: {e}")
        return False

    return True


def test_evaluation_script():
    """Test if MS MARCO evaluation script exists"""
    print("\nTesting evaluation script...")

    import config

    if os.path.exists(config.MSMARCO_EVAL_SCRIPT):
        print(f"✓ Evaluation script found: {config.MSMARCO_EVAL_SCRIPT}")
        return True
    else:
        print(f"⚠ Evaluation script not found: {config.MSMARCO_EVAL_SCRIPT}")
        print("  Download from: https://github.com/microsoft/MSMARCO-Passage-Ranking")
        return False


def test_gpu():
    """Test GPU availability"""
    print("\nTesting GPU...")

    if torch.cuda.is_available():
        print("✓ CUDA available")
        print(f"  Device: {torch.cuda.get_device_name()}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")
    else:
        print("⚠ CUDA not available, will use CPU")
        print("  GPU recommended for faster training")

    return True


def create_directories():
    """Create necessary directories if they don't exist"""
    print("\nCreating necessary directories...")

    import config

    dirs_to_create = [
        config.EMBEDDING_DIR,
        config.MODEL_SAVE_DIR,
        os.path.dirname(config.MSMARCO_EVAL_SCRIPT),
    ]

    for dir_path in dirs_to_create:
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            print(f"✓ Created directory: {dir_path}")
        else:
            print(f"✓ Directory exists: {dir_path}")


def main():
    """Run all tests"""
    print("MS MARCO Passage Ranking Setup Test")
    print("=" * 40)

    tests = [
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("Data Paths", test_data_paths),
        ("Embedding Paths", test_embedding_paths),
        ("Models", test_models),
        ("Evaluation Script", test_evaluation_script),
        ("GPU", test_gpu),
    ]

    results = {}

    for test_name, test_func in tests:
        print(f"\n{test_name}")
        print("-" * 20)
        results[test_name] = test_func()

    # Create directories
    create_directories()

    # Summary
    print("\n" + "=" * 40)
    print("SUMMARY")
    print("=" * 40)

    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:<20} {status}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 40)
    if all_passed:
        print("✓ All tests passed! You're ready to start training.")
    else:
        print("⚠ Some tests failed. Please check the issues above.")
        print("\nNext steps:")
        print("1. Install missing dependencies: pip install -r requirements.txt")
        print("2. Download MS MARCO data to data/msmarco_v1/")
        print("3. Run preprocess_embeddings.py to generate embeddings")
        print("4. Download the MS MARCO evaluation script")

    return all_passed


if __name__ == "__main__":
    main()
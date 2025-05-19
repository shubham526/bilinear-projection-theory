#!/usr/bin/env python
# test_setup.py
import os
import torch


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
        import ir_datasets
        print(f"✓ ir_datasets {ir_datasets.__version__}")
    except ImportError:
        print("✗ ir_datasets not installed")
        return False
    except AttributeError:
        print("✓ ir_datasets (version unknown)")

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
        'EMBEDDING_DIR', 'MODEL_CONFIGS',
        'DEVICE', 'LEARNING_RATE', 'BATCH_SIZE',
        'QUERY_EMBEDDINGS_PATH', 'PASSAGE_EMBEDDINGS_PATH',
        'QUERY_ID_TO_IDX_PATH', 'PASSAGE_ID_TO_IDX_PATH'
    ]

    for var in required_vars:
        if hasattr(config, var):
            print(f"✓ {var} defined")
        else:
            print(f"✗ {var} not defined")
            return False

    return True


def test_ir_datasets():
    """Test if MS MARCO data can be accessed via ir_datasets"""
    print("\nTesting ir_datasets access to MS MARCO...")

    try:
        import ir_datasets

        # Test base MS MARCO access
        print("Loading msmarco-passage...")
        dataset = ir_datasets.load("msmarco-passage")

        # Count documents (without loading all)
        try:
            doc_count = dataset.docs_count()
            print(f"✓ Document count: {doc_count:,}")
        except:
            print("⚠ Could not get document count")

        # Test first document access
        try:
            first_doc = next(dataset.docs_iter())
            print(f"✓ First document loaded: ID {first_doc.doc_id}")
        except Exception as e:
            print(f"✗ Error loading document: {e}")
            return False

        # Test dev queries access
        print("\nLoading dev queries...")
        dev_dataset = ir_datasets.load("msmarco-passage/dev/small")

        # Test first query access
        try:
            first_query = next(dev_dataset.queries_iter())
            print(f"✓ First dev query loaded: ID {first_query.query_id}")
        except Exception as e:
            print(f"✗ Error loading dev query: {e}")
            return False

        # Test qrels access
        try:
            first_qrel = next(dev_dataset.qrels_iter())
            print(f"✓ First qrel loaded: Query {first_qrel.query_id}, Doc {first_qrel.doc_id}")
        except Exception as e:
            print(f"✗ Error loading qrels: {e}")
            return False

        # Test train triples access
        print("\nLoading training triples...")
        train_dataset = ir_datasets.load("msmarco-passage/train/triples-small")

        # Test first triple access
        try:
            first_docpair = next(train_dataset.docpairs_iter())
            print(
                f"✓ First docpair loaded: Query {first_docpair.query_id}, Docs {first_docpair.doc_id_a}, {first_docpair.doc_id_b}")
        except Exception as e:
            print(f"✗ Error loading docpairs: {e}")
            return False

        return True

    except Exception as e:
        print(f"✗ Error using ir_datasets: {e}")
        return False


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

        # Get embedding dimension
        embedding_dim = getattr(config, 'EMBEDDING_DIM', 768)  # Default to 768 if not specified

        # Test each model type
        for model_name, model_config in config.MODEL_CONFIGS.items():
            try:
                model = get_model(model_name, model_config)
                print(f"✓ {model_name} model created successfully")

                # Test with dummy input
                dummy_q = torch.randn(2, embedding_dim)
                dummy_p = torch.randn(2, embedding_dim)
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

    # Check if using traditional method
    traditional_path = getattr(config, 'MSMARCO_EVAL_SCRIPT', None)

    # Check if the script exists in the config location
    if traditional_path and os.path.exists(traditional_path):
        print(f"✓ Evaluation script found: {traditional_path}")
        return True

    # Also check if the script is available via ir_datasets
    try:
        import ir_datasets
        # Check common locations where ir_datasets might store the script
        possible_paths = [
            os.path.join(os.path.dirname(ir_datasets.__file__), 'etc', 'msmarco', 'msmarco_eval.py'),
            os.path.expanduser('~/.ir_datasets/msmarco/msmarco_eval.py')
        ]

        for path in possible_paths:
            if os.path.exists(path):
                print(f"✓ Evaluation script found through ir_datasets: {path}")
                return True

        # If we can't find it, check if we can download it
        import requests
        url = "https://raw.githubusercontent.com/microsoft/MSMARCO-Passage-Ranking/master/ms_marco_eval.py"
        try:
            response = requests.head(url)
            if response.status_code == 200:
                print(f"✓ Evaluation script available for download from: {url}")
                print("  Will be automatically downloaded when needed")
                return True
        except:
            pass

        print("⚠ Evaluation script not found")
        print("  It will be downloaded automatically when needed")
        return True

    except ImportError:
        print("⚠ Evaluation script not found: {traditional_path}")
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
    ]

    # Add eval script dir if it exists in config
    eval_script_dir = getattr(config, 'MSMARCO_EVAL_SCRIPT', None)
    if eval_script_dir:
        dirs_to_create.append(os.path.dirname(eval_script_dir))

    for dir_path in dirs_to_create:
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            print(f"✓ Created directory: {dir_path}")
        else:
            print(f"✓ Directory exists: {dir_path}")


def main():
    """Run all tests"""
    print("MS MARCO Passage Ranking Setup Test (ir_datasets version)")
    print("=" * 60)

    tests = [
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("ir_datasets", test_ir_datasets),
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
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:<20} {status}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All tests passed! You're ready to start training.")
    else:
        print("⚠ Some tests failed. Please check the issues above.")
        print("\nNext steps:")
        print("1. Install missing dependencies: pip install -r requirements.txt ir_datasets")
        print("2. Run preprocess_embeddings_ir_datasets.py to generate embeddings")
        print("3. Use main_train_ir_datasets.py for training")

    return all_passed


if __name__ == "__main__":
    main()
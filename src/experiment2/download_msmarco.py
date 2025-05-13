# download_msmarco.py
"""
Script to download MS MARCO passage ranking data
"""
import os
import requests
import tarfile
from tqdm import tqdm
import config


def download_file(url, local_path, desc=None):
    """Download a file with progress bar"""
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    if os.path.exists(local_path):
        print(f"File already exists: {local_path}")
        return True

    print(f"Downloading {desc or os.path.basename(local_path)}...")

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        with open(local_path, 'wb') as f:
            if total_size == 0:
                f.write(response.content)
            else:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))

        print(f"Downloaded: {local_path}")
        return True

    except Exception as e:
        print(f"Error downloading {url}: {e}")
        if os.path.exists(local_path):
            os.remove(local_path)
        return False


def extract_tar_gz(tar_path, extract_to):
    """Extract tar.gz file"""
    print(f"Extracting {tar_path}...")

    try:
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(extract_to)
        print(f"Extracted to: {extract_to}")
        return True
    except Exception as e:
        print(f"Error extracting {tar_path}: {e}")
        return False


def download_msmarco_data():
    """Download all necessary MS MARCO files"""
    print("MS MARCO Data Download Script")
    print("=" * 50)

    # Create base directory
    os.makedirs(config.MSMARCO_V1_DIR, exist_ok=True)

    # URLs for MS MARCO files
    files_to_download = [
        {
            'url': 'https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz',
            'local_path': os.path.join(config.MSMARCO_V1_DIR, 'collection.tar.gz'),
            'desc': 'Collection (passages)',
            'extract': True
        },
        {
            'url': 'https://msmarco.blob.core.windows.net/msmarcoranking/triples.train.small.tar.gz',
            'local_path': os.path.join(config.MSMARCO_V1_DIR, 'triples.train.small.tar.gz'),
            'desc': 'Training triples',
            'extract': True
        },
        {
            'url': 'https://msmarco.blob.core.windows.net/msmarcoranking/queries.dev.small.tar.gz',
            'local_path': os.path.join(config.MSMARCO_V1_DIR, 'queries.dev.small.tar.gz'),
            'desc': 'Dev queries',
            'extract': True
        },
        {
            'url': 'https://msmarco.blob.core.windows.net/msmarcoranking/qrels.dev.small.tar.gz',
            'local_path': os.path.join(config.MSMARCO_V1_DIR, 'qrels.dev.small.tar.gz'),
            'desc': 'Dev qrels',
            'extract': True
        },
        {
            'url': 'https://msmarco.blob.core.windows.net/msmarcoranking/top1000.dev.tar.gz',
            'local_path': os.path.join(config.MSMARCO_V1_DIR, 'top1000.dev.tar.gz'),
            'desc': 'Top 1000 candidates',
            'extract': True
        }
    ]

    # Download files
    for file_info in files_to_download:
        success = download_file(
            file_info['url'],
            file_info['local_path'],
            file_info['desc']
        )

        if success and file_info.get('extract', False):
            extract_tar_gz(file_info['local_path'], config.MSMARCO_V1_DIR)
            # Optionally remove the tar.gz file after extraction
            # os.remove(file_info['local_path'])

    print("\nDownload complete!")
    print("\nDownloaded files:")
    for root, dirs, files in os.walk(config.MSMARCO_V1_DIR):
        for file in files:
            file_path = os.path.join(root, file)
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"  {file}: {size_mb:.1f} MB")


def download_eval_script():
    """Download the MS MARCO evaluation script"""
    print("\nDownloading MS MARCO evaluation script...")

    os.makedirs(os.path.dirname(config.MSMARCO_EVAL_SCRIPT), exist_ok=True)

    eval_script_url = "https://raw.githubusercontent.com/microsoft/MSMARCO-Passage-Ranking/master/ms_marco_eval.py"

    success = download_file(eval_script_url, config.MSMARCO_EVAL_SCRIPT, "Evaluation script")

    if success:
        print("Evaluation script downloaded successfully!")
    else:
        print("Failed to download evaluation script. Please download manually:")
        print(f"  URL: {eval_script_url}")
        print(f"  Save to: {config.MSMARCO_EVAL_SCRIPT}")


def main():
    """Main function"""
    print("This script will download MS MARCO passage ranking data.")
    print("Warning: This will download several GB of data.")

    response = input("\nProceed with download? (y/n): ")
    if response.lower() != 'y':
        print("Download cancelled.")
        return

    # Download MS MARCO data
    download_msmarco_data()

    # Download evaluation script
    download_eval_script()

    print("\n" + "=" * 50)
    print("Download complete!")
    print("\nNext steps:")
    print("1. Verify all files are present")
    print("2. Run test_setup.py to check your setup")
    print("3. Run preprocess_embeddings.py to generate embeddings")
    print("4. Run main_train.py to start training")


if __name__ == "__main__":
    main()
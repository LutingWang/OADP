#!/usr/bin/env python3
"""
Script to download ImageNet-21K dataset using the synset IDs from a text file.
"""

import os
import sys
import argparse
import requests
import tarfile
from tqdm import tqdm
import concurrent.futures
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Download ImageNet-21K dataset')
    parser.add_argument('--output_dir', type=str, default='./imagenet21k',
                        help='Directory to save the downloaded files')
    parser.add_argument('--synset_file', type=str, default='imagenet21k.txt',
                        help='Path to the file containing synset IDs')
    parser.add_argument('--max_workers', type=int, default=4,
                        help='Maximum number of concurrent downloads')
    parser.add_argument('--base_url', type=str, 
                        default='https://image-net.org/data/winter21_whole/',
                        help='Base URL for downloading synsets')
    return parser.parse_args()

def download_file(url, output_path, max_retries=3, retry_delay=5):
    """Download a file from a URL to the specified output path with retry capability."""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024  # 1 Kibibyte
            
            # Check if file exists and is partially downloaded
            mode = 'ab' if os.path.exists(output_path) else 'wb'
            initial_pos = os.path.getsize(output_path) if mode == 'ab' else 0
            
            # Set the range header to resume download
            if initial_pos > 0:
                headers = {'Range': f'bytes={initial_pos}-'}
                response = requests.get(url, stream=True, headers=headers)
                if response.status_code == 206:  # Partial content
                    logger.info(f"Resuming download from byte {initial_pos}")
                    total_size = int(response.headers.get('content-length', 0)) + initial_pos
                else:
                    # If server doesn't support range requests, start over
                    mode = 'wb'
                    initial_pos = 0
            
            with open(output_path, mode) as f, tqdm(
                    desc=os.path.basename(output_path),
                    total=total_size,
                    unit='iB',
                    unit_scale=True,
                    unit_divisor=1024,
                    initial=initial_pos
            ) as bar:
                for data in response.iter_content(block_size):
                    size = f.write(data)
                    bar.update(size)
            
            # Verify file size if content-length was provided
            if total_size > 0 and os.path.getsize(output_path) != total_size:
                logger.warning(f"Downloaded file size doesn't match expected size for {url}. Retrying...")
                continue
                
            return True
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"Attempt {attempt+1}/{max_retries} failed for {url}: {e}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                # Increase retry delay for next attempt (exponential backoff)
                retry_delay *= 2
            else:
                logger.error(f"Failed to download {url} after {max_retries} attempts")
                if os.path.exists(output_path) and os.path.getsize(output_path) == 0:
                    os.remove(output_path)
                return False
    
    return False

def extract_tar(tar_path, extract_path):
    """Extract a tar file to the specified path."""
    try:
        with tarfile.open(tar_path) as tar:
            tar.extractall(path=extract_path)
        return True
    except Exception as e:
        logger.error(f"Error extracting {tar_path}: {e}")
        return False

def download_and_extract_synset(synset_id, base_url, output_dir, extract=True):
    """Download and optionally extract a synset tar file."""
    url = f"{base_url}{synset_id}.tar"
    tar_path = os.path.join(output_dir, f"{synset_id}.tar")
    extract_path = os.path.join(output_dir, synset_id)
    
    # Create extract directory if it doesn't exist
    if extract and not os.path.exists(extract_path):
        os.makedirs(extract_path, exist_ok=True)
    
    # Skip if already downloaded and extracted
    if extract and os.path.exists(extract_path) and len(os.listdir(extract_path)) > 0:
        logger.info(f"Synset {synset_id} already downloaded and extracted. Skipping.")
        return True
    
    # Download the tar file
    if not os.path.exists(tar_path):
        logger.info(f"Downloading {synset_id} from {url}")
        if not download_file(url, tar_path):
            return False
    
    # Extract the tar file
    if extract:
        logger.info(f"Extracting {synset_id}")
        if not extract_tar(tar_path, extract_path):
            return False
        
        # Remove the tar file after extraction
        os.remove(tar_path)
    
    return True

def read_synset_ids(file_path):
    """Read synset IDs from a file, skipping empty lines and lines starting with '...'."""
    synset_ids = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('...'):
                synset_ids.append(line)
    return synset_ids

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Read synset IDs from file
    synset_ids = read_synset_ids(args.synset_file)
    logger.info(f"Found {len(synset_ids)} synset IDs in {args.synset_file}")
    
    # Download and extract synsets in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(
                download_and_extract_synset, 
                synset_id, 
                args.base_url, 
                args.output_dir
            ): synset_id for synset_id in synset_ids
        }
        
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            synset_id = futures[future]
            try:
                success = future.result()
                status = "Success" if success else "Failed"
            except Exception as e:
                logger.error(f"Error processing {synset_id}: {e}")
                status = "Error"
            
            completed += 1
            logger.info(f"Progress: {completed}/{len(synset_ids)} - {synset_id}: {status}")
    
    logger.info("Download completed!")

if __name__ == "__main__":
    main()

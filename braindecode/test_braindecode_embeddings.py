import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from braindecode_embeddings import get_model, extract_pool4_features, extract_embeddings_for_specific_eeg
from const import TRAIN_EEG_DIR, TRAIN_CSV_PATH


def test_single_example():
    """Test Deep4Net pool_4 embedding extraction on a single example"""
    print("Testing Deep4Net pool_4 embedding extraction on a single EEG file...")

    # Get a specific EEG ID
    train_df = pd.read_csv(TRAIN_CSV_PATH)
    eeg_id = str(train_df['eeg_id'].iloc[0])

    # Load and process the EEG data
    import os
    import pyarrow.parquet as pq
    import torch

    parquet_path = os.path.join(TRAIN_EEG_DIR, f"{eeg_id}.parquet")
    print(f"Using EEG file: {parquet_path}")

    # Read the file
    table = pq.read_table(parquet_path)
    df = table.to_pandas()
    eeg_data = df.values

    # Normalize and prepare data
    eeg_data_norm = (eeg_data - eeg_data.min()) / (eeg_data.max() - eeg_data.min())
    eeg_tensor = torch.tensor(eeg_data_norm.T, dtype=torch.float32).unsqueeze(0)

    print(f"EEG data shape: {eeg_data.shape}")
    print(f"EEG tensor shape: {eeg_tensor.shape}")

    # Create model and extract embeddings
    model = get_model(n_channels=eeg_tensor.shape[1], input_window_samples=eeg_tensor.shape[2])

    # Extract embeddings from pool_4 layer
    features = extract_pool4_features(model, eeg_tensor)

    # Print stats
    print(f"Embedding shape: {features.shape}")
    print(f"Number of embeddings: {features.size}")
    print(f"First few values: {features.flatten()[:5]}")

    # Expected output is about 1400 features (200 channels * 7 time steps * 1)
    expected_size = 1400

    if features.size != expected_size:
        print(f"WARNING: Feature size ({features.size}) does not match expected size ({expected_size})")
    else:
        print(f"SUCCESS: Feature size matches expected 1400 features from pool_4 layer")


def test_specific_eeg_extraction(eeg_id=None, eeg_sub_id=None):
    """
    Test extraction of embeddings for a specific EEG ID and sub-ID

    Args:
        eeg_id (str): The EEG ID to test. If None, uses the first one from training data.
        eeg_sub_id (str/int): The EEG sub-ID to test. If None, uses the first sub-ID from training data.
    """
    print("Testing extraction of Deep4Net pool_4 embeddings for a specific EEG...")

    # If no IDs provided, get the first one from the training data
    if eeg_id is None or eeg_sub_id is None:
        train_df = pd.read_csv(TRAIN_CSV_PATH)
        sample_row = train_df.iloc[0]
        eeg_id = str(sample_row['eeg_id']) if eeg_id is None else eeg_id
        eeg_sub_id = sample_row['eeg_sub_id'] if eeg_sub_id is None else eeg_sub_id

    print(f"Using EEG ID: {eeg_id}, sub-ID: {eeg_sub_id}")

    # Extract embeddings
    embeddings = extract_embeddings_for_specific_eeg(eeg_id, eeg_sub_id)

    if embeddings is None:
        print(f"Failed to extract embeddings")
        return

    # Print stats
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Number of embeddings: {embeddings.size}")
    print(f"First few values: {embeddings.flatten()[:5]}")

    # Expected size check
    expected_size = 1400
    if embeddings.size != expected_size:
        print(f"WARNING: Feature size ({embeddings.size}) does not match expected size ({expected_size})")
    else:
        print(f"SUCCESS: Feature size matches expected 1400 features from pool_4 layer")


def run_batch_test(n_samples=10):
    """Run performance test for Deep4Net pool_4 extraction"""
    print(f"Running performance test with {n_samples} samples...")

    # Get a random sample
    train_df = pd.read_csv(TRAIN_CSV_PATH)
    if len(train_df) < n_samples:
        print(f"Warning: Requested {n_samples} samples but only {len(train_df)} are available")
        samples = train_df
    else:
        samples = train_df.sample(n=n_samples, random_state=42)

    # Initialize timing and results
    start_time = time.time()
    extraction_times = []
    processed_count = 0
    embedding_counts = []

    # Process each sample
    for _, row in tqdm(samples.iterrows(), total=len(samples)):
        eeg_id = str(row['eeg_id'])
        eeg_sub_id = row['eeg_sub_id']

        # Time the extraction process
        sample_start = time.time()

        try:
            # Extract embeddings
            embeddings = extract_embeddings_for_specific_eeg(eeg_id, eeg_sub_id)

            if embeddings is not None:
                # Count embeddings
                embedding_counts.append(embeddings.size)
                processed_count += 1

                # Record time for this sample
                sample_time = time.time() - sample_start
                extraction_times.append(sample_time)
        except Exception as e:
            print(f"Error processing EEG {eeg_id}: {e}")

    # Calculate statistics
    total_time = time.time() - start_time

    # Prepare results
    print("\nPerformance Results:")
    print(f"Processed {processed_count} samples")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average time per sample: {np.mean(extraction_times):.4f} seconds")
    print(f"Embeddings per sample: {np.mean(embedding_counts):.0f}")

    expected_size = 1400
    avg_embeddings = np.mean(embedding_counts) if embedding_counts else 0

    if abs(avg_embeddings - expected_size) > 10:  # Allow some flexibility
        print(f"WARNING: Average feature size ({avg_embeddings:.0f}) differs from expected size ({expected_size})")
    else:
        print(f"SUCCESS: Average feature size approximately matches expected 1400 features from pool_4 layer")


def test(specific_eeg=False, eeg_id=None, eeg_sub_id=None, batch_size=10):
    """
    Run all tests for Deep4Net pool_4 embeddings extraction

    Args:
        specific_eeg (bool): Whether to test specific EEG extraction
        eeg_id (str): Specific EEG ID to test
        eeg_sub_id (str/int): Specific EEG sub-ID to test
        batch_size (int): Number of samples to test in batch mode
    """
    # Test extraction on a single example
    test_single_example()

    # Test specific EEG extraction if requested
    if specific_eeg:
        print("\n" + "=" * 80)
        test_specific_eeg_extraction(eeg_id, eeg_sub_id)

    # Run batch performance test
    print("\n" + "=" * 80)
    run_batch_test(n_samples=batch_size)

    print("\nAll tests completed.")


if __name__ == "__main__":
    # Run the tests with default settings
    test()
import time

from braindecode_embeddings import extract_braindecode_embeddings, extract_embeddings_for_specific_eeg
from const import TRAIN_EEG_DIR, TRAIN_CSV_PATH, BRAINDECODE_OUT_PATH, BRAINDECODE_OUT_PATH_TEST, \
    TEST_SAMPLE_SIZE_BRAINDECODE


def test_single_eeg():
    """Test embedding extraction on a single EEG"""
    import torch
    import os
    import pandas as pd
    import pyarrow.parquet as pq
    from braindecode_embeddings import get_model, extract_pool4_features

    # Get a specific EEG ID
    train_df = pd.read_csv(TRAIN_CSV_PATH)
    eeg_id = str(train_df['eeg_id'].iloc[0])

    # Load the EEG file
    parquet_path = os.path.join(TRAIN_EEG_DIR, f"{eeg_id}.parquet")
    print(f"Loading EEG file: {parquet_path}")

    table = pq.read_table(parquet_path)
    df = table.to_pandas()
    eeg_data = df.values

    # Normalize and prepare data
    eeg_data_norm = (eeg_data - eeg_data.min()) / (eeg_data.max() - eeg_data.min())
    eeg_tensor = torch.tensor(eeg_data_norm.T, dtype=torch.float32).unsqueeze(0)

    # Create model and extract embeddings
    print("\nTesting Deep4Net pool_4 layer extraction")
    model = get_model(n_channels=eeg_tensor.shape[1], input_window_samples=eeg_tensor.shape[2])
    features = extract_pool4_features(model, eeg_tensor)

    print(f"Embedding shape: {features.shape}")
    print(f"Number of embeddings: {features.size}")
    print(f"First few values: {features.flatten()[:5]}")


def extract_specific_eeg_and_save(eeg_id, eeg_sub_id, output_file=None):
    """
    Extract embeddings for a specific EEG ID and sub-ID combination and save to CSV.

    Args:
        eeg_id (str): The EEG ID
        eeg_sub_id (int or str): The EEG sub ID
        output_file (str): Path to save the CSV file. If None, generates a default name.

    Returns:
        dict: Dictionary with embeddings information
    """
    import pandas as pd
    import numpy as np

    start_time = time.time()
    print(f"Extracting Deep4Net pool_4 embeddings for EEG ID: {eeg_id}, sub-ID: {eeg_sub_id}")

    # Extract the embeddings
    embeddings = extract_embeddings_for_specific_eeg(eeg_id, eeg_sub_id)

    if embeddings is None:
        print(f"Failed to extract embeddings for EEG ID: {eeg_id}, sub-ID: {eeg_sub_id}")
        return None

    # Create result dictionary
    result = {
        'eeg_id': eeg_id,
        'eeg_sub_id': eeg_sub_id
    }

    # Add embeddings
    for i, val in enumerate(embeddings.flatten()):
        result[f'embedding_{i}'] = float(val)

    # Create output filename if not provided
    if output_file is None:
        output_file = f"{eeg_id}_{eeg_sub_id}_deep4_pool4_embeddings.csv"

    # Save to CSV
    df = pd.DataFrame([result])
    df.to_csv(output_file, index=False)

    elapsed_time = time.time() - start_time
    print(f"Embeddings extraction completed in {elapsed_time:.2f} seconds")
    print(f"Extracted {len(embeddings.flatten())} embeddings")
    print(f"Output saved to: {output_file}")

    return result


def main(test_mode=True, specific_eeg=None):
    """
    Main function to run the embedding extraction process

    Parameters:
        test_mode (bool): If True, run in test mode with sampled data
        specific_eeg (tuple): If provided, extract embeddings for this specific (eeg_id, eeg_sub_id) pair
    """
    # If a specific EEG is requested, extract embeddings just for that
    if specific_eeg:
        eeg_id, eeg_sub_id = specific_eeg
        print(f"Extracting embeddings for specific EEG: ID={eeg_id}, Sub-ID={eeg_sub_id}")
        output_file = f"{eeg_id}_{eeg_sub_id}_deep4_pool4_embeddings.csv"
        result = extract_specific_eeg_and_save(eeg_id, eeg_sub_id, output_file)
        return result

    # First, run a test on a single EEG
    print("Testing extraction on single EEG:")
    test_single_eeg()

    # Run extraction on dataset
    if test_mode:
        output_path = BRAINDECODE_OUT_PATH_TEST
        sample_size = TEST_SAMPLE_SIZE_BRAINDECODE
        print(f"\nRunning extraction in TEST MODE with {sample_size * 100:.1f}% of data")
    else:
        output_path = BRAINDECODE_OUT_PATH
        sample_size = None
        print(f"\nRunning extraction in PRODUCTION MODE with full dataset")

    embeddings_df, _ = extract_braindecode_embeddings(
        output_file=output_path,
        sample=sample_size
    )

    # Display summary
    print("\nExtraction complete.")
    print(f"Total samples processed: {len(embeddings_df)}")
    embedding_cols = [col for col in embeddings_df.columns if col.startswith('embedding_')]
    print(f"Total embeddings per sample: {len(embedding_cols)}")
    print(f"Output saved to: {output_path}")

    return embeddings_df


if __name__ == "__main__":
    # By default, run in test mode
    main(test_mode=True)
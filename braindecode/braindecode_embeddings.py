import os
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from const import TRAIN_EEG_DIR, TRAIN_CSV_PATH, OUTPUT_DIR, BRAINDECODE_OUT_PATH, BRAINDECODE_OUT_PATH_TEST, \
    TEST_SAMPLE_SIZE_BRAINDECODE

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Enable performance optimizations if CUDA is available
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    print("CUDA optimizations enabled")

# Set maximum sequence length for processing
MAX_SEQUENCE_LENGTH = 1000  # Adjust based on memory constraints


class EEGSubsampleDataset(Dataset):
    """Dataset for loading EEG data and their subsamples"""

    def __init__(self, data_df, eeg_dir, cache_size=100, max_seq_len=MAX_SEQUENCE_LENGTH):
        """
        Args:
            data_df: DataFrame containing eeg_id, eeg_sub_id, etc.
            eeg_dir: Directory containing EEG files
            cache_size: Number of EEGs to cache in memory
            max_seq_len: Maximum sequence length to use (will crop longer sequences)
        """
        self.data_df = data_df
        self.eeg_dir = eeg_dir
        self.max_seq_len = max_seq_len

        # Initialize cache for EEGs to reduce disk I/O
        self.cache = {}
        self.cache_size = cache_size
        self.cache_hits = 0
        self.cache_misses = 0

        # Ensure that we have both eeg_id and eeg_sub_id columns
        if 'eeg_id' not in data_df.columns or 'eeg_sub_id' not in data_df.columns:
            raise ValueError("DataFrame must contain 'eeg_id' and 'eeg_sub_id' columns")

        print(f"Dataset contains {len(data_df)} EEG subsamples")
        print(f"Unique EEG IDs: {len(data_df['eeg_id'].unique())}")

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]
        eeg_id = str(row['eeg_id'])
        eeg_sub_id = row['eeg_sub_id']

        # Cache key is just the EEG ID (we cache the full EEG data)
        cache_key = eeg_id

        # Try to get full EEG data from cache first
        if cache_key in self.cache:
            eeg_data = self.cache[cache_key]
            self.cache_hits += 1
        else:
            # Load EEG data from file
            parquet_path = os.path.join(self.eeg_dir, f"{eeg_id}.parquet")
            self.cache_misses += 1

            try:
                # Read parquet file
                table = pq.read_table(parquet_path)
                df = table.to_pandas()

                # Get EEG data
                eeg_data = df.values

                # Add to cache
                if len(self.cache) >= self.cache_size:
                    # Remove a random item if cache is full
                    remove_key = next(iter(self.cache))
                    del self.cache[remove_key]

                self.cache[cache_key] = eeg_data

            except Exception as e:
                print(f"Error loading {parquet_path}: {e}")
                # Return placeholder in case of error
                return {
                    'eeg_data': torch.zeros((1, 20, self.max_seq_len), dtype=torch.float32),
                    'eeg_id': eeg_id,
                    'eeg_sub_id': eeg_sub_id,
                    'error': True
                }

        try:
            # Extract subsample based on eeg_sub_id
            # This assumes that the subsamples are continuous segments
            subsample = self.extract_subsample(eeg_data, eeg_sub_id, row)

            # Preprocess EEG subsample
            # 1. Normalize
            if subsample.max() > subsample.min():
                subsample_norm = (subsample - subsample.min()) / (subsample.max() - subsample.min())
            else:
                subsample_norm = np.zeros_like(subsample)

            # 2. Transpose to (channels, time_steps)
            subsample_norm = subsample_norm.T

            # 3. Crop or pad to max_seq_len
            n_channels, n_times = subsample_norm.shape
            if n_times > self.max_seq_len:
                # Crop to max_seq_len
                subsample_norm = subsample_norm[:, :self.max_seq_len]
            elif n_times < self.max_seq_len:
                # Pad to max_seq_len
                padding = np.zeros((n_channels, self.max_seq_len - n_times))
                subsample_norm = np.concatenate([subsample_norm, padding], axis=1)

            # 4. Add batch dimension and convert to tensor
            eeg_tensor = torch.tensor(subsample_norm, dtype=torch.float32).unsqueeze(0)

            # Return all metadata along with preprocessed tensor
            result = {
                'eeg_data': eeg_tensor,
                'eeg_id': eeg_id,
                'eeg_sub_id': eeg_sub_id
            }

            # Add any other metadata from the row
            for col in self.data_df.columns:
                if col not in result and col not in ['eeg_id', 'eeg_sub_id']:
                    result[col] = row[col]

            return result

        except Exception as e:
            print(f"Error preprocessing EEG data for {eeg_id}, sub_id {eeg_sub_id}: {e}")
            # Return placeholder in case of error
            return {
                'eeg_data': torch.zeros((1, 20, self.max_seq_len), dtype=torch.float32),
                'eeg_id': eeg_id,
                'eeg_sub_id': eeg_sub_id,
                'error': True
            }

    def extract_subsample(self, eeg_data, eeg_sub_id, row):
        """
        Extract subsample from the full EEG data based on the sub-ID

        Args:
            eeg_data: Full EEG data
            eeg_sub_id: Subsample ID
            row: DataFrame row with metadata

        Returns:
            Subsample of the EEG data
        """
        # If there are specific columns in the dataframe that contain
        # information about how to extract the subsample (e.g., start, end indices),
        # use those values here

        # Method 1: If subsamples are segments with specific start/end indices
        if 'start_index' in row and 'end_index' in row:
            start_idx = row['start_index']
            end_idx = row['end_index']
            return eeg_data[start_idx:end_idx]

        # Method 2: If subsamples are of fixed size with a window size
        elif 'window_size' in row:
            window_size = row['window_size']
            start_idx = int(eeg_sub_id) * window_size
            end_idx = start_idx + window_size

            # Ensure indices are within bounds
            if start_idx >= len(eeg_data):
                start_idx = max(0, len(eeg_data) - window_size)
            end_idx = min(end_idx, len(eeg_data))

            return eeg_data[start_idx:end_idx]

        # Method 3: If there's an offset column specifying the subsample position
        elif 'offset_seconds' in row and 'sampling_rate' in row:
            offset_seconds = row['offset_seconds']
            sampling_rate = row['sampling_rate']
            window_size = int(2.0 * sampling_rate)  # Assuming 2-second windows
            start_idx = int(offset_seconds * sampling_rate)
            end_idx = start_idx + window_size

            # Ensure indices are within bounds
            if start_idx >= len(eeg_data):
                start_idx = max(0, len(eeg_data) - window_size)
            end_idx = min(end_idx, len(eeg_data))

            return eeg_data[start_idx:end_idx]

        # Default method: Use sub_id as an index for fixed-size windows
        # This assumes that sub_ids increment by 1 and represent consecutive segments
        else:
            # Determine a reasonable window size based on the data size
            total_samples = len(eeg_data)
            window_size = min(2000, total_samples // 10)  # Use at most 2000 samples or 1/10 of the data

            start_idx = int(eeg_sub_id) * window_size
            end_idx = min(start_idx + window_size, total_samples)

            # Ensure start index is in bounds
            if start_idx >= total_samples:
                start_idx = max(0, total_samples - window_size)
                end_idx = total_samples

            return eeg_data[start_idx:end_idx]


def custom_collate(batch):
    """
    Custom collate function to handle tensors of different sizes

    Args:
        batch: List of items returned by __getitem__

    Returns:
        Dictionary with batched data
    """
    # Filter out examples with errors
    valid_samples = [b for b in batch if 'error' not in b or not b['error']]

    if len(valid_samples) == 0:
        # Return empty batch if all samples have errors
        return {
            'eeg_data': [],
            'eeg_id': [],
            'eeg_sub_id': [],
            'error': True
        }

    # Group by keys
    result = {key: [] for key in valid_samples[0].keys()}

    # Collect all items by key
    for sample in valid_samples:
        for key, value in sample.items():
            result[key].append(value)

    return result


def extract_eeg_features(eeg_data):
    """
    Extract features from EEG data without using pre-trained models

    Args:
        eeg_data: EEG data tensor with shape (batch, channels, time)

    Returns:
        features: Extracted features as a numpy array
    """
    with torch.no_grad():
        try:
            # Ensure the data is properly shaped
            if eeg_data.dim() == 4:  # Batch of batches
                eeg_data = eeg_data.squeeze(0)

            if eeg_data.dim() == 3:  # Batch, channels, time
                # Get the features from each channel
                n_channels = eeg_data.shape[1]

                # Create a feature vector for each channel
                features = []
                for ch in range(n_channels):
                    channel_data = eeg_data[0, ch, :]  # Get data for this channel

                    # Calculate standard time-domain features
                    mean_val = torch.mean(channel_data).item()
                    std_val = torch.std(channel_data).item()
                    max_val = torch.max(channel_data).item()
                    min_val = torch.min(channel_data).item()

                    # Calculate frequency domain features using FFT
                    fft_vals = torch.abs(torch.fft.rfft(channel_data))
                    dom_freq = torch.argmax(fft_vals).item()

                    # Add features for this channel
                    features.extend([mean_val, std_val, max_val, min_val, dom_freq])

                # Add some cross-channel features
                cross_corr = torch.mean(torch.corrcoef(eeg_data[0])).item()
                features.append(cross_corr)

                # Create the embedding vector
                embeddings = np.array(features)

                return embeddings
            else:
                raise ValueError(f"Unexpected EEG data shape: {eeg_data.shape}")

        except Exception as e:
            raise ValueError(f"Error extracting features: {e}")


def extract_braindecode_embeddings(model_name='shallow', data_folder=TRAIN_EEG_DIR, train_csv=TRAIN_CSV_PATH,
                                   output_file=BRAINDECODE_OUT_PATH, batch_size=16, num_workers=2, sample=None):
    """
    Extract embeddings using direct feature extraction (not using Braindecode models)

    Args:
        model_name: Not used, kept for compatibility (feature extraction is model-independent)
        data_folder: Path to EEG data folder
        train_csv: Path to train CSV file
        output_file: Path to save embeddings
        batch_size: Batch size for processing
        num_workers: Number of worker processes
        sample: Fraction of data to use (for testing)

    Returns:
        DataFrame with embeddings for each EEG subsample
    """
    start_time = time.time()

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Load training data
    train_df = pd.read_csv(train_csv)
    print(f"Loaded training data with {len(train_df)} rows")

    # Apply sampling if requested
    if sample:
        train_df = train_df.sample(frac=sample, random_state=42)
        print(f"Using sample of {len(train_df)} rows ({sample * 100:.1f}% of data)")

    # Create dataset and dataloader for subsamples
    dataset = EEGSubsampleDataset(train_df, data_folder, cache_size=100, max_seq_len=MAX_SEQUENCE_LENGTH)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=2 if num_workers > 0 else None,
        collate_fn=custom_collate
    )

    # Extract embeddings
    all_embeddings = []

    for batch in tqdm(dataloader, desc="Extracting embeddings"):
        # Skip errors or empty batches
        if 'error' in batch and batch['error']:
            continue
        if len(batch['eeg_data']) == 0:
            continue

        # Process batch
        eeg_data_list = batch['eeg_data']
        eeg_ids = batch['eeg_id']
        eeg_sub_ids = batch['eeg_sub_id']

        # Extract embeddings for each EEG subsample in the batch
        for i in range(len(eeg_ids)):
            # Process each EEG subsample individually
            single_eeg = eeg_data_list[i].to(device)

            try:
                # Extract embeddings using our feature extraction function
                embeddings = extract_eeg_features(single_eeg.unsqueeze(0) if single_eeg.dim() == 3 else single_eeg)

                # Create a row with metadata
                row = {
                    'eeg_id': eeg_ids[i],
                    'eeg_sub_id': eeg_sub_ids[i]
                }

                # Add any other metadata from the batch
                for key in batch.keys():
                    if key not in ['eeg_data', 'eeg_id', 'eeg_sub_id', 'error'] and i < len(batch[key]):
                        row[key] = batch[key][i]

                # Add embedding values
                for j, val in enumerate(embeddings):
                    row[f'embedding_{j}'] = val if isinstance(val, (int, float)) else float(val)

                all_embeddings.append(row)

            except Exception as e:
                print(f"Error extracting embeddings for EEG {eeg_ids[i]}, sub_id {eeg_sub_ids[i]}: {e}")

    # Create DataFrame
    embeddings_df = pd.DataFrame(all_embeddings)
    print(f"Created embeddings for {len(embeddings_df)} EEG subsamples")

    # Report cache statistics
    cache_total = dataset.cache_hits + dataset.cache_misses
    if cache_total > 0:
        cache_hit_rate = dataset.cache_hits / cache_total * 100
        print(f"Cache hit rate: {cache_hit_rate:.2f}% ({dataset.cache_hits}/{cache_total})")

    # Save to CSV
    embeddings_df.to_csv(output_file, index=False)
    print(f"Saved embeddings to {output_file}")

    # Print elapsed time
    elapsed_time = time.time() - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    print(f"Total processing time: {int(minutes)} minutes {seconds:.2f} seconds")

    return embeddings_df, train_df.columns


def visualize_embeddings(embeddings_df, output_dir=OUTPUT_DIR):
    """
    Visualize the embeddings using t-SNE

    Args:
        embeddings_df: DataFrame with embeddings
        output_dir: Directory to save visualizations
    """
    try:
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
        import plotly.express as px
        import numpy as np

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        print("Creating t-SNE visualization...")

        # Get embedding columns
        embedding_cols = [col for col in embeddings_df.columns if col.startswith('embedding_')]

        # Add expert_consensus label from training data if available
        train_df = pd.read_csv(TRAIN_CSV_PATH)
        if 'expert_consensus' in train_df.columns:
            # Create mapping from (eeg_id, eeg_sub_id) to label
            label_map = {}
            for _, row in train_df.iterrows():
                key = (row['eeg_id'], row['eeg_sub_id'])
                label_map[key] = row['expert_consensus']

            # Add labels to embeddings
            labels = []
            for _, row in embeddings_df.iterrows():
                key = (row['eeg_id'], row['eeg_sub_id'])
                labels.append(label_map.get(key, 'Unknown'))

            embeddings_df['label'] = labels
        else:
            # Use EEG ID as label if no expert consensus available
            embeddings_df['label'] = embeddings_df['eeg_id'].astype(str)

        # Check for and remove NaN values
        X = embeddings_df[embedding_cols].values

        # Identify rows with NaN values
        nan_mask = np.isnan(X).any(axis=1)
        if nan_mask.any():
            print(f"Warning: Found {nan_mask.sum()} rows with NaN values. These will be removed for visualization.")
            # Filter out rows with NaN values
            valid_indices = ~nan_mask
            X = X[valid_indices]
            labels = np.array(labels)[valid_indices] if len(labels) > 0 else []

        # Select a sample if there are too many points
        if len(X) > 1000:
            print(f"Sampling 1000 points from {len(X)} for visualization")
            sample_indices = np.random.choice(len(X), 1000, replace=False)
            X = X[sample_indices]
            labels = np.array(labels)[sample_indices] if len(labels) > 0 else []

        # Check if we have enough data for visualization
        if len(X) < 5:
            print("Not enough valid data points for t-SNE visualization (minimum 5 required)")
            return None

        # Perform t-SNE
        print(f"Running t-SNE on {len(X)} data points")
        tsne = TSNE(n_components=3, random_state=42, perplexity=min(30, len(X) - 1))
        X_tsne = tsne.fit_transform(X)

        # Create and save 3D visualization
        df_vis = pd.DataFrame({
            'x': X_tsne[:, 0],
            'y': X_tsne[:, 1],
            'z': X_tsne[:, 2],
            'label': labels
        })

        fig = px.scatter_3d(
            df_vis, x='x', y='y', z='z',
            color='label',
            title='3D t-SNE of EEG Subsample Embeddings',
            opacity=0.7
        )

        fig.update_layout(
            scene=dict(
                xaxis_title='t-SNE 1',
                yaxis_title='t-SNE 2',
                zaxis_title='t-SNE 3'
            ),
            width=900,
            height=700
        )

        # Save as HTML
        html_path = os.path.join(output_dir, 'eeg_embeddings_3d.html')
        fig.write_html(html_path)
        print(f"3D visualization saved to {html_path}")

        return fig

    except ImportError:
        print("Visualization requires sklearn, matplotlib, and plotly. Skipping visualization.")
        return None
    except Exception as e:
        print(f"Error during visualization: {str(e)}")
        print("Skipping visualization.")
        return None


def compare_feature_sets(test_size=0.1):
    """
    Compare different feature extraction methods

    Args:
        test_size: Fraction of data to use for testing

    Returns:
        DataFrame with comparison results
    """
    print("Comparing different feature extraction methods...")

    # Feature extraction methods to compare
    methods = {
        'time_domain': {'n_features': None},
        'frequency_domain': {'n_features': None},
        'combined': {'n_features': None}
    }

    results = []

    # Implementation would vary based on specific features to extract
    # This is a stub for compatibility
    print("Feature comparison not implemented in this version")

    return pd.DataFrame(results)


def main(model_name='shallow', test_mode=True, visualize=True, compare=False):
    """
    Main function to extract EEG embeddings

    Args:
        model_name: Name of the model to use (not used, kept for compatibility)
        test_mode: Whether to run in test mode with a sample of data
        visualize: Whether to create visualizations
        compare: Whether to compare different feature extraction methods

    Returns:
        DataFrame with embeddings
    """
    # Set file paths based on mode
    if test_mode:
        output_file = BRAINDECODE_OUT_PATH_TEST
        sample_size = TEST_SAMPLE_SIZE_BRAINDECODE
        print(f"Running in TEST MODE with {sample_size * 100:.1f}% of data")
    else:
        output_file = BRAINDECODE_OUT_PATH
        sample_size = None
        print("Running in PRODUCTION MODE with full dataset")

    # Compare models if requested
    if compare:
        compare_results = compare_feature_sets(test_size=0.02)
        print("\nFeature comparison results:")
        print(compare_results)
        return compare_results

    # Choose optimal batch size and workers based on hardware
    if torch.cuda.is_available():
        batch_size = 16  # Larger batches for GPU
        num_workers = 4  # More workers for GPU
    else:
        batch_size = 1  # Set to 1 to avoid batching issues on CPU
        num_workers = 0  # Set to 0 to avoid multiprocessing issues on CPU

    # Extract embeddings
    embeddings_df, original_columns = extract_braindecode_embeddings(
        model_name=model_name,  # Not used but kept for compatibility
        output_file=output_file,
        batch_size=batch_size,
        num_workers=num_workers,
        sample=sample_size
    )

    # Create visualizations if requested
    if visualize and embeddings_df is not None and len(embeddings_df) > 0:
        visualize_embeddings(embeddings_df)

    # Display summary
    print("\nEmbedding extraction complete.")
    print(f"Total samples processed: {len(embeddings_df)}")

    # Count embedding dimensions
    embedding_cols = [col for col in embeddings_df.columns if col.startswith('embedding_')]
    print(f"Total embeddings per sample: {len(embedding_cols)}")
    print(f"Output saved to: {output_file}")

    # Display sample of the extracted embeddings
    print("\nSample of extracted embeddings:")
    if not embeddings_df.empty and len(embedding_cols) > 0:
        print(embeddings_df[['eeg_id', 'eeg_sub_id'] + embedding_cols[:5]].head())
    else:
        print("No embeddings extracted or empty result.")

    print("EEG embedding extraction complete!")
    return embeddings_df


def test_mode_run():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    start_time = time.time()

    # Run test or comparison
    # By default, run in test mode with 'shallow' model
    # Change to main(model_name='shallow', test_mode=False) to run on the full dataset
    # Change to main(model_name='deep4', test_mode=True) to use different model
    # Change to main(compare=True) to compare models
    embeddings_df = main(model_name='shallow', test_mode=True, visualize=True, compare=True)

    elapsed_time = time.time() - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    print(f"Complete process finished in {int(minutes)} minutes {seconds:.2f} seconds")


if __name__ == "__main__":
    test_mode_run()
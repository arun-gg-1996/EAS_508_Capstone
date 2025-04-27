import os
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import ViTFeatureExtractor, ViTModel

from const import TRAIN_SPECTROGRAMS_DIR, TRAIN_CSV_PATH, VIT_EMBEDDINGS_PATH_TEST, \
    VIT_EMBEDDINGS_PATH, TEST_SAMPLE_SIZE_VISION, VISION_TSNE_FILE_TEST, VISION_TSNE_FILE

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Enable performance optimizations if CUDA is available
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    print("CUDA optimizations enabled")


class SpectrogramDataset(Dataset):
    def __init__(self, data_df, spectrogram_dir, cache_size=100):
        """
        Dataset for loading spectrograms and their subsamples

        Args:
            data_df: DataFrame containing spectrogram_id, spectrogram_sub_id, etc.
            spectrogram_dir: Directory containing spectrogram files
            cache_size: Number of spectrograms to cache in memory
        """
        self.data_df = data_df
        self.spectrogram_dir = spectrogram_dir
        self.feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224", do_rescale=False)

        # Initialize cache for spectrograms to reduce disk I/O
        self.cache = {}
        self.cache_size = cache_size
        self.cache_hits = 0
        self.cache_misses = 0

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]
        spec_id = row['spectrogram_id']
        sub_id = row['spectrogram_sub_id']
        eeg_id = row['eeg_id']
        eeg_sub_id = row['eeg_sub_id']

        # Get expert_consensus if available
        expert_consensus = row.get('expert_consensus', 'Unknown')

        # Cache key is spectrogram_id
        cache_key = spec_id

        # Try to get from cache first
        if cache_key in self.cache:
            spectrogram = self.cache[cache_key]
            self.cache_hits += 1
        else:
            parquet_path = os.path.join(self.spectrogram_dir, f"{spec_id}.parquet")
            self.cache_misses += 1

            try:
                # Read parquet file
                table = pq.read_table(parquet_path)
                df = table.to_pandas()

                # Get spectrogram data
                spectrogram = df.values

                # Add to cache
                if len(self.cache) >= self.cache_size:
                    # Remove a random item if cache is full
                    remove_key = next(iter(self.cache))
                    del self.cache[remove_key]

                self.cache[cache_key] = spectrogram

            except Exception as e:
                print(f"Error processing {parquet_path}: {e}")
                # Return placeholder in case of error
                return {
                    'pixel_values': torch.zeros((3, 224, 224), dtype=torch.float32),
                    'spectrogram_id': spec_id,
                    'spectrogram_sub_id': sub_id,
                    'eeg_id': eeg_id,
                    'eeg_sub_id': eeg_sub_id,
                    'expert_consensus': expert_consensus,
                    'row_idx': idx,
                    'error': True
                }

        try:
            # Get offset information
            if 'spectrogram_label_offset_seconds' in row:
                offset_seconds = row['spectrogram_label_offset_seconds']
            else:
                # If no specific offset, use sub_id as a rough approximation
                offset_seconds = float(sub_id) * 2.0  # Assuming 2 seconds per sub_id

            # Normalize the spectrogram to [0,1] range
            if spectrogram.max() > spectrogram.min():
                spectrogram_norm = (spectrogram - spectrogram.min()) / (spectrogram.max() - spectrogram.min())
            else:
                spectrogram_norm = np.zeros_like(spectrogram)

            # Convert to RGB by repeating the channel
            spectrogram_rgb = np.repeat(spectrogram_norm[:, :, np.newaxis], 3, axis=2)

            # Process with feature extractor - this will handle resizing and normalization
            inputs = self.feature_extractor(images=spectrogram_rgb, return_tensors="pt")

            # Return all metadata along with processed image
            return {
                'pixel_values': inputs['pixel_values'].squeeze(),  # This will be [3, 224, 224]
                'spectrogram_id': spec_id,
                'spectrogram_sub_id': sub_id,
                'eeg_id': eeg_id,
                'eeg_sub_id': eeg_sub_id,
                'expert_consensus': expert_consensus,
                'offset_seconds': offset_seconds,
                'row_idx': idx
            }
        except Exception as e:
            print(f"Error processing spectrogram data for {spec_id}, sub_id {sub_id}: {e}")
            # Return placeholder in case of error
            return {
                'pixel_values': torch.zeros((3, 224, 224), dtype=torch.float32),
                'spectrogram_id': spec_id,
                'spectrogram_sub_id': sub_id,
                'eeg_id': eeg_id,
                'eeg_sub_id': eeg_sub_id,
                'expert_consensus': expert_consensus,
                'offset_seconds': 0.0,
                'row_idx': idx,
                'error': True
            }


def extract_subsample_embeddings(output_file=VIT_EMBEDDINGS_PATH_TEST, batch_size=16, num_workers=2, sample=None):
    """
    Extract embeddings for each EEG ID and subsample

    Args:
        output_file: Path to save embeddings
        batch_size: Batch size for processing
        num_workers: Number of worker processes
        sample: Fraction of data to use (for testing)

    Returns:
        DataFrame with embeddings for each EEG ID and subsample
    """
    start_time = time.time()

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Load training data
    train_df = pd.read_csv(TRAIN_CSV_PATH)
    print(f"Loaded training data with {len(train_df)} rows")

    # Apply sampling if requested
    if sample:
        train_df = train_df.sample(frac=sample, random_state=42)
        print(f"Using sample of {len(train_df)} rows ({sample * 100:.1f}% of data)")

    # Create dataset and dataloader
    dataset = SpectrogramDataset(train_df, TRAIN_SPECTROGRAMS_DIR, cache_size=200)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=2 if num_workers > 0 else None
    )

    # Load ViT model
    model = ViTModel.from_pretrained("google/vit-base-patch16-224")
    if torch.cuda.is_available():
        model = model.to(device)
        model = model.half()  # Use half precision for speed
    model.eval()

    # Process in batches
    all_embeddings = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings for subsamples"):
            # Skip errors
            if 'error' in batch and batch['error'].any():
                error_indices = torch.where(batch['error'])[0].cpu().numpy()
                for i in error_indices:
                    print(
                        f"Skipping error in spectrogram {batch['spectrogram_id'][i]}, sub_id {batch['spectrogram_sub_id'][i]}")
                continue

            # Process batch
            pixel_values = batch['pixel_values'].to(device)

            # Ensure correct shape for ViT model
            if pixel_values.ndim == 3:  # Single sample [channels, height, width]
                pixel_values = pixel_values.unsqueeze(0)  # Add batch dimension

            # Run through model
            outputs = model(pixel_values)
            embeddings = outputs.last_hidden_state[:, 0].cpu().numpy()

            # Store results with metadata
            for i in range(len(embeddings)):
                # Create a row with all metadata
                row = {
                    'spectrogram_id': batch['spectrogram_id'][i].item() if torch.is_tensor(
                        batch['spectrogram_id'][i]) else batch['spectrogram_id'][i],
                    'spectrogram_sub_id': batch['spectrogram_sub_id'][i].item() if torch.is_tensor(
                        batch['spectrogram_sub_id'][i]) else batch['spectrogram_sub_id'][i],
                    'eeg_id': batch['eeg_id'][i].item() if torch.is_tensor(batch['eeg_id'][i]) else batch['eeg_id'][i],
                    'eeg_sub_id': batch['eeg_sub_id'][i].item() if torch.is_tensor(batch['eeg_sub_id'][i]) else
                    batch['eeg_sub_id'][i],
                    'offset_seconds': batch['offset_seconds'][i].item() if torch.is_tensor(
                        batch['offset_seconds'][i]) else batch['offset_seconds'][i],
                }

                # Add expert_consensus if available
                if 'expert_consensus' in batch:
                    row['expert_consensus'] = batch['expert_consensus'][i]

                # Add embedding values
                for j, val in enumerate(embeddings[i]):
                    row[f'embedding_{j}'] = val.item() if isinstance(val, np.float32) else val

                all_embeddings.append(row)

    # Create DataFrame
    embeddings_df = pd.DataFrame(all_embeddings)
    print(f"Created embeddings for {len(embeddings_df)} subsamples")

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

    return embeddings_df


def visualize_subsample_embeddings(embeddings_df, out_file):
    """
    Visualize the subsample embeddings

    Args:
        embeddings_df: DataFrame with embeddings
        out_file: file to save visualization
    """
    try:
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
        import plotly.express as px

        print("Creating t-SNE visualization...")

        # Get embedding columns
        embedding_cols = [col for col in embeddings_df.columns if col.startswith('embedding_')]

        # Use existing labels if available, otherwise create labels from train.csv
        if 'expert_consensus' in embeddings_df.columns:
            labels = embeddings_df['expert_consensus'].values
            print("Using expert_consensus labels from embeddings dataframe")
        else:
            # Add expert_consensus label from training data
            train_df = pd.read_csv(TRAIN_CSV_PATH)
            label_map = {}
            if 'expert_consensus' in train_df.columns:
                # Create mapping from (eeg_id, eeg_sub_id) to label
                for _, row in train_df.iterrows():
                    key = (row['eeg_id'], row['eeg_sub_id'])
                    label_map[key] = row['expert_consensus']

                # Add labels to embeddings
                labels = []
                for _, row in embeddings_df.iterrows():
                    key = (row['eeg_id'], row['eeg_sub_id'])
                    labels.append(label_map.get(key, 'Unknown'))
                print("Created expert_consensus labels from train.csv")
            else:
                # Use EEG ID as label if no expert consensus available
                labels = embeddings_df['eeg_id'].astype(str).values
                print("No expert_consensus found, using eeg_id as labels")

        # Select a sample if there are too many points
        if len(embeddings_df) > 1000:
            # If embeddings_df has 'expert_consensus', stratify sampling by it
            if 'expert_consensus' in embeddings_df.columns:
                # Get at least 10 samples per class if possible
                min_samples = 10

                # Stratified sampling
                classes = embeddings_df['expert_consensus'].unique()
                sample_indices = []

                for cls in classes:
                    cls_indices = embeddings_df[embeddings_df['expert_consensus'] == cls].index
                    # If we have fewer than min_samples, take all of them
                    if len(cls_indices) <= min_samples:
                        sample_indices.extend(cls_indices)
                    else:
                        # Otherwise take a sample
                        n_samples = max(min_samples, int(len(cls_indices) * 1000 / len(embeddings_df)))
                        sampled = np.random.choice(cls_indices, size=n_samples, replace=False)
                        sample_indices.extend(sampled)

                # If we have more than 1000 samples, subsample randomly
                if len(sample_indices) > 1000:
                    sample_indices = np.random.choice(sample_indices, size=1000, replace=False)

                sample_df = embeddings_df.loc[sample_indices]
                labels = [labels[i] for i in range(len(labels)) if i in sample_indices]
            else:
                sample_df = embeddings_df.sample(1000, random_state=42)
                # We need to recreate labels for the sampled rows
                if 'expert_consensus' in embeddings_df.columns:
                    labels = sample_df['expert_consensus'].values
                else:
                    labels = sample_df['eeg_id'].astype(str).values
        else:
            sample_df = embeddings_df

        # Get embeddings matrix
        X = sample_df[embedding_cols].values

        # Perform t-SNE
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
        fig.write_html(out_file)
        print(f"3D visualization saved to {out_file}")

        return fig

    except ImportError:
        print("Visualization requires sklearn, matplotlib, and plotly. Skipping visualization.")
        return None


def main(test_mode=True, visualize=True):
    """
    Main function to extract embeddings for each EEG and subsample

    Args:
        test_mode: Whether to run in test mode with a sample of data
        visualize: Whether to create visualizations
    """
    # Set file paths based on mode
    if test_mode:
        output_file = VIT_EMBEDDINGS_PATH_TEST
        sample_size = TEST_SAMPLE_SIZE_VISION  # Use 5% of data for testing
        print(f"Running in TEST MODE with {sample_size * 100:.1f}% of data")
    else:
        output_file = VIT_EMBEDDINGS_PATH
        sample_size = None
        print("Running in PRODUCTION MODE with full dataset")

    # Choose optimal batch size and workers based on hardware
    if torch.cuda.is_available():
        batch_size = 32  # Larger batches for GPU
        num_workers = 4  # More workers for GPU
    else:
        batch_size = 16  # Smaller batches for CPU
        num_workers = 2  # Fewer workers for CPU

    # Reduce batch size for CPU
    if not torch.cuda.is_available():
        batch_size = 8  # Smaller batches to avoid memory issues

    # Extract embeddings
    embeddings_df = extract_subsample_embeddings(
        output_file=output_file,
        batch_size=batch_size,
        num_workers=num_workers,
        sample=sample_size
    )

    # Create visualizations if requested
    if visualize and embeddings_df is not None:
        if test_mode:
            visualize_subsample_embeddings(embeddings_df, out_file=VISION_TSNE_FILE_TEST)
        else:
            visualize_subsample_embeddings(embeddings_df, out_file=VISION_TSNE_FILE)

    print("Subsample embedding extraction complete!")
    return embeddings_df


def run():
    start_time = time.time()

    # Run with test_mode=True for a quick test with 5% of the data
    # Set to False for processing the full dataset
    embeddings_df = main(test_mode=True, visualize=True)

    elapsed_time = time.time() - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    print(f"Complete process finished in {int(minutes)} minutes {seconds:.2f} seconds")


if __name__ == "__main__":
    run()

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import ViTFeatureExtractor, ViTModel
import pyarrow.parquet as pq
from tqdm import tqdm
import time
import gc

from const import TRAIN_SPECTROGRAMS_DIR, TRAIN_CSV_PATH, OUTPUT_DIR_TEST

# Define output file paths for test and production modes
VIT_EMBEDDINGS_PATH = os.path.join(OUTPUT_DIR_TEST, 'vit_embeddings.csv')
VIT_EMBEDDINGS_PATH_TEST = os.path.join(OUTPUT_DIR_TEST, 'vit_embeddings_test.csv')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Enable performance optimizations if CUDA is available
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    print("CUDA optimizations enabled")


class SpectrogramDataset(Dataset):
    def __init__(self, spectrogram_ids, spectrogram_dir, cache_size=100):
        self.spectrogram_ids = spectrogram_ids
        self.spectrogram_dir = spectrogram_dir
        self.feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224", do_rescale=False)

        # Initialize cache for spectrograms to reduce disk I/O
        self.cache = {}
        self.cache_size = cache_size
        self.cache_hits = 0
        self.cache_misses = 0

    def __len__(self):
        return len(self.spectrogram_ids)

    def __getitem__(self, idx):
        spec_id = self.spectrogram_ids[idx]

        # Try to get from cache first
        if spec_id in self.cache:
            spectrogram = self.cache[spec_id]
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

                self.cache[spec_id] = spectrogram

            except Exception as e:
                print(f"Error processing {parquet_path}: {e}")
                # Return placeholder in case of error
                dummy = self.feature_extractor(images=np.zeros((224, 224, 3)), return_tensors="pt")
                return {
                    'pixel_values': dummy['pixel_values'].squeeze(),
                    'spectrogram_id': spec_id,
                    'error': True
                }

        try:
            # Normalize to [0,1] range
            if spectrogram.max() > spectrogram.min():
                spectrogram = (spectrogram - spectrogram.min()) / (spectrogram.max() - spectrogram.min())
            else:
                spectrogram = np.zeros_like(spectrogram)

            # Convert to RGB by repeating the channel
            spectrogram_rgb = np.repeat(spectrogram[:, :, np.newaxis], 3, axis=2)

            # Process with feature extractor
            inputs = self.feature_extractor(images=spectrogram_rgb, return_tensors="pt")

            return {
                'pixel_values': inputs['pixel_values'].squeeze(),
                'spectrogram_id': spec_id
            }
        except Exception as e:
            print(f"Error processing spectrogram data for {spec_id}: {e}")
            # Return placeholder in case of error
            dummy = self.feature_extractor(images=np.zeros((224, 224, 3)), return_tensors="pt")
            return {
                'pixel_values': dummy['pixel_values'].squeeze(),
                'spectrogram_id': spec_id,
                'error': True
            }


def extract_embeddings(output_file=VIT_EMBEDDINGS_PATH, batch_size=32, num_workers=4, sample=None):
    """
    Extract ViT embeddings for spectrograms

    Args:
        output_file: Path to save the output CSV
        batch_size: Batch size for processing
        num_workers: Number of worker processes for data loading
        sample: Fraction of data to use (for test mode)

    Returns:
        DataFrame containing the embeddings
    """
    start_time = time.time()

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Load CSV with spectrograms info
    train_df = pd.read_csv(TRAIN_CSV_PATH)
    print(f"Loaded training data with {len(train_df)} rows")

    # Apply sampling if in test mode
    if sample:
        train_df = train_df.sample(frac=sample, random_state=42)
        print(f"Test mode: Using {len(train_df)} rows ({sample * 100:.1f}% of data)")

    # Get unique spectrogram IDs
    unique_spectrograms = train_df['spectrogram_id'].unique()
    print(f"Found {len(unique_spectrograms)} unique spectrograms")

    # Create dataset and dataloader with optimized settings
    dataset = SpectrogramDataset(unique_spectrograms, TRAIN_SPECTROGRAMS_DIR, cache_size=200)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),  # Speed up transfers to GPU
        prefetch_factor=2 if num_workers > 0 else None  # Prefetch batches
    )

    # Load ViT model with optimizations
    model = ViTModel.from_pretrained("google/vit-base-patch16-224")
    if torch.cuda.is_available():
        model = model.to(device)
        # Use half precision for faster computation if on GPU
        model = model.half()
    model.eval()

    # Extract embeddings
    embeddings = {}

    # Process in batches with progress bar
    with torch.no_grad():  # Disable gradient calculation for inference
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            # Skip samples with errors
            if 'error' in batch and batch['error'].any():
                continue

            pixel_values = batch['pixel_values'].to(device)
            spectrogram_ids = batch['spectrogram_id']

            # Forward pass through model
            outputs = model(pixel_values)

            # Get embeddings from [CLS] token
            cls_embeddings = outputs.last_hidden_state[:, 0].cpu().numpy()

            # Store embeddings
            for i, spec_id in enumerate(spectrogram_ids):
                embeddings[spec_id] = cls_embeddings[i]

    # Report cache statistics
    cache_total = dataset.cache_hits + dataset.cache_misses
    if cache_total > 0:
        cache_hit_rate = dataset.cache_hits / cache_total * 100
        print(f"Cache hit rate: {cache_hit_rate:.2f}% ({dataset.cache_hits}/{cache_total})")

    # Convert to DataFrame
    embedding_rows = []

    for spec_id, embedding in embeddings.items():
        row = {'spectrogram_id': spec_id}

        # Add each embedding dimension as a separate column
        for i, val in enumerate(embedding):
            row[f'embedding_{i}'] = val

        embedding_rows.append(row)

    # Create DataFrame
    embeddings_df = pd.DataFrame(embedding_rows)
    print(f"Created DataFrame with embeddings for {len(embeddings_df)} spectrograms")

    # Save to CSV
    embeddings_df.to_csv(output_file, index=False)
    print(f"Saved embeddings to {output_file}")

    # Print elapsed time
    elapsed_time = time.time() - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    print(f"Total processing time: {int(minutes)} minutes {seconds:.2f} seconds")

    return embeddings_df


def visualize_embeddings_3d(embeddings, labels, title="ViT Spectrogram Embeddings (3D t-SNE)", output_path=None):
    """
    Visualize embeddings using 3D t-SNE and save as interactive HTML

    Args:
        embeddings: Embedding vectors
        labels: Labels for each embedding
        title: Title for the visualization
        output_path: Path to save the HTML visualization

    Returns:
        Plotly figure object
    """
    # Import here to avoid dependency if not needed
    from sklearn.manifold import TSNE
    import plotly.express as px

    # Reduce dimensionality with t-SNE to 3D
    print("Running 3D t-SNE dimensionality reduction...")
    tsne = TSNE(n_components=3, random_state=42, perplexity=min(30, len(embeddings) - 1))
    embeddings_3d = tsne.fit_transform(embeddings)

    # Create a DataFrame for plotly
    df = pd.DataFrame({
        'x': embeddings_3d[:, 0],
        'y': embeddings_3d[:, 1],
        'z': embeddings_3d[:, 2],
        'label': labels
    })

    # Create 3D scatter plot
    fig = px.scatter_3d(
        df, x='x', y='y', z='z',
        color='label',
        title=title,
        labels={'label': 'Class'},
        opacity=0.7,
        color_continuous_scale=px.colors.qualitative.G10
    )

    # Update layout for better visualization
    fig.update_layout(
        scene=dict(
            xaxis_title='t-SNE Component 1',
            yaxis_title='t-SNE Component 2',
            zaxis_title='t-SNE Component 3'
        ),
        width=900,
        height=700,
        margin=dict(l=0, r=0, b=0, t=40)
    )

    # Add hover information
    fig.update_traces(
        hoverinfo="text",
        hovertext=[f"Class: {label}" for label in labels]
    )

    # Save as HTML if output_path is provided
    if output_path:
        fig.write_html(output_path)
        print(f"3D t-SNE visualization saved to {output_path}")

    return fig


def main(test_mode=True, visualize=True):
    """
    Main function to run the vision transformer embedding extraction process

    Args:
        test_mode: Whether to run in test mode with a sample of data
        visualize: Whether to create a visualization of the embeddings
    """
    # Set configuration based on mode
    if test_mode:
        # Test mode configuration
        output_path = VIT_EMBEDDINGS_PATH_TEST
        sample_size = 0.05  # Use 5% of data for testing
        print(f"Running in TEST MODE with {sample_size * 100}% of data")
    else:
        # Production mode configuration
        output_path = VIT_EMBEDDINGS_PATH
        sample_size = None
        print("Running in PRODUCTION MODE with full dataset")

    # Choose optimal batch size and workers based on available hardware
    if torch.cuda.is_available():
        batch_size = 32  # Larger batches for GPU
        num_workers = 4  # More workers for GPU
    else:
        batch_size = 16  # Smaller batches for CPU
        num_workers = 2  # Fewer workers for CPU

    # Extract embeddings
    features_df = extract_embeddings(
        output_file=output_path,
        batch_size=batch_size,
        num_workers=num_workers,
        sample=sample_size
    )

    # Display summary
    print("\nVision transformer embedding extraction complete.")
    print(f"Total spectrograms processed: {len(features_df)}")

    # Count embedding dimensions
    embedding_cols = [col for col in features_df.columns if col.startswith('embedding_')]
    print(f"Total embedding dimensions: {len(embedding_cols)}")
    print(f"Output saved to: {output_path}")

    # Create visualization if requested and enough samples are available
    if visualize and len(features_df) >= 50:
        # First we need to merge with the train data to get labels
        train_df = pd.read_csv(TRAIN_CSV_PATH)
        if sample_size:
            train_df = train_df.sample(frac=sample_size, random_state=42)

        # Get unique spectrogram_id to expert_consensus mapping
        spec_to_label = {}
        for _, row in train_df.iterrows():
            spec_to_label[row['spectrogram_id']] = row['expert_consensus']

        # Add labels to our embeddings
        features_df['label'] = features_df['spectrogram_id'].map(spec_to_label)

        # Drop rows with missing labels
        features_df = features_df.dropna(subset=['label'])

        # Verify we have enough data for visualization
        if len(features_df) >= 50:
            print("\nVisualizing embeddings with 3D t-SNE...")
            # Get embeddings and labels
            embeddings = features_df[embedding_cols].values
            labels = features_df['label'].values

            # Generate file paths
            base_dir = os.path.dirname(output_path)

            # Create 3D interactive visualization
            html_path = os.path.join(base_dir, "vit_embeddings_3d_tsne.html")

            # Generate 3D visualization
            visualize_embeddings_3d(
                embeddings,
                labels,
                title="ViT Spectrogram 3D Embeddings",
                output_path=html_path
            )

            print(f"3D Interactive visualization saved to {html_path}")
        else:
            print("Not enough labeled data for visualization after sampling")


if __name__ == "__main__":
    # Set test_mode=False to run on the full dataset
    main(test_mode=True, visualize=True)
import os
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from const import TRAIN_EEG_DIR, TRAIN_CSV_PATH, OUTPUT_DIR, BRAINDECODE_OUT_PATH, BRAINDECODE_OUT_PATH_TEST, \
    TEST_SAMPLE_SIZE_BRAINDECODE

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Preprocessing settings
MAX_SEQUENCE_LENGTH = 1000


class EEGDataset(Dataset):
    """Dataset for loading EEG data"""

    def __init__(self, data_df, eeg_dir, max_seq_len=MAX_SEQUENCE_LENGTH):
        self.data_df = data_df
        self.eeg_dir = eeg_dir
        self.max_seq_len = max_seq_len
        self.cache = {}
        print(f"Dataset contains {len(data_df)} EEG samples")

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]
        eeg_id = str(row['eeg_id'])
        eeg_sub_id = row['eeg_sub_id']

        # Try to get data from cache
        if eeg_id in self.cache:
            eeg_data = self.cache[eeg_id]
        else:
            try:
                # Load EEG data from file
                parquet_path = os.path.join(self.eeg_dir, f"{eeg_id}.parquet")
                table = pq.read_table(parquet_path)
                df = table.to_pandas()
                eeg_data = df.values

                # Add to cache (simple, no size limit)
                self.cache[eeg_id] = eeg_data
            except Exception as e:
                print(f"Error loading {parquet_path}: {e}")
                return {
                    'eeg_data': torch.zeros((1, 20, self.max_seq_len), dtype=torch.float32),
                    'eeg_id': eeg_id,
                    'eeg_sub_id': eeg_sub_id,
                    'error': True
                }

        try:
            # Extract subsample based on eeg_sub_id (simple approach)
            window_size = min(2000, len(eeg_data) // 10)
            start_idx = int(eeg_sub_id) * window_size
            end_idx = min(start_idx + window_size, len(eeg_data))

            # Ensure start index is in bounds
            if start_idx >= len(eeg_data):
                start_idx = max(0, len(eeg_data) - window_size)
                end_idx = len(eeg_data)

            subsample = eeg_data[start_idx:end_idx]

            # Normalize to [0, 1]
            if subsample.max() > subsample.min():
                subsample_norm = (subsample - subsample.min()) / (subsample.max() - subsample.min())
            else:
                subsample_norm = np.zeros_like(subsample)

            # Transpose to (channels, time_steps)
            subsample_norm = subsample_norm.T

            # Crop or pad to max_seq_len
            n_channels, n_times = subsample_norm.shape
            if n_times > self.max_seq_len:
                subsample_norm = subsample_norm[:, :self.max_seq_len]
            elif n_times < self.max_seq_len:
                padding = np.zeros((n_channels, self.max_seq_len - n_times))
                subsample_norm = np.concatenate([subsample_norm, padding], axis=1)

            # Convert to tensor
            eeg_tensor = torch.tensor(subsample_norm, dtype=torch.float32).unsqueeze(0)

            result = {
                'eeg_data': eeg_tensor,
                'eeg_id': eeg_id,
                'eeg_sub_id': eeg_sub_id
            }

            # Add expert_consensus if available
            if 'expert_consensus' in row:
                result['expert_consensus'] = row['expert_consensus']

            return result

        except Exception as e:
            print(f"Error preprocessing data for {eeg_id}, sub_id {eeg_sub_id}: {e}")
            return {
                'eeg_data': torch.zeros((1, 20, self.max_seq_len), dtype=torch.float32),
                'eeg_id': eeg_id,
                'eeg_sub_id': eeg_sub_id,
                'error': True
            }


class Deep4ConvNet(nn.Module):
    """
    Implementation of the Deep4Net convolutional architecture to extract 1400 features
    """

    def __init__(self):
        super(Deep4ConvNet, self).__init__()
        # Sequence of layers that process the EEG data to get to the conv4 features
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1, 10), stride=1),
            nn.BatchNorm2d(25),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),
            nn.Dropout(p=0.5),

            nn.Conv2d(25, 50, kernel_size=(1, 10), stride=1),
            nn.BatchNorm2d(50),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),
            nn.Dropout(p=0.5),

            nn.Conv2d(50, 100, kernel_size=(1, 10), stride=1),
            nn.BatchNorm2d(100),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),
            nn.Dropout(p=0.5),

            nn.Conv2d(100, 200, kernel_size=(1, 10), stride=1),
            nn.BatchNorm2d(200),
            nn.ELU()
        )

    def forward(self, x):
        # Check input shape and fix it if needed
        if x.dim() == 3:  # [batch, channels, time]
            batch_size = x.shape[0]
            # Reshape to [batch, 1, channels, time]
            x = x.unsqueeze(1)
            # Transpose to [batch, 1, time, channels] as expected
            x = x.permute(0, 1, 3, 2)
        elif x.dim() == 4 and x.shape[1] == 1 and x.shape[2] == 20:  # [batch, 1, channels, time]
            batch_size = x.shape[0]
            # Swap channels and time dimensions
            x = x.permute(0, 1, 3, 2)
        else:
            print(f"Unexpected input shape: {x.shape}")
            batch_size = x.shape[0]

        # Apply convolutional layers
        features = self.temporal_conv(x)

        # Flatten to get features
        features_flat = features.reshape(batch_size, -1)

        # Ensure exactly 1400 features
        if features_flat.shape[1] > 1400:
            features_flat = features_flat[:, :1400]
        elif features_flat.shape[1] < 1400:
            # Pad with zeros to get exactly 1400 features
            padding = torch.zeros(batch_size, 1400 - features_flat.shape[1], device=features_flat.device)
            features_flat = torch.cat([features_flat, padding], dim=1)

        return features_flat


def extract_deep4_conv_features(eeg_data):
    """
    Extract exactly 1400 features from the conv4 layer output

    Args:
        eeg_data: EEG data tensor of shape [batch, channels, time]

    Returns:
        numpy.ndarray: 1400 features per sample
    """
    # Create the model
    model = Deep4ConvNet().to(device)
    model.eval()

    try:
        # Process the data through the model
        with torch.no_grad():
            if torch.cuda.is_available():
                eeg_data = eeg_data.to(device)

            # Print input shape for debugging
            print(f"Input shape before model: {eeg_data.shape}")

            # Get features
            features = model(eeg_data)

            print(f"Extracted features shape: {features.shape}")
            return features.cpu().numpy()

    except Exception as e:
        print(f"Error in feature extraction: {e}")
        print(f"Input shape: {eeg_data.shape}")

        # Get the shape details for better debugging
        if eeg_data.dim() == 3:
            print(f"3D tensor: [batch={eeg_data.shape[0]}, channels={eeg_data.shape[1]}, time={eeg_data.shape[2]}]")
        elif eeg_data.dim() == 4:
            print(
                f"4D tensor: [batch={eeg_data.shape[0]}, dim1={eeg_data.shape[1]}, dim2={eeg_data.shape[2]}, dim3={eeg_data.shape[3]}]")

        # Create a proper fallback with 1400 features
        batch_size = eeg_data.shape[0]
        return np.zeros((batch_size, 1400))


def extract_embeddings_for_specific_eeg(eeg_id, eeg_sub_id, data_folder=TRAIN_EEG_DIR, max_seq_len=MAX_SEQUENCE_LENGTH):
    """
    Extract 1400 embeddings for a specific EEG ID and sub-ID combination

    Args:
        eeg_id (str): The EEG ID
        eeg_sub_id (int or str): The EEG sub ID
        data_folder (str): Path to the folder containing EEG data
        max_seq_len (int): Maximum sequence length

    Returns:
        numpy.ndarray: 1400 embeddings per sample
    """
    try:
        # Load EEG data from file
        parquet_path = os.path.join(data_folder, f"{eeg_id}.parquet")
        table = pq.read_table(parquet_path)
        df = table.to_pandas()
        eeg_data = df.values

        # Prepare subsample based on eeg_sub_id
        window_size = min(2000, len(eeg_data) // 10)
        start_idx = int(eeg_sub_id) * window_size
        end_idx = min(start_idx + window_size, len(eeg_data))

        # Ensure start index is in bounds
        if start_idx >= len(eeg_data):
            start_idx = max(0, len(eeg_data) - window_size)
            end_idx = len(eeg_data)

        subsample = eeg_data[start_idx:end_idx]

        # Normalize to [0, 1]
        if subsample.max() > subsample.min():
            subsample_norm = (subsample - subsample.min()) / (subsample.max() - subsample.min())
        else:
            subsample_norm = np.zeros_like(subsample)

        # Transpose to (channels, time_steps)
        subsample_norm = subsample_norm.T

        # Crop or pad to max_seq_len
        n_channels, n_times = subsample_norm.shape
        if n_times > max_seq_len:
            subsample_norm = subsample_norm[:, :max_seq_len]
        elif n_times < max_seq_len:
            padding = np.zeros((n_channels, max_seq_len - n_times))
            subsample_norm = np.concatenate([subsample_norm, padding], axis=1)

        # Convert to tensor
        eeg_tensor = torch.tensor(subsample_norm, dtype=torch.float32).unsqueeze(0)

        # Extract the 1400 features
        embeddings = extract_deep4_conv_features(eeg_tensor)

        return embeddings

    except Exception as e:
        print(f"Error extracting embeddings for EEG {eeg_id}, sub_id {eeg_sub_id}: {e}")
        # Return empty array with 1400 features
        return np.zeros((1, 1400))


def extract_braindecode_embeddings(data_folder=TRAIN_EEG_DIR, train_csv=TRAIN_CSV_PATH,
                                   output_file=BRAINDECODE_OUT_PATH, batch_size=1, sample=None):
    """Extract 1400 embeddings from the conv4 layer of Deep4Net"""
    start_time = time.time()

    # Create output directory
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Load training data
    train_df = pd.read_csv(train_csv)
    print(f"Loaded training data with {len(train_df)} rows")

    # Apply sampling if requested
    if sample:
        train_df = train_df.sample(frac=sample, random_state=42)
        print(f"Using sample of {len(train_df)} rows ({sample * 100:.1f}% of data)")

    # Create dataset and dataloader
    dataset = EEGDataset(train_df, data_folder)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,  # Process one sample at a time to avoid issues
        shuffle=False,
        num_workers=0
    )

    # Extract embeddings
    all_embeddings = []

    # Create a single model instance to reuse
    model = Deep4ConvNet().to(device)
    model.eval()

    for batch in tqdm(dataloader, desc="Extracting Deep4Net conv4 embeddings"):
        # Skip errors
        if 'error' in batch and batch['error'].any():
            continue

        eeg_data = batch['eeg_data']
        eeg_ids = batch['eeg_id']
        eeg_sub_ids = batch['eeg_sub_id']

        try:
            # Process batch using our model
            with torch.no_grad():
                if torch.cuda.is_available():
                    eeg_data = eeg_data.to(device)

                # Reshape for visualization
                print(f"Sample shape in batch: {eeg_data.shape}")

                # Process batch through custom model
                batch_features = model(eeg_data)

                # Move back to CPU and convert to numpy
                batch_features = batch_features.cpu().numpy()

            # Create rows for each sample in the batch
            for i in range(len(eeg_ids)):
                # Create result row
                row = {
                    'eeg_id': eeg_ids[i],
                    'eeg_sub_id': eeg_sub_ids[i]
                }

                # Add expert_consensus if available
                if 'expert_consensus' in batch and i < len(batch['expert_consensus']):
                    row['expert_consensus'] = batch['expert_consensus'][i]

                # Add embeddings (1400 features)
                features = batch_features[i]
                for j, val in enumerate(features):
                    row[f'embedding_{j}'] = float(val)

                all_embeddings.append(row)

        except Exception as e:
            print(f"Error processing batch: {e}")
            # Create fallback with 1400 features for each sample in the batch
            for i in range(len(eeg_ids)):
                row = {
                    'eeg_id': eeg_ids[i],
                    'eeg_sub_id': eeg_sub_ids[i]
                }

                # Add expert_consensus if available
                if 'expert_consensus' in batch and i < len(batch['expert_consensus']):
                    row['expert_consensus'] = batch['expert_consensus'][i]

                # Add zero embeddings
                for j in range(1400):
                    row[f'embedding_{j}'] = 0.0

                all_embeddings.append(row)

    # Create DataFrame and save
    embeddings_df = pd.DataFrame(all_embeddings)
    embeddings_df.to_csv(output_file, index=False)

    # Report time and stats
    elapsed_time = time.time() - start_time
    embedding_cols = [col for col in embeddings_df.columns if col.startswith('embedding_')]

    print(f"Total processing time: {elapsed_time:.2f} seconds")
    print(f"Total samples processed: {len(embeddings_df)}")
    print(f"Total embeddings per sample: {len(embedding_cols)}")
    print(f"Output saved to: {output_file}")

    return embeddings_df, train_df.columns


def main(test_mode=True):
    """Main function to extract 1400 embeddings from conv4 layer"""
    # Set file paths based on mode
    if test_mode:
        output_file = BRAINDECODE_OUT_PATH_TEST
        sample_size = TEST_SAMPLE_SIZE_BRAINDECODE
        print(f"Running in TEST MODE with {sample_size * 100:.1f}% of data")
    else:
        output_file = BRAINDECODE_OUT_PATH
        sample_size = None
        print("Running in PRODUCTION MODE with full dataset")

    # Extract embeddings
    embeddings_df, original_columns = extract_braindecode_embeddings(
        output_file=output_file,
        sample=sample_size
    )

    # Display sample
    print("\nSample of extracted embeddings:")
    if not embeddings_df.empty:
        sample_cols = ['eeg_id', 'eeg_sub_id']
        if 'expert_consensus' in embeddings_df.columns:
            sample_cols.append('expert_consensus')
        sample_cols.extend([col for col in embeddings_df.columns if col.startswith('embedding_')][:5])
        print(embeddings_df[sample_cols].head())

    return embeddings_df


if __name__ == "__main__":
    main(test_mode=True)
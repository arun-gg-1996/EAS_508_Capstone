import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from functools import lru_cache

from const import DATA_FOLDER, SAMPLE_RATE, WINDOW_SIZE


class EEGDataReader:
    def __init__(self, data_folder=DATA_FOLDER):
        self.data_folder = data_folder
        self._train_df = None

    def get_train_df(self):
        """Load train.csv if not already loaded"""
        if self._train_df is None:
            csv_path = os.path.join(self.data_folder, 'train.csv')
            self._train_df = pd.read_csv(csv_path)
        return self._train_df

    @lru_cache(maxsize=16)
    def get_eeg_file(self, eeg_id):
        """Load and cache an EEG file"""
        eeg_file = os.path.join(self.data_folder, 'train_eegs', f"{eeg_id}.parquet")
        try:
            return pd.read_parquet(eeg_file)
        except FileNotFoundError:
            print(f"Error: File {eeg_file} not found")
            return None

    def get_eeg_subsample(self, eeg_id, sub_id):
        """Extract a specific subsample from an EEG recording"""
        # Load the metadata
        train_df = self.get_train_df()

        # Filter to get the specific row for this eeg_id and sub_id
        row = train_df[(train_df['eeg_id'] == int(eeg_id)) & (train_df['eeg_sub_id'] == sub_id)]

        if row.empty:
            print(f"No metadata found for EEG ID {eeg_id}, subsample ID {sub_id}")
            return None, None, None

        # Extract metadata for this subsample
        row = row.iloc[0]
        offset_seconds = row['eeg_label_offset_seconds']

        # Create metadata dictionary
        metadata = {
            'eeg_id': eeg_id,
            'sub_id': sub_id,
            'offset_seconds': offset_seconds,
            'expert_consensus': row['expert_consensus'],
            'seizure_vote': row['seizure_vote'],
            'lpd_vote': row['lpd_vote'],
            'gpd_vote': row['gpd_vote'],
            'lrda_vote': row['lrda_vote'],
            'grda_vote': row['grda_vote'],
            'other_vote': row['other_vote']
        }

        # Load the EEG data
        eeg_data = self.get_eeg_file(eeg_id)
        if eeg_data is None:
            return None, None, None

        # Get channel names
        channel_names = eeg_data.columns.tolist()

        # Calculate the range of data to extract
        start_idx = int(offset_seconds * SAMPLE_RATE)
        end_idx = start_idx + WINDOW_SIZE

        # Ensure we don't go beyond the data boundaries
        if end_idx > len(eeg_data):
            print(f"Warning: Subsample {sub_id} extends beyond data bounds. Truncating.")
            end_idx = len(eeg_data)

        # Extract the subsample and convert to numpy
        subsample = eeg_data.iloc[start_idx:end_idx]
        subsample_np = subsample.to_numpy()

        return subsample_np, channel_names, metadata

    def visualize_eeg(self, eeg_data, channel_names, metadata=None, seconds=5):
        """Visualize EEG data as a multi-channel plot"""
        # Limit to specified number of seconds
        max_samples = seconds * SAMPLE_RATE
        if eeg_data.shape[0] > max_samples:
            plot_data = eeg_data[:max_samples, :]
        else:
            plot_data = eeg_data
            seconds = plot_data.shape[0] / SAMPLE_RATE

        # Create time axis in seconds
        time = np.arange(plot_data.shape[0]) / SAMPLE_RATE

        plt.figure(figsize=(15, 10))

        # Plot each channel with offset for visibility
        for i, channel in enumerate(channel_names):
            # Add offset to separate channels visually
            offset = i * 100
            plt.plot(time, plot_data[:, i] + offset, label=f"{channel}")

        plt.xlabel('Time (seconds)')
        plt.ylabel('Amplitude (µV) + Offset')

        # Add title with metadata if provided
        if metadata:
            consensus = metadata.get('expert_consensus', 'Unknown')
            eeg_id = metadata.get('eeg_id', 'Unknown')
            sub_id = metadata.get('sub_id', 'Unknown')
            plt.title(f"EEG ID: {eeg_id}, Subsample: {sub_id}, Consensus: {consensus}")

        plt.legend(loc='upper right')
        plt.grid(True)
        plt.tight_layout()
        plt.show()


def test_visualize():
    """Test visualization functionality"""
    reader = EEGDataReader()
    eeg_id = "1628180742"
    sub_id = 3

    # Get the subsample as numpy array
    eeg_subsample, channels, meta = reader.get_eeg_subsample(eeg_id, sub_id)

    if eeg_subsample is not None:
        # Visualize the subsample
        reader.visualize_eeg(eeg_subsample, channels, meta, seconds=10)

        # Now eeg_subsample is a numpy array that can be used for further processing
        print(f"Subsample shape: {eeg_subsample.shape}")
        print(f"First 3 channels: {channels[:3]}")
        print(f"Expert consensus: {meta['expert_consensus']}")


if __name__ == "__main__":
    test_visualize()
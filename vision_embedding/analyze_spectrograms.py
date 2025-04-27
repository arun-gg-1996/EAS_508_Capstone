import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

from const import (
    TRAIN_CSV_PATH,
    TRAIN_SPECTROGRAMS_DIR,
    OUTPUT_DIR_ANALYSIS_MAIN
)


def analyze_spectrograms(max_files=None):
    """
    Analyze properties of the spectrogram files

    Args:
        max_files: Maximum number of files to analyze (None = analyze all)
    """
    # Set output directory
    output_dir = OUTPUT_DIR_ANALYSIS_MAIN
    print(f"Using output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    print("Loading training data...")
    train_df = pd.read_csv(TRAIN_CSV_PATH)
    print(f"Found {len(train_df)} rows in training data")

    # Get unique spectrogram IDs
    unique_spectrograms = train_df['spectrogram_id'].unique()
    print(f"Found {len(unique_spectrograms)} unique spectrograms")

    # Limit files if max_files is specified
    if max_files and max_files < len(unique_spectrograms):
        spectrograms_to_analyze = unique_spectrograms[:max_files]
        print(f"Analyzing first {len(spectrograms_to_analyze)} spectrograms")
    else:
        spectrograms_to_analyze = unique_spectrograms
        print(f"Analyzing all {len(spectrograms_to_analyze)} spectrograms")

    # Collect statistics
    stats = {
        'spectrogram_id': [],
        'shape': [],
        'min_value': [],
        'max_value': [],
        'mean_value': [],
        'std_value': [],
        'has_nan': [],
        'size_kb': []
    }

    # Maximum number of files to visualize
    max_visualizations = 10
    num_visualized = 0

    # Analyze each spectrogram
    for spec_id in tqdm(spectrograms_to_analyze, desc="Analyzing spectrograms"):
        parquet_path = os.path.join(TRAIN_SPECTROGRAMS_DIR, f"{spec_id}.parquet")

        try:
            # Read parquet file
            table = pq.read_table(parquet_path)
            df = table.to_pandas()
            data = df.values

            # Record statistics
            stats['spectrogram_id'].append(spec_id)
            stats['shape'].append(str(data.shape))
            stats['min_value'].append(float(np.nanmin(data)))
            stats['max_value'].append(float(np.nanmax(data)))
            stats['mean_value'].append(float(np.nanmean(data)))
            stats['std_value'].append(float(np.nanstd(data)))
            stats['has_nan'].append(bool(np.isnan(data).any()))
            stats['size_kb'].append(os.path.getsize(parquet_path) / 1024)

            # Only visualize the first max_visualizations files
            if num_visualized < max_visualizations:
                visualize_spectrogram(data, spec_id, output_dir)
                num_visualized += 1

        except Exception as e:
            print(f"Error processing {parquet_path}: {e}")

    # Create stats DataFrame
    stats_df = pd.DataFrame(stats)
    print(f"Created statistics for {len(stats_df)} spectrograms")

    # Save statistics to CSV
    stats_file = os.path.join(output_dir, 'spectrogram_stats.csv')
    stats_df.to_csv(stats_file, index=False)
    print(f"Saved statistics to {stats_file}")

    # Create summary visualizations
    create_summary_visualizations(stats_df, output_dir)

    return stats_df, output_dir


def visualize_spectrogram(data, spec_id, output_dir):
    """
    Create and save a visualization of a spectrogram

    Args:
        data: Spectrogram data array
        spec_id: Spectrogram ID
        output_dir: Directory to save visualization
    """
    plt.figure(figsize=(12, 8))

    # Plot the spectrogram
    plt.subplot(2, 2, 1)
    plt.imshow(data, aspect='auto', cmap='viridis')
    plt.colorbar(label='Magnitude')
    plt.title(f'Spectrogram {spec_id}')
    plt.xlabel('Time')
    plt.ylabel('Frequency')

    # Plot histogram of values
    plt.subplot(2, 2, 2)
    plt.hist(data.flatten(), bins=50, alpha=0.75)
    plt.title('Histogram of Values')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    # Plot row and column averages
    plt.subplot(2, 2, 3)
    plt.plot(np.mean(data, axis=0))
    plt.title('Column Averages (Time)')
    plt.xlabel('Time Index')
    plt.ylabel('Average Value')

    plt.subplot(2, 2, 4)
    plt.plot(np.mean(data, axis=1))
    plt.title('Row Averages (Frequency)')
    plt.xlabel('Frequency Index')
    plt.ylabel('Average Value')

    plt.tight_layout()

    # Save the visualization
    output_file = os.path.join(output_dir, f'spectrogram_{spec_id}.png')
    plt.savefig(output_file)
    plt.close()


def create_summary_visualizations(stats_df, output_dir):
    """
    Create summary visualizations of spectrogram statistics

    Args:
        stats_df: DataFrame with spectrogram statistics
        output_dir: Directory to save visualizations
    """
    # Create figure for summary stats
    plt.figure(figsize=(16, 12))

    # Plot distribution of min values
    plt.subplot(2, 2, 1)
    plt.hist(stats_df['min_value'], bins=20, alpha=0.75)
    plt.title('Distribution of Minimum Values')
    plt.xlabel('Minimum Value')
    plt.ylabel('Frequency')

    # Plot distribution of max values
    plt.subplot(2, 2, 2)
    plt.hist(stats_df['max_value'], bins=20, alpha=0.75)
    plt.title('Distribution of Maximum Values')
    plt.xlabel('Maximum Value')
    plt.ylabel('Frequency')

    # Plot distribution of means
    plt.subplot(2, 2, 3)
    plt.hist(stats_df['mean_value'], bins=20, alpha=0.75)
    plt.title('Distribution of Mean Values')
    plt.xlabel('Mean Value')
    plt.ylabel('Frequency')

    # Plot distribution of standard deviations
    plt.subplot(2, 2, 4)
    plt.hist(stats_df['std_value'], bins=20, alpha=0.75)
    plt.title('Distribution of Standard Deviations')
    plt.xlabel('Standard Deviation')
    plt.ylabel('Frequency')

    plt.tight_layout()

    # Save the visualization
    output_file = os.path.join(output_dir, 'spectrogram_summary_stats.png')
    plt.savefig(output_file)
    plt.close()


def analyze_class_distribution(output_dir):
    """
    Analyze the distribution of classes in the dataset

    Args:
        output_dir: Directory to save visualizations
    """
    print("Analyzing class distribution...")
    train_df = pd.read_csv(TRAIN_CSV_PATH)

    if 'expert_consensus' in train_df.columns:
        # Count classes
        class_counts = train_df['expert_consensus'].value_counts()

        # Print class distribution
        print("\nClass distribution:")
        for class_name, count in class_counts.items():
            percentage = count / len(train_df) * 100
            print(f"  {class_name}: {count} samples ({percentage:.2f}%)")

        # Create bar chart
        plt.figure(figsize=(12, 6))
        class_counts.plot(kind='bar')
        plt.title('Distribution of Classes')
        plt.xlabel('Class')
        plt.ylabel('Number of Samples')
        plt.tight_layout()

        # Save the visualization
        output_file = os.path.join(output_dir, 'class_distribution.png')
        plt.savefig(output_file)
        plt.close()

        return class_counts
    else:
        print("No 'expert_consensus' column found in the dataset")
        return None


def main():
    print("Starting Spectrogram Analysis")
    print("=" * 50)

    # Analyze spectrogram properties - analyze all files
    stats_df, output_dir = analyze_spectrograms()

    # Analyze class distribution
    class_counts = analyze_class_distribution(output_dir)

    print("\nAnalysis complete. Results saved to:", output_dir)


if __name__ == "__main__":
    main()
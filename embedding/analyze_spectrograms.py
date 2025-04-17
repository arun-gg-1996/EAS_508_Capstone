import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

from const import TRAIN_CSV_PATH, TRAIN_SPECTROGRAMS_DIR, OUTPUT_DIR_ANALYSIS_TEST


def analyze_spectrograms(sample_size=0.05, max_files=50):
    """
    Analyze properties of the spectrogram files

    Args:
        sample_size: Fraction of files to analyze
        max_files: Maximum number of files to analyze
    """
    print("Loading training data...")
    train_df = pd.read_csv(TRAIN_CSV_PATH)
    print(f"Found {len(train_df)} rows in training data")

    # Get unique spectrogram IDs
    unique_spectrograms = train_df['spectrogram_id'].unique()
    print(f"Found {len(unique_spectrograms)} unique spectrograms")

    # Apply sampling
    if sample_size < 1.0:
        np.random.seed(42)
        sample_count = min(int(len(unique_spectrograms) * sample_size), max_files)
        sampled_spectrograms = np.random.choice(unique_spectrograms, size=sample_count, replace=False)
        print(f"Analyzing {len(sampled_spectrograms)} sampled spectrograms")
    else:
        sampled_spectrograms = unique_spectrograms[:max_files]
        print(f"Analyzing first {len(sampled_spectrograms)} spectrograms")

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

    # Analyze each sampled spectrogram
    for spec_id in tqdm(sampled_spectrograms, desc="Analyzing spectrograms"):
        parquet_path = os.path.join(TRAIN_SPECTROGRAMS_DIR, f"{spec_id}.parquet")

        try:
            # Read parquet file
            table = pq.read_table(parquet_path)
            df = table.to_pandas()

            # Get spectrogram data
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

            # If this is one of the first 10 files, save a visualization
            if len(stats['spectrogram_id']) <= 10:
                visualize_spectrogram(data, spec_id)

        except Exception as e:
            print(f"Error processing {parquet_path}: {e}")

    # Create stats DataFrame
    stats_df = pd.DataFrame(stats)
    print(f"Created statistics for {len(stats_df)} spectrograms")

    # Save statistics to CSV
    stats_file = os.path.join(OUTPUT_DIR_ANALYSIS_TEST, 'spectrogram_stats.csv')
    stats_df.to_csv(stats_file, index=False)
    print(f"Saved statistics to {stats_file}")

    # Create summary visualizations
    create_summary_visualizations(stats_df)

    return stats_df


def visualize_spectrogram(data, spec_id):
    """
    Create and save a visualization of a spectrogram

    Args:
        data: Spectrogram data array
        spec_id: Spectrogram ID
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
    output_file = os.path.join(OUTPUT_DIR_ANALYSIS_TEST, f'spectrogram_{spec_id}.png')
    plt.savefig(output_file)
    plt.close()


def create_summary_visualizations(stats_df):
    """
    Create summary visualizations of spectrogram statistics

    Args:
        stats_df: DataFrame with spectrogram statistics
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

    # Save the OUTPUT_DIR_ANALYSIS
    output_file = os.path.join(OUTPUT_DIR_ANALYSIS_TEST, 'spectrogram_summary_stats.png')
    plt.savefig(output_file)
    plt.close()

    # Create a figure for file sizes
    plt.figure(figsize=(10, 6))
    plt.hist(stats_df['size_kb'], bins=20, alpha=0.75)
    plt.title('Distribution of File Sizes')
    plt.xlabel('File Size (KB)')
    plt.ylabel('Frequency')
    plt.tight_layout()

    # Save the visualization
    output_file = os.path.join(OUTPUT_DIR_ANALYSIS_TEST, 'spectrogram_file_sizes.png')
    plt.savefig(output_file)
    plt.close()

    # Create a pie chart for shapes if there are different shapes
    shapes = stats_df['shape'].value_counts()
    if len(shapes) > 1:
        plt.figure(figsize=(10, 8))
        plt.pie(shapes, labels=shapes.index, autopct='%1.1f%%')
        plt.title('Distribution of Spectrogram Shapes')
        plt.tight_layout()

        # Save the visualization
        output_file = os.path.join(OUTPUT_DIR_ANALYSIS_TEST, 'spectrogram_shapes.png')
        plt.savefig(output_file)
        plt.close()


def analyze_spectrogram_sub_ids():
    """
    Analyze the relationship between spectrogram_id and spectrogram_sub_id
    """
    print("Analyzing spectrogram_id and spectrogram_sub_id relationships...")
    train_df = pd.read_csv(TRAIN_CSV_PATH)

    # Count sub_ids per spectrogram_id
    sub_ids_per_spec = train_df.groupby('spectrogram_id')['spectrogram_sub_id'].nunique()

    # Create statistics
    min_sub_ids = sub_ids_per_spec.min()
    max_sub_ids = sub_ids_per_spec.max()
    mean_sub_ids = sub_ids_per_spec.mean()
    median_sub_ids = sub_ids_per_spec.median()

    print(f"Sub_ids per spectrogram_id:")
    print(f"  Min: {min_sub_ids}")
    print(f"  Max: {max_sub_ids}")
    print(f"  Mean: {mean_sub_ids:.2f}")
    print(f"  Median: {median_sub_ids}")

    # Create histogram of sub_ids per spectrogram
    plt.figure(figsize=(10, 6))
    plt.hist(sub_ids_per_spec, bins=max(20, max_sub_ids), alpha=0.75)
    plt.title('Number of sub_ids per spectrogram_id')
    plt.xlabel('Number of sub_ids')
    plt.ylabel('Number of spectrograms')
    plt.tight_layout()

    # Save the visualization
    output_file = os.path.join(OUTPUT_DIR_ANALYSIS_TEST, 'sub_ids_per_spectrogram.png')
    plt.savefig(output_file)
    plt.close()

    # Analyze the relationship between eeg_id and spectrogram_id
    print("\nAnalyzing eeg_id and spectrogram_id relationships...")
    specs_per_eeg = train_df.groupby('eeg_id')['spectrogram_id'].nunique()

    min_specs = specs_per_eeg.min()
    max_specs = specs_per_eeg.max()
    mean_specs = specs_per_eeg.mean()
    median_specs = specs_per_eeg.median()

    print(f"Spectrograms per eeg_id:")
    print(f"  Min: {min_specs}")
    print(f"  Max: {max_specs}")
    print(f"  Mean: {mean_specs:.2f}")
    print(f"  Median: {median_specs}")

    # Create histogram of spectrograms per eeg
    plt.figure(figsize=(10, 6))
    plt.hist(specs_per_eeg, bins=min(20, max_specs), alpha=0.75)
    plt.title('Number of spectrograms per eeg_id')
    plt.xlabel('Number of spectrograms')
    plt.ylabel('Number of EEGs')
    plt.tight_layout()

    # Save the visualization
    output_file = os.path.join(OUTPUT_DIR_ANALYSIS_TEST, 'spectrograms_per_eeg.png')
    plt.savefig(output_file)
    plt.close()

    return {
        'sub_ids_per_spec_stats': {
            'min': min_sub_ids,
            'max': max_sub_ids,
            'mean': mean_sub_ids,
            'median': median_sub_ids
        },
        'specs_per_eeg_stats': {
            'min': min_specs,
            'max': max_specs,
            'mean': mean_specs,
            'median': median_specs
        }
    }


def analyze_class_distribution():
    """
    Analyze the distribution of classes in the dataset
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
        output_file = os.path.join(OUTPUT_DIR_ANALYSIS_TEST, 'class_distribution.png')
        plt.savefig(output_file)
        plt.close()

        # Create pie chart
        plt.figure(figsize=(10, 8))
        plt.pie(class_counts, labels=class_counts.index, autopct='%1.1f%%')
        plt.title('Distribution of Classes')
        plt.tight_layout()

        # Save the visualization
        output_file = os.path.join(OUTPUT_DIR_ANALYSIS_TEST, 'class_distribution_pie.png')
        plt.savefig(output_file)
        plt.close()

        return class_counts
    else:
        print("No 'expert_consensus' column found in the dataset")
        return None


if __name__ == "__main__":
    print("Starting Spectrogram Analysis")
    print("=" * 50)

    # Analyze spectrogram properties
    stats_df = analyze_spectrograms(sample_size=0.1, max_files=100)

    # Analyze spectrogram_sub_id relationships
    id_stats = analyze_spectrogram_sub_ids()

    # Analyze class distribution
    class_counts = analyze_class_distribution()

    print("\nAnalysis complete. Results saved to:", OUTPUT_DIR_ANALYSIS_TEST)

import time

import numpy as np
from tqdm import tqdm

from const import DATA_FOLDER
from data_reader import EEGDataReader
from time_domain_features import EEGFeatureExtractor


class PerformanceTester:
    def __init__(self, data_folder=DATA_FOLDER):
        self.data_reader = EEGDataReader(data_folder)
        self.feature_extractor = EEGFeatureExtractor()

        print(f"Number of data samples available: {self.data_reader.get_train_df().shape[0]}")

    def run_benchmark(self, n_samples=100, verbose=True):
        """
        Benchmark feature extraction on multiple EEG samples
        """
        # Get the train data
        train_df = self.data_reader.get_train_df()

        # Get a random sample
        if len(train_df) < n_samples:
            samples = train_df
        else:
            samples = train_df.sample(n=n_samples, random_state=42)

        # Initialize timing and results
        start_time = time.time()
        extraction_times = []
        processed_count = 0

        # Process each sample
        iterator = tqdm(samples.iterrows(), total=len(samples)) if verbose else samples.iterrows()

        for _, row in iterator:
            eeg_id = str(row['eeg_id'])
            sub_id = row['eeg_sub_id']

            # Time the extraction process
            sample_start = time.time()

            # Get the subsample and extract features
            eeg_subsample, _, _ = self.data_reader.get_eeg_subsample(eeg_id, sub_id)

            if eeg_subsample is not None:
                self.feature_extractor.extract_features(eeg_subsample)
                processed_count += 1

                # Record time for this sample
                sample_time = time.time() - sample_start
                extraction_times.append(sample_time)

        # Calculate statistics
        total_time = time.time() - start_time

        # Prepare results
        results = {
            'total_time': total_time,
            'processed_count': processed_count,
            'avg_time': np.mean(extraction_times) if extraction_times else 0,
            'min_time': min(extraction_times) if extraction_times else 0,
            'max_time': max(extraction_times) if extraction_times else 0
        }

        if verbose:
            self._print_results(results)

        return results

    def _print_results(self, results):
        """Print benchmark results in a formatted way"""
        print(f"\nProcessed {results['processed_count']} samples")
        print(f"Total time: {results['total_time']:.2f} seconds")
        print(f"Average time per sample: {results['avg_time']:.4f} seconds")
        print(f"Min time: {results['min_time']:.4f} seconds")
        print(f"Max time: {results['max_time']:.4f} seconds")


def test():
    n_samples = 100  # Reduced from 10000 for simplicity
    print(f"Testing feature extraction on {n_samples} EEG samples...")
    tester = PerformanceTester()
    tester.run_benchmark(n_samples=n_samples)

    # Test a single example for reference
    print("\nTesting single example for reference:")
    extractor = EEGFeatureExtractor()
    reader = EEGDataReader()
    features, _ = extractor.process_example("1628180742", 3, reader)

    if features:
        # Display a few features
        for i, (key, value) in enumerate(list(features.items())[:5]):
            print(f"{key}: {value:.6f}")
        print("...")


if __name__ == "__main__":
    test()
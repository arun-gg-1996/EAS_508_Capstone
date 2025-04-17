import time

import numpy as np
from tqdm import tqdm

from const import DATA_FOLDER, SAMPLE_RATE
from data_reader import EEGDataReader
from nonlinear_domain_features import EEGNonlinearExtractor


class NonlinearPerformanceTester:
    def __init__(self, data_folder=DATA_FOLDER):
        self.data_reader = EEGDataReader(data_folder)
        self.feature_extractor = EEGNonlinearExtractor()

        print(f"Number of data samples available: {self.data_reader.get_train_df().shape[0]}")

    def run_benchmark(self, n_samples=50, verbose=True, fs=SAMPLE_RATE):
        """
        Benchmark non-linear feature extraction on multiple EEG samples

        Parameters:
            n_samples (int): Number of samples to process
            verbose (bool): Whether to print progress and results
            fs (int): Sampling frequency in Hz

        Returns:
            dict: Performance metrics
        """
        # Get the train data
        train_df = self.data_reader.get_train_df()

        # Get a random sample
        if len(train_df) < n_samples:
            if verbose:
                print(f"Warning: Requested {n_samples} samples but only {len(train_df)} are available")
            samples = train_df
        else:
            samples = train_df.sample(n=n_samples, random_state=42)

        # Initialize timing and results
        start_time = time.time()
        extraction_times = []
        processed_count = 0
        feature_counts = []

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
                features = self.feature_extractor.extract_features(eeg_subsample, fs=fs)
                processed_count += 1
                feature_counts.append(len(features))

                # Record time for this sample
                sample_time = time.time() - sample_start
                extraction_times.append(sample_time)

        # Calculate statistics
        total_time = time.time() - start_time
        avg_feature_count = np.mean(feature_counts) if feature_counts else 0

        # Prepare results
        results = {
            'total_time': total_time,
            'processed_count': processed_count,
            'avg_time': np.mean(extraction_times) if extraction_times else 0,
            'min_time': min(extraction_times) if extraction_times else 0,
            'max_time': max(extraction_times) if extraction_times else 0,
            'avg_feature_count': avg_feature_count
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
        print(f"Average number of features per sample: {results['avg_feature_count']:.1f}")


def test():
    n_samples = 30  # Using fewer samples as non-linear features are computationally intensive
    print(f"Testing non-linear domain feature extraction on {n_samples} EEG samples...")
    tester = NonlinearPerformanceTester()
    tester.run_benchmark(n_samples=n_samples)

    # Test a single example for reference
    print("\nTesting single example for reference:")
    extractor = EEGNonlinearExtractor()
    reader = EEGDataReader()

    # Get timing for a single example
    start_time = time.time()
    features, meta = extractor.process_example("1628180742", 3, reader)
    total_time = time.time() - start_time

    if features:
        # Display a few features
        print("Example features:")
        for i, (key, value) in enumerate(list(features.items())[:5]):
            print(f"{key}: {value:.6f}")
        print("...")

        print(f"Number of features: {len(features)}")
        print(f"Total processing time: {total_time:.4f} seconds")


if __name__ == "__main__":
    test()
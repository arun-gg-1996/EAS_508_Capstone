import time

import numpy as np
from tqdm import tqdm

from const import DATA_FOLDER, SAMPLE_RATE
from data_reader import EEGDataReader
from time_frequency_features import EEGWaveletExtractor


class WaveletPerformanceTester:
    def __init__(self, data_folder=DATA_FOLDER, wavelet='db4', level=5):
        self.data_reader = EEGDataReader(data_folder)
        self.feature_extractor = EEGWaveletExtractor(wavelet=wavelet, level=level)
        self.wavelet_type = wavelet
        self.level = level

        print(f"Number of data samples available: {self.data_reader.get_train_df().shape[0]}")
        print(f"Using wavelet: {self.wavelet_type}, decomposition level: {self.level}")

    def run_benchmark(self, n_samples=50, verbose=True, fs=SAMPLE_RATE):
        """
        Benchmark wavelet feature extraction on multiple EEG samples

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


def test_different_wavelets():
    """Test performance with different wavelet types"""
    n_samples = 10
    wavelets = ['db4', 'sym4', 'coif3', 'haar']

    print("Comparing performance of different wavelet types:")
    for wavelet in wavelets:
        print(f"\nTesting with wavelet type: {wavelet}")
        tester = WaveletPerformanceTester(wavelet=wavelet)
        tester.run_benchmark(n_samples=n_samples)


def test():
    n_samples = 50  # Using fewer samples as wavelet transform is computationally intensive
    print(f"Testing wavelet feature extraction on {n_samples} EEG samples...")
    tester = WaveletPerformanceTester()
    tester.run_benchmark(n_samples=n_samples)

    # Test a single example for reference
    print("\nTesting single example for reference:")
    extractor = EEGWaveletExtractor()
    reader = EEGDataReader()
    features, _ = extractor.process_example("1628180742", 3, reader)

    if features:
        # Display a few features
        for i, (key, value) in enumerate(list(features.items())[:5]):
            print(f"{key}: {value:.6f}")
        print("...")

    print(f"Number of features: {len(features)}")

    # Uncomment to test different wavelet types
    # test_different_wavelets()


if __name__ == "__main__":
    test()
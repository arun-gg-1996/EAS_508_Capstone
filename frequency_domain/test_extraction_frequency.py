import time

import numpy as np
from tqdm import tqdm

from scipy import signal
from const import DATA_FOLDER, SAMPLE_RATE
from data_reader import EEGDataReader
from frequency_domain_features import EEGFrequencyExtractor


class FrequencyPerformanceTester:
    def __init__(self, data_folder=DATA_FOLDER):
        self.data_reader = EEGDataReader(data_folder)
        self.feature_extractor = EEGFrequencyExtractor()

        print(f"Number of data samples available: {self.data_reader.get_train_df().shape[0]}")

    def run_benchmark(self, n_samples=20, verbose=True, fs=SAMPLE_RATE):
        """
        Benchmark frequency feature extraction on multiple EEG samples

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
                self.feature_extractor.extract_features(eeg_subsample, fs=fs)
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


def test_feature_timing():
    """Test timing of individual frequency bands"""
    print("\nTesting timing of frequency band calculations...")
    extractor = EEGFrequencyExtractor()
    reader = EEGDataReader()

    eeg_id = "1628180742"
    sub_id = 3

    # Get the sample data
    eeg_subsample, _, _ = reader.get_eeg_subsample(eeg_id, sub_id)
    if eeg_subsample is not None:
        # Test PSD calculation timing
        start_time = time.time()
        channel_data = eeg_subsample[:, 0]  # Use first channel
        freqs, psd = signal.welch(channel_data, fs=SAMPLE_RATE, nperseg=min(256, len(channel_data)))
        psd_time = time.time() - start_time

        # Test band power calculations
        band_times = {}
        for band_name, (low_freq, high_freq) in extractor.FREQ_BANDS.items():
            start_time = time.time()
            band_idx = np.logical_and(freqs >= low_freq, freqs <= high_freq)
            band_power = np.sum(psd[band_idx])
            band_times[band_name] = time.time() - start_time

        # Report results
        print(f"PSD calculation time: {psd_time:.6f} seconds")
        print("\nBand power calculation times:")
        for band, elapsed in band_times.items():
            print(f"{band:10s}: {elapsed:.6f} seconds")


def test():
    n_samples = 10
    print(f"Testing frequency domain feature extraction on {n_samples} EEG samples...")
    tester = FrequencyPerformanceTester()
    tester.run_benchmark(n_samples=n_samples)

    # Test a single example for reference
    print("\nTesting single example for reference:")
    extractor = EEGFrequencyExtractor()
    reader = EEGDataReader()

    start_time = time.time()
    features, _ = extractor.process_example("1628180742", 3, reader)
    total_time = time.time() - start_time

    if features:
        # Group features by type
        band_features = [k for k in features.keys() if any(band in k for band in
                                                           ['delta', 'theta', 'alpha', 'beta', 'gamma'])]
        other_features = [k for k in features.keys() if k not in band_features]

        # Display counts
        print(f"Band power features: {len(band_features)}")
        print(f"Other features: {len(other_features)}")

        # Sample of each category
        if band_features:
            print("\nExample band power features:")
            for k in band_features[:3]:
                print(f"{k}: {features[k]:.6f}")

        if other_features:
            print("\nExample other features:")
            for k in other_features[:3]:
                print(f"{k}: {features[k]:.6f}")

        print(f"\nTotal features: {len(features)}")
        print(f"Processing time: {total_time:.4f} seconds")

    # Test individual feature timing
    from scipy import signal
    test_feature_timing()


if __name__ == "__main__":
    test()
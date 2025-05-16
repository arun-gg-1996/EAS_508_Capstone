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

    def run_benchmark(self, n_samples=20, verbose=True, fs=SAMPLE_RATE):
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


def test_feature_timing():
    """Test timing of individual non-linear features"""
    print("\nTesting timing of individual non-linear features...")
    extractor = EEGNonlinearExtractor()
    reader = EEGDataReader()

    # Process a few samples and record timing for each feature
    feature_times = {}
    n_test_samples = 3
    test_ids = [("1628180742", 3), ("2277392603", 0), ("722738444", 4)]

    for eeg_id, sub_id in test_ids[:n_test_samples]:
        print(f"\nProcessing sample: EEG ID {eeg_id}, sub_id {sub_id}")

        # Get the sample data
        eeg_subsample, _, _ = reader.get_eeg_subsample(eeg_id, sub_id)
        if eeg_subsample is None:
            continue

        # Test each feature category

        # Fractal Dimensions
        for func_name, func in [
            ("petrosian_fd", lambda x: ant.petrosian_fd(x)),
            ("katz_fd", lambda x: ant.katz_fd(x)),
            ("higuchi_fd", lambda x: ant.higuchi_fd(x, kmax=10))
        ]:
            channel_data = eeg_subsample[:, 0]  # Use first channel for testing
            start_time = time.time()
            try:
                _ = func(channel_data)
                elapsed = time.time() - start_time

                if func_name in feature_times:
                    feature_times[func_name].append(elapsed)
                else:
                    feature_times[func_name] = [elapsed]

            except Exception as e:
                print(f"Error in {func_name}: {e}")

        # DFA
        start_time = time.time()
        try:
            _ = ant.detrended_fluctuation(channel_data)
            elapsed = time.time() - start_time

            if "dfa" in feature_times:
                feature_times["dfa"].append(elapsed)
            else:
                feature_times["dfa"] = [elapsed]

        except Exception as e:
            print(f"Error in DFA: {e}")

        # Entropy measures
        for func_name, func in [
            ("sample_entropy", lambda x: ant.sample_entropy(x, order=2, metric='chebyshev')),
            ("perm_entropy", lambda x: ant.perm_entropy(x, order=3, normalize=True)),
            ("spectral_entropy", lambda x: ant.spectral_entropy(x, sf=SAMPLE_RATE, method='welch', normalize=True))
        ]:
            start_time = time.time()
            try:
                _ = func(channel_data)
                elapsed = time.time() - start_time

                if func_name in feature_times:
                    feature_times[func_name].append(elapsed)
                else:
                    feature_times[func_name] = [elapsed]

            except Exception as e:
                print(f"Error in {func_name}: {e}")

        # Lempel-Ziv complexity
        start_time = time.time()
        try:
            binary_seq = (channel_data > np.median(channel_data)).astype(int)
            _ = ant.lziv_complexity(''.join(binary_seq.astype(str)), normalize=True)
            elapsed = time.time() - start_time

            if "lziv_complexity" in feature_times:
                feature_times["lziv_complexity"].append(elapsed)
            else:
                feature_times["lziv_complexity"] = [elapsed]

        except Exception as e:
            print(f"Error in LZ complexity: {e}")

    # Report timing statistics
    print("\nFeature timing statistics (seconds per feature):")
    print("-" * 60)
    print(f"{'Feature':<20} {'Min':<10} {'Max':<10} {'Avg':<10} {'Samples':<10}")
    print("-" * 60)

    for feature, times in sorted(feature_times.items(), key=lambda x: np.mean(x[1]), reverse=True):
        min_time = min(times)
        max_time = max(times)
        avg_time = np.mean(times)

        print(f"{feature:<20} {min_time:<10.6f} {max_time:<10.6f} {avg_time:<10.6f} {len(times):<10}")


def test():
    n_samples = 5  # Using fewer samples as non-linear features are computationally intensive
    print(f"Testing non-linear domain feature extraction on {n_samples} EEG samples...")
    tester = NonlinearPerformanceTester()
    tester.run_benchmark(n_samples=n_samples)

    # Test a single example for reference
    print("\nTesting single example for reference:")
    extractor = EEGNonlinearExtractor()
    reader = EEGDataReader()

    # Get timing for a single example
    start_time = time.time()
    features, _ = extractor.process_example("1628180742", 3, reader)
    total_time = time.time() - start_time

    if features:
        # Group features by type
        fd_features = [k for k in features.keys() if any(x in k for x in ['_fd', '_dfa'])]
        entropy_features = [k for k in features.keys() if 'entropy' in k]
        other_features = [k for k in features.keys() if k not in fd_features and k not in entropy_features]

        # Display count by category
        print(f"Fractal dimension features: {len(fd_features)}")
        print(f"Entropy features: {len(entropy_features)}")
        print(f"Other features: {len(other_features)}")

        # Display example features from each category
        if fd_features:
            print("\nExample fractal dimension features:")
            for k in fd_features[:3]:
                print(f"{k}: {features[k]:.6f}")

        if entropy_features:
            print("\nExample entropy features:")
            for k in entropy_features[:3]:
                print(f"{k}: {features[k]:.6f}")

        if other_features:
            print("\nExample other features:")
            for k in other_features[:3]:
                print(f"{k}: {features[k]:.6f}")

        print(f"\nNumber of features: {len(features)}")
        print(f"Total processing time: {total_time:.4f} seconds")

    # Test individual feature timing
    test_feature_timing()


if __name__ == "__main__":
    # Import antropy only when needed for feature timing
    import antropy as ant

    test()
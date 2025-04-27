import time

import numpy as np
from tqdm import tqdm

from braindecode_embeddings import create_model, extract_embeddings_feature
from const import TRAIN_EEG_DIR, TRAIN_CSV_PATH


class PerformanceTester:
    def __init__(self, data_folder=TRAIN_EEG_DIR):
        """
        Initialize performance tester
        
        Parameters:
            data_folder (str): Path to the data folder
        """
        self.data_folder = data_folder
        import pandas as pd
        self.train_df = pd.read_csv(TRAIN_CSV_PATH)
        print(f"Number of data samples available: {self.train_df.shape[0]}")

    def run_benchmark(self, model_name='shallow', n_samples=100, verbose=True):
        """
        Benchmark feature extraction on multiple EEG samples
        
        Parameters:
            model_name (str): Name of model to use ('shallow', 'deep4', 'eegnet')
            n_samples (int): Number of samples to process
            verbose (bool): Whether to print progress and results
            
        Returns:
            dict: Performance metrics
        """
        # Get a random sample
        if len(self.train_df) < n_samples:
            if verbose:
                print(f"Warning: Requested {n_samples} samples but only {len(self.train_df)} are available")
            samples = self.train_df
        else:
            samples = self.train_df.sample(n=n_samples, random_state=42)

        # Initialize timing and results
        start_time = time.time()
        extraction_times = []
        processed_count = 0

        # Process each sample
        iterator = tqdm(samples.iterrows(), total=len(samples)) if verbose else samples.iterrows()

        import torch
        import os
        import pyarrow.parquet as pq

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create model with default parameters first to be adjusted later
        model = None

        for _, row in iterator:
            eeg_id = str(row['eeg_id'])

            # Time the extraction process
            sample_start = time.time()

            try:
                # Load EEG data
                parquet_path = os.path.join(self.data_folder, f"{eeg_id}.parquet")
                table = pq.read_table(parquet_path)
                df = table.to_pandas()
                eeg_data = df.values

                # Normalize the data
                if eeg_data.max() > eeg_data.min():
                    eeg_data_norm = (eeg_data - eeg_data.min()) / (eeg_data.max() - eeg_data.min())
                else:
                    eeg_data_norm = np.zeros_like(eeg_data)

                # Convert to tensor
                eeg_tensor = torch.tensor(eeg_data_norm.T, dtype=torch.float32).unsqueeze(0)

                # Create or adjust model if needed
                if model is None:
                    n_chans = eeg_tensor.shape[1]
                    n_times = eeg_tensor.shape[2]
                    model = create_model(model_name, n_chans, n_times)
                    if torch.cuda.is_available():
                        model = model.to(device)
                        model = model.half()
                    model.eval()

                # Extract features
                with torch.no_grad():
                    eeg_tensor = eeg_tensor.to(device)
                    embeddings = extract_embeddings_feature(model, eeg_tensor)

                processed_count += 1

                # Record time for this sample
                sample_time = time.time() - sample_start
                extraction_times.append(sample_time)

            except Exception as e:
                if verbose:
                    print(f"Error processing EEG {eeg_id}: {e}")

        # Calculate statistics
        total_time = time.time() - start_time

        # Prepare results
        results = {
            'model_name': model_name,
            'total_time': total_time,
            'processed_count': processed_count,
            'avg_time': np.mean(extraction_times) if extraction_times else 0,
            'min_time': min(extraction_times) if extraction_times else 0,
            'max_time': max(extraction_times) if extraction_times else 0
        }

        if verbose:
            self._print_results(results)

        return results

    def compare_models(self, n_samples=20, verbose=True):
        """
        Compare performance across different models
        
        Parameters:
            n_samples (int): Number of samples to test per model
            verbose (bool): Whether to print progress
            
        Returns:
            dict: Benchmark results for each model
        """
        models = ['shallow', 'deep4', 'eegnet']
        results = {}

        for model_name in models:
            if verbose:
                print(f"\nBenchmarking model: {model_name}")

            model_results = self.run_benchmark(
                model_name=model_name,
                n_samples=n_samples,
                verbose=verbose
            )

            results[model_name] = model_results

        if verbose:
            self._print_comparison(results)

        return results

    def _print_results(self, results):
        """Print benchmark results in a formatted way"""
        print(f"\nResults for model: {results['model_name']}")
        print(f"Processed {results['processed_count']} samples")
        print(f"Total time: {results['total_time']:.2f} seconds")
        print(f"Average time per sample: {results['avg_time']:.4f} seconds")
        print(f"Min time: {results['min_time']:.4f} seconds")
        print(f"Max time: {results['max_time']:.4f} seconds")

    def _print_comparison(self, results_dict):
        """Print comparison of benchmark results across models"""
        print("\nModel Comparison:")
        print("-" * 60)
        print(f"{'Model':<10} {'Avg Time (s)':<15} {'Total Time (s)':<15} {'Samples':<10}")
        print("-" * 60)

        for model_name, results in results_dict.items():
            print(
                f"{model_name:<10} {results['avg_time']:<15.4f} {results['total_time']:<15.2f} {results['processed_count']:<10}")


def test_single_example():
    """Test embedding extraction on a single example"""
    print("Testing embedding extraction on a single EEG file...")

    # Get a specific EEG ID
    import pandas as pd
    train_df = pd.read_csv(TRAIN_CSV_PATH)
    eeg_id = str(train_df['eeg_id'].iloc[0])

    # Load and process the EEG data
    import os
    import pyarrow.parquet as pq
    import torch

    parquet_path = os.path.join(TRAIN_EEG_DIR, f"{eeg_id}.parquet")
    print(f"Using EEG file: {parquet_path}")

    # Read the file
    table = pq.read_table(parquet_path)
    df = table.to_pandas()
    eeg_data = df.values

    # Normalize and prepare data
    eeg_data_norm = (eeg_data - eeg_data.min()) / (eeg_data.max() - eeg_data.min())
    eeg_tensor = torch.tensor(eeg_data_norm.T, dtype=torch.float32).unsqueeze(0)

    print(f"EEG data shape: {eeg_data.shape}")
    print(f"EEG tensor shape: {eeg_tensor.shape}")

    # Create models and extract embeddings
    models = ['shallow', 'deep4', 'eegnet']

    for model_name in models:
        print(f"\nTesting model: {model_name}")

        # Create model
        model = create_model(model_name, eeg_tensor.shape[1], eeg_tensor.shape[2])

        # Extract embeddings
        embeddings = extract_embeddings_feature(model, eeg_tensor)

        # Print stats
        print(f"Embedding shape: {embeddings.shape}")
        print(f"Embedding size: {embeddings.size}")
        print(f"First few values: {embeddings.flatten()[:5]}")


def run_batch_test():
    """Run performance tests for all models"""
    print("Running performance tests...")

    tester = PerformanceTester()

    # Test with small number of samples first
    print("\nQuick test with 10 samples:")
    tester.compare_models(n_samples=10)

    # Run more comprehensive test
    print("\nComprehensive test with 50 samples:")
    results = tester.compare_models(n_samples=50)

    # Determine best model based on speed
    best_model = min(results.items(), key=lambda x: x[1]['avg_time'])[0]
    print(f"\nBest performing model based on speed: {best_model}")

    # Print recommendation
    print("\nRecommendation:")
    if best_model == 'shallow':
        print("ShallowFBCSPNet is fastest and suitable for quick experiments.")
    elif best_model == 'deep4':
        print("Deep4Net provides good balance between performance and accuracy.")
    else:
        print("EEGNetv4 is optimized for motor imagery and ERPs.")


def test():
    """Run all tests"""
    # Test a single example
    test_single_example()

    # Run batch performance tests
    run_batch_test()


if __name__ == "__main__":
    test()

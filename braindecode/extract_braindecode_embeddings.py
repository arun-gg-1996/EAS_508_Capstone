import time

from braindecode_embeddings import extract_braindecode_embeddings
from const import TRAIN_EEG_DIR, TRAIN_CSV_PATH, BRAINDECODE_OUT_PATH, BRAINDECODE_OUT_PATH_TEST, \
    TEST_SAMPLE_SIZE_BRAINDECODE


def test_single_eeg():
    """Test embedding extraction on a single EEG"""
    from braindecode_embeddings import extract_embeddings_feature, create_model
    import torch
    import os
    import pandas as pd
    import pyarrow.parquet as pq

    # Get a specific EEG ID
    train_df = pd.read_csv(TRAIN_CSV_PATH)
    eeg_id = str(train_df['eeg_id'].iloc[0])

    # Load the EEG file
    parquet_path = os.path.join(TRAIN_EEG_DIR, f"{eeg_id}.parquet")
    print(f"Loading EEG file: {parquet_path}")

    table = pq.read_table(parquet_path)
    df = table.to_pandas()
    eeg_data = df.values

    # Normalize and prepare data
    eeg_data_norm = (eeg_data - eeg_data.min()) / (eeg_data.max() - eeg_data.min())
    eeg_tensor = torch.tensor(eeg_data_norm.T, dtype=torch.float32).unsqueeze(0)

    # Initialize models
    models = {
        'shallow': create_model('shallow', eeg_tensor.shape[1], eeg_tensor.shape[2]),
        'deep4': create_model('deep4', eeg_tensor.shape[1], eeg_tensor.shape[2]),
        'eegnet': create_model('eegnet', eeg_tensor.shape[1], eeg_tensor.shape[2])
    }

    # Extract and compare embeddings for each model
    for model_name, model in models.items():
        embeddings = extract_embeddings_feature(model, eeg_tensor)

        print(f"\nModel: {model_name}")
        print(f"Embedding shape: {embeddings.shape}")
        print(f"Embedding size: {embeddings.size}")
        print(f"Sample values: {embeddings.flatten()[:5]}")

    print("\nTest completed successfully")


class PerformanceTester:
    def __init__(self, model_name='shallow'):
        """Initialize performance tester with the chosen model"""
        self.model_name = model_name
        print(f"Initializing performance tester with model: {model_name}")

    def run_benchmark(self, n_samples=100, verbose=True):
        """
        Benchmark embedding extraction on multiple EEG samples

        Parameters:
            n_samples (int): Number of samples to process
            verbose (bool): Whether to print progress and results

        Returns:
            dict: Performance metrics
        """
        import pandas as pd

        # Get the train data
        train_df = pd.read_csv(TRAIN_CSV_PATH)

        # Get a random sample
        if len(train_df) < n_samples:
            if verbose:
                print(f"Warning: Requested {n_samples} samples but only {len(train_df)} are available")
            samples = train_df
        else:
            samples = train_df.sample(n=n_samples, random_state=42)

        # Initialize timing and results
        start_time = time.time()

        if verbose:
            print(f"Running benchmark with {self.model_name} model on {len(samples)} samples")

        # Set appropriate sample size
        sample_size = len(samples) / len(train_df)

        # Process samples with the extraction function
        output_file = f"benchmark_{self.model_name}_{n_samples}_samples.csv"

        # Measure extraction time
        embeddings_df, _ = extract_braindecode_embeddings(
            model_name=self.model_name,
            output_file=output_file,
            sample=sample_size,
            batch_size=16 if n_samples > 16 else n_samples
        )

        # Calculate statistics
        total_time = time.time() - start_time

        # Prepare results
        results = {
            'model_name': self.model_name,
            'total_time': total_time,
            'samples_processed': len(embeddings_df),
            'avg_time_per_sample': total_time / (len(embeddings_df) if len(embeddings_df) > 0 else 1),
            'embeddings_per_sample': len(
                [c for c in embeddings_df.columns if c.startswith('embedding_')]) if not embeddings_df.empty else 0
        }

        if verbose:
            self._print_results(results)

        return results

    def _print_results(self, results):
        """Print benchmark results in a formatted way"""
        print("\nBenchmark Results:")
        print(f"Model: {results['model_name']}")
        print(f"Processed {results['samples_processed']} samples")
        print(f"Total time: {results['total_time']:.2f} seconds")
        print(f"Average time per sample: {results['avg_time_per_sample']:.4f} seconds")
        print(f"Embeddings per sample: {results['embeddings_per_sample']}")


def main(test_mode=True):
    """
    Main function to run the embedding extraction process

    Parameters:
        test_mode (bool): If True, run in test mode with sampled data
    """
    # Choose the model to use
    model_name = 'shallow'  # Options: 'shallow', 'deep4', 'eegnet'

    # Run embedding extraction
    if test_mode:
        # Test mode configuration
        output_path = BRAINDECODE_OUT_PATH_TEST
        sample_size = TEST_SAMPLE_SIZE_BRAINDECODE
        print(f"Running in TEST MODE with {sample_size * 100}% of data")
        embeddings_df, original_columns = extract_braindecode_embeddings(
            model_name=model_name,
            output_file=output_path,
            sample=sample_size
        )
    else:
        # Production mode configuration
        output_path = BRAINDECODE_OUT_PATH
        print("Running in PRODUCTION MODE with full dataset")
        embeddings_df, original_columns = extract_braindecode_embeddings(
            model_name=model_name,
            output_file=output_path
        )

    # Display summary
    print("\nEmbedding extraction complete.")
    print(f"Total samples processed: {len(embeddings_df)}")
    print(f"Total embeddings per sample: {len(embeddings_df.columns) - len(original_columns)}")
    print(f"Output saved to: {output_path}")

    # Display sample of the extracted embeddings
    print("\nSample of extracted embeddings:")
    embedding_cols = [col for col in embeddings_df.columns if col.startswith('embedding_')][:5]
    print(embeddings_df[['eeg_id'] + embedding_cols].head())


if __name__ == "__main__":
    # By default, run in test mode
    # Change to main(False) to run on the full dataset
    main(test_mode=True)

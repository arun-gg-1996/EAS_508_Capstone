import pandas as pd
from tqdm import tqdm

from const import DATA_FOLDER, SAMPLE_RATE, NONLINEAR_DOMAIN_OUT_PATH_TEST, NONLINEAR_DOMAIN_OUT_PATH, \
    TEST_SAMPLE_SIZE_NONLINEAR
from data_reader import EEGDataReader
from nonlinear_domain_features import EEGNonlinearExtractor


def extract_all_nonlinear_features(data_folder=DATA_FOLDER, output_file=NONLINEAR_DOMAIN_OUT_PATH, sample=None,
                                   fs=SAMPLE_RATE):
    """
    Extract non-linear domain features for all EEG files in train.csv
    and save them to a CSV file.

    Parameters:
        data_folder (str): Path to the data folder
        output_file (str): Path to save the output CSV file
        sample (float): A number indicating what percent of dataset to run the feature extractor on
        fs (int): Sampling frequency in Hz

    Returns:
        pd.DataFrame: DataFrame containing all extracted features
    """
    # Initialize the necessary objects
    data_reader = EEGDataReader(data_folder)
    feature_extractor = EEGNonlinearExtractor()

    # Load the train.csv data
    if sample:
        train_df = data_reader.get_train_df().sample(frac=sample, random_state=42)
    else:
        train_df = data_reader.get_train_df()

    print(f"Total samples in train.csv: {len(train_df)}")

    # Create a list to store all feature dictionaries
    all_features = []

    # Process each sample in the training data
    for idx, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Extracting non-linear features"):
        eeg_id = str(row['eeg_id'])
        sub_id = row['eeg_sub_id']

        # Extract features for this sample
        features, meta = feature_extractor.process_example(eeg_id, sub_id, data_reader, fs=fs)

        if features:
            # Add identifiers and labels to the features dictionary
            features['eeg_id'] = eeg_id
            features['eeg_sub_id'] = sub_id

            # Add expert_consensus label if available
            if 'expert_consensus' in row:
                features['expert_consensus'] = row['expert_consensus']

            # Append to our collection
            all_features.append(features)
        else:
            print(f"Warning: Could not extract non-linear features for eeg_id={eeg_id}, sub_id={sub_id}")

    # Convert the list of dictionaries to a DataFrame
    features_df = pd.DataFrame(all_features)
    print(f"Extracted non-linear features for {len(features_df)} samples")

    # Save to CSV
    features_df.to_csv(output_file, index=False)
    print(f"Non-linear features saved to {output_file}")

    return features_df


def main(test_mode=True, fs=SAMPLE_RATE):
    """
    Main function to run the non-linear domain feature extraction process

    Parameters:
        test_mode (bool): If True, run in test mode with sampled data
        fs (int): Sampling frequency in Hz
    """
    if test_mode:
        # Test mode configuration
        output_path = NONLINEAR_DOMAIN_OUT_PATH_TEST
        sample_size = TEST_SAMPLE_SIZE_NONLINEAR
        print(f"Running in TEST MODE with {sample_size * 100}% of data")
        features_df = extract_all_nonlinear_features(
            DATA_FOLDER, output_path, sample=sample_size, fs=fs
        )
    else:
        # Production mode configuration
        output_path = NONLINEAR_DOMAIN_OUT_PATH
        print("Running in PRODUCTION MODE with full dataset")
        features_df = extract_all_nonlinear_features(
            DATA_FOLDER, output_path, fs=fs
        )

    # Display summary
    print("\nNon-linear domain feature extraction complete.")
    print(f"Total samples processed: {len(features_df)}")

    # Count feature columns
    feature_cols = [col for col in features_df.columns
                    if col not in ['eeg_id', 'eeg_sub_id', 'expert_consensus']]
    print(f"Total features per sample: {len(feature_cols)}")

    print(f"Output saved to: {output_path}")

    # Display sample of the extracted features
    print("\nSample of extracted non-linear features:")
    # Show fractal dimension and complexity features for the first channel
    sample_cols = [col for col in features_df.columns
                   if 'channel_0' in col and col.endswith(('_fd', '_entropy', '_dfa'))][:5]

    if sample_cols:
        print(features_df[['eeg_id', 'eeg_sub_id', 'expert_consensus'] + sample_cols].head())


if __name__ == "__main__":
    # By default, run in test mode with a sampling frequency of 200 Hz
    # Change to main(False) to run on the full dataset
    main(test_mode=False, fs=SAMPLE_RATE)

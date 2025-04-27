import pandas as pd
from tqdm import tqdm

from const import DATA_FOLDER, TIME_DOMAIN_OUT_PATH, TIME_DOMAIN_OUT_PATH_TEST, TEST_SAMPLE_SIZE_TIME
from data_reader import EEGDataReader
from time_domain_features import EEGFeatureExtractor


def extract_all_features(data_folder=DATA_FOLDER, output_file="eeg_time_domain_features.csv", sample=None):
    """
    Extract time domain features for all EEG files in train.csv
    and save them to a CSV file.

    Parameters:
        data_folder (str): Path to the data folder
        output_file (str): Path to save the output CSV file
        sample (float): A number indicating what percent of dataset do we run the feature extractor on

    Returns:
        pd.DataFrame: DataFrame containing all extracted features
    """
    # Initialize the necessary objects
    data_reader = EEGDataReader(data_folder)
    feature_extractor = EEGFeatureExtractor()

    # Load the train.csv data
    if sample:
        train_df = data_reader.get_train_df().sample(frac=sample, random_state=42)
    else:
        train_df = data_reader.get_train_df()

    print(f"Total samples in train.csv: {len(train_df)}")

    # Create a list to store all feature dictionaries
    all_features = []

    # Process each sample in the training data
    for idx, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Extracting features"):
        eeg_id = str(row['eeg_id'])
        sub_id = row['eeg_sub_id']

        # Extract features for this sample
        features, meta = feature_extractor.process_example(eeg_id, sub_id, data_reader)

        if features:
            # Add identifiers and labels to the features dictionary
            features['eeg_id'] = eeg_id
            features['eeg_sub_id'] = sub_id

            # Add all metadata from the row for reference
            for col in train_df.columns:
                if col not in features:
                    features[col] = row[col]

            # Append to our collection
            all_features.append(features)
        else:
            print(f"Warning: Could not extract features for eeg_id={eeg_id}, sub_id={sub_id}")

    # Convert the list of dictionaries to a DataFrame
    features_df = pd.DataFrame(all_features)
    print(f"Extracted features for {len(features_df)} samples")

    # Save to CSV
    features_df.to_csv(output_file, index=False)
    print(f"Features saved to {output_file}")

    return features_df, train_df.columns


def main(test_mode=False):
    """
    Main function to run the feature extraction process

    Parameters:
        test_mode (bool): If True, run in test mode with sampled data
    """
    if test_mode:
        # Test mode configuration
        output_path = TIME_DOMAIN_OUT_PATH_TEST
        sample_size = TEST_SAMPLE_SIZE_TIME
        print(f"Running in TEST MODE with {sample_size * 100}% of data")
        features_df, original_columns = extract_all_features(DATA_FOLDER, output_path, sample=sample_size)
    else:
        # Production mode configuration
        output_path = TIME_DOMAIN_OUT_PATH
        print("Running in PRODUCTION MODE with full dataset")
        features_df, original_columns = extract_all_features(DATA_FOLDER, output_path)

    # Display summary
    print("\nFeature extraction complete.")
    print(f"Total samples processed: {len(features_df)}")
    print(f"Total features per sample: {len(features_df.columns) - len(original_columns)}")
    print(f"Output saved to: {output_path}")

    # Display sample of the extracted features
    print("\nSample of extracted features:")
    feature_cols = [col for col in features_df.columns if 'channel_' in col][:5]
    print(features_df[['eeg_id', 'eeg_sub_id', 'expert_consensus'] + feature_cols].head())


if __name__ == "__main__":
    # By default, run in test mode
    # Change to main(False) to run on the full dataset
    main(test_mode=False)

import os

# Get the directory containing the script
current_folder = os.path.dirname(os.path.abspath(__file__))

DATA_FOLDER = os.path.join(current_folder, "data")

# Constants for EEG processing
SAMPLE_RATE = 200  # EEG sample rate in Hz
WINDOW_DURATION = 50  # seconds
WINDOW_SIZE = SAMPLE_RATE * WINDOW_DURATION  # 50 seconds window at 200Hz

# create main folder
if not os.path.exists(os.path.join(current_folder, "out", "main")):
    os.makedirs(os.path.join(current_folder, "out", "main"))
    print(f"Created directory: {os.path.join(current_folder, 'out', 'main')}")

# create test folder
if not os.path.exists(os.path.join(current_folder, "out", "test")):
    os.makedirs(os.path.join(current_folder, "out", "test"))
    print(f"Created directory: {os.path.join(current_folder, 'out', 'test')}")

############################## TIME DOMAIN FEATURES ##############################
TIME_DOMAIN_OUT_PATH = os.path.join(current_folder, "out", "main", "eeg_time_domain_features.csv")
TIME_DOMAIN_OUT_PATH_TEST = os.path.join(current_folder, "out", "test", "eeg_time_domain_features_test.csv")
TEST_SAMPLE_SIZE_TIME = 0.2

############################## FREQUENCY DOMAIN FEATURES ##############################
FREQ_DOMAIN_OUT_PATH = os.path.join(current_folder, "out", "main", "eeg_frequency_domain_features.csv")
FREQ_DOMAIN_OUT_PATH_TEST = os.path.join(current_folder, "out", "test", "eeg_frequency_domain_features_test.csv")
TEST_SAMPLE_SIZE_FREQ = 0.05

############################## FREQUENCY DOMAIN FEATURES ##############################
WAVELET_DOMAIN_OUT_PATH = os.path.join(current_folder, "out", "main", "eeg_wavelet_domain_features.csv")
WAVELET_DOMAIN_OUT_PATH_TEST = os.path.join(current_folder, "out", "test", "eeg_wavelet_domain_features_test.csv")
TEST_SAMPLE_SIZE_WAVELET = 0.05

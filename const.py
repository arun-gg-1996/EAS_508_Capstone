import os

# Get the directory containing the script
current_folder = os.path.dirname(os.path.abspath(__file__))

DATA_FOLDER = os.path.join(current_folder, "data")

TRAIN_CSV_PATH = os.path.join(DATA_FOLDER, 'train.csv')

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

############################## NON LINEAR DOMAIN FEATURES ##############################
NONLINEAR_DOMAIN_OUT_PATH_TEST = os.path.join(current_folder, "out", "test", "eeg_nonlinear_domain_features_test.csv")
NONLINEAR_DOMAIN_OUT_PATH = os.path.join(current_folder, "out", "main", "eeg_nonlinear_domain_features.csv")
TEST_SAMPLE_SIZE_NONLINEAR = 0.05

############################## EMBEDDING FEATURES ##############################
BRAINDECODE_OUT_PATH = os.path.join(current_folder, "out", "test", "eeg_embedding_domain_features.csv")
BRAINDECODE_OUT_PATH_TEST = os.path.join(current_folder, "out", "test", "eeg_embedding_domain_features_test.csv")
TEST_SAMPLE_SIZE_BRAINDECODE = 0.05
TRAIN_EEG_DIR = os.path.join(current_folder, "data", "train_eegs")
TRAIN_CSV_PATH = os.path.join(current_folder, "data", "train.csv")
OUTPUT_DIR = os.path.join(current_folder, "out")

############################## VISION FEATURES #################################
# Path configurations - adjust these to match your file structure
# Define output file paths for test and production modes
TRAIN_SPECTROGRAMS_DIR = os.path.join(DATA_FOLDER, 'train_spectrograms')
OUTPUT_DIR_TEST = os.path.join(current_folder, "out", "test")
TEST_SAMPLE_SIZE_VISION = 0.02
VIT_EMBEDDINGS_PATH = os.path.join(OUTPUT_DIR_TEST, 'vit_embeddings.csv')
VIT_EMBEDDINGS_PATH_TEST = os.path.join(OUTPUT_DIR_TEST, 'vit_embeddings_test.csv')

OUTPUT_DIR_ANALYSIS_MAIN = os.path.join(current_folder, 'out', 'main', 'analysis')
OUTPUT_DIR_ANALYSIS_TEST = os.path.join(current_folder, 'out', 'test', 'analysis')

VISION_TSNE_FILE = os.path.join(OUTPUT_DIR_ANALYSIS_MAIN, 'tsne_embeddding.html')
VISION_TSNE_FILE_TEST = os.path.join(OUTPUT_DIR_ANALYSIS_TEST, 'tsne_embeddding_test.html')

# Create output directory
os.makedirs(OUTPUT_DIR_ANALYSIS_TEST, exist_ok=True)
os.makedirs(OUTPUT_DIR_ANALYSIS_MAIN, exist_ok=True)

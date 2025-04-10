import os

# Get the directory containing the script
current_folder = os.path.dirname(os.path.abspath(__file__))

DATA_FOLDER = os.path.join(current_folder, "data")

# Constants for EEG processing
SAMPLE_RATE = 200  # EEG sample rate in Hz
WINDOW_DURATION = 50  # seconds
WINDOW_SIZE = SAMPLE_RATE * WINDOW_DURATION  # 50 seconds window at 200Hz
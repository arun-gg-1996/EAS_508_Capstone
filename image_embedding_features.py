import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import signal
from torchvision import models, transforms


def create_spectrogram_image(eeg_signal, fs=250, nperseg=256, noverlap=None, cmap='viridis'):
    """
    Create a spectrogram image from EEG signal.

    Parameters:
    -----------
    eeg_signal : numpy.ndarray
        Single channel EEG signal
    fs : float
        Sampling frequency in Hz
    nperseg : int
        Length of each segment
    noverlap : int or None
        Number of points to overlap between segments
    cmap : str
        Matplotlib colormap for the spectrogram

    Returns:
    --------
    PIL.Image
        Spectrogram as a PIL Image
    """
    # Create a figure
    plt.figure(figsize=(4, 4), dpi=100)

    # Create spectrogram
    if noverlap is None:
        noverlap = nperseg // 2

    # Use scipy.signal.spectrogram for more control
    f, t, Sxx = signal.spectrogram(eeg_signal, fs=fs, nperseg=nperseg, noverlap=noverlap)

    # Apply logarithmic scaling to the power
    Sxx_log = 10 * np.log10(Sxx + 1e-10)

    # Apply normalization for better contrast
    vmin = max(Sxx_log.min(), -20)  # Clip very low values for better visualization
    vmax = Sxx_log.max()

    # Plot spectrogram
    plt.pcolormesh(t, f, Sxx_log, cmap=cmap, vmin=vmin, vmax=vmax, shading='gouraud')
    plt.axis('off')  # No axes for better feature extraction

    # Save figure to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()
    buf.seek(0)

    # Open image from buffer
    img = Image.open(buf)
    return img


def create_mel_spectrogram_image(eeg_signal, fs=250, n_mels=64, n_fft=512, hop_length=None, cmap='viridis'):
    """
    Create a Mel spectrogram image from EEG signal.

    Parameters:
    -----------
    eeg_signal : numpy.ndarray
        Single channel EEG signal
    fs : float
        Sampling frequency in Hz
    n_mels : int
        Number of Mel bands to generate
    n_fft : int
        Length of the FFT window
    hop_length : int or None
        Number of samples between successive frames
    cmap : str
        Matplotlib colormap for the spectrogram

    Returns:
    --------
    PIL.Image
        Mel spectrogram as a PIL Image
    """
    try:
        import librosa
    except ImportError:
        raise ImportError("librosa is required for Mel spectrogram extraction. Install it with 'pip install librosa'.")

    # Set default hop length if not provided
    if hop_length is None:
        hop_length = n_fft // 4

    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=eeg_signal,
        sr=fs,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=0.5,  # Lowest EEG frequency of interest
        fmax=min(fs / 2, 100)  # Highest EEG frequency of interest, capped at 100 Hz
    )

    # Convert to dB scale
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Create a figure
    plt.figure(figsize=(4, 4), dpi=100)

    # Plot mel spectrogram
    plt.imshow(mel_spec_db, aspect='auto', origin='lower', cmap=cmap)
    plt.axis('off')  # No axes for better feature extraction

    # Save figure to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()
    buf.seek(0)

    # Open image from buffer
    img = Image.open(buf)
    return img


def create_scalogram_image(eeg_signal, fs=250, wavelet='morl', scales=None, cmap='viridis'):
    """
    Create a scalogram (continuous wavelet transform) image from EEG signal.

    Parameters:
    -----------
    eeg_signal : numpy.ndarray
        Single channel EEG signal
    fs : float
        Sampling frequency in Hz
    wavelet : str
        Wavelet to use for transform
    scales : numpy.ndarray or None
        Scales for the wavelet transform. If None, scales are automatically determined
    cmap : str
        Matplotlib colormap for the scalogram

    Returns:
    --------
    PIL.Image
        Scalogram as a PIL Image
    """
    try:
        from pywt import cwt
    except ImportError:
        raise ImportError("PyWavelets is required for scalogram creation. Install it with 'pip install PyWavelets'.")

    # Set default scales if not provided
    if scales is None:
        # Calculate appropriate scales for EEG frequencies of interest (0.5-100 Hz)
        min_scale = fs / (2 * 100)  # Scale for 100 Hz
        max_scale = fs / (2 * 0.5)  # Scale for 0.5 Hz
        scales = np.logspace(np.log10(min_scale), np.log10(max_scale), 64)

    # Compute continuous wavelet transform
    coef, freqs = cwt(eeg_signal, scales, wavelet, 1.0 / fs)

    # Convert to power
    power = np.abs(coef) ** 2

    # Apply logarithmic scaling
    power_log = np.log10(power + 1e-10)

    # Create a figure
    plt.figure(figsize=(4, 4), dpi=100)

    # Plot scalogram
    plt.imshow(power_log, aspect='auto', cmap=cmap)
    plt.axis('off')  # No axes for better feature extraction

    # Save figure to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()
    buf.seek(0)

    # Open image from buffer
    img = Image.open(buf)
    return img


def create_multiscale_image(eeg_signal, fs=250, cmap='viridis'):
    """
    Create a multi-view image with different time-frequency representations combined.

    Parameters:
    -----------
    eeg_signal : numpy.ndarray
        Single channel EEG signal
    fs : float
        Sampling frequency in Hz
    cmap : str
        Matplotlib colormap

    Returns:
    --------
    PIL.Image
        Multi-scale image as a PIL Image
    """
    # Create a figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=100)

    # 1. Spectrogram
    f, t, Sxx = signal.spectrogram(eeg_signal, fs=fs, nperseg=min(256, len(eeg_signal) // 8), noverlap=None)
    Sxx_log = 10 * np.log10(Sxx + 1e-10)
    vmin = max(Sxx_log.min(), -20)
    axes[0].pcolormesh(t, f, Sxx_log, cmap=cmap, vmin=vmin, shading='gouraud')
    axes[0].set_title('Spectrogram')
    axes[0].set_xlabel('Time [s]')
    axes[0].set_ylabel('Frequency [Hz]')

    # 2. Mel spectrogram (if librosa is available)
    try:
        import librosa
        mel_spec = librosa.feature.melspectrogram(
            y=eeg_signal,
            sr=fs,
            n_fft=512,
            hop_length=128,
            n_mels=64,
            fmin=0.5,
            fmax=min(fs / 2, 100)
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        axes[1].imshow(mel_spec_db, aspect='auto', origin='lower', cmap=cmap)
        axes[1].set_title('Mel Spectrogram')
        axes[1].set_xlabel('Time Frame')
        axes[1].set_ylabel('Mel Band')
    except ImportError:
        axes[1].text(0.5, 0.5, 'Librosa not available', ha='center', va='center')

    # 3. Scalogram (if PyWavelets is available)
    try:
        from pywt import cwt
        min_scale = fs / (2 * 100)
        max_scale = fs / (2 * 0.5)
        scales = np.logspace(np.log10(min_scale), np.log10(max_scale), 64)
        coef, freqs = cwt(eeg_signal, scales, 'morl', 1.0 / fs)
        power = np.abs(coef) ** 2
        power_log = np.log10(power + 1e-10)
        axes[2].imshow(power_log, aspect='auto', cmap=cmap)
        axes[2].set_title('Scalogram (CWT)')
        axes[2].set_xlabel('Time')
        axes[2].set_ylabel('Scale')
    except ImportError:
        axes[2].text(0.5, 0.5, 'PyWavelets not available', ha='center', va='center')

    # Adjust layout
    plt.tight_layout()

    # Save figure to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)

    # Open image from buffer
    img = Image.open(buf)
    return img


def extract_image_embedding(img, model_name='efficientnet_b0'):
    """
    Extract embeddings from an image using a pre-trained model.

    Parameters:
    -----------
    img : PIL.Image
        Input image
    model_name : str
        Name of the pre-trained model to use

    Returns:
    --------
    numpy.ndarray
        Image embeddings
    """
    # Define preprocessing
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Preprocess image
    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0)  # Add batch dimension

    # Load pre-trained model
    if model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=True)
        # Remove classifier
        model = torch.nn.Sequential(*list(model.children())[:-1])
    elif model_name == 'resnet18':
        model = models.resnet18(pretrained=True)
        # Remove classifier
        model = torch.nn.Sequential(*list(model.children())[:-1])
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
        # Remove classifier
        model = torch.nn.Sequential(*list(model.children())[:-1])
    elif model_name == 'mobilenet_v3_small':
        model = models.mobilenet_v3_small(pretrained=True)
        # Remove classifier
        model = torch.nn.Sequential(*list(model.children())[:-1])
    elif model_name == 'densenet121':
        model = models.densenet121(pretrained=True)
        # Remove classifier
        model = torch.nn.Sequential(*list(model.children())[:-1])
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # Set model to evaluation mode
    model.eval()

    # Extract features
    with torch.no_grad():
        features = model(input_batch)

    # Flatten features
    embeddings = features.squeeze().flatten().numpy()

    return embeddings


def extract_embeddings_from_eeg(eeg_signal, fs=250, model_name='efficientnet_b0', image_type='all'):
    """
    Extract embeddings from EEG signal using different time-frequency representations.

    Parameters:
    -----------
    eeg_signal : numpy.ndarray
        Single channel EEG signal
    fs : float
        Sampling frequency in Hz
    model_name : str
        Name of the pre-trained model to use
    image_type : str
        Type of image to create ('spectrogram', 'mel', 'scalogram', 'multi', or 'all')

    Returns:
    --------
    dict
        Dictionary of embeddings for different image types
    """
    embeddings = {}

    if image_type == 'spectrogram' or image_type == 'all':
        # Create spectrogram image
        spec_img = create_spectrogram_image(eeg_signal, fs)
        # Extract embeddings
        spec_embeddings = extract_image_embedding(spec_img, model_name)
        # Store with prefix
        for i, val in enumerate(spec_embeddings):
            embeddings[f'spec_emb_{i}'] = val

    if image_type == 'mel' or image_type == 'all':
        try:
            # Create mel spectrogram image
            mel_img = create_mel_spectrogram_image(eeg_signal, fs)
            # Extract embeddings
            mel_embeddings = extract_image_embedding(mel_img, model_name)
            # Store with prefix
            for i, val in enumerate(mel_embeddings):
                embeddings[f'mel_emb_{i}'] = val
        except ImportError as e:
            print(f"Skipping Mel spectrogram: {e}")

    if image_type == 'scalogram' or image_type == 'all':
        try:
            # Create scalogram image
            scal_img = create_scalogram_image(eeg_signal, fs)
            # Extract embeddings
            scal_embeddings = extract_image_embedding(scal_img, model_name)
            # Store with prefix
            for i, val in enumerate(scal_embeddings):
                embeddings[f'scal_emb_{i}'] = val
        except ImportError as e:
            print(f"Skipping scalogram: {e}")

    if image_type == 'multi' or image_type == 'all':
        # Create multi-scale image
        multi_img = create_multiscale_image(eeg_signal, fs)
        # Extract embeddings
        multi_embeddings = extract_image_embedding(multi_img, model_name)
        # Store with prefix
        for i, val in enumerate(multi_embeddings):
            embeddings[f'multi_emb_{i}'] = val

    return embeddings


def dimensionality_reduction(embeddings, n_components=100):
    """
    Reduce dimensionality of embeddings using PCA.

    Parameters:
    -----------
    embeddings : numpy.ndarray
        Embeddings with shape (n_samples, n_features)
    n_components : int
        Number of components to keep

    Returns:
    --------
    numpy.ndarray
        Reduced embeddings with shape (n_samples, n_components)
    """
    from sklearn.decomposition import PCA

    # Check if dimensionality reduction is needed
    if embeddings.shape[1] <= n_components:
        return embeddings

    # Apply PCA
    pca = PCA(n_components=n_components)
    reduced_embeddings = pca.fit_transform(embeddings)

    return reduced_embeddings


def extract_multichannel_image_embeddings(eeg_signals, fs=250, model_name='efficientnet_b0', image_type='spectrogram',
                                          reduce_dim=True, n_components=100):
    """
    Extract image embeddings for multiple EEG channels.

    Parameters:
    -----------
    eeg_signals : numpy.ndarray
        Multi-channel EEG signals with shape (n_channels, n_samples)
    fs : float
        Sampling frequency in Hz
    model_name : str
        Name of the pre-trained model to use
    image_type : str
        Type of image to create ('spectrogram', 'mel', 'scalogram', 'multi', or 'all')
    reduce_dim : bool
        Whether to reduce dimensionality of embeddings
    n_components : int
        Number of components to keep after dimensionality reduction

    Returns:
    --------
    pandas.DataFrame
        DataFrame with image embeddings for all channels
    """
    all_channel_embeddings = []

    for ch_idx, eeg_signal in enumerate(eeg_signals):
        # Extract embeddings for this channel
        channel_embeddings = extract_embeddings_from_eeg(eeg_signal, fs, model_name, image_type)

        # Add channel index
        channel_embeddings['channel'] = ch_idx

        all_channel_embeddings.append(channel_embeddings)

    # Convert to DataFrame
    embeddings_df = pd.DataFrame(all_channel_embeddings)

    # Dimensionality reduction if requested
    if reduce_dim:
        # Select only embedding columns
        emb_cols = [col for col in embeddings_df.columns if ('emb_' in col and col != 'channel')]

        if emb_cols:
            # Extract embeddings
            embeddings = embeddings_df[emb_cols].values

            # Apply dimensionality reduction
            reduced_embeddings = dimensionality_reduction(embeddings, n_components)

            # Replace embeddings with reduced version
            for i in range(reduced_embeddings.shape[1]):
                embeddings_df[f'reduced_emb_{i}'] = reduced_embeddings[:, i]

            # Drop original embeddings to save memory
            embeddings_df.drop(columns=emb_cols, inplace=True)

    return embeddings_df

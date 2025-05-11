import numpy as np
import pandas as pd
from scipy.signal import welch, stft
from scipy.stats import skew, kurtosis, entropy
from scipy.integrate import simpson
import pywt
import os

# EEG Frequency Bands
bands = {
    'Delta': (0.5, 4),
    'Theta': (4, 8),
    'Alpha': (8, 12),
    'Beta': (12, 30),
    'Gamma': (30, 100)
}

def extract_spatial_features(eeg_data):
    """Extract spatial (time-domain) features from EEG signals."""
    features = pd.DataFrame()
    
    for col in eeg_data.columns:
        features[f'Zero-crossing Rate_{col}'] = [((np.diff(np.sign(eeg_data[col]), axis=0) != 0).sum(axis=0)) / len(eeg_data[col])]
        features[f'Peak-to-Peak Amplitude_{col}'] = [eeg_data[col].max(axis=0) - eeg_data[col].min(axis=0)]
        features[f'RMS_{col}'] = [np.sqrt((eeg_data[col]**2).mean(axis=0))]  # Root Mean Square
    
    #print(f'Spatial: {features.head()}')
    #print(f'Size: {features.shape}')
    return features

def extract_frequency_features(eeg_data, fs=1000):
    """Compute Power Spectral Density (PSD) and extract band power."""
    freq_features = pd.DataFrame()
    
    for col in eeg_data.columns:
        f, Pxx = welch(eeg_data[col], fs=fs, nperseg=fs*2)
        
        total_power = simpson(Pxx, f)  # Total Power (Integral of PSD)

        for band, (low, high) in bands.items():
            band_power = simpson(Pxx[(f >= low) & (f < high)], f[(f >= low) & (f < high)])
            freq_features[f'{band} Power_{col}'] = [band_power]
            freq_features[f'{band} Relative Power (%)_{col}'] = [(band_power / total_power) * 100]
            #freq_features.loc[col, f'{band} Power'] = band_power
            #freq_features.loc[col, f'{band} Relative Power (%)'] = (band_power / total_power) * 100
        
        # Spectral Entropy: Measures randomness in frequency components
        P_norm = Pxx / np.sum(Pxx)
        spectral_entropy = -np.sum(P_norm * np.log2(P_norm))
        freq_features[f'Spectral Entropy_{col}'] = [spectral_entropy]
        #freq_features.loc[col, 'Spectral Entropy'] = spectral_entropy
        
        # Dominant Frequency: Frequency with highest power
        dominant_freq = f[np.argmax(Pxx)]
        freq_features[f'Dominant Frequency_{col}'] = [dominant_freq]
        #freq_features.loc[col, 'Dominant Frequency'] = dominant_freq
        
    #print(f'Freq: {freq_features.head()}')
    #print(f'Size: {freq_features.shape}')
    return freq_features


# **TIME-FREQUENCY FEATURES (STFT)**
def extract_time_frequency_features(eeg_data, fs=1000):
    """Extract features from Short-Time Fourier Transform (STFT)."""
    time_freq_features = pd.DataFrame()
    
    for col in eeg_data.columns:
        f, t, Zxx = stft(eeg_data[col], fs=fs, nperseg=fs//2)
        
        # Compute total spectral power
        total_spectral_power = np.sum(np.abs(Zxx)**2)
        
        # Compute Spectral Centroid
        spectral_centroid = np.sum(f[:, None] * np.abs(Zxx)) / np.sum(np.abs(Zxx))

        time_freq_features[f'Total Spectral Power_{col}'] = [total_spectral_power]
        time_freq_features[f'Spectral Centroid_{col}'] = [spectral_centroid]
        #time_freq_features.loc[col, 'Total Spectral Power'] = total_spectral_power
        #time_freq_features.loc[col, 'Spectral Centroid'] = spectral_centroid
        
    #print(f'Time Frequency: {time_freq_features.head()}')
    #print(f'Size: {time_freq_features.shape}')
    return time_freq_features

# **WAVELET FEATURES**
def extract_wavelet_features(eeg_data, wavelet='db4', level=5):
    """Extract wavelet energy and entropy features."""
    wavelet_features = pd.DataFrame()
    
    for col in eeg_data.columns:
        coeffs = pywt.wavedec(eeg_data[col], wavelet, level=level)
        
        # Compute Energy for each level
        wavelet_energy = [np.sum(np.square(c)) for c in coeffs]
        wavelet_entropy = entropy(wavelet_energy)

        for i, energy in enumerate(wavelet_energy):
            wavelet_features[f'Wavelet Energy L{i}_{col}'] = [energy]
            #wavelet_features.loc[col, f'Wavelet Energy L{i}'] = energy
        
        wavelet_features[f'Wavelet Entropy_{col}'] = [wavelet_entropy]
        #wavelet_features.loc[col, 'Wavelet Entropy'] = wavelet_entropy
    
    #print(f'Wavelet: {wavelet_features.head()}')
    #print(f'Size: {wavelet_features.shape}')
    return wavelet_features

# **STATISTICAL FEATURES**
def extract_statistical_features(eeg_data):
    """Extract all statistical features from EEG signals."""
    features = pd.DataFrame()
    
    for col in eeg_data.columns:
        features[f'Mean_{col}'] = [eeg_data[col].mean(axis=0)]
        features[f'Median_{col}'] = [eeg_data[col].median(axis=0)]
        features[f'Variance_{col}'] = [eeg_data[col].var(axis=0)]
        features[f'IQR_{col}'] = [np.percentile(eeg_data[col], 75) - np.percentile(eeg_data[col], 25)]
        features[f'Skewness_{col}'] = [skew(eeg_data[col])]
        features[f'Kurtosis_{col}'] = [kurtosis(eeg_data[col])]

    #print(f'Statistical: {features.head()}')
    #print(f'Size: {features.shape}')
    return features


files = [f for f in os.listdir('.') if f.endswith('.csv')]
i = 0

print(len(files))

features_ndf = pd.DataFrame()
for file in files:
    try:
        df = pd.read_csv(file) 
        features_df = pd.DataFrame()
        cols = ['Ch1', 'Ch2', 'Ch3', 'Ch4', 'Ch5', 'Ch6', 'Ch7', 'Ch8']
        #df.columns = cols     
        for col in cols:
            spatial_ft = extract_spatial_features(df[[col]])  # Use double brackets
            frequency_ft = extract_frequency_features(df[[col]])
            time_frequency_ft = extract_time_frequency_features(df[[col]])
            wavelet_frequency_ft = extract_wavelet_features(df[[col]])
            statistical_frequency_ft = extract_statistical_features(df[[col]])
            features = [
                features_df,
                spatial_ft,
                frequency_ft,
                time_frequency_ft,
                wavelet_frequency_ft,
                statistical_frequency_ft
            ]  
            features_df = pd.concat(features, axis=1)
        
        features_df['Label'] = df['Label'].iloc[0]
        
        ft = [features_ndf, features_df]
        features_ndf = pd.concat(ft, axis=0)
        i += 1
        print(i)
        
    except Exception as e:
        print(f"Error processing {file}: {e}")
  
print(features_ndf.head())      
print(features_ndf.shape)

features_ndf.to_csv(f'RS_Features.csv', index=False)

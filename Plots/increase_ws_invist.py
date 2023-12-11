import numpy as np
import librosa
import scipy.fftpack
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import librosa
import scipy.fftpack
from sklearn.model_selection import train_test_split
from gammatone import gtgram
import os
import csv
import sys
import pickle
import scipy.signal
from scipy.stats import entropy
import numpy as np
import librosa
from gammatone import gtgram
from gammatone.filters import erb_filterbank, make_erb_filters, centre_freqs
folder_path = "/home/yassin.terraf/OSD/Codes/spafe"
sys.path.append(folder_path)
from spafe.features import gfcc
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm
import numpy as np
import librosa
import scipy.fftpack
from sklearn.model_selection import train_test_split
from gammatone import gtgram
import os
import csv
import sys
import pickle
import scipy.signal
from scipy.stats import entropy
import numpy as np
import librosa
from gammatone import gtgram
from gammatone.filters import erb_filterbank, make_erb_filters, centre_freqs
folder_path = "/home/yassin.terraf/OSD/Codes/spafe"
sys.path.append(folder_path)
from spafe.features import pncc
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable



def compute_basic_pncc(audio, sr=25000, n_pncc=12, n_gfb=40):
    audio=remove_silence(audio)
    pnccs=pncc.compute_pncc(audio)
    return pnccs


def compute_E_pncc_TCA(audio, sr=25000, n_pncc=12, n_gfb=40, context_size=5):

    basic_pncc = compute_basic_pncc(audio, sr, n_pncc, n_gfb)

    # Temporal averaging
    averaged_pncc = temporal_context_average(basic_pncc, context=context_size)

    return averaged_pncc  # or return combined_features if combined

def preemphasis(signal):
    return np.append(signal[0], signal[1:] - 0.97 * signal[:-1])

def remove_silence(audio_data):
    clips = librosa.effects.split(audio_data, top_db=10)
    wav_data = []
    for c in clips:
        data = audio_data[c[0]: c[1]]
        wav_data.extend(data)
    return np.array(wav_data)





def preemphasis(signal):
    return np.append(signal[0], signal[1:] - 0.97 * signal[:-1])


def remove_silence(audio_data):
    clips = librosa.effects.split(audio_data, top_db=10)
    wav_data = []
    for c in clips:
        data = audio_data[c[0]: c[1]]
        wav_data.extend(data)
    return np.array(wav_data)



def compute_basic_gfcc(audio, sr=25000, n_gfcc=12, n_gfb=40):

    # Compute Basic GFCC for original audio
    frame_length, hop_length = int(sr * 0.025), int(sr * 0.01)
    

    # Pre-processing, like silence removal
    audio = remove_silence(audio)
    
    # Compute enhanced GFCC
    gamma_gram = gtgram.gtgram(audio, sr, window_time=frame_length/sr, hop_time=hop_length/sr, channels=n_gfb, f_min=50)
    log10_gamma_gram = np.log10(gamma_gram)
    gfccs = scipy.fftpack.dct(log10_gamma_gram, axis=0, norm='ortho')[1:n_gfcc+1]

    return gfccs
    
def compute_basic_mfcc(signal, sr=25000, n_mfcc=12, n_mfb=40):
    audio = remove_silence(signal)
    pre_emphasized = preemphasis(audio)
    frame_length, hop_length = int(sr * 0.025), int(sr * 0.01)
    n_fft = 1024
    S = librosa.stft(y=pre_emphasized, n_fft=n_fft, hop_length=hop_length, win_length=frame_length, window='hamming')
    mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mfb)
    mel_energy = np.dot(mel_basis, np.abs(S) ** 2)
    mfb = np.log(mel_energy)
    mfccs = scipy.fftpack.dct(mfb, axis=0, norm='ortho')[1:n_mfcc + 1]
    return mfccs

def temporal_context_average(features, context):
    num_features, num_frames = features.shape
    padded_features = np.pad(features, ((0, 0), (context//2, context//2)), mode='reflect')
    averaged_features = np.zeros_like(features)
    for i in range(num_frames):
        averaged_features[:, i] = np.mean(padded_features[:, i:i + context], axis=1)
    return averaged_features

def compute_E_mfcc_TCA(audio, sr=25000, n_mfcc=12, n_mfb=40, context_size=5):
    basic_mfcc = compute_basic_mfcc(audio, sr, n_mfcc, n_mfb)
    averaged_mfcc = temporal_context_average(basic_mfcc, context_size)
    return averaged_mfcc

def plot_features(features, title, ax, sr, hop_length):
    im = ax.imshow(features, aspect='auto', origin='lower', interpolation='none', cmap='viridis')
    # Dynamically adjust the color limits for better contrast
    percentile_low, percentile_high = np.percentile(features, [5, 95])
    im.set_clim(vmin=percentile_low, vmax=percentile_high)
    # Calculate the total duration of the audio in seconds
    total_duration_sec = features.shape[1] * hop_length / sr
    times = np.arange(features.shape[1]) * hop_length / sr
    # Set the ticks and labels on the x-axis
    ax.set_xticks(np.arange(0, len(times), len(times) // 10))
    ax.set_xticklabels(["{:.2f}".format(t) for t in ax.get_xticks()], fontsize=14)
    # Set title and axis labels with standardized font size
    ax.set_title(title, fontsize=16)
    ax.set_ylabel('MFCC Coefficients', fontsize=15)
    ax.set_xlabel('Time (s)', fontsize=15)

def visualize_combined_features(audio_samples, file_names, context_sizes, custom_contexts=None):
    # Increase the figure size
    fig_width = 9 * (len(context_sizes) + 1)  # 9 inches for each subplot column
    fig_height = 6 * len(audio_samples)  # 6 inches for each subplot row for increased height
    fig, axs = plt.subplots(len(audio_samples), len(context_sizes) + 1, figsize=(fig_width, fig_height))

    # Increase font sizes
    title_fontsize = 18  # Increased for better visibility
    label_fontsize = 14
    tick_labelsize = 12

    max_val = 0  # Variable to store the maximum value for the colorbar

    for i, audio_path in enumerate(audio_samples):
        y, sr = librosa.load(audio_path, sr=25000)
        basic_features = compute_basic_mfcc(y, sr)

        # Check if the file needs custom context sizes
        if custom_contexts and audio_path in custom_contexts:
            current_context_sizes = custom_contexts[audio_path]
        else:
            current_context_sizes = context_sizes

        # Plot the conventional features with larger font size
        plot_features(basic_features, f'Conventional MFCC- {file_names[i]}', axs[i, 0], sr, hop_length=int(sr * 0.01))
        axs[i, 0].set_title(f'Conventional MFCC - {file_names[i]}', fontsize=title_fontsize)
        axs[i, 0].set_ylabel('MFCC Coefficients', fontsize=label_fontsize)
        axs[i, 0].tick_params(axis='both', which='major', labelsize=tick_labelsize)

        # Find the max value for the colorbar range
        max_val = max(max_val, basic_features.max())

        for j, context_size in enumerate(current_context_sizes):
            enhanced_features = compute_E_mfcc_TCA(y, sr, context_size=context_size)
            plot_features(enhanced_features, f'TCEF-MFCC (Context size {context_size})', axs[i, j+1], sr, hop_length=int(sr * 0.01))
            axs[i, j+1].set_title(f'TCEF-MFCC (Context size {context_size})', fontsize=title_fontsize)
            axs[i, j+1].tick_params(axis='both', which='major', labelsize=tick_labelsize)

            # Update the max value for the colorbar range
            max_val = max(max_val, enhanced_features.max())

    # Set the color normalization and create a scalar mappable for the colorbar
    norm = Normalize(vmin=0, vmax=max_val)
    sm = ScalarMappable(norm=norm, cmap='viridis')

    # Add the colorbar to the right of the subplots
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # Position the colorbar
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.ax.set_ylabel('Scale', fontsize=label_fontsize)
    cbar.ax.tick_params(labelsize=tick_labelsize)

    # Adjust layout to prevent overlap
    plt.tight_layout(pad=3.0, rect=[0, 0, 0.9, 1])  # Adjust the right side to make room for the colorbar

    # Save the figure
    save_path = "lustre/rsc_mgt_wi-st-sccs-lj7uaansp4q/users/yassin.terraf/test_mfccccs/combined_features_contexts.png"  # Change this to your preferred save path
    plt.savefig(save_path, dpi=300, bbox_inches='tight')  # bbox_inches='tight' to include all content
    print(f"Saved combined plot to {save_path}")

audio_samples = [
    '/home/yassin.terraf/lustre/rsc_mgt_wi-st-sccs-lj7uaansp4q/yassin.terraf/FeatureExtraction/dataset/raw/Grid_clean/s1/bbaf2n.wav',
    '/home/yassin.terraf/lustre/rsc_mgt_wi-st-sccs-lj7uaansp4q/yassin.terraf/FeatureExtraction/dataset/raw/Grid_noise_reverb/s1/bbaf2n.wav',
    '/home/yassin.terraf/lustre/rsc_mgt_wi-st-sccs-lj7uaansp4q/yassin.terraf/FeatureExtraction/dataset/raw/Ravdess_clean/Actor_12/03-01-02-02-02-02-12.wav',
    '/home/yassin.terraf/lustre/rsc_mgt_wi-st-sccs-lj7uaansp4q/yassin.terraf/FeatureExtraction/dataset/raw/RAVDESS_noise_reverb/Actor_12/03-01-02-02-02-02-12.wav'
]

file_names = [
    'Neutral speech', 'Neutral speech with NR',
    'Emotional speech', 'Emotional speech with NR'
]
# context_sizes = [3, 6, 9]
# custom_contexts = {
#     '/path': [5,10, 15],
#     '/path':  [5,10, 15],
#     '/path':  [2,4, 6],

# }

visualize_combined_features(audio_samples, file_names, context_sizes)



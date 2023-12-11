import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def remove_silence(audio_data, sr):
    clips = librosa.effects.split(audio_data, top_db=10)
    wav_data = []
    for c in clips:
        data = audio_data[c[0]: c[1]]
        wav_data.extend(data)
    return np.array(wav_data)

# Load audios
conditions = [
    "Neutral Speech",
    "Neutral Speech with NR",
    "Emotional Speech",
    "Emotional Speech with NR"
]
audio_paths = [
    "lustre/rsc_mgt_wi-st-sccs-lj7uaansp4q/yassin.terraf/FeatureExtraction/dataset/raw/Grid_clean/s1/bbaf2n.wav",
    "lustre/rsc_mgt_wi-st-sccs-lj7uaansp4q/yassin.terraf/FeatureExtraction/dataset/raw/testfiles/GridOutput.wav",
    "/home/yassin.terraf/lustre/rsc_mgt_wi-st-sccs-lj7uaansp4q/yassin.terraf/FeatureExtraction/dataset/raw/Ravdess_clean/Actor_04/03-01-05-01-01-01-04.wav",
    "lustre/rsc_mgt_wi-st-sccs-lj7uaansp4q/yassin.terraf/FeatureExtraction/dataset/raw/testfiles/RAVDESSOutput.wav",
]
spectrograms = []

# Load and compute the spectrograms
for path in audio_paths:
    y, sr = librosa.load(path, sr=None)
    y = remove_silence(y, sr)  # Remove silence
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    spectrograms.append(D)

# Plotting
fig, axs = plt.subplots(1, 4, figsize=(34, 8))

for i, (D, condition) in enumerate(zip(spectrograms, conditions)):
    ax = axs[i]
    img = librosa.display.specshow(D, sr=sr, x_axis='time', cmap='jet', ax=ax)
    ax.set_yscale('log')
    ax.set_ylim(bottom=10)  # Adjust as needed, to avoid log(0) issues.
    ax.set_title(f" {condition}", fontsize=24)  # Increased title fontsize
    
    # Get the duration of the audio 'y' and set x-axis ticks accordingly
    duration = len(y) / sr
    ax.set_xticks([0, duration/2, duration])
    ax.tick_params(axis='both', which='major', labelsize=16)  # Increase tick label fontsize
    
    if i == 0:
        ax.set_ylabel("Frequency (Hz)", fontsize=16)  # Increase y-axis label fontsize
    ax.set_xlabel("Time (s)", fontsize=16)  # Increase x-axis label fontsize

# Add a single colorbar to the right of all subplots
cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])
cbar = fig.colorbar(img, cax=cbar_ax, format='%+2.0f dB')
cbar.ax.tick_params(labelsize=16)  # Increase colorbar tick label fontsize

# Reducing space between subplots
plt.subplots_adjust(wspace=0.1)

# Save the plot to a file
plt.savefig("plot_spectrograms.png", dpi=300, bbox_inches='tight')
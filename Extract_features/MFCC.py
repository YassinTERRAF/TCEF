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



def preemphasis(signal):
    return np.append(signal[0], signal[1:] - 0.97 * signal[:-1])


def remove_silence(audio_data):
    clips = librosa.effects.split(audio_data, top_db=10)
    wav_data = []
    for c in clips:
        data = audio_data[c[0]: c[1]]
        wav_data.extend(data)
    return np.array(wav_data)




def compute_basic_mfcc(signal, sr=25000, n_mfcc=12, n_mfb=40):

    audio = remove_silence(signal)
    # Apply pre-emphasis filter
    pre_emphasized = preemphasis(audio)
    
    # STFT parameters
    frame_length, hop_length = int(sr * 0.025), int(sr * 0.01)  # Explicitly cast to integers
    n_fft = 1024
    
    # Calculate STFT
    S = librosa.stft(y=pre_emphasized, n_fft=n_fft, hop_length=hop_length, win_length=frame_length, window='hamming')
    
    # Apply mel filter bank
    mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mfb)
    mel_energy = np.dot(mel_basis, np.abs(S) ** 2)
    
    # Apply log to get MFB, using a small constant to avoid log(0)
    mfb = np.log(mel_energy)
    
    # Apply DCT to get MFCCs excluding the 0th coefficient and only consider the next 12
    mfccs = scipy.fftpack.dct(mfb, axis=0, norm='ortho')[1:n_mfcc + 1]
    return mfccs


def compute_basic_mfcc_with_derivatives(audio, sr=25000, n_mfcc=12, n_gfb=40):
    mfccs = compute_basic_mfcc(audio, sr, n_mfcc, n_gfb)
    mfcc_delta = librosa.feature.delta(mfccs, width=5)
    mfcc_delta2 = librosa.feature.delta(mfccs, order=2, width=5)
    return np.vstack([mfccs, mfcc_delta, mfcc_delta2])





def temporal_context_average(features, context):
    num_features, num_frames = features.shape
    averaged_features = np.zeros((num_features, num_frames))
    
    # Averaging frames within the context window for the bulk of frames
    for i in range(num_frames - context + 1):
        averaged_features[:, i] = np.mean(features[:, i:i + context], axis=1)
    
    # Handle the remaining frames with a decreasing context
    for i in range(num_frames - context + 1, num_frames):
        remaining_context = num_frames - i
        averaged_features[:, i] = np.mean(features[:, i:i + remaining_context], axis=1)

    return averaged_features

    
def compute_E_mfcc_TCA(audio, sr=25000, n_mfcc=12, n_gfb=40, context_size=1):

    basic_mfcc = compute_basic_mfcc(audio, sr, n_mfcc, n_gfb)

    # Temporal averaging
    averaged_mfcc = temporal_context_average(basic_mfcc, context=context_size)

    return averaged_mfcc  # or return combined_features if combined


def compute_E_mfcc_TCA_with_TCA_derivatives(audio, sr=25000, n_mfcc=12, n_gfb=40, context_size=1):


    basic_mfcc = compute_basic_mfcc(audio, sr, n_mfcc, n_gfb)


    # Temporal averaging
    averaged_mfcc = temporal_context_average(basic_mfcc, context=context_size)


    # Calculate the first and second derivatives of the TCA GFCCs
    delta = librosa.feature.delta(averaged_mfcc ,width=5)
    delta_delta = librosa.feature.delta(averaged_mfcc, order=2, width=5)
    
    # Stack the TCA GFCCs, its first and second derivatives
    combined_features = np.vstack((averaged_mfcc, delta, delta_delta))
    
    return combined_features



def compute_E_mfcc_TCA_and_basic_mfcc(audio, sr=25000, n_mfcc=12, n_gfb=40, context_size=1):
    basic_mfcc = compute_basic_mfcc(audio, sr, n_mfcc, n_gfb)
    averaged_mfcc = compute_E_mfcc_TCA(audio, sr, n_mfcc, n_gfb, context_size)
    
    # Stack the basic MFCC and the TCA MFCC features vertically
    combined_features = np.vstack((basic_mfcc, averaged_mfcc))


    return combined_features



def compute_E_mfcc_TCA_and_basic_mfcc_with_basic_mfcc_derivatives(audio, sr=25000, n_mfcc=12, n_gfb=40, context_size=1):
    
    basic_mfcc = compute_basic_mfcc(audio, sr, n_mfcc, n_gfb)

    # Compute the enhanced MFCC with TCA
    averaged_mfcc = compute_E_mfcc_TCA(audio, sr, n_mfcc, n_gfb, context_size)

    # Compute the first (delta) and second (delta-delta) derivatives of TCA MFCC using librosa
    delta_basic_MFCC = librosa.feature.delta(averaged_mfcc, width=5)

    delta_delta_basic_MFCC= librosa.feature.delta(averaged_mfcc, order=2, width=5)
    
    # Stack the basic MFCC, the TCA MFCC, and its derivatives vertically
    combined_features = np.vstack((basic_mfcc, averaged_mfcc, delta_basic_MFCC, delta_delta_basic_MFCC))
    

    return combined_features




def compute_E_mfcc_TCA_with_basic_mfcc_and_mfcc_TCA_derivatives(audio, sr=25000, n_mfcc=12, n_gfb=40, context_size=1):


    basic_mfcc = compute_basic_mfcc(audio, sr, n_mfcc, n_gfb)
    
    # Compute the enhanced MFCC with TCA
    averaged_mfcc = compute_E_mfcc_TCA(audio, sr, n_mfcc, n_gfb, context_size)
    

    # Compute the first (delta) and second (delta-delta) derivatives of TCA MFCC using librosa
    delta_TCA = librosa.feature.delta(averaged_mfcc, width=5)

    delta_delta_TCA= librosa.feature.delta(averaged_mfcc, order=2, width=5)


    # Stack the basic MFCC, the TCA MFCC, and its derivatives vertically
    combined_features = np.vstack((basic_mfcc, averaged_mfcc, delta_TCA, delta_delta_TCA))
    
    return combined_features



def compute_E_mfcc_TCA_with_basic_mfcc_and_basic_mfcc_and_mfcc_TCA_derivatives(audio, sr=25000, n_mfcc=12, n_gfb=40, context_size=1):
    
    
    basic_mfcc = compute_basic_mfcc(audio, sr, n_mfcc, n_gfb)
    
    # Compute the enhanced MFCC with TCA
    averaged_mfcc = compute_E_mfcc_TCA(audio, sr, n_mfcc, n_gfb, context_size)
    
    # Compute the first (delta) and second (delta-delta) derivatives of basic MFCC using librosa
    delta_basic_mfcc = librosa.feature.delta(basic_mfcc,width=5)
    delta_delta_basic_mfcc = librosa.feature.delta(delta_basic_mfcc, order=2, width=5)
    
    # Compute the first (delta) and second (delta-delta) derivatives of TCA MFCC using librosa
    delta_TCA = librosa.feature.delta(averaged_mfcc , width=5)
    delta_delta_TCA= librosa.feature.delta(averaged_mfcc, order=2, width=5)

    
    # Stack the basic MFCC, the TCA MFCC, and their derivatives vertically
    combined_features = np.vstack((basic_mfcc, averaged_mfcc, delta_basic_mfcc, delta_delta_basic_mfcc, delta_TCA, delta_delta_TCA))
    
    return combined_features





def compute_num_frames(y, sr, frame_duration=0.025, frame_shift=0.01):
    hop_length = int(frame_shift * sr)
    return len(librosa.frames_to_time(np.arange(len(y) // hop_length), sr=sr, hop_length=hop_length))+1



def extract_and_save_features(dataset_folder, save_folder, feature_fn):
    np.random.seed(0)  # For reproducibility
    
    # Ensure save_folder exists
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    actor_folders = [f for f in os.listdir(dataset_folder) if "s" in f]
    
    all_features = []
    all_labels = []

    for actor in tqdm(actor_folders):
        actor_path = os.path.join(dataset_folder, actor)
        
        if not os.path.exists(actor_path):
            print(f"Folder {actor_path} does not exist. Skipping...")
            continue
        
        actor_files = [f for f in os.listdir(actor_path) if f.endswith(".wav")]
        
        for f in actor_files:
            audio_path = os.path.join(actor_path, f)
            audio, sr = librosa.load(audio_path,sr=None)
            
            try:
                features = feature_fn(audio, sr)
            except (librosa.util.exceptions.ParameterError, AttributeError, IndexError) as e:
                # Skip the audio if there's a ParameterError from librosa (likely due to insufficient frames)
                continue
            
            for frame_features in features.T:
                all_features.append(frame_features)
                all_labels.append(actor)

    # Step 2: Split into train, test, and validation sets
    features_train, features_temp, labels_train, labels_temp = train_test_split(
        all_features, all_labels, test_size=0.3, random_state=42
    )

    features_test, features_val, labels_test, labels_val = train_test_split(
        features_temp, labels_temp, test_size=1/3, random_state=42
    )

    # Step 3: Write to CSV files
    with open(os.path.join(save_folder, f'features_train_{feature_fn.__name__}.csv'), 'w', newline='') as f_train, open(os.path.join(save_folder, f'labels_train_{feature_fn.__name__}.csv'), 'w', newline='') as l_train:
        features_writer = csv.writer(f_train)
        labels_writer = csv.writer(l_train)
        for feature, label in zip(features_train, labels_train):
            features_writer.writerow(feature)
            labels_writer.writerow([label])

    with open(os.path.join(save_folder, f'features_test_{feature_fn.__name__}.csv'), 'w', newline='') as f_test, open(os.path.join(save_folder, f'labels_test_{feature_fn.__name__}.csv'), 'w', newline='') as l_test:
        features_writer = csv.writer(f_test)
        labels_writer = csv.writer(l_test)
        for feature, label in zip(features_test, labels_test):
            features_writer.writerow(feature)
            labels_writer.writerow([label])

    with open(os.path.join(save_folder, f'features_val_{feature_fn.__name__}.csv'), 'w', newline='') as f_val, open(os.path.join(save_folder, f'labels_val_{feature_fn.__name__}.csv'), 'w', newline='') as l_val:
        features_writer = csv.writer(f_val)
        labels_writer = csv.writer(l_val)
        for feature, label in zip(features_val, labels_val):
            features_writer.writerow(feature)
            labels_writer.writerow([label])




def extract_and_save_features_seq(dataset_folder, save_folder, feature_fn, sequence_length=3):
    np.random.seed(0)  # For reproducibility
    
    # Ensure save_folder exists
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    actor_folders = [f for f in os.listdir(dataset_folder) if "s" in f]
    
    all_features = []
    all_labels = []

    for actor in tqdm(actor_folders):
        actor_path = os.path.join(dataset_folder, actor)
        
        if not os.path.exists(actor_path):
            print(f"Folder {actor_path} does not exist. Skipping...")
            continue
        
        actor_files = [f for f in os.listdir(actor_path) if f.endswith(".wav")]
        
        for f in actor_files:
            audio_path = os.path.join(actor_path, f)
            audio, sr = librosa.load(audio_path, sr=None)

            try:
                features = feature_fn(audio, sr).T  # Transposing to iterate over frames
            except (librosa.util.exceptions.ParameterError, AttributeError, IndexError) as e:

                # Skip the audio if there's a ParameterError from librosa (likely due to insufficient frames)
                continue

            # Create sequences
            for i in range(0, len(features) - sequence_length + 1):
                all_features.append(features[i:i+sequence_length])
                all_labels.append(actor)
            
            # Handle padding for the last few frames
            if len(features) % sequence_length != 0:
                remaining = len(features) % sequence_length
                pad_length = sequence_length - remaining
                padding = np.zeros((pad_length, features.shape[1]))
                sequence = np.vstack((features[-remaining:], padding))
                all_features.append(sequence)
                all_labels.append(actor)

    # Step 2: Split into train, test, and validation sets
    features_train, features_temp, labels_train, labels_temp = train_test_split(
        all_features, all_labels, test_size=0.3, random_state=42
    )

    features_test, features_val, labels_test, labels_val = train_test_split(
        features_temp, labels_temp, test_size=1/3, random_state=42
    )

    # Step 3: Write sequences to CSV files
    def save_to_csv(filepath, data):
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            for row in data:
                # Flatten the sequence into a single row
                flat_list = [item for sublist in row for item in sublist]
                writer.writerow(flat_list)

    save_to_csv(os.path.join(save_folder, f'features_train_{feature_fn.__name__}.csv'), features_train)
    save_to_csv(os.path.join(save_folder, f'features_test_{feature_fn.__name__}.csv'), features_test)
    save_to_csv(os.path.join(save_folder, f'features_val_{feature_fn.__name__}.csv'), features_val)

    # Save labels as before
    with open(os.path.join(save_folder, f'labels_train_{feature_fn.__name__}.csv'), 'w', newline='') as l_train:
        writer = csv.writer(l_train)
        writer.writerows([[label] for label in labels_train])

    with open(os.path.join(save_folder, f'labels_test_{feature_fn.__name__}.csv'), 'w', newline='') as l_test:
        writer = csv.writer(l_test)
        writer.writerows([[label] for label in labels_test])

    with open(os.path.join(save_folder, f'labels_val_{feature_fn.__name__}.csv'), 'w', newline='') as l_val:
        writer = csv.writer(l_val)
        writer.writerows([[label] for label in labels_val])



if __name__ == '__main__':
    dataset_folder = "/PATH/Grid_clean"
    save_folder = "/PATH_to_save_features_for_1DCNN_model/Grid_clean"
    save_folder_for_LSTM = "/PATH_to_save_features_for_LSTM_model/Grid_clean"
    feature_fns = [
                   compute_basic_mfcc,
                   compute_basic_mfcc_with_derivatives,
                   compute_E_mfcc_TCA,
                   compute_E_mfcc_TCA_with_TCA_derivatives,
                   compute_E_mfcc_TCA_and_basic_mfcc,
                   compute_E_mfcc_TCA_and_basic_mfcc_with_basic_mfcc_derivatives,
                   compute_E_mfcc_TCA_with_basic_mfcc_and_mfcc_TCA_derivatives,
                   compute_E_mfcc_TCA_with_basic_mfcc_and_basic_mfcc_and_mfcc_TCA_derivatives
                  ]    
    
    for feature_fn in feature_fns:
        
        print(f"Extracting features using {feature_fn.__name__}...")
        extract_and_save_features(dataset_folder, save_folder, feature_fn)
        extract_and_save_features_seq(dataset_folder, save_folder_for_LSTM, feature_fn)
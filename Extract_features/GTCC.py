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
sys.path.append(folder_path)
from spafe.features import gfcc




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
    
    
    

def compute_basic_gfcc_with_derivatives(audio, sr=25000, n_gfcc=12, n_gfb=40):
    gfccs = compute_basic_gfcc(audio, sr, n_gfcc, n_gfb)
    gfcc_delta = librosa.feature.delta(gfccs, width=5)
    gfcc_delta2 = librosa.feature.delta(gfccs, order=2, width=5)
    return np.vstack([gfccs, gfcc_delta, gfcc_delta2])





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
    
def compute_E_gfcc_TCA(audio, sr=25000, n_gfcc=12, n_gfb=40, context_size=1):



    basic_gfcc = compute_basic_gfcc(audio, sr, n_gfcc, n_gfb)

    # Temporal averaging
    averaged_gfcc = temporal_context_average(basic_gfcc, context=context_size)

    return averaged_gfcc  # or return combined_features if combined


def compute_E_gfcc_TCA_with_TCA_derivatives(audio, sr=25000, n_gfcc=12, n_gfb=40, context_size=1):

    basic_gfcc = compute_basic_gfcc(audio, sr, n_gfcc, n_gfb)

    # Temporal averaging
    averaged_gfcc = temporal_context_average(basic_gfcc, context=context_size)

    # Calculate the first and second derivatives of the TCA GFCCs
    delta = librosa.feature.delta(averaged_gfcc, width=5)
    delta_delta = librosa.feature.delta(averaged_gfcc, order=2, width=5)
    
    # Stack the TCA GFCCs, its first and second derivatives
    combined_features = np.vstack((averaged_gfcc, delta, delta_delta))
    
    return combined_features


def compute_E_gfcc_TCA_and_basic_gfcc(audio, sr=25000, n_gfcc=12, n_gfb=40, context_size=1):
    basic_gfcc = compute_basic_gfcc(audio, sr, n_gfcc, n_gfb)
    averaged_gfcc = compute_E_gfcc_TCA(audio, sr, n_gfcc, n_gfb, context_size)
    
    # Stack the basic GFCC and the TCA GFCC features vertically
    combined_features = np.vstack((basic_gfcc, averaged_gfcc))


    return combined_features



def compute_E_gfcc_TCA_and_basic_gfcc_with_basic_gfcc_derivatives(audio, sr=25000, n_gfcc=12, n_gfb=40, context_size=1):
    
    basic_gfcc = compute_basic_gfcc(audio, sr, n_gfcc, n_gfb)

    # Compute the enhanced GFCC with TCA
    averaged_gfcc = compute_E_gfcc_TCA(audio, sr, n_gfcc, n_gfb, context_size)

    # Compute the first (delta) and second (delta-delta) derivatives of TCA GFCC using librosa
    delta_basic_GFCC = librosa.feature.delta(averaged_gfcc,width=5)

    delta_delta_basic_GFCC= librosa.feature.delta(averaged_gfcc, order=2, width=5)
    
    # Stack the basic GFCC, the TCA GFCC, and its derivatives vertically
    combined_features = np.vstack((basic_gfcc, averaged_gfcc, delta_basic_GFCC, delta_delta_basic_GFCC))
    

    return combined_features




def compute_E_gfcc_TCA_with_basic_gfcc_and_gfcc_TCA_derivatives(audio, sr=25000, n_gfcc=12, n_gfb=40, context_size=1):


    basic_gfcc = compute_basic_gfcc(audio, sr, n_gfcc, n_gfb)
    
    # Compute the enhanced GFCC with TCA
    averaged_gfcc = compute_E_gfcc_TCA(audio, sr, n_gfcc, n_gfb, context_size)
    

    # Compute the first (delta) and second (delta-delta) derivatives of TCA GFCC using librosa
    delta_TCA = librosa.feature.delta(averaged_gfcc,width=5)

    delta_delta_TCA= librosa.feature.delta(averaged_gfcc, order=2, width=5)


    # Stack the basic GFCC, the TCA GFCC, and its derivatives vertically
    combined_features = np.vstack((basic_gfcc, averaged_gfcc, delta_TCA, delta_delta_TCA))
    
    return combined_features



def compute_E_gfcc_TCA_with_basic_gfcc_and_basic_gfcc_and_gfcc_TCA_derivatives(audio, sr=25000, n_gfcc=12, n_gfb=40, context_size=1):
    
    
    basic_gfcc = compute_basic_gfcc(audio, sr, n_gfcc, n_gfb)
    
    # Compute the enhanced GFCC with TCA
    averaged_gfcc = compute_E_gfcc_TCA(audio, sr, n_gfcc, n_gfb, context_size)
    
    # Compute the first (delta) and second (delta-delta) derivatives of basic GFCC using librosa
    delta_basic_gfcc = librosa.feature.delta(basic_gfcc,width=5)
    delta_delta_basic_gfcc = librosa.feature.delta(delta_basic_gfcc,order=2, width=5)
    
    # Compute the first (delta) and second (delta-delta) derivatives of TCA GFCC using librosa
    delta_TCA = librosa.feature.delta(averaged_gfcc,width=5)
    delta_delta_TCA= librosa.feature.delta(averaged_gfcc, order=2, width=5)

    
    # Stack the basic GFCC, the TCA GFCC, and their derivatives vertically
    combined_features = np.vstack((basic_gfcc, averaged_gfcc, delta_basic_gfcc, delta_delta_basic_gfcc, delta_TCA, delta_delta_TCA))
    
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
            except (librosa.util.exceptions.ParameterError, AttributeError,IndexError) as e:
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
                   compute_basic_gfcc,
                   compute_basic_gfcc_with_derivatives,
                   compute_E_gfcc_TCA,
                   compute_E_gfcc_TCA_with_TCA_derivatives,
                   compute_E_gfcc_TCA_and_basic_gfcc,
                   compute_E_gfcc_TCA_and_basic_gfcc_with_basic_gfcc_derivatives,
                   compute_E_gfcc_TCA_with_basic_gfcc_and_gfcc_TCA_derivatives,
                   compute_E_gfcc_TCA_with_basic_gfcc_and_basic_gfcc_and_gfcc_TCA_derivatives
                  ]    
    
    for feature_fn in feature_fns:
        
        print(f"Extracting features using {feature_fn.__name__}...")
        extract_and_save_features(dataset_folder, save_folder, feature_fn)
        extract_and_save_features_seq(dataset_folder, save_folder_for_LSTM, feature_fn)
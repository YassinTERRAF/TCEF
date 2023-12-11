import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import os
import csv
from tensorflow.keras.layers import LeakyReLU, LayerNormalization

from tensorflow.keras.regularizers import l2

class SimpleRNNModel:
    def __init__(self, input_shape, output_dim):
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.model = self.build_model()

    def build_model(self):
        input_layer = Input(shape=self.input_shape)

        lstm = LSTM(32, return_sequences=True)(input_layer)
        lstm = LayerNormalization()(lstm)
        lstm = LSTM(32, return_sequences=False)(lstm)
        lstm = LayerNormalization()(lstm)

        # Adding L2 regularization to the Dense layer
        dense = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(lstm)
        output = Dense(self.output_dim, activation='softmax')(dense)

        model = Model(inputs=input_layer, outputs=output)
        return model

    def load_weights(self, filepath):
        self.model.load_weights(filepath)


class F1ScoreCallback(Callback):
    def __init__(self, validation_data):
        super(F1ScoreCallback, self).__init__()
        self.validation_data = validation_data

    def on_epoch_end(self, epoch, logs={}):
        val_predict = np.argmax(self.model.predict(self.validation_data[0]), axis=1)
        val_true = np.argmax(self.validation_data[1], axis=1)
        val_f1 = f1_score(val_true, val_predict, average='weighted')
        logs['val_f1_score'] = val_f1
        print(f" — val_f1_score: {val_f1:.4f}")



def load_data_seq(features_file, labels_file):
    # We need to update the data loading function to reshape data into sequences
    features, labels = [], []
    with open(features_file, 'r') as f, open(labels_file, 'r') as l:
        csv_reader_features = csv.reader(f)
        csv_reader_labels = csv.reader(l)
        for row_f, row_l in zip(csv_reader_features, csv_reader_labels):
            sequence_length = int(len(row_f) / 12)  # Assuming 13 for GFCC features, adjust if different
            features.append(np.array(row_f, dtype=float).reshape(sequence_length, 12))
            labels.append(row_l[0])
    return np.array(features), np.array(labels)

def initialize_seq_model(input_shape, num_classes):
    model = SimpleRNNModel(input_shape, num_classes)
    optimizer = Adam(learning_rate=0.001)  
    model.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model
    

def main(feature_technique):
    save_folder = "Path_to_features_to_train/dataset_name"
    # File paths
    train_feature_file = os.path.join(save_folder, f'features_train_{feature_technique}.csv')
    train_label_file = os.path.join(save_folder, f'labels_train_{feature_technique}.csv')
    val_feature_file = os.path.join(save_folder, f'features_val_{feature_technique}.csv')
    val_label_file = os.path.join(save_folder, f'labels_val_{feature_technique}.csv')
    test_feature_file = os.path.join(save_folder, f'features_test_{feature_technique}.csv')
    test_label_file = os.path.join(save_folder, f'labels_test_{feature_technique}.csv')

    # Load data using the sequence function
    train_features, train_labels = load_data_seq(train_feature_file, train_label_file)
    val_features, val_labels = load_data_seq(val_feature_file, val_label_file)
    test_features, test_labels = load_data_seq(test_feature_file, test_label_file)


    # Label encoding
    label_encoder = LabelEncoder()
    train_labels = label_encoder.fit_transform(train_labels)
    val_labels = label_encoder.transform(val_labels)
    test_labels_int = label_encoder.transform(test_labels)

    num_classes = len(np.unique(train_labels))
    train_labels = to_categorical(train_labels, num_classes=num_classes)
    val_labels = to_categorical(val_labels, num_classes=num_classes)

    accuracies, precisions, recalls, f1s = [], [], [], []
    log_file_path = os.path.join(log_dir, f'logs_{feature_technique}.txt')
    
    with open(log_file_path, 'a') as log_file:
        for run in range(2):   # Change this to 10 if you want 10 runs
            print(f"Run {run+1}/5")  # Change this to 10 if you want 10 runs
            
            seq_model = initialize_seq_model(train_features.shape[1:], num_classes)
            f1_callback = F1ScoreCallback(validation_data=(val_features, val_labels))
            early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min', restore_best_weights=True)
            checkpoint = ModelCheckpoint("/PATH_TO_CheckPoint/"+f'best_weights_{feature_technique}_clean_lstm_1.h5', save_best_only=True, monitor='val_loss', mode='min', verbose=1)

            seq_model.model.fit(train_features, train_labels, 
                                validation_data=(val_features, val_labels), 
                                epochs=55, batch_size=32, 
                                callbacks=[checkpoint, f1_callback, early_stopping])
            seq_model.load_weights("/PATH_TO_CheckPoint/"+f'best_weights_{feature_technique}_clean_lstm_1.h5')

            predictions = seq_model.model.predict(test_features)
            predictions_int = np.argmax(predictions, axis=1)

            accuracy = accuracy_score(test_labels_int, predictions_int)
            precision = precision_score(test_labels_int, predictions_int, average='weighted')
            recall = recall_score(test_labels_int, predictions_int, average='weighted')
            f1 = f1_score(test_labels_int, predictions_int, average='weighted')

            accuracies.append(accuracy)
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

            log_file.write(f"Run {run+1}\nAccuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1 Score: {f1}\n\n")
            print(f"Run {run+1}\nAccuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1 Score: {f1}\n\n")

        avg_accuracy = np.mean(accuracies)
        avg_precision = np.mean(precisions)
        avg_recall = np.mean(recalls)
        avg_f1 = np.mean(f1s)

        print(f"\nAverage Results over 5 runs:\nAccuracy: {avg_accuracy:.4f}\nPrecision: {avg_precision:.4f}\nRecall: {avg_recall:.4f}\nF1 Score: {avg_f1:.4f}\n")

if __name__ == "__main__":
    main('Feature_Extraction_Technique')
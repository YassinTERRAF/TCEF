import numpy as np
import csv
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Activation, LayerNormalization, MaxPooling1D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import Callback, EarlyStopping
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.layers import Dropout, ReLU, BatchNormalization

from tensorflow.keras.regularizers import l2

class CNNModel:
    def __init__(self, input_shape, output_dim):
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()

        # Convolutional layer with Dropout and Batch Normalization
        model.add(Conv1D(64, 3, padding='same', input_shape=self.input_shape))
        model.add(ReLU())

        # Additional convolutional layers with regularization
        for _ in range(2):
            model.add(Conv1D(64, 3, padding='same'))
            model.add(BatchNormalization())
            model.add(ReLU())
            model.add(MaxPooling1D(pool_size=2))

        # Flatten and dense layers with L2 regularization
        model.add(Flatten())
        model.add(Dense(64, activation='relu',kernel_regularizer=l2(0.01)))  # Adding L2 regularization
        model.add(Dense(self.output_dim, activation='softmax'))

        return model






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

def load_data(features_file, labels_file):
    features, labels = [], []
    with open(features_file, 'r') as f, open(labels_file, 'r') as l:
        csv_reader_features = csv.reader(f)
        csv_reader_labels = csv.reader(l)
        for row_f, row_l in zip(csv_reader_features, csv_reader_labels):
            features.append([float(i) for i in row_f])
            labels.append(row_l[0])
    return np.array(features), np.array(labels)



def initialize_model(input_shape, num_classes):
    model = CNNModel(input_shape, num_classes)
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

    # Load data
    train_features, train_labels = load_data(train_feature_file, train_label_file)
    val_features, val_labels = load_data(val_feature_file, val_label_file)
    test_features, test_labels = load_data(test_feature_file, test_label_file)

    # Label encoding
    label_encoder = LabelEncoder()
    train_labels = label_encoder.fit_transform(train_labels)
    val_labels = label_encoder.transform(val_labels)
    test_labels_int = label_encoder.transform(test_labels)

    num_classes = len(np.unique(train_labels))
    train_labels = to_categorical(train_labels, num_classes=num_classes)
    val_labels = to_categorical(val_labels, num_classes=num_classes)

    train_features = train_features[:, :, np.newaxis]
    val_features = val_features[:, :, np.newaxis]
    test_features = test_features[:, :, np.newaxis]

    accuracies, precisions, recalls, f1s = [], [], [], []

    log_file_path = os.path.join(log_dir, f'logs_{feature_technique}.txt')
    
    with open(log_file_path, 'a') as log_file:
        for run in range(2):
            print(f"Run {run+1}/5")

            cnn_model = initialize_model(train_features.shape[1:], num_classes)

            f1_callback = F1ScoreCallback(validation_data=(val_features, val_labels))
            early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min', restore_best_weights=True)
            checkpoint = ModelCheckpoint("/CheckPoint_PATH/"+f'best_weights_{feature_technique}_clean_1dcnn_1.h5', 
                                         save_best_only=True, 
                                         monitor='val_loss', 
                                         mode='min', 
                                         verbose=1)

            cnn_model.model.fit(train_features, train_labels, 
                                validation_data=(val_features, val_labels), 
                                epochs=100, batch_size=32, 
                                callbacks=[checkpoint, f1_callback, early_stopping])

            cnn_model.model.load_weights("/CheckPoint_PATH/"+f'best_weights_{feature_technique}_clean_1dcnn_1.h5')

            predictions = cnn_model.model.predict(test_features)
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
    main('Feature_extraction_Technique')
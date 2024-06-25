# Temporal Context Enhanced Features - TCEF 
## Robust Feature Extraction Using Temporal Context Averaging for Speaker Identification in Diverse Acoustic Environments
## This is the official Tensorflow implementation of TCEF.

## Citation
@article{terraf2024robust,
  title={Robust Feature Extraction Using Temporal Context Averaging for Speaker Identification in Diverse Acoustic Environments},
  author={Terraf, Yassin and Iraqi, Youssef},
  journal={IEEE Access},
  year={2024},
  publisher={IEEE}
}
## Overview
TCEF is an innovative approach to enhance audio feature representation by incorporating temporal context. This repository contains the implementation of TCEF and its application on two datasets: GRID and Ravdess. It also includes tools for adding noise and reverberation to these datasets to create augmented versions.

## Structure
The repository is organized into the following main directories:

- **Datasets**: This folder contains samples of audios in different datasets used in TCEF.
- To download RAVDESS-NR dataset: https://zenodo.org/records/10528908 //https://zenodo.org/records/3625687
- To download GRID-NR dataset: https://zenodo.org/records/10528941
  


- **Extract_features**: Contains scripts for extracting both conventional and TCEF features using three different feature extraction techniques: MFCC, GTCC, and PNCC.

## Feature Extraction
Feature extraction can be performed using the following scripts:
- `MFCC.py`: For extracting MFCC features.
- `GTCC.py`: For extracting GTCC features.
- `PNCC.py`: For extracting PNCC features.

You can adjust the context window size for temporal context averaging by modifying the `context_size` variable in each script. The `sequence_length` variable can be adjusted to change the number of frames considered for sequence length in the LSTM network.

## Model Training
Two Python scripts are provided for training models using both conventional and TCEF features:
- `1Dcnn.py`: Used to train TCEF features and conventional features using a 1D CNN for frame-level analysis.
- `lstm.py`: Used to train TCEF features and conventional features using an LSTM for sequence-level analysis.

## Generating Plots
All generated plots presented in the paper can be created through the scripts available in the `plots` folder.

## Usage
To use this repository for feature extraction and model training, follow these steps:

1. Clone the repository to your local machine.
2. Navigate to the desired script for feature extraction or model training.
3. Adjust the parameters like `context_size` and `sequence_length` as per your requirement.
4. Run the script to perform feature extraction or to train the models.


## License
This project is released under the Apache 2.0 license.
## Contact
Email: yassin.terraf@um6p.ma

Linkedin: https://www.linkedin.com/in/yassin-terraf-206597151/

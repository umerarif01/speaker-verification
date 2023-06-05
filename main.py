# pip install numpy scipy librosa scikit-learn
import os
import numpy as np
import scipy.io.wavfile as wav
import librosa
from sklearn.mixture import GaussianMixture

def extract_mfcc(audio_signal, sample_rate):
    # Perform MFCC feature extraction using librosa
    mfcc_features = librosa.feature.mfcc(y=audio_signal, sr=sample_rate)
    return mfcc_features.T

def train_gmm(features, num_components):
    # Train a Gaussian Mixture Model (GMM) using sklearn's GaussianMixture
    gmm = GaussianMixture(n_components=num_components)
    gmm.fit(features)
    return gmm

def match_gmm(features, gmm):
    # Calculate log-likelihood scores for each feature vector
    log_likelihoods = gmm.score_samples(features)
    return log_likelihoods

def load_audio_signal(file_path):
    # Load audio file using scipy's wavfile module
    sample_rate, audio_signal = wav.read(file_path)
    audio_signal = audio_signal.astype(np.float32) / 32767.0  # Convert to floating-point in the range [-1, 1]
    return audio_signal, sample_rate

def load_audio_files(folder_path):
    audio_files = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".wav"):
            audio_files.append(os.path.join(folder_path, file_name))
    return audio_files

def train_speaker_verification_model(folder_path, num_components):
    # Load audio files from the specified folder
    audio_files = load_audio_files(folder_path)

    # Initialize an empty array to store reference features and file paths
    reference_features = []
    reference_file_paths = []

    # Extract features from each audio file
    for file_path in audio_files:
        print(file_path)
        audio_signal, sample_rate = load_audio_signal(file_path)
        features = extract_mfcc(audio_signal, sample_rate)
        reference_features.append(features)
        reference_file_paths.append(file_path)

    # Concatenate all reference features
    reference_features = np.concatenate(reference_features)

    # Normalize the reference features
    reference_features = (reference_features - np.mean(reference_features)) / np.std(reference_features)

    # Train a Gaussian Mixture Model (GMM) using the reference features
    gmm = train_gmm(reference_features, num_components)

    return gmm, reference_file_paths

def verify_speaker(gmm, reference_file_paths, audio_file_path, threshold):
    # Load the verification audio file
    verification_signal, verification_sample_rate = load_audio_signal(audio_file_path)

    # Extract features from the verification audio
    verification_features = extract_mfcc(verification_signal, verification_sample_rate)

    # Normalize the verification features
    verification_features = (verification_features - np.mean(verification_features)) / np.std(verification_features)

    # Match the verification features with the trained GMM
    verification_scores = match_gmm(verification_features, gmm)

    # Find the maximum log-likelihood score from the verification scores
    max_verification_score = np.max(verification_scores)

    # Find the maximum log-likelihood score from the reference audio files
    max_reference_scores = []
    for file_path in reference_file_paths:
        audio_signal, sample_rate = load_audio_signal(file_path)
        reference_features = extract_mfcc(audio_signal, sample_rate)
        reference_scores = match_gmm(reference_features, gmm)
        max_reference_scores.append(np.max(reference_scores))

    # Compare the maximum verification score with the maximum reference scores
    if max_verification_score >= np.max(max_reference_scores) - threshold:
        print("Speaker verified.")
    else:
        print("Speaker not verified.")

def main():
    # Set the path to the reference audio folder
    reference_folder_path = "f0001"

    # Set the path to the verification audio file
    verification_audio_path = "umer2.wav"

    # Set the number of components for the GMM
    num_components = 16

    # Set the threshold for speaker verification
    threshold = 5.0

    # Train the speaker verification model
    gmm, reference_file_paths = train_speaker_verification_model(reference_folder_path, num_components)

    # Verify the speaker using the verification audio file
    verify_speaker(gmm, reference_file_paths, verification_audio_path, threshold)

if __name__ == "__main__":
    main()

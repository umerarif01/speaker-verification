# Speaker Verification Project

This project implements a speaker verification system using MFCC (Mel-Frequency Cepstral Coefficients) features and an SVM (Support Vector Machine) classifier. The system enables the authentication of claimed speaker identities by comparing voice samples with enrolled templates.

## Features

- Extraction of MFCC features from audio files
- Training an SVM model with a linear kernel
- Prediction of speakers based on trained model
- Evaluation of model accuracy using classification metrics
- Basic voice recording and speaker prediction functionality

## Usage

### 1. Training the Model

- Place the voice samples of known speakers in the `voice-samples-2` directory.
- Run the `train_model.py` script to extract MFCC features, train the SVM model, and save the trained model as `SVC_model_34ssamples.joblib`.

### 2. Predicting the Speaker

- Place the audio files to be predicted in the `test2` directory.
- Run the `predict_speaker.py` script to predict the speaker for each audio file and view the results.

### Additional Considerations

- Ensure the required dependencies (`librosa`, `numpy`, `sklearn`, `sounddevice`, `soundfile`, `joblib`) are installed.
- Adjust the parameters in the scripts, such as the number of MFCC coefficients or SVM kernel, as per your project requirements.
- Experiment with different datasets, feature extraction techniques, and classification models to enhance the system's accuracy and performance.
- Collaborate, share, and contribute to the project by submitting pull requests and suggestions.

## Future Enhancements

- Include functionality for verifying unknown speakers.
- Explore advanced techniques such as deep learning models for speaker verification.
- Provide an option for user enrollment and management.
- Improve robustness by handling noisy environments and speaker variations.

## License

This project is licensed under the [MIT License](LICENSE).

Feel free to clone, modify, and use the code for academic, research, or personal purposes.


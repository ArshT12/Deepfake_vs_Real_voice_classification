# Deepfake Voice Detection

A machine learning project for distinguishing between authentic human speech and AI-generated deepfake audio using acoustic feature analysis and XGBoost classification.

## Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Dataset Information](#-dataset-information)
- [Methodology](#-methodology)
- [Results](#-results)
- [Hugging Face API](#-hugging-face-api)
- [Mobile Application](#-mobile-application)
- [Installation & Setup](#-installation--setup)
- [Running the Notebooks](#-running-the-notebooks)
- [Future Development](#-future-development)
- [Citation](#-citation)
- [License](#-license)
- [Contact](#-contact)

## Overview

The proliferation of AI-generated synthetic speech presents significant challenges for authenticating audio content. This project addresses this problem by using machine learning to analyze acoustic features for distinguishing between real human voices and deepfakes.

Our approach extracts 34 acoustic features from audio samples and employs an optimized XGBoost classifier to detect deepfakes with high accuracy. The model achieves 98.9% accuracy on test data and is deployed as an accessible API via Hugging Face Spaces.

## Key Features

- **Acoustic Feature Analysis**: Extraction and analysis of 34 acoustic features from audio samples
- **XGBoost Classification**: Optimized gradient boosting model for deepfake detection
- **High Performance**: 98.86% accuracy with excellent precision-recall balance
- **API Deployment**: Publicly accessible Hugging Face Spaces API
- **Cross-Platform Application**: Companion web/iOS application for user-friendly interaction
- **Comprehensive Jupyter Notebooks**: Detailed exploration, analysis, and model development

## Dataset Information

The model is trained on a multi-source, balanced dataset comprising:

1. **MLAAD (Multi-Language Audio Anti-Spoofing Dataset)**
   - Multilingual benchmark with diverse TTS artifacts
   - 35+ languages and 59-91 distinct TTS systems
   - Gold standard for audio spoofing research

2. **FakeAVCeleb (Audio variant)**
   - Fine-grained multimodal deepfake audio
   - Extracted from deepfake videos
   - Captures lip-sync optimized speech artifacts

3. **YouTube Real Speech**
   - Long-form genuine recordings of natural speech
   - Diverse acoustic environments and speaker demographics
   - Natural prosody and spontaneous speech elements

4. **YouTube Gaming Fake Commentary**
   - Synthetic gaming commentary with background noise
   - Challenging real-world conditions with ambient noise
   - Tests detection under noisy, expressive conditions

The final dataset contains 31,403 real and 19,596 fake audio samples (5-10s each) after balancing and preprocessing.

## Methodology

### Feature Extraction

We extract 34 acoustic features from each audio sample:

1. **Mel-Frequency Cepstral Coefficients (MFCCs)**
   - 13 mean coefficients (mfcc_mean_1 through mfcc_mean_13)
   - 13 standard deviation coefficients (mfcc_std_1 through mfcc_std_13)
   - Captures vocal tract configuration and spectral envelope

2. **Zero-Crossing Rate (ZCR)**
   - Frequency with which the signal changes from positive to negative
   - Effective at detecting high-frequency vocoder artifacts

3. **Root Mean Square (RMS) Energy**
   - Overall amplitude/loudness measure
   - Reveals unnatural energy patterns in synthetic speech

4. **Spectral Features**
   - Centroid: "Brightness" of the sound
   - Bandwidth: Spread of frequencies around the centroid
   - Rolloff: Frequency below which 85% of energy is contained
   - Contrast: Difference between peaks and valleys in the spectrum

5. **Pitch Statistics**
   - Mean pitch (fundamental frequency)
   - Pitch standard deviation (intonation variability)

### Model Selection

After evaluating multiple machine learning approaches:

| Model | Accuracy | Precision | Recall | F1 Score |
|----------------------|----------|-----------|--------|----------|
| Logistic Regression | 0.8906 | 0.8996 | 0.9256 | 0.9124 |
| Random Forest | 0.9799 | 0.9744 | 0.9935 | 0.9838 |
| **XGBoost** | **0.9871** | **0.9861** | **0.9930** | **0.9895** |
| SVM (linear kernel) | 0.8954 | 0.8973 | 0.9374 | 0.9169 |
| DNN (4 Dense layers) | 0.9889 | - | - | - |
| CNN (Conv1D) | 0.9937 | - | - | - |
| LSTM | 0.9659 | - | - | - |

We selected **XGBoost** for its excellent balance of accuracy, interpretability, and deployment efficiency.

### Hyperparameter Optimization

The final XGBoost model uses the following optimized hyperparameters:
- max_depth: 7
- learning_rate: 0.1
- n_estimators: 200

Discovered through systematic grid search and cross-validation.

## Results

### Classification Performance

| Metric | Value |
|--------|-------|
| Accuracy | 0.9886 |
| Precision | 0.9867 |
| Recall | 0.9949 |
| F1 Score | 0.9908 |

### Feature Importance Analysis

Most discriminative features for deepfake detection:

1. Zero-crossing rate (ZCR)
2. Spectral contrast
3. Spectral rolloff
4. MFCC standard deviation (coefficient 12)
5. MFCC standard deviation (coefficient 2)

### Threshold Optimization

| Threshold | Accuracy | Precision | Recall | F1 Score |
|-----------|----------|-----------|--------|----------|
| 0.50 | 0.9886 | 0.9867 | 0.9949 | 0.9908 |
| 0.60 | 0.9891 | 0.9894 | 0.9930 | 0.9912 |
| 0.70 | 0.9880 | 0.9917 | 0.9889 | 0.9903 |
| 0.80 | 0.9872 | 0.9950 | 0.9841 | 0.9895 |
| 0.90 | 0.9797 | 0.9971 | 0.9699 | 0.9833 |
| 0.95 | 0.9700 | 0.9983 | 0.9529 | 0.9751 |
| 0.99 | 0.9095 | 1.0000 | 0.8530 | 0.9207 |

## Hugging Face API

The model is deployed as a user-friendly API on Hugging Face Spaces:

- **URL**: [https://huggingface.co/spaces/ArshTandon/deepfake-detection-api](https://huggingface.co/spaces/ArshTandon/deepfake-detection-api)
- **Interface**: Gradio web UI for direct file upload and analysis
- **REST API**: Available for programmatic access

### API Implementation

```python
import gradio as gr
import joblib
import numpy as np
import librosa

MODEL_PATH = "deepfake_voice_detector.pkl"
SCALER_PATH = "feature_scaler.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

def extract_features(filepath):
    y, sr = librosa.load(filepath, sr=None)
    
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    mfccs_std = np.std(mfccs, axis=1)
    
    zcr_mean = np.mean(librosa.feature.zero_crossing_rate(y))
    rms_mean = np.mean(librosa.feature.rms(y=y))
    
    spec_cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    spec_roll = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    spec_con = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))
    
    pitches, * = librosa.piptrack(y=y, sr=sr)
    pitches = pitches[pitches > 0]
    p_mean = np.mean(pitches) if pitches.size else 0.0
    p_std = np.std(pitches) if pitches.size else 0.0
    
    feats = np.hstack([
        mfccs_mean, mfccs_std, zcr_mean, rms_mean,
        spec_cent, spec_bw, spec_roll, spec_con,
        p_mean, p_std
    ])
    
    return feats.reshape(1, -1)

def predict_deepfake(audio_path):
    feats = extract_features(audio_path)
    feats_scaled = scaler.transform(feats)
    proba = model.predict_proba(feats_scaled)[0]
    pred = model.predict(feats_scaled)[0]
    
    # training used 1=Real, 0=Fake
    is_fake = (pred == 0)
    confidence = proba[pred] * 100
    label = "🔴 Deepfake detected!" if is_fake else "🟢 Audio appears authentic."
    
    return f"{label}\nConfidence: {confidence:.2f}%"

iface = gr.Interface(
    fn=predict_deepfake,
    inputs=gr.Audio(type="filepath"),
    outputs="text",
    title="Audio Deepfake Detector",
    description="Upload any WAV/MP3 file and this will tell you if it's a deepfake – with confidence."
)

if __name__ == "__main__":
    iface.launch()
```

### API Dependencies

```
gradio>=3.50.2
numpy>=1.24.3
librosa>=0.10.1
scikit-learn>=1.3.0
xgboost>=1.7.6
pandas>=1.5.3
tqdm>=4.65.0
```

## Mobile Application

A cross-platform mobile application called "Audio Truth Teller" provides a user-friendly interface:

- **Technologies**: React, TypeScript, Capacitor, Tailwind CSS
- **Platforms**: Web Progressive Web App (PWA) and iOS native app
- **Features**:
  - Real-time recording
  - Audio file upload
  - Analysis results with confidence scoring
  - Settings configuration

## Installation & Setup

### Prerequisites

- Python 3.8+
- Jupyter Notebook
- Git

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/ArshT12/Deepfake_vs_Real_voice_classification.git
cd Deepfake_vs_Real_voice_classification

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install numpy pandas librosa scikit-learn xgboost jupyter matplotlib seaborn tqdm
```

## Running the Notebooks

### Feature Extraction

1. Navigate to the `Feature extraction` directory
2. Open `feature_extraction.ipynb` in Jupyter Notebook
3. Follow the instructions to extract features from audio files

```bash
jupyter notebook "Feature extraction/feature_extraction.ipynb"
```

### Model Training

1. Navigate to the `Model_1` directory
2. Open `model_training.ipynb` in Jupyter Notebook
3. Execute the cells to train and evaluate the models

```bash
jupyter notebook "Model_1/model_training.ipynb"
```

### Hyperparameter Tuning

```bash
jupyter notebook "Model_1/hyperparameter_tuning.ipynb"
```

## Future Development

Planned enhancements for this project:

1. **Code Restructuring**: Refactor notebook code into modular Python scripts
2. **Advanced Models**: Explore transformer-based approaches for sequence modeling
3. **Expanded Dataset**: Include newer deepfake generation techniques
4. **Real-time Analysis**: Optimize for streaming audio processing
5. **Multi-class Classification**: Distinguish between different deepfake generation methods

## Contact

Arsh Tandon - [@ArshTandon](https://github.com/ArshT12)

Project Link: [https://github.com/ArshT12/Deepfake_vs_Real_voice_classification](https://github.com/ArshT12/Deepfake_vs_Real_voice_classification)

---

<p align="center">

</p>
<p align="center">
  <i>Protecting audio authenticity in an age of synthetic media.</i>
</p>

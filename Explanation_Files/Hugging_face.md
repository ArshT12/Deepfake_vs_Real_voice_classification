# Deepfake Voice Detection API - Technical Documentation

**Version:** 1.0.0  
**Last Updated:** 2025-04-30  
**Deployment URL:** [https://huggingface.co/spaces/ArshTandon/deepfake-detection-api](https://huggingface.co/spaces/ArshTandon/deepfake-detection-api)

---

## 1. Overview

This document provides technical details for the Deepfake Voice Detection API, a machine learning service deployed on Hugging Face Spaces that analyzes audio files to determine whether they contain authentic human speech or AI-generated deepfakes. The API is built on an XGBoost model trained on acoustic features extracted from a diverse dataset of real and synthetic voice samples.

### 1.1 Key Features

- Real-time audio analysis through a Gradio web interface
- Support for common audio formats (WAV, MP3)
- Advanced acoustic feature extraction via Librosa
- Confidence scoring of classification results
- Simple REST API endpoint for programmatic access
- Lightweight deployment on Hugging Face Spaces

---

## 2. Technical Architecture

The API is deployed as a Gradio application on Hugging Face Spaces, consisting of three main components:

1. **Feature Extraction Pipeline**: Uses Librosa to process audio and extract 34 acoustic features
2. **Pre-trained Model**: XGBoost classifier trained on an extensive dataset of real and synthetic speech
3. **Web Interface**: Gradio-based UI for direct file upload and analysis

### 2.1 Architectural Diagram

```
┌────────────────┐    ┌───────────────┐    ┌──────────────┐
│                │    │               │    │              │
│  Audio Input   │───▶│  Feature      │───▶│  XGBoost     │
│  (WAV/MP3)     │    │  Extraction   │    │  Classifier  │
│                │    │               │    │              │
└────────────────┘    └───────────────┘    └──────┬───────┘
                                                  │
                                                  ▼
┌────────────────┐    ┌───────────────┐    ┌──────────────┐
│                │    │               │    │              │
│  Gradio UI     │◀───│  Confidence   │◀───│  Prediction  │
│  Response      │    │  Calculation  │    │  Results     │
│                │    │               │    │              │
└────────────────┘    └───────────────┘    └──────────────┘
```

---

## 3. Backend Implementation

### 3.1 Core Files

The API consists of the following key files:

- **app.py**: Main application entry point and Gradio interface
- **deepfake_voice_detector.pkl**: XGBoost model serialized with joblib
- **feature_scaler.pkl**: StandardScaler for feature normalization
- **requirements.txt**: Python dependencies

### 3.2 Full Application Code

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

### 3.3 Dependencies

The API requires the following Python packages:

```
gradio>=3.50.2
numpy>=1.24.3
librosa>=0.10.1
scikit-learn>=1.3.0
xgboost>=1.7.6
pandas>=1.5.3
tqdm>=4.65.0
```

---

## 4. Feature Extraction Process

The API extracts 34 acoustic features from each audio file, creating a comprehensive acoustic fingerprint for analysis:

### 4.1 Feature Categories

1. **Mel-Frequency Cepstral Coefficients (MFCCs)**
   - 13 mean coefficients (`mfcc_mean_1` through `mfcc_mean_13`)
   - 13 standard deviation coefficients (`mfcc_std_1` through `mfcc_std_13`)
   - Captures spectral envelope and vocal tract configuration

2. **Zero-Crossing Rate (ZCR)**
   - Single feature representing frequency content
   - Higher in synthetic speech due to vocoder artifacts

3. **Root Mean Square (RMS) Energy**
   - Overall amplitude/loudness measure
   - Captures dynamic range differences in real vs. synthetic speech

4. **Spectral Features**
   - **Centroid**: Center of mass of the spectrum ("brightness")
   - **Bandwidth**: Spread of frequencies around the centroid
   - **Rolloff**: Frequency below which 85% of energy is contained
   - **Contrast**: Difference between peaks and valleys in the spectrum

5. **Pitch Statistics**
   - **Mean**: Average fundamental frequency
   - **Standard Deviation**: Variability in pitch (intonation)

### 4.2 Feature Extraction Implementation

```python
def extract_features(filepath):
    # Load audio file
    y, sr = librosa.load(filepath, sr=None)
    
    # Extract MFCCs (13 coefficients)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)  # Mean of each coefficient
    mfccs_std = np.std(mfccs, axis=1)    # Standard deviation of each coefficient
    
    # Zero-crossing rate
    zcr_mean = np.mean(librosa.feature.zero_crossing_rate(y))
    
    # Root Mean Square Energy
    rms_mean = np.mean(librosa.feature.rms(y=y))
    
    # Spectral features
    spec_cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    spec_roll = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    spec_con = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))
    
    # Pitch statistics
    pitches, _ = librosa.piptrack(y=y, sr=sr)
    pitches = pitches[pitches > 0]  # Filter for valid pitch values
    p_mean = np.mean(pitches) if pitches.size else 0.0
    p_std = np.std(pitches) if pitches.size else 0.0
    
    # Combine all features into a single vector
    feats = np.hstack([
        mfccs_mean, mfccs_std, zcr_mean, rms_mean,
        spec_cent, spec_bw, spec_roll, spec_con,
        p_mean, p_std
    ])
    
    return feats.reshape(1, -1)  # Return as 2D array for prediction
```

---

## 5. Model Details

The deepfake detection model is an XGBoost classifier optimized for audio classification:

### 5.1 Model Specifications

- **Algorithm**: XGBoost Classifier
- **Feature Dimension**: 34 acoustic features
- **Hyperparameters**:
  - max_depth: 7
  - learning_rate: 0.1
  - n_estimators: 200
  - objective: binary:logistic
- **Performance Metrics**:
  - Accuracy: 98.86%
  - Precision: 98.67%
  - Recall: 99.49%
  - F1 Score: 99.08%

### 5.2 Feature Importance

Based on model analysis, the most important features for deepfake detection are:

1. Zero-crossing rate (zcr)
2. Spectral contrast
3. MFCC standard deviation (coefficient 12)
4. Spectral rolloff
5. MFCC standard deviation (coefficient 2)

These features are particularly effective at identifying artifacts introduced by neural vocoders and speech synthesis algorithms.

---

## 6. API Usage

### 6.1 Web Interface

The API includes a Gradio web interface for direct interaction:

1. Navigate to [https://huggingface.co/spaces/ArshTandon/deepfake-detection-api](https://huggingface.co/spaces/ArshTandon/deepfake-detection-api)
2. Upload an audio file (WAV or MP3) using the file picker
3. Click "Submit" to process the file
4. View the classification result and confidence score

### 6.2 REST API Access

For programmatic access, use the Hugging Face Spaces API endpoint:

```python
import requests

API_URL = "https://arshtandon-deepfake-detection-api.hf.space/api/predict"

def analyze_audio(file_path):
    with open(file_path, "rb") as f:
        files = {"audio": f}
        response = requests.post(API_URL, files=files)
    
    return response.json()

# Example usage
result = analyze_audio("sample.wav")
print(result)
```

### 6.3 Expected Response Format

The API returns a text string with two components:

1. Classification result: "🔴 Deepfake detected!" or "🟢 Audio appears authentic."
2. Confidence percentage: "Confidence: 95.42%"

Example response:
```
🔴 Deepfake detected!
Confidence: 95.42%
```

### 6.4 Mobile App Integration

To integrate with the Audio Truth Teller mobile application:

```typescript
// TypeScript example for Audio Truth Teller app
async function analyzeAudioFile(audioUri: string): Promise<{
  label: string;
  confidence: number;
}> {
  const formData = new FormData();
  
  // Create file object from URI
  const fileInfo = await getFileInfo(audioUri);
  formData.append('audio', {
    uri: audioUri,
    name: fileInfo.name,
    type: fileInfo.mimeType
  });
  
  try {
    const response = await fetch(
      'https://arshtandon-deepfake-detection-api.hf.space/api/predict',
      {
        method: 'POST',
        body: formData,
        headers: {
          'Accept': 'application/json',
        }
      }
    );
    
    const text = await response.text();
    
    // Parse response text
    const isFake = text.includes('Deepfake detected');
    const confidenceMatch = text.match(/Confidence: (\d+\.\d+)%/);
    const confidence = confidenceMatch ? parseFloat(confidenceMatch[1]) / 100 : 0;
    
    return {
      label: isFake ? 'Fake' : 'Real',
      confidence: confidence
    };
  } catch (error) {
    console.error('API Error:', error);
    throw new Error('Failed to analyze audio');
  }
}
```

---

## 7. Performance Considerations

### 7.1 Response Time

- **Small files** (<1MB): Typically 1-3 seconds
- **Medium files** (1-5MB): 3-5 seconds
- **Large files** (>5MB): 5-10+ seconds

Response time scales primarily with audio duration due to feature extraction computational complexity.

### 7.2 Rate Limits

Hugging Face Spaces enforces the following default limits:

- **Concurrent requests**: Max 5
- **Requests per minute**: ≈50 (approximate)
- **File size**: Max 25MB

### 7.3 Memory Usage

- Model and scaler size: ≈15MB combined
- Runtime memory: ≈200-500MB depending on audio size
- Peak memory during feature extraction: Up to 1GB for very long audio files

### 7.4 Optimization Tips

1. **Trim silence** from audio before uploading
2. **Compress audio** to reduce file size (MP3 ≈128kbps is sufficient)
3. **Split long recordings** into 10-30 second segments
4. **Implement client-side caching** to avoid redundant analysis of the same file

---

## 8. Deployment Notes

### 8.1 Hugging Face Spaces Configuration

The API is deployed using Hugging Face Spaces with the following configuration:

- **Framework**: Gradio
- **Hardware**: CPU (Standard)
- **Space Visibility**: Public
- **Sleep Mode**: Enabled (awakens on request)
- **Persistent Storage**: Enabled for model files

### 8.2 Files in Repository

- **app.py**: Main application code
- **deepfake_voice_detector.pkl**: Serialized XGBoost model
- **feature_scaler.pkl**: Serialized StandardScaler
- **requirements.txt**: Dependencies specification
- **README.md**: Public documentation

### 8.3 Model Updates

The model can be updated by replacing the PKL files and restarting the space:

1. Upload new model file via Hugging Face interface
2. Replace references if filenames change
3. Space will automatically restart to load new model

---

## 9. Future Enhancements

Planned improvements for future releases:

1. **Multi-class classification**: Distinguish between different types of deepfakes
2. **Segment-level analysis**: Provide frame-by-frame confidence scoring
3. **Batch processing**: Support for analyzing multiple files in one request
4. **Asynchronous API**: Webhook-based results for long audio files
5. **Explainability features**: Highlight which acoustic features triggered detection

---

## 10. Troubleshooting

| Issue | Possible Cause | Solution |
|-------|---------------|----------|
| "Internal Server Error" | File format not supported | Ensure audio is WAV or MP3 format |
| Timeout error | File too large or complex | Split file into smaller segments |
| Low confidence score | Ambiguous audio characteristics | Try a longer audio sample |
| "Model not found" error | Space recently restarted | Wait a few seconds and try again |
| Inconsistent results | Audio quality issues | Ensure clean recording with minimal background noise |

For technical support or to report issues, open a discussion on the Hugging Face Space.

---




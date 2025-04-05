# Audio Feature Selection for Deepfake Detection

## Overview
This document outlines the key audio features used for training machine learning and deep learning models for deepfake audio detection. The selected features are chosen based on their ability to capture meaningful patterns from speech signals, which help distinguish between real and synthetic (fake) audio.

---

## Selected Features

### 1. Mel-Frequency Cepstral Coefficients (MFCC)
**Definition:** MFCCs represent the short-term power spectrum of a sound, modeled on human hearing sensitivity.

**Why We Use It:**
- MFCCs are highly effective at capturing the timbral characteristics of speech.
- They are widely used in speech and speaker recognition systems.

**Pros:**
- Mimics human auditory system.
- Extracts useful phonetic information.

**Cons:**
- Sensitive to background noise.

---

### 2. Zero Crossing Rate (ZCR)
**Definition:** ZCR measures how frequently the signal changes sign (crosses the zero amplitude line).

**Why We Use It:**
- Helps detect voiced vs. unvoiced segments.
- Useful for identifying speech activity.

**Pros:**
- Fast to compute.
- Good indicator of noisiness or energy transitions.

**Cons:**
- Less effective with complex or low-frequency signals.

---

### 3. Root Mean Square Energy (RMS)
**Definition:** RMS energy measures the amplitude of the audio signal over time.

**Why We Use It:**
- Highlights segments with significant energy (speech vs. silence).
- Helps in activity detection and quality analysis.

**Pros:**
- Simple and robust.
- Helps in detecting silence or speech presence.

**Cons:**
- Alone, it lacks detail about the frequency content.

---

### 4. Spectral Centroid
**Definition:** Represents the "center of mass" of the audio spectrum.

**Why We Use It:**
- Indicates whether a sound is "bright" or "dull".
- Useful for distinguishing synthetic from natural speech.

**Pros:**
- Describes overall tone.

**Cons:**
- Sensitive to high-frequency noise.

---

### 5. Spectral Bandwidth
**Definition:** Measures the width of the frequency band in which most of the signal energy is concentrated.

**Why We Use It:**
- Identifies how frequencies are spread in the spectrum.
- Complements the spectral centroid.

**Pros:**
- Highlights harmonic vs. noisy signals.

**Cons:**
- Can be redundant with other spectral features.

---

### 6. Spectral Contrast
**Definition:** Measures the contrast between peaks and valleys in the spectrum.

**Why We Use It:**
- Differentiates between harmonic and percussive sound textures.
- Useful in speech quality and clarity analysis.

**Pros:**
- Provides a high-level view of spectral variation.

**Cons:**
- Computationally more complex.

---

## Summary
| Feature            | Captures                          | Strengths                                   | Limitations                    |
|--------------------|-----------------------------------|---------------------------------------------|-------------------------------|
| MFCC               | Timbre and speech patterns        | Highly informative for voice features       | Sensitive to noise            |
| ZCR                | Signal sign changes               | Fast, good for activity detection           | Not informative alone         |
| RMS                | Signal energy                     | Detects speech/silence transitions          | No frequency detail           |
| Spectral Centroid  | Frequency distribution center     | Identifies brightness of sound              | Affected by noise             |
| Spectral Bandwidth | Spread of frequencies             | Complements centroid                        | Can be redundant              |
| Spectral Contrast  | Peaks vs valleys in frequency     | Highlights harmonic structure differences   | Expensive to compute          |

---

## Conclusion
These selected features together provide a comprehensive view of both temporal and spectral characteristics of audio signals. This multi-dimensional representation helps machine learning models better distinguish between real and fake audio data, improving classification performance and robustness.


# Enhanced Feature Extraction Rationale for Deepfake Audio Detection

_This document provides a comprehensive technical rationale for our chosen audio features, detailing their computation methodology, underlying signal-processing principles, detection significance, and how they collectively enable discrimination between real and deepfake speech._

---

## 1. Mel-Frequency Cepstral Coefficients (MFCCs)

### Theoretical Foundation
MFCCs represent the short-term power spectrum of sound on a perceptual (mel) frequency scale, approximating the human auditory system's nonlinear frequency resolution. They effectively capture the vocal tract configuration that shapes the speech spectrum.

### Computation Methodology
MFCCs are derived through a multi-stage signal processing pipeline:
1. **Framing**: Segmentation of the waveform into short frames (typically 20-40ms windows)
2. **Windowing**: Application of a Hamming window to reduce spectral leakage
3. **Spectral Analysis**: Computation of the magnitude spectrum via Fast Fourier Transform (FFT)
4. **Mel Filtering**: Mapping the power spectrum onto the mel scale using a triangular filter bank (typically 26-40 filters)
5. **Logarithmic Compression**: Taking the logarithm of filter-bank energies to mimic human loudness perception
6. **Decorrelation**: Computing the Discrete Cosine Transform (DCT) of the log energies
7. **Coefficient Selection**: Retaining the first 13 coefficients (excluding the 0th coefficient which represents average energy)

### Implementation Details
```python
# MFCCs (13 coefficients)
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
mfccs_mean = np.mean(mfccs, axis=1)
mfccs_std = np.std(mfccs, axis=1)
```

### Features Extracted
- **`mfcc_mean_1` through `mfcc_mean_13`**: Mean of each coefficient over time, capturing the overall spectral envelope characteristics
- **`mfcc_std_1` through `mfcc_std_13`**: Standard deviation of each coefficient, representing temporal variability in formant structure and articulation

### Deepfake Detection Significance
- **Spectral envelope analysis**: Deepfake TTS often fails to perfectly reproduce natural formant trajectories and resonance patterns of human vocal tracts
- **Temporal consistency detection**: Synthetic speech typically exhibits either:
  * Overly smooth MFCC trajectories (lacking natural micro-variations)
  * Unnaturally jittery MFCC trajectories (frame-to-frame incoherence)
- **Coefficient-specific insights**:
  * Lower-order coefficients (1-4): Capture overall spectral shape, often revealing vocoder artifacts
  * Mid-order coefficients (5-9): Represent formant structure, detecting unnatural articulation patterns
  * Higher-order coefficients (10-13): Capture fine spectral details, helping identify high-frequency synthesis artifacts
- **Empirical validation**: MFCCs have demonstrated consistent effectiveness in prior deepfake detection research, with feature importance analysis confirming their discriminative power

---

## 2. Zero-Crossing Rate (ZCR)

### Theoretical Foundation
ZCR quantifies the frequency with which the audio signal transitions from positive to negative amplitude (or vice versa) within a frame, indicating frequency content and voiced/unvoiced characteristics.

### Mathematical Definition
The zero-crossing rate is formally defined as:

\[ \mathrm{ZCR} = \frac{1}{N-1} \sum_{n=1}^{N-1} \mathbb{I}(x[n] \cdot x[n-1] < 0 ) \]

Where:
- \(\mathbb{I}\) is the indicator function (1 if condition is true, 0 otherwise)
- \(N\) is the frame length
- \(x[n]\) is the audio signal at sample \(n\)

### Implementation Details
```python
# Zero Crossing Rate
zcr = librosa.feature.zero_crossing_rate(y)
zcr_mean = np.mean(zcr)
```

### Feature Extracted
- **`zcr`**: Mean zero-crossing rate across all frames in the audio sample

### Deepfake Detection Significance
- **High-frequency artifact detection**: Neural vocoders and waveform generators often introduce subtle high-frequency noise or quantization artifacts that increase the ZCR
- **Voicing characterization**: Natural speech has predictable ZCR patterns:
  * Low ZCR for voiced sounds (vowels)
  * High ZCR for unvoiced sounds (fricatives, plosives)
  * Synthetic voices frequently misrepresent this ratio or create unnatural transitions
- **Temporal consistency**: The frame-to-frame stability of ZCR in human speech follows linguistic patterns that TTS systems may not fully replicate
- **Compression artifact sensitivity**: ZCR effectively captures post-processing artifacts from compression codecs commonly applied to deepfake audio

---

## 3. Root Mean Square (RMS) Energy

### Theoretical Foundation
RMS energy measures the frame-wise acoustic power or loudness of the signal, capturing both overall volume and dynamic variations that reflect natural prosodic patterns.

### Mathematical Definition
The RMS energy for a frame is calculated as:

\[ \mathrm{RMS} = \sqrt{\frac{1}{N} \sum_{n=1}^{N} x[n]^2 } \]

Where:
- \(N\) is the frame length
- \(x[n]\) is the amplitude of the audio signal at sample \(n\)

### Implementation Details
```python
# Root Mean Square Energy
rms = librosa.feature.rms(y=y)
rms_mean = np.mean(rms)
```

### Feature Extracted
- **`rms`**: Mean RMS energy across all frames of the audio clip

### Deepfake Detection Significance
- **Dynamic range analysis**: Natural speech exhibits characteristic amplitude variations:
  * Stress patterns at the word level
  * Phrase-level energy contours
  * Micro-level variations from articulation
  * Many TTS systems produce flattened or exaggerated dynamics
- **Vocoder smoothing detection**: Neural vocoders often normalize energy excessively, creating unnaturally consistent loudness patterns
- **Breath and pause patterns**: Human speech contains natural energy dips for breathing and pauses that synthetic speech may not accurately reproduce
- **Recording condition authenticity**: RMS patterns reflect recording environments and microphone proximity in ways that generated audio often fails to simulate

---

## 4. Spectral Centroid

### Theoretical Foundation
The spectral centroid represents the "center of mass" of the spectrum, providing a measure of the predominant frequency or "brightness" of the sound.

### Mathematical Definition
The spectral centroid is calculated as the weighted mean of frequencies in the spectrum:

\[ \mathrm{Centroid} = \frac{\sum_{k} f[k] \cdot X[k]}{\sum_{k} X[k]} \]

Where:
- \(f[k]\) is the frequency at bin \(k\)
- \(X[k]\) is the magnitude at frequency bin \(k\)

### Implementation Details
```python
# Spectral features
spec_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
```

### Feature Extracted
- **`spec_centroid`**: Mean spectral centroid measured in Hz, averaged across all frames

### Deepfake Detection Significance
- **Timbre brightness characterization**: The centroid directly correlates with perceived brightness of sound:
  * Human voices have characteristic centroid distributions based on physiology and articulation
  * Synthetic voices often exhibit subtle shifts in spectral balance
- **Artifact detection capabilities**: Synthesis algorithms may produce:
  * Unnatural spectral peaks (increasing the centroid)
  * Missing high-frequency components (decreasing the centroid)
  * Frame-to-frame centroid patterns that don't follow natural speech trajectories
- **Age and gender verification**: The spectral centroid varies predictably with speaker characteristics, helping verify the consistency of purported speaker identity
- **Recording authenticity**: Microphone characteristics and room acoustics influence the centroid in ways that synthetic generation may not accurately simulate

---

## 5. Spectral Bandwidth

### Theoretical Foundation
Spectral bandwidth measures the dispersion or spread of the spectrum around the centroid, indicating the concentration or diffusion of spectral energy and overall richness of the signal.

### Mathematical Definition
The spectral bandwidth is the weighted standard deviation of frequencies around the centroid:

\[ \mathrm{Bandwidth} = \sqrt{\frac{\sum_{k} (f[k] - \mathrm{Centroid})^2 \cdot X[k]}{\sum_{k} X[k]} } \]

Where:
- \(f[k]\) is the frequency at bin \(k\)
- \(X[k]\) is the magnitude at frequency bin \(k\)
- \(\mathrm{Centroid}\) is the spectral centroid

### Implementation Details
```python
spec_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
```

### Feature Extracted
- **`spec_bandwidth`**: Mean spectral bandwidth in Hz, averaged across all frames

### Deepfake Detection Significance
- **Harmonic richness assessment**: Human voices have characteristic bandwidth patterns:
  * Vowels exhibit specific bandwidth properties based on formant structure
  * Consonants show broader bandwidth distribution
  * Synthetic voices can be either:
    - Overly narrow (missing spectral richness)
    - Unnaturally broad (excessive noise or artifacts)
- **Voice quality indicators**: Bandwidth correlates with voice qualities like breathiness, creakiness, and nasality that TTS systems struggle to authentically reproduce
- **Emotional speech detection**: Natural emotional speech affects bandwidth in predictable ways that synthetic speech often fails to capture
- **Algorithm fingerprinting**: Different voice synthesis technologies produce characteristic bandwidth patterns, enabling identification of specific generation methods

---

## 6. Spectral Rolloff

### Theoretical Foundation
The spectral rolloff represents the frequency below which a specified percentage (typically 85%) of the spectral energy is contained, providing insight into the high-frequency content distribution.

### Mathematical Definition
The P% spectral rolloff frequency \(f_R\) is defined such that:

\[ \sum_{k=0}^{k_R} X[k] = P\% \cdot \sum_{k=0}^{K-1} X[k] \]

Where:
- \(X[k]\) is the magnitude at frequency bin \(k\)
- \(k_R\) is the bin index corresponding to the rolloff frequency \(f_R\)
- \(K\) is the total number of frequency bins
- \(P\) is typically set to 85

### Implementation Details
```python
spec_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
```

### Feature Extracted
- **`spec_rolloff`**: Mean spectral rolloff frequency in Hz, averaged across all frames

### Deepfake Detection Significance
- **High-frequency artifact detection**: Rolloff effectively captures:
  * Neural vocoder noise artifacts in higher frequencies
  * Oversmoothing of high-frequency content
  * Unnatural spectral tails characteristic of certain synthesis methods
- **Natural acoustic environment verification**: Real recordings exhibit:
  * Microphone frequency response effects
  * Room acoustics that attenuate high frequencies in predictable ways
  * Background noise with specific rolloff signatures
  * These environmental factors are difficult for deepfakes to simulate accurately
- **Voice characteristics verification**: Different phonemes have characteristic rolloff patterns; deviations can indicate synthesis
- **Post-processing detection**: Audio enhancement or compression applied to mask deepfakes often creates distinctive rolloff patterns

---

## 7. Spectral Contrast

### Theoretical Foundation
Spectral contrast measures the difference between peaks (high energy) and valleys (low energy) in the spectrum across multiple sub-bands, quantifying the clarity of harmonic structure versus noise floor.

### Mathematical Definition
For each sub-band, the contrast is calculated as:

\[ \mathrm{Contrast} = 10\log_{10}\frac{\mathrm{MeanPeak}}{\mathrm{MeanValley}} \]

Where:
- \(\mathrm{MeanPeak}\) is the average of the highest percentile of energies in the sub-band
- \(\mathrm{MeanValley}\) is the average of the lowest percentile of energies in the sub-band

### Implementation Details
```python
spec_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))
```

### Feature Extracted
- **`spec_contrast`**: Mean spectral contrast averaged across all sub-bands and frames

### Deepfake Detection Significance
- **Harmonic structure analysis**: Spectral contrast effectively characterizes:
  * Clarity of vocal harmonics relative to noise
  * Presence of formant structure
  * Synthesis algorithms often:
    - Oversmooth valleys, reducing contrast
    - Create artificial peaks, increasing contrast unnaturally
- **Voicing clarity verification**: Clear voiced segments in human speech have high contrast between harmonic peaks and noise valleys
- **Generation algorithm fingerprinting**: Different synthesis methods produce characteristic contrast patterns:
  * WaveNet-based models tend toward higher contrast
  * GAN-based approaches often show specific contrast anomalies
  * Waveform interpolation methods create distinctive sub-band contrast patterns
- **Environmental authenticity**: Background noise affects contrast in predictable ways that deepfakes struggle to replicate

---

## 8. Pitch (Fundamental Frequency)

### Theoretical Foundation
Pitch, or fundamental frequency (F0), represents the perceived tonality of voiced speech segments, determined by vocal fold vibration rates and constituting a primary component of speech prosody.

### Extraction Methodology
Pitch estimation involves specialized algorithms to identify the fundamental frequency:
1. **Time-frequency analysis**: Convert signal to time-frequency representation
2. **Candidate identification**: Detect potential F0 candidates at each frame
3. **Trajectory modeling**: Apply continuity constraints to select the most likely F0 path
4. **Post-processing**: Filter to retain only voiced frames with reliable pitch estimates

### Implementation Details
```python
# Pitch (from piptrack)
pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
pitches = pitches[pitches > 0]
pitch_mean = np.mean(pitches) if len(pitches) > 0 else 0
pitch_std = np.std(pitches) if len(pitches) > 0 else 0
```

### Features Extracted
- **`pitch_mean`**: Average fundamental frequency (F0) across voiced frames, measured in Hz
- **`pitch_std`**: Standard deviation of F0, reflecting pitch variability and intonation patterns

### Deepfake Detection Significance
- **Prosody modeling assessment**: Pitch contours reveal sophisticated aspects of speech:
  * Intonation patterns (questions, statements, emphasis)
  * Emotional expression through pitch modulation
  * Micro-prosody (subtle pitch variations within syllables)
  * Many TTS systems produce:
    - Monotonic or stepped pitch contours
    - Unnatural transitions between voiced segments
    - Limited range of pitch variation
- **Speaker consistency verification**: Pitch statistics help verify speaker identity claims:
  * Natural speakers maintain consistent pitch ranges
  * Deepfakes may fail to maintain pitch consistency when generating longer utterances
- **Context-appropriateness assessment**: Pitch patterns should match linguistic and emotional context:
  * Questions exhibit rising terminal contours
  * Emphasized words show specific pitch peaks
  * Synthetic speech often fails these context-appropriateness tests
- **Inter-frame coherence**: Natural pitch transitions follow continuous trajectories; discontinuities can indicate splicing or synthesis boundaries

---

## 9. Temporal Dynamics and Combined Feature Analysis

### Multi-scale Temporal Context
Our feature extraction approach captures dynamics at three critical time scales:
- **Frame-level dynamics** (20-25ms): Instantaneous spectral characteristics
- **Mid-term patterns** (100-500ms): Syllable and phoneme-level transitions
- **Long-term trends** (1-10s): Word and phrase-level prosodic patterns

### Statistical Aggregation Strategy
- **Central tendency (mean)**: Captures overall characteristic patterns
- **Dispersion (standard deviation)**: Quantifies variability and expressive range
- This dual approach ensures both static characteristics and dynamic behavior are represented

### Feature Complementarity Matrix

| Feature Category | Captures Spectral Artifacts | Reveals Temporal Artifacts | Sensitive to Prosody | Reflects Voice Quality |
|------------------|:---------------------------:|:--------------------------:|:--------------------:|:---------------------:|
| MFCCs            | +++                         | ++                         | +                    | +++                   |
| ZCR              | ++                          | +                          | -                    | ++                    |
| RMS              | -                           | +++                        | +++                  | +                     |
| Spectral Centroid| +++                         | +                          | +                    | ++                    |
| Spectral Bandwidth| ++                         | +                          | -                    | +++                   |
| Spectral Rolloff | +++                         | +                          | -                    | ++                    |
| Spectral Contrast| +++                         | ++                         | -                    | +++                   |
| Pitch            | -                           | ++                         | +++                  | ++                    |

*Legend: +++ (strong), ++ (moderate), + (some), - (minimal)*

### Integration Benefits
- **Multimodal detection capability**: Different features excel at detecting different types of synthesis artifacts
- **Robustness to evasion**: Manipulating all features simultaneously to avoid detection is challenging
- **Interpretability advantage**: Statistical features provide explainable insights into detection decisions
- **Computational efficiency**: Fixed-dimensional feature vector enables rapid inference

---

## 10. Feature Vector Construction and Implementation

### Feature Vector Summary

| Group | Count | Feature Names | Implementation in Code |
|-------|-------|---------------|------------------------|
| MFCC Mean | 13 | mfcc_mean_1 … mfcc_mean_13 | `mfccs_mean = np.mean(mfccs, axis=1)` |
| MFCC Std | 13 | mfcc_std_1 … mfcc_std_13 | `mfccs_std = np.std(mfccs, axis=1)` |
| ZCR | 1 | zcr | `zcr_mean = np.mean(zcr)` |
| RMS | 1 | rms | `rms_mean = np.mean(rms)` |
| Spectral Centroid | 1 | spec_centroid | `spec_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))` |
| Spectral Bandwidth | 1 | spec_bandwidth | `spec_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))` |
| Spectral Rolloff | 1 | spec_rolloff | `spec_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))` |
| Spectral Contrast | 1 | spec_contrast | `spec_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))` |
| Pitch Statistics | 2 | pitch_mean, pitch_std | `pitch_mean = np.mean(pitches) if len(pitches) > 0 else 0` |

**Total dimension:** 34 features
### Interpretability Benefits

The extracted features provide not just classification power but also interpretability:

1. **Feature importance analysis**: Models like XGBoost can rank features by importance, revealing which acoustic properties most distinguish real from synthetic speech
2. **Correlation analysis**: Understanding correlations between features helps identify redundancy and critical feature combinations
3. **Threshold-based rules**: Simple rules based on feature thresholds can be derived for transparent detection logic
4. **Visualization potential**: Features can be plotted in reduced dimensionality space (PCA, t-SNE) to visualize clustering patterns

By combining signal processing expertise with machine learning, this feature set creates a robust foundation for deepfake audio detection that balances performance with explainability.

---
# Audio Truth Teller — Comprehensive Build & Deployment Guide

**Version:** 1.0.0  
**Last Updated:** 2025-04-30  
**Authors:** Report Writing Team

---

## Table of Contents
1. [Application Overview](#1-application-overview)  
2. [Technology Stack](#2-technology-stack)  
3. [Prerequisites](#3-prerequisites)  
4. [Environment Setup](#4-environment-setup)  
5. [Project Structure](#5-project-structure)  
6. [Configuration Details](#6-configuration-details)  
7. [Web Build & Run](#7-web-build--run)  
8. [iOS Build & Run](#8-ios-build--run)  
9. [Git & GitHub Integration](#9-git--github-integration)  
10. [Continuous Integration & Deployment](#10-continuous-integration--deployment)  
11. [Troubleshooting Guide](#11-troubleshooting-guide)  
12. [Advanced Configuration](#12-advanced-configuration)  
13. [Appendix](#13-appendix)

---

## 1. Application Overview

**Audio Truth Teller** is a sophisticated cross-platform mobile-first web application designed to analyze audio recordings and classify them as either **authentic human speech** or **AI-generated deepfakes**. Leveraging cutting-edge machine learning technology through an API interface, the application provides users with accessible tools to verify audio authenticity.

### 1.1 Key Features

- **Real-time Audio Recording**: Capture voice samples directly through device microphone
- **File Upload Analysis**: Support for .wav/.mp3 audio file analysis
- **Configurable Backend**: Flexible API endpoint configuration for deepfake detection service
- **Confidence Scoring**: Numerical assessment of classification reliability
- **Cross-Platform Compatibility**: Seamless experience across web browsers and iOS devices
- **Intuitive User Interface**: Modern, responsive design with clear visual feedback
- **Privacy-Focused**: Local processing where possible, minimizing data transmission

### 1.2 User Workflow

1. User initiates recording or uploads audio file
2. Application processes audio and extracts acoustic features
3. Features are sent to backend API for analysis
4. Results are displayed with classification and confidence score
5. Option to save, share, or analyze another audio sample

This guide provides comprehensive instructions for developers to clone, configure, build, and deploy the application across both web and iOS platforms.

---

## 2. Technology Stack

Audio Truth Teller leverages a modern stack of web and mobile technologies, each carefully selected for performance, developer productivity, and cross-platform compatibility:

| Technology | Role | Implementation Benefits |
|------------|------|-------------------------|
| **React** | UI library | Component-based architecture enables rapid iteration on complex UI flows (recording, playback, result display) while ensuring maintainable code through reusable components |
| **TypeScript** | Programming language | Static typing catches errors at compile time and provides superior IDE autocompletion, improving code quality and developer confidence as the application grows in complexity |
| **Vite** | Build tool | Lightning-fast development server with native ES module support and on-demand hot module replacement reduces rebuild times to under 100ms, enabling real-time feedback during development |
| **Tailwind CSS** | Styling framework | Utility-first approach allows composing responsive, modern layouts quickly without writing custom CSS, while enforcing design tokens for consistent UI |
| **Shadcn UI** | Component library | Pre-built, accessible components styled with Tailwind provide ready-made buttons, cards, dialogs, and form controls, accelerating interface development without sacrificing quality |
| **React Router** | Navigation | Declarative routing manages navigation between home screen, recording view, file picker, results, and settings pages, synchronizing URLs with application state |
| **TanStack Query** | Data management | Handles asynchronous state management for fetching, caching, and updating data, simplifying API interactions with automated retries, caching, and loading states |
| **Lucide React** | Icon library | Lightweight SVG icons provide clear visual indicators (play, record, upload) without significant bundle size impact |
| **Capacitor** | Native bridge | Cross-platform bridge by Ionic wraps the PWA in a native shell, exposing device APIs like microphone and filesystem under a consistent JavaScript interface |
| **capacitor-voice-recorder** | Audio plugin | Streamlines microphone permission requests and audio capture, abstracting away low-level AVFoundation details on iOS |
| **@capacitor/filesystem** | File handling | Provides read/write access to device local storage for loading user-selected audio files and saving temporary recordings |
| **@capacitor/dialog** | UI notifications | Offers native alert and confirmation dialogs for consistent, platform-native feedback |
| **Git & GitHub** | Version control | Tracks code changes, feature branches, and integrates with GitHub Actions for CI/CD, maintaining code history while excluding large audio assets |

---

## 3. Prerequisites

Ensure your development environment has the following tools installed and properly configured before beginning:

| Tool | Minimum Version | Installation | Verification |
|------|-----------------|--------------|-------------|
| **Node.js & npm** | v16.x or later | https://nodejs.org | `node -v && npm -v` |
| **npm CLI** | v9.x or later | Bundled with Node.js | `npm -v` |
| **Git** | v2.39.5+ | https://git-scm.com | `git --version` |
| **Xcode** | 15.x | Mac App Store | Open Xcode, check "About Xcode" |
| **CocoaPods** | v1.11.3+ | `sudo gem install cocoapods` | `pod --version` |
| **Capacitor CLI** | 7.x | `npm install -g @capacitor/cli` | `cap --version` |
| **VS Code** (recommended) | Latest stable | https://code.visualstudio.com | Open VS Code, check "About" |

### 3.1 Development Environment Recommendations

- **Operating System**: macOS 12+ (Monterey or later) for full iOS development
- **Hardware**: Apple Silicon Mac (M1/M2/M3) recommended for optimal iOS simulator performance
- **Browser**: Chrome/Safari latest version for testing
- **VS Code Extensions**:
  - ESLint
  - Prettier
  - Tailwind CSS IntelliSense
  - GitHub Copilot (optional)

### 3.2 iOS Development Requirements

- Active Apple Developer account for testing on physical devices
- iPhone or iPad running iOS 14+ for physical device testing
- Xcode Command Line Tools installed (`xcode-select --install`)

---

## 4. Environment Setup

Follow these steps to set up your development environment:

### 4.1 Repository Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/ArshT12/audio-truth-teller-app.git
   cd audio-truth-teller-app
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Verify installation**
   ```bash
   npm list --depth=0
   ```
   Confirm key dependencies are installed correctly (react, typescript, vite, etc.)

### 4.2 Environment Configuration

1. **Configure environment variables**
   - No sensitive keys required for client
   - Ensure `capacitor.config.ts` is correctly configured
   - Remove any placeholder entries (e.g., `lovable` references)

2. **Verify project root structure**
   ```bash
   ls -la
   ```
   Ensure the following directories/files exist:
   ```text
   ├── src/
   ├── public/
   ├── ios/
   ├── capacitor.config.ts
   ├── package.json
   └── vite.config.ts
   ```

### 4.3 Development Editor Setup

1. **Launch VS Code in project directory**
   ```bash
   code .
   ```

2. **Install recommended extensions**
   - Open Extensions view (Ctrl+Shift+X / Cmd+Shift+X)
   - Search for and install the recommended extensions listed in section 3.1

3. **Configure editor settings** (optional)
   Create or update `.vscode/settings.json`:
   ```json
   {
     "editor.formatOnSave": true,
     "editor.defaultFormatter": "esbenp.prettier-vscode",
     "editor.codeActionsOnSave": {
       "source.fixAll.eslint": true
     },
     "typescript.tsdk": "node_modules/typescript/lib"
   }
   ```

---

## 5. Project Structure

Understanding the project's organization is essential for efficient development:

```text
audio-truth-teller-app/
├── src/                      # React/TypeScript source
│   ├── components/           # Reusable UI components
│   │   ├── ui/               # Shadcn UI components
│   │   ├── audio-player/     # Audio playback components
│   │   ├── record-button/    # Recording interface
│   │   └── results/          # Result display components
│   ├── hooks/                # Custom React hooks
│   │   ├── use-audio-recorder.ts  # Recording hook
│   │   ├── use-file-upload.ts     # File handling hook
│   │   └── use-detection-api.ts   # API communication hook
│   ├── lib/                  # Utility functions
│   │   ├── api.ts            # API client
│   │   ├── audio-utils.ts    # Audio processing utilities
│   │   └── validation.ts     # Input validation
│   ├── pages/                # Route-based pages
│   │   ├── home/             # Landing page
│   │   ├── record/           # Recording interface
│   │   ├── results/          # Analysis results
│   │   └── settings/         # Application settings
│   ├── App.tsx               # Root component & router
│   └── main.tsx              # Entry point
├── public/                   # Static assets & index.html
│   ├── assets/               # Images, icons, etc.
│   └── index.html            # HTML entry point
├── ios/                      # iOS native project
│   ├── App/                  # Xcode workspace
│   └── Podfile               # CocoaPods dependencies
├── capacitor.config.ts       # Capacitor configuration
├── package.json              # npm scripts & dependencies
├── vite.config.ts            # Vite build settings
├── tsconfig.json             # TypeScript configuration
├── tailwind.config.js        # Tailwind CSS settings
└── README.md                 # Project overview & documentation
```

### 5.1 Key Component Structure

Core functionality is organized into well-defined components and hooks:

- **AudioRecorder**: Handles microphone access, recording, and audio capture
- **FileUploader**: Manages file selection and validation
- **ResultView**: Displays classification results and confidence scores
- **SettingsPanel**: Configures API endpoints and application preferences

### 5.2 Code Organization Principles

- **Component-Based Architecture**: UI elements are modular and reusable
- **Custom Hook Pattern**: Complex logic is encapsulated in custom hooks
- **Separation of Concerns**: Clear distinction between UI, business logic, and API communication
- **Consistent Naming**: Descriptive, consistent naming conventions throughout

---

## 6. Configuration Details

Several configuration files control the application's behavior across platforms:

### 6.1 Capacitor Configuration (`capacitor.config.ts`)

```typescript
import { CapacitorConfig } from '@capacitor/cli';

const config: CapacitorConfig = {
  appId: 'com.yourorg.audio-truth-teller',
  appName: 'AudioTruthTeller',
  webDir: 'dist',
  plugins: {
    VoiceRecorder: {
      requestAudioPermission: true,
    },
  },
  server: {
    // Development settings (remove for production)
    url: 'http://localhost:4123',
    cleartext: true
  }
};

export default config;
```

**Configuration Options:**
- **appId**: Unique reverse-DNS identifier for app stores (change to your organization)
- **appName**: Display name shown on device
- **webDir**: Output directory from Vite build process
- **plugins**: Plugin-specific configurations
  - **VoiceRecorder**: Controls microphone permission behavior
- **server**: Local development server settings (remove for production)

### 6.2 Vite Configuration (`vite.config.ts`)

```typescript
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react-swc';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  resolve: { 
    alias: { '@': path.resolve(__dirname, 'src') } 
  },
  server: { 
    host: '0.0.0.0', 
    port: 4123 
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
    minify: 'terser',
    terserOptions: {
      compress: {
        drop_console: false,  // Keep for debugging in production
      }
    }
  }
});
```

**Key Settings:**
- **plugins**: Vite plugins (react-swc for faster compilation)
- **resolve.alias**: Path aliases for cleaner imports
- **server**: Development server configuration
- **build**: Production build options
  - **outDir**: Must match `webDir` in capacitor.config.ts
  - **sourcemap**: Useful for debugging production issues
  - **minify**: Code optimization settings

### 6.3 HTML Metadata (`public/index.html`)

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no" />
  <title>Audio Truth Teller</title>
  <meta name="description" content="Detect whether audio clips are real or AI-generated deepfakes." />
  <meta name="author" content="Your Team or Company" />

  <!-- Favicon -->
  <link rel="icon" type="image/png" href="/assets/favicon.png" />
  <link rel="apple-touch-icon" href="/assets/apple-touch-icon.png" />

  <!-- Open Graph / Social Cards -->
  <meta property="og:title" content="Audio Truth Teller" />
  <meta property="og:description" content="AI-based audio deepfake detection in your browser or iOS." />
  <meta property="og:image" content="/assets/og-image.png" />
  <meta property="og:type" content="website" />

  <!-- Twitter -->
  <meta name="twitter:card" content="summary_large_image" />
  <meta name="twitter:site" content="@YourTwitterHandle" />
  <meta name="twitter:image" content="/assets/twitter-image.png" />

  <!-- PWA Support -->
  <meta name="theme-color" content="#ffffff" />
  <link rel="manifest" href="/manifest.json" />
</head>
<body>
  <div id="root"></div>
  <script type="module" src="/src/main.tsx"></script>
</body>
</html>
```

**Important Metadata:**
- **viewport**: Mobile-specific settings to prevent zooming
- **description**: SEO and link preview description
- **Open Graph/Twitter**: Social sharing preview data
- **PWA Support**: Progressive Web App configuration

### 6.4 TypeScript Configuration (`tsconfig.json`)

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true,
    "baseUrl": ".",
    "paths": {
      "@/*": ["./src/*"]
    }
  },
  "include": ["src"],
  "references": [{ "path": "./tsconfig.node.json" }]
}
```

**Key TypeScript Settings:**
- **strict**: Enforces type checking rules
- **paths**: Configures path aliases to match Vite configuration
- **include**: Specifies files to be included in compilation

---

## 7. Web Build & Run

Follow these steps to build and run the application for web:

### 7.1 Development Environment

1. **Start development server**
   ```bash
   npm run dev
   ```
   This launches a hot-reloading development server at `http://localhost:4123`

2. **Development features**
   - Real-time reloading on code changes
   - Error overlay for TypeScript/React errors
   - Network access via local IP (for testing on real devices)

### 7.2 Production Build

1. **Create optimized build**
   ```bash
   npm run build
   ```
   This generates production-ready files in the `/dist` directory:
   - JavaScript bundle with tree-shaking and minification
   - Optimized assets with content-based hashing
   - HTML with injected resource references

2. **Preview production build locally**
   ```bash
   npm run preview
   ```
   Serves the production build at `http://localhost:4173` for verification

### 7.3 Advanced Build Options

1. **Custom environment configuration**
   ```bash
   # Development with mock API
   npm run dev:mock
   
   # Production build with specific API target
   VITE_API_URL=https://api.example.com npm run build
   ```

2. **Analyze bundle size** (optional)
   ```bash
   npm run build:analyze
   ```
   Generates a visual report of bundle composition and size

### 7.4 Web Deployment Options

1. **Static hosting** (Netlify, Vercel, GitHub Pages)
   - Upload the `dist` directory to your preferred static hosting provider
   - Configure redirects for SPA routing (`_redirects` or `vercel.json`)

2. **Server deployment**
   - Copy `dist` to web server directory
   - Configure server for SPA routing (redirect 404s to index.html)

---

## 8. iOS Build & Run

Building for iOS requires additional steps to bridge the web application to native capabilities:

### 8.1 Prepare Native Project

1. **Build web assets first**
   ```bash
   npm run build
   ```
   Ensures the latest web code is ready for native bundling

2. **Sync with Capacitor**
   ```bash
   npx cap sync ios
   ```
   This command:
   - Copies web assets from `/dist` → `ios/App/App/public`
   - Updates native plugins based on package.json
   - Runs pod install for iOS dependencies

3. **Update native configuration**
   Open `ios/App/App/Info.plist` and ensure these entries exist:
   ```xml
   <key>NSMicrophoneUsageDescription</key>
   <string>This app requires microphone access to record audio for deepfake analysis.</string>
   
   <key>UIBackgroundModes</key>
   <array>
     <string>audio</string>
   </array>
   ```

### 8.2 Open and Build in Xcode

1. **Launch Xcode project**
   ```bash
   npx cap open ios
   ```
   This opens the generated Xcode workspace

2. **Configure signing and capabilities**
   - Select the "App" target
   - Navigate to "Signing & Capabilities"
   - Choose your development team
   - Verify "Microphone" capability is enabled

3. **Build and run**
   - Select a simulator or connected device
   - Click the Play button (▶️) to build and run
   - First launch may take several minutes to compile

### 8.3 Run from Command Line

Alternatively, use the Capacitor CLI to build and run:

1. **List available simulators**
   ```bash
   xcrun simctl list devices
   ```
   Note the UUID of your target device

2. **Run on specific simulator**
   ```bash
   npx cap run ios --target=DEVICE_UUID
   ```
   Replace `DEVICE_UUID` with the identifier from the previous command

3. **Run on connected device**
   ```bash
   npx cap run ios --device
   ```
   This will prompt you to select from connected devices

### 8.4 Troubleshooting iOS Builds

- **Pod installation issues**:
  ```bash
  cd ios/App
  pod install --repo-update
  ```

- **Build failures**:
  - Check Xcode logs for specific errors
  - Verify minimum iOS version (iOS 14+)
  - Confirm correct provisioning profile for physical devices

---

## 9. Git & GitHub Integration

Effective version control is essential for maintaining code quality and team collaboration:

### 9.1 Repository Setup

1. **Configure remote origin**
   ```bash
   git remote add origin https://github.com/ArshT12/audio-truth-teller-app.git
   git branch -M main
   git push -u origin main
   ```

2. **Verify remote configuration**
   ```bash
   git remote -v
   ```
   Should display the fetch and push URLs for origin

### 9.2 Gitignore Configuration

The `.gitignore` file excludes unnecessary files from version control:

```gitignore
# Build outputs
dist/
www/
.vite/

# Dependencies
node_modules/
ios/App/Pods/

# Environment
.env
.env.local
.env.development
.env.production

# IDEs and editors
.idea/
.vscode/*
!.vscode/extensions.json
!.vscode/settings.json

# Operating system
.DS_Store
Thumbs.db

# Capacitor
ios/App/App/public/
android/app/src/main/assets/public/

# Logs
logs
*.log
npm-debug.log*

# Testing
coverage/

# Temporary files
tmp/
temp/
*.tmp
```

### 9.3 Branching Strategy

Follow a well-defined branching model:

1. **Main branch** (`main`)
   - Production-ready code
   - Protected from direct pushes
   - Requires pull request and code review

2. **Development branch** (`develop`)
   - Integration branch for features
   - Deploys to staging/development environment

3. **Feature branches** (`feature/<name>`)
   - Created from `develop`
   - Merged back to `develop` via pull request
   - Example: `feature/audio-recording-ui`

4. **Hotfix branches** (`hotfix/<name>`)
   - Created from `main`
   - Merged to both `main` and `develop`
   - Used for critical production fixes

### 9.4 Commit Conventions

Follow semantic commit messages:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting changes
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Build process or tools

**Example**:
```
feat(audio): add waveform visualization

Implement real-time waveform display during recording to provide
visual feedback to users about audio levels.

Closes #42
```

---

## 10. Continuous Integration & Deployment

Automate testing, building, and deployment processes:

### 10.1 GitHub Actions Configuration

Create `.github/workflows/main.yml`:

```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: 16
          cache: 'npm'
      - run: npm ci
      - run: npm run lint
      - run: npm run test
  
  build-web:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: 16
          cache: 'npm'
      - run: npm ci
      - run: npm run build
      - name: Upload build artifacts
        uses: actions/upload-artifact@v3
        with:
          name: web-build
          path: dist/
  
  deploy-web:
    needs: build-web
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/download-artifact@v3
        with:
          name: web-build
          path: dist
      - name: Deploy to Netlify
        uses: nwtgck/actions-netlify@v2
        with:
          publish-dir: './dist'
          production-branch: main
          github-token: ${{ secrets.GITHUB_TOKEN }}
          deploy-message: "Deploy from GitHub Actions"
        env:
          NETLIFY_AUTH_TOKEN: ${{ secrets.NETLIFY_AUTH_TOKEN }}
          NETLIFY_SITE_ID: ${{ secrets.NETLIFY_SITE_ID }}
```

### 10.2 Environment Variables

Configure the following secrets in GitHub repository settings:

1. **API Configuration**
   - `VITE_API_ENDPOINT`: Backend API URL
   - `VITE_API_KEY`: API authentication key (if required)

2. **Deployment Credentials**
   - `NETLIFY_AUTH_TOKEN`: Netlify authentication token
   - `NETLIFY_SITE_ID`: Netlify site identifier

### 10.3 Deployment Environments

Configure multiple environments for different stages:

1. **Development**
   - Deploys from `develop` branch
   - Uses development API endpoints
   - Staging URL: `dev.audio-truth-teller.example.com`

2. **Production**
   - Deploys from `main` branch
   - Uses production API endpoints
   - Production URL: `audio-truth-teller.example.com`

### 10.4 iOS CI Pipeline (Advanced)

For automated iOS builds:

1. **Configure iOS certificates in GitHub Actions**
   - Use `fastlane match` to manage certificates
   - Store Apple credentials securely in GitHub Secrets

2. **Create iOS build workflow**
   ```yaml
   ios-build:
     runs-on: macos-latest
     steps:
       - uses: actions/checkout@v3
       - uses: actions/setup-node@v3
       - run: npm ci
       - run: npm run build
       - run: npx cap sync ios
       - name: Set up Ruby
         uses: ruby/setup-ruby@v1
       - name: Install Fastlane
         run: gem install fastlane
       - name: Build iOS App
         run: fastlane ios build
   ```

---

## 11. Troubleshooting Guide

Common issues and their solutions:

### 11.1 Web Development Issues

| Issue | Possible Cause | Solution |
|-------|---------------|----------|
| `Module not found` error | Incorrect import path | Check import path and `@` alias configuration |
| Hot reload not working | Watcher limitations | Restart dev server with `npm run dev` |
| CORS errors | API endpoint restrictions | Add appropriate CORS headers on API or use a proxy |
| TypeScript errors | Type mismatches | Check type definitions and interfaces |

### 11.2 iOS Build Problems

| Issue | Possible Cause | Solution |
|-------|---------------|----------|
| `NSMicrophoneUsageDescription` crash | Missing Info.plist key | Add the key in `ios/App/App/Info.plist` |
| `Cannot find module 'lovable-tagger'` | Leftover AI scaffold plugin | Remove from `package.json` & `vite.config.ts` |
| `webDir` build assets not found | Wrong build folder | Ensure `npm run build` completes successfully and outputs to `dist` |
| iOS native dependencies fail | CocoaPods mismatch | Run `cd ios/App && pod install --repo-update` |
| Code signing errors | Missing profile | Set up proper code signing in Xcode |

### 11.3 Recording and Audio Issues

| Issue | Possible Cause | Solution |
|-------|---------------|----------|
| No microphone access | Permission denied | Check Info.plist and permission request timing |
| Recording fails silently | Capacitor plugin error | Check console logs and ensure plugin is properly installed |
| Audio playback issues | Format compatibility | Verify audio format is supported (WAV/MP3) |
| File upload errors | Size limitations | Check file size limits and implement validation |

### 11.4 API Connection Problems

| Issue | Possible Cause | Solution |
|-------|---------------|----------|
| API timeout | Network issues or server load | Implement retry logic and timeout handling |
| Authentication failure | Invalid API key | Verify API key configuration in settings |
| Incorrect response format | API version mismatch | Update expected response format in code |

---

## 12. Advanced Configuration

Fine-tune the application for specific deployment scenarios:

### 12.1 API Integration

Configure the API client in `src/lib/api.ts`:

```typescript
import axios from 'axios';

// Create API client instance
const apiClient = axios.create({
  baseURL: import.meta.env.VITE_API_ENDPOINT || 'https://api.default-url.com',
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
  }
});

// Optional authentication interceptor
apiClient.interceptors.request.use(config => {
  const apiKey = import.meta.env.VITE_API_KEY;
  if (apiKey) {
    config.headers['X-API-Key'] = apiKey;
  }
  return config;
});

// Audio analysis endpoint
export async function analyzeAudio(audioData: Blob): Promise<{
  label: 'Real' | 'Fake';
  confidence: number;
}> {
  const formData = new FormData();
  formData.append('audio', audioData);
  
  try {
    const response = await apiClient.post('/analyze', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      }
    });
    
    return response.data;
  } catch (error) {
    console.error('API Error:', error);
    throw new Error('Failed to analyze audio');
  }
}
```

### 12.2 Performance Optimization

1. **Code splitting**
   ```typescript
   // In src/App.tsx
   import { lazy, Suspense } from 'react';
   
   const Results = lazy(() => import('./pages/results/Results'));
   const Settings = lazy(() => import('./pages/settings/Settings'));
   
   // In router
   <Suspense fallback={<LoadingSpinner />}>
     <Route path="/results" element={<Results />} />
   </Suspense>
   ```

2. **Audio processing optimization**
   ```typescript
   // In src/hooks/use-audio-recorder.ts
   const optimizeAudioForUpload = (blob: Blob): Promise<Blob> => {
     return new Promise((resolve) => {
       // Check if compression needed
       if (blob.size < 1024 * 1024) {
         // Under 1MB, use as is
         resolve(blob);
         return;
       }
       
       // Implement compression logic for larger files
       // ...compression code...
     });
   };
   ```

### 12.3 Custom Capacitor Plugin Configuration

For advanced native functionality, customize plugin behavior:

```typescript
// In capacitor.config.ts
export default {
  // ...other config
  plugins: {
    VoiceRecorder: {
      requestAudioPermission: true,
      sampleRate: 44100,
      bitRate: 128000,
      audioChannelConfig: 'MONO',
      encoderBitRate: 128000
    },
    CapacitorHttp: {
      enabled: true
    },
    SplashScreen: {
      launchShowDuration: 2000,
# Smart Posture and Ergonomics Tracker

[![Python 3.13](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![PySide6](https://img.shields.io/badge/GUI-PySide6%20(Qt%206.12)-green.svg)](https://wiki.qt.io/Qt_for_Python)
[![MediaPipe Tasks](https://img.shields.io/badge/Vision-MediaPipe%20Tasks-orange.svg)](https://developers.google.com/mediapipe)
[![Tests Passing](https://img.shields.io/badge/Tests-68%20Passed-brightgreen.svg)](https://docs.pytest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

A modern, privacy-first desktop application for Windows 11 designed to improve workplace ergonomics by detecting sustained posture anomalies in real time without recording, saving, or transmitting raw video frames.

---

## 🌟 Key Features

1. **Intelligent Posture Estimation**:
   - Detects **slouching** (ear-to-shoulder vertical compression ratio < 0.72).
   - Detects **forward-head craning** (relative head delta > 0.08).
   - Detects **shoulder tilt misalignment** (> 7.5° difference).
   - Detects **lateral spinal leaning** (> 0.07 lateral offset).
   - Detects **screen proximity** (inter-ocular eye distance expansion > 28% over baseline).
   - Distinguishes body visibility and transient harmless movement from sustained poor posture using temporal smoothing (EMA $\alpha = 0.30$) and hysteresis.

2. **Personalized Multi-Sample Calibration**:
   - Capture a 30-sample baseline of your comfortable, optimal upright posture in seconds.
   - All subsequent posture comparisons adapt to your personal body proportions, screen distance, and camera placement.

3. **Windows 11 Notification Tray Integration**:
   - Native system tray icon in the Windows taskbar (bottom-right next to the clock).
   - Ergonomic alerts and warnings pop up as native Windows 11 Toast notifications so you stay informed without interrupting your focus.
   - Quick access context menu to open dashboard, toggle monitoring, or test the overlay.

4. **Multi-Camera & Phone Webcam Auto-Detection**:
   - Automatically probes and detects connected webcams (built-in laptop cameras, USB webcams, and phones connected via DroidCam, Iriun, Camo, or Windows Phone Link).
   - DirectShow (`CAP_DSHOW`) priority guarantees seamless compatibility with virtual phone webcam drivers.
   - Live **"🔄 Scan Camera"** button on the dashboard header detects newly connected phone webcams without restarting.
   - Supports direct Wi-Fi RTSP / HTTP phone streams (e.g. `http://192.168.x.x:8080/video`).

5. **Full-Screen Soft-Blur Overlay**:
   - Triggers when 3 consecutive posture warnings are ignored.
   - **Instant Force-Trigger (<kbd>F12</kbd> or <kbd>Ctrl+T</kbd>)**: Test the full-screen overlay on demand with 1 click or shortcut!
   - Built as a topmost, frameless, non-activating window (`Qt.WindowType.Tool` + `WA_ShowWithoutActivating`) that **never minimizes, closes, resizes, or steals focus from your active application**.
   - Hardware-accelerated **Windows 11 DWM Acrylic blur** (`DwmSetWindowAttribute` with `DWMWA_SYSTEMBACKDROP_TYPE = 38`) with a dark translucent fallback scrim (`rgba(10, 12, 18, 205)`).
   - Live **green recovery state** ("Posture corrected!") that automatically fades away after 3 seconds of sustained upright posture.
   - Reliable emergency dismiss shortcuts (<kbd>Esc</kbd> or <kbd>Ctrl+Alt+P</kbd>).

6. **Adaptive Power Monitoring**:
   - **Plugged into AC**: Runs continuous monitoring at 3–5 FPS with non-blocking background threading.
   - **Laptop on Battery**: Enforces battery conservation mode with 5-minute periodic burst checks (capturing a 3–5 second sample at $640 \times 480$) and releasing webcam hardware immediately after each check.

7. **100% Privacy Guaranteed**:
   - All computer-vision analysis takes place strictly in volatile RAM.
   - Zero raw video frames, images, or face meshes are ever saved to disk or transmitted over the network.
   - SQLite database schema stores only anonymized numeric session metrics and event timestamps.

8. **Zero-Webcam Interactive Simulator**:
   - Includes a built-in `MockPostureGenerator` and interactive simulation dropdown right on the dashboard.
   - Allows complete testing and demonstration of all ergonomic states (slouching, craning, tilting, recovery, empty chair) without requiring physical camera hardware.


---

## 🏗️ Technical Architecture

```text
smart-posture-tracker/
├── app.py                      # Application bootstrap & Windows 11 DLL isolation
├── requirements.txt            # Locked dependencies (PySide6, OpenCV, MediaPipe, etc.)
├── models/
│   └── pose_landmarker_lite.task # MediaPipe Tasks Lite model file
├── src/
│   ├── core/
│   │   ├── calibration.py      # Multi-sample ergonomic baseline calibration
│   │   ├── posture_engine.py   # Multi-landmark geometry, EMA smoothing, hysteresis
│   │   ├── reminder_manager.py # State machine, rate limiter, escalation engine
│   │   └── mock_generator.py   # Synthetic skeleton generator for zero-hardware demo
│   ├── vision/
│   │   ├── camera.py           # Thread-safe OpenCV camera manager with burst support
│   │   └── pose_detector.py    # MediaPipe Tasks Vision wrapper
│   ├── system/
│   │   ├── power_manager.py    # Battery & AC power detection via psutil
│   │   └── session_logger.py   # Privacy-first SQLite session & event analytics
│   └── ui/
│       ├── dashboard.py        # Modern PySide6 desktop dashboard
│       ├── overlay.py          # Non-activating Windows 11 Acrylic blur overlay
│       ├── settings.py         # Configurable ergonomics & threshold dialog
│       └── worker.py           # QThread background orchestration engine
├── tests/                      # 67 Automated pytest unit & integration tests
│   ├── test_database.py        # SQLite session lifecycle & privacy compliance audit
│   ├── test_environment.py     # Windows 11 environment & dependency validation
│   ├── test_overlay.py         # Overlay flags, Acrylic API safety & recovery states
│   ├── test_posture_engine.py  # Geometry, EMA filter, hysteresis & deviations
│   ├── test_power_manager.py   # AC/Battery transitions & fallback handling
│   ├── test_reminders.py       # Rate limiting, 3-warning escalation & recovery hold
│   ├── test_ui.py              # PySide6 settings dialog & main dashboard UI
│   └── test_vision.py          # OpenCV capture & MediaPipe extraction
└── docs/                       # Architectural specifications & guides
```

---

## 🚀 Getting Started

### 1. Prerequisites
- **Operating System**: Windows 11 (64-bit)
- **Python**: 3.10 to 3.13 (verified on Python 3.13.9)
- **Webcam**: Any built-in or USB webcam (or use the built-in simulator)

### 2. Virtual Environment Setup
Open PowerShell in the project directory:

```powershell
# Navigate to the project directory
cd C:\Users\iamsi\.gemini\antigravity\scratch\smart-posture-tracker

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 3. Launching the Application

Run the application:
```powershell
.\.venv\Scripts\python.exe app.py
```

---

## 🎮 How to Use the Application

1. **Start Monitoring**:
   - Click the green **▶ Start Monitoring** button.
   - If a physical webcam is connected, live video appears in the viewport.
   - If no webcam is available, the built-in simulator automatically takes over seamlessly.

2. **Calibrate Your Posture**:
   - Sit upright in your ideal comfortable working position.
   - Click **🎯 Calibrate Posture Baseline**.
   - Maintain your upright posture for 3 seconds while 30 samples are recorded.

3. **Testing the Full-Screen Soft-Blur Overlay (Instant Force-Trigger)**:
   - Click the red **"🚨 Test Blur Overlay Now (F12)"** button on the dashboard, or press **<kbd>F12</kbd>** (or **<kbd>Ctrl+T</kbd>**) at any time.
   - The full-screen Acrylic blur overlay immediately appears over your desktop.
   - Test the message, dimming, and emergency dismissal with <kbd>Esc</kbd>.

4. **Connecting a Phone as a Webcam**:
   - Start your phone webcam app (e.g. **DroidCam**, **Iriun Webcam**, **Camo**, or **Windows Phone Link**).
   - In the tracker dashboard header, click **"🔄 Scan Camera"**.
   - The app automatically probes all connected video devices, selects your phone camera, and streams live video!
   - You can also explicitly select your camera device or enter an IP stream URL in **⚙ Settings**.

5. **Windows Notification Tray Alerts**:
   - The app lives in your Windows Taskbar tray (bottom-right next to the clock).
   - Posture alerts, warnings, and recovery confirmations appear as native Windows 11 Toast popups.
   - Right-click the tray icon to quickly open the dashboard, toggle monitoring, or trigger the overlay test.

6. **Demonstrating Posture Reminders & Escalation**:
   - Use the **Simulation State** dropdown under the camera viewport to test any posture:
     - Select `SLOUCHING`: Notice the status bar changes to orange/red with slouch compression metric.
     - Keep it slouching for 20 seconds to trigger **Warning 1**.
     - Notice the 3-stage escalation ladder (`Warning 1` $\rightarrow$ `Warning 2` $\rightarrow$ `Warning 3`).
     - Once 3 warnings are ignored, the **Full-Screen Soft-Blur Overlay** escalates automatically.
     - Switch the simulation back to `GOOD`: Watch the overlay transition to the green **"Posture corrected!"** recovery hold and smoothly dismiss after 3 seconds.

7. **Customizing Settings**:
   - Click **⚙ Settings** in the top bar to customize:
     - Reminder persistence thresholds (default: 20 seconds)
     - Warning cooldown interval (default: 5 minutes)
     - Escalation threshold (default: 3 warnings)
     - Recovery confirmation hold (default: 3 seconds)
     - Windows Tray notifications, sound alerts, and event history logging.

---

## 🧪 Automated Testing

The project includes an exhaustive test suite with **68 tests** covering 100% of the subsystems:


```powershell
.\.venv\Scripts\pytest.exe -v
```

### Test Coverage Highlights:
- **`test_environment.py` (7 tests)**: Windows 11 DLL isolation, Python 3.13 compatibility, PySide6, OpenCV, MediaPipe imports.
- **`test_vision.py` (9 tests)**: Frame acquisition, camera error recovery, MediaPipe landmarking, tilt angle math.
- **`test_posture_engine.py` (14 tests)**: Calibration baselines, slouch detection, forward craning, tilt, EMA filter, hysteresis.
- **`test_ui.py` (5 tests)**: Settings serialization, main dashboard lifecycle, live UI metric updates.
- **`test_reminders.py` (11 tests)**: 20s transient filter, 5-minute cooldown, 3-warning escalation, 3s recovery hold.
- **`test_overlay.py` (9 tests)**: Non-activating window flags, Acrylic blur API, green recovery hold, emergency shortcuts.
- **`test_power_manager.py` (6 tests)**: Battery vs AC detection, desktop fallback, worker scheduling.
- **`test_database.py` (6 tests)**: SQLite session lifecycle, event logging, calibration profiles, schema privacy audit.

---

## 🔒 Privacy & Security

The Smart Posture and Ergonomics Tracker is engineered from the ground up to respect user privacy:
- **Zero Local Video Storage**: No webcam frames or images are saved to disk or database.
- **Zero Cloud Transmission**: All inference runs locally on your machine via CPU/GPU.
- **Audited Relational Schema**: The database schema strictly prohibits `BLOB`, `IMAGE`, or `VIDEO` columns, validated continuously by automated regression tests (`test_privacy_compliance_guarantee`).

---

## ⚠️ Ergonomic Disclaimer

*Notice: This application provides automated workplace ergonomic posture tracking and habit awareness only. It is not a medical device and does not diagnose, treat, or prevent any spinal, orthopedic, or medical conditions.*


## Make Ui Dynamic
# IrisKeys

IrisKeys is a Windows eye-control prototype for finals/demo use. It combines webcam gaze tracking, calibration, assistive magnetism, demo selection, and OS cursor control in one product flow.

## What It Does

- 9-point calibration with saved local calibration data
- Demo mode with fullscreen targets, dwell arming, and confirm flow
- OS mode with real Windows cursor control
- Smart Magnetism assist for easier target acquisition
- Dwell click flow and blink click support
- Transparent OS overlay for visible gaze cursor feedback
- Floating accessibility toolbar
- Voice typing, on-screen keyboard, pause/resume, and next right-click toggle
- Launcher UI for demo and OS launches without terminal arguments
- Hardware kill-switch in OS mode with `F12` or `ESC`

## Requirements

- Windows
- Webcam
- Python 3.11 recommended
- A working virtual environment at `C:\Users\User\infomatrix\venv`

## Install

From `C:\Users\User\infomatrix`:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Main Dependencies

- `opencv-python`
- `mediapipe`
- `numpy`
- `filterpy`
- `pyautogui`
- `pywin32`
- `SpeechRecognition`
- `PyAudio`
- `PyQt5`
- `scipy`

## Launch Options

### 1. Launcher

```powershell
.\venv\Scripts\python.exe src\launcher.py
```

The launcher is the recommended entry point for finals.

It provides:

- `Launch Demo Mode`
- `Launch OS Mode`
- `Smart Magnetism Assist`
- `Dwell Click`

Both launch buttons automatically start calibration first, then continue into the selected mode without requiring a second click.

### 2. Demo Mode

```powershell
.\venv\Scripts\python.exe src\main.py --mode demo --assist on --click dwell --os-click off --drift off
```

Use this for:

- rehearsal
- calibration checks
- presentation sandbox

### 3. OS Mode

```powershell
.\venv\Scripts\python.exe src\main.py --mode os --assist on --click dwell --os-click on --drift off
```

Use this for:

- real Windows cursor control
- overlay + toolbar
- finals field conditions

## Safety

In OS mode:

- `F12` = emergency stop
- `ESC` = emergency stop

These immediately stop OS control.

## Click Behavior

### Blink Click

- Blink click can trigger OS clicks when mouse control is active
- If the toolbar has `Right Click` armed, the next blink click becomes a right click
- After that single click, it resets back to left click

### Dwell Click

- Dwell click is enabled with `--click dwell`
- In the demo interaction flow, dwell arms the target and confirm triggers the click/output path
- If `Right Click` is armed in the toolbar, the next dwell-confirm OS click becomes a right click

## Floating Toolbar

In OS mode, the toolbar provides:

- `Keyboard`
  - opens Windows On-Screen Keyboard
- `Voice Type`
  - listens to speech and pastes recognized text into the active field
- `Right Click`
  - makes the next OS click right-click, then resets to left
- `Pause`
  - pauses tracking output so the cursor stops moving

## Voice Typing Notes

Voice typing currently:

- calibrates for ambient noise
- tries `ru-RU`, `ru-KZ`, then `en-US`
- pastes text through the Windows clipboard for better Unicode support

For best results:

- click into a text field first
- speak one short phrase at a time
- keep the microphone close and reduce background noise

## Files

Core entry points:

- `C:\Users\User\infomatrix\src\main.py`
- `C:\Users\User\infomatrix\src\launcher.py`
- `C:\Users\User\infomatrix\src\overlay.py`
- `C:\Users\User\infomatrix\src\toolbar.py`

Calibration artifacts are local-only and should not be committed:

- `C:\Users\User\infomatrix\calibration\`
- `C:\Users\User\infomatrix\src\calibration\`

## Quick Verification

Check interpreter:

```powershell
.\venv\Scripts\python.exe -c "import sys; print(sys.executable)"
```

Check required imports:

```powershell
.\venv\Scripts\python.exe -c "import cv2, mediapipe, numpy, win32api, speech_recognition, pyaudio; print('imports ok')"
```

## Final Instructions

1. Launch `src\launcher.py`
2. Enable `Smart Magnetism Assist`
3. Choose whether `Dwell Click` is needed
4. Run calibration
5. Demo Mode
6. OS Mode

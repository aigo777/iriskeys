
# Webcam Gaze Assist (MediaPipe FaceMesh)

## What it is
A webcam-only gaze interaction demo. We don’t aim for perfect Tobii-level accuracy. Instead, we build an **assistive system** that works despite noisy gaze using:
- short **per-user calibration**
- stable **dwell + confirm** selection
- optional **gentle magnetism** (no snapping)

## How it works (short)
Webcam → FaceMesh/iris landmarks → raw gaze point → **calibration mapping(ML per user)** → screen cursor → **5-target demo UI**  
Selection:
- Look at a target to **arm** it (progress ring fills)
- Press **SPACE** to confirm (**SUCCESS**)
- Pressing SPACE off-target counts as **FALSE_SELECT**

## Install
**Python 3.11 recommended**

### Create venv
**Windows (PowerShell)**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
````

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

## Run

### Calibration

```bash
python src/main.py calibrate
```

### Demo (safe)

```bash
python src/main.py --assist off --os-click off
```

### Demo with magnetism

```bash
python src/main.py --assist on --os-click off
```

### Demo with OS click (use carefully)

```bash
python src/main.py --assist on --os-click on
```

## Controls

* **SPACE**: confirm selection (when ARMED)
* **ESC**: cancel selection
* **P**: pause/unpause
* **Q**: quit
* **K**: start calibration
* **M**: toggle mouse mode (use gaze as OS mouse, then switch tabs/apps by alt+tab)

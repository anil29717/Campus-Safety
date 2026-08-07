# Campus Safety AI — setup on a new PC

Guide to install and run the **face_dashboard** app (FastAPI backend + React admin UI) on another machine.

---

## 1. Prerequisites

| Software | Version | Notes |
|----------|---------|--------|
| **Python** | 3.10 – 3.12 | [python.org](https://www.python.org/downloads/) — check “Add to PATH” on Windows |
| **Node.js** | 18+ | [nodejs.org](https://nodejs.org/) — includes `npm` |
| **Git** | any recent | To clone the repository |
| **Webcam** | optional | Required for live camera features |

**Hardware:** CPU works; NVIDIA GPU optional (faster YOLO inference).

---

## 2. Get the code

```powershell
git clone <your-repo-url> sanskrituniversity
cd sanskrituniversity\ultralytics
```

If you copy the folder manually (USB / ZIP), open a terminal in the `ultralytics` directory.

---

## 3. Python environment

### Windows (PowerShell)

```powershell
cd ultralytics
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
pip install -r face_dashboard\requirements-admin.txt
```

### Linux / macOS

```bash
cd ultralytics
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
pip install -r face_dashboard/requirements-admin.txt
```

`pip install -e .` installs **Ultralytics** (YOLO, OpenCV, PyTorch, etc.) from this repo.

---

## 4. React admin UI

```powershell
cd face_dashboard\admin-ui
npm install
```

---

## 5. First run — models download automatically

On first camera start, these files are downloaded into `face_dashboard/models/`:

- YuNet face detector (`.onnx`)
- SFace face recognizer (`.onnx`)
- YOLO weights (`yolo26n.pt`, `yolo11n-pose.pt`, etc.)
- Fire/smoke model (only if that feature is enabled)

**Internet is required** for the first run. Allow a few minutes on a slow connection.

---

## 6. Run the application

Use **two terminals**. Keep the Python venv activated in terminal 1.

### Terminal 1 — API backend (port 8000)

```powershell
cd ultralytics
.\.venv\Scripts\Activate.ps1
python -m face_dashboard.api.server
```

You should see: `Uvicorn running on http://0.0.0.0:8000`

### Terminal 2 — Admin dashboard (port 5173)

```powershell
cd ultralytics\face_dashboard\admin-ui
npm run dev
```

Open in browser: **http://localhost:5173**

### Start the camera

1. Go to **Live Camera** or **Dashboard → Feature Control**
2. Set camera index (`0` = default webcam)
3. Click **Start**
4. Toggle features (face recognition, zones, etc.) as needed

---

## 7. Optional: Streamlit dashboard

```powershell
cd ultralytics
.\.venv\Scripts\Activate.ps1
pip install streamlit
streamlit run face_dashboard/app.py
```

Opens at **http://localhost:8501**

---

## 8. Optional: OpenCV live window (`live_fast`)

```powershell
cd ultralytics
.\.venv\Scripts\Activate.ps1
python -m face_dashboard.live_fast
```

Press **N** in the window to register a new face from the camera.

---

## 8b. CP Plus IP camera (instead of laptop webcam)

1. Copy the example env file and edit with your camera details:

```powershell
cd ultralytics
copy .env.example .env
```

2. Set in `.env`:

```
CAMERA_IP=172.16.40.2
CAMERA_USER=admin
CAMERA_PASSWORD=your_password
CAMERA_CHANNEL=1
CAMERA_SUBTYPE=1
```

`subtype=1` uses the lower-resolution sub-stream (faster for face recognition).

3. Set how many NVR channels to list (default 4):

```
CAMERA_CHANNELS_COUNT=4
```

4. Restart the API server, then open **Camera Roles** (`/cameras`) to assign:
   - **Entry** — clock-in when a face is seen
   - **Exit** — clock-out when a face is seen
   - **Restricted** — safety zone monitoring
   - **General** — normal AI monitoring

5. On **Live Camera**, pick a camera and click **Start**. Use **Grid** to preview all channels.

The laptop and CP Plus camera must be on the **same network** (same LAN). Test RTSP:

```powershell
python -c "import cv2; c=cv2.VideoCapture('rtsp://admin:Admin%%401234@172.16.40.2:554/cam/realmonitor?channel=1&subtype=1', cv2.CAP_FFMPEG); print('opened:', c.isOpened())"
```

(Password `@` must be `%40` in the URL.)

---

## 9. Folder layout

```
ultralytics/
├── .venv/                    # Python virtual environment (not in git)
├── data/                     # Runtime data (not in git)
│   ├── known_faces/          # Registered persons
│   ├── unknown_faces/        # Unknown visitor photos
│   ├── entries/              # Entry / exit log
│   ├── activity/             # Emotion & behavior log
│   ├── safety/               # Zones & incidents
│   └── config/features.json  # Feature toggles
├── face_dashboard/
│   ├── api/                  # FastAPI server
│   ├── admin-ui/             # React dashboard
│   ├── models/               # Downloaded ONNX / YOLO weights
│   └── services/             # Detection feature modules
```

---

## 10. Troubleshooting

| Problem | Fix |
|---------|-----|
| `Cannot open camera 0` | Try index `1` or `2`; close other apps using the webcam |
| Black video in browser | Refresh the page; ensure API is running on port 8000 |
| `ModuleNotFoundError: face_dashboard` | Run commands from `ultralytics/` with venv active |
| Slow FPS on CPU | Disable heavy features (Fighting, Fire/Smoke, Behavior) in **Features** page |
| Port 8000 in use | Stop other processes or change port in `face_dashboard/api/server.py` |
| `npm` not found | Install Node.js and restart the terminal |

### Test API health

```powershell
curl.exe http://127.0.0.1:8000/api/health
curl.exe http://127.0.0.1:8000/api/status
```

---

## 11. Production build (optional)

Build the React app for static hosting:

```powershell
cd face_dashboard\admin-ui
npm run build
```

Serve `dist/` with any static file server; point API requests to your backend URL.

---

## Quick start (copy-paste)

```powershell
# One-time setup
cd ultralytics
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
pip install -r face_dashboard\requirements-admin.txt
cd face_dashboard\admin-ui
npm install
cd ..\..

# Every time you run
# Terminal 1:
cd ultralytics
.\.venv\Scripts\Activate.ps1
python -m face_dashboard.api.server

# Terminal 2:
cd ultralytics\face_dashboard\admin-ui
npm run dev
```

Then open **http://localhost:5173** and start the camera.

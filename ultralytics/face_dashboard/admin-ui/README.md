# Campus Safety Admin UI

React + Tailwind CSS admin dashboard for the campus safety AI system.

## Prerequisites

- Python venv with `ultralytics` + `face_dashboard` dependencies
- Node.js 18+

## 1. Install API dependencies

```powershell
cd D:\sanskrituniversity\sanskrituniversity\ultralytics
.\.venv\Scripts\Activate.ps1
pip install fastapi uvicorn
```

## 2. Start the API (port 8000)

```powershell
python -m face_dashboard.api.server
```

## 3. Install & run React admin (port 5173)

```powershell
cd face_dashboard\admin-ui
npm install
npm run dev
```

Open **http://localhost:5173**

## Pages

| Page | Description |
|------|-------------|
| Dashboard | Live metrics, alerts, activity |
| Live Camera | MJPEG stream + start/stop |
| Persons | Registered face database |
| Objects | Live YOLO object detection |
| Entry History | Attendance / presence log |
| Emotion & Behavior | Registered user activity |
| Incidents | Unattended bag, zone violations |
| Safety Zones | Restricted & crowd zone config |

## API endpoints

- `GET /api/status` — live metrics
- `GET /api/camera/stream` — MJPEG feed
- `POST /api/camera/start` / `stop`
- `GET /api/persons`, `/entries`, `/activity`, `/incidents`, `/zones`

## Register faces

Use Streamlit or `live_fast` (press **N**) to add persons — they appear in the admin **Persons** page.

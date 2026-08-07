@echo off
echo Starting Campus Safety AI...

echo Starting API Server...
start "API Server" cmd /k ".\.venv\Scripts\activate.bat && python -m face_dashboard.api.server"

echo Starting Admin UI...
cd face_dashboard\admin-ui
start "Admin UI" cmd /k "npm run dev"

echo Both servers are starting in separate windows!
echo Once they are ready, open http://localhost:5173 in your browser.
pause

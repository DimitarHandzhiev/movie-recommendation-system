@echo off

echo ===============================
echo Starting Movie Recommender App
echo ===============================

REM --- Start Backend ---
echo Starting backend...
REM set TMDB_API_KEY=your_api_key
cd backend
start "Backend" cmd /k "..\.venv\Scripts\activate && python -m uvicorn main:app --reload --port 8000"

REM --- Wait a bit for backend to boot ---
timeout /t 5 > nul

REM --- Start Frontend ---
echo Starting frontend...
cd ../frontend
start "Frontend" cmd /k "npm start"

REM --- Wait a bit for frontend ---
timeout /t 5 > nul

REM --- Open browser ---
echo Opening browser...
start http://localhost:3000

echo ===============================
echo App started!
echo ===============================
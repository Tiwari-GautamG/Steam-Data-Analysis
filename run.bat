@echo off
echo Starting Steam Analytics Dashboard...
echo.
echo Make sure your MySQL database is running with:
echo - Database: steam_data
echo - Table: steamout
echo - User credentials configured in app.py
echo.
python app.py
pause

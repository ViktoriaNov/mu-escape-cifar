@echo off
REM Eksperyment diagnostyczny patience (rozdz. 6)
REM Szacowany czas: ~1.5-3 h na zbiór (RTX 3050)

cd /d "%~dp0.."
call .venv\Scripts\activate.bat 2>nul

echo === CIFAR-10 ===
python -m thesis_code.experiments --patience --dataset CIFAR10

echo.
echo === CIFAR-100 (opcjonalnie) ===
set /p RUN100="Uruchomic CIFAR-100? (t/n): "
if /i "%RUN100%"=="t" (
    python -m thesis_code.experiments --patience --dataset CIFAR100
)

echo.
echo Gotowe. Sprawdz: results\CIFAR10\diagnostyka_patience.csv
echo              oraz results\CIFAR10\diagnostyka_sharpness.png
pause

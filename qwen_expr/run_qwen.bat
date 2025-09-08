@echo off
REM Batch script to run Qwen3-Coder-480B-A35B-Instruct for vulnerability detection

echo Running Qwen3-Coder-480B-A35B-Instruct Vulnerability Detection
echo ================================================================

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Check if data path is provided as argument
if "%~1"=="" (
    echo Usage: %0 ^<path_to_test_data.jsonl^> [output_folder]
    echo.
    echo Example: %0 ..\data\primevul_test.jsonl .\results
    pause
    exit /b 1
)

set DATA_PATH=%1
set OUTPUT_FOLDER=%2

REM Set default output folder if not provided
if "%OUTPUT_FOLDER%"=="" set OUTPUT_FOLDER=.\output

REM Check if data file exists
if not exist "%DATA_PATH%" (
    echo Error: Data file not found: %DATA_PATH%
    pause
    exit /b 1
)

echo Data file: %DATA_PATH%
echo Output folder: %OUTPUT_FOLDER%
echo.

REM Create output directory
if not exist "%OUTPUT_FOLDER%" mkdir "%OUTPUT_FOLDER%"

REM Run the vulnerability detection
echo Starting vulnerability detection with Qwen3-Coder-480B-A35B-Instruct...
echo Using Chain-of-Thought prompting with few-shot examples
echo.

python run_qwen_480b.py --data_path "%DATA_PATH%" --output_folder "%OUTPUT_FOLDER%" --strategy cot

if %errorlevel% equ 0 (
    echo.
    echo ================================================================
    echo SUCCESS: Vulnerability detection completed!
    echo Results saved to: %OUTPUT_FOLDER%
    echo.
    echo To calculate VD-Score, run:
    echo python ..\calc_vd_score.py --pred_file "%OUTPUT_FOLDER%\predictions.txt" --test_file "%DATA_PATH%"
) else (
    echo.
    echo ERROR: Vulnerability detection failed!
)

echo.
pause

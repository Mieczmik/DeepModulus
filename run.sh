ts=$(date +"%Y%m%d_%H%M%S")
mkdir -p logs
LOG_FILE="logs/run_${ts}.log"

PYTHON_SCRIPT="Run.py"

source .venv/bin/activate
export DDE_BACKEND=pytorch
nohup python3 -u "$PYTHON_SCRIPT" > "$LOG_FILE" 2>&1 &
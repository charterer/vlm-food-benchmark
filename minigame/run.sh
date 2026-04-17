#!/bin/bash
cd "$(dirname "$0")"

# Default port is 5050, can be overridden with PORT=XXXX ./run.sh
PORT="${PORT:-5050}"

# Mode flags (BLIND_MODE, SINGLE_BLIND_MODE)
# BLIND_MODE: Original blind mode with bidirectional linking
# SINGLE_BLIND_MODE: Two-phase approach (merged list → equivalencies)

if ! python3 -c "import flask" 2>/dev/null; then
    echo "Installing Flask..."
    pip install flask
fi

echo "=========================================="
echo "  Food Annotation Game"
echo "=========================================="
echo ""
echo "Server: http://localhost:$PORT"
echo ""
if [ "$SINGLE_BLIND_MODE" = "1" ] || [ "$SINGLE_BLIND_MODE" = "true" ]; then
    echo "*** SINGLE-BLIND MODE ***"
    echo "  Phase 1: Review merged ingredient list"
    echo "  Phase 2: Draw equivalency lines"
    echo ""
elif [ "$BLIND_MODE" = "1" ] || [ "$BLIND_MODE" = "true" ]; then
    echo "*** BLIND MODE ***"
    echo ""
fi
echo "Usage:"
echo "  ./run.sh                        # Normal mode"
echo "  BLIND_MODE=1 ./run.sh           # Blind mode"
echo "  SINGLE_BLIND_MODE=1 ./run.sh    # Single-blind mode (two-phase)"
echo "  PORT=5055 ./run.sh              # Custom port"
echo ""
echo "After playing, export results:"
echo "  python3 export_results.py"
echo ""

PORT=$PORT BLIND_MODE=$BLIND_MODE SINGLE_BLIND_MODE=$SINGLE_BLIND_MODE python3 app.py

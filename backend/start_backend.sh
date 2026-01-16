#!/bin/bash

# ============================================================================
# OTT Churn Analytics - Backend Startup Script
# Purpose: Start FastAPI backend with automatic port cleanup
# ============================================================================

echo "🚀 Starting OTT Churn Analytics Backend..."
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Port to check and use
PORT=8000

# ============================================================================
# 1. PROCESS DISCOVERY - Check if port 8000 is in use
# ============================================================================

echo -e "${BLUE}[1/3]${NC} Checking for existing processes on port ${PORT}..."

# Use lsof to find PID of process using port 8000
PID=$(lsof -ti:${PORT} 2>/dev/null)

if [ -z "$PID" ]; then
    echo -e "${GREEN}✅ Port ${PORT} is free${NC}"
    echo ""
else
    # ========================================================================
    # 2. CONDITIONAL KILL - Kill the process if found
    # ========================================================================
    
    echo -e "${YELLOW}⚠️  Found process(es) using port ${PORT} (PID: $PID)${NC}"
    echo -e "${YELLOW}Clearing port ${PORT}...${NC}"
    
    # Kill the process(es) gracefully first, then force kill if needed
    for pid in $PID; do
        echo -e "${YELLOW}Killing process $pid...${NC}"
        kill -9 $pid 2>/dev/null
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✅ Successfully killed PID $pid${NC}"
        else
            echo -e "${RED}❌ Failed to kill PID $pid${NC}"
        fi
    done
    
    # Give the system a moment to release the port
    sleep 2
    echo -e "${GREEN}✅ Port ${PORT} is now clear${NC}"
    echo ""
fi

# ============================================================================
# 3. STARTUP - Start the FastAPI backend
# ============================================================================

echo -e "${BLUE}[2/3]${NC} Starting FastAPI server..."
echo -e "${BLUE}[3/3]${NC} Backend available at: http://127.0.0.1:${PORT}"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Verify main.py exists
if [ ! -f "$SCRIPT_DIR/main.py" ]; then
    echo -e "${RED}❌ Error: main.py not found in $SCRIPT_DIR${NC}"
    exit 1
fi

# Verify models folder exists
if [ ! -d "$SCRIPT_DIR/models" ]; then
    echo -e "${RED}❌ Error: models/ folder not found in $SCRIPT_DIR${NC}"
    exit 1
fi

echo -e "${GREEN}✅ All prerequisites verified${NC}"
echo ""

# Start the FastAPI backend with uvicorn
echo "📡 Uvicorn is starting..."
echo -e "${YELLOW}Press Ctrl+C to stop the server${NC}"
echo ""
echo "---" 

# Execute uvicorn with reload enabled
cd "$SCRIPT_DIR"
python -m uvicorn main:app --host 127.0.0.1 --port ${PORT} --reload

# If we get here, the server was stopped
echo ""
echo "---"
echo -e "${YELLOW}Backend server stopped${NC}"

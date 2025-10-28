#!/bin/bash

# Healthcare Assistant Streaming Services Manager
# Manages all microservices for the streaming setup

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="/Users/aaravraina/Documents/HackHarvard/marco"
BACKEND_PORT=8001
FRONTEND_PORT=3001
VENV_PATH="$PROJECT_ROOT/venv"
BACKEND_SCRIPT="$PROJECT_ROOT/src/fastapi_stagehand_server_simple.py"
FRONTEND_DIR="$PROJECT_ROOT/healthcare-assistant-ts"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if port is in use
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 0  # Port is in use
    else
        return 1  # Port is free
    fi
}

# Function to kill processes on specific ports
kill_port() {
    local port=$1
    print_status "Checking port $port..."
    if check_port $port; then
        print_warning "Port $port is in use, killing processes..."
        lsof -ti:$port | xargs kill -9 2>/dev/null || true
        sleep 2
    else
        print_success "Port $port is free"
    fi
}

# Function to wait for service to be ready
wait_for_service() {
    local url=$1
    local service_name=$2
    local max_attempts=30
    local attempt=1
    
    print_status "Waiting for $service_name to be ready..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s "$url" >/dev/null 2>&1; then
            print_success "$service_name is ready!"
            return 0
        fi
        
        echo -n "."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    print_error "$service_name failed to start after $max_attempts attempts"
    return 1
}

# Function to start backend service
start_backend() {
    print_status "Starting backend service on port $BACKEND_PORT..."
    
    # Kill any existing processes
    kill_port $BACKEND_PORT
    
    # Start backend in background
    cd "$PROJECT_ROOT"
    source "$VENV_PATH/bin/activate"
    
    # Start the backend service
    nohup python "$BACKEND_SCRIPT" > backend.log 2>&1 &
    BACKEND_PID=$!
    echo $BACKEND_PID > backend.pid
    
    # Wait for backend to be ready
    if wait_for_service "http://localhost:$BACKEND_PORT" "Backend"; then
        print_success "Backend service started successfully (PID: $BACKEND_PID)"
        return 0
    else
        print_error "Backend service failed to start"
        return 1
    fi
}

# Function to start frontend service
start_frontend() {
    print_status "Starting frontend service on port $FRONTEND_PORT..."
    
    # Kill any existing processes
    kill_port $FRONTEND_PORT
    
    # Start frontend in background
    cd "$FRONTEND_DIR"
    
    # Start the frontend service
    nohup pnpm dev --port $FRONTEND_PORT > frontend.log 2>&1 &
    FRONTEND_PID=$!
    echo $FRONTEND_PID > frontend.pid
    
    # Wait for frontend to be ready
    if wait_for_service "http://localhost:$FRONTEND_PORT" "Frontend"; then
        print_success "Frontend service started successfully (PID: $FRONTEND_PID)"
        return 0
    else
        print_error "Frontend service failed to start"
        return 1
    fi
}

# Function to stop all services
stop_services() {
    print_status "Stopping all services..."
    
    # Stop backend
    if [ -f "$PROJECT_ROOT/backend.pid" ]; then
        BACKEND_PID=$(cat "$PROJECT_ROOT/backend.pid")
        if kill -0 $BACKEND_PID 2>/dev/null; then
            print_status "Stopping backend (PID: $BACKEND_PID)..."
            kill $BACKEND_PID
            rm -f "$PROJECT_ROOT/backend.pid"
        fi
    fi
    
    # Stop frontend
    if [ -f "$FRONTEND_DIR/frontend.pid" ]; then
        FRONTEND_PID=$(cat "$FRONTEND_DIR/frontend.pid")
        if kill -0 $FRONTEND_PID 2>/dev/null; then
            print_status "Stopping frontend (PID: $FRONTEND_PID)..."
            kill $FRONTEND_PID
            rm -f "$FRONTEND_DIR/frontend.pid"
        fi
    fi
    
    # Kill any remaining processes
    kill_port $BACKEND_PORT
    kill_port $FRONTEND_PORT
    
    print_success "All services stopped"
}

# Function to show status
show_status() {
    print_status "Service Status:"
    echo "=================="
    
    # Check backend
    if check_port $BACKEND_PORT; then
        print_success "Backend: Running on port $BACKEND_PORT"
        curl -s "http://localhost:$BACKEND_PORT" | jq . 2>/dev/null || echo "Backend responding"
    else
        print_error "Backend: Not running"
    fi
    
    # Check frontend
    if check_port $FRONTEND_PORT; then
        print_success "Frontend: Running on port $FRONTEND_PORT"
    else
        print_error "Frontend: Not running"
    fi
    
    echo ""
    print_status "Access URLs:"
    echo "Frontend UI: http://localhost:$FRONTEND_PORT"
    echo "Backend API: http://localhost:$BACKEND_PORT"
    echo "API Docs: http://localhost:$BACKEND_PORT/docs"
}

# Function to show logs
show_logs() {
    local service=$1
    
    case $service in
        "backend")
            if [ -f "$PROJECT_ROOT/backend.log" ]; then
                tail -f "$PROJECT_ROOT/backend.log"
            else
                print_error "Backend log not found"
            fi
            ;;
        "frontend")
            if [ -f "$FRONTEND_DIR/frontend.log" ]; then
                tail -f "$FRONTEND_DIR/frontend.log"
            else
                print_error "Frontend log not found"
            fi
            ;;
        *)
            print_error "Usage: $0 logs [backend|frontend]"
            ;;
    esac
}

# Main script logic
case "${1:-start}" in
    "start")
        print_status "Starting Healthcare Assistant Streaming Services..."
        echo "=================================================="
        
        # Start backend first
        if start_backend; then
            # Start frontend
            if start_frontend; then
                echo ""
                print_success "All services started successfully!"
                echo ""
                show_status
                echo ""
                print_status "To stop services: $0 stop"
                print_status "To view logs: $0 logs [backend|frontend]"
            else
                print_error "Frontend failed to start"
                stop_services
                exit 1
            fi
        else
            print_error "Backend failed to start"
            exit 1
        fi
        ;;
    
    "stop")
        stop_services
        ;;
    
    "restart")
        print_status "Restarting all services..."
        stop_services
        sleep 3
        $0 start
        ;;
    
    "status")
        show_status
        ;;
    
    "logs")
        show_logs $2
        ;;
    
    *)
        echo "Healthcare Assistant Streaming Services Manager"
        echo "Usage: $0 {start|stop|restart|status|logs [backend|frontend]}"
        echo ""
        echo "Commands:"
        echo "  start   - Start all services"
        echo "  stop    - Stop all services"
        echo "  restart - Restart all services"
        echo "  status  - Show service status"
        echo "  logs    - Show service logs"
        exit 1
        ;;
esac


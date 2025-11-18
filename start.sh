#!/bin/bash
# Credit Intelligence Platform - Quick Start Script

set -e

echo "🚀 Starting Credit Intelligence Platform..."
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if we're using Docker or local
if command -v docker-compose &> /dev/null; then
    echo -e "${GREEN}✓${NC} Docker Compose found"
    
    echo ""
    echo "Choose deployment method:"
    echo "1) Docker Compose (Recommended)"
    echo "2) Local PM2"
    read -p "Select [1-2]: " choice
    
    if [ "$choice" = "1" ]; then
        echo ""
        echo -e "${YELLOW}Starting services with Docker Compose...${NC}"
        docker-compose up -d
        
        echo ""
        echo -e "${GREEN}✓${NC} Services started!"
        echo ""
        echo "Access points:"
        echo "  - Frontend:  http://localhost:8501"
        echo "  - Backend:   http://localhost:8000"
        echo "  - API Docs:  http://localhost:8000/api/v1/docs"
        echo ""
        echo "View logs:"
        echo "  docker-compose logs -f"
        
    else
        echo ""
        echo -e "${YELLOW}Starting services with PM2...${NC}"
        
        # Check Python
        if ! command -v python3 &> /dev/null; then
            echo -e "${RED}✗${NC} Python 3 not found. Please install Python 3.12+"
            exit 1
        fi
        
        # Install dependencies
        echo "Installing backend dependencies..."
        cd backend && pip install -q -r requirements.txt
        
        echo "Installing frontend dependencies..."
        cd ../frontend && pip install -q -r requirements.txt
        cd ..
        
        # Start with PM2
        echo "Starting services with PM2..."
        pm2 start ecosystem.config.cjs
        
        echo ""
        echo -e "${GREEN}✓${NC} Services started!"
        echo ""
        echo "Access points:"
        echo "  - Frontend:  http://localhost:8501"
        echo "  - Backend:   http://localhost:8000"
        echo "  - API Docs:  http://localhost:8000/api/v1/docs"
        echo ""
        echo "PM2 commands:"
        echo "  pm2 list          - View all services"
        echo "  pm2 logs          - View logs"
        echo "  pm2 restart all   - Restart services"
        echo "  pm2 stop all      - Stop services"
    fi
else
    echo -e "${RED}✗${NC} Docker Compose not found"
    echo "Installing with PM2..."
    
    # Same PM2 logic as above
    cd backend && pip install -q -r requirements.txt
    cd ../frontend && pip install -q -r requirements.txt
    cd ..
    pm2 start ecosystem.config.cjs
    
    echo ""
    echo -e "${GREEN}✓${NC} Services started with PM2!"
fi

echo ""
echo -e "${GREEN}🎉 Credit Intelligence Platform is ready!${NC}"
echo ""

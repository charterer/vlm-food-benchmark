#!/bin/bash
# Deployment script for the Nutrition Game to AWS server
# Run this locally from your minigame directory

set -e

SERVER="ubuntu@3.82.138.84"
KEY="/Users/everett/Downloads/lightsail-us-east-1.pem"
REMOTE_DIR="/home/ubuntu/minigame"

echo "=== Deploying Nutrition Game to $SERVER ==="

# Step 1: Create remote directory
echo "[1/5] Creating remote directory..."
ssh -i "$KEY" "$SERVER" "mkdir -p $REMOTE_DIR"

# Step 2: Sync files (excluding large image directories if needed)
echo "[2/5] Syncing files..."
rsync -avz --progress \
    -e "ssh -i $KEY" \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.git' \
    --exclude 'exports/*' \
    --exclude 'nginx' \
    ./ "$SERVER:$REMOTE_DIR/"

# Step 3: Install dependencies on server
echo "[3/5] Installing Python dependencies..."
ssh -i "$KEY" "$SERVER" "cd $REMOTE_DIR && pip3 install -r requirements.txt"

# Step 4: Copy nginx configs
echo "[4/5] Copying nginx configuration..."
scp -i "$KEY" nginx/apps-proxy.conf "$SERVER:/tmp/apps-proxy.conf"
scp -i "$KEY" nginx/index.html "$SERVER:/tmp/index.html"

# Step 5: Print remaining manual steps
echo ""
echo "=== Files synced! Now SSH in and run these commands: ==="
echo ""
echo "ssh -i $KEY $SERVER"
echo ""
echo "# Install nginx config"
echo "sudo cp /tmp/apps-proxy.conf /etc/nginx/sites-available/apps-proxy"
echo "sudo cp /tmp/index.html /var/www/html/index.html"
echo ""
echo "# Disable old site configs (keep as backup)"
echo "sudo rm -f /etc/nginx/sites-enabled/dfkviewer-5001"
echo "sudo rm -f /etc/nginx/sites-enabled/canvis-5005"
echo "sudo rm -f /etc/nginx/sites-enabled/ash-image-bank"
echo ""
echo "# Enable new unified config"
echo "sudo ln -sf /etc/nginx/sites-available/apps-proxy /etc/nginx/sites-enabled/"
echo ""
echo "# Test and reload nginx"
echo "sudo nginx -t"
echo "sudo systemctl reload nginx"
echo ""
echo "# Start the Nutrition Game (use tmux or screen to keep it running)"
echo "cd $REMOTE_DIR"
echo "tmux new -s nutritiongame"
echo "BLIND_MODE=1 PORT=5100 python3 app.py"
echo "# Press Ctrl+B then D to detach from tmux"
echo ""
echo "# Or use systemd service (see nutritiongame.service)"

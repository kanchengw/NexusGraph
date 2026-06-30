#!/bin/sh
# Grafana dashboard bootstrap script
# Used as an init container or docker entrypoint override

set -e

GRAFANA_URL="${GRAFANA_URL:-http://localhost:3000}"
GRAFANA_USER="${GRAFANA_USER:-admin}"
GRAFANA_PASSWORD="${GRAFANA_PASSWORD:-admin}"

echo "Waiting for Grafana to be ready..."
until curl -s -f -u "$GRAFANA_USER:$GRAFANA_PASSWORD" "$GRAFANA_URL/api/health" > /dev/null 2>&1; do
  sleep 2
done

echo "Importing dashboards..."
DASHBOARD_DIR="/etc/grafana/provisioning/dashboards/json"

for file in "$DASHBOARD_DIR"/*.json; do
  if [ -f "$file" ]; then
    echo "Importing: $file"
    response=$(curl -s -X POST "$GRAFANA_URL/api/dashboards/db" \
      -u "$GRAFANA_USER:$GRAFANA_PASSWORD" \
      -H "Content-Type: application/json" \
      -d "$(cat "$file")")
    
    status=$(echo "$response" | grep -o '"status"[^,]*' | cut -d'"' -f4)
    if [ "$status" = "success" ]; then
      echo "  -> Imported successfully"
    else
      echo "  -> Failed: $response"
    fi
  fi
done

echo "Dashboard import complete"

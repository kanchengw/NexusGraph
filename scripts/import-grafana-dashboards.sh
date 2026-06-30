#!/bin/bash
# Grafana dashboard bootstrap - import dashboards via API
set -e

GRAFANA_URL="http://admin:admin@localhost:3000"
DASHBOARD_DIR="./grafana/dashboards/json"

echo "Waiting for Grafana..."
until curl -sf -o /dev/null "$GRAFANA_URL/api/health"; do
    sleep 2
done

for f in "$DASHBOARD_DIR"/*.json; do
    [ -f "$f" ] || continue
    echo "Importing: $(basename $f)"
    RESPONSE=$(curl -s -X POST "$GRAFANA_URL/api/dashboards/db" \
        -H "Content-Type: application/json" \
        -d "@$f")
    STATUS=$(echo "$RESPONSE" | grep -o '"status"[^,]*' | cut -d'"' -f4)
    [ "$STATUS" = "success" ] && echo "  -> OK" || echo "  -> FAIL: $RESPONSE"
done
echo "Dashboard import complete"
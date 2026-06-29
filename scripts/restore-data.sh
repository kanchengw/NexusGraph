#!/bin/bash
# NexusGraph Data Restore Script
# Usage: bash scripts/restore-data.sh <backup_file>

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: bash scripts/restore-data.sh <backup_file>"
    echo "Example: bash scripts/restore-data.sh backups/nexusgraph-backup-20260629_120000.tar.gz"
    exit 1
fi

BACKUP_FILE="$1"
if [ ! -f "${BACKUP_FILE}" ]; then
    echo "Error: Backup file not found: ${BACKUP_FILE}"
    exit 1
fi

echo "=== NexusGraph Restore ==="
echo "Backup file: ${BACKUP_FILE}"

# Extract
TMP_DIR=$(mktemp -d)
tar xzf "${BACKUP_FILE}" -C "${TMP_DIR}"
EXTRACTED_DIR=$(find "${TMP_DIR}" -maxdepth 1 -type d | tail -1)
echo "Extracted to: ${EXTRACTED_DIR}"

# 1. Restore Neo4j
if docker ps --format "{{.Names}}" | grep -q "graphrag-neo4j"; then
    echo "[1/3] Restoring Neo4j data..."
    NEO4J_FILE="${EXTRACTED_DIR}/neo4j-data.json"
    if [ -f "${NEO4J_FILE}" ]; then
        docker cp "${NEO4J_FILE}" graphrag-neo4j:/data/
        echo "  -> Neo4j data restored. Run in Neo4j Browser: CALL apoc.import.json('file:///data/neo4j-data.json')"
    else
        echo "  -> No Neo4j backup found, skipping"
    fi
else
    echo "[1/3] Neo4j container not running, skipping"
fi

# 2. Restore PostgreSQL
if docker ps --format "{{.Names}}" | grep -q "graphrag-pg"; then
    echo "[2/3] Restoring PostgreSQL..."
    PG_FILE="${EXTRACTED_DIR}/nexusgraph_pg.dump"
    if [ -f "${PG_FILE}" ]; then
        docker cp "${PG_FILE}" graphrag-pg:/tmp/
        docker exec graphrag-pg pg_restore -U myuser -d mydb --clean --if-exists /tmp/nexusgraph_pg.dump
        echo "  -> PostgreSQL restored"
    else
        echo "  -> No PostgreSQL backup found, skipping"
    fi
else
    echo "[2/3] PostgreSQL container not running, skipping"
fi

# 3. Restore config files
echo "[3/3] Restoring configuration files..."
if ls "${EXTRACTED_DIR}"/.env.* 1>/dev/null 2>&1; then
    cp "${EXTRACTED_DIR}"/.env.* ./ 2>/dev/null || true
    echo "  -> Config files restored"
fi

# Cleanup
rm -rf "${TMP_DIR}"
echo ""
echo "Restore complete. Restart containers: docker compose --profile online up -d"

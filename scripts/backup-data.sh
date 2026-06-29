#!/bin/bash
# NexusGraph Data Backup Script
# Usage: bash scripts/backup-data.sh [output_dir]

set -euo pipefail

OUTPUT_DIR="${1:-./backups}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="${OUTPUT_DIR}/${TIMESTAMP}"

mkdir -p "${BACKUP_DIR}"

echo "=== NexusGraph Backup: ${TIMESTAMP} ==="

# 1. Neo4j dump (requires running container)
if docker ps --format "{{.Names}}" | grep -q "graphrag-neo4j"; then
    echo "[1/3] Exporting Neo4j data..."
    docker exec graphrag-neo4j cypher-shell -u neo4j -p "${NEO4J_PASSWORD:-neo4jpassword}" \
        "CALL apoc.export.json.all('neo4j-data.json', {useTypes: true})"
    docker cp graphrag-neo4j:/var/lib/neo4j/neo4j-data.json "${BACKUP_DIR}/" 2>/dev/null || \
    docker cp graphrag-neo4j:/data/neo4j-data.json "${BACKUP_DIR}/" 2>/dev/null || true
    echo "  -> Neo4j export saved"
else
    echo "[1/3] Neo4j container not running, skipping"
fi

# 2. PostgreSQL dump (requires running container)
if docker ps --format "{{.Names}}" | grep -q "graphrag-pg"; then
    echo "[2/3] Dumping PostgreSQL..."
    docker exec graphrag-pg pg_dump -U myuser -d mydb --format=custom -f /tmp/nexusgraph_pg.dump
    docker cp graphrag-pg:/tmp/nexusgraph_pg.dump "${BACKUP_DIR}/"
    echo "  -> PostgreSQL dump saved"
else
    echo "[2/3] PostgreSQL container not running, skipping"
fi

# 3. Configuration backup
echo "[3/3] Backing up configuration files..."
cp .env.* "${BACKUP_DIR}/" 2>/dev/null || true
cp docker-compose.yml "${BACKUP_DIR}/"
cp prometheus/prometheus.yml "${BACKUP_DIR}/" 2>/dev/null || true
echo "  -> Config backed up"

# Package
cd "${OUTPUT_DIR}"
tar czf "nexusgraph-backup-${TIMESTAMP}.tar.gz" "${TIMESTAMP}"
rm -rf "${TIMESTAMP}"
cd - > /dev/null

echo ""
echo "Backup complete: ${OUTPUT_DIR}/nexusgraph-backup-${TIMESTAMP}.tar.gz"

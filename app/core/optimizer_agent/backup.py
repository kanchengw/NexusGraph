"""Neo4j backup/restore for optimization rollback."""

from __future__ import annotations
import asyncio
import os
from datetime import datetime
from app.core.logging import logger
from app.core.config import settings

BACKUP_DIR = "/data/backups"


async def create_backup(tag: str = "") -> str:
    """Create a Neo4j database dump before applying optimization.

    Returns:
        Path to the backup dump file.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{tag}" if tag else ""
    filename = f"flywheel_{ts}{suffix}.dump"
    container = "graphrag-neo4j"

    cmd = [
        "docker", "exec", container,
        "neo4j-admin", "database", "dump", "neo4j",
        f"--to-path={BACKUP_DIR}"
    ]

    logger.info("backup_starting", container=container, filename=filename)
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await proc.communicate()

    if proc.returncode != 0:
        logger.error("backup_failed", error=stderr.decode())
        raise RuntimeError(f"Neo4j backup failed: {stderr.decode()}")

    backup_path = f"{BACKUP_DIR}/{filename}"
    logger.info("backup_complete", path=backup_path)
    return backup_path


async def restore_backup(backup_path: str) -> None:
    """Restore Neo4j from a backup dump.

    Args:
        backup_path: Path to the backup dump file (inside container).
    """
    container = "graphrag-neo4j"

    load_cmd = [
        "docker", "exec", container,
        "neo4j-admin", "database", "load", "neo4j",
        f"--from-path={backup_path}",
        "--overwrite-destination=true"
    ]

    logger.info("restore_starting", backup=backup_path)
    proc = await asyncio.create_subprocess_exec(
        *load_cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await proc.communicate()

    if proc.returncode != 0:
        logger.error("restore_failed", error=stderr.decode())
        raise RuntimeError(f"Neo4j restore failed: {stderr.decode()}")

    restart_cmd = ["docker", "restart", container]
    proc = await asyncio.create_subprocess_exec(*restart_cmd)
    await proc.communicate()

    logger.info("restore_complete", backup=backup_path)


async def cleanup_backup(backup_path: str) -> None:
    """Remove a backup file after successful verification."""
    container = "graphrag-neo4j"
    rm_cmd = ["docker", "exec", container, "rm", "-f", backup_path]
    proc = await asyncio.create_subprocess_exec(*rm_cmd)
    await proc.communicate()
    logger.info("backup_cleaned", path=backup_path)

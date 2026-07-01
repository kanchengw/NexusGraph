"""Offline Admin Server — standalone FastAPI app on port 8001.
Usage:
    python run_offline_server.py
    # Then open http://localhost:8001/admin
"""
from __future__ import annotations
import os
import asyncio
import selectors

os.environ.setdefault("APP_ENV", "development")
os.environ["HF_DATASETS_OFFLINE"] = "1"
asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from app.core.offline_admin.router import router
from app.models.database import init_db

app = FastAPI(title="NexusGraph Offline Admin", version="1.0.0")

# Init PipelineSession table on startup
@app.on_event("startup")
def startup():
    init_db()

# Mount admin API
app.include_router(router)

# Serve admin UI
@app.get("/admin", include_in_schema=False)
async def admin_ui():
    return FileResponse("app/static/admin/index.html")

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "offline-admin"}

if __name__ == "__main__":
    uvicorn.run("run_offline_server:app", host="0.0.0.0", port=8001, reload=True)

"""Feedback API endpoints for data flywheel."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sqlmodel import Session, SQLModel, create_engine

from app.core.logging import logger
from app.models.feedback import Feedback

router = APIRouter(prefix="/graphrag/feedback", tags=["GraphRAG"])

# Use localhost directly (not Docker internal hostname)
_FEEDBACK_DB_URL = "postgresql://myuser:mypassword@localhost:5432/mydb"
_engine = create_engine(_FEEDBACK_DB_URL, pool_pre_ping=True)


class FeedbackRequest(BaseModel):
    session_id: str = Field(..., description="Session identifier")
    query: str = Field(..., description="User query")
    response: str = Field(..., description="GraphRAG response")
    rating: int = Field(..., ge=0, le=5, description="User rating (0-5)")
    correction: str | None = Field(default=None, description="User correction text")


class FeedbackResponse(BaseModel):
    id: int
    status: str = "stored"


@router.post("", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest) -> FeedbackResponse:
    try:
        with Session(_engine) as session:
            fb = Feedback(
                session_id=request.session_id,
                query=request.query,
                response=request.response,
                rating=request.rating,
                correction=request.correction,
            )
            session.add(fb)
            session.commit()
            session.refresh(fb)
            logger.info("feedback_stored", feedback_id=fb.id, rating=request.rating)
            return FeedbackResponse(id=fb.id)
    except Exception as e:
        logger.error("feedback_store_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_feedback_stats():
    try:
        with Session(_engine) as session:
            total = session.query(Feedback).count()
            if total == 0:
                return {"total": 0, "avg_rating": 0, "with_correction": 0}
            avg = session.query(Feedback.rating).all()
            avg_rating = sum(r[0] for r in avg) / len(avg) if avg else 0
            with_correction = session.query(Feedback).filter(Feedback.correction.isnot(None)).count()
            return {"total": total, "avg_rating": round(avg_rating, 2), "with_correction": with_correction}
    except Exception as e:
        logger.error("feedback_stats_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

"""Observability module for the application."""

from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

from app.core.config import settings
from app.core.logging import logger

# Singleton Langfuse instance
_langfuse: Langfuse | None = None


def langfuse_init():
    """Initialize Langfuse."""
    global _langfuse
    if not settings.LANGFUSE_TRACING_ENABLED:
        logger.debug("langfuse_tracing_disabled")
        return

    _langfuse = Langfuse(
        tracing_enabled=settings.LANGFUSE_TRACING_ENABLED,
        public_key=settings.LANGFUSE_PUBLIC_KEY,
        secret_key=settings.LANGFUSE_SECRET_KEY,
        host=settings.LANGFUSE_HOST,
        environment=settings.ENVIRONMENT.value,
        debug=settings.DEBUG,
    )

    try:
        if _langfuse.auth_check():
            logger.debug("langfuse_auth_success")
        else:
            logger.warning("langfuse_auth_failure")
    except Exception:
        logger.exception("langfuse_auth_check_failed")


def get_langfuse() -> Langfuse | None:
    """Get the singleton Langfuse instance."""
    return _langfuse


def get_langfuse_callback_handler() -> CallbackHandler:
    """Create a Langfuse CallbackHandler for tracking LLM interactions."""
    return CallbackHandler()


langfuse_callback_handler = get_langfuse_callback_handler()



def langfuse_flush() -> None:
    """Flush pending Langfuse traces before shutdown."""
    global _langfuse
    if _langfuse:
        try:
            _langfuse.shutdown()
            logger.debug("langfuse_traces_flushed")
        except Exception as e:
            logger.warning("langfuse_flush_failed", error=str(e))

def langfuse_force_flush() -> None:
    """Force-flush pending Langfuse traces immediately (non-destructive, keeps SDK running)."""
    global _langfuse
    if _langfuse:
        try:
            _langfuse.flush()
        except Exception as e:
            logger.warning("langfuse_force_flush_failed", error=str(e))


def langfuse_start_trace(lf, trace_id: str, name: str, question: str):
    """Start a trace-level observation using Langfuse 4.x SDK API (start_observation with as_type='chain').
    
    Returns: (trace_observation, trace_id, succeeded)
    - trace_observation: the LangfuseSpan object to .end() later, or None
    - trace_id: the trace ID (same as input)
    - succeeded: bool
    """
    if not lf:
        return None, trace_id, False
    try:
        from langfuse.types import TraceContext
        obs = lf.start_observation(
            trace_context=TraceContext(trace_id=trace_id),
            name=name,
            as_type="chain",
            input=question,
        )
        return obs, trace_id, True
    except Exception:
        return None, trace_id, False


def langfuse_span(lf, trace_id: str, name: str, question: str, output: str):
    """Create a span within a trace using Langfuse 4.x SDK API."""
    if not lf or not trace_id:
        return
    try:
        from langfuse.types import TraceContext
        span = lf.start_observation(
            trace_context=TraceContext(trace_id=trace_id),
            name=name,
            as_type="retriever",
            input=question,
            output=output,
        )
        span.end()
    except Exception:
        pass


def langfuse_end_trace(trace_obs, output: str = "", metadata: dict = None, status: str = None):
    """End a trace-level observation with final output and metadata."""
    if trace_obs is None:
        return
    try:
        kwargs = {"output": output}
        if metadata:
            kwargs["metadata"] = metadata
        if status:
            kwargs["level"] = "ERROR" if status == "error" else "DEFAULT"
        trace_obs.end(**kwargs)
    except Exception:
        pass

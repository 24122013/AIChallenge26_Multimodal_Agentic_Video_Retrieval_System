"""Canonical application-level pipelines."""

__all__ = ["Candidate", "OnlinePipeline", "OnlinePipelineConfig"]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(name)
    from backend.app.pipelines import online_pipeline

    return getattr(online_pipeline, name)


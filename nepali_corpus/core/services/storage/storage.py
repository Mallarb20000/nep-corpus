"""Storage service abstractions for Nepali Corpus."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict


class StorageService(BaseModel, ABC):
    """Abstract base class for storage services."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the storage backend."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close the storage backend."""
        ...

    @abstractmethod
    def create_session(self) -> "StorageSession":
        """Create a scoped storage session."""
        ...


class StorageSession(BaseModel, ABC):
    """Abstract base class for storage sessions."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    async def store_training_document(self, doc) -> str:
        """Store a single training document."""
        ...

    @abstractmethod
    async def store_training_documents(self, docs) -> int:
        """Store multiple training documents."""
        ...

    @abstractmethod
    async def list_recent_documents(self, limit: int = 50):
        """List recent documents."""
        ...

    @abstractmethod
    async def get_stats(self) -> dict:
        """Aggregate corpus statistics."""
        ...

    @abstractmethod
    async def seen_url(self, url: str) -> bool:
        """Return True if URL has been seen before."""
        ...

    @abstractmethod
    async def mark_url(self, url: str) -> None:
        """Mark URL as seen."""
        ...

    @abstractmethod
    async def count_urls(self) -> int:
        """Return count of seen URLs."""
        ...

    # --- Pipeline run tracking (optional; default no-ops for backward compat) ---

    async def create_pipeline_run(
        self,
        run_id: str,
        sources: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
        output_dir: Optional[str] = None,
    ) -> int:
        """Create a pipeline run record. Returns the DB id."""
        return 0

    async def update_pipeline_run(self, run_id: str, **kwargs: Any) -> None:
        """Update a pipeline run record."""
        pass

    async def create_pipeline_job(
        self,
        pipeline_run_id: int,
        job_type: str,
        source_id: str,
        source_name: Optional[str] = None,
        category: Optional[str] = None,
        scraper_class: Optional[str] = None,
    ) -> int:
        """Create a pipeline job record. Returns the DB id."""
        return 0

    async def update_pipeline_job(self, job_id: int, **kwargs: Any) -> None:
        """Update a pipeline job record."""
        pass

    async def get_pending_jobs(self, run_id: str, job_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get pending/interrupted jobs for a run, optionally filtered by job_type."""
        return []

    async def get_run_status(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get the full status of a pipeline run including job counts."""
        return None

    async def list_runs(self, limit: int = 20) -> List[Dict[str, Any]]:
        """List recent pipeline runs."""
        return []


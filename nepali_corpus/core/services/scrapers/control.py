from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from pydantic import ConfigDict

from nepali_corpus.core.models import RawRecord, TrainingDocument
from nepali_corpus.core.models.base import BaseEntity
from nepali_corpus.core.services.scrapers.pdf import PdfJob, extract_pdfs, HAS_PYMUPDF
from nepali_corpus.core.services.scrapers import (
    dao_scraper,
    ekantipur_scraper,
    govt_scraper,
    news_rss_scraper,
    regulatory_scraper,
)
from nepali_corpus.core.services.scrapers.registry import load_registry
from nepali_corpus.core.services.storage.env_storage import EnvStorageService
from nepali_corpus.core.utils import JsonlWriter
from nepali_corpus.core.utils.content_types import identify_content_type
from nepali_corpus.pipeline.runner import enrich_records, to_training_docs

logger = logging.getLogger("nepali_corpus.scrapers.control")


class ScrapeJob(BaseEntity):
    """A runnable scrape job dispatched by the coordinator.

    Inherits from:
        BaseEntity – common model config
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    category: str
    func: Callable[[], List[RawRecord]]
    scraper_class: str = "generic"


class ScrapeState:
    def __init__(self) -> None:
        self.running = False
        self.paused = False
        self.urls_crawled = 0
        self.urls_failed = 0
        self.docs_saved = 0
        self.pdf_saved = 0
        self.start_time: Optional[float] = None
        self.current_sources: List[str] = []
        self.errors: List[str] = []
        self.source_stats: Dict[str, Dict[str, int]] = {}

    def reset(self) -> None:
        self.running = False
        self.paused = False
        self.urls_crawled = 0
        self.urls_failed = 0
        self.docs_saved = 0
        self.pdf_saved = 0
        self.start_time = None
        self.current_sources = []
        self.errors = []
        self.source_stats = {}

    def record_source(self, source_id: str, crawled: int = 0, saved: int = 0, failed: int = 0) -> None:
        stats = self.source_stats.setdefault(source_id, {"crawled": 0, "saved": 0, "failed": 0})
        stats["crawled"] += crawled
        stats["saved"] += saved
        stats["failed"] += failed

    def add_error(self, message: str) -> None:
        self.errors.append(message)
        if len(self.errors) > 200:
            self.errors.pop(0)

    def speed_urls_per_min(self) -> float:
        if not self.start_time:
            return 0.0
        elapsed = (time.time() - self.start_time) / 60
        return round(self.urls_crawled / elapsed, 1) if elapsed > 0 else 0.0

    def elapsed_str(self) -> str:
        if not self.start_time:
            return "00:00:00"
        sec = int(time.time() - self.start_time)
        h, m, s = sec // 3600, (sec % 3600) // 60, sec % 60
        return f"{h:02d}:{m:02d}:{s:02d}"

    def to_dict(self) -> dict:
        return {
            "running": self.running,
            "paused": self.paused,
            "urls_crawled": self.urls_crawled,
            "urls_failed": self.urls_failed,
            "docs_saved": self.docs_saved,
            "pdf_saved": self.pdf_saved,
            "speed": self.speed_urls_per_min(),
            "elapsed": self.elapsed_str(),
            "current_sources": self.current_sources[-5:],
            "recent_errors": self.errors[-10:],
            "source_stats": self.source_stats,
        }


class ScrapeCoordinator:
    def __init__(self, storage: EnvStorageService) -> None:
        self._storage = storage
        self.state = ScrapeState()
        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        self._shutdown_event = asyncio.Event()

        # Run tracking state
        self._run_id: Optional[str] = None
        self._db_run_id: Optional[int] = None
        self._job_db_ids: Dict[str, int] = {}  # job.name -> DB pipeline_jobs.id
        self._log_handler: Optional[logging.Handler] = None
        self._log_handler: Optional[logging.Handler] = None

    def is_running(self) -> bool:
        return self.state.running

    def request_shutdown(self) -> None:
        """Signal graceful shutdown (from signal handler)."""
        self._shutdown_event.set()
        self._stop_event.set()

    async def start(
        self,
        *,
        workers: int = 4,
        max_pages: Optional[int] = None,
        categories: Optional[List[str]] = None,
        pdf_enabled: bool = False,
        gzip_output: bool = False,
        output_path: str = "data/raw/raw.jsonl",
        pdf_output_dir: str = "data/pdfs",
        govt_registry_path: Optional[str] = None,
        govt_registry_groups: Optional[List[str]] = None,
        run_id: Optional[str] = None,
        output_dir: Optional[str] = None,
        log_file: Optional[str] = None,
    ) -> None:
        if self._task and not self._task.done():
            raise RuntimeError("Scraper already running")
        self._stop_event = asyncio.Event()
        self._shutdown_event = asyncio.Event()
        self.state.reset()
        self.state.running = True
        self.state.start_time = time.time()

        # Generate run_id if not provided
        self._run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")

        self._task = asyncio.create_task(
            self._run(
                workers=workers,
                max_pages=max_pages,
                categories=categories,
                pdf_enabled=pdf_enabled,
                gzip_output=gzip_output,
                output_path=output_path,
                pdf_output_dir=pdf_output_dir,
                govt_registry_path=govt_registry_path,
                govt_registry_groups=govt_registry_groups,
                output_dir=output_dir,
                log_file=log_file,
            )
        )

    async def stop(self) -> None:
        self._stop_event.set()
        self.state.running = False
        self.state.paused = False
        if self._task:
            await asyncio.sleep(0)

    def pause(self) -> None:
        self.state.paused = True

    def resume_paused(self) -> None:
        self.state.paused = False

    async def resume_run(
        self,
        run_id: str,
        *,
        workers: int = 4,
        max_pages: Optional[int] = None,
        pdf_enabled: bool = False,
        gzip_output: bool = False,
        output_path: str = "data/raw/raw.jsonl",
        pdf_output_dir: str = "data/pdfs",
        govt_registry_path: Optional[str] = None,
        govt_registry_groups: Optional[List[str]] = None,
        output_dir: Optional[str] = None,
        log_file: Optional[str] = None,
    ) -> None:
        """Resume an interrupted run by re-dispatching pending jobs."""
        if self._task and not self._task.done():
            raise RuntimeError("Scraper already running")
        self._stop_event = asyncio.Event()
        self._shutdown_event = asyncio.Event()
        self.state.reset()
        self.state.running = True
        self.state.start_time = time.time()
        self._run_id = run_id

        self._task = asyncio.create_task(
            self._resume(
                run_id=run_id,
                workers=workers,
                max_pages=max_pages,
                pdf_enabled=pdf_enabled,
                gzip_output=gzip_output,
                output_path=output_path,
                pdf_output_dir=pdf_output_dir,
                govt_registry_path=govt_registry_path,
                govt_registry_groups=govt_registry_groups,
                output_dir=output_dir,
                log_file=log_file,
            )
        )

    def _build_jobs(
        self,
        categories: Optional[List[str]],
        max_pages: Optional[int],
        govt_registry_path: Optional[str],
        govt_registry_groups: Optional[List[str]],
    ) -> List[ScrapeJob]:
        """Build scrape job list from selected categories."""
        categories = categories or ["Gov", "News"]
        selected = {c.lower() for c in categories}

        registry_path = govt_registry_path
        if not registry_path:
            default = os.path.join("sources", "govt_sources_registry.yaml")
            if os.path.exists(default):
                registry_path = default

        registry_entries = None
        if registry_path:
            registry_entries = load_registry(registry_path, groups=govt_registry_groups)

        jobs: List[ScrapeJob] = []

        # --- Gov Category ---
        if "gov" in selected or "government" in selected:
            if registry_entries:
                for entry in registry_entries:
                    if entry.scraper_class == "ministry_generic":
                        from nepali_corpus.core.services.scrapers.govt_scraper import (
                            MinistryScraper,
                            post_to_raw,
                            MinistryConfig,
                        )
                        cfg = MinistryConfig(
                            source_id=entry.source_id,
                            name=entry.name or entry.source_id,
                            name_ne=entry.name_ne or entry.source_id,
                            base_url=entry.base_url,
                            endpoints=entry.endpoints,
                            priority=entry.priority,
                        )
                        jobs.append(
                            ScrapeJob(
                                name=f"gov:{entry.source_id}",
                                category="Gov",
                                scraper_class="ministry_generic",
                                func=lambda c=cfg: [
                                    post_to_raw(p)
                                    for posts in MinistryScraper(c)
                                    .scrape_all(max_pages_per_endpoint=max_pages or 3)
                                    .values()
                                    for p in posts
                                ],
                            )
                        )
                    elif entry.scraper_class == "regulatory":
                        from nepali_corpus.core.services.scrapers.regulatory_scraper import (
                            RegulatoryScraper,
                        )
                        jobs.append(
                            ScrapeJob(
                                name=f"reg:{entry.source_id}",
                                category="Gov",
                                scraper_class="regulatory",
                                func=lambda e=entry: RegulatoryScraper(e).scrape(
                                    pages=max_pages or 1
                                ),
                            )
                        )
                    else:
                        # Fallback for constitutional, parliament, etc.
                        from nepali_corpus.core.services.scrapers.regulatory_scraper import (
                            RegulatoryScraper,
                        )
                        jobs.append(
                            ScrapeJob(
                                name=f"gov:{entry.source_id}",
                                category="Gov",
                                scraper_class=entry.scraper_class or "generic",
                                func=lambda e=entry: RegulatoryScraper(e).scrape(
                                    pages=max_pages or 1
                                ),
                            )
                        )

            if not govt_registry_groups:
                jobs.append(
                    ScrapeJob(
                        name="dao",
                        category="Gov",
                        scraper_class="dao",
                        func=lambda: dao_scraper.fetch_raw_records(pages=max_pages or 2),
                    )
                )

        # --- News Category ---
        if "news" in selected:
            jobs.append(
                ScrapeJob(
                    name="news_rss",
                    category="News",
                    scraper_class="rss",
                    func=lambda: news_rss_scraper.fetch_raw_records(),
                )
            )
            jobs.append(
                ScrapeJob(
                    name="ekantipur",
                    category="News",
                    scraper_class="ekantipur",
                    func=lambda: ekantipur_scraper.fetch_raw_records(max_pages=max_pages or 3),
                )
            )

        # --- Social Category ---
        if "social" in selected:
            from .social_scraper import NitterScraper
            from ..dashboard.sources import load_social_sources

            social_sources = load_social_sources()
            scraper = NitterScraper()

            for s in social_sources:
                sid = s["id"]
                if sid.startswith("social:"):
                    username = sid.split(":", 1)[1]
                    jobs.append(
                        ScrapeJob(
                            name=f"social:{username}",
                            category="Social",
                            scraper_class="nitter",
                            func=lambda u=username: scraper.fetch_user_tweets(
                                u, max_pages=max_pages or 1
                            ),
                        )
                    )
                elif sid.startswith("hashtag:") or sid.startswith("search:"):
                    query = s["name"]
                    jobs.append(
                        ScrapeJob(
                            name=f"social_search:{sid}",
                            category="Social",
                            scraper_class="nitter",
                            func=lambda q=query: scraper.fetch_search_tweets(
                                q, max_pages=max_pages or 1
                            ),
                        )
                    )

        return jobs

    async def _run(
        self,
        *,
        workers: int,
        max_pages: Optional[int],
        categories: Optional[List[str]],
        pdf_enabled: bool,
        gzip_output: bool,
        output_path: str,
        pdf_output_dir: str,
        govt_registry_path: Optional[str],
        govt_registry_groups: Optional[List[str]],
        output_dir: Optional[str],
        log_file: Optional[str] = None,
    ) -> None:
        if log_file:
            self._setup_file_logging(log_file)

        session = self._storage.create_session()

        jobs = self._build_jobs(categories, max_pages, govt_registry_path, govt_registry_groups)

        if not jobs:
            self.state.add_error("No matching scrapers for selected categories")
            self.state.running = False
            return

        # --- Create pipeline run in DB ---
        config = {
            "workers": workers,
            "max_pages": max_pages,
            "categories": categories,
            "pdf_enabled": pdf_enabled,
            "gzip_output": gzip_output,
        }
        try:
            self._db_run_id = await session.create_pipeline_run(
                run_id=self._run_id,
                sources=[j.name for j in jobs],
                categories=categories,
                config=config,
                output_dir=output_dir or os.path.dirname(output_path),
            )
            await session.update_pipeline_run(self._run_id, total_jobs=len(jobs))
        except Exception as e:
            logger.warning("Failed to create pipeline run record: %s", e)
            self._db_run_id = 0

        # --- Create pipeline jobs in DB ---
        for job in jobs:
            try:
                db_id = await session.create_pipeline_job(
                    pipeline_run_id=self._db_run_id,
                    job_type="scrape",
                    source_id=job.name,
                    source_name=job.name,
                    category=job.category,
                    scraper_class=job.scraper_class,
                )
                self._job_db_ids[job.name] = db_id
            except Exception as e:
                logger.warning("Failed to create job record for %s: %s", job.name, e)

        # --- Execute jobs (Scrape) ---
        writer = JsonlWriter(output_path, gzip_output=gzip_output, append=True)
        try:
            await self._execute_jobs(
                jobs=jobs,
                session=session,
                writer=writer,
                workers=workers,
                pdf_enabled=pdf_enabled,
                pdf_output_dir=pdf_output_dir,
            )
        finally:
            writer.flush()
            writer.close()

        # --- Enrichment Phase ---
        if not self._stop_event.is_set() and self.state.urls_crawled > 0:
            await self._run_enrichment(session, output_path, gzip_output, workers)

        # --- Finalize run ---
        await self._finalize_run(session)
        self._cleanup_file_logging()

    async def _resume(
        self,
        *,
        run_id: str,
        workers: int,
        max_pages: Optional[int],
        pdf_enabled: bool,
        gzip_output: bool,
        output_path: str,
        pdf_output_dir: str,
        govt_registry_path: Optional[str],
        govt_registry_groups: Optional[List[str]],
        output_dir: Optional[str],
        log_file: Optional[str] = None,
    ) -> None:
        """Resume an interrupted run."""
        if log_file:
            self._setup_file_logging(log_file)

        session = self._storage.create_session()

        # Get pending jobs from DB
        pending = await session.get_pending_jobs(run_id, job_type="scrape")
        if not pending:
            logger.info("No pending jobs to resume for run %s", run_id)
            self.state.running = False
            return

        # Get the run's DB id
        run_status = await session.get_run_status(run_id)
        if run_status:
            self._db_run_id = run_status["id"]

        await session.update_pipeline_run(run_id, status="running")

        # Build a name -> func map from all possible jobs
        all_jobs = self._build_jobs(None, max_pages, govt_registry_path, govt_registry_groups)
        func_map = {j.name: j.func for j in all_jobs}

        # Rebuild job list from pending DB records
        jobs: List[ScrapeJob] = []
        for pj in pending:
            source_id = pj["source_id"]
            func = func_map.get(source_id)
            if func is None:
                logger.warning("No scraper func found for source %s, skipping", source_id)
                continue
            self._job_db_ids[source_id] = pj["id"]
            # Increment attempt number
            await session.update_pipeline_job(pj["id"], attempt_number=pj["attempt_number"] + 1)
            jobs.append(
                ScrapeJob(
                    name=source_id,
                    category=pj.get("category", ""),
                    scraper_class=pj.get("scraper_class", "generic"),
                    func=func,
                )
            )

        if not jobs:
            logger.info("No resumable jobs found for run %s", run_id)
            self.state.running = False
            return

        writer = JsonlWriter(output_path, gzip_output=gzip_output, append=True)
        try:
            await self._execute_jobs(
                jobs=jobs,
                session=session,
                writer=writer,
                workers=workers,
                pdf_enabled=pdf_enabled,
                pdf_output_dir=pdf_output_dir,
            )
        finally:
            writer.flush()
            writer.close()

        # --- Enrichment Phase (for any items that need it) ---
        if not self._stop_event.is_set():
            await self._run_enrichment(session, output_path, gzip_output, workers)

        await self._finalize_run(session)

    async def _execute_jobs(
        self,
        *,
        jobs: List[ScrapeJob],
        session: Any,
        writer: JsonlWriter,
        workers: int,
        pdf_enabled: bool,
        pdf_output_dir: str,
    ) -> None:
        """Execute scrape jobs in a thread pool and write results."""
        loop = asyncio.get_running_loop()
        pending: List[asyncio.Future] = []

        with ThreadPoolExecutor(max_workers=workers) as pool:
            for job in jobs:
                # Mark job as running
                db_id = self._job_db_ids.get(job.name)
                if db_id:
                    try:
                        await session.update_pipeline_job(
                            db_id,
                            status="running",
                            started_at=datetime.now(timezone.utc),
                        )
                    except Exception:
                        pass

                def _run_job(j=job):
                    try:
                        return j, j.func()
                    except Exception as e:
                        return j, e

                future = loop.run_in_executor(pool, _run_job)
                pending.append(future)

            pdf_jobs: List[PdfJob] = []

            for coro in asyncio.as_completed(pending):
                if self._stop_event.is_set():
                    break
                while self.state.paused and not self._stop_event.is_set():
                    await asyncio.sleep(0.5)

                job_start = time.time()
                try:
                    job, records = await coro
                    if isinstance(records, Exception):
                        raise records
                except Exception as exc:
                    job_name = "unknown"
                    try:
                        job_name = job.name
                    except Exception:
                        pass

                    msg = f"{job_name} failed: {exc}"
                    logger.error(msg)
                    self.state.urls_failed += 1
                    self.state.record_source(job_name, failed=1)
                    self.state.add_error(msg)

                    # Update job status to failed in DB
                    db_id = self._job_db_ids.get(job_name)
                    if db_id:
                        try:
                            await session.update_pipeline_job(
                                db_id,
                                status="failed",
                                error_message=str(exc)[:500],
                                completed_at=datetime.now(timezone.utc),
                                duration_ms=(time.time() - job_start) * 1000,
                            )
                        except Exception:
                            pass
                    continue

                self.state.current_sources.append(job.name)
                if len(self.state.current_sources) > 20:
                    self.state.current_sources.pop(0)

                if not records:
                    # Job completed with no records
                    db_id = self._job_db_ids.get(job.name)
                    if db_id:
                        try:
                            await session.update_pipeline_job(
                                db_id,
                                status="completed",
                                records_crawled=0,
                                records_saved=0,
                                completed_at=datetime.now(timezone.utc),
                                duration_ms=(time.time() - job_start) * 1000,
                            )
                        except Exception:
                            pass
                    continue

                seen = 0
                saved_count = 0
                last_url = None
                saved_records = []
                for record in records:
                    seen += 1
                    self.state.record_source(record.source_id, crawled=1)
                    try:
                        if await session.seen_url(record.url):
                            continue
                        await session.mark_url(record.url)
                        await session.store_raw_records([record])
                    except Exception as exc:
                        logger.error("Failed to store raw record for %s: %s", record.url, exc)
                    writer.write(record)
                    saved_records.append(record)
                    saved_count += 1
                    last_url = record.url
                    self.state.record_source(record.source_id, saved=1)

                    if pdf_enabled:
                        # If the main URL is a PDF and has no content yet, queue it for extraction
                        if record.content_type == "pdf" and not (record.content or "").strip():
                            pdf_jobs.append(
                                PdfJob(
                                    url=record.url,
                                    source_id=record.source_id,
                                    source_name=record.source_name,
                                    category=record.category,
                                )
                            )

                        urls = record.raw_meta.get("attachment_urls") if record.raw_meta else None
                        if urls:
                            for url in urls:
                                if url:
                                    # Avoid duplicates if it's the same as the main URL
                                    if url == record.url and record.content_type == "pdf":
                                        continue
                                    pdf_jobs.append(
                                        PdfJob(
                                            url=url,
                                            source_id=record.source_id,
                                            source_name=record.source_name,
                                            category=record.category,
                                        )
                                    )

                # Incremental DB sync — training documents
                if saved_records:
                    try:
                        import hashlib

                        training_docs = []
                        for r in saved_records:
                            text = (r.content or "").strip()
                            if len(text) < 100:
                                continue
                                
                            doc_id = hashlib.md5(r.url.encode()).hexdigest()
                            doc = TrainingDocument(
                                id=doc_id,
                                url=r.url,
                                source_id=r.source_id,
                                source_name=r.source_name,
                                language=r.language or "ne",
                                text=text,
                                published_at=r.published_at,
                                date_bs=r.raw_meta.get("date_bs") if r.raw_meta else None,
                                category=r.category,
                                content_type=r.content_type,
                                tags=[],
                            )
                            training_docs.append(doc)

                        if training_docs:
                            await session.store_training_documents(training_docs)
                        self.state.docs_saved += len(training_docs)
                    except Exception as e:
                        logger.error(f"Failed incremental DB sync: {e}")

                self.state.urls_crawled += seen

                # Update job status in DB
                db_id = self._job_db_ids.get(job.name)
                if db_id:
                    try:
                        await session.update_pipeline_job(
                            db_id,
                            status="completed",
                            records_crawled=seen,
                            records_saved=saved_count,
                            last_url=last_url,
                            completed_at=datetime.now(timezone.utc),
                            duration_ms=(time.time() - job_start) * 1000,
                        )
                    except Exception:
                        pass

        # Handle PDF extraction
        if pdf_enabled and pdf_jobs:
            if not HAS_PYMUPDF:
                self.state.add_error("PyMuPDF not installed; PDF extraction skipped")
            else:
                try:
                    pdf_records = await extract_pdfs(
                        pdf_jobs,
                        output_dir=pdf_output_dir,
                        max_workers=max(2, workers),
                        seen_url=session.seen_url,
                        mark_url=session.mark_url,
                    )
                    for rec in pdf_records:
                        writer.write(rec)
                        self.state.record_source(rec.source_id, crawled=1, saved=1)
                        # Store in DB as well
                        try:
                            await session.store_raw_records([rec])
                            
                            # Also store as training document if text is long enough
                            if len((rec.content or "").strip()) >= 100:
                                doc_id = hashlib.md5(rec.url.encode()).hexdigest()
                                doc = TrainingDocument(
                                    id=doc_id,
                                    url=rec.url,
                                    source_id=rec.source_id,
                                    source_name=rec.source_name,
                                    language=rec.language or "ne",
                                    text=(rec.content or "").strip(),
                                    published_at=rec.published_at,
                                    date_bs=rec.raw_meta.get("date_bs") if rec.raw_meta else None,
                                    category=rec.category,
                                    content_type=rec.content_type,
                                    tags=[],
                                )
                                await session.store_training_documents([doc])
                        except Exception as exc:
                            logger.error("Failed to store PDF record/doc in DB: %s", exc)

                    self.state.pdf_saved += len(pdf_records)
                    self.state.urls_crawled += len(pdf_records)
                    self.state.docs_saved += len(pdf_records)
                except Exception as exc:
                    self.state.add_error(f"PDF extraction failed: {exc}")

        # If shutdown was requested, mark in-flight jobs as interrupted
        if self._shutdown_event.is_set():
            for job in jobs:
                db_id = self._job_db_ids.get(job.name)
                if db_id:
                    try:
                        # Only update jobs that are still 'running'
                        await session.update_pipeline_job(db_id, status="interrupted")
                    except Exception:
                        pass

    async def _run_enrichment(
        self,
        session: Any,
        output_path: str,
        gzip_output: bool,
        workers: int,
    ) -> None:
        """Run enrichment on the records produced by this run."""
        from nepali_corpus.pipeline.runner import load_raw_jsonl, save_raw_jsonl

        # Create enrichment job record
        job_id = await session.create_pipeline_job(
            pipeline_run_id=self._db_run_id,
            job_type="enrich",
            source_id="run_enrichment",
            source_name="Full Text Enrichment",
            category="Pipeline",
            scraper_class="enricher",
        )
        
        await session.update_pipeline_job(
            job_id, 
            status="running", 
            started_at=datetime.now(timezone.utc)
        )

        start_time = time.time()
        try:
            # Load the current raw records from the output file
            records = load_raw_jsonl(output_path)
            if not records:
                await session.update_pipeline_job(job_id, status="completed", records_saved=0)
                return

            logger.info("Starting enrichment for %s records...", len(records))
            
            # Run parallel enrichment
            # Note: enrich_records uses a thread pool internally
            enriched_pairs = enrich_records(records, cache_dir="data/html_cache")
            
            enriched_records = []
            for rec, content in enriched_pairs:
                if content:
                    rec.content = content
                enriched_records.append(rec)
            
            # Save enriched data back to raw file (overwriting with full content)
            save_raw_jsonl(enriched_records, output_path, gzip_output=gzip_output)
            
            # Sync to training_documents in DB
            import hashlib

            training_docs = []
            for r in enriched_records:
                # Skip if text is empty or just generic boilerplate (short)
                text = (r.content or "").strip()
                if len(text) < 100:
                    continue
                    
                doc_id = hashlib.md5(r.url.encode()).hexdigest()
                doc = TrainingDocument(
                    id=doc_id,
                    url=r.url,
                    source_id=r.source_id,
                    source_name=r.source_name,
                    language=r.language or "ne",
                    text=text,
                    published_at=r.published_at,
                    date_bs=r.raw_meta.get("date_bs") if r.raw_meta else None,
                    category=r.category,
                    content_type=r.content_type,
                    tags=[],
                )
                training_docs.append(doc)

            if training_docs:
                await session.store_training_documents(training_docs)
                self.state.docs_saved += len(training_docs)
                logger.info("Saved %s training documents with text", len(training_docs))
            else:
                logger.info("No training documents with text to save")
            
            # ALSO update the raw_records table in DB so the CONTENT field is populated
            await session.store_raw_records(enriched_records)
            
            await session.update_pipeline_job(
                job_id,
                status="completed",
                records_saved=len(enriched_records),
                completed_at=datetime.now(timezone.utc),
                duration_ms=(time.time() - start_time) * 1000,
            )
            logger.info("Enrichment complete for %s records", len(enriched_records))

        except Exception as e:
            logger.error("Enrichment failed: %s", e)
            await session.update_pipeline_job(
                job_id,
                status="failed",
                error_message=str(e),
                completed_at=datetime.now(timezone.utc),
            )

    async def _finalize_run(self, session: Any) -> None:
        """Update pipeline run with final counts."""
        if not self._run_id:
            self.state.running = False
            return

        final_status = "interrupted" if self._shutdown_event.is_set() else "completed"

        try:
            await session.update_pipeline_run(
                self._run_id,
                status=final_status,
                completed_jobs=self.state.docs_saved,
                total_records_scraped=self.state.urls_crawled,
                total_records_saved=self.state.docs_saved,
                completed_at=datetime.now(timezone.utc),
            )
        except Exception as e:
            logger.warning("Failed to finalize pipeline run: %s", e)

        self.state.running = False

    def write_checkpoint(self, output_dir: str) -> None:
        """Write a checkpoint.json to the run directory."""
        checkpoint = {
            "run_id": self._run_id,
            "status": "interrupted" if self._shutdown_event.is_set() else "completed",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "urls_crawled": self.state.urls_crawled,
            "docs_saved": self.state.docs_saved,
            "urls_failed": self.state.urls_failed,
            "source_stats": self.state.source_stats,
            "errors": self.state.errors[-20:],
        }
        path = os.path.join(output_dir, "checkpoint.json")
        os.makedirs(output_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, indent=2, ensure_ascii=False)
        logger.info("Checkpoint written to %s", path)

    def _setup_file_logging(self, log_file: str) -> None:
        """Set up logging to a file."""
        try:
            # Ensure directory exists
            log_dir = os.path.dirname(log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)

            # Add file handler to the root logger to capture everything
            root_logger = logging.getLogger()
            handler = logging.FileHandler(log_file, encoding="utf-8", mode='a')
            handler.setFormatter(
                logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
            )
            root_logger.addHandler(handler)
            root_logger.setLevel(logging.INFO)
            self._log_handler = handler
            logger.info("Logging to file: %s", log_file)
        except Exception as e:
            logger.warning("Failed to set up file logging to %s: %s", log_file, e)

    def _cleanup_file_logging(self) -> None:
        """Remove file handler from root logger."""
        if self._log_handler:
            try:
                root_logger = logging.getLogger()
                root_logger.removeHandler(self._log_handler)
                self._log_handler.close()
                self._log_handler = None
            except Exception:
                pass


__all__ = ["ScrapeCoordinator", "ScrapeState"]

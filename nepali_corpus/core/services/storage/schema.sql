CREATE TABLE IF NOT EXISTS training_documents (
    id TEXT PRIMARY KEY,
    url TEXT UNIQUE,
    source_id TEXT,
    source_name TEXT,
    language TEXT,
    text TEXT,
    published_at TEXT,
    date_bs TEXT,
    category TEXT,
    content_type TEXT,
    province TEXT,
    district TEXT,
    tags JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_training_documents_source_id ON training_documents(source_id);
CREATE INDEX IF NOT EXISTS idx_training_documents_language ON training_documents(language);
CREATE INDEX IF NOT EXISTS idx_training_documents_created_at ON training_documents(created_at);

CREATE TABLE IF NOT EXISTS visited_urls (
    url_hash TEXT PRIMARY KEY,
    url TEXT NOT NULL,
    first_seen_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_visited_urls_first_seen_at ON visited_urls(first_seen_at);

CREATE TABLE IF NOT EXISTS raw_records (
    url TEXT PRIMARY KEY,
    source_id TEXT,
    source_name TEXT,
    title TEXT,
    summary TEXT,
    content TEXT,
    language TEXT,
    published_at TEXT,
    date_bs TEXT,
    category TEXT,
    content_type TEXT,
    fetched_at TEXT,
    raw_meta JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_raw_records_source_id ON raw_records(source_id);
CREATE INDEX IF NOT EXISTS idx_raw_records_created_at ON raw_records(created_at);

-- Pipeline run tracking (one row per CLI invocation)
CREATE TABLE IF NOT EXISTS pipeline_runs (
    id SERIAL PRIMARY KEY,
    run_id TEXT NOT NULL UNIQUE,
    sources_requested TEXT[],
    categories TEXT[],
    config_json JSONB DEFAULT '{}',
    total_jobs INTEGER DEFAULT 0,
    completed_jobs INTEGER DEFAULT 0,
    failed_jobs INTEGER DEFAULT 0,
    total_records_scraped INTEGER DEFAULT 0,
    total_records_saved INTEGER DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'pending',
    output_dir TEXT,
    started_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_pipeline_runs_status ON pipeline_runs(status);
CREATE INDEX IF NOT EXISTS idx_pipeline_runs_started_at ON pipeline_runs(started_at);

-- Per-source job within a pipeline run (scrape, enrich, or export)
CREATE TABLE IF NOT EXISTS pipeline_jobs (
    id SERIAL PRIMARY KEY,
    pipeline_run_id INTEGER NOT NULL REFERENCES pipeline_runs(id) ON DELETE CASCADE,
    job_type TEXT NOT NULL DEFAULT 'scrape',
    source_id TEXT NOT NULL,
    source_name TEXT,
    category TEXT,
    scraper_class TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    error_message TEXT,
    records_crawled INTEGER DEFAULT 0,
    records_saved INTEGER DEFAULT 0,
    records_failed INTEGER DEFAULT 0,
    pages_scraped INTEGER DEFAULT 0,
    last_url TEXT,
    attempt_number INTEGER DEFAULT 1,
    max_attempts INTEGER DEFAULT 3,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    duration_ms DOUBLE PRECISION,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_pipeline_jobs_run_id ON pipeline_jobs(pipeline_run_id);
CREATE INDEX IF NOT EXISTS idx_pipeline_jobs_status ON pipeline_jobs(status);
CREATE INDEX IF NOT EXISTS idx_pipeline_jobs_job_type ON pipeline_jobs(job_type);
CREATE INDEX IF NOT EXISTS idx_pipeline_jobs_source_id ON pipeline_jobs(source_id);

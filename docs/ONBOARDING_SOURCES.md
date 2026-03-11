# Onboarding New Sources to Nepali Corpus

This guide explains how to add new data sources (Government, News, Social Media) to the Nepali Corpus pipeline. The system is designed to be highly configurable, minimizing the need to write new Python code for standard sites.

## 1. Government Sources

Government sources are managed in `sources/govt_sources_registry.yaml`. This file dictates what gets scraped, how often, and which scraper to use.

### Adding a Standard Ministry
Most Federal Ministries use a standard template (WordPress-based or similar) with consistent endpoints for notices and press releases. 

To add a new ministry, open `sources/govt_sources_registry.yaml` and add an entry under `federal_ministries`:

```yaml
  - id: new_ministry_id
    name: Ministry of Something
    name_ne: केही मन्त्रालय
    base_url: https://something.gov.np
    scraper_class: ministry_generic
    endpoints:
      press_release_en: /en/page/press-release
      press_release_ne: /page/press-release
      notice_en: /en/page/notice
      notice_ne: /page/notice
    priority: 2
    poll_interval_mins: 120
```

*   **`scraper_class: ministry_generic`**: Tells the pipeline to use the standard pagination scraper.
*   **`endpoints`**: Define the paths to specific categories. The scraper will paginate through them automatically.

### Adding Regulatory or Constitutional Bodies
If a site doesn't fit the generic structure, it usually falls under the `regulatory` scraper, which recursively searches for links containing keywords (like 'notice', 'सूचना', 'press').

```yaml
  - id: new_regulator
    name: Some Regulatory Board
    name_ne: नियामक बोर्ड
    base_url: https://board.gov.np
    scraper_class: regulatory
    priority: 2
    poll_interval_mins: 120
```

### Adding Custom Scrapers
If the site relies heavily on APIs or complex JavaScript for its content, you may need a custom scraper:
1. Create a new scraper in `nepali_corpus/core/services/scrapers/` (e.g., `custom_scraper.py`).
2. Make sure it returns a list of dictionaries mapping to the `RawRecord` schema.
3. Hook it up in `nepali_corpus/core/services/scrapers/control.py` inside the `_build_jobs` method:
   ```python
   elif entry.scraper_class == "custom_scraper_name":
       from .custom_scraper import CustomScraper
       jobs.append(ScrapeJob(
           name=f"gov:{entry.source_id}",
           category="Gov",
           scraper_class="custom_scraper_name",
           func=lambda e=entry: CustomScraper(e).scrape()
       ))
   ```

---

## 2. Social Media Sources

Social media sources (Twitter/X) are scraped without an API using Nitter instances. They are configured in `sources/social_sources.yaml`.

You can track specific accounts, hashtags, or raw text searches. 

### Adding a Tracked Account
Add an entry under `accounts`:
```yaml
  - username: KathmanduPost
    name: The Kathmandu Post
    category: news
```

### Adding a Hashtag
Add an entry under `hashtags` (the # symbol is handled automatically):
```yaml
  - tag: NepalElection
    name: Nepal Election
```

### Adding a Text Search
For broader listening, add an entry under `searches`:
```yaml
  - query: "Nepal budget 2082"
    name: Budget Discussions
```

When you run the `coordinator` with the `Social` category, it will automatically distribute these across the Nitter instances defined at the top of the file.

---

## 3. Adding a Completely New Domain (e.g., Financial/Market Data)

If you are expanding the corpus into a completely new domain, such as financial market data (NEPSE), you shouldn't just append to the existing government or social YAMLs.

Instead, you should establish a new source configuration file and hook it into the overarching pipeline.

### Step 1: Create a New Source Configuration File
Create a new YAML file in the `sources/` directory (e.g., `sources/market_sources.yaml`):

```yaml
# sources/market_sources.yaml
markets:
  - id: nepse
    name: Nepal Stock Exchange
    base_url: https://nepalstock.com
    scraper_class: nepse_scraper
    priority: 1
    poll_interval_mins: 15
```

### Step 2: Create a Specialized Scraper
Certain applications like NEPSE are highly dynamic, use WebSockets, or require API scraping. Create a dedicated scraper such as `nepali_corpus/core/services/scrapers/nepse_scraper.py` that handles this platform's specific structure and yields records conforming to the pipeline's standard `RawRecord` format.

### Step 3: Wire it into the Scrape Coordinator
Finally, you must modify `nepali_corpus/core/services/scrapers/control.py` to recognize this new category (e.g. `'Market'`) and spin up the corresponding jobs.

In the `_build_jobs` method, you would:
1. Load your new YAML file utilizing the existing configuration manager.
2. Add a new block for your category:

```python
        # --- Market Category ---
        if "market" in selected:
            from .nepse_scraper import NepseScraper
            market_entries = load_registry("sources/market_sources.yaml")
            
            for entry in market_entries:
                if entry.scraper_class == "nepse_scraper":
                    jobs.append(
                        ScrapeJob(
                            name=f"market:{entry.source_id}",
                            category="Market",
                            scraper_class="nepse",
                            func=lambda e=entry: NepseScraper(e).fetch()
                        )
                    )
```

Now you can effortlessly invoke the entire pipeline for your new category with:
```bash
python scripts/corpus_cli.py coordinator --categories Market
```

---

## 4. Testing Your New Source

Once configured, verify the source by running the pipeline targeted specifically at the group you updated.

**Test a specific government group (e.g., federal ministries):**
```bash
python scripts/corpus_cli.py coordinator --categories Gov --govt-groups federal_ministries --max-pages 1
```

**Test social media scrapers:**
```bash
python scripts/corpus_cli.py coordinator --categories Social --max-pages 1
```

### Verification Steps
1. Open the dashboard (`http://localhost:8000`).
2. Navigate to **Scraper Logs** to see live traversal.
3. Check the **Dataset Viewer** under the `raw_records` and `training_documents` tables. Ensure the `content` (extracted text) is cleanly parsed and not full of boilerplate navigation text.

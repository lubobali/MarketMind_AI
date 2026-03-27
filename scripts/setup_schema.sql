-- MarketMind AI — Unity Catalog Schema Setup
-- Run this in a Databricks SQL editor or notebook before running any pipelines.
--
-- Prerequisites:
--   - Unity Catalog enabled on your workspace
--   - CREATE SCHEMA permission on the bootcamp_students catalog
--
-- Usage:
--   1. Open a SQL editor in Databricks
--   2. Paste and run this script
--   3. Verify: SHOW TABLES IN bootcamp_students.lubo_marketmind_ai

-- Create schema (idempotent)
CREATE SCHEMA IF NOT EXISTS bootcamp_students.lubo_marketmind_ai
  COMMENT 'MarketMind AI — Stock market streaming + NLP + AI agent capstone';

-- Grant usage (adjust principal as needed for your workspace)
-- GRANT USAGE ON SCHEMA bootcamp_students.lubo_marketmind_ai TO `data@lubobali.com`;
-- GRANT CREATE TABLE ON SCHEMA bootcamp_students.lubo_marketmind_ai TO `data@lubobali.com`;
-- GRANT SELECT ON SCHEMA bootcamp_students.lubo_marketmind_ai TO `data@lubobali.com`;

-- Create the raw news staging table (used by news producer before DLT)
CREATE TABLE IF NOT EXISTS bootcamp_students.lubo_marketmind_ai.raw_stock_news (
  headline STRING,
  summary STRING,
  source STRING,
  url STRING,
  symbols ARRAY<STRING>,
  published_at STRING
)
USING DELTA
COMMENT 'Raw financial news articles — staging table for DLT Bronze ingestion';

-- Verify
SHOW TABLES IN bootcamp_students.lubo_marketmind_ai;

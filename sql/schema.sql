-- ============================================================
-- MDM 主数据管理系统 — PostgreSQL 数据库建表脚本
-- ============================================================
-- 运行方式: psql -U mdm_user -d mdm_db -f sql/schema.sql
-- ============================================================

-- 1. 源数据记录表（CRM / ERP / ECommerce 原始记录）
CREATE TABLE IF NOT EXISTS source_records (
    id              SERIAL PRIMARY KEY,
    record_id       VARCHAR(64)   NOT NULL,
    source          VARCHAR(32)   NOT NULL,  -- CRM, ERP, ECommerce
    entity_type     VARCHAR(16)   NOT NULL,  -- customer, product
    canonical_id    VARCHAR(64),             -- 真值标签（生成数据用）

    -- 客户字段
    company_name    VARCHAR(256),
    region          VARCHAR(64),
    city            VARCHAR(64),
    address         TEXT,
    phone           VARCHAR(48),
    tax_id          VARCHAR(48),
    website         VARCHAR(256),
    email           VARCHAR(128),
    contact_person  VARCHAR(64),

    -- 商品字段
    product_name    VARCHAR(256),
    category        VARCHAR(64),
    brand           VARCHAR(64),
    model           VARCHAR(64),
    sku             VARCHAR(48),
    specification   VARCHAR(128),
    upc             VARCHAR(48),
    price           VARCHAR(32),

    source_created_at VARCHAR(32),
    created_at      TIMESTAMP DEFAULT NOW(),

    UNIQUE (record_id, source, entity_type)
);

CREATE INDEX idx_source_records_source ON source_records(source);
CREATE INDEX idx_source_records_entity ON source_records(entity_type);
CREATE INDEX idx_source_records_canonical ON source_records(canonical_id);
CREATE INDEX idx_source_records_region ON source_records(region);
CREATE INDEX idx_source_records_category ON source_records(category);


-- 2. 黄金客户表
CREATE TABLE IF NOT EXISTS golden_customers (
    id                  SERIAL PRIMARY KEY,
    canonical_customer_id VARCHAR(64) UNIQUE NOT NULL,
    company_name        VARCHAR(256),
    region              VARCHAR(64),
    city                VARCHAR(64),
    address             TEXT,
    phone               VARCHAR(48),
    tax_id              VARCHAR(48),
    website             VARCHAR(256),
    email               VARCHAR(128),
    contact_person      VARCHAR(64),
    source_created_at   VARCHAR(32),
    sources             VARCHAR(128),        -- 来源系统列表，用 ; 分隔
    record_count        INTEGER DEFAULT 1,
    merged_at           TIMESTAMP DEFAULT NOW()
);


-- 3. 黄金商品表
CREATE TABLE IF NOT EXISTS golden_products (
    id                  SERIAL PRIMARY KEY,
    canonical_product_id VARCHAR(64) UNIQUE NOT NULL,
    product_name        VARCHAR(256),
    category            VARCHAR(64),
    brand               VARCHAR(64),
    model               VARCHAR(64),
    sku                 VARCHAR(48),
    specification       VARCHAR(128),
    upc                 VARCHAR(48),
    price               VARCHAR(32),
    source_created_at   VARCHAR(32),
    sources             VARCHAR(128),
    record_count        INTEGER DEFAULT 1,
    merged_at           TIMESTAMP DEFAULT NOW()
);


-- 4. 匹配候选对表
CREATE TABLE IF NOT EXISTS match_candidates (
    id                  SERIAL PRIMARY KEY,
    entity_type         VARCHAR(16)   NOT NULL,
    record_id_left      VARCHAR(64)   NOT NULL,
    record_id_right     VARCHAR(64)   NOT NULL,
    source_left         VARCHAR(32),
    source_right        VARCHAR(32),
    pair_source         VARCHAR(32),         -- 如 CRM-ERP
    match_score         REAL,
    semantic_name_score REAL,                -- Embedding 相似度
    decision_band       VARCHAR(16),         -- auto_merge / review / no_match
    is_true_match       BOOLEAN DEFAULT FALSE,

    -- 审核增强字段
    dify_decision       VARCHAR(16),
    dify_confidence     REAL,
    dify_reasoning      TEXT,

    created_at          TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_candidates_entity ON match_candidates(entity_type);
CREATE INDEX idx_candidates_band   ON match_candidates(decision_band);
CREATE INDEX idx_candidates_score  ON match_candidates(match_score);


-- 5. 审核日志表
CREATE TABLE IF NOT EXISTS review_log (
    id              SERIAL PRIMARY KEY,
    entity_type     VARCHAR(16) NOT NULL,
    record_id_left  VARCHAR(64) NOT NULL,
    source_left     VARCHAR(32),
    record_id_right VARCHAR(64) NOT NULL,
    source_right    VARCHAR(32),
    match_score     REAL,
    decision        VARCHAR(16) NOT NULL,    -- 合并 / 不合并 / 保留待定
    comment         TEXT,
    updated_at      TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_review_log_decision ON review_log(decision);
CREATE INDEX idx_review_log_entity    ON review_log(entity_type);


-- 6. 管道指标表
CREATE TABLE IF NOT EXISTS match_metrics (
    id              SERIAL PRIMARY KEY,
    entity_type     VARCHAR(16)  NOT NULL,
    run_at          TIMESTAMP DEFAULT NOW(),
    total_candidates INTEGER,
    auto_merge_count  INTEGER,
    review_count      INTEGER,
    no_match_count    INTEGER,
    auto_merge_true_ratio  REAL,
    review_true_ratio      REAL,
    no_match_true_ratio    REAL,
    extra_data       JSONB
);


-- 7. 阈值配置表
CREATE TABLE IF NOT EXISTS threshold_config (
    id              SERIAL PRIMARY KEY,
    entity_type     VARCHAR(16) UNIQUE NOT NULL,
    auto_merge_threshold  REAL NOT NULL DEFAULT 0.80,
    review_lower_threshold REAL NOT NULL DEFAULT 0.50,
    human_adjusted  BOOLEAN DEFAULT FALSE,
    updated_at      TIMESTAMP DEFAULT NOW()
);


-- 8. Embedding 缓存表（替代 Redis 部分功能）
CREATE TABLE IF NOT EXISTS embedding_cache (
    id              SERIAL PRIMARY KEY,
    entity_type     VARCHAR(16) NOT NULL,
    record_id       VARCHAR(64) NOT NULL,
    text_content    TEXT,
    embedding       BYTEA,                  -- 序列化的 numpy 向量
    model_name      VARCHAR(128),
    created_at      TIMESTAMP DEFAULT NOW(),

    UNIQUE (entity_type, record_id, model_name)
);

CREATE INDEX idx_embedding_cache_lookup ON embedding_cache(entity_type, record_id);

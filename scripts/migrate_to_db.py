"""
数据迁移 — CSV → 数据库
==========================
将所有管道输出数据从 CSV 迁移到 SQLite/PostgreSQL

用法:
  python migrate_to_db.py             # 完整迁移（SQLite）
  python migrate_to_db.py --pg        # 迁移到 PostgreSQL
"""

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from db_manager import DBManager
from cache_manager import CacheManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def migrate_source_data(db: DBManager) -> None:
    """迁移原始数据"""
    source_files = {
        "customer": [
            ("crm_customers.csv", "CRM"),
            ("erp_customers.csv", "ERP"),
            ("ecommerce_customers.csv", "ECommerce"),
        ],
        "product": [
            ("erp_products.csv", "ERP"),
            ("ecommerce_products.csv", "ECommerce"),
        ],
    }

    for entity_type, files in source_files.items():
        for filename, source_name in files:
            path = DATA_DIR / filename
            if not path.exists():
                logger.warning("跳过: %s", filename)
                continue
            df = pd.read_csv(path, dtype=str).fillna("")
            df["record_id"] = df.get("record_id", "")
            df["source"] = source_name
            count = db.insert_source_records(df, entity_type)
            logger.info("  %s (%s): %d 条", filename, source_name, count)


def migrate_golden_data(db: DBManager) -> None:
    """迁移黄金记录"""
    golden_files = {
        "customer": "final_golden_customers.csv",
        "product": "final_golden_products.csv",
    }
    for entity_type, filename in golden_files.items():
        path = DATA_DIR / filename
        if path.exists():
            df = pd.read_csv(path, dtype=str).fillna("")
            db.insert_golden(df, entity_type)
            logger.info("黄金%s: %d 条", entity_type, len(df))


def migrate_candidates(db: DBManager) -> None:
    """迁移匹配候选对"""
    candidate_files = {
        "customer": "customer_match_candidates.csv",
        "product": "product_match_candidates.csv",
    }
    for entity_type, filename in candidate_files.items():
        path = DATA_DIR / filename
        if not path.exists():
            continue
        df = pd.read_csv(path, dtype=str).fillna("")
        # 类型转换
        for col in ["match_score", "semantic_name_score"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "is_true_match" in df.columns:
            df["is_true_match"] = df["is_true_match"].astype(str).str.lower() == "true"

        db.insert_candidates(df, entity_type)
        logger.info("候选对 %s: %d 条", entity_type, len(df))


def migrate_review_log(db: DBManager) -> None:
    """迁移审核日志"""
    log_files = {
        "customer": "customer_review_log.csv",
        "product": "product_review_log.csv",
    }
    for entity_type, filename in log_files.items():
        path = DATA_DIR / filename
        if not path.exists():
            continue
        df = pd.read_csv(path, dtype=str).fillna("")
        db.insert_review_log(df, entity_type)
        logger.info("审核日志 %s: %d 条", entity_type, len(df))


def migrate_enriched_data(db: DBManager) -> None:
    """将 Dify 增强结果同步到 match_candidates"""
    enriched_files = {
        "customer": "customer_review_enriched.csv",
        "product": "product_review_enriched.csv",
    }
    for entity_type, filename in enriched_files.items():
        path = DATA_DIR / filename
        if not path.exists():
            continue
        df = pd.read_csv(path, dtype=str).fillna("")
        if "dify_decision" not in df.columns:
            continue
        for _, row in df.iterrows():
            db.conn.execute(
                """UPDATE match_candidates
                   SET dify_decision = ?, dify_confidence = ?, dify_reasoning = ?
                   WHERE entity_type = ? AND record_id_left = ? AND record_id_right = ?""",
                [
                    row.get("dify_decision", ""),
                    float(row.get("dify_confidence", 0) or 0),
                    str(row.get("dify_reasoning", ""))[:200],
                    entity_type,
                    row["record_id_left"],
                    row["record_id_right"],
                ],
            )
        db.conn.commit()
        logger.info("Dify 增强 %s: %d 条已同步", entity_type, len(df))


def migrate_metrics(db: DBManager) -> None:
    """迁移评估指标"""
    metric_files = {
        "customer": "customer_match_metrics.json",
        "product": "product_match_metrics.json",
    }
    import json
    for entity_type, filename in metric_files.items():
        path = DATA_DIR / filename
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        db.save_metrics(metrics, entity_type)
        logger.info("指标 %s: 已保存", entity_type)


def migrate_thresholds(db: DBManager) -> None:
    """迁移优化阈值"""
    import json
    config_path = DATA_DIR / "optimized_thresholds.json"
    if not config_path.exists():
        return
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    for entity_type in ["customer", "product"]:
        if entity_type in config:
            ec = config[entity_type]
            db.save_thresholds(
                entity_type,
                ec.get("auto_merge_threshold", 0.80),
                ec.get("review_lower_threshold", 0.50),
                ec.get("human_adjusted", False),
            )
        logger.info("阈值 %s: 已保存", entity_type)


def migrate_embedding_cache(cache: CacheManager) -> None:
    """将已有的 Embedding 缓存迁移到新缓存系统"""
    import pickle
    model_cache = BASE_DIR / ".model_cache"
    if not model_cache.exists():
        return
    # 如果 embedder 之前缓存了向量在本地 pickle 文件中，这里不做迁移
    # 因为实际向量是 sentence-transformers 缓存在 .model_cache 中的
    logger.info("Embedding 模型缓存: %s (由 sentence-transformers 管理)", model_cache)


def main() -> None:
    use_pg = "--pg" in sys.argv

    logger.info("=" * 50)
    logger.info("数据迁移: CSV → %s", "PostgreSQL" if use_pg else "SQLite")
    logger.info("=" * 50)

    db = DBManager(use_pg=use_pg)
    cache = CacheManager()

    # 1. 建表
    db.init_schema()

    # 2. 迁移数据
    migrate_source_data(db)
    migrate_golden_data(db)
    migrate_candidates(db)
    migrate_review_log(db)
    migrate_enriched_data(db)
    migrate_metrics(db)
    migrate_thresholds(db)
    migrate_embedding_cache(cache)

    # 3. 统计
    logger.info("=" * 50)
    for entity in ["customer", "product"]:
        stats = db.get_stats(entity)
        logger.info("%s 数据库统计: %s", entity, stats)

    # 4. 缓存统计
    logger.info("缓存统计: %s", cache.stats())

    db.close()
    logger.info("迁移完成！数据库: %s", BASE_DIR / "data" / "mdm.db")


if __name__ == "__main__":
    main()

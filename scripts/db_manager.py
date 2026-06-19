"""
数据库管理器 — SQLite (本地测试) + PostgreSQL (生产环境)
==========================================================
用法：
  from scripts.db_manager import DBManager

  db = DBManager()                        # 默认 SQLite
  db = DBManager(use_pg=True)             # PostgreSQL（需要配置 pg_config.json）
  db.init_schema()                        # 建表
  db.insert_source_records(df)            # 导入数据
  db.get_review_queue("customer")         # 查询审核队列
"""

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
SQLITE_PATH = BASE_DIR / "data" / "mdm.db"
PG_CONFIG_PATH = BASE_DIR / "sql" / "pg_config.json"


def _load_pg_config() -> dict:
    if PG_CONFIG_PATH.exists():
        with open(PG_CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


class DBManager:
    """统一数据库接口，支持 SQLite 和 PostgreSQL"""

    def __init__(self, use_pg: bool = False):
        self.use_pg = use_pg
        self._conn = None

    # ── 连接管理 ──

    @property
    def conn(self):
        if self._conn is None:
            if self.use_pg:
                self._conn = self._connect_pg()
            else:
                self._conn = self._connect_sqlite()
        return self._conn

    def _connect_sqlite(self):
        path = str(SQLITE_PATH)
        logger.info("连接 SQLite: %s", path)
        conn = sqlite3.connect(path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _connect_pg(self):
        try:
            import psycopg2
            cfg = _load_pg_config()
            conn = psycopg2.connect(
                host=cfg.get("host", "localhost"),
                port=cfg.get("port", 5432),
                dbname=cfg.get("database", "mdm_db"),
                user=cfg.get("user", "mdm_user"),
                password=cfg.get("password", ""),
            )
            logger.info("连接 PostgreSQL: %s:%s/%s", cfg.get("host"), cfg.get("port"), cfg.get("database"))
            return conn
        except ImportError:
            logger.warning("psycopg2 未安装，回退到 SQLite")
            self.use_pg = False
            return self._connect_sqlite()
        except Exception as e:
            logger.warning("PostgreSQL 连接失败 (%s)，回退到 SQLite", e)
            self.use_pg = False
            return self._connect_sqlite()

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    # ── 初始化表结构 ──

    def init_schema(self) -> None:
        if self.use_pg:
            self._init_pg_schema()
        else:
            self._init_sqlite_schema()

    def _init_sqlite_schema(self) -> None:
        c = self.conn
        c.executescript("""
        CREATE TABLE IF NOT EXISTS source_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            record_id TEXT NOT NULL,
            source TEXT NOT NULL,
            entity_type TEXT NOT NULL,
            canonical_id TEXT,
            company_name TEXT, region TEXT, city TEXT, address TEXT,
            phone TEXT, tax_id TEXT, website TEXT, email TEXT, contact_person TEXT,
            product_name TEXT, category TEXT, brand TEXT, model TEXT,
            sku TEXT, specification TEXT, upc TEXT, price TEXT,
            source_created_at TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            UNIQUE(record_id, source, entity_type)
        );

        CREATE TABLE IF NOT EXISTS golden_customers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            canonical_customer_id TEXT UNIQUE NOT NULL,
            company_name TEXT, region TEXT, city TEXT, address TEXT,
            phone TEXT, tax_id TEXT, website TEXT, email TEXT, contact_person TEXT,
            source_created_at TEXT, sources TEXT, record_count INTEGER DEFAULT 1,
            merged_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS golden_products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            canonical_product_id TEXT UNIQUE NOT NULL,
            product_name TEXT, category TEXT, brand TEXT, model TEXT,
            sku TEXT, specification TEXT, upc TEXT, price TEXT,
            source_created_at TEXT, sources TEXT, record_count INTEGER DEFAULT 1,
            merged_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS match_candidates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_type TEXT NOT NULL,
            record_id_left TEXT NOT NULL, record_id_right TEXT NOT NULL,
            source_left TEXT, source_right TEXT, pair_source TEXT,
            match_score REAL, semantic_name_score REAL,
            decision_band TEXT, is_true_match INTEGER DEFAULT 0,
            dify_decision TEXT, dify_confidence REAL, dify_reasoning TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS review_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_type TEXT NOT NULL,
            record_id_left TEXT NOT NULL, source_left TEXT,
            record_id_right TEXT NOT NULL, source_right TEXT,
            match_score REAL, decision TEXT NOT NULL, comment TEXT,
            updated_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS match_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_type TEXT NOT NULL, run_at TEXT DEFAULT (datetime('now')),
            total_candidates INTEGER, auto_merge_count INTEGER,
            review_count INTEGER, no_match_count INTEGER,
            auto_merge_true_ratio REAL, review_true_ratio REAL,
            no_match_true_ratio REAL, extra_data TEXT
        );

        CREATE TABLE IF NOT EXISTS threshold_config (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_type TEXT UNIQUE NOT NULL,
            auto_merge_threshold REAL NOT NULL DEFAULT 0.80,
            review_lower_threshold REAL NOT NULL DEFAULT 0.50,
            human_adjusted INTEGER DEFAULT 0,
            updated_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS embedding_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_type TEXT NOT NULL, record_id TEXT NOT NULL,
            text_content TEXT, embedding BLOB, model_name TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            UNIQUE(entity_type, record_id, model_name)
        );
        """)
        self.conn.commit()
        logger.info("SQLite 表结构初始化完成")

    def _init_pg_schema(self) -> None:
        schema_path = BASE_DIR / "sql" / "schema.sql"
        if schema_path.exists():
            sql = schema_path.read_text(encoding="utf-8")
            with self.conn.cursor() as cur:
                cur.execute(sql)
            self.conn.commit()
            logger.info("PostgreSQL 表结构初始化完成")

    # ── 数据导入 ──

    def insert_source_records(self, df: pd.DataFrame, entity_type: str) -> int:
        """将 CSV 数据导入 source_records 表"""
        df = df.copy()
        df["entity_type"] = entity_type
        df["canonical_id"] = df.get("canonical_customer_id", df.get("canonical_product_id", ""))

        # 统一列名
        col_map = {
            "entity_type": "entity_type", "canonical_id": "canonical_id",
            "record_id": "record_id", "source": "source",
            "company_name": "company_name", "region": "region",
            "city": "city", "address": "address", "phone": "phone",
            "tax_id": "tax_id", "website": "website", "email": "email",
            "contact_person": "contact_person",
            "product_name": "product_name", "category": "category",
            "brand": "brand", "model": "model", "sku": "sku",
            "specification": "specification", "upc": "upc", "price": "price",
            "source_created_at": "source_created_at",
        }
        cols = [k for k in col_map if k in df.columns]
        sub = df[cols].copy()

        if self.use_pg:
            import psycopg2.extras
            rows = sub.where(sub.notna(), None).values.tolist()
            sql = f"INSERT INTO source_records ({','.join(cols)}) VALUES ({','.join(['%s']*len(cols))}) ON CONFLICT (record_id, source) DO NOTHING"
            with self.conn.cursor() as cur:
                psycopg2.extras.execute_batch(cur, sql, rows, page_size=500)
        else:
            sub.to_sql("source_records", self.conn, if_exists="append", index=False)
        self.conn.commit()
        logger.info("导入 %d 条 %s 记录到 source_records", len(sub), entity_type)
        return len(sub)

    def insert_golden(self, df: pd.DataFrame, entity_type: str) -> None:
        """导入黄金记录"""
        table = "golden_customers" if entity_type == "customer" else "golden_products"
        df.to_sql(table, self.conn, if_exists="replace", index=False)
        self.conn.commit()
        logger.info("导入 %d 条到 %s", len(df), table)

    def insert_candidates(self, df: pd.DataFrame, entity_type: str) -> None:
        """导入匹配候选对"""
        df = df.copy()
        df["entity_type"] = entity_type
        # 只保留必要的列
        cols = ["entity_type", "record_id_left", "record_id_right",
                "source_left", "source_right", "pair_source",
                "match_score", "semantic_name_score", "decision_band",
                "is_true_match"]
        available = [c for c in cols if c in df.columns]
        df[available].to_sql("match_candidates", self.conn, if_exists="append", index=False)
        self.conn.commit()
        logger.info("导入 %d 条候选对到 match_candidates", len(df))

    def insert_review_log(self, df: pd.DataFrame, entity_type: str) -> None:
        """导入审核日志"""
        df = df.copy()
        df["entity_type"] = entity_type
        df.to_sql("review_log", self.conn, if_exists="append", index=False)
        self.conn.commit()
        logger.info("导入 %d 条审核日志", len(df))

    def save_metrics(self, metrics: dict, entity_type: str) -> None:
        """保存管道指标"""
        df = pd.DataFrame([{**metrics, "entity_type": entity_type, "run_at": datetime.now().isoformat()}])
        df.to_sql("match_metrics", self.conn, if_exists="append", index=False)
        self.conn.commit()

    def save_thresholds(self, entity_type: str, auto_t: float, review_low: float, human_adj: bool = False) -> None:
        """保存优化阈值"""
        df = pd.DataFrame([{"entity_type": entity_type, "auto_merge_threshold": auto_t,
                            "review_lower_threshold": review_low, "human_adjusted": int(human_adj)}])
        df.to_sql("threshold_config", self.conn, if_exists="replace", index=False)
        self.conn.commit()

    # ── 查询接口 ──

    def query(self, sql: str, params=None) -> pd.DataFrame:
        return pd.read_sql_query(sql, self.conn, params=params)

    def get_review_queue(self, entity_type: str) -> pd.DataFrame:
        return self.query(
            "SELECT * FROM match_candidates WHERE entity_type = ? AND decision_band = 'review' "
            "ORDER BY match_score DESC",
            params=[entity_type],
        )

    def get_golden(self, entity_type: str) -> pd.DataFrame:
        table = "golden_customers" if entity_type == "customer" else "golden_products"
        return self.query(f"SELECT * FROM {table}")

    def get_review_log(self, entity_type: str) -> pd.DataFrame:
        return self.query(
            "SELECT * FROM review_log WHERE entity_type = ? ORDER BY updated_at DESC",
            params=[entity_type],
        )

    def get_metrics(self, entity_type: str, limit: int = 5) -> pd.DataFrame:
        return self.query(
            "SELECT * FROM match_metrics WHERE entity_type = ? ORDER BY run_at DESC LIMIT ?",
            params=[entity_type, limit],
        )

    def get_thresholds(self, entity_type: str) -> tuple:
        df = self.query(
            "SELECT * FROM threshold_config WHERE entity_type = ?", params=[entity_type]
        )
        if df.empty:
            return (0.80, 0.50)
        row = df.iloc[0]
        return (row["auto_merge_threshold"], row["review_lower_threshold"])

    # ── 统计查询 ──

    def get_stats(self, entity_type: str) -> dict:
        """获取实体匹配统计数据"""
        candidates = self.query(
            "SELECT COUNT(*) as total, decision_band FROM match_candidates "
            "WHERE entity_type = ? GROUP BY decision_band",
            params=[entity_type],
        )
        review_log = self.query(
            "SELECT COUNT(*) as cnt, decision FROM review_log "
            "WHERE entity_type = ? GROUP BY decision",
            params=[entity_type],
        )
        golden = self.query(
            f"SELECT COUNT(*) as cnt FROM {'golden_customers' if entity_type == 'customer' else 'golden_products'}"
        )
        return {
            "candidates": candidates.to_dict("records"),
            "review_log": review_log.to_dict("records"),
            "golden_count": int(golden.iloc[0]["cnt"]) if not golden.empty else 0,
        }


# ── 便捷函数 ──

def get_db() -> DBManager:
    """获取默认数据库实例（SQLite）"""
    return DBManager()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    db = DBManager()
    db.init_schema()
    print("SQLite 数据库初始化完成:", SQLITE_PATH)
    print("表:", db.query("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").values.flatten())

"""
缓存管理器 — Redis + 本地文件回退
==================================
用法：
  from scripts.cache_manager import CacheManager

  cache = CacheManager()                          # 默认本地缓存
  cache = CacheManager(use_redis=True, redis_config={"host": "localhost"})
  cache.set_embedding("customer", "CRM-001", vector)  # 缓存向量
  vec = cache.get_embedding("customer", "CRM-001")    # 读取缓存
  stats = cache.stats()                               # 缓存统计
"""

import json
import logging
import pickle
import time
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
LOCAL_CACHE_DIR = BASE_DIR / "data" / "cache"
REDIS_CONFIG_PATH = BASE_DIR / "sql" / "redis_config.json"


def _load_redis_config() -> dict:
    if REDIS_CONFIG_PATH.exists():
        with open(REDIS_CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


class CacheManager:
    """统一的缓存接口，支持 Redis 和本地文件两种后端"""

    def __init__(self, use_redis: bool = False, redis_config: Optional[dict] = None):
        self.use_redis = use_redis
        self._redis = None
        self._redis_config = redis_config or _load_redis_config()

        # 确保本地缓存目录存在
        LOCAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ── Redis 连接 ──

    @property
    def redis(self):
        if self._redis is None and self.use_redis:
            try:
                import redis
                cfg = self._redis_config
                self._redis = redis.Redis(
                    host=cfg.get("host", "localhost"),
                    port=cfg.get("port", 6379),
                    db=cfg.get("db", 0),
                    decode_responses=False,
                )
                self._redis.ping()
                logger.info("Redis 连接成功: %s:%s", cfg.get("host"), cfg.get("port"))
            except ImportError:
                logger.warning("redis-py 未安装，回退到本地缓存")
                self.use_redis = False
            except Exception as e:
                logger.warning("Redis 连接失败 (%s)，回退到本地缓存", e)
                self.use_redis = False
        return self._redis

    # ── Embedding 缓存 ──

    def _emb_key(self, entity_type: str, record_id: str) -> str:
        return f"emb:{entity_type}:{record_id}"

    def set_embedding(self, entity_type: str, record_id: str, vector: np.ndarray) -> None:
        """缓存 Embedding 向量"""
        if self.use_redis and self.redis:
            data = pickle.dumps(vector)
            self.redis.set(self._emb_key(entity_type, record_id), data)
            self.redis.expire(self._emb_key(entity_type, record_id), 86400 * 7)  # 7 天过期
        else:
            path = LOCAL_CACHE_DIR / self._emb_key(entity_type, record_id)
            with open(path, "wb") as f:
                pickle.dump(vector, f)

    def get_embedding(self, entity_type: str, record_id: str) -> Optional[np.ndarray]:
        """读取缓存的 Embedding 向量"""
        if self.use_redis and self.redis:
            data = self.redis.get(self._emb_key(entity_type, record_id))
            if data:
                return pickle.loads(data)
            return None
        else:
            path = LOCAL_CACHE_DIR / self._emb_key(entity_type, record_id)
            if path.exists():
                with open(path, "rb") as f:
                    return pickle.load(f)
            return None

    def batch_set_embeddings(self, entity_type: str, emb_map: dict) -> None:
        """批量缓存 Embedding {record_id: vector}"""
        for rid, vec in emb_map.items():
            self.set_embedding(entity_type, rid, vec)
        logger.info("缓存 %d 个 %s Embedding", len(emb_map), entity_type)

    def batch_get_embeddings(self, entity_type: str, record_ids: list[str]) -> dict:
        """批量读取缓存 {record_id: vector or None}"""
        result = {}
        for rid in record_ids:
            result[rid] = self.get_embedding(entity_type, rid)
        hits = sum(1 for v in result.values() if v is not None)
        logger.info("Embedding 缓存命中: %d/%d (%.1f%%)", hits, len(record_ids),
                     hits / len(record_ids) * 100 if record_ids else 0)
        return result

    # ── 通用 K-V 缓存 ──

    def set(self, key: str, value, expire: int = 3600) -> None:
        """设置通用缓存"""
        if self.use_redis and self.redis:
            self.redis.set(key, pickle.dumps(value), ex=expire)
        else:
            path = LOCAL_CACHE_DIR / f"{key}.pkl"
            data = {"value": value, "expire": time.time() + expire}
            with open(path, "wb") as f:
                pickle.dump(data, f)

    def get(self, key: str):
        """读取通用缓存"""
        if self.use_redis and self.redis:
            data = self.redis.get(key)
            return pickle.loads(data) if data else None
        else:
            path = LOCAL_CACHE_DIR / f"{key}.pkl"
            if path.exists():
                with open(path, "rb") as f:
                    data = pickle.load(f)
                if time.time() < data["expire"]:
                    return data["value"]
                path.unlink(missing_ok=True)
            return None

    # ── 统计 ──

    def stats(self) -> dict:
        """缓存统计"""
        if self.use_redis and self.redis:
            info = self.redis.info("keyspace")
            return {"backend": "redis", "keyspace": info}
        else:
            files = list(LOCAL_CACHE_DIR.glob("*"))
            total_size = sum(f.stat().st_size for f in files if f.is_file())
            return {
                "backend": "local",
                "file_count": len(files),
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "cache_dir": str(LOCAL_CACHE_DIR),
            }

    def clear(self) -> None:
        """清空缓存"""
        if self.use_redis and self.redis:
            self.redis.flushdb()
        else:
            for f in LOCAL_CACHE_DIR.glob("*"):
                f.unlink()
        logger.info("缓存已清空")


# ── 便捷函数 ──

def get_cache() -> CacheManager:
    return CacheManager()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 测试本地缓存
    cache = CacheManager()
    vec = np.random.randn(1024).astype(np.float32)

    cache.set_embedding("customer", "TEST-001", vec)
    loaded = cache.get_embedding("customer", "TEST-001")

    print(f"写入: {vec[:5]}")
    print(f"读取: {loaded[:5] if loaded is not None else 'None'}")
    print(f"匹配: {np.allclose(vec, loaded) if loaded is not None else False}")
    print(f"统计: {cache.stats()}")

    cache.clear()
    print("缓存已清空")

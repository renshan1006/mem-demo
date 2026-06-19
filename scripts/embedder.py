"""
语义相似度模块 — 基于 bge-large-zh-v1.5 做实体名称匹配
==========================================================
用法：
  from scripts.embedder import Embedder
  emb = Embedder()                        # 自动加载模型
  sim = emb.name_similarity("北京Alpha科技", "北京阿尔法科技")  # → 0.85~
  sim = emb.pairwise_similarity(names_a, names_b)            # 批量计算
"""

import logging
import os
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# 国内优先使用 HF 镜像
if not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

logger = logging.getLogger(__name__)

MODEL_NAME = "BAAI/bge-large-zh-v1.5"
CACHE_DIR = Path(__file__).resolve().parent.parent / ".model_cache"


class Embedder:
    """语义嵌入器，封装 sentence-transformers 模型"""

    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self._model = None
        self._available = None  # None = 未检测, True/False

    @property
    def available(self) -> bool:
        """检测模型是否可用"""
        if self._available is None:
            try:
                import sentence_transformers  # noqa: F401
                self._available = True
            except ImportError:
                logger.warning("sentence-transformers 未安装，回退到编辑距离匹配")
                self._available = False
        return self._available

    @property
    def model(self):
        """懒加载模型（失败则回退到编辑距离）"""
        if self._model is None and self.available:
            from sentence_transformers import SentenceTransformer

            logger.info("加载模型 %s ...", self.model_name)
            try:
                self._model = SentenceTransformer(
                    self.model_name,
                    cache_folder=str(CACHE_DIR),
                )
            except Exception as e:
                logger.warning("模型加载失败 (%s)，回退到编辑距离匹配。", e)
                self._available = False
                self._model = None
                return None
            logger.info("模型加载完成。")
        return self._model

    def encode(self, texts: list[str], show_progress: bool = True) -> np.ndarray:
        """将文本列表编码为归一化向量矩阵 [N, D]"""
        if not self.available or self.model is None:
            return np.zeros((len(texts), 1))
        clean = [str(t).strip() if pd.notna(t) else "" for t in texts]
        embeddings = self.model.encode(
            clean,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
            batch_size=128,
        )
        return np.array(embeddings)

    def name_similarity(self, name_a: str, name_b: str) -> float:
        """两个名称的语义相似度 (0~1)"""
        if not name_a or not name_b:
            return 0.0
        if self.available and self.model is not None:
            vecs = self.encode([str(name_a), str(name_b)], show_progress=False)
            sim = float(np.dot(vecs[0], vecs[1]))
            return max(0.0, min(1.0, sim))
        # 回退：编辑距离
        if not name_a or not name_b:
            return 0.0
        return SequenceMatcher(None, str(name_a), str(name_b)).ratio()

    def pairwise_similarity(
        self,
        names_a: list[str],
        names_b: list[str],
    ) -> np.ndarray:
        """
        批量计算两组名称的逐对余弦相似度
        返回 shape [len(names_a), len(names_b)]
        """
        if not self.available or self.model is None:
            # 回退：逐对编辑距离
            result = np.zeros((len(names_a), len(names_b)))
            for i, a in enumerate(names_a):
                for j, b in enumerate(names_b):
                    result[i, j] = SequenceMatcher(None, str(a), str(b)).ratio()
            return result

        emb_a = self.encode(names_a, show_progress=True)
        emb_b = self.encode(names_b, show_progress=True)
        # 余弦相似度 = 点积 (已归一化)
        sim = np.dot(emb_a, emb_b.T)
        return np.clip(sim, 0.0, 1.0)

    def embed_dataframe(
        self,
        df: pd.DataFrame,
        text_col: str,
        key_col: str = "record_id",
    ) -> dict:
        """
        对 DataFrame 中唯一文本做编码，返回 {key: embedding_vector} 映射
        """
        unique_texts = df[[key_col, text_col]].drop_duplicates(subset=key_col)
        names = unique_texts[text_col].tolist()
        keys = unique_texts[key_col].tolist()
        vectors = self.encode(names, show_progress=True)
        return dict(zip(keys, vectors))


# 全局单例
_embedder: Optional[Embedder] = None


def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder


def embedding_similarity(name_a: str, name_b: str) -> float:
    """快捷函数：两个名称的语义相似度"""
    return get_embedder().name_similarity(name_a, name_b)

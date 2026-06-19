"""
商品主数据匹配管道 — v2.0
==========================
多策略匹配：阻塞规则（品类） + Embedding 语义相似度 + 规则加权打分
Embedding 模型：bge-large-zh-v1.5
"""

import json
import logging
import time
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd

from embedder import get_embedder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

PRODUCT_SOURCES = {
    "ERP": "erp_products.csv",
    "ECommerce": "ecommerce_products.csv",
}

OUTPUT_FILES = {
    "candidate": DATA_DIR / "product_match_candidates.csv",
    "review_queue": DATA_DIR / "product_review_queue.csv",
    "golden": DATA_DIR / "golden_products.csv",
    "metrics": DATA_DIR / "product_match_metrics.json",
}

# ── 权重配置 ──
# SKU/UPC 是商品唯一标识的核心，权重高于名称
WEIGHTS = {
    "name": 0.20,
    "brand": 0.15,
    "model": 0.10,
    "sku": 0.25,
    "upc": 0.15,
    "spec": 0.05,
}
# 标识符不匹配时的分数上限（SKU/UPC 不同 = 大概率不同商品变体）
IDENTIFIER_MISMATCH_CAP = 0.55
EMBED_BLEND = 0.80  # Embedding 与编辑距离混合比


# ═══════════════════════════════════════════════════════════════════════════════
# 文本预处理
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_for_edit(val: str) -> str:
    if not isinstance(val, str):
        return ""
    text = val.strip().lower()
    for ch in ["有限公司", "公司", "科技", "alpha"]:
        text = text.replace(ch, "")
    return "".join(ch for ch in text if ch.isalnum())


def normalize_sku(val: str) -> str:
    if not isinstance(val, str):
        return ""
    return "".join(ch for ch in val.upper() if ch.isalnum())


def normalize_upc(val: str) -> str:
    if not isinstance(val, str):
        return ""
    return "".join(ch for ch in val if ch.isdigit())


def edit_similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


# ═══════════════════════════════════════════════════════════════════════════════
# 数据加载
# ═══════════════════════════════════════════════════════════════════════════════

def load_products() -> dict:
    sources = {}
    for source_name, filename in PRODUCT_SOURCES.items():
        path = DATA_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"缺少数据文件：{path}")
        df = pd.read_csv(path, dtype=str).fillna("")
        df["source"] = source_name
        # 原始文本（Embedding 用）
        df["product_name_raw"] = df["product_name"].str.strip()
        df["brand_raw"] = df["brand"].str.strip()
        df["model_raw"] = df["model"].str.strip()
        # 规范化文本（编辑距离用）
        df["norm_product_name"] = df["product_name"].apply(normalize_for_edit)
        df["norm_brand"] = df["brand"].apply(normalize_for_edit)
        df["norm_model"] = df["model"].apply(normalize_for_edit)
        df["norm_sku"] = df["sku"].apply(normalize_sku)
        df["norm_upc"] = df["upc"].apply(normalize_upc)
        df["norm_specification"] = df["specification"].apply(normalize_for_edit)
        df["block_category"] = df["category"].fillna("").astype(str).str.strip().str.lower()
        sources[source_name] = df

    logger.info("数据加载完成: %s", {k: len(v) for k, v in sources.items()})
    return sources


# ═══════════════════════════════════════════════════════════════════════════════
# Embedding 预计算 — 商品名 & 品牌 & 型号
# ═══════════════════════════════════════════════════════════════════════════════

def precompute_product_embeddings(sources: dict) -> dict:
    """
    预计算商品名、品牌、型号的 Embedding
    返回 {"name": {rid: vec}, "brand": {rid: vec}, "model": {rid: vec}}
    """
    embedder = get_embedder()
    if not embedder.available:
        logger.warning("Embedding 模型不可用，回退到编辑距离匹配。")
        return {}

    fields = ["product_name_raw", "brand_raw", "model_raw"]
    result = {}

    for field in fields:
        key_name = field.replace("_raw", "")
        all_records = {}
        for source_name, df in sources.items():
            for _, row in df.iterrows():
                rid = row["record_id"]
                if rid not in all_records:
                    all_records[rid] = row.get(field, "")

        rids = list(all_records.keys())
        texts = [all_records[r] for r in rids]
        logger.info("为 %d 个唯一 %s 计算 Embedding ...", len(texts), key_name)

        t0 = time.time()
        vectors = embedder.encode(texts, show_progress=True)
        logger.info("  %s Embedding 完成，耗时 %.1f 秒", key_name, time.time() - t0)

        result[key_name] = {rid: vec for rid, vec in zip(rids, vectors)}

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 候选对生成 & 打分
# ═══════════════════════════════════════════════════════════════════════════════

def build_candidate_pairs(sources: dict, emb_maps: dict) -> pd.DataFrame:
    embedder = get_embedder()
    has_embeddings = embedder.available and len(emb_maps) > 0

    pairs = []
    source_list = list(sources.items())

    for i in range(len(source_list)):
        left_name, left_df = source_list[i]
        for j in range(i + 1, len(source_list)):
            right_name, right_df = source_list[j]

            merged = pd.merge(
                left_df, right_df,
                on="block_category",
                suffixes=("_left", "_right"),
                how="inner",
            )
            if merged.empty:
                continue

            merged = merged[
                ~(
                    (merged["product_name_left"] == merged["product_name_right"])
                    & (merged["record_id_left"] == merged["record_id_right"])
                )
            ]

            n_pairs = len(merged)
            logger.info("  源对 %s-%s: %d 个候选对", left_name, right_name, n_pairs)

            def _semantic_scores(field_key: str) -> np.ndarray:
                """为指定字段计算语义相似度"""
                scores = np.zeros(n_pairs, dtype=np.float32)
                if has_embeddings and field_key in emb_maps:
                    emb_map = emb_maps[field_key]
                    left_vecs = np.array([emb_map.get(rid, np.zeros(768)) for rid in merged["record_id_left"]])
                    right_vecs = np.array([emb_map.get(rid, np.zeros(768)) for rid in merged["record_id_right"]])
                    left_n = np.linalg.norm(left_vecs, axis=1)
                    right_n = np.linalg.norm(right_vecs, axis=1)
                    valid = (left_n > 0) & (right_n > 0)
                    if valid.any():
                        cos = np.sum(left_vecs[valid] * right_vecs[valid], axis=1)
                        scores[valid] = np.clip(cos, 0.0, 1.0)
                return scores

            # 名称相似度
            name_embed = _semantic_scores("product_name")
            name_edit = np.array([
                edit_similarity(r["norm_product_name_left"], r["norm_product_name_right"])
                for _, r in merged.iterrows()
            ], dtype=np.float32)
            name_scores = (
                EMBED_BLEND * name_embed + (1 - EMBED_BLEND) * name_edit
                if has_embeddings else name_edit
            )

            # 品牌相似度
            brand_embed = _semantic_scores("brand")
            brand_edit = np.array([
                edit_similarity(r["norm_brand_left"], r["norm_brand_right"])
                for _, r in merged.iterrows()
            ], dtype=np.float32)
            brand_scores = (
                EMBED_BLEND * brand_embed + (1 - EMBED_BLEND) * brand_edit
                if has_embeddings else brand_edit
            )

            # 型号相似度
            model_embed = _semantic_scores("model")
            model_edit = np.array([
                edit_similarity(r["norm_model_left"], r["norm_model_right"])
                for _, r in merged.iterrows()
            ], dtype=np.float32)
            model_scores = (
                EMBED_BLEND * model_embed + (1 - EMBED_BLEND) * model_edit
                if has_embeddings else model_edit
            )

            # SKU / UPC 精确匹配
            sku_match = (
                (merged["norm_sku_left"].values != "")
                & (merged["norm_sku_left"].values == merged["norm_sku_right"].values)
            ).astype(np.float32)

            upc_match = (
                (merged["norm_upc_left"].values != "")
                & (merged["norm_upc_left"].values == merged["norm_upc_right"].values)
            ).astype(np.float32)

            # 规格相似度（编辑距离就够了）
            spec_scores = np.array([
                edit_similarity(r["norm_specification_left"], r["norm_specification_right"])
                for _, r in merged.iterrows()
            ], dtype=np.float32)

            # ── 加权总分 ──
            scores = (
                name_scores * WEIGHTS["name"]
                + brand_scores * WEIGHTS["brand"]
                + model_scores * WEIGHTS["model"]
                + sku_match * WEIGHTS["sku"]
                + upc_match * WEIGHTS["upc"]
                + spec_scores * WEIGHTS["spec"]
            )
            # 强信号加分
            scores = np.where(
                (sku_match.astype(bool)) | (upc_match.astype(bool)),
                np.minimum(1.0, scores + 0.05),
                scores,
            )
            scores = np.where(
                (brand_scores > 0.9) & (model_scores > 0.9),
                np.minimum(1.0, scores + 0.03),
                scores,
            )
            # ── 标识符不匹配惩罚 ──
            # 如果 SKU 两边都有值但不匹配 → 不同商品变体，分数上限
            sku_both_present = (
                (merged["norm_sku_left"].values != "")
                & (merged["norm_sku_right"].values != "")
            )
            sku_conflict = sku_both_present & (~sku_match.astype(bool))
            # UPC 同理
            upc_both_present = (
                (merged["norm_upc_left"].values != "")
                & (merged["norm_upc_right"].values != "")
            )
            upc_conflict = upc_both_present & (~upc_match.astype(bool))
            # 任一标识符冲突 → cap 分数
            has_identifier_conflict = sku_conflict | upc_conflict
            scores = np.where(
                has_identifier_conflict,
                np.minimum(scores, IDENTIFIER_MISMATCH_CAP),
                scores,
            )
            scores = np.round(scores, 4)

            merged = merged.copy()
            merged["match_score"] = scores
            merged["pair_source"] = f"{left_name}-{right_name}"
            merged["is_true_match"] = (
                merged["canonical_product_id_left"] == merged["canonical_product_id_right"]
            )

            pairs.append(merged)

    return pd.concat(pairs, ignore_index=True) if pairs else pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════════
# 分类 & 输出
# ═══════════════════════════════════════════════════════════════════════════════

def load_thresholds(entity: str) -> tuple[float, float]:
    """从优化配置文件加载阈值，若不存在则使用默认值"""
    config_path = DATA_DIR / "optimized_thresholds.json"
    defaults = (0.80, 0.50)
    if not config_path.exists():
        return defaults
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        entity_config = config.get(entity, {})
        auto = entity_config.get("auto_merge_threshold", defaults[0])
        review_low = entity_config.get("review_lower_threshold", defaults[1])
        logger.info("使用优化阈值: auto_merge≥%.2f, review≥%.2f", auto, review_low)
        return (auto, review_low)
    except Exception:
        return defaults


def categorize_matches(candidates: pd.DataFrame, entity: str = "product") -> pd.DataFrame:
    auto_t, review_low = load_thresholds(entity)

    def label(score: float) -> str:
        if score >= auto_t:
            return "auto_merge"
        if score >= review_low:
            return "review"
        return "no_match"

    candidates = candidates.copy()
    candidates["decision_band"] = candidates["match_score"].apply(label)
    return candidates


def save_candidate_files(candidates: pd.DataFrame) -> None:
    if candidates.empty:
        logger.warning("没有生成任何候选匹配对。")
        return
    candidates.to_csv(OUTPUT_FILES["candidate"], index=False, encoding="utf-8-sig")
    review_queue = candidates[candidates["decision_band"] == "review"].sort_values(
        "match_score", ascending=False
    )
    review_queue.to_csv(OUTPUT_FILES["review_queue"], index=False, encoding="utf-8-sig")
    logger.info("候选匹配文件: %s (%d 条)", OUTPUT_FILES["candidate"].name, len(candidates))
    logger.info("审核队列文件: %s (%d 条)", OUTPUT_FILES["review_queue"].name, len(review_queue))


def build_golden_products(sources: dict) -> pd.DataFrame:
    all_products = pd.concat(sources.values(), ignore_index=True)
    grouped = []
    for canonical_id, group in all_products.groupby("canonical_product_id", sort=False):
        best = group.iloc[0]
        merged = {
            "canonical_product_id": canonical_id,
            "product_name": best["product_name"],
            "category": best["category"],
            "brand": best["brand"],
            "model": best["model"],
            "sku": best["sku"],
            "specification": best["specification"],
            "upc": best["upc"],
            "price": best["price"],
            "source_created_at": best["source_created_at"],
            "sources": ";".join(sorted(group["source"].unique())),
            "record_count": len(group),
        }
        grouped.append(merged)
    golden = pd.DataFrame(grouped)
    golden.to_csv(OUTPUT_FILES["golden"], index=False, encoding="utf-8-sig")
    logger.info("黄金商品表: %s (%d 条)", OUTPUT_FILES["golden"].name, len(golden))
    return golden


def compute_metrics(candidates: pd.DataFrame) -> dict:
    if candidates.empty:
        return {}
    m = {
        "total_candidates": int(len(candidates)),
        "auto_merge_count": int((candidates["decision_band"] == "auto_merge").sum()),
        "review_count": int((candidates["decision_band"] == "review").sum()),
        "no_match_count": int((candidates["decision_band"] == "no_match").sum()),
        "auto_merge_true_ratio": float(
            candidates.loc[candidates["decision_band"] == "auto_merge", "is_true_match"].mean() or 0
        ),
        "review_true_ratio": float(
            candidates.loc[candidates["decision_band"] == "review", "is_true_match"].mean() or 0
        ),
        "no_match_true_ratio": float(
            candidates.loc[candidates["decision_band"] == "no_match", "is_true_match"].mean() or 0
        ),
    }
    with open(OUTPUT_FILES["metrics"], "w", encoding="utf-8") as fp:
        json.dump(m, fp, ensure_ascii=False, indent=2)
    logger.info("评估指标: %s", OUTPUT_FILES["metrics"].name)
    return m


# ═══════════════════════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    t_total = time.time()

    logger.info("=" * 50)
    logger.info("商品主数据匹配管道 v2.0 (Embedding 增强)")
    logger.info("=" * 50)

    logger.info("[1/4] 加载商品数据 ...")
    sources = load_products()

    logger.info("[2/4] 预计算商品 Embedding (名称/品牌/型号) ...")
    emb_maps = precompute_product_embeddings(sources)

    logger.info("[3/4] 生成候选匹配对并打分 ...")
    t_candidates = time.time()
    candidates = build_candidate_pairs(sources, emb_maps)
    logger.info("候选对生成耗时: %.1f 秒", time.time() - t_candidates)

    logger.info("[4/4] 分类、保存、评估 ...")
    candidates = categorize_matches(candidates, "product")
    save_candidate_files(candidates)
    golden = build_golden_products(sources)
    metrics = compute_metrics(candidates)

    logger.info("=" * 50)
    logger.info("管道完成！总耗时: %.1f 秒", time.time() - t_total)
    logger.info("指标摘要:")
    for k, v in metrics.items():
        if isinstance(v, float):
            logger.info("  %s: %.4f", k, v)
        else:
            logger.info("  %s: %s", k, v)
    logger.info("=" * 50)


if __name__ == "__main__":
    main()

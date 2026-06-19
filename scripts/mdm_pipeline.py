"""
客户主数据匹配管道 — v2.0
==========================
多策略匹配：阻塞规则（地区） + Embedding 语义相似度 + 规则加权打分
Embedding 模型：bge-large-zh-v1.5（通过 sentence-transformers）
回退方案：模型不可用时使用 difflib 编辑距离
"""

import json
import logging
import time
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd

# 导入共享 embedder
from embedder import get_embedder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

CUSTOMER_SOURCES = {
    "CRM": "crm_customers.csv",
    "ERP": "erp_customers.csv",
    "ECommerce": "ecommerce_customers.csv",
}

OUTPUT_FILES = {
    "candidate": DATA_DIR / "customer_match_candidates.csv",
    "review_queue": DATA_DIR / "customer_review_queue.csv",
    "golden": DATA_DIR / "golden_customers.csv",
    "metrics": DATA_DIR / "customer_match_metrics.json",
}

# ── 权重配置 ──
# 名称权重提升（语义匹配更可靠），税号仍是强信号
WEIGHTS = {
    "name": 0.45,
    "tax_id": 0.30,
    "phone": 0.15,
    "address": 0.10,
}
# Embedding 相似度与编辑距离的混合比例
EMBED_BLEND = 0.80  # 80% 用语义, 20% 用编辑距离做平滑


# ═══════════════════════════════════════════════════════════════════════════════
# 文本预处理
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_company(name: str) -> str:
    """轻量规范化：保留原始文本用于 Embedding，只做基础清理"""
    if not isinstance(name, str):
        return ""
    return name.strip()


def normalize_for_edit(name: str) -> str:
    """编辑距离用：去后缀、只留字母数字"""
    if not isinstance(name, str):
        return ""
    text = name.strip().lower()
    for ch in ["有限公司", "公司", "科技", "信息技术", "贸易", "实业", "集团", "商务", "国际"]:
        text = text.replace(ch, "")
    return "".join(ch for ch in text if ch.isalnum())


def normalize_phone(phone: str) -> str:
    if not isinstance(phone, str):
        return ""
    return "".join(ch for ch in phone if ch.isdigit())


def normalize_address(address: str) -> str:
    if not isinstance(address, str):
        return ""
    text = address.strip().lower()
    for ch in ["号楼", "号院", "号", "栋", "园", "楼"]:
        text = text.replace(ch, "")
    return "".join(ch for ch in text if ch.isalnum())


def edit_similarity(left: str, right: str) -> float:
    """编辑距离相似度（回退方案）"""
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


# ═══════════════════════════════════════════════════════════════════════════════
# 数据加载
# ═══════════════════════════════════════════════════════════════════════════════

def load_customers() -> dict:
    """加载并预处理所有数据源的客户数据"""
    sources = {}
    for source_name, filename in CUSTOMER_SOURCES.items():
        path = DATA_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"缺少数据文件：{path}")
        df = pd.read_csv(path, dtype=str).fillna("")
        df["source"] = source_name
        df["company_name_raw"] = df["company_name"].apply(normalize_company)
        df["norm_company"] = df["company_name"].apply(normalize_for_edit)
        df["norm_phone"] = df["phone"].apply(normalize_phone)
        df["norm_address"] = df["address"].apply(normalize_address)
        df["block_region"] = df["region"].fillna("").astype(str).str.strip().str.lower()
        sources[source_name] = df

    logger.info(
        "数据加载完成: %s",
        {k: len(v) for k, v in sources.items()},
    )
    return sources


# ═══════════════════════════════════════════════════════════════════════════════
# Embedding 预计算
# ═══════════════════════════════════════════════════════════════════════════════

def precompute_name_embeddings(sources: dict) -> dict:
    """
    对所有数据源中的唯一原始公司名称预计算 Embedding
    返回 {record_id: np.ndarray} 映射
    """
    embedder = get_embedder()
    if not embedder.available:
        logger.warning("Embedding 模型不可用，将回退到编辑距离匹配。")
        return {}

    # 收集所有唯一的 record_id 和对应的原始公司名
    all_records = {}
    for source_name, df in sources.items():
        for _, row in df.iterrows():
            rid = row["record_id"]
            if rid not in all_records:
                all_records[rid] = row["company_name_raw"]

    rids = list(all_records.keys())
    names = [all_records[r] for r in rids]
    logger.info("为 %d 个唯一客户名称计算 Embedding ...", len(names))

    t0 = time.time()
    vectors = embedder.encode(names, show_progress=True)
    elapsed = time.time() - t0
    logger.info("Embedding 完成，耗时 %.1f 秒 (%.0f 条/秒)", elapsed, len(names) / elapsed)

    return {rid: vec for rid, vec in zip(rids, vectors)}


# ═══════════════════════════════════════════════════════════════════════════════
# 候选对生成 & 打分
# ═══════════════════════════════════════════════════════════════════════════════

def build_candidate_pairs(sources: dict, emb_map: dict) -> pd.DataFrame:
    """
    按地区阻塞生成候选对，用 Embedding + 规则加权打分
    """
    embedder = get_embedder()
    has_embeddings = embedder.available and len(emb_map) > 0

    pairs = []
    source_list = list(sources.items())

    for i in range(len(source_list)):
        left_name, left_df = source_list[i]
        for j in range(i + 1, len(source_list)):
            right_name, right_df = source_list[j]

            # 按阻塞键合并
            merged = pd.merge(
                left_df, right_df,
                on="block_region",
                suffixes=("_left", "_right"),
                how="inner",
            )
            if merged.empty:
                continue

            # 去掉完全相同的记录
            merged = merged[
                ~(
                    (merged["company_name_left"] == merged["company_name_right"])
                    & (merged["record_id_left"] == merged["record_id_right"])
                )
            ]

            n_pairs = len(merged)
            logger.info("  源对 %s-%s: %d 个候选对", left_name, right_name, n_pairs)

            # ── 计算名称相似度 ──
            name_scores = np.zeros(n_pairs, dtype=np.float32)

            if has_embeddings:
                # 向量化计算：从 emb_map 取出向量做批量点积
                left_vecs = np.array([emb_map.get(rid, np.zeros(768)) for rid in merged["record_id_left"]])
                right_vecs = np.array([emb_map.get(rid, np.zeros(768)) for rid in merged["record_id_right"]])
                # 只对向量非零的行计算（回退行为）
                left_norms = np.linalg.norm(left_vecs, axis=1)
                right_norms = np.linalg.norm(right_vecs, axis=1)
                valid = (left_norms > 0) & (right_norms > 0)

                if valid.any():
                    cos_sim = np.sum(left_vecs[valid] * right_vecs[valid], axis=1)
                    name_scores[valid] = np.clip(cos_sim, 0.0, 1.0)

            # 编辑距离分数（混合作平滑）
            edit_scores = np.array([
                edit_similarity(r["norm_company_left"], r["norm_company_right"])
                for _, r in merged.iterrows()
            ], dtype=np.float32)

            if has_embeddings:
                name_scores = EMBED_BLEND * name_scores + (1 - EMBED_BLEND) * edit_scores
            else:
                name_scores = edit_scores

            # ── 规则打分 ──
            tax_match = (
                (merged["tax_id_left"].values != "")
                & (merged["tax_id_left"].values == merged["tax_id_right"].values)
            ).astype(np.float32)

            phone_match = (
                (merged["norm_phone_left"].values != "")
                & (merged["norm_phone_left"].values == merged["norm_phone_right"].values)
            ).astype(np.float32)

            address_scores = np.array([
                edit_similarity(r["norm_address_left"], r["norm_address_right"])
                for _, r in merged.iterrows()
            ], dtype=np.float32)

            # ── 加权总分 ──
            scores = (
                name_scores * WEIGHTS["name"]
                + tax_match * WEIGHTS["tax_id"]
                + phone_match * WEIGHTS["phone"]
                + address_scores * WEIGHTS["address"]
            )
            # 强信号加分
            scores = np.where(tax_match.astype(bool), np.minimum(1.0, scores + 0.10), scores)
            scores = np.where(phone_match.astype(bool), np.minimum(1.0, scores + 0.05), scores)
            scores = np.round(scores, 4)

            merged = merged.copy()
            merged["match_score"] = scores
            merged["pair_source"] = f"{left_name}-{right_name}"
            merged["is_true_match"] = (
                merged["canonical_customer_id_left"] == merged["canonical_customer_id_right"]
            )
            # 保留语义分数用于分析
            if has_embeddings:
                merged["semantic_name_score"] = np.round(name_scores, 4)

            pairs.append(merged)

    result = pd.concat(pairs, ignore_index=True) if pairs else pd.DataFrame()
    return result


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


def categorize_matches(candidates: pd.DataFrame, entity: str = "customer") -> pd.DataFrame:
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


def build_golden_customers(sources: dict) -> pd.DataFrame:
    all_customers = pd.concat(sources.values(), ignore_index=True)
    grouped = []
    for canonical_id, group in all_customers.groupby("canonical_customer_id", sort=False):
        best = group.iloc[0]
        merged = {
            "canonical_customer_id": canonical_id,
            "company_name": best["company_name"],
            "region": best["region"],
            "city": best["city"],
            "address": best["address"],
            "phone": best["phone"],
            "tax_id": best["tax_id"],
            "website": best["website"],
            "email": best["email"],
            "contact_person": best["contact_person"],
            "source_created_at": best["source_created_at"],
            "sources": ";".join(sorted(group["source"].unique())),
            "record_count": len(group),
        }
        grouped.append(merged)
    golden = pd.DataFrame(grouped)
    golden.to_csv(OUTPUT_FILES["golden"], index=False, encoding="utf-8-sig")
    logger.info("黄金客户表: %s (%d 条)", OUTPUT_FILES["golden"].name, len(golden))
    return golden


def compute_metrics(candidates: pd.DataFrame, golden: pd.DataFrame) -> dict:
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
    # 如果 auto_merge_true_ratio >= 0.98，说明 Embedding 匹配效果好
    # 如果 review_true_ratio 比之前高，说明语义匹配把更多真匹配拉到了审核队列
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
    logger.info("客户主数据匹配管道 v2.0 (Embedding 增强)")
    logger.info("=" * 50)

    # 1. 加载数据
    logger.info("[1/5] 加载客户数据 ...")
    sources = load_customers()

    # 2. 预计算 Embedding
    logger.info("[2/5] 预计算名称 Embedding ...")
    emb_map = precompute_name_embeddings(sources)

    # 3. 生成候选对 & 打分
    logger.info("[3/5] 生成候选匹配对并打分 ...")
    t_candidates = time.time()
    candidates = build_candidate_pairs(sources, emb_map)
    logger.info("候选对生成耗时: %.1f 秒", time.time() - t_candidates)

    # 4. 分类 & 保存
    logger.info("[4/5] 分类并保存结果 ...")
    candidates = categorize_matches(candidates, "customer")
    save_candidate_files(candidates)
    golden = build_golden_customers(sources)

    # 5. 评估指标
    logger.info("[5/5] 计算评估指标 ...")
    metrics = compute_metrics(candidates, golden)

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

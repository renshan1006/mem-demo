"""
审核反馈回训 — 基于人工决策 + 真值标签自动优化匹配阈值
========================================================
策略：
  1. 加载候选匹配文件（含 is_true_match 真值标签）
  2. 加载审核日志（人工 merge/not-merge 决策）
  3. 在 [0.3, 0.95] 区间以 0.01 步长扫描所有可能的阈值组合
  4. 优化目标：auto_merge 精确率 ≥ 95%，最小化 review 队列
  5. 输出 optimized_thresholds.json 供管道读取
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CONFIG_FILE = DATA_DIR / "optimized_thresholds.json"

CANDIDATE_FILES = {
    "customer": DATA_DIR / "customer_match_candidates.csv",
    "product": DATA_DIR / "product_match_candidates.csv",
}
LOG_FILES = {
    "customer": DATA_DIR / "customer_review_log.csv",
    "product": DATA_DIR / "product_review_log.csv",
}

# 默认阈值（优化前的基线）
DEFAULT_THRESHOLDS = {
    "auto_merge": 0.80,
    "review_lower": 0.50,
}
# 约束条件
MIN_AUTO_PRECISION = 0.95  # auto_merge 最低精确率
MAX_REVIEW_FNR = 0.20      # review 区间最大假阴性率（漏掉的真匹配）
SCAN_RANGE = np.arange(0.30, 0.96, 0.01)


def load_data(entity: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """加载候选数据 + 审核日志"""
    cand_file = CANDIDATE_FILES[entity]
    if not cand_file.exists():
        raise FileNotFoundError(f"候选文件不存在: {cand_file}")

    candidates = pd.read_csv(cand_file, dtype=str).fillna("")
    candidates["match_score"] = pd.to_numeric(candidates["match_score"], errors="coerce")
    candidates["is_true_match"] = candidates["is_true_match"].astype(str).str.lower() == "true"

    log_file = LOG_FILES[entity]
    if log_file.exists():
        review_log = pd.read_csv(log_file, dtype=str).fillna("")
    else:
        review_log = pd.DataFrame(columns=["record_id_left", "record_id_right", "decision", "match_score"])

    return candidates, review_log


def analyze_review_feedback(review_log: pd.DataFrame) -> dict:
    """分析人工审核模式的统计信息"""
    if review_log.empty:
        return {"total_decisions": 0, "merge_scores": [], "not_merge_scores": [], "pending_scores": []}

    merge_scores = pd.to_numeric(
        review_log[review_log["decision"] == "合并"]["match_score"], errors="coerce"
    ).tolist()
    not_merge_scores = pd.to_numeric(
        review_log[review_log["decision"] == "不合并"]["match_score"], errors="coerce"
    ).tolist()
    pending_scores = pd.to_numeric(
        review_log[review_log["decision"] == "保留待定"]["match_score"], errors="coerce"
    ).tolist()

    return {
        "total_decisions": len(review_log),
        "merge_count": len(merge_scores),
        "not_merge_count": len(not_merge_scores),
        "pending_count": len(pending_scores),
        "merge_scores": merge_scores,
        "not_merge_scores": not_merge_scores,
        "merge_mean": float(np.mean(merge_scores)) if merge_scores else None,
        "not_merge_mean": float(np.mean(not_merge_scores)) if not_merge_scores else None,
    }


def find_best_thresholds(candidates: pd.DataFrame, review_log: pd.DataFrame) -> dict:
    """
    扫描阈值空间，寻找最优配置。

    优化目标（按优先级）：
      1. auto_merge 精确率 ≥ MIN_AUTO_PRECISION
      2. review 区间假阴性率 ≤ MAX_REVIEW_FNR
      3. review 队列尽量小 → 最小化人工工作量
    """
    scores = candidates["match_score"].values
    labels = candidates["is_true_match"].values
    n_total = len(candidates)
    n_true = int(labels.sum())

    logger.info("  总候选对: %d, 真匹配: %d (%.1f%%)", n_total, n_true, n_true / n_total * 100)

    best = None
    best_score = -1.0

    results = []
    for auto_thresh in SCAN_RANGE:
        for review_lower in SCAN_RANGE:
            if review_lower >= auto_thresh:
                continue  # review 下限必须 < auto_merge 上限

            # 分区
            auto_mask = scores >= auto_thresh
            review_mask = (scores >= review_lower) & (scores < auto_thresh)
            no_match_mask = scores < review_lower

            n_auto = int(auto_mask.sum())
            n_review = int(review_mask.sum())
            n_no = int(no_match_mask.sum())

            if n_auto == 0:
                continue

            # auto_merge 精确率
            auto_tp = int((auto_mask & labels).sum())
            auto_precision = auto_tp / n_auto

            if auto_precision < MIN_AUTO_PRECISION:
                continue  # 不满足最低精确率

            # review 区间假阴性（漏掉的真匹配）
            review_tp = int((review_mask & labels).sum())
            review_fn = int((review_mask & ~labels).sum())  # 实际是 FP，这里关注遗漏

            # no_match 区间假阴性（真匹配被判定为不同）
            no_match_fn = int((no_match_mask & labels).sum())
            no_match_fnr = no_match_fn / n_true if n_true > 0 else 0

            if no_match_fnr > MAX_REVIEW_FNR:
                continue  # 遗漏太多真匹配

            # 综合评分：auto_precision 高 + review 小 + fnr 低
            # 我们希望 auto_merge 多覆盖真匹配，review 尽量小
            review_ratio = n_review / n_total
            auto_recall = auto_tp / n_true if n_true > 0 else 0

            score = (
                auto_precision * 0.40          # 合并准确性最重要
                + auto_recall * 0.30           # 合并覆盖率
                + (1.0 - review_ratio) * 0.20 # 审核队列小
                + (1.0 - no_match_fnr) * 0.10 # 假阴性少
            )

            results.append({
                "auto_threshold": round(auto_thresh, 2),
                "review_lower": round(review_lower, 2),
                "auto_count": n_auto,
                "review_count": n_review,
                "no_match_count": n_no,
                "auto_precision": round(auto_precision, 4),
                "auto_recall": round(auto_recall, 4),
                "no_match_fnr": round(no_match_fnr, 4),
                "review_ratio": round(review_ratio, 4),
                "composite_score": round(score, 4),
            })

            if score > best_score:
                best_score = score
                best = results[-1]

    # 排序：综合分降序
    results.sort(key=lambda r: r["composite_score"], reverse=True)

    return {
        "best": best,
        "top5": results[:5],
        "baseline": _baseline_metrics(candidates, 0.80, 0.50),
        "search_space": len(results),
    }


def _baseline_metrics(candidates: pd.DataFrame, auto_t: float, review_low: float) -> dict:
    """计算基线指标"""
    scores = candidates["match_score"].values
    labels = candidates["is_true_match"].values

    auto_mask = scores >= auto_t
    review_mask = (scores >= review_low) & (scores < auto_t)

    auto_tp = int((auto_mask & labels).sum())
    return {
        "auto_threshold": auto_t,
        "review_lower": review_low,
        "auto_count": int(auto_mask.sum()),
        "review_count": int(review_mask.sum()),
        "auto_precision": round(auto_tp / max(int(auto_mask.sum()), 1), 4),
    }


def human_insight_adjustment(best: dict, review_feedback: dict) -> dict:
    """
    用人工审核数据微调阈值：
    - 如果人工「合并」的分数中位数低于 auto_threshold → 降低 auto 阈值
    - 如果人工「不合并」的分数中位数高于 review_lower → 提高 review 下限
    """
    if not best:
        return best

    adjusted = dict(best)
    merge_scores = review_feedback.get("merge_scores", [])
    not_merge_scores = review_feedback.get("not_merge_scores", [])

    if merge_scores:
        merge_median = np.median(merge_scores)
        if merge_median < adjusted["auto_threshold"]:
            # 人工倾向于在更低分数就合并 → 降低阈值
            new_auto = max(adjusted["review_lower"] + 0.05, merge_median - 0.03)
            logger.info(
                "  人工合并中位数 %.4f < 当前阈值 %.2f → 建议降至 %.2f",
                merge_median, adjusted["auto_threshold"], new_auto,
            )
            adjusted["auto_threshold"] = round(new_auto, 2)
            adjusted["human_adjusted"] = True

    if not_merge_scores:
        not_merge_median = np.median(not_merge_scores)
        if not_merge_median > adjusted["review_lower"]:
            # 人工倾向于在更高分数就不合并 → 提高 review 下限
            new_lower = min(adjusted["auto_threshold"] - 0.05, not_merge_median + 0.02)
            logger.info(
                "  人工不合并中位数 %.4f > 当前下限 %.2f → 建议升至 %.2f",
                not_merge_median, adjusted["review_lower"], new_lower,
            )
            adjusted["review_lower"] = round(new_lower, 2)
            adjusted["human_adjusted"] = True

    return adjusted


def save_config(entity: str, thresholds: dict, metrics: dict) -> None:
    """保存优化后的配置"""
    config = {
        "entity": entity,
        "auto_merge_threshold": thresholds["auto_threshold"],
        "review_lower_threshold": thresholds["review_lower"],
        "human_adjusted": thresholds.get("human_adjusted", False),
        "metrics": {
            "auto_precision": thresholds.get("auto_precision"),
            "auto_count": thresholds.get("auto_count"),
            "review_count": thresholds.get("review_count"),
        },
        "scan_details": {
            "search_space": metrics.get("search_space"),
            "top_candidates": metrics.get("top5", []),
        },
    }

    # 如果已有配置则合并（保留另一个 entity 的配置）
    existing = {}
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            existing = json.load(f)

    existing[entity] = config

    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)

    logger.info("配置已保存: %s → %s", entity, CONFIG_FILE.name)


def print_report(entity: str, review_feedback: dict, result: dict, final: dict) -> None:
    """打印可读报告"""
    print()
    print("=" * 60)
    print(f"  {entity} 阈值优化报告")
    print("=" * 60)

    # 审核数据
    fb = review_feedback
    print(f"\n[Review] 审核反馈数据:")
    print(f"  总决策数: {fb['total_decisions']}")
    print(f"  合并: {fb['merge_count']} 条 (中位数 {fb['merge_mean'] or 'N/A'})")
    print(f"  不合并: {fb['not_merge_count']} 条 (中位数 {fb['not_merge_mean'] or 'N/A'})")

    # 基线
    bl = result["baseline"]
    print(f"\n[BASELINE] (auto>={bl['auto_threshold']}, review>={bl['review_lower']}):")
    print(f"  auto_merge: {bl['auto_count']} 条, 精确率 {bl['auto_precision']:.1%}")

    # 最优
    b = result["best"]
    print(f"\n[BEST] 最优配置:")
    print(f"  auto_merge 阈值: ≥ {b['auto_threshold']:.2f}")
    print(f"  review 下限:     ≥ {b['review_lower']:.2f}")
    print(f"  auto_merge: {b['auto_count']} 条, 精确率 {b['auto_precision']:.1%}, 召回率 {b['auto_recall']:.1%}")
    print(f"  review:     {b['review_count']} 条 ({b['review_ratio']:.1%})")
    print(f"  no_match:   {b['no_match_count']} 条, 假阴性率 {b['no_match_fnr']:.1%}")
    print(f"  综合评分:   {b['composite_score']:.4f}")

    # 最终（含人工微调）
    if final.get("human_adjusted"):
        print(f"\n[ADJUSTED] 人工微调后:")
        print(f"  auto_merge 阈值: ≥ {final['auto_threshold']:.2f}")
        print(f"  review 下限:     ≥ {final['review_lower']:.2f}")

    # 对比
    delta_auto = final.get("auto_count", b["auto_count"]) - bl["auto_count"]
    delta_review = final.get("review_count", b["review_count"]) - bl["review_count"]
    print(f"\n[DELTA] 相比基线:")
    print(f"  auto_merge: {delta_auto:+d} 条")
    print(f"  review:     {delta_review:+d} 条")
    print("=" * 60)


def main() -> None:
    for entity in ["customer", "product"]:
        logger.info("=" * 50)
        logger.info("分析实体: %s", entity)

        # 1. 加载数据
        candidates, review_log = load_data(entity)
        review_feedback = analyze_review_feedback(review_log)
        logger.info(
            "审核日志: %d 条决策 (合并 %d / 不合并 %d)",
            review_feedback["total_decisions"],
            review_feedback["merge_count"],
            review_feedback["not_merge_count"],
        )

        # 2. 扫描最优阈值
        logger.info("扫描阈值空间 (%d 个组合)...", len(SCAN_RANGE) ** 2)
        result = find_best_thresholds(candidates, review_log)

        if result["best"] is None:
            logger.warning("未找到满足约束的阈值组合，使用默认值。")
            continue

        # 3. 人工决策微调
        final_thresholds = human_insight_adjustment(result["best"], review_feedback)

        # 4. 保存
        save_config(entity, final_thresholds, result)

        # 5. 报告
        print_report(entity, review_feedback, result, final_thresholds)

    print(f"\n[DONE] 阈值配置文件: {CONFIG_FILE}")


if __name__ == "__main__":
    main()

"""
DataGuardian — 统一评价框架

用途:
    对所有智能体的输出做统一评价，对比 Baseline 和 Agent 方法。

支持的评价类型:
    - 匹配任务 (entity matching): Precision / Recall / F1 / Accuracy
    - 质量任务 (data quality): 检出率 / 误报率
    - 标准化任务 (normalization): 准确率 / 覆盖率

用法:
    from scripts.evaluator import Evaluator, EvalResult

    eval = Evaluator(ground_truth=df_labels)
    result = eval.evaluate(predictions, name="baseline")
    print(result.summary())
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import json
import time


@dataclass
class EvalResult:
    """单次评价结果"""
    name: str                                    # 方法名称（如 "baseline", "siamese", "bert"）
    task_type: str = "matching"                  # 任务类型
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    accuracy: float = 0.0
    total_samples: int = 0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    true_negatives: int = 0
    elapsed_ms: float = 0.0
    extra_metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def f1_percent(self) -> str:
        return f"{self.f1 * 100:.1f}%"

    @property
    def precision_percent(self) -> str:
        return f"{self.precision * 100:.1f}%"

    @property
    def recall_percent(self) -> str:
        return f"{self.recall * 100:.1f}%"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "task_type": self.task_type,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "accuracy": round(self.accuracy, 4),
            "total_samples": self.total_samples,
            "tp": self.true_positives,
            "fp": self.false_positives,
            "fn": self.false_negatives,
            "tn": self.true_negatives,
            "elapsed_ms": round(self.elapsed_ms, 1),
            **self.extra_metrics,
        }

    def summary(self) -> str:
        lines = [
            f"====== {self.name} ({self.task_type}) ======",
            f"  samples: {self.total_samples}",
            f"  Precision: {self.precision_percent}",
            f"  Recall:    {self.recall_percent}",
            f"  F1:        {self.f1_percent}",
            f"  Accuracy:  {self.accuracy * 100:.1f}%",
            f"  TP={self.true_positives}  FP={self.false_positives}",
            f"  FN={self.false_negatives}  TN={self.true_negatives}",
        ]
        if self.extra_metrics:
            lines.append("  ---")
            for k, v in self.extra_metrics.items():
                lines.append(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        return "\n".join(lines)


class Evaluator:
    """
    统一评价器。

    用法:
        eval = Evaluator()

        # 方式1：从真值标签评价匹配结果
        eval.set_ground_truth(ground_truth_pairs)
        result = eval.evaluate_matching(predictions, name="baseline")

        # 方式2：批量对比
        comparison = eval.compare({
            "baseline": baseline_preds,
            "agent_v2": agent_preds,
        })
    """

    def __init__(self, ground_truth: Any = None):
        """
        参数:
            ground_truth: 真值数据。
                匹配任务: set of (id_a, id_b) 表示真正匹配的对
                质量任务: DataFrame with 'issue_type' and 'found' columns
        """
        self._ground_truth = ground_truth
        self._results: List[EvalResult] = []

    def set_ground_truth(self, ground_truth: Any) -> None:
        """设置真值数据"""
        self._ground_truth = ground_truth

    # ── 匹配任务评价 ───────────────────────────────

    def evaluate_matching(self,
                          predictions: set,
                          name: str = "model",
                          total_candidates: int = 0,
                          elapsed_ms: float = 0.0,
                          **extra) -> EvalResult:
        """
        评价实体匹配结果。

        参数:
            predictions: set of (id_a, id_b) — 模型判定为"匹配"的对
            name: 方法名称
            total_candidates: 候选对总数（用于计算 TN）
            elapsed_ms: 执行耗时

        返回:
            EvalResult
        """
        if self._ground_truth is None:
            raise ValueError("请先调用 set_ground_truth() 设置真值")

        gt = self._ground_truth  # set of true match pairs
        pred = set(predictions)

        tp = len(pred & gt)       # 正确识别的匹配
        fp = len(pred - gt)       # 误报（实际不匹配但判定匹配）
        fn = len(gt - pred)       # 漏报（实际匹配但没识别出来）

        total = len(pred | gt)
        precision = tp / len(pred) if len(pred) > 0 else 0.0
        recall = tp / len(gt) if len(gt) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # Accuracy 需要 TN
        tn = total_candidates - tp - fp - fn if total_candidates > 0 else 0
        accuracy = (tp + tn) / total_candidates if total_candidates > 0 else 0.0

        result = EvalResult(
            name=name,
            task_type="matching",
            precision=precision,
            recall=recall,
            f1=f1,
            accuracy=accuracy,
            total_samples=len(gt),
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            true_negatives=tn,
            elapsed_ms=elapsed_ms,
            extra_metrics=extra,
        )
        self._results.append(result)
        return result

    # ── 分类任务评价（质量检测、标准化判定等）───────

    def evaluate_classification(self,
                                y_true: List[int],
                                y_pred: List[int],
                                name: str = "model",
                                elapsed_ms: float = 0.0,
                                **extra) -> EvalResult:
        """
        评价分类结果。

        参数:
            y_true: 真实标签列表 (1=正例, 0=负例)
            y_pred: 预测标签列表
        """
        if len(y_true) != len(y_pred):
            raise ValueError(f"长度不一致: y_true={len(y_true)}, y_pred={len(y_pred)}")

        tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
        tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)

        total = len(y_true)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / total if total > 0 else 0.0

        result = EvalResult(
            name=name,
            task_type="classification",
            precision=precision,
            recall=recall,
            f1=f1,
            accuracy=accuracy,
            total_samples=total,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            true_negatives=tn,
            elapsed_ms=elapsed_ms,
            extra_metrics=extra,
        )
        self._results.append(result)
        return result

    # ── 批量对比 ────────────────────────────────────

    def compare(self,
                predictions_map: Dict[str, set],
                total_candidates: int = 0) -> List[EvalResult]:
        """
        批量对比多个方法。

        参数:
            predictions_map: {方法名: 预测匹配对集合}
            total_candidates: 候选对总数

        返回:
            [EvalResult, ...]  按 F1 降序排列

        用法:
            results = evaluator.compare({
                "baseline": baseline_preds,
                "agent_v2": agent_preds,
                "agent_v3": agent_v3_preds,
            })
        """
        results = []
        for name, preds in predictions_map.items():
            result = self.evaluate_matching(preds, name=name,
                                            total_candidates=total_candidates)
            results.append(result)

        results.sort(key=lambda r: r.f1, reverse=True)
        return results

    # ── 报告 ────────────────────────────────────────

    def report(self) -> str:
        """生成多方法对比报告"""
        if not self._results:
            return "(暂无评价结果)"

        lines = ["",
                  "====== DataGuardian Evaluation Report ======", ""]

        # 表头
        header = f"{'Method':<20} {'Prec':>8} {'Recall':>8} {'F1':>8} {'Acc':>8} {'Time':>8}"
        lines.append(header)
        lines.append("-" * 65)

        for r in sorted(self._results, key=lambda x: x.f1, reverse=True):
            row = (f"{r.name:<20} {r.precision_percent:>8} {r.recall_percent:>8} "
                   f"{r.f1_percent:>8} {r.accuracy*100:>7.1f}% {r.elapsed_ms:>7.0f}ms")
            lines.append(row)

        lines.append("")
        return "\n".join(lines)

    def to_dataframe(self):
        """转为 pandas DataFrame（给 Streamlit 用）"""
        import pandas as pd
        return pd.DataFrame([r.to_dict() for r in self._results])

    def save(self, path: str) -> None:
        """保存评价结果到 JSON"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in self._results], f, ensure_ascii=False, indent=2)

    def clear(self) -> None:
        """清除历史结果"""
        self._results.clear()


# ── 命令行入口 ───────────────────────────────────────────────

if __name__ == "__main__":
    # 演示：用假数据跑一次评价
    print("Evaluator Demo\n")

    gt = {("A", "B"), ("C", "D"), ("E", "F"), ("G", "H"), ("I", "J")}  # 5对真匹配

    baseline_preds = {("A", "B"), ("E", "F"), ("X", "Y")}   # 找到2对真的 + 1对误报
    agent_preds = {("A", "B"), ("C", "D"), ("E", "F"), ("G", "H")}  # 找到4对真的

    evaluator = Evaluator(ground_truth=gt)
    evaluator.evaluate_matching(baseline_preds, name="Baseline (difflib)",
                                total_candidates=100)
    evaluator.evaluate_matching(agent_preds, name="Agent (Embed+LLM)",
                                total_candidates=100)

    print(evaluator.report())

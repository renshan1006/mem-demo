"""
DataGuardian — 实体匹配智能体

封装现有的客户/商品匹配管道为 Agent 接口。
这是组员的参考范例：看我怎么把已有代码包装成 Agent。

用法:
    from scripts.match_agent import EntityMatchAgent
    from scripts.agent_registry import registry

    agent = EntityMatchAgent()
    registry.register(agent)

    result = registry.call("entity_match", AgentTask(
        action="match_customers",
        params={"use_dify": True},
    ))
"""

import sys
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd

# 把 scripts 加入路径，以便调用兄弟模块
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

# 兼容直接运行和模块导入两种方式
try:
    from .base_agent import BaseAgent, AgentTask, AgentResult
except ImportError:
    from base_agent import BaseAgent, AgentTask, AgentResult

logger = logging.getLogger(__name__)

BASE_DIR = _THIS_DIR.parent
DATA_DIR = BASE_DIR / "data"


class EntityMatchAgent(BaseAgent):
    """
    实体匹配智能体 — 多策略实体匹配 + LLM辅助审核。

    支持的动作:
        - match_customers : 运行客户匹配管道
        - match_products  : 运行商品匹配管道
        - metrics         : 返回当前匹配指标
        - run_dify        : 对审核队列运行 Dify LLM 增强
        - status          : 返回系统状态摘要
    """

    name = "entity_match"
    description = "多策略实体匹配智能体（Embedding语义+规则加权+Dify LLM辅助）"

    def get_actions(self) -> List[str]:
        return ["match_customers", "match_products", "metrics", "run_dify", "status"]

    def health_check(self) -> bool:
        """检查 embedder 和数据文件是否可用"""
        try:
            from embedder import get_embedder
            emb = get_embedder()
            if not emb.available:
                logger.warning("Embedding 模型不可用，降级到编辑距离模式")
                # 不返回 False — 系统有降级方案，仍然可用
            return True
        except Exception:
            return False

    def run(self, task: AgentTask) -> AgentResult:
        action = task.action

        if action == "match_customers":
            return self._match_customers(task)
        elif action == "match_products":
            return self._match_products(task)
        elif action == "metrics":
            return self._get_metrics(task)
        elif action == "run_dify":
            return self._run_dify(task)
        elif action == "status":
            return self._status(task)
        else:
            return self._fail(
                error=f"未知动作: '{action}'",
                summary=f"支持的动作: {self.get_actions()}",
            )

    # ── 客户匹配 ─────────────────────────────────────

    def _match_customers(self, task: AgentTask) -> AgentResult:
        """运行客户匹配管道全流程"""
        try:
            import mdm_pipeline
        except ImportError as e:
            return self._fail(error=f"无法导入 mdm_pipeline: {e}")

        use_dify = task.params.get("use_dify", False)
        t0 = time.time()

        try:
            sources = mdm_pipeline.load_customers()
            emb_map = mdm_pipeline.precompute_name_embeddings(sources)
            candidates = mdm_pipeline.build_candidate_pairs(sources, emb_map)
            candidates = mdm_pipeline.categorize_matches(candidates, "customer")
            mdm_pipeline.save_candidate_files(candidates)
            golden = mdm_pipeline.build_golden_customers(sources)
            metrics = mdm_pipeline.compute_metrics(candidates, golden)

            if use_dify:
                dify_result = self._run_dify_enrich("customer",
                    max_samples=task.params.get("dify_max_samples"))
                metrics["dify"] = dify_result

            elapsed = time.time() - t0

            return self._ok(
                data=metrics,
                summary=f"客户匹配完成: {metrics['auto_merge_count']} 自动合并, "
                       f"{metrics['review_count']} 需审核",
                details={
                    "entity_type": "customer",
                    "total_candidates": metrics["total_candidates"],
                    "auto_merge": metrics["auto_merge_count"],
                    "review": metrics["review_count"],
                    "no_match": metrics["no_match_count"],
                    "auto_merge_true_ratio": f"{metrics['auto_merge_true_ratio']:.1%}",
                    "golden_entities": len(golden),
                    "elapsed_seconds": round(elapsed, 1),
                },
                elapsed_ms=elapsed * 1000,
            )

        except FileNotFoundError as e:
            return self._fail(error=str(e), summary="数据文件缺失，请先运行 generate_dataset.py")
        except Exception as e:
            return self._fail(error=str(e), summary="客户匹配管道执行失败")

    # ── 商品匹配 ─────────────────────────────────────

    def _match_products(self, task: AgentTask) -> AgentResult:
        """运行商品匹配管道全流程"""
        try:
            import product_pipeline
        except ImportError as e:
            return self._fail(error=f"无法导入 product_pipeline: {e}")

        t0 = time.time()

        try:
            sources = product_pipeline.load_products()
            emb_map = product_pipeline.precompute_product_embeddings(sources)
            candidates = product_pipeline.build_product_candidates(sources, emb_map)
            candidates = product_pipeline.categorize_matches(candidates, "product")
            product_pipeline.save_product_files(candidates)
            golden = product_pipeline.build_golden_products(sources)
            metrics = product_pipeline.compute_product_metrics(candidates, golden)

            elapsed = time.time() - t0

            return self._ok(
                data=metrics,
                summary=f"商品匹配完成: {metrics['auto_merge_count']} 自动合并, "
                       f"{metrics['review_count']} 需审核",
                details={
                    "entity_type": "product",
                    "total_candidates": metrics["total_candidates"],
                    "auto_merge": metrics["auto_merge_count"],
                    "review": metrics["review_count"],
                    "golden_entities": len(golden),
                    "elapsed_seconds": round(elapsed, 1),
                },
                elapsed_ms=elapsed * 1000,
            )

        except Exception as e:
            return self._fail(error=str(e), summary="商品匹配管道执行失败")

    # ── Dify 增强 ────────────────────────────────────

    def _run_dify(self, task: AgentTask) -> AgentResult:
        """对现有审核队列运行 Dify LLM 增强"""
        entity_type = task.params.get("entity_type", "customer")
        max_samples = task.params.get("max_samples")
        return self._run_dify_enrich(entity_type, max_samples)

    def _run_dify_enrich(self, entity_type: str,
                         max_samples: int = None) -> Dict[str, Any]:
        """内部：调用 dify_enrich 模块"""
        try:
            import dify_enrich
        except ImportError:
            return {"error": "dify_enrich 模块不可用"}
        except Exception as e:
            return {"error": str(e)}

        try:
            dify_enrich.main(entity=entity_type, max_samples=max_samples)
            return {
                "status": "done",
                "entity_type": entity_type,
                "max_samples": max_samples,
            }
        except Exception as e:
            return {"error": str(e), "status": "failed"}

    # ── 指标查询 ─────────────────────────────────────

    def _get_metrics(self, task: AgentTask) -> AgentResult:
        """读取已保存的匹配指标"""
        all_metrics = {}

        customer_metrics_path = DATA_DIR / "customer_match_metrics.json"
        if customer_metrics_path.exists():
            with open(customer_metrics_path, "r", encoding="utf-8") as f:
                all_metrics["customer"] = json.load(f)

        product_metrics_path = DATA_DIR / "product_match_metrics.json"
        if product_metrics_path.exists():
            with open(product_metrics_path, "r", encoding="utf-8") as f:
                all_metrics["product"] = json.load(f)

        final_customer_path = DATA_DIR / "final_customer_metrics.json"
        if final_customer_path.exists():
            with open(final_customer_path, "r", encoding="utf-8") as f:
                all_metrics["final_customer"] = json.load(f)

        if not all_metrics:
            return self._fail(error="未找到任何指标文件", summary="请先运行匹配管道")

        return self._ok(
            data=all_metrics,
            summary=f"已加载 {len(all_metrics)} 套指标",
        )

    # ── 状态 ─────────────────────────────────────────

    def _status(self, task: AgentTask) -> AgentResult:
        """返回系统状态摘要"""
        files_status = {}
        for f in ["crm_customers.csv", "customer_match_candidates.csv",
                   "customer_review_queue.csv", "golden_customers.csv",
                   "final_golden_customers.csv"]:
            path = DATA_DIR / f
            files_status[f] = "exists" if path.exists() else "missing"

        # 简单统计
        review_count = 0
        review_path = DATA_DIR / "customer_review_queue.csv"
        if review_path.exists():
            try:
                df = pd.read_csv(review_path)
                review_count = len(df)
            except Exception:
                pass

        return self._ok(
            data={"files": files_status, "pending_review": review_count},
            summary=f"数据文件就绪，待审核 {review_count} 条",
        )


# ── 命令行入口（方便组员测试）───────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    print("EntityMatchAgent 测试\n")

    agent = EntityMatchAgent()
    print(f"名称: {agent.name}")
    print(f"描述: {agent.description}")
    print(f"动作: {agent.get_actions()}")
    print(f"健康: {agent.health_check()}")
    print()

    # 测试状态查询（不跑完整管道，快）
    result = agent.run(AgentTask(action="status"))
    print(f"status 结果: {result.summary}")
    print(f"  success={result.success}")
    print(f"  data={result.data}")
    print(f"  elapsed={result.elapsed_ms:.0f}ms")

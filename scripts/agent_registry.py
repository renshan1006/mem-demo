"""
DataGuardian — 智能体注册与调度中心

功能:
    1. 注册/注销智能体
    2. 按名称路由任务
    3. 健康检查
    4. 调用日志记录
    5. 列出所有可用智能体

用法:
    from scripts.agent_registry import registry

    # 注册
    registry.register(my_agent)

    # 调用
    result = registry.call("entity_match", AgentTask(action="match", params={...}))

    # 查看
    print(registry.summary())
"""

from typing import Dict, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
import logging

try:
    from .base_agent import BaseAgent, AgentTask, AgentResult
except ImportError:
    from base_agent import BaseAgent, AgentTask, AgentResult

logger = logging.getLogger(__name__)


@dataclass
class CallRecord:
    """单次调用记录"""
    agent_name: str
    action: str
    success: bool
    summary: str
    elapsed_ms: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


class AgentRegistry:
    """
    智能体注册中心（单例模式）。

    职责:
        - 维护 name → Agent 的映射
        - 提供统一的 call() 入口
        - 记录调用历史
    """

    def __init__(self):
        self._agents: Dict[str, BaseAgent] = {}
        self._history: List[CallRecord] = []

    # ── 注册 / 注销 ─────────────────────────────────

    def register(self, agent: BaseAgent) -> None:
        """注册一个智能体。如果同名已存在则覆盖。"""
        if not isinstance(agent, BaseAgent):
            raise TypeError(f"agent 必须继承 BaseAgent，实际类型: {type(agent)}")
        if agent.name in self._agents:
            logger.warning(f"智能体 '{agent.name}' 已存在，将被覆盖")
        self._agents[agent.name] = agent
        logger.info(f"智能体已注册: {agent.name} — {agent.description}")

    def unregister(self, name: str) -> bool:
        """注销一个智能体。返回是否成功。"""
        if name in self._agents:
            del self._agents[name]
            logger.info(f"智能体已注销: {name}")
            return True
        return False

    # ── 查询 ────────────────────────────────────────

    def get(self, name: str) -> Optional[BaseAgent]:
        """按名称获取智能体"""
        return self._agents.get(name)

    def list_all(self) -> List[BaseAgent]:
        """列出所有已注册的智能体"""
        return list(self._agents.values())

    def list_names(self) -> List[str]:
        """列出所有智能体名称"""
        return list(self._agents.keys())

    def summary(self) -> str:
        """返回多行摘要，供终端或 UI 展示"""
        if not self._agents:
            return "(暂无已注册的智能体)"
        lines = [f"已注册 {len(self._agents)} 个智能体:"]
        for agent in self._agents.values():
            actions = agent.get_actions()
            healthy = "OK" if agent.health_check() else "!!"
            extra = f" | 动作: {', '.join(actions)}" if actions else ""
            lines.append(f"  [{healthy}] {agent.name} — {agent.description}{extra}")
        return "\n".join(lines)

    # ── 调用 ────────────────────────────────────────

    def call(self, agent_name: str, task: AgentTask) -> AgentResult:
        """
        调用指定智能体。

        参数:
            agent_name: 智能体名称
            task: AgentTask 对象

        返回:
            AgentResult 对象。如果智能体不存在，返回 success=False 的结果。
        """
        agent = self._agents.get(agent_name)
        if agent is None:
            msg = f"智能体 '{agent_name}' 未注册。可用: {self.list_names()}"
            logger.error(msg)
            return AgentResult(success=False, summary="智能体未找到", error=msg)

        logger.info(f"调用 [{agent_name}] → {task.action}")
        result = agent._timed_run(task)

        # 记录历史
        self._history.append(CallRecord(
            agent_name=agent_name,
            action=task.action,
            success=result.success,
            summary=result.summary,
            elapsed_ms=result.elapsed_ms,
        ))

        return result

    def call_safe(self, agent_name: str, task: AgentTask,
                  fallback: Optional[AgentResult] = None) -> AgentResult:
        """
        安全调用：即使智能体不存在或报错也返回结果（不抛异常）。

        参数:
            agent_name: 智能体名称
            task: AgentTask 对象
            fallback: 智能体不存在时的默认返回值
        """
        try:
            return self.call(agent_name, task)
        except Exception as e:
            logger.exception(f"调用 [{agent_name}] 时发生未捕获异常")
            return fallback or AgentResult(
                success=False,
                summary=f"调用 [{agent_name}] 异常",
                error=str(e),
            )

    # ── 健康检查 ────────────────────────────────────

    def health_check_all(self) -> Dict[str, bool]:
        """对所有已注册智能体执行健康检查"""
        return {name: agent.health_check() for name, agent in self._agents.items()}

    def healthy_agents(self) -> List[str]:
        """返回所有健康智能体的名称"""
        return [name for name, ok in self.health_check_all().items() if ok]

    # ── 历史 ────────────────────────────────────────

    def recent_calls(self, n: int = 20) -> List[CallRecord]:
        """返回最近 n 条调用记录"""
        return self._history[-n:]

    def call_stats(self) -> Dict[str, int]:
        """按智能体统计调用次数"""
        stats: Dict[str, int] = {}
        for r in self._history:
            stats[r.agent_name] = stats.get(r.agent_name, 0) + 1
        return stats

    # ── 重置 ────────────────────────────────────────

    def clear(self) -> None:
        """清空所有注册和记录（仅用于测试）"""
        self._agents.clear()
        self._history.clear()


# ── 全局单例 ────────────────────────────────────────────────

registry = AgentRegistry()

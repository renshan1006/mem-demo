"""
DataGuardian — 智能体统一接口基类

所有智能体必须继承 BaseAgent 并实现 name、description、run() 三个成员。
这是整个系统的"接口合约"，组长维护，组员遵守。
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import time
import logging

logger = logging.getLogger(__name__)


# ── 统一数据结构 ────────────────────────────────────────────

@dataclass
class AgentTask:
    """发给智能体的任务对象"""
    action: str                          # 动作名，如 "scan", "match", "normalize"
    params: Dict[str, Any] = field(default_factory=dict)   # 动作参数
    data: Any = None                     # 可选：输入数据（DataFrame / dict / list）
    context: Dict[str, Any] = field(default_factory=dict)  # 可选：上下文信息


@dataclass
class AgentResult:
    """智能体返回的结果对象"""
    success: bool
    data: Any = None                     # 核心输出
    summary: str = ""                    # 一句话结果描述
    details: Dict[str, Any] = field(default_factory=dict)  # 详细信息（给前端展示）
    error: Optional[str] = None          # 错误信息
    elapsed_ms: float = 0.0              # 执行耗时（毫秒）


# ── 基类 ────────────────────────────────────────────────────

class BaseAgent(ABC):
    """
    所有智能体的统一基类。

    子类必须实现:
        name        — 唯一标识，如 "entity_match"
        description — 功能描述，一句话
        run(task)   — 执行业务逻辑，返回 AgentResult

    子类可选实现:
        health_check()  — 健康检查
        get_actions()   — 返回支持的动作列表

    用法示例:
        class MyAgent(BaseAgent):
            name = "my_agent"
            description = "示例智能体"

            def run(self, task: AgentTask) -> AgentResult:
                return AgentResult(success=True, summary="完成")
    """

    # ── 子类必须覆盖 ──
    @property
    @abstractmethod
    def name(self) -> str:
        """智能体唯一名称。命名规则：小写字母 + 下划线，如 'entity_match'"""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """一句话功能描述"""
        ...

    @abstractmethod
    def run(self, task: AgentTask) -> AgentResult:
        """
        执行任务。

        参数:
            task: AgentTask 对象，包含 action/params/data/context

        返回:
            AgentResult 对象，必须设置 success=True/False
        """
        ...

    # ── 子类可选覆盖 ──
    def health_check(self) -> bool:
        """健康检查。返回 True 表示可用，False 表示不可用。"""
        return True

    def get_actions(self) -> List[str]:
        """返回支持的动作列表。默认返回空，子类按需覆盖。"""
        return []

    # ── 基类提供的工具方法 ──
    def _ok(self, data: Any = None,
            summary: str = "完成",
            details: Optional[Dict[str, Any]] = None,
            elapsed_ms: float = 0.0) -> AgentResult:
        """快捷构造成功结果"""
        return AgentResult(
            success=True,
            data=data,
            summary=summary,
            details=details or {},
            elapsed_ms=elapsed_ms,
        )

    def _fail(self, error: str,
              summary: str = "执行失败",
              data: Any = None) -> AgentResult:
        """快捷构造失败结果"""
        return AgentResult(
            success=False,
            data=data,
            summary=summary,
            error=error,
        )

    def _timed_run(self, task: AgentTask) -> AgentResult:
        """带计时的执行包装，子类可在 run() 内部调用"""
        t0 = time.perf_counter()
        try:
            result = self.run(task)
            result.elapsed_ms = (time.perf_counter() - t0) * 1000
            return result
        except Exception as e:
            elapsed = (time.perf_counter() - t0) * 1000
            logger.exception(f"[{self.name}] 执行异常: {e}")
            return AgentResult(
                success=False,
                summary=f"{self.name} 执行异常",
                error=str(e),
                elapsed_ms=elapsed,
            )

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name='{self.name}'>"

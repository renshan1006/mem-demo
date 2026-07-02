"""
DataGuardian — 数据质量评估智能体 (M2)

对源 CSV 做 5 维质量扫描（缺失值 / 重复 / 格式 / 异常值 / 一致性），
按 0-100 综合评分生成中文 JSON + Markdown 报告。
支持 Dify LLM 增强（可选，未配置时自动降级为 Baseline 纯统计报告）。

用法:
    from scripts.quality_agent import DataQualityAgent
    from scripts.agent_registry import registry

    agent = DataQualityAgent()
    registry.register(agent)

    result = registry.call("data_quality", AgentTask(
        action="scan",
        params={"source": "crm_customers.csv"},
    ))

CLI:
    python quality_agent.py                                    # 自检
    python quality_agent.py --source crm_customers.csv         # 单源报告
    python quality_agent.py --compare --source a.csv --source b.csv  # 跨源对比
"""

import sys
import json
import logging
import re
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

import pandas as pd
import numpy as np

# 把 scripts 加入路径，以便调用兄弟模块
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

# 兼容直接运行和模块导入
try:
    from .base_agent import BaseAgent, AgentTask, AgentResult
except ImportError:
    from base_agent import BaseAgent, AgentTask, AgentResult

logger = logging.getLogger(__name__)

BASE_DIR = _THIS_DIR.parent
DATA_DIR = BASE_DIR / "data"
DIFY_CONFIG_FILE = BASE_DIR / "dify" / "config.json"

# ── 字段常量 ────────────────────────────────────────────────

CUSTOMER_RELEVANT_FIELDS = [
    "company_name", "region", "city", "address", "phone", "tax_id",
    "website", "email", "contact_person", "source_created_at",
]

PRODUCT_RELEVANT_FIELDS = [
    "product_name", "category", "brand", "model", "sku",
    "specification", "upc", "price", "source_created_at",
]

CUSTOMER_SCORE_FIELDS = CUSTOMER_RELEVANT_FIELDS  # 用于评分公式的分母
PRODUCT_SCORE_FIELDS = PRODUCT_RELEVANT_FIELDS

CHINESE_PROVINCES = [
    "北京市", "上海市", "广东省", "浙江省", "江苏省", "四川省", "山东省",
    "湖北省", "陕西省", "湖南省", "天津市", "重庆市", "河北省", "山西省",
    "辽宁省", "吉林省", "黑龙江省", "安徽省", "福建省", "江西省", "河南省",
    "海南省", "贵州省", "云南省", "甘肃省", "青海省", "台湾省",
    "内蒙古自治区", "广西壮族自治区", "西藏自治区", "宁夏回族自治区",
    "新疆维吾尔自治区", "香港特别行政区", "澳门特别行政区",
]

# ── 评分权重 ────────────────────────────────────────────────

WEIGHTS = {
    "completeness": 0.30,
    "uniqueness":   0.20,
    "validity":     0.25,
    "consistency":  0.25,
}

BANDS = [
    (90, "优秀"),
    (75, "良好"),
    (60, "一般"),
    (0,  "较差"),
]


# ══════════════════════════════════════════════════════════════
# 轻量文本处理（仅用于跨源一致性检测中的归一化比对）
# ══════════════════════════════════════════════════════════════

def _normalize_company_for_cmp(name: str) -> str:
    """轻量归一化：去后缀、去空格、统一 Alpha↔α，用于跨源名称比对"""
    if not isinstance(name, str):
        return ""
    text = name.strip().lower()
    for s in ["有限公司", "科技有限公司", "科技公司", "信息技术有限公司",
              "贸易有限公司", "实业有限公司", "集团", "商务", "国际",
              "公司", "科技", "信息技术"]:
        text = text.replace(s, "")
    text = text.replace("alpha", "").replace("α", "").replace(" ", "")
    return text


def _extract_address_unit(address: str) -> str:
    """提取地址的尾单元 token（号/号楼/号院/栋/园/楼）"""
    if not isinstance(address, str):
        return ""
    m = re.search(r"(号院|号楼|号|栋|园|楼)[楼]?$", address.strip())
    return m.group(0) if m else ""


def _extract_phone_digits_len(phone: str) -> int:
    """去格式后返回纯数字位数"""
    if not isinstance(phone, str):
        return 0
    return len(re.sub(r"\D", "", phone))


# ── 评分 ────────────────────────────────────────────────────

def _compute_band(score: float) -> str:
    for threshold, label in BANDS:
        if score >= threshold:
            return label
    return "较差"


# ══════════════════════════════════════════════════════════════
# DataQualityAgent
# ══════════════════════════════════════════════════════════════

class DataQualityAgent(BaseAgent):
    """
    数据质量评估智能体 — Baseline 统计检测 + Dify LLM 增强中文报告。

    支持的动作：
        - scan    : 扫描指定数据源，返回质量问题列表 + 评分
        - report  : 生成中文质量报告（JSON + Markdown）并写入磁盘
        - compare : 对比多个数据源的质量并给出跨源一致性分析
    """

    name = "data_quality"
    description = "数据质量评估智能体（Baseline统计 + Dify LLM增强中文报告）"

    def get_actions(self) -> List[str]:
        return ["scan", "report", "compare"]

    def health_check(self) -> bool:
        return DATA_DIR.exists() and any(DATA_DIR.glob("*.csv"))

    def run(self, task: AgentTask) -> AgentResult:
        action = task.action

        if action == "scan":
            return self._scan(task)
        elif action == "report":
            return self._report(task)
        elif action == "compare":
            return self._compare(task)
        else:
            return self._fail(
                error=f"未知动作: '{action}'",
                summary=f"支持的动作: {self.get_actions()}",
            )

    # ── 加载 / 分发 ─────────────────────────────────────────

    def _resolve_source(self, source: str) -> Tuple[pd.DataFrame, str]:
        """解析 source 名 → (df, entity_type)"""
        path = Path(source) if Path(source).exists() else DATA_DIR / source
        if not path.exists():
            raise FileNotFoundError(f"数据文件不存在: {path}")
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        return df, self._detect_entity(df)

    def _detect_entity(self, df: pd.DataFrame) -> str:
        if "canonical_customer_id" in df.columns:
            return "customer"
        if "canonical_product_id" in df.columns:
            return "product"
        return "unknown"

    def _numeric_column(self, df: pd.DataFrame) -> Optional[str]:
        if "price" in df.columns:
            return "price"
        return None

    def _score_fields(self, entity_type: str) -> List[str]:
        return CUSTOMER_SCORE_FIELDS if entity_type == "customer" else PRODUCT_SCORE_FIELDS

    # ══════════════════════════════════════════════════════════
    # 5 Baseline 检测器
    # ══════════════════════════════════════════════════════════

    # ── 缺失值 ──────────────────────────────────────────────

    def _detect_missing(self, df: pd.DataFrame, entity_type: str) -> dict:
        """统计 NaN + 空串"""
        total_cells = len(df) * len(self._score_fields(entity_type))
        issues: Dict[str, int] = {}
        missing_cells = 0
        for col in self._score_fields(entity_type):
            if col not in df.columns:
                continue
            col_missing = (df[col].isna() | (df[col] == "")).sum()
            if col_missing > 0:
                issues[col] = int(col_missing)
                missing_cells += int(col_missing)

        return {
            "issue_count": len(issues),
            "missing_cells": missing_cells,
            "total_cells": total_cells,
            "fields": issues,
        }

    # ── 重复行 ──────────────────────────────────────────────

    def _detect_duplicates(self, df: pd.DataFrame, entity_type: str) -> dict:
        """
        完全重复行检测。
        注意：本合成数据集单源内 canonical_*_id 不重复（重复是跨源的），
        所以 duplicated() 预期 ≈0。真正有意义的"重复/冲突"信号
        在 _detect_consistency_cross_source 中按 canonical_*_id 分组呈现。
        """
        exact_dup = int(df.duplicated().sum())

        # 单源内 canonical_id 重复（按设计应为 0）
        id_col = "canonical_customer_id" if entity_type == "customer" else "canonical_product_id"
        id_col = id_col if id_col in df.columns else None
        canonical_dup = 0
        if id_col:
            canonical_dup = int(df[id_col].duplicated().sum())

        return {
            "issue_count": exact_dup + canonical_dup,
            "duplicated_rows": exact_dup,
            "canonical_id_duplicates": canonical_dup,
            "note": "本数据集单源 ≈0；重复均为跨源（见一致性检测）",
        }

    # ── 格式异常 ────────────────────────────────────────────

    def _detect_format(self, df: pd.DataFrame, entity_type: str) -> dict:
        """正则校验各字段格式"""
        checks: Dict[str, Dict] = {}

        # phone: 座机 XXX-XXXX-XXXX 或 手机 1xxxxxxxxxx(11-12位)
        if "phone" in df.columns:
            phone_pat = re.compile(r"^(\d{3}-\d{4}-\d{4}|1\d{10,11})$")
            valid = df["phone"].astype(str).str.match(phone_pat)
            invalid_count = int((~valid).sum())
            if invalid_count > 0:
                checks["phone"] = {
                    "invalid": invalid_count,
                    "total": len(df),
                    "pattern": "座机(XXX-XXXX-XXXX) 或 手机(1xxxxxxxxxx)",
                    "sample": df.loc[~valid, "phone"].head(5).tolist(),
                }

        # tax_id: 18 位字母数字
        if "tax_id" in df.columns:
            tax_pat = re.compile(r"^[0-9A-Z]{18}$")
            valid = df["tax_id"].astype(str).str.match(tax_pat)
            invalid_count = int((~valid).sum())
            if invalid_count > 0:
                checks["tax_id"] = {
                    "invalid": invalid_count,
                    "total": len(df),
                    "pattern": "18位字母数字",
                    "sample": df.loc[~valid, "tax_id"].head(5).tolist(),
                }

        # email
        if "email" in df.columns:
            email_pat = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
            valid = df["email"].astype(str).str.match(email_pat)
            invalid_count = int((~valid).sum())
            if invalid_count > 0:
                checks["email"] = {
                    "invalid": invalid_count,
                    "total": len(df),
                    "pattern": "xxx@xxx.xxx",
                    "sample": df.loc[~valid, "email"].head(5).tolist(),
                }

        # website
        if "website" in df.columns:
            web_pat = re.compile(r"^[\w.-]+\.[a-zA-Z]{2,}")
            valid = df["website"].astype(str).str.match(web_pat)
            invalid_count = int((~valid).sum())
            if invalid_count > 0:
                checks["website"] = {
                    "invalid": invalid_count,
                    "total": len(df),
                    "pattern": "xxx.xxx",
                    "sample": df.loc[~valid, "website"].head(5).tolist(),
                }

        # source_created_at
        if "source_created_at" in df.columns:
            date_pat = re.compile(r"^\d{4}-\d{2}-\d{2}$")
            valid = df["source_created_at"].astype(str).str.match(date_pat)
            invalid_count = int((~valid).sum())
            if invalid_count > 0:
                checks["source_created_at"] = {
                    "invalid": invalid_count,
                    "total": len(df),
                    "pattern": "YYYY-MM-DD",
                    "sample": df.loc[~valid, "source_created_at"].head(5).tolist(),
                }

        # (商品) upc
        if entity_type == "product" and "upc" in df.columns:
            upc_pat = re.compile(r"^\d{12}$")
            valid = df["upc"].astype(str).str.match(upc_pat)
            invalid_count = int((~valid).sum())
            if invalid_count > 0:
                checks["upc"] = {
                    "invalid": invalid_count,
                    "total": len(df),
                    "pattern": "12位数字",
                    "sample": df.loc[~valid, "upc"].head(5).tolist(),
                }

        # (商品) price 可解析
        if entity_type == "product" and "price" in df.columns:
            valid = pd.to_numeric(df["price"], errors="coerce").notna()
            invalid_count = int((~valid).sum())
            if invalid_count > 0:
                checks["price_parseable"] = {
                    "invalid": invalid_count,
                    "total": len(df),
                    "pattern": "可解析浮点数",
                    "sample": df.loc[~valid, "price"].head(5).tolist(),
                }

        total_invalid = sum(c["invalid"] for c in checks.values())
        total_checked = sum(c["total"] for c in checks.values()) if checks else 1

        return {
            "issue_count": len(checks),
            "total_invalid_cells": total_invalid,
            "total_checked_cells": total_checked,
            "fields": checks,
        }

    # ── 异常值 ──────────────────────────────────────────────

    def _detect_outliers(self, df: pd.DataFrame, entity_type: str) -> dict:
        """IQR 异常值检测（仅商品 price）"""
        numeric_col = self._numeric_column(df)
        if numeric_col is None or entity_type != "product":
            return {"applicable": False, "note": "客户数据无数值字段，跳过"}

        series = pd.to_numeric(df[numeric_col], errors="coerce")
        valid = series.dropna()
        if len(valid) < 10:
            return {"applicable": True, "note": "样本量太小不足以做 IQR"}

        q1 = valid.quantile(0.25)
        q3 = valid.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outlier_mask = (valid < lower) | (valid > upper)
        outlier_count = int(outlier_mask.sum())
        outliers = valid[outlier_mask].tolist()

        return {
            "applicable": True,
            "field": numeric_col,
            "count": outlier_count,
            "total": len(valid),
            "q1": round(float(q1), 2),
            "q3": round(float(q3), 2),
            "iqr": round(float(iqr), 2),
            "lower_fence": round(lower, 2),
            "upper_fence": round(upper, 2),
            "samples": outliers[:10],
            "note": (
                "±5–8% 价格注入变异大多落在 IQR 界内（IQR 对此幅度稳健），"
                "因此此处异常值极少。价格差异由跨源一致性检测的 max/min 比 >1.05 呈现。"
            ),
        }

    # ── 单源一致性 ──────────────────────────────────────────

    def _detect_consistency_single(self, df: pd.DataFrame, entity_type: str) -> dict:
        """分类字段频次分布 + region 省内校验"""
        fields: Dict[str, Any] = {}

        if "region" in df.columns:
            unknown = df[~df["region"].isin(CHINESE_PROVINCES)]
            if len(unknown) > 0:
                fields["region_unknown"] = {
                    "count": int(len(unknown)),
                    "sample": unknown["region"].value_counts().head(5).to_dict(),
                }

        return {
            "issue_count": len(fields),
            "fields": fields,
        }

    # ── 跨源一致性（核心区分点）───────────────────────────

    def _detect_consistency_cross_source(
        self,
        sources: Dict[str, pd.DataFrame],
        entity_type: str,
        id_col: str,
    ) -> dict:
        """按 canonical_*_id 分组，检测同一实体在多个源中的字段冲突"""
        combined_list = []
        for src_name, sdf in sources.items():
            df_tag = sdf.copy()
            df_tag["__source"] = src_name
            combined_list.append(df_tag)

        combined = pd.concat(combined_list, ignore_index=True)
        groups = combined.groupby(id_col)
        shared = groups.filter(lambda g: g["__source"].nunique() >= 2)
        shared_ids = shared[id_col].unique()
        shared_count = len(shared_ids)

        if shared_count == 0:
            return {
                "shared_entities": 0,
                "conflict_groups": 0,
                "by_field": {},
                "note": "无跨源共享实体（canonical_*_id 单源独有）",
            }

        by_field_conflicts: Dict[str, int] = {}
        conflict_samples: List[Dict] = []
        conflict_id_set: set = set()

        for gid, grp in shared.groupby(id_col):
            has_conflict = False
            group_conflicts = {}

            # 公司/商品 名称（归一化比对）
            name_col = "company_name" if entity_type == "customer" else "product_name"
            if name_col in grp.columns:
                norm_names = grp[name_col].apply(_normalize_company_for_cmp)
                if norm_names.nunique() > 1:
                    by_field_conflicts[name_col] = by_field_conflicts.get(name_col, 0) + 1
                    has_conflict = True
                    group_conflicts[name_col] = grp[name_col].tolist()

            # 地址单元
            if "address" in grp.columns:
                units = grp["address"].apply(_extract_address_unit)
                non_empty = units[units != ""]
                if non_empty.nunique() > 1:
                    by_field_conflicts["address"] = by_field_conflicts.get("address", 0) + 1
                    has_conflict = True
                    group_conflicts["address"] = grp["address"].tolist()

            # 电话格式/号码
            if "phone" in grp.columns:
                digit_lens = grp["phone"].apply(_extract_phone_digits_len)
                if digit_lens.nunique() > 1:
                    by_field_conflicts["phone"] = by_field_conflicts.get("phone", 0) + 1
                    has_conflict = True
                    group_conflicts["phone"] = grp["phone"].tolist()
                else:
                    raw_digits = grp["phone"].apply(lambda x: re.sub(r"\D", "", str(x)))
                    if raw_digits.nunique() > 1:
                        by_field_conflicts["phone"] = by_field_conflicts.get("phone", 0) + 1
                        has_conflict = True
                        group_conflicts["phone_dict"] = grp["phone"].tolist()

            # 税号
            if "tax_id" in grp.columns and grp["tax_id"].nunique() > 1:
                by_field_conflicts["tax_id"] = by_field_conflicts.get("tax_id", 0) + 1
                has_conflict = True
                group_conflicts["tax_id"] = grp["tax_id"].tolist()

            # 产品价格 (max/min > 1.05)
            if entity_type == "product" and "price" in grp.columns:
                prices = pd.to_numeric(grp["price"], errors="coerce").dropna()
                if len(prices) >= 2:
                    pmax, pmin = prices.max(), prices.min()
                    if pmin > 0 and pmax / pmin > 1.05:
                        by_field_conflicts["price"] = by_field_conflicts.get("price", 0) + 1
                        has_conflict = True
                        group_conflicts["price"] = grp["price"].tolist()

            if has_conflict:
                conflict_id_set.add(gid)
                if len(conflict_samples) < 10:
                    conflict_samples.append({
                        "canonical_id": gid,
                        "source_count": int(grp["__source"].nunique()),
                        "conflicts": group_conflicts,
                    })

        return {
            "shared_entities": shared_count,
            "conflict_groups": len(conflict_id_set),
            "by_field": by_field_conflicts,
            "samples": conflict_samples,
        }

    # ══════════════════════════════════════════════════════════
    # LLM 增强（自包含，不依赖 dify_client.py）
    # ══════════════════════════════════════════════════════════

    def _env_or_config(self, key: str, default: str = "") -> str:
        """优先环境变量，其次 dify/config.json"""
        env_key = f"DIFY_{key.upper()}"
        if env_key in os.environ:
            return os.environ[env_key]
        if DIFY_CONFIG_FILE.exists():
            try:
                with open(DIFY_CONFIG_FILE, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                return cfg.get(key, default)
            except Exception:
                return default
        return default

    def _call_quality_llm(self, inputs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        自包含 Dify 质量工作流调用（不动 dify_client.py —— 组长文件）。
        失败返回 None，绝不抛异常。
        """
        import requests
        import os

        api_key = self._env_or_config("quality_api_key")
        if not api_key:
            logger.info("Dify quality_api_key 未配置，跳过 LLM 增强")
            return None

        base_url = self._env_or_config("quality_base_url", "https://api.dify.ai").rstrip("/")
        endpoint = f"{base_url}/v1/workflows/run"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "inputs": inputs,
            "response_mode": "blocking",
            "user": "quality-agent",
        }

        try:
            resp = requests.post(endpoint, json=payload, headers=headers, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            if "data" in data and "outputs" in data["data"]:
                return data["data"]["outputs"]
            return data
        except Exception as e:
            logger.warning("Dify quality LLM 调用失败: %s", e)
            return None

    def _enhance_report_with_llm(self, scan_payload: dict) -> Dict[str, Any]:
        """对一次 scan 结果做 LLM 增强；不可用时降级"""
        try:
            # 提取紧凑摘要给 LLM
            sources_data = scan_payload.get("sources", [])
            if not sources_data:
                return {"available": False, "note": "无扫描数据可供 LLM 分析"}

            summary = []
            for s in sources_data:
                dims = s.get("dimensions", {})
                fmt = dims.get("format", {}).get("fields", {})
                dup = dims.get("duplicates", {})
                missing = dims.get("missing", {})
                outliers = dims.get("outliers", {})
                consistency = dims.get("consistency", {})

                summary.append({
                    "source": s.get("source"),
                    "row_count": s.get("row_count"),
                    "score": s.get("score"),
                    "missing": {
                        f: missing.get("fields", {}).get(f, 0)
                        for f in missing.get("fields", {})
                    },
                    "duplicate_rows": dup.get("duplicated_rows", 0),
                    "format_issues": {
                        f: fmt.get(f, {}).get("invalid", 0) for f in fmt
                    },
                    "outliers": {
                        "count": outliers.get("count", 0) if outliers.get("applicable") else None,
                    },
                    "cross_source": consistency.get("cross", {}),
                })

            inputs = {
                "quality_payload": json.dumps(summary, ensure_ascii=False),
                "entity_type": scan_payload.get("entity_type", "unknown"),
                "source": ", ".join(s["source"] for s in sources_data),
            }

            llm_result = self._call_quality_llm(inputs)
            if llm_result is None:
                return {"available": False, "note": "Dify quality 工作流未配置或调用失败"}

            return {
                "available": True,
                "补全建议": str(llm_result.get("补全建议", "")),
                "去重建议": str(llm_result.get("去重建议", "")),
                "异常原因": str(llm_result.get("异常原因", "")),
                "整改优先级": llm_result.get("整改优先级", []),
                "总结": str(llm_result.get("总结", "")),
            }
        except Exception as e:
            logger.warning("LLM 增强异常（已降级）: %s", e)
            return {"available": False, "error": str(e)}

    # ══════════════════════════════════════════════════════════
    # 评分
    # ══════════════════════════════════════════════════════════

    def _score_source(self, df: pd.DataFrame, dims: dict,
                      entity_type: str, cross_consistency: Optional[dict] = None) -> dict:
        """计算 0–100 加权综合评分"""

        # 完整性
        missing = dims.get("missing", {})
        if missing.get("total_cells", 0) > 0:
            completeness = 100.0 * (1 - missing["missing_cells"] / missing["total_cells"])
        else:
            completeness = 100.0

        # 唯一性
        dup = dims.get("duplicates", {})
        uniqueness = 100.0
        if len(df) > 0:
            uniqueness = 100.0 * (1 - dup.get("duplicated_rows", 0) / len(df))

        # 有效性
        fmt = dims.get("format", {})
        checked = fmt.get("total_checked_cells", 1)
        invalid = fmt.get("total_invalid_cells", 0)
        validity = 100.0 * (1 - invalid / checked) if checked > 0 else 100.0

        # 一致性
        if cross_consistency and cross_consistency.get("shared_entities", 0) > 0:
            shared = cross_consistency["shared_entities"]
            conflicts = cross_consistency["conflict_groups"]
            consistency = 100.0 * (1 - conflicts / shared) if shared > 0 else 100.0
        else:
            consistency_single = dims.get("consistency", {}).get("fields", {})
            consistency = 100.0 if not consistency_single else 80.0

        score = round(
            WEIGHTS["completeness"] * completeness +
            WEIGHTS["uniqueness"]   * uniqueness +
            WEIGHTS["validity"]     * validity +
            WEIGHTS["consistency"]  * consistency
        )
        score = max(0, min(100, score))

        return {
            "score": score,
            "band": _compute_band(score),
            "breakdown": {
                "completeness": round(completeness, 1),
                "uniqueness":   round(uniqueness, 1),
                "validity":     round(validity, 1),
                "consistency":  round(consistency, 1),
            },
            "weights": WEIGHTS,
        }

    # ══════════════════════════════════════════════════════════
    # 动作: scan / report / compare
    # ══════════════════════════════════════════════════════════

    def _scan(self, task: AgentTask) -> AgentResult:
        source = task.params.get("source")
        if not source:
            return self._fail(error="缺少必需参数 'source'")

        entity_type = task.params.get("entity")
        use_llm = task.params.get("use_llm", True)
        t0 = time.perf_counter()

        try:
            df, detected_entity = self._resolve_source(source)
            entity_type = entity_type or detected_entity

            dims = {
                "missing":    self._detect_missing(df, entity_type),
                "duplicates": self._detect_duplicates(df, entity_type),
                "format":     self._detect_format(df, entity_type),
                "outliers":   self._detect_outliers(df, entity_type),
                "consistency": {},
            }

            # 跨源一致性（仅当调用方提供时）
            cross_sources_raw = task.params.get("cross_sources") or task.context.get("cross_sources")
            cross_result = None
            if cross_sources_raw:
                loaded = {}
                for sname, sdf_or_path in cross_sources_raw.items():
                    if isinstance(sdf_or_path, pd.DataFrame):
                        loaded[sname] = sdf_or_path
                    else:
                        loaded[sname], _ = self._resolve_source(str(sdf_or_path))
                # 加入当前 df
                loaded[source] = df
                id_col = "canonical_customer_id" if entity_type == "customer" else "canonical_product_id"
                cross_result = self._detect_consistency_cross_source(loaded, entity_type, id_col)
                dims["consistency"] = {"cross": cross_result}
            else:
                dims["consistency"] = {"intra": self._detect_consistency_single(df, entity_type)}

            scoring = self._score_source(df, dims, entity_type, cross_result)

            payload = {
                "agent": self.name,
                "entity_type": entity_type,
                "source": source,
                "row_count": len(df),
                **scoring,
                "dimensions": dims,
            }

            elapsed = (time.perf_counter() - t0) * 1000

            # LLM 增强
            llm_data = {"available": False}
            if use_llm:
                llm_data = self._enhance_report_with_llm({"entity_type": entity_type, "sources": [payload]})
            payload["llm"] = llm_data

            return self._ok(
                data=payload,
                summary=f"{source} 质量评分 {scoring['score']}/100（{scoring['band']}），"
                       f"检出 {sum(d.get('issue_count', 0) for d in dims.values())} 类问题",
                details={
                    "row_count": len(df),
                    "score": scoring["score"],
                    "band": scoring["band"],
                    "llm_available": llm_data.get("available", False),
                },
                elapsed_ms=elapsed,
            )

        except FileNotFoundError as e:
            return self._fail(error=str(e), summary="数据文件缺失")
        except Exception as e:
            logger.exception("scan 异常")
            return self._fail(error=str(e), summary="质量扫描异常")

    def _report(self, task: AgentTask) -> AgentResult:
        """生成中文报告并写入 data/quality_report.{json,md}"""
        sources_list = task.params.get("sources", [task.params.get("source")])
        sources_list = [s for s in sources_list if s]

        if not sources_list:
            return self._fail(error="缺少 'source' 或 'sources' 参数")

        use_llm = task.params.get("use_llm", True)
        out_dir = Path(task.params.get("out_dir") or (BASE_DIR / "data"))
        out_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.perf_counter()
        all_scans = []
        entity_type = None
        cross_result = None

        try:
            # 逐源 scan
            loaded_sources: Dict[str, pd.DataFrame] = {}
            for src in sources_list:
                df, et = self._resolve_source(src)
                entity_type = entity_type or et
                loaded_sources[src] = df

                dims = {
                    "missing":    self._detect_missing(df, entity_type),
                    "duplicates": self._detect_duplicates(df, entity_type),
                    "format":     self._detect_format(df, entity_type),
                    "outliers":   self._detect_outliers(df, entity_type),
                    "consistency": {},
                }
                # 单源内（先不留空，后面做跨源）
                all_scans.append((src, df, dims))

            # 跨源一致性（如果有多个源）
            if len(loaded_sources) >= 2:
                id_col = "canonical_customer_id" if entity_type == "customer" else "canonical_product_id"
                cross_result = self._detect_consistency_cross_source(
                    loaded_sources, entity_type, id_col
                )

            # 组装
            source_entries = []
            for src_name, df, dims in all_scans:
                dims["consistency"] = {}
                scoring = self._score_source(df, dims, entity_type, cross_result)
                entry = {
                    "source": src_name,
                    "row_count": len(df),
                    **scoring,
                    "dimensions": dims,
                }
                source_entries.append(entry)

            payload = {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "agent": self.name,
                "entity_type": entity_type,
                "sources": source_entries,
                "cross_source_summary": cross_result,
                "compare_scorecard": self._build_scorecard(source_entries, cross_result),
            }

            # LLM
            if use_llm:
                payload["llm"] = self._enhance_report_with_llm(payload)
            else:
                payload["llm"] = {"available": False}

            # 写磁盘
            json_path = out_dir / "quality_report.json"
            md_path = out_dir / "quality_report.md"

            json_str = self._build_json_report(payload)
            with open(json_path, "w", encoding="utf-8") as f:
                f.write(json_str)

            md_str = self._build_markdown_report(payload)
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(md_str)

            elapsed = (time.perf_counter() - t0) * 1000

            return self._ok(
                data=payload,
                summary=f"质量报告已生成 → {json_path} / {md_path}",
                details={
                    "json_report": str(json_path),
                    "md_report": str(md_path),
                    "source_count": len(sources_list),
                },
                elapsed_ms=elapsed,
            )

        except Exception as e:
            logger.exception("report 异常")
            return self._fail(error=str(e), summary="报告生成异常")

    def _compare(self, task: AgentTask) -> AgentResult:
        """对比多个源的质量（走 report 再提取 scorecard）"""
        sources_list = task.params.get("sources", [])
        if len(sources_list) < 2:
            return self._fail(error="compare 需要至少 2 个源，请用 --source 指定多个")

        # 复用 _report 逻辑
        task.params["use_llm"] = task.params.get("use_llm", True)
        result = self._report(task)
        if not result.success:
            return result

        data = result.data or {}
        scorecard = data.get("compare_scorecard", {})
        ranking = scorecard.get("ranking", [])

        return self._ok(
            data=data,
            summary=f"跨源对比完成。评分排名: {' > '.join(ranking) if ranking else '(无)'}",
            details=scorecard,
            elapsed_ms=result.elapsed_ms,
        )

    def _build_scorecard(self, sources: List[dict],
                         cross_source: Optional[dict] = None) -> dict:
        ranked = sorted(sources, key=lambda s: s.get("score", 0), reverse=True)
        return {
            "sources": [
                {
                    "source": s["source"],
                    "score": s["score"],
                    "band": s.get("band", ""),
                    "issue_total": sum(
                        d.get("issue_count", 0)
                        for d in s.get("dimensions", {}).values()
                        if isinstance(d, dict)
                    ),
                    "rank": i + 1,
                }
                for i, s in enumerate(ranked)
            ],
            "ranking": [s["source"] for s in ranked],
            "best": ranked[0]["source"] if ranked else None,
        }

    # ══════════════════════════════════════════════════════════
    # 报告构建器
    # ══════════════════════════════════════════════════════════

    def _build_json_report(self, payload: dict) -> str:
        return json.dumps(payload, ensure_ascii=False, indent=2, default=str)

    def _build_markdown_report(self, payload: dict) -> str:
        """生成中文质量评估报告 Markdown"""
        lines = [
            "# 数据质量评估报告",
            "",
            f"> 生成时间：{payload.get('generated_at', '')}",
            f"> 智能体：{payload.get('agent', '')}",
            f"> 实体类型：{payload.get('entity_type', '')}",
            "",
            "---",
            "",
            "## 一、总体评分",
            "",
        ]

        sources = payload.get("sources", [])
        # 总体评分表
        lines.append("| 数据源 | 行数 | 评分 | 等级 |")
        lines.append("|--------|-----:|:----:|------|")
        for s in sources:
            lines.append(f"| {s['source']} | {s['row_count']} | {s['score']}/100 | {s.get('band', '')} |")
        lines.append("")

        # 一句话结论
        if sources:
            avg_score = round(np.mean([s["score"] for s in sources]))
            lines.append(f"**结论**：{len(sources)} 个数据源的平均质量评分为 **{avg_score}/100**，"
                         f"整体等级 **{_compute_band(avg_score)}**。")
        lines.append("")

        # 等级说明
        lines.append("| 评分 | 等级 |")
        lines.append("|:----:|------|")
        for thresh, band in BANDS:
            lines.append(f"| ≥{thresh} | {band} |")
        lines.append("")

        # 二、分维度评分明细
        lines.append("## 二、分维度评分明细")
        lines.append("")
        for s in sources:
            breakdown = s.get("score_breakdown", {})
            lines.append(f"### {s['source']}（{s.get('band', '')}，{s['score']}/100）")
            lines.append("")
            lines.append("| 维度 | 得分 | 权重 |")
            lines.append("|------|:---:|:---:|")
            for dim_name in ["completeness", "uniqueness", "validity", "consistency"]:
                score = breakdown.get(dim_name, "-")
                weight = WEIGHTS.get(dim_name, 0)
                weight_pct = f"{int(weight*100)}%"
                dim_cn = {
                    "completeness": "完整性",
                    "uniqueness": "唯一性",
                    "validity": "有效性",
                    "consistency": "一致性",
                }.get(dim_name, dim_name)
                lines.append(f"| {dim_cn}（{weight_pct}） | {score} | {weight_pct} |")
            lines.append("")

            # 每维度详情
            dims = s.get("dimensions", {})

            missing = dims.get("missing", {})
            if missing.get("issue_count", 0) > 0:
                lines.append(f"#### 2.1 完整性 — 缺失 {missing.get('missing_cells', 0)} 个单元格")
                lines.append("")
                for fld, cnt in sorted(missing.get("fields", {}).items(), key=lambda x: -x[1]):
                    lines.append(f"- `{fld}`：{cnt} 缺失")
                lines.append("")
            else:
                lines.append("#### 2.1 完整性 — 无缺失值 ✅")
                lines.append("")

            dup = dims.get("duplicates", {})
            lines.append(f"#### 2.2 唯一性 — 完全重复行 {dup.get('duplicated_rows', 0)}")
            if dup.get("note"):
                lines.append(f"> {dup['note']}")
            lines.append("")

            fmt = dims.get("format", {})
            if fmt.get("issue_count", 0) > 0:
                lines.append(f"#### 2.3 有效性 — {fmt.get('issue_count', 0)} 个字段存在格式异常")
                lines.append("")
                for fld, info in fmt.get("fields", {}).items():
                    invalid = info.get("invalid", 0)
                    total = info.get("total", 0)
                    pct = f"{invalid/total*100:.1f}%" if total > 0 else "0%"
                    lines.append(f"- `{fld}`：{invalid}/{total} 非法（{pct}），匹配规则 `{info.get('pattern', '')}`")
                    sample = info.get("sample", [])[:3]
                    if sample:
                        lines.append(f"  样本：{', '.join(str(x) for x in sample)}")
                lines.append("")
            else:
                lines.append("#### 2.3 有效性 — 格式均合规 ✅")
                lines.append("")

            outliers = dims.get("outliers", {})
            if outliers.get("applicable"):
                lines.append(f"#### 2.5 异常值 — {outliers.get('count', 0)} 个（{outliers.get('field', '')}，IQR 法）")
                if outliers.get("note"):
                    lines.append(f"> {outliers['note']}")
                if outliers.get("samples"):
                    lines.append(f"样本：{outliers['samples'][:5]}")
                lines.append("")

            consistency = dims.get("consistency", {})
            cons_cross = consistency.get("cross")
            if cons_cross:
                shared = cons_cross.get("shared_entities", 0)
                conflicts = cons_cross.get("conflict_groups", 0)
                lines.append(f"#### 2.4 一致性 — 跨源共享实体 {shared}，冲突组 {conflicts}")
                by_field = cons_cross.get("by_field", {})
                if by_field:
                    for fld, cnt in sorted(by_field.items(), key=lambda x: -x[1]):
                        lines.append(f"- `{fld}`：{cnt} 组冲突")
                lines.append("")

        # 三、跨源一致性深析
        cross = payload.get("cross_source_summary")
        if cross and cross.get("shared_entities", 0) > 0:
            lines.append("## 三、跨源一致性深析")
            lines.append("")
            shared = cross.get("shared_entities", 0)
            conflicts = cross.get("conflict_groups", 0)
            lines.append(f"- 共享实体：**{shared}**")
            lines.append(f"- 存在字段冲突的实体：**{conflicts}**（{conflicts/shared*100:.1f}% 冲突率）" if shared > 0 else "")
            lines.append("")

            by_field = cross.get("by_field", {})
            if by_field:
                lines.append("| 字段 | 冲突组数 |")
                lines.append("|------|:------:|")
                for fld, cnt in sorted(by_field.items(), key=lambda x: -x[1]):
                    lines.append(f"| {fld} | {cnt} |")
                lines.append("")

            samples = cross.get("samples", [])[:5]
            if samples:
                lines.append("**冲突样本**：")
                lines.append("")
                for smp in samples:
                    lines.append(f"- `{smp['canonical_id']}`（{smp['source_count']} 个源）")
                    for fld, vals in smp.get("conflicts", {}).items():
                        lines.append(f"  - {fld}：{' / '.join(str(v) for v in vals)}")
                lines.append("")

        # 四、LLM 整改建议
        llm = payload.get("llm") or next((s.get("llm") for s in sources if s.get("llm")), None)
        lines.append("## 四、LLM 整改建议")
        lines.append("")
        if llm and llm.get("available"):
            if llm.get("补全建议"):
                lines.append(f"**补全建议**：{llm['补全建议']}")
            if llm.get("去重建议"):
                lines.append(f"**去重建议**：{llm['去重建议']}")
            if llm.get("异常原因"):
                lines.append(f"**异常原因分析**：{llm['异常原因']}")
            prio = llm.get("整改优先级", [])
            if prio:
                lines.append(f"**整改优先级**：{' → '.join(str(p) for p in prio)}")
            if llm.get("总结"):
                lines.append(f"**总结**：{llm['总结']}")
        else:
            lines.append("> ⚠️ Dify 质量评估工作流未配置，已跳过 LLM 增强。")
            lines.append("> 如需启用，请将 `dify/prompts/质量评估提示词.md` 导入 Dify 并发布工作流，")
            lines.append("> 再将 `quality_api_key` 填入 `dify/config.json`。")
            lines.append("")
        lines.append("")

        # 五、评分方法说明
        lines.append("## 五、评分方法说明")
        lines.append("")
        lines.append("| 维度 | 权重 | 方法 |")
        lines.append("|------|:---:|------|")
        lines.append("| 完整性 | 30% | (1 − 缺失单元格/总单元格) × 100 |")
        lines.append("| 唯一性 | 20% | (1 − 重复行/总行) × 100 |")
        lines.append("| 有效性 | 25% | (合法单元格/受检单元格) × 100，各正则字段平均 |")
        lines.append("| 一致性 | 25% | 跨源：(1 − 冲突组/共享组) × 100；单源：分类分布校验 |")
        lines.append("")
        lines.append(f"**综合评分** = 30%×完整性 + 20%×唯一性 + 25%×有效性 + 25%×一致性，取值 0–100。")
        lines.append("")

        return "\n".join(lines)


# ── 注册助手 ────────────────────────────────────────────────

def register_quality_agent(registry: Any) -> None:
    """向 Registry 注册本 Agent（供组长 bootstrap 调用）"""
    registry.register(DataQualityAgent())


# ── 命令行入口 ──────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import os

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    # 无参数 → 自检模式
    if len(sys.argv) == 1:
        print("DataQualityAgent 自检\n")
        agent = DataQualityAgent()
        print(f"名称: {agent.name}")
        print(f"描述: {agent.description}")
        print(f"动作: {agent.get_actions()}")
        print(f"健康: {agent.health_check()}")
        print()

        # 快速扫描 crm_customers.csv
        result = agent.run(AgentTask(
            action="scan",
            params={"source": "crm_customers.csv", "use_llm": False},
        ))
        print(f"scan 结果: {result.summary}")
        print(f"  success={result.success}")
        if result.data:
            scoring = result.data
            print(f"  score={scoring.get('score')}/{scoring.get('band')}")
            print(f"  row_count={scoring.get('row_count')}")
        print(f"  elapsed={result.elapsed_ms:.0f}ms")
        sys.exit(0 if result.success else 1)

    # argparse 模式
    parser = argparse.ArgumentParser(
        description="DataGuardian 数据质量评估智能体 (M2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python quality_agent.py                                  # 自检
  python quality_agent.py --source crm_customers.csv       # 单源报告
  python quality_agent.py --compare --source crm_customers.csv --source erp_customers.csv  # 跨源对比
        """,
    )
    parser.add_argument("--source", action="append", default=[],
                       help="数据源 CSV 路径或 data/ 下文件名（可重复）")
    parser.add_argument("--entity", choices=["customer", "product"],
                       help="实体类型（默认自动检测）")
    parser.add_argument("--compare", action="store_true",
                       help="跨源对比模式（需 --source ≥2）")
    parser.add_argument("--no-llm", action="store_true",
                       help="禁用 Dify LLM 增强（仅 Baseline 统计）")
    parser.add_argument("--out", default=str(BASE_DIR / "data"),
                       help="报告输出目录（默认 data/）")
    parser.add_argument("--action", choices=["scan", "report", "compare"],
                       help="显式指定动作（默认按 flag 推导）")

    args = parser.parse_args()

    # 推导动作
    action = args.action
    if not action:
        if args.compare:
            action = "compare"
        elif args.source:
            action = "report"
        else:
            print("错误: 请传 --source 或 --compare")
            sys.exit(1)

    task = AgentTask(
        action=action,
        params={
            "sources": args.source if args.source else [args.source],
            "source": args.source[0] if args.source else None,
            "entity": args.entity,
            "use_llm": not args.no_llm,
            "out_dir": args.out,
        },
    )

    agent = DataQualityAgent()
    result = agent.run(task)

    print()
    print(f"结果: {result.summary}")
    print(f"  success={result.success}, elapsed={result.elapsed_ms:.0f}ms")
    if not result.success:
        print(f"  error={result.error}")
    sys.exit(0 if result.success else 1)

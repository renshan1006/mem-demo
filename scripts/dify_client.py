"""
Dify MDM 智能体 API 客户端
============================
调用 Dify Workflow API 进行实体匹配判断。

用法：
  from scripts.dify_client import DifyMDMClient

  client = DifyMDMClient(api_key="app-xxx", base_url="https://cloud.dify.ai")
  result = client.match_customers(customer_a, customer_b)
  # → {"decision": "merge", "confidence": 0.92, "reasoning": "...", ...}
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# 配置路径
BASE_DIR = Path(__file__).resolve().parent.parent
DIFY_CONFIG_FILE = BASE_DIR / "dify" / "config.json"


def _load_config() -> dict:
    """加载 Dify 配置文件"""
    if DIFY_CONFIG_FILE.exists():
        with open(DIFY_CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _env_or_config(key: str, default: str = "") -> str:
    """优先从环境变量读取，其次从配置文件"""
    env_key = f"DIFY_{key.upper()}"
    return os.environ.get(env_key) or _load_config().get(key, default)


class DifyMDMClient:
    """Dify MDM 智能体 API 客户端"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.api_key = api_key or _env_or_config("api_key")
        self.base_url = (base_url or _env_or_config("base_url", "https://api.dify.ai")).rstrip("/")
        self.endpoint = f"{self.base_url}/v1/workflows/run"

        if not self.api_key:
            logger.warning(
                "Dify API Key 未配置。请设置环境变量 DIFY_API_KEY 或在 dify/config.json 中配置。"
                "匹配将回退到本地管道。"
            )

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def _call_workflow(self, inputs: dict, timeout: int = 60) -> Optional[dict]:
        """调用 Dify Workflow API"""
        if not self.available:
            return None

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "inputs": inputs,
            "response_mode": "blocking",
            "user": "mdm-pipeline",
        }

        try:
            resp = requests.post(
                self.endpoint,
                json=payload,
                headers=headers,
                timeout=timeout,
            )
            resp.raise_for_status()
            data = resp.json()

            # Dify workflow 返回格式: {"data": {"outputs": {...}}}
            if "data" in data and "outputs" in data["data"]:
                return data["data"]["outputs"]
            return data

        except requests.exceptions.Timeout:
            logger.error("Dify API 请求超时 (>%ds)", timeout)
            return None
        except requests.exceptions.RequestException as e:
            logger.error("Dify API 请求失败: %s", e)
            return None
        except Exception as e:
            logger.error("Dify API 异常: %s", e)
            return None

    # ── 单对匹配 ──

    def match_customers(self, record_a: dict, record_b: dict) -> dict:
        """匹配两条客户记录"""
        inputs = {
            "entity_type": "customer",
            "source_a": record_a.get("source", ""),
            "source_b": record_b.get("source", ""),
            "company_name_a": record_a.get("company_name", ""),
            "region_a": record_a.get("region", ""),
            "city_a": record_a.get("city", ""),
            "address_a": record_a.get("address", ""),
            "phone_a": record_a.get("phone", ""),
            "tax_id_a": record_a.get("tax_id", ""),
            "company_name_b": record_b.get("company_name", ""),
            "region_b": record_b.get("region", ""),
            "city_b": record_b.get("city", ""),
            "address_b": record_b.get("address", ""),
            "phone_b": record_b.get("phone", ""),
            "tax_id_b": record_b.get("tax_id", ""),
        }
        result = self._call_workflow(inputs)
        if result is None:
            return {"decision": "review", "confidence": 0.5, "reasoning": "Dify 不可用", "signals": {}}
        return result

    def match_products(self, record_a: dict, record_b: dict) -> dict:
        """匹配两条商品记录"""
        inputs = {
            "entity_type": "product",
            "source_a": record_a.get("source", ""),
            "source_b": record_b.get("source", ""),
            "product_name_a": record_a.get("product_name", ""),
            "category_a": record_a.get("category", ""),
            "brand_a": record_a.get("brand", ""),
            "model_a": record_a.get("model", ""),
            "sku_a": record_a.get("sku", ""),
            "specification_a": record_a.get("specification", ""),
            "upc_a": record_a.get("upc", ""),
            "product_name_b": record_b.get("product_name", ""),
            "category_b": record_b.get("category", ""),
            "brand_b": record_b.get("brand", ""),
            "model_b": record_b.get("model", ""),
            "sku_b": record_b.get("sku", ""),
            "specification_b": record_b.get("specification", ""),
            "upc_b": record_b.get("upc", ""),
        }
        result = self._call_workflow(inputs)
        if result is None:
            return {"decision": "review", "confidence": 0.5, "reasoning": "Dify 不可用", "signals": {}}
        return result

    # ── 批量匹配（用于审核队列） ──

    def batch_match(
        self,
        entity_type: str,
        pairs: list[tuple[dict, dict]],
        delay: float = 0.5,
    ) -> list[dict]:
        """
        批量匹配，带有请求间隔避免 API 限流

        Args:
            entity_type: "customer" | "product"
            pairs: [(record_a, record_b), ...]
            delay: 请求间隔秒数

        Returns:
            [{"decision": ..., "confidence": ..., ...}, ...]
        """
        results = []
        match_fn = self.match_customers if entity_type == "customer" else self.match_products

        for i, (rec_a, rec_b) in enumerate(pairs):
            if i > 0:
                time.sleep(delay)
            logger.info("Dify 匹配 %d/%d ...", i + 1, len(pairs))
            result = match_fn(rec_a, rec_b)
            results.append(result)

        return results


# ── 便捷函数 ──

def create_dify_client() -> DifyMDMClient:
    """创建 Dify 客户端（自动读取配置）"""
    return DifyMDMClient()


if __name__ == "__main__":
    # 测试：本地模拟（无需 API Key）
    logging.basicConfig(level=logging.INFO)

    client = DifyMDMClient()
    print(f"Dify API 可用: {client.available}")

    if client.available:
        result = client.match_customers(
            {
                "source": "CRM",
                "company_name": "北京阿尔法科技有限公司",
                "region": "北京市",
                "city": "北京市",
                "address": "北京市海淀区中关村大街1号",
                "phone": "010-12345678",
                "tax_id": "91110108MA01XXXXX",
            },
            {
                "source": "ERP",
                "company_name": "北京Alpha科技公司",
                "region": "北京市",
                "city": "北京市",
                "address": "北京市海淀区中关村大街1号楼",
                "phone": "01012345678",
                "tax_id": "91110108MA01XXXXX",
            },
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("请在 dify/config.json 中配置 api_key 后重试。")

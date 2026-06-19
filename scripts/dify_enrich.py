"""
Dify 审核队列增强 — 用 LLM 对模糊匹配对做二次判断
====================================================
用法：
  python dify_enrich.py customer   # 用 Dify 增强客户审核队列
  python dify_enrich.py product    # 用 Dify 增强商品审核队列

输出：
  - data/customer_review_enriched.csv （新增 dify_decision, dify_confidence, dify_reasoning 列）
  - data/product_review_enriched.csv
"""

import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

# 添加 scripts 目录到 path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dify_client import DifyMDMClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

QUEUE_FILES = {
    "customer": DATA_DIR / "customer_review_queue.csv",
    "product": DATA_DIR / "product_review_queue.csv",
}
OUTPUT_FILES = {
    "customer": DATA_DIR / "customer_review_enriched.csv",
    "product": DATA_DIR / "product_review_enriched.csv",
}

# 客户字段映射：queue CSV 列名 → Dify API 字段名
CUSTOMER_FIELD_MAP = {
    "company_name_left": "company_name",
    "company_name_right": "company_name",
    "region_left": "region",
    "region_right": "region",
    "city_left": "city",
    "city_right": "city",
    "address_left": "address",
    "address_right": "address",
    "phone_left": "phone",
    "phone_right": "phone",
    "tax_id_left": "tax_id",
    "tax_id_right": "tax_id",
}

PRODUCT_FIELD_MAP = {
    "product_name_left": "product_name",
    "product_name_right": "product_name",
    "category_left": "category",
    "category_right": "category",
    "brand_left": "brand",
    "brand_right": "brand",
    "model_left": "model",
    "model_right": "model",
    "sku_left": "sku",
    "sku_right": "sku",
    "specification_left": "specification",
    "specification_right": "specification",
    "upc_left": "upc",
    "upc_right": "upc",
}


def row_to_record(row: pd.Series, side: str, field_map: dict) -> dict:
    """将 DataFrame 行转为 Dify API 需要的记录字典"""
    suffix = "_left" if side == "left" else "_right"
    record = {"source": row.get(f"source{suffix}", "")}
    for csv_col, api_field in field_map.items():
        if csv_col.endswith(suffix):
            record[api_field] = str(row.get(csv_col, ""))
    return record


def enrich_review_queue(entity: str, max_samples: int = 0, delay: float = 0.3) -> pd.DataFrame:
    """
    用 Dify LLM 增强审核队列

    Args:
        entity: "customer" | "product"
        max_samples: 最大处理数（0=全部）
        delay: API 请求间隔
    """
    queue_file = QUEUE_FILES[entity]
    if not queue_file.exists():
        logger.error("审核队列文件不存在: %s", queue_file)
        return pd.DataFrame()

    queue_df = pd.read_csv(queue_file, dtype=str).fillna("")
    if queue_df.empty:
        logger.info("审核队列为空，无需增强。")
        return queue_df

    client = DifyMDMClient()
    if not client.available:
        logger.warning("Dify 不可用，跳过增强。请在 dify/config.json 中配置 api_key。")
        queue_df["dify_decision"] = "skipped"
        queue_df["dify_confidence"] = ""
        queue_df["dify_reasoning"] = "Dify 未配置"
        return queue_df

    field_map = CUSTOMER_FIELD_MAP if entity == "customer" else PRODUCT_FIELD_MAP

    samples = queue_df.head(max_samples) if max_samples > 0 else queue_df
    total = len(samples)
    logger.info("开始用 Dify 增强 %d 条 %s 审核记录 ...", total, entity)

    decisions = []
    confidences = []
    reasonings = []

    for idx, (_, row) in enumerate(samples.iterrows()):
        rec_a = row_to_record(row, "left", field_map)
        rec_b = row_to_record(row, "right", field_map)

        if entity == "customer":
            result = client.match_customers(rec_a, rec_b)
        else:
            result = client.match_products(rec_a, rec_b)

        decisions.append(result.get("decision", "error"))
        confidences.append(result.get("confidence", 0))
        reasonings.append(str(result.get("reasoning", ""))[:200])

        if (idx + 1) % 10 == 0:
            logger.info("  进度: %d/%d", idx + 1, total)

        if idx < total - 1:
            time.sleep(delay)

    result_df = samples.copy()
    result_df["dify_decision"] = decisions
    result_df["dify_confidence"] = confidences
    result_df["dify_reasoning"] = reasonings

    # 如果只处理了部分，拼接未处理的行
    if max_samples > 0 and max_samples < len(queue_df):
        rest = queue_df.iloc[max_samples:].copy()
        rest["dify_decision"] = "skipped"
        rest["dify_confidence"] = ""
        rest["dify_reasoning"] = "超出采样上限"
        result_df = pd.concat([result_df, rest], ignore_index=True)

    output_file = OUTPUT_FILES[entity]
    result_df.to_csv(output_file, index=False, encoding="utf-8-sig")
    logger.info("增强结果已保存: %s (%d 条)", output_file.name, len(result_df))

    # 统计
    if decisions:
        merge_n = sum(1 for d in decisions if d == "merge")
        review_n = sum(1 for d in decisions if d == "review")
        no_match_n = sum(1 for d in decisions if d == "no_match")
        logger.info(
            "Dify 判断分布: merge=%d, review=%d, no_match=%d",
            merge_n, review_n, no_match_n,
        )

    return result_df


def main():
    if len(sys.argv) < 2:
        print("用法: python dify_enrich.py <customer|product> [max_samples]")
        print("示例: python dify_enrich.py customer 50")
        sys.exit(1)

    entity = sys.argv[1]
    if entity not in ("customer", "product"):
        print(f"无效实体: {entity}，请使用 customer 或 product")
        sys.exit(1)

    max_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    enrich_review_queue(entity, max_samples=max_samples)
    print(f"\n完成！增强文件: {OUTPUT_FILES[entity]}")


if __name__ == "__main__":
    main()

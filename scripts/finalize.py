"""
黄金记录生成 — 自动合并 + 人工审核 → 聚类 → 最终黄金记录
============================================================
用法:
  python finalize.py customer    # 生成客户黄金记录
  python finalize.py product     # 生成商品黄金记录
"""

import json
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

CONFIG = {
    "customer": {
        "source_files": {
            "CRM": "crm_customers.csv",
            "ERP": "erp_customers.csv",
            "ECommerce": "ecommerce_customers.csv",
        },
        "review_log": "customer_review_log.csv",
        "candidates": "customer_match_candidates.csv",
        "output": "final_golden_customers.csv",
        "metrics": "final_customer_metrics.json",
        "score_fields": [
            "company_name", "address", "phone", "tax_id",
            "website", "email", "contact_person",
        ],
        "golden_fields": [
            "company_name", "region", "city", "address",
            "phone", "tax_id", "website", "email",
            "contact_person", "source_created_at",
        ],
        "id_prefix": "FINAL",
        "canonical_col": "canonical_customer_id",
    },
    "product": {
        "source_files": {
            "ERP": "erp_products.csv",
            "ECommerce": "ecommerce_products.csv",
        },
        "review_log": "product_review_log.csv",
        "candidates": "product_match_candidates.csv",
        "output": "final_golden_products.csv",
        "metrics": "final_product_metrics.json",
        "score_fields": [
            "product_name", "category", "brand", "model",
            "sku", "specification", "upc", "price",
        ],
        "golden_fields": [
            "product_name", "category", "brand", "model",
            "sku", "specification", "upc", "price",
            "source_created_at",
        ],
        "id_prefix": "FINAL-P",
        "canonical_col": "canonical_product_id",
    },
}


class UnionFind:
    def __init__(self, items=None):
        self.parent = {}
        if items:
            for item in items:
                self.parent[item] = item

    def find(self, item):
        if item not in self.parent:
            self.parent[item] = item
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def load_source_data(entity: str) -> pd.DataFrame:
    cfg = CONFIG[entity]
    frames = []
    for source_name, filename in cfg["source_files"].items():
        path = DATA_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"缺少数据文件：{path}")
        df = pd.read_csv(path, dtype=str).fillna("")
        df["source"] = source_name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_auto_merges(entity: str) -> list[tuple[str, str]]:
    path = DATA_DIR / CONFIG[entity]["candidates"]
    if not path.exists():
        return []
    df = pd.read_csv(path, dtype=str).fillna("")
    auto = df[df["decision_band"] == "auto_merge"]
    return list(zip(auto["record_id_left"], auto["record_id_right"]))


def load_review_merges(entity: str) -> list[tuple[str, str]]:
    path = DATA_DIR / CONFIG[entity]["review_log"]
    if not path.exists():
        return []
    df = pd.read_csv(path, dtype=str).fillna("")
    pairs = []
    for _, row in df.iterrows():
        decision = str(row.get("decision", "")).strip()
        if any(t in decision for t in ["合", "同", "merge", "yes"]):
            pairs.append((row["record_id_left"], row["record_id_right"]))
    return pairs


def choose_best_record(group: pd.DataFrame, score_fields: list[str]) -> pd.Series:
    def completeness(row):
        return sum(bool(str(row[f]).strip()) for f in score_fields)

    return group.loc[group.apply(completeness, axis=1).idxmax()]


def build_golden(entity: str, all_data: pd.DataFrame, merge_pairs: list) -> pd.DataFrame:
    cfg = CONFIG[entity]
    uf = UnionFind(all_data["record_id"].tolist())
    for a, b in merge_pairs:
        uf.union(a, b)

    all_data["cluster_root"] = all_data["record_id"].apply(uf.find)
    grouped = []
    for cluster_id, group in all_data.groupby("cluster_root", sort=False):
        best = choose_best_record(group, cfg["score_fields"])
        record = {
            "final_entity_id": f"{cfg['id_prefix']}-{len(grouped) + 1:05d}",
            "cluster_root": cluster_id,
            f"canonical_{'customer' if entity == 'customer' else 'product'}_ids":
                ";".join(sorted(group[cfg["canonical_col"]].unique())),
            **{f: best[f] for f in cfg["golden_fields"]},
            "sources": ";".join(sorted(group["source"].unique())),
            "record_count": len(group),
            "record_ids": ";".join(sorted(group["record_id"].tolist())),
        }
        grouped.append(record)
    return pd.DataFrame(grouped)


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in CONFIG:
        print("用法: python finalize.py <customer|product>")
        sys.exit(1)

    entity = sys.argv[1]
    cfg = CONFIG[entity]
    entity_cn = "客户" if entity == "customer" else "商品"

    print(f"加载{entity_cn}数据...")
    all_data = load_source_data(entity)
    print(f"读取 {len(all_data)} 条{entity_cn}记录。")

    auto_merges = load_auto_merges(entity)
    manual_merges = load_review_merges(entity)
    merge_pairs = auto_merges + manual_merges
    print(f"自动合并对: {len(auto_merges)}，人工合并对: {len(manual_merges)}。")

    golden = build_golden(entity, all_data, merge_pairs)

    output_path = DATA_DIR / cfg["output"]
    golden.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"已生成最终黄金{entity_cn}表：{output_path.name} ({len(golden)} 条)")

    metrics = {
        "final_entity_count": int(len(golden)),
        "auto_merge_pairs": int(len(auto_merges)),
        "manual_merge_pairs": int(len(manual_merges)),
        "total_merge_pairs": int(len(merge_pairs)),
    }
    metrics_path = DATA_DIR / cfg["metrics"]
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"已生成最终指标文件：{metrics_path.name}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

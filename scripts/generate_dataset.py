import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

random.seed(42)
np.random.seed(42)
faker = Faker("zh_CN")

CHINESE_PROVINCES = [
    "北京市", "上海市", "广东省", "浙江省", "江苏省", "四川省", "山东省", "湖北省", "陕西省", "湖南省"
]
CITIES = {
    "北京市": ["北京市"],
    "上海市": ["上海市"],
    "广东省": ["广州市", "深圳市", "东莞市", "佛山市"],
    "浙江省": ["杭州市", "宁波市", "温州市", "绍兴市"],
    "江苏省": ["南京市", "苏州市", "无锡市", "常州市"],
    "四川省": ["成都市", "绵阳市", "德阳市", "宜宾市"],
    "山东省": ["青岛市", "济南市", "烟台市", "潍坊市"],
    "湖北省": ["武汉市", "宜昌市", "襄阳市", "荆州市"],
    "陕西省": ["西安市", "咸阳市", "宝鸡市", "渭南市"],
    "湖南省": ["长沙市", "株洲市", "湘潭市", "常德市"]
}

COMPANY_SUFFIXES = ["有限公司", "科技公司", "科技有限公司", "信息技术有限公司", "贸易有限公司", "实业有限公司"]
COMPANY_PREFIXES = ["北京", "上海", "广州", "深圳", "杭州", "南京", "成都", "青岛", "武汉", "长沙"]
COMPANY_ADJECTIVES = ["星辰", "宏达", "先锋", "瑞丰", "智科", "卓越", "恒信", "启迪", "天宇", "安联"]
PRODUCT_CATEGORIES = ["电子产品", "办公用品", "工业设备", "消费品", "软件服务", "家具", "食品", "医疗器械"]
PRODUCT_BRANDS = ["Alpha", "贝特", "宏鹏", "赛博", "天翼", "智造", "优品", "快优", "浩宇", "聚合"]
PRODUCT_MODELS = ["T100", "X5", "A8", "P30", "B2", "C9", "Z3", "M7", "S1", "L6"]
PRODUCT_SPECIFICATIONS = ["64GB", "128GB", "256GB", "512GB", "1TB", "双频", "高速", "工业级", "标准", "升级版"]

PHONE_PREFIXES = ["010", "021", "020", "025", "028", "0532", "0755", "0571", "0731", "027"]


def generate_tax_id() -> str:
    base = ''.join(random.choices("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=18))
    return base


def normalize_company_name(name: str) -> str:
    return name.replace("有限公司", "公司").replace("科技", "科").replace(" ", "")


def random_company_name() -> str:
    prefix = random.choice(COMPANY_PREFIXES)
    adjective = random.choice(COMPANY_ADJECTIVES)
    suffix = random.choice(COMPANY_SUFFIXES)
    if random.random() < 0.3:
        return f"{prefix}{adjective}{suffix}"
    mid = random.choice(["", "国际", "集团", "商务", "创新"])
    return f"{prefix}{adjective}{mid}{suffix}"


def company_name_variants(base_name: str) -> str:
    patterns = [
        lambda x: x,
        lambda x: x.replace("有限公司", "有限公司"),
        lambda x: x.replace("科技", "科技"),
        lambda x: x.replace("科技", "Alpha"),
        lambda x: x.replace("信息技术", "信息技术"),
        lambda x: x.replace("公司", "公司"),
    ]
    candidate = random.choice(patterns)(base_name)
    if random.random() < 0.35:
        candidate = candidate.replace("科技", "科").replace("有限公司", "公司")
    if random.random() < 0.2:
        candidate = candidate.replace("北京", "北京 ").replace("上海", "上海 ")
    if random.random() < 0.2:
        candidate = candidate.replace("有限公司", "有限公司")
    if random.random() < 0.15:
        candidate = candidate.replace("Alpha", "α")
    if random.random() < 0.1:
        candidate = candidate.replace("北京", "北京Alpha")
    return candidate


def random_address(region: str, city: str) -> str:
    street = faker.street_name()
    number = random.randint(1, 399)
    unit = random.choice(["号", "号楼", "号院", "栋"])
    return f"{region}{city}{street}{number}{unit}"


def random_phone() -> str:
    if random.random() < 0.5:
        return f"{random.choice(PHONE_PREFIXES)}-{random.randint(1000,9999)}-{random.randint(1000,9999)}"
    return f"1{random.randint(300,999)}{random.randint(10000000,99999999)}"


def random_email(company: str) -> str:
    local = company.replace("有限公司", "").replace("公司", "").replace("科技", "").replace(" ", "")
    local = ''.join(ch for ch in local if ch.isalnum())[:8].lower() or "contact"
    return f"{local}@example.com"


def random_product_name(category: str, brand: str, model: str) -> str:
    descriptor = random.choice(["Pro", "Max", "Lite", "Plus", "X", "S", "Classic"])
    if category == "工业设备":
        descriptor = random.choice(["系列", "套装", "模块", "平台"])
    return f"{brand} {model} {descriptor}"


def generate_customer_base(n_customers: int) -> list[dict]:
    customers = []
    for i in range(1, n_customers + 1):
        company = random_company_name()
        region = random.choice(CHINESE_PROVINCES)
        city = random.choice(CITIES[region])
        customers.append(
            {
                "canonical_customer_id": f"CUST{str(i).zfill(5)}",
                "company_name": company,
                "region": region,
                "city": city,
                "address": random_address(region, city),
                "phone": random_phone(),
                "tax_id": generate_tax_id(),
                "website": f"www.{company.replace('有限公司','').replace('公司','').replace('科技','').lower()}.com",
                "email": random_email(company),
                "contact_person": faker.name(),
                "source_created_at": faker.date_time_between(start_date='-3y', end_date='now').strftime('%Y-%m-%d'),
            }
        )
    return customers


def generate_product_base(n_products: int) -> list[dict]:
    products = []
    for i in range(1, n_products + 1):
        category = random.choice(PRODUCT_CATEGORIES)
        brand = random.choice(PRODUCT_BRANDS)
        model = random.choice(PRODUCT_MODELS)
        products.append(
            {
                "canonical_product_id": f"PROD{str(i).zfill(5)}",
                "product_name": random_product_name(category, brand, model),
                "category": category,
                "brand": brand,
                "model": model,
                "sku": f"SKU{random.randint(100000,999999)}",
                "specification": random.choice(PRODUCT_SPECIFICATIONS),
                "upc": f"{random.randint(100000000000,999999999999)}",
                "price": round(random.uniform(120, 9800), 2),
                "source_created_at": faker.date_time_between(start_date='-3y', end_date='now').strftime('%Y-%m-%d'),
            }
        )
    return products


def apply_customer_variation(base: dict) -> dict:
    company_name = company_name_variants(base["company_name"])
    if random.random() < 0.4:
        region = base["region"]
        city = base["city"]
    else:
        region = base["region"]
        city = base["city"]
    address = base["address"]
    if random.random() < 0.25:
        address = address.replace("号", "号楼").replace("栋", "栋").replace("院", "园")
    return {
        "company_name": company_name,
        "region": region,
        "city": city,
        "address": address,
        "phone": base["phone"] if random.random() < 0.7 else random_phone(),
        "tax_id": base["tax_id"] if random.random() < 0.9 else generate_tax_id(),
        "website": base["website"],
        "email": base["email"],
        "contact_person": base["contact_person"],
        "source_created_at": base["source_created_at"],
    }


def apply_product_variation(base: dict) -> dict:
    name = base["product_name"]
    if random.random() < 0.4:
        name = name.replace("Max", "Pro").replace("Plus", "LITE").replace("Classic", "标准")
    return {
        "product_name": name,
        "category": base["category"],
        "brand": base["brand"],
        "model": base["model"],
        "sku": base["sku"],
        "specification": base["specification"],
        "upc": base["upc"],
        "price": base["price"] if random.random() < 0.7 else round(base["price"] * random.uniform(0.95, 1.08), 2),
        "source_created_at": base["source_created_at"],
    }


def build_source_dataset(source_name: str, canonical_ids: list[str], base_data: dict[str, dict], entity_type: str) -> pd.DataFrame:
    rows = []
    for idx, cid in enumerate(canonical_ids, start=1):
        record_id = f"{source_name[:3].upper()}-{str(idx).zfill(6)}"
        base = base_data[cid]
        if entity_type == "customer":
            record = apply_customer_variation(base)
            record.update(
                {
                    "record_id": record_id,
                    "source": source_name,
                    "canonical_customer_id": cid,
                }
            )
        else:
            record = apply_product_variation(base)
            record.update(
                {
                    "record_id": record_id,
                    "source": source_name,
                    "canonical_product_id": cid,
                }
            )
        rows.append(record)
    return pd.DataFrame(rows)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"生成 {path.name}: {len(df)} 条记录")


def main() -> None:
    customer_total = 4200
    product_total = 2000
    customer_duplicate_pairs = 840
    product_duplicate_pairs = 400

    # 客户基准实体
    total_customer_entities = customer_total - customer_duplicate_pairs
    customer_base = generate_customer_base(total_customer_entities)
    customer_base_map = {item["canonical_customer_id"]: item for item in customer_base}

    # 产品基准实体
    total_product_entities = product_total - product_duplicate_pairs
    product_base = generate_product_base(total_product_entities)
    product_base_map = {item["canonical_product_id"]: item for item in product_base}

    # 生成跨源重复 ID
    duplicated_customer_ids = random.sample(list(customer_base_map.keys()), customer_duplicate_pairs)
    duplicated_product_ids = random.sample(list(product_base_map.keys()), product_duplicate_pairs)

    # 客户来源分配，确保重复实体在两个不同来源间出现，整体重复率约 20%
    crm_erp_ids = set(duplicated_customer_ids[:280])
    crm_ecommerce_ids = set(duplicated_customer_ids[280:560])
    erp_ecommerce_ids = set(duplicated_customer_ids[560:])

    duplicate_customer_ids = set(duplicated_customer_ids)
    remaining_customer_ids = list(set(customer_base_map.keys()) - duplicate_customer_ids)
    random.shuffle(remaining_customer_ids)

    crm_customer_ids = set(crm_erp_ids | crm_ecommerce_ids)
    erp_customer_ids = set(crm_erp_ids | erp_ecommerce_ids)
    ecommerce_customer_ids = set(crm_ecommerce_ids | erp_ecommerce_ids)

    while len(crm_customer_ids) < 1600:
        crm_customer_ids.add(remaining_customer_ids.pop())
    while len(erp_customer_ids) < 1400:
        erp_customer_ids.add(remaining_customer_ids.pop())
    while len(ecommerce_customer_ids) < 1200:
        ecommerce_customer_ids.add(remaining_customer_ids.pop())

    # 商品来源分配，重复实体在 ERP 和 ECommerce 两个来源出现，整体重复率约 20%
    duplicate_product_ids = set(duplicated_product_ids)
    remaining_product_ids = list(set(product_base_map.keys()) - duplicate_product_ids)
    random.shuffle(remaining_product_ids)

    erp_product_ids = set(duplicate_product_ids)
    ecommerce_product_ids = set(duplicate_product_ids)

    while len(erp_product_ids) < 1000:
        erp_product_ids.add(remaining_product_ids.pop())
    while len(ecommerce_product_ids) < 1000:
        ecommerce_product_ids.add(remaining_product_ids.pop())

    # 生成数据集
    crm_df = build_source_dataset("CRM", list(crm_customer_ids), customer_base_map, "customer")
    erp_cust_df = build_source_dataset("ERP", list(erp_customer_ids), customer_base_map, "customer")
    ecommerce_df = build_source_dataset("ECommerce", list(ecommerce_customer_ids), customer_base_map, "customer")
    erp_prod_df = build_source_dataset("ERP", list(erp_product_ids), product_base_map, "product")
    ecommerce_prod_df = build_source_dataset("ECommerce", list(ecommerce_product_ids), product_base_map, "product")

    write_csv(crm_df, DATA_DIR / "crm_customers.csv")
    write_csv(erp_cust_df, DATA_DIR / "erp_customers.csv")
    write_csv(ecommerce_df, DATA_DIR / "ecommerce_customers.csv")
    write_csv(erp_prod_df, DATA_DIR / "erp_products.csv")
    write_csv(ecommerce_prod_df, DATA_DIR / "ecommerce_products.csv")

    summary = {
        "crm_customers": len(crm_df),
        "erp_customers": len(erp_cust_df),
        "ecommerce_customers": len(ecommerce_df),
        "erp_products": len(erp_prod_df),
        "ecommerce_products": len(ecommerce_prod_df),
        "total_records": len(crm_df) + len(erp_cust_df) + len(ecommerce_df) + len(erp_prod_df) + len(ecommerce_prod_df),
        "customer_duplicate_pairs": customer_duplicate_pairs,
        "product_duplicate_pairs": product_duplicate_pairs,
    }
    summary_path = DATA_DIR / "dataset_summary.json"
    pd.Series(summary).to_json(summary_path, force_ascii=False, indent=2)
    print(f"生成 summary: {summary_path.name}")


if __name__ == "__main__":
    main()

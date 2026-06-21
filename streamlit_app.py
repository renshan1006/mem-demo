"""
MDM 主数据管理智能清洗系统 — 人工审核界面
============================================
功能：
  Tab 1 - 仪表板：匹配统计、分数分布、决策概览
  Tab 2 - 审核队列：可筛选的待审核列表，点击进入详情
  Tab 3 - 详情审核：双侧对比卡片 + 审核决策表单
  Tab 4 - 审核历史：分页日志表 + 导出功能
  Tab 5 - 黄金记录：最终主数据指标与下载
"""

import json
import textwrap
from datetime import datetime
from math import ceil
from pathlib import Path
import pandas as pd
import streamlit as st

# ── 路径常量 ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

QUEUE_FILES = {
    "customer": DATA_DIR / "customer_review_queue.csv",
    "product": DATA_DIR / "product_review_queue.csv",
}
LOG_FILES = {
    "customer": DATA_DIR / "customer_review_log.csv",
    "product": DATA_DIR / "product_review_log.csv",
}
METRICS_FILES = {
    "customer": DATA_DIR / "customer_match_metrics.json",
    "product": DATA_DIR / "product_match_metrics.json",
}
FINAL_METRICS_FILES = {
    "customer": DATA_DIR / "final_customer_metrics.json",
    "product": DATA_DIR / "final_product_metrics.json",
}
GOLDEN_FILES = {
    "customer": DATA_DIR / "final_golden_customers.csv",
    "product": DATA_DIR / "final_golden_products.csv",
}

# ── 字段配置 ────────────────────────────────────────────────────────────────
FIELD_CONFIG = {
    "customer": {
        "left": [
            ("record_id_left", "Record ID"),
            ("company_name_left", "公司名称"),
            ("region_left", "地区"),
            ("city_left", "城市"),
            ("address_left", "地址"),
            ("phone_left", "电话"),
            ("tax_id_left", "税号"),
            ("contact_person_left", "联系人"),
        ],
        "right": [
            ("record_id_right", "Record ID"),
            ("company_name_right", "公司名称"),
            ("region_right", "地区"),
            ("city_right", "城市"),
            ("address_right", "地址"),
            ("phone_right", "电话"),
            ("tax_id_right", "税号"),
            ("contact_person_right", "联系人"),
        ],
    },
    "product": {
        "left": [
            ("record_id_left", "Record ID"),
            ("product_name_left", "商品名称"),
            ("category_left", "品类"),
            ("brand_left", "品牌"),
            ("model_left", "型号"),
            ("sku_left", "SKU"),
            ("specification_left", "规格"),
            ("upc_left", "条码"),
            ("price_left", "价格"),
        ],
        "right": [
            ("record_id_right", "Record ID"),
            ("product_name_right", "商品名称"),
            ("category_right", "品类"),
            ("brand_right", "品牌"),
            ("model_right", "型号"),
            ("sku_right", "SKU"),
            ("specification_right", "规格"),
            ("upc_right", "条码"),
            ("price_right", "价格"),
        ],
    },
}

HIGHLIGHT_DIFF_FIELDS = {
    "customer": {"phone", "tax_id"},
    "product": {"sku", "upc", "price"},
}

SOURCE_LABEL_MAP = {
    "CRM": "CRM 系统",
    "ERP": "ERP 系统",
    "ECommerce": "电商平台",
}

DECISION_LABEL_MAP = {
    "合并": "✅ 合并",
    "不合并": "❌ 不合并",
    "保留待定": "⏳ 保留待定",
}

STATUS_COLOR = {
    "合并": "success",
    "不合并": "neutral",
    "保留待定": "warn",
    "高": "success",
    "中": "warn",
    "低": "neutral",
}


# ═══════════════════════════════════════════════════════════════════════════════
# 数据加载 / 持久化
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=5)
def load_review_queue(entity: str) -> pd.DataFrame:
    """加载审核队列（带短缓存以支持实时刷新）"""
    queue_file = QUEUE_FILES[entity]
    if not queue_file.exists():
        return pd.DataFrame()
    return pd.read_csv(queue_file, dtype=str).fillna("")


def load_review_log(entity: str) -> pd.DataFrame:
    """加载审核日志"""
    log_file = LOG_FILES[entity]
    if log_file.exists():
        return pd.read_csv(log_file, dtype=str).fillna("")
    return pd.DataFrame(
        columns=[
            "record_id_left", "source_left", "record_id_right", "source_right",
            "match_score", "decision", "comment", "updated_at",
        ]
    )


def load_json(filepath: Path) -> dict:
    """安全加载 JSON 文件"""
    if filepath.exists():
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_golden(entity: str) -> pd.DataFrame:
    """加载最终黄金记录"""
    gf = GOLDEN_FILES[entity]
    if gf.exists():
        return pd.read_csv(gf, dtype=str).fillna("")
    return pd.DataFrame()


def append_review_log(entity: str, row: dict) -> None:
    """追加一条审核决策到日志"""
    log_file = LOG_FILES[entity]
    log_df = load_review_log(entity)
    log_df = pd.concat([log_df, pd.DataFrame([row])], ignore_index=True)
    log_df.to_csv(log_file, index=False, encoding="utf-8-sig")


def remove_last_log_entry(entity: str) -> bool:
    """撤销最后一条审核日志（用于 Undo）"""
    log_file = LOG_FILES[entity]
    log_df = load_review_log(entity)
    if log_df.empty:
        return False
    log_df = log_df.iloc[:-1]
    log_df.to_csv(log_file, index=False, encoding="utf-8-sig")
    return True


def get_reviewed_pairs(entity: str) -> set:
    """获取已审核的 (record_id_left, record_id_right) 集合"""
    log_df = load_review_log(entity)
    if log_df.empty:
        return set()
    return set(
        (str(r["record_id_left"]), str(r["record_id_right"]))
        for _, r in log_df.iterrows()
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 格式化工具
# ═══════════════════════════════════════════════════════════════════════════════

def fmt_score(score) -> str:
    try:
        return f"{float(score):.2f}"
    except (ValueError, TypeError):
        return str(score)


def fmt_num(value) -> str:
    try:
        return f"{int(value):,}"
    except (ValueError, TypeError):
        return str(value)


def score_to_confidence(score: float) -> tuple[str, str, str]:
    """
    返回 (置信度标签, 颜色hex, 图标)
    高分 >= 0.80: 高置信度 → 可自动合并
    中分 0.65-0.80: 中等置信度 → 需审核
    低分 < 0.65: 低置信度 → 倾向不合并
    """
    if score >= 0.80:
        return "高", "#34D399", "🟢"
    elif score >= 0.65:
        return "中", "#FBBF24", "🟡"
    else:
        return "低", "#F472B6", "🔴"


def esc(text) -> str:
    """HTML 转义"""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def _md(html: str) -> None:
    """st.markdown 包装：自动去缩进 + unsafe_allow_html"""
    st.markdown(textwrap.dedent(html), unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# CSS 样式（Playful Geometric 设计系统）
# ═══════════════════════════════════════════════════════════════════════════════

CSS = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

  /* ═══════════════════════════════════════════════════════════════════════════
     CSS VARIABLES (Design Tokens)
     ═══════════════════════════════════════════════════════════════════════════ */
  :root {
    --bg: #FFFDF5;
    --fg: #1E293B;
    --muted: #F1F5F9;
    --muted-fg: #64748B;
    --accent: #8B5CF6;
    --accent-hover: #7C3AED;
    --secondary: #F472B6;
    --tertiary: #FBBF24;
    --quaternary: #34D399;
    --border: #E2E8F0;
    --border-dark: #1E293B;
    --card: #FFFFFF;
    --radius-sm: 8px;
    --radius-md: 16px;
    --radius-lg: 24px;
    --radius-full: 9999px;
    --shadow: 4px 4px 0px 0px #1E293B;
    --shadow-hover: 6px 6px 0px 0px #1E293B;
    --shadow-active: 2px 2px 0px 0px #1E293B;
    --bounce: cubic-bezier(0.34, 1.56, 0.64, 1);
  }

  /* ═══════════════════════════════════════════════════════════════════════════
     GLOBAL
     ═══════════════════════════════════════════════════════════════════════════ */
  .stApp {
    background: var(--bg);
    font-family: 'Plus Jakarta Sans', system-ui, sans-serif;
    color: var(--fg);
  }
  .stMain {
    background-image: radial-gradient(circle, #CBD5E1 1.2px, transparent 1.2px);
    background-size: 22px 22px;
  }
  /* Top-right decorative blob */
  .stMain::before {
    content: '';
    position: fixed; top: -120px; right: -100px;
    width: 400px; height: 400px;
    border-radius: 50%;
    background: rgba(139, 92, 246, 0.04);
    border: 3px solid rgba(139, 92, 246, 0.08);
    pointer-events: none; z-index: 0;
  }

  /* ═══════════════════════════════════════════════════════════════════════════
     TYPOGRAPHY
     ═══════════════════════════════════════════════════════════════════════════ */
  h1, h2, h3, h4, h5, h6 {
    font-family: 'Outfit', system-ui, sans-serif !important;
    font-weight: 700 !important;
    color: var(--fg) !important;
    letter-spacing: -0.02em;
  }
  h3 { font-size: 1.25rem !important; }
  p, div, span, label, input, textarea, select, button, td, th, li, a {
    font-family: 'Plus Jakarta Sans', system-ui, sans-serif;
  }

  /* ═══════════════════════════════════════════════════════════════════════════
     CARDS — "Sticker" cards with bouncy hover
     ═══════════════════════════════════════════════════════════════════════════ */
  .card {
    background: var(--card);
    border: 2px solid var(--border-dark);
    border-radius: var(--radius-md);
    padding: 20px;
    box-shadow: var(--shadow);
    margin-bottom: 14px;
    transition: transform 0.3s var(--bounce), box-shadow 0.3s var(--bounce);
    animation: card-enter 0.5s var(--bounce) both;
    position: relative;
  }
  .card:hover {
    box-shadow: var(--shadow-hover);
    transform: rotate(-0.5deg) scale(1.01);
  }
  .card-sm { padding: 14px; border-radius: 12px; }

  /* Staggered card entrance */
  .card:nth-child(1) { animation-delay: 0.05s; }
  .card:nth-child(2) { animation-delay: 0.1s; }
  .card:nth-child(3) { animation-delay: 0.15s; }
  .card:nth-child(4) { animation-delay: 0.2s; }
  .card:nth-child(5) { animation-delay: 0.25s; }

  /* ── Decorative card variants ── */
  .card-accent { border-color: var(--accent); box-shadow: 4px 4px 0px 0px var(--accent); }
  .card-accent:hover { box-shadow: 6px 6px 0px 0px var(--accent); }
  .card-pink { border-color: var(--secondary); box-shadow: 4px 4px 0px 0px var(--secondary); }
  .card-pink:hover { box-shadow: 6px 6px 0px 0px var(--secondary); }
  .card-amber { border-color: var(--tertiary); box-shadow: 4px 4px 0px 0px var(--tertiary); }
  .card-amber:hover { box-shadow: 6px 6px 0px 0px var(--tertiary); }
  .card-emerald { border-color: var(--quaternary); box-shadow: 4px 4px 0px 0px var(--quaternary); }
  .card-emerald:hover { box-shadow: 6px 6px 0px 0px var(--quaternary); }

  /* ── Blob-radius card ── */
  .card-blob { border-radius: var(--radius-lg) 8px var(--radius-lg) 8px; }

  /* ── Floating icon circle (top-right corner) ── */
  .card-icon-circle {
    width: 42px; height: 42px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
    border: 2px solid var(--border-dark);
    position: absolute; top: -21px; right: 16px;
    box-shadow: 2px 2px 0px 0px var(--border-dark);
    z-index: 1;
  }
  .card-icon-circle.violet { background: var(--accent); }
  .card-icon-circle.pink { background: var(--secondary); }
  .card-icon-circle.amber { background: var(--tertiary); }
  .card-icon-circle.emerald { background: var(--quaternary); }

  /* ── Sidebar card (compact) ── */
  .sidebar-card {
    background: var(--card); border: 2px solid var(--border-dark);
    border-radius: var(--radius-md); padding: 14px 12px 12px 12px;
    position: relative; margin-top: 20px;
    transition: transform 0.25s var(--bounce), box-shadow 0.25s var(--bounce);
  }
  .sidebar-card-accent { border-color: var(--accent); box-shadow: 3px 3px 0px 0px var(--accent); }
  .sidebar-card-accent:hover { box-shadow: 5px 5px 0px 0px var(--accent); transform: translateY(-1px); }
  .sidebar-card-emerald { border-color: var(--quaternary); box-shadow: 3px 3px 0px 0px var(--quaternary); }
  .sidebar-card-emerald:hover { box-shadow: 5px 5px 0px 0px var(--quaternary); transform: translateY(-1px); }
  .sidebar-card-pink { border-color: var(--secondary); box-shadow: 3px 3px 0px 0px var(--secondary); }
  .sidebar-card-pink:hover { box-shadow: 5px 5px 0px 0px var(--secondary); transform: translateY(-1px); }

  /* ── Sidebar icon circle (smaller) ── */
  .sidebar-icon {
    width: 32px; height: 32px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 14px;
    border: 2px solid var(--border-dark);
    position: absolute; top: -16px; right: 10px;
    box-shadow: 2px 2px 0px 0px var(--border-dark);
    z-index: 1;
  }
  .sidebar-icon.violet { background: var(--accent); }
  .sidebar-icon.pink { background: var(--secondary); }
  .sidebar-icon.amber { background: var(--tertiary); }
  .sidebar-icon.emerald { background: var(--quaternary); }

  /* ── Sidebar metric text (compact) ── */
  .sidebar-metric-val {
    font-family: 'Outfit', system-ui, sans-serif;
    font-size: 20px; font-weight: 800; color: var(--fg);
    line-height: 1.2;
  }
  .sidebar-metric-lbl {
    font-size: 11px; color: var(--muted-fg); font-weight: 500;
    margin-top: 2px;
  }
  .sidebar-metric-sub {
    font-size: 10px; color: #94A3B8; margin-top: 2px;
  }

  /* ═══════════════════════════════════════════════════════════════════════════
     HERO / PAGE HEADER
     ═══════════════════════════════════════════════════════════════════════════ */
  .hero-section {
    position: relative;
    padding: 28px 0 8px 0;
  }
  .page-title {
    font-family: 'Outfit', system-ui, sans-serif;
    font-size: 30px; font-weight: 800; color: var(--fg);
    margin: 0 0 6px 0;
    position: relative; display: inline-block;
  }
  .page-title::after {
    content: '';
    position: absolute;
    bottom: -6px; left: 0; right: 0;
    height: 4px; border-radius: 2px;
    background: repeating-linear-gradient(
      90deg,
      var(--accent) 0px, var(--accent) 12px,
      var(--secondary) 12px, var(--secondary) 24px,
      var(--tertiary) 24px, var(--tertiary) 36px,
      var(--quaternary) 36px, var(--quaternary) 48px
    );
  }
  .page-subtitle {
    font-size: 13px; color: var(--muted-fg); margin: 0;
    font-family: 'Plus Jakarta Sans', system-ui, sans-serif;
  }
  .hero-stats {
    display: flex; gap: 24px; margin-top: 12px;
  }
  .hero-stat {
    text-align: center;
  }
  .hero-stat-val {
    font-family: 'Outfit', system-ui, sans-serif;
    font-size: 22px; font-weight: 800; color: var(--fg);
  }
  .hero-stat-lbl {
    font-size: 10px; color: var(--muted-fg);
    text-transform: uppercase; letter-spacing: 0.05em;
  }

  /* ═══════════════════════════════════════════════════════════════════════════
     METRICS
     ═══════════════════════════════════════════════════════════════════════════ */
  .metric-val {
    font-family: 'Outfit', system-ui, sans-serif;
    font-size: 32px; font-weight: 800; color: var(--fg);
    margin-bottom: 4px; line-height: 1.1;
  }
  .metric-val.sm { font-size: 24px; }
  .metric-lbl {
    font-size: 11px; color: var(--muted-fg); letter-spacing: 0.06em;
    text-transform: uppercase; font-weight: 600;
  }
  .metric-sub {
    font-size: 12px; color: #94A3B8; margin-top: 4px;
  }

  /* ═══════════════════════════════════════════════════════════════════════════
     PROGRESS BARS
     ═══════════════════════════════════════════════════════════════════════════ */
  .progress-track {
    width: 100%; height: 12px; border-radius: var(--radius-full);
    background: var(--border); overflow: hidden; margin-top: 8px;
    border: 2px solid var(--border-dark);
  }
  .progress-fill {
    height: 100%; border-radius: var(--radius-full);
    background: var(--accent);
    transition: width 0.6s var(--bounce);
  }
  .progress-fill.high { background: var(--quaternary); }
  .progress-fill.mid  { background: var(--tertiary); }
  .progress-fill.low  { background: var(--secondary); }

  /* ═══════════════════════════════════════════════════════════════════════════
     SCORE GAUGE
     ═══════════════════════════════════════════════════════════════════════════ */
  .score-gauge {
    display: flex; align-items: center; gap: 10px;
  }
  .score-ring {
    width: 72px; height: 72px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-family: 'Outfit', system-ui, sans-serif;
    font-size: 22px; font-weight: 700; color: var(--fg);
    border: 3px solid var(--border-dark);
    box-shadow: 3px 3px 0px 0px var(--border-dark);
    background: var(--card);
  }

  /* ═══════════════════════════════════════════════════════════════════════════
     GRIDS
     ═══════════════════════════════════════════════════════════════════════════ */
  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
  .grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 14px; }
  .grid-4 { display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 14px; }

  /* ═══════════════════════════════════════════════════════════════════════════
     COMPARE CARDS
     ═══════════════════════════════════════════════════════════════════════════ */
  .compare-card {
    background: #F8FAFC;
    border: 2px solid var(--border);
    border-radius: var(--radius-md); padding: 18px;
    transition: transform 0.3s var(--bounce), box-shadow 0.3s var(--bounce);
  }
  .compare-card:hover {
    box-shadow: 4px 4px 0px 0px var(--border);
    transform: translateY(-2px);
  }
  .compare-header {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 14px; padding-bottom: 10px;
    border-bottom: 2px dashed #CBD5E1;
  }
  .compare-title {
    font-family: 'Outfit', system-ui, sans-serif;
    color: var(--fg); font-size: 15px; font-weight: 700;
  }
  .compare-badge {
    color: #475569; font-size: 11px;
    background: #F1F5F9; padding: 4px 12px; border-radius: var(--radius-full);
    border: 1px solid var(--border);
  }
  .compare-row {
    display: grid; grid-template-columns: 1fr 1.4fr; align-items: center;
    gap: 6px 16px; padding: 9px 0;
    border-bottom: 1px solid #F1F5F9;
  }
  .compare-row:last-child { border-bottom: none; }
  .compare-row.header {
    font-size: 11px; color: var(--muted-fg); letter-spacing: 0.06em;
    text-transform: uppercase; padding-bottom: 8px;
    border-bottom: 2px solid var(--border);
  }
  .compare-label { font-size: 11px; color: var(--muted-fg); font-weight: 500; }
  .compare-value {
    font-size: 13px; color: var(--fg); word-break: break-word;
    text-align: right; font-weight: 500;
  }
  .compare-value.diff {
    color: #D97706; font-weight: 700;
    background: #FEF3C7; padding: 2px 8px; border-radius: 4px;
    border: 1px solid var(--tertiary);
  }
  .compare-value.match { color: #059669; }

  /* ═══════════════════════════════════════════════════════════════════════════
     FORMS — Candy Buttons & Playful Inputs
     ═══════════════════════════════════════════════════════════════════════════ */
  .form-section {
    background: #F8FAFC;
    border: 2px solid var(--border);
    border-radius: var(--radius-md); padding: 18px;
  }
  .stRadio > div {
    background: #F8FAFC !important;
    padding: 10px !important; border-radius: var(--radius-md) !important;
    border: 2px solid var(--border) !important;
  }
  .stRadio label {
    color: var(--fg) !important;
    font-family: 'Plus Jakarta Sans', system-ui, sans-serif !important;
  }
  .stTextArea>div textarea {
    border-radius: 12px !important;
    background: var(--card) !important;
    color: var(--fg) !important;
    border: 2px solid var(--border) !important;
    font-family: 'Plus Jakarta Sans', system-ui, sans-serif !important;
    transition: all 0.25s var(--bounce) !important;
  }
  .stTextArea>div textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 4px 4px 0px 0px var(--accent) !important;
    outline: none !important;
  }

  /* ── Candy Button (Primary) ── */
  .stButton>button {
    border-radius: var(--radius-full) !important;
    padding: 0.6rem 1.6rem !important;
    background-color: var(--accent) !important;
    color: #FFFFFF !important;
    border: 2px solid var(--border-dark) !important;
    font-weight: 700 !important;
    font-family: 'Plus Jakarta Sans', system-ui, sans-serif !important;
    box-shadow: var(--shadow) !important;
    transition: all 0.25s var(--bounce) !important;
    letter-spacing: 0.01em;
  }
  .stButton>button:hover {
    transform: translate(-2px, -2px) !important;
    box-shadow: var(--shadow-hover) !important;
    background-color: var(--accent-hover) !important;
  }
  .stButton>button:active {
    transform: translate(2px, 2px) !important;
    box-shadow: var(--shadow-active) !important;
  }
  .stButton>button:disabled {
    opacity: 0.45; transform: none !important; box-shadow: var(--shadow) !important;
  }

  /* ── Secondary Button (transparent → yellow fill on hover) ── */
  .btn-skip>button {
    background-color: transparent !important;
    color: var(--fg) !important;
    border: 2px solid var(--border-dark) !important;
    box-shadow: none !important;
  }
  .btn-skip>button:hover {
    background-color: var(--tertiary) !important;
    color: var(--fg) !important;
    transform: translate(-2px, -2px) !important;
    box-shadow: var(--shadow-hover) !important;
  }
  .btn-undo>button {
    background-color: #FEF2F2 !important;
    color: #DC2626 !important;
    border: 2px solid #FECACA !important;
  }
  .btn-undo>button:hover {
    background-color: #FEE2E2 !important;
    border-color: #FCA5A5 !important;
  }

  /* ═══════════════════════════════════════════════════════════════════════════
     TABLES
     ═══════════════════════════════════════════════════════════════════════════ */
  .review-table {
    width: 100%; border-collapse: separate; border-spacing: 0 6px;
  }
  .review-table th, .review-table td {
    padding: 10px 12px; text-align: left; font-size: 12px;
  }
  .review-table th {
    color: var(--muted-fg); font-weight: 600; letter-spacing: 0.03em;
    text-transform: uppercase; font-size: 10px;
  }
  .review-table td {
    background: #F8FAFC; color: #334155;
    border-top: 1px solid #F1F5F9;
  }
  .review-table td:first-child { border-radius: var(--radius-sm) 0 0 var(--radius-sm); }
  .review-table td:last-child { border-radius: 0 var(--radius-sm) var(--radius-sm) 0; }
  .review-table tr:hover td { background: #EDE9FE; }

  /* ═══════════════════════════════════════════════════════════════════════════
     BADGES
     ═══════════════════════════════════════════════════════════════════════════ */
  .badge {
    display: inline-flex; align-items: center; gap: 4px;
    border-radius: var(--radius-full); padding: 3px 12px;
    font-size: 11px; font-weight: 600;
    border: 2px solid var(--border-dark);
  }
  .badge-success { background: #D1FAE5; color: #065F46; border-color: var(--quaternary); }
  .badge-neutral { background: #F1F5F9; color: #475569; border-color: #CBD5E1; }
  .badge-warn { background: #FEF3C7; color: #92400E; border-color: var(--tertiary); }
  .badge-accent { background: #EDE9FE; color: #5B21B6; border-color: var(--accent); }
  .badge-pink { background: #FCE7F3; color: #9D174D; border-color: var(--secondary); }
  .badge-emerald { background: #D1FAE5; color: #065F46; border-color: var(--quaternary); }
  .badge-amber { background: #FEF3C7; color: #92400E; border-color: var(--tertiary); }

  /* ═══════════════════════════════════════════════════════════════════════════
     PAGINATION
     ═══════════════════════════════════════════════════════════════════════════ */
  .pager {
    display: flex; justify-content: space-between; align-items: center;
    margin-top: 14px; color: var(--muted-fg); font-size: 12px;
  }
  .pager-btn {
    border: 2px solid var(--border); background: var(--card);
    border-radius: var(--radius-full); color: var(--fg); padding: 6px 14px;
    cursor: pointer; font-size: 12px; font-weight: 600;
    transition: all 0.25s var(--bounce);
  }
  .pager-btn:hover {
    background: #F8FAFC; border-color: #94A3B8;
    transform: translateY(-1px);
  }

  /* ═══════════════════════════════════════════════════════════════════════════
     DATA TABLE (Queue Preview)
     ═══════════════════════════════════════════════════════════════════════════ */
  .data-table {
    width: 100%; border-collapse: separate; border-spacing: 0 4px;
  }
  .data-table th {
    color: var(--muted-fg); font-size: 10px; font-weight: 600;
    letter-spacing: 0.04em; padding: 8px 10px; text-align: left;
    position: sticky; top: 0; background: var(--bg); z-index: 1;
    text-transform: uppercase;
  }
  .data-table td {
    padding: 8px 10px; font-size: 12px; color: #334155;
    background: #F8FAFC; border-top: 1px solid #F1F5F9;
  }
  .data-table tr:hover td { background: #EDE9FE; cursor: pointer; }
  .data-table tr.reviewed td { opacity: 0.5; }

  /* ── Queue row card style ── */
  .queue-row {
    background: var(--card);
    border: 2px solid var(--border);
    border-radius: 12px;
    padding: 12px 16px;
    margin-bottom: 8px;
    display: flex; align-items: center; gap: 12px;
    transition: all 0.25s var(--bounce);
    cursor: pointer;
  }
  .queue-row:hover {
    border-color: var(--accent);
    box-shadow: 3px 3px 0px 0px var(--accent);
    transform: translateX(4px);
  }
  .queue-row.reviewed { opacity: 0.55; }
  .queue-row.reviewed:hover { border-color: var(--border); box-shadow: none; transform: none; }

  /* ═══════════════════════════════════════════════════════════════════════════
     TAB NAVIGATION (Vertical Sidebar Nav) — Playful Geometric
     ═══════════════════════════════════════════════════════════════════════════ */
  [data-testid="stSidebar"] div[data-testid="stRadio"] > div[role="radiogroup"] {
    gap: 6px;
    background: transparent;
    border: none;
    border-radius: 0;
    padding: 0;
    display: flex;
    flex-direction: column;
    box-shadow: none;
  }
  [data-testid="stSidebar"] div[data-testid="stRadio"] label {
    border-radius: var(--radius-md);
    color: var(--fg) !important;
    font-size: 14px;
    padding: 11px 16px;
    transition: all 0.3s var(--bounce);
    cursor: pointer;
    font-weight: 600;
    font-family: 'Plus Jakarta Sans', system-ui, sans-serif;
    border: 2px solid transparent;
    margin: 0;
    position: relative;
    background: #FAFAFA;
  }
  [data-testid="stSidebar"] div[data-testid="stRadio"] label:hover {
    background: #F1F5F9;
    border-color: var(--border);
    transform: translateX(3px);
    box-shadow: 2px 2px 0px 0px var(--border);
  }
  [data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input:checked) {
    background: #EDE9FE !important;
    color: var(--accent) !important;
    font-weight: 700;
    border-color: var(--border-dark);
    box-shadow: 3px 3px 0px 0px var(--accent);
    transform: translateX(2px);
  }
  /* 选中项左侧色条 */
  [data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input:checked)::before {
    content: '';
    position: absolute;
    left: 0; top: 8px; bottom: 8px;
    width: 4px;
    border-radius: 0 4px 4px 0;
    background: var(--accent);
  }

  /* ═══════════════════════════════════════════════════════════════════════════
     OTHER STREAMLIT OVERRIDES
     ═══════════════════════════════════════════════════════════════════════════ */
  .stSelectbox>div {
    border-radius: 12px !important;
    background: var(--card) !important;
    border: 2px solid var(--border) !important;
    transition: all 0.25s var(--bounce) !important;
  }
  .stSelectbox>div:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 4px 4px 0px 0px var(--accent) !important;
  }
  .stSelectbox label {
    color: var(--muted-fg) !important; font-size: 12px !important;
    font-family: 'Plus Jakarta Sans', system-ui, sans-serif !important;
  }
  .stSlider>div>div { color: var(--fg) !important; }
  hr { border-color: var(--border); margin: 20px 0; border-width: 2px; }
  .entity-tag {
    display: inline-block; padding: 4px 14px; border-radius: var(--radius-full);
    font-size: 12px; font-weight: 600;
    border: 2px solid var(--border-dark);
  }
  .entity-tag.customer { background: #EDE9FE; color: #5B21B6; border-color: var(--accent); }
  .entity-tag.product { background: #D1FAE5; color: #065F46; border-color: var(--quaternary); }

  /* ── Streamlit progress bar ── */
  .stProgress > div > div {
    background-color: var(--accent) !important;
    border-radius: var(--radius-full) !important;
  }
  .stProgress > div {
    background-color: var(--border) !important;
    border-radius: var(--radius-full) !important;
    border: 2px solid var(--border-dark);
  }

  /* ── Streamlit expander ── */
  .streamlit-expanderHeader {
    font-family: 'Outfit', system-ui, sans-serif !important;
    font-weight: 700 !important;
    color: var(--fg) !important;
    border: 2px solid var(--border) !important;
    border-radius: 12px !important;
    transition: all 0.25s var(--bounce) !important;
  }
  .streamlit-expanderHeader:hover {
    border-color: var(--accent) !important;
  }

  /* ── Streamlit metric widget ── */
  [data-testid="stMetricValue"] {
    font-family: 'Outfit', system-ui, sans-serif !important;
    font-weight: 800 !important;
    color: var(--fg) !important;
  }
  [data-testid="stMetricLabel"] {
    font-family: 'Plus Jakarta Sans', system-ui, sans-serif !important;
    color: var(--muted-fg) !important;
  }

  /* ── Streamlit form ── */
  [data-testid="stForm"] {
    border: 2px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    padding: 16px !important;
    background: #F8FAFC !important;
  }

  /* ═══════════════════════════════════════════════════════════════════════════
     ANIMATIONS
     ═══════════════════════════════════════════════════════════════════════════ */
  @keyframes card-enter {
    0% { opacity: 0; transform: scale(0.92) translateY(12px); }
    70% { transform: scale(1.02); }
    100% { opacity: 1; transform: scale(1) translateY(0); }
  }
  @keyframes float {
    0%, 100% { transform: translateY(0px) rotate(0deg); }
    25% { transform: translateY(-6px) rotate(3deg); }
    75% { transform: translateY(4px) rotate(-3deg); }
  }
  @keyframes float-reverse {
    0%, 100% { transform: translateY(0px) rotate(0deg); }
    25% { transform: translateY(6px) rotate(-3deg); }
    75% { transform: translateY(-4px) rotate(3deg); }
  }
  @keyframes wiggle {
    0%, 100% { transform: rotate(0deg); }
    25% { transform: rotate(-3deg); }
    75% { transform: rotate(3deg); }
  }
  @keyframes pop-in {
    0% { transform: scale(0); opacity: 0; }
    70% { transform: scale(1.15); }
    100% { transform: scale(1); opacity: 1; }
  }
  @keyframes marquee {
    0% { transform: translateX(0); }
    100% { transform: translateX(-50%); }
  }
  @keyframes pulse-ring {
    0%, 100% { box-shadow: 3px 3px 0px 0px var(--border-dark); }
    50% { box-shadow: 5px 5px 0px 0px var(--accent); }
  }

  /* ═══════════════════════════════════════════════════════════════════════════
     DECORATIVE ELEMENTS
     ═══════════════════════════════════════════════════════════════════════════ */
  .deco-circle {
    border-radius: 50%; border: 2px solid var(--border-dark);
    position: absolute; pointer-events: none; z-index: 0;
  }

  /* ── Confetti container ── */
  .confetti-container {
    position: fixed; top: 0; left: 0; width: 100%; height: 100%;
    pointer-events: none; z-index: 0; overflow: hidden;
  }
  .confetti-piece {
    position: absolute; border: 2px solid var(--border-dark); opacity: 0.65;
  }
  .confetti-piece:nth-child(1) {
    top: 6%; left: 4%; width: 20px; height: 20px;
    border-radius: 50%; background: var(--secondary);
    animation: float 4.5s ease-in-out infinite;
  }
  .confetti-piece:nth-child(2) {
    top: 10%; right: 6%; width: 16px; height: 16px;
    background: var(--tertiary);
    animation: float-reverse 5.5s ease-in-out infinite 0.5s;
  }
  .confetti-piece:nth-child(3) {
    top: 18%; left: 10%; width: 0; height: 0;
    border-left: 10px solid transparent;
    border-right: 10px solid transparent;
    border-bottom: 18px solid var(--quaternary);
    background: transparent;
    animation: float 3.8s ease-in-out infinite 1s;
  }
  .confetti-piece:nth-child(4) {
    top: 4%; right: 12%; width: 18px; height: 18px;
    border-radius: 4px; background: var(--accent);
    animation: float-reverse 4.8s ease-in-out infinite 0.8s;
  }
  .confetti-piece:nth-child(5) {
    top: 22%; left: 2%; width: 0; height: 0;
    border-left: 9px solid transparent;
    border-right: 9px solid transparent;
    border-bottom: 16px solid var(--secondary);
    background: transparent;
    animation: float 6s ease-in-out infinite 1.5s;
  }
  .confetti-piece:nth-child(6) {
    top: 6%; right: 22%; width: 14px; height: 14px;
    border-radius: 50%; background: var(--tertiary);
    animation: float-reverse 4s ease-in-out infinite 2s;
  }
  .confetti-piece:nth-child(7) {
    top: 12%; left: 18%; width: 22px; height: 10px;
    border-radius: var(--radius-full); background: var(--accent);
    animation: wiggle 4.5s ease-in-out infinite 0.3s;
  }
  .confetti-piece:nth-child(8) {
    top: 26%; right: 4%; width: 0; height: 0;
    border-left: 12px solid transparent;
    border-right: 12px solid transparent;
    border-bottom: 22px solid var(--quaternary);
    background: transparent;
    animation: float 5s ease-in-out infinite 1.2s;
  }

  /* ── Squiggly divider ── */
  .squiggle-divider {
    height: 12px; margin: 10px 0;
    background: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 120 12' preserveAspectRatio='none'%3E%3Cpath d='M0,6 Q10,0 20,6 T40,6 T60,6 T80,6 T100,6 T120,6' fill='none' stroke='%238B5CF6' stroke-width='2.5'/%3E%3C/svg%3E") repeat-x;
    background-size: 60px 12px;
  }

  /* ── Diagonal stripe pattern (for hero backgrounds) ── */
  .bg-stripes {
    background-image: repeating-linear-gradient(
      45deg,
      transparent,
      transparent 8px,
      rgba(139, 92, 246, 0.04) 8px,
      rgba(139, 92, 246, 0.04) 10px
    );
  }

  /* ── Dot accent ── */
  .dot-accent {
    display: inline-block; width: 8px; height: 8px;
    border-radius: 50%; border: 2px solid var(--border-dark);
    margin-right: 6px; vertical-align: middle;
  }
  .dot-accent.violet { background: var(--accent); }
  .dot-accent.pink { background: var(--secondary); }
  .dot-accent.amber { background: var(--tertiary); }
  .dot-accent.emerald { background: var(--quaternary); }

  /* ═══════════════════════════════════════════════════════════════════════════
     SIDEBAR
     ═══════════════════════════════════════════════════════════════════════════ */
  [data-testid="stSidebar"] {
    background: var(--bg);
    border-right: 2px solid var(--border-dark);
  }
  [data-testid="stSidebar"] .stSelectbox>div {
    background: var(--card) !important;
    border: 2px solid var(--border) !important;
    border-radius: 12px !important;
  }

  /* ═══════════════════════════════════════════════════════════════════════════
     SECTION HEADER
     ═══════════════════════════════════════════════════════════════════════════ */
  .section-header {
    display: flex; align-items: center; gap: 10px;
    margin-bottom: 16px; padding-bottom: 10px;
    border-bottom: 2px solid var(--border);
  }
  .section-header-icon {
    width: 36px; height: 36px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 16px; border: 2px solid var(--border-dark);
    flex-shrink: 0;
  }
  .section-header-icon.violet { background: var(--accent); }
  .section-header-icon.pink { background: var(--secondary); }
  .section-header-icon.amber { background: var(--tertiary); }
  .section-header-icon.emerald { background: var(--quaternary); }

  .section-header-title {
    font-family: 'Outfit', system-ui, sans-serif;
    font-size: 18px; font-weight: 700; color: var(--fg);
  }
  .section-header-sub {
    font-size: 12px; color: var(--muted-fg);
  }

  /* ═══════════════════════════════════════════════════════════════════════════
     ACCESSIBILITY: prefers-reduced-motion
     ═══════════════════════════════════════════════════════════════════════════ */
  @media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
      animation-duration: 0.01ms !important;
      animation-iteration-count: 1 !important;
      transition-duration: 0.01ms !important;
    }
    .card:hover { transform: none; }
    .stButton>button:hover { transform: none !important; }
    .queue-row:hover { transform: none; }
    .sidebar-card:hover { transform: none; }
  }

  /* ═══════════════════════════════════════════════════════════════════════════
     RESPONSIVE
     ═══════════════════════════════════════════════════════════════════════════ */
  @media (max-width: 768px) {
    .grid-2, .grid-3, .grid-4 { grid-template-columns: 1fr; }
    .page-title { font-size: 22px; }
    .card { padding: 14px; }
    .hero-stats { gap: 12px; }
    .hero-stat-val { font-size: 18px; }
    .card-icon-circle { display: none; }
    .sidebar-icon { display: none; }
  }
</style>
"""

# ═══════════════════════════════════════════════════════════════════════════════
# 组件：对比卡片
# ═══════════════════════════════════════════════════════════════════════════════

def render_compare_card(entity: str, row: pd.Series, side: str) -> str:
    """渲染一侧的对比卡片，返回 HTML 字符串"""
    prefix = "left" if side == "left" else "right"
    source_raw = row.get(f"source_{prefix}", "")
    source_label = SOURCE_LABEL_MAP.get(source_raw, source_raw)

    field_cfg = FIELD_CONFIG[entity][side]
    rows_html_parts = []
    for field_name, label in field_cfg:
        value = row.get(field_name, "")
        # 检测是否与另一侧不同
        other_field = field_name.replace(f"_{prefix}_", "_right_" if prefix == "left" else "_left_")
        other_value = row.get(other_field, "")
        is_different = str(value) != str(other_value)
        field_key = field_name.replace(f"_{prefix}_", "")
        is_highlight = is_different and field_key in HIGHLIGHT_DIFF_FIELDS.get(entity, set())

        css_class = ""
        if is_highlight:
            css_class = "diff"
        elif not is_different and value:
            css_class = "match"

        rows_html_parts.append(
            f"<div class='compare-row'>"
            f"<div class='compare-label'>{esc(label)}</div>"
            f"<div class='compare-value {css_class}'>{esc(value)}</div>"
            f"</div>"
        )

    return textwrap.dedent(f"""\
    <div class='compare-card'>
      <div class='compare-header'>
        <div class='compare-title'>{esc(source_label)}</div>
        <div class='compare-badge'>ID: {esc(row.get(f"record_id_{prefix}", ""))}</div>
      </div>
      <div class='compare-row header'>
        <div class='compare-label'>字段</div>
        <div class='compare-value'>值</div>
      </div>
      {"".join(rows_html_parts)}
    </div>\
    """)


# ═══════════════════════════════════════════════════════════════════════════════
# 组件：评分仪表
# ═══════════════════════════════════════════════════════════════════════════════

def render_score_gauge(score: float) -> str:
    """渲染评分仪表盘 HTML"""
    pct = min(max(score, 0), 1) * 100
    label, color, icon = score_to_confidence(score)

    if pct >= 80:
        bar_class = "high"
    elif pct >= 65:
        bar_class = "mid"
    else:
        bar_class = "low"

    return textwrap.dedent(f"""\
    <div class='card card-sm' style='text-align:center;'>
      <div class='metric-lbl'>匹配分数</div>
      <div style='font-size:36px;font-weight:700;color:{color};margin:8px 0;'>
        {score:.2f}
      </div>
      <div class='badge badge-{STATUS_COLOR.get(label, "neutral")}' style='justify-content:center;'>
        {icon} {label} 置信度
      </div>
      <div class='progress-track'>
        <div class='progress-fill {bar_class}' style='width:{pct:.0f}%;'></div>
      </div>
      <div class='metric-sub' style='margin-top:8px;'>
        ≥0.80 自动合并 · 0.65-0.80 审核 · &lt;0.65 不合并
      </div>
    </div>\
    """)


# ═══════════════════════════════════════════════════════════════════════════════
# 组件：审核日志表格
# ═══════════════════════════════════════════════════════════════════════════════

def render_log_table(log_df: pd.DataFrame, page_key: str, page_size: int = 10) -> None:
    """渲染分页审核日志表格"""
    if log_df.empty:
        st.info("暂无审核记录")
        return

    if page_key not in st.session_state:
        st.session_state[page_key] = 1

    total = len(log_df)
    page_count = max(1, ceil(total / page_size))
    current = min(max(1, st.session_state[page_key]), page_count)
    start = (current - 1) * page_size
    page_data = log_df.iloc[start:start + page_size].copy()

    rows_html = ""
    for _, row in page_data.iterrows():
        decision = str(row.get("decision", ""))
        color = STATUS_COLOR.get(decision, "neutral")
        label = DECISION_LABEL_MAP.get(decision, decision)
        rows_html += f"""
        <tr>
          <td>{esc(row.get("updated_at", ""))}</td>
          <td><code>{esc(row.get("record_id_left", ""))}</code></td>
          <td><code>{esc(row.get("record_id_right", ""))}</code></td>
          <td><strong>{fmt_score(row.get("match_score", ""))}</strong></td>
          <td><span class='badge badge-{color}'>{label}</span></td>
          <td>{esc(row.get("comment", ""))}</td>
        </tr>
        """

    st.markdown(
        f"""
        <div class='card'>
          <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;'>
            <span style='color:#1E293B;font-size:16px;font-weight:600;'>📋 审核记录</span>
            <span style='color:#64748B;font-size:12px;'>共 {total} 条</span>
          </div>
          <div style='max-height:500px;overflow-y:auto;'>
            <table class='review-table'>
              <thead>
                <tr>
                  <th>时间</th><th>左侧 ID</th><th>右侧 ID</th>
                  <th>分数</th><th>决策</th><th>备注</th>
                </tr>
              </thead>
              <tbody>{rows_html}</tbody>
            </table>
          </div>
          <div class='pager'>
            <button class='pager-btn' onclick='return false;'>第 {current} / {page_count} 页</button>
            <span>每页 {page_size} 条</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 分页按钮（使用 Streamlit 原生按钮）
    c1, c2, c3 = st.columns([1, 2, 1])
    with c1:
        if st.button("◀ 上一页", disabled=(current <= 1), key=f"{page_key}_prev"):
            st.session_state[page_key] = current - 1
            st.rerun()
    with c3:
        if st.button("下一页 ▶", disabled=(current >= page_count), key=f"{page_key}_next"):
            st.session_state[page_key] = current + 1
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# 页面：仪表板
# ═══════════════════════════════════════════════════════════════════════════════

def render_dashboard(entity: str) -> None:
    """渲染仪表板 Tab"""
    st.markdown(
        """<div class='section-header'>
          <div class='section-header-icon violet'>📊</div>
          <div>
            <div class='section-header-title'>匹配管道仪表板</div>
            <div class='section-header-sub'>全局概览 · 实时统计 · 分数分布</div>
          </div>
        </div>""",
        unsafe_allow_html=True,
    )

    queue_df = load_review_queue(entity)
    log_df = load_review_log(entity)
    metrics = load_json(METRICS_FILES[entity])
    final_metrics = load_json(FINAL_METRICS_FILES[entity])

    # 计算审核统计
    total_queue = len(queue_df)
    total_log = len(log_df)
    remaining = max(0, total_queue - len(get_reviewed_pairs(entity)))

    merge_count = len(log_df[log_df["decision"] == "合并"]) if not log_df.empty else 0
    not_merge_count = len(log_df[log_df["decision"] == "不合并"]) if not log_df.empty else 0
    pending_count = len(log_df[log_df["decision"] == "保留待定"]) if not log_df.empty else 0

    merge_rate = (merge_count / total_log * 100) if total_log > 0 else 0

    # ── 顶部指标卡片（带浮动图标） ──
    _md(f"""\
    <div class='grid-4'>
      <div class='card card-sm card-accent' style='margin-top:22px;'>
        <div class='card-icon-circle violet'>📥</div>
        <div class='metric-val'>{fmt_num(total_queue)}</div>
        <div class='metric-lbl'>待审核候选对</div>
        <div class='metric-sub'>剩余 {remaining} 条未审核</div>
      </div>
      <div class='card card-sm card-emerald' style='margin-top:22px;'>
        <div class='card-icon-circle emerald'>✅</div>
        <div class='metric-val'>{fmt_num(total_log)}</div>
        <div class='metric-lbl'>已审核记录</div>
        <div class='metric-sub'>进度 {total_log/max(total_queue,1)*100:.1f}%</div>
      </div>
      <div class='card card-sm card-emerald' style='margin-top:22px;'>
        <div class='card-icon-circle emerald'>🔗</div>
        <div class='metric-val' style='color:var(--quaternary);'>{merge_count}</div>
        <div class='metric-lbl'>合并决策</div>
        <div class='metric-sub'>合并率 {merge_rate:.1f}%</div>
      </div>
      <div class='card card-sm card-pink' style='margin-top:22px;'>
        <div class='card-icon-circle pink'>❌</div>
        <div class='metric-val' style='color:var(--secondary);'>{not_merge_count}</div>
        <div class='metric-lbl'>不合并决策</div>
        <div class='metric-sub'>待定 {pending_count} 条</div>
      </div>
    </div>\
    """)

    # ── 第二行：管道指标 + 审核进度 ──
    col1, col2 = st.columns([1, 1])

    with col1:
        _md(f"""\
        <div class='card' style='margin-top:22px;'>
          <div class='card-icon-circle violet'>🔧</div>
          <div style='color:var(--fg);font-size:14px;font-weight:600;margin-bottom:12px;margin-top:4px;'>
            自动匹配管道指标
          </div>
          <table style='width:100%;color:#334155;font-size:13px;'>
            <tr><td style='padding:6px 0;color:var(--muted-fg);'>总候选对数</td>
                <td style='text-align:right;'><strong>{fmt_num(metrics.get("total_candidates", 0))}</strong></td></tr>
            <tr><td style='padding:6px 0;color:var(--muted-fg);'>自动合并 (≥0.80)</td>
                <td style='text-align:right;color:var(--quaternary);'><strong>{fmt_num(metrics.get("auto_merge_count", 0))}</strong></td></tr>
            <tr><td style='padding:6px 0;color:var(--muted-fg);'>需审核 (0.50-0.80)</td>
                <td style='text-align:right;color:var(--tertiary);'><strong>{fmt_num(metrics.get("review_count", 0))}</strong></td></tr>
            <tr><td style='padding:6px 0;color:var(--muted-fg);'>判定不同 (&lt;0.50)</td>
                <td style='text-align:right;color:var(--secondary);'><strong>{fmt_num(metrics.get("no_match_count", 0))}</strong></td></tr>
            <tr><td style='padding:6px 0;color:var(--muted-fg);'>审核队列中真匹配比例</td>
                <td style='text-align:right;'><strong>{metrics.get("review_true_ratio", 0):.1%}</strong></td></tr>
          </table>
        </div>\
        """)

    with col2:
        _md(f"""\
        <div class='card' style='margin-top:22px;'>
          <div class='card-icon-circle emerald'>🏆</div>
          <div style='color:var(--fg);font-size:14px;font-weight:600;margin-bottom:12px;margin-top:4px;'>
            最终黄金记录
          </div>
          <table style='width:100%;color:#334155;font-size:13px;'>
            <tr><td style='padding:6px 0;color:var(--muted-fg);'>最终实体数</td>
                <td style='text-align:right;'><strong style='font-size:22px;color:var(--fg);'>{fmt_num(final_metrics.get("final_entity_count", 0))}</strong></td></tr>
            <tr><td style='padding:6px 0;color:var(--muted-fg);'>自动合并对</td>
                <td style='text-align:right;color:var(--quaternary);'><strong>{fmt_num(final_metrics.get("auto_merge_pairs", 0))}</strong></td></tr>
            <tr><td style='padding:6px 0;color:var(--muted-fg);'>人工合并对</td>
                <td style='text-align:right;color:var(--tertiary);'><strong>{fmt_num(final_metrics.get("manual_merge_pairs", 0))}</strong></td></tr>
            <tr><td style='padding:6px 0;color:var(--muted-fg);'>总合并对</td>
                <td style='text-align:right;'><strong>{fmt_num(final_metrics.get("total_merge_pairs", 0))}</strong></td></tr>
            <tr><td style='padding:6px 0;color:var(--muted-fg);'>去重率</td>
                <td style='text-align:right;'><strong>{final_metrics.get("dedup_rate", "N/A")}</strong></td></tr>
          </table>
        </div>\
        """)

    # ── 分数分布 ──
    if not queue_df.empty:
        _md("""<div class='card' style='margin-top:22px;'>
          <div class='card-icon-circle amber'>📈</div>
          <div style='color:var(--fg);font-size:14px;font-weight:600;margin-bottom:12px;margin-top:4px;'>审核队列分数分布</div></div>""")
        scores = pd.to_numeric(queue_df["match_score"], errors="coerce").dropna()
        if len(scores) > 0:
            try:
                import plotly.graph_objects as go  # type: ignore
            except ImportError:
                st.info("💡 安装 `plotly` 可查看交互式分数分布图：`pip install plotly`")
                return

            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=scores,
                nbinsx=30,
                marker_color='#8B5CF6',
                marker_line_color='rgba(0,0,0,0.08)',
                marker_line_width=1,
                hovertemplate='分数区间: %{x:.3f}<br>数量: %{y}<extra></extra>',
            ))
            # 添加阈值线
            for threshold, color, label in [(0.50, "#F472B6", "低/中分界"), (0.80, "#34D399", "中/高分界")]:
                fig.add_vline(
                    x=threshold, line_dash="dash", line_color=color,
                    annotation_text=label, annotation_position="top",
                    opacity=0.7, line_width=1.5,
                )
            fig.update_layout(
                template="plotly_white",
                paper_bgcolor="#FFFFFF",
                plot_bgcolor="#F8FAFC",
                font=dict(color="#64748B", size=11),
                margin=dict(l=20, r=20, t=10, b=10),
                height=280,
                xaxis_title="匹配分数",
                yaxis_title="候选对数量",
                bargap=0.05,
            )
            st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 页面：审核队列（可筛选表格）
# ═══════════════════════════════════════════════════════════════════════════════

def render_review_queue(entity: str) -> None:
    """渲染审核队列 Tab —— 可筛选、可点击跳转的数据表格"""
    st.markdown(
        """<div class='section-header'>
          <div class='section-header-icon amber'>🔍</div>
          <div>
            <div class='section-header-title'>审核队列浏览器</div>
            <div class='section-header-sub'>筛选 · 排序 · 快速跳转审核</div>
          </div>
        </div>""",
        unsafe_allow_html=True,
    )

    queue_df = load_review_queue(entity)
    reviewed_pairs = get_reviewed_pairs(entity)

    if queue_df.empty:
        entity_name = "客户" if entity == "customer" else "商品"
        st.warning(f"当前没有{entity_name}审核队列数据。请先运行匹配管道生成队列文件。")
        return

    # ── 筛选器 ──
    with st.container():
        st.markdown("<div class='card card-sm'>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
        with c1:
            score_min, score_max = st.slider(
                "匹配分数范围",
                min_value=0.0, max_value=1.0,
                value=(0.50, 0.80),
                step=0.01,
                key=f"score_range_{entity}",
            )
        with c2:
            pair_sources = ["全部"] + sorted(queue_df["pair_source"].unique().tolist())
            selected_source = st.selectbox(
                "源对类型",
                pair_sources,
                key=f"source_filter_{entity}",
            )
        with c3:
            show_filter = st.radio(
                "审核状态",
                ["全部", "待审核", "已审核"],
                horizontal=True,
                key=f"reviewed_filter_{entity}",
            )
        with c4:
            sort_by = st.selectbox(
                "排序方式",
                ["分数降序", "分数升序", "公司名称升序" if entity == "customer" else "商品名称升序"],
                key=f"sort_filter_{entity}",
            )
        st.markdown("</div>", unsafe_allow_html=True)

    # ── 筛选 ──
    filtered = queue_df.copy()
    filtered["_score_num"] = pd.to_numeric(filtered["match_score"], errors="coerce")
    filtered = filtered[
        (filtered["_score_num"] >= score_min) & (filtered["_score_num"] <= score_max)
    ]

    if selected_source != "全部":
        filtered = filtered[filtered["pair_source"] == selected_source]

    reviewed_set = reviewed_pairs
    if show_filter == "待审核":
        filtered = filtered[
            ~filtered.apply(
                lambda r: (str(r["record_id_left"]), str(r["record_id_right"])) in reviewed_set,
                axis=1,
            )
        ]
    elif show_filter == "已审核":
        filtered = filtered[
            filtered.apply(
                lambda r: (str(r["record_id_left"]), str(r["record_id_right"])) in reviewed_set,
                axis=1,
            )
        ]

    # 排序
    if sort_by == "分数降序":
        filtered = filtered.sort_values("_score_num", ascending=False)
    elif sort_by == "分数升序":
        filtered = filtered.sort_values("_score_num", ascending=True)
    elif "名称" in sort_by:
        name_col = "company_name_left" if entity == "customer" else "product_name_left"
        filtered = filtered.sort_values(name_col)

    # ── 分页 ──
    table_page_key = f"queue_page_{entity}"
    if table_page_key not in st.session_state:
        st.session_state[table_page_key] = 1

    page_size = 15
    total = len(filtered)
    page_count = max(1, ceil(total / page_size))
    current_page = min(max(1, st.session_state[table_page_key]), page_count)
    start = (current_page - 1) * page_size
    page_data = filtered.iloc[start:start + page_size]

    st.markdown(
        f"<div style='color:#64748B;font-size:12px;margin-bottom:8px;'>"
        f"显示 {start+1}-{min(start+page_size, total)} / 共 {total} 条"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── 生成可交互表格 ──
    for idx, (_, row) in enumerate(page_data.iterrows()):
        global_idx = start + idx
        pair = (str(row["record_id_left"]), str(row["record_id_right"]))
        is_reviewed = pair in reviewed_set
        score_val = float(row["match_score"])
        _, color, icon = score_to_confidence(score_val)

        row_class = "reviewed" if is_reviewed else ""
        reviewed_tag = '<span class="badge badge-success" style="font-size:10px;">✓ 已审核</span>' if is_reviewed else ""

        name_left = row.get("company_name_left" if entity == "customer" else "product_name_left", "")
        name_right = row.get("company_name_right" if entity == "customer" else "product_name_right", "")
        source_left = SOURCE_LABEL_MAP.get(row.get("source_left", ""), row.get("source_left", ""))
        source_right = SOURCE_LABEL_MAP.get(row.get("source_right", ""), row.get("source_right", ""))

        # 每一行是一个可点击的按钮
        col1, col2, col3, col4, col5, col6 = st.columns([3, 2, 1.5, 1.5, 1, 1.5])
        with col1:
            st.markdown(
                f"<div style='font-size:12px;color:#1E293B;'>{esc(name_left)}</div>"
                f"<div style='font-size:11px;color:#64748B;'>{esc(name_right)}</div>",
                unsafe_allow_html=True,
            )
        with col2:
            region_or_cat = row.get("region_left" if entity == "customer" else "category_left", "")
            st.markdown(
                f"<div style='font-size:11px;color:#64748B;'>{esc(source_left)} ↔ {esc(source_right)}</div>"
                f"<div style='font-size:10px;color:#94A3B8;'>{esc(region_or_cat)}</div>",
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                f"<span style='color:{color};font-weight:700;font-size:14px;'>{score_val:.4f}</span>"
                f"<span style='font-size:10px;color:#64748B;margin-left:4px;'>{icon}</span>",
                unsafe_allow_html=True,
            )
        with col4:
            st.markdown(reviewed_tag, unsafe_allow_html=True)
        with col5:
            st.markdown(
                f"<span style='font-size:10px;color:#64748B;'>#{global_idx+1}</span>",
                unsafe_allow_html=True,
            )
        with col6:
            if st.button("🔍 审核", key=f"review_btn_{entity}_{global_idx}", use_container_width=True):
                st.session_state[f"current_index_{entity}"] = int(
                    queue_df[
                        (queue_df["record_id_left"] == row["record_id_left"])
                        & (queue_df["record_id_right"] == row["record_id_right"])
                    ].index[0]
                )
                # 切换到详情审核 tab（索引 2）
                st.session_state["active_tab"] = 2
                st.rerun()

        st.markdown("<hr style='margin:4px 0;opacity:0.3;'>", unsafe_allow_html=True)

    # ── 分页导航 ──
    c1, c2, c3, c4 = st.columns([1, 2, 2, 1])
    with c1:
        if st.button("◀◀ 首页", disabled=(current_page <= 1), key=f"queue_first_{entity}"):
            st.session_state[table_page_key] = 1
            st.rerun()
    with c2:
        if st.button("◀ 上一页", disabled=(current_page <= 1), key=f"queue_prev_{entity}"):
            st.session_state[table_page_key] = current_page - 1
            st.rerun()
    with c4:
        if st.button("下一页 ▶", disabled=(current_page >= page_count), key=f"queue_next_{entity}"):
            st.session_state[table_page_key] = current_page + 1
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# 页面：详情审核
# ═══════════════════════════════════════════════════════════════════════════════

def render_detail_review(entity: str) -> None:
    """渲染详情审核 Tab —— 双侧对比 + 决策表单"""
    queue_df = load_review_queue(entity)
    log_df = load_review_log(entity)
    reviewed_pairs = get_reviewed_pairs(entity)

    if queue_df.empty:
        entity_name = "客户" if entity == "customer" else "商品"
        st.warning(f"当前没有{entity_name}审核队列数据。")
        return

    st.markdown(
        """<div class='section-header'>
          <div class='section-header-icon pink'>📝</div>
          <div>
            <div class='section-header-title'>详情审核</div>
            <div class='section-header-sub'>双侧对比 · 智能建议 · 决策提交</div>
          </div>
        </div>""",
        unsafe_allow_html=True,
    )

    # ── Dify 增强数据 ──
    dify_data = {}
    enriched_file = DATA_DIR / f"{entity}_review_enriched.csv"
    if enriched_file.exists():
        try:
            enriched_df = pd.read_csv(enriched_file, dtype=str).fillna("")
            enriched_df["_left"] = enriched_df["record_id_left"].astype(str)
            enriched_df["_right"] = enriched_df["record_id_right"].astype(str)
            dify_data = {
                (r["_left"], r["_right"]): {
                    "decision": r.get("dify_decision", ""),
                    "confidence": r.get("dify_confidence", ""),
                    "reasoning": r.get("dify_reasoning", ""),
                }
                for _, r in enriched_df.iterrows()
            }
        except Exception:
            pass

    # ── 会话状态初始化 ──
    session_key = f"current_index_{entity}"
    if session_key not in st.session_state:
        st.session_state[session_key] = 0

    max_index = len(queue_df) - 1
    current_index = min(max(0, st.session_state[session_key]), max_index)
    st.session_state[session_key] = current_index
    current_row = queue_df.iloc[current_index]
    current_pair = (str(current_row["record_id_left"]), str(current_row["record_id_right"]))
    is_already_reviewed = current_pair in reviewed_pairs

    score_val = float(current_row["match_score"])
    conf_label, conf_color, conf_icon = score_to_confidence(score_val)

    # ── Dify 建议 ──
    dify = dify_data.get(current_pair, {})
    has_dify = bool(dify.get("decision"))
    if has_dify:
        dify_decision = dify["decision"]
        dify_conf = dify["confidence"]
        dify_reason = dify["reasoning"]
        if dify_decision == "merge":
            dify_color = "#34D399"; dify_icon = "🤖"; dify_label = "LLM 建议合并"
        elif dify_decision == "no_match":
            dify_color = "#F472B6"; dify_icon = "🤖"; dify_label = "LLM 建议不合并"
        else:
            dify_color = "#FBBF24"; dify_icon = "🤖"; dify_label = "LLM 建议保留"

    # ── 顶部导航栏 ──
    _md(f"""\
    <div class='card' style='display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;'>
      <div style='display:flex;align-items:center;gap:16px;'>
        <span style='color:#1E293B;font-weight:600;'>📝 审核详情</span>
        <span class='entity-tag {"customer" if entity == "customer" else "product"}'>
          {"👤 客户" if entity == "customer" else "📦 商品"}
        </span>
        <span class='badge badge-{STATUS_COLOR.get(conf_label, "neutral")}'>
          {conf_icon} {conf_label}置信度
        </span>
        {f'<span class="badge badge-success">✓ 已审核</span>' if is_already_reviewed else '<span class="badge badge-warn">⏳ 待审核</span>'}
      </div>
      <div style='color:#64748B;font-size:13px;'>
        第 <strong style='color:#1E293B;'>{current_index + 1}</strong> / {len(queue_df)} 条
      </div>
    </div>\
    """)

    # ── 导航按钮行 ──
    nav_col1, nav_col2, nav_col3, nav_col4, nav_col5 = st.columns([1, 1, 2, 1, 1])
    with nav_col1:
        if st.button("⏮ 首条", disabled=(current_index == 0), key=f"nav_first_{entity}", use_container_width=True):
            st.session_state[session_key] = 0
            st.rerun()
    with nav_col2:
        if st.button("◀ 上一条", disabled=(current_index == 0), key=f"nav_prev_{entity}", use_container_width=True):
            st.session_state[session_key] = current_index - 1
            st.rerun()
    with nav_col3:
        jump_val = st.number_input(
            "跳转到",
            min_value=1, max_value=len(queue_df),
            value=current_index + 1, step=1,
            key=f"nav_jumpval_{entity}",
            label_visibility="collapsed",
        )
        if st.button("🎯 跳转", key=f"nav_jump_{entity}"):
            st.session_state[session_key] = int(jump_val) - 1
            st.rerun()
    with nav_col4:
        if st.button("下一条 ▶", disabled=(current_index >= max_index), key=f"nav_next_{entity}", use_container_width=True):
            st.session_state[session_key] = current_index + 1
            st.rerun()
    with nav_col5:
        if st.button("⏭ 末条", disabled=(current_index >= max_index), key=f"nav_last_{entity}", use_container_width=True):
            st.session_state[session_key] = max_index
            st.rerun()

    # ── 主内容区：评分 + 双侧对比 ──
    gauge_col, compare_col = st.columns([1, 3])

    with gauge_col:
        st.markdown(render_score_gauge(score_val), unsafe_allow_html=True)

        # 源对信息
        pair_source = current_row.get("pair_source", "")
        _md(f"""\
        <div class='card card-sm' style='text-align:center;margin-top:12px;'>
          <div class='metric-lbl'>源对类型</div>
          <div style='font-size:16px;color:#1E293B;margin-top:4px;'>{esc(pair_source)}</div>
          <div style='font-size:11px;color:#94A3B8;margin-top:4px;'>
            {SOURCE_LABEL_MAP.get(current_row.get("source_left",""), "")} ↔ {SOURCE_LABEL_MAP.get(current_row.get("source_right",""), "")}
          </div>
        </div>\
        """)

        # Dify LLM 建议
        if has_dify:
            dify_conf_val = float(dify_conf) if dify_conf else 0
            _md(f"""\
            <div class='card card-sm' style='text-align:center;margin-top:12px;border-color:{dify_color};'>
              <div class='metric-lbl'>🤖 Dify LLM 建议</div>
              <div style='font-size:20px;font-weight:700;color:{dify_color};margin:6px 0;'>
                {dify_icon} {dify_label}
              </div>
              <div style='font-size:11px;color:#64748B;'>置信度 {dify_conf_val:.2f}</div>
              <div style='font-size:11px;color:#94A3B8;margin-top:4px;line-height:1.4;'>
                {esc(dify_reason[:150])}
              </div>
            </div>\
            """)

        # 重复提交警告
        if is_already_reviewed:
            prev_decisions = log_df[
                (log_df["record_id_left"] == str(current_row["record_id_left"]))
                & (log_df["record_id_right"] == str(current_row["record_id_right"]))
            ]
            st.warning(
                f"⚠️ 此候选对已审核过！\n\n"
                f"最近决策：**{prev_decisions.iloc[-1].get('decision', 'N/A')}**\n\n"
                f"审核时间：{prev_decisions.iloc[-1].get('updated_at', 'N/A')}"
            )

    with compare_col:
        left_html = render_compare_card(entity, current_row, "left")
        right_html = render_compare_card(entity, current_row, "right")
        st.markdown(
            f"<div style='display:grid;grid-template-columns:1fr 1fr;gap:14px;'>{left_html}{right_html}</div>",
            unsafe_allow_html=True,
        )

    # ── 决策表单（key 绑定 current_index，导航时自动重置） ──
    st.markdown("<hr>", unsafe_allow_html=True)

    with st.form(key=f"review_form_{entity}_{current_index}"):
        form_col1, form_col2, form_col3 = st.columns([2, 1, 1])

        with form_col1:
            st.markdown(
                "<div style='color:#64748B;font-size:12px;margin-bottom:8px;'>审核决策</div>",
                unsafe_allow_html=True,
            )
            decision = st.radio(
                "审核决策",
                ["合并", "不合并", "保留待定"],
                index=0,
                horizontal=True,
                label_visibility="collapsed",
                key=f"decision_{entity}_{current_index}",
            )
        with form_col2:
            st.markdown(
                "<div style='color:#64748B;font-size:12px;margin-bottom:8px;'>备注（可选）</div>",
                unsafe_allow_html=True,
            )
            comment = st.text_area(
                "备注",
                placeholder="填写审核备注…",
                label_visibility="collapsed",
                key=f"comment_{entity}_{current_index}",
            )
        with form_col3:
            st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)
            submitted = st.form_submit_button("✅ 提交决策", use_container_width=True)

        if submitted:
            append_review_log(
                entity,
                {
                    "record_id_left": current_row["record_id_left"],
                    "source_left": current_row["source_left"],
                    "record_id_right": current_row["record_id_right"],
                    "source_right": current_row["source_right"],
                    "match_score": current_row["match_score"],
                    "decision": decision,
                    "comment": comment,
                    "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                },
            )
            st.toast(f"✅ 决策「{decision}」已记录", icon="✅")
            if current_index < max_index:
                st.session_state[session_key] += 1
            st.rerun()

    # ── 操作按钮行：跳过 / 撤销 / 下一未审核 ──
    op_col1, op_col2, op_col3 = st.columns([1, 1, 1])
    with op_col1:
        if st.button("⏭ 跳过此条", key=f"skip_{entity}_{current_index}", use_container_width=True):
            if current_index < max_index:
                st.session_state[session_key] += 1
                st.rerun()
            else:
                st.info("已是最后一条")
    with op_col2:
        if st.button("↩ 撤销上一条", key=f"undo_{entity}", use_container_width=True):
            if remove_last_log_entry(entity):
                st.toast("已撤销上一条审核决策", icon="↩")
                st.rerun()
            else:
                st.info("没有可撤销的记录")
    with op_col3:
        if st.button("⏩ 下一未审核", key=f"next_unreviewed_{entity}_{current_index}", use_container_width=True):
            found = False
            for i in range(current_index + 1, len(queue_df)):
                r = queue_df.iloc[i]
                if (str(r["record_id_left"]), str(r["record_id_right"])) not in reviewed_pairs:
                    st.session_state[session_key] = i
                    found = True
                    break
            if found:
                st.rerun()
            else:
                st.info("后面没有未审核记录了")


# ═══════════════════════════════════════════════════════════════════════════════
# 页面：审核历史 + 导出
# ═══════════════════════════════════════════════════════════════════════════════

def render_history(entity: str) -> None:
    """渲染审核历史 Tab"""
    st.markdown(
        """<div class='section-header'>
          <div class='section-header-icon emerald'>📋</div>
          <div>
            <div class='section-header-title'>审核历史与导出</div>
            <div class='section-header-sub'>筛选 · 分页 · CSV 导出</div>
          </div>
        </div>""",
        unsafe_allow_html=True,
    )

    log_df = load_review_log(entity)

    if log_df.empty:
        st.info("暂无审核记录。在「详情审核」中完成审核后，记录将显示在此处。")
        return

    # ── 筛选器 ──
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        decision_filter = st.multiselect(
            "决策筛选",
            ["合并", "不合并", "保留待定"],
            default=[],
            key=f"hist_decision_{entity}",
        )
    with c2:
        score_range = st.slider(
            "分数范围",
            0.0, 1.0, (0.0, 1.0), 0.01,
            key=f"hist_score_{entity}",
        )
    with c3:
        search_term = st.text_input(
            "搜索 ID / 备注",
            placeholder="输入 Record ID 或备注关键词…",
            key=f"hist_search_{entity}",
        )

    filtered_log = log_df.copy()
    if decision_filter:
        filtered_log = filtered_log[filtered_log["decision"].isin(decision_filter)]
    filtered_log = filtered_log[
        pd.to_numeric(filtered_log["match_score"], errors="coerce").between(*score_range)
    ]
    if search_term:
        mask = (
            filtered_log["record_id_left"].str.contains(search_term, case=False, na=False)
            | filtered_log["record_id_right"].str.contains(search_term, case=False, na=False)
            | filtered_log["comment"].str.contains(search_term, case=False, na=False)
        )
        filtered_log = filtered_log[mask]

    st.markdown(
        f"<div style='color:#64748B;font-size:12px;margin-bottom:8px;'>"
        f"共 {len(filtered_log)} 条记录（总计 {len(log_df)} 条）"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── 决策分布小统计 ──
    if not filtered_log.empty:
        stats_cols = st.columns(4)
        for i, dec in enumerate(["合并", "不合并", "保留待定"]):
            cnt = len(filtered_log[filtered_log["decision"] == dec])
            with stats_cols[i]:
                st.metric(
                    DECISION_LABEL_MAP.get(dec, dec),
                    cnt,
                    delta=f"{cnt/len(filtered_log)*100:.0f}%" if len(filtered_log) > 0 else None,
                )
        with stats_cols[3]:
            st.metric("📊 平均分数", f"{pd.to_numeric(filtered_log['match_score'], errors='coerce').mean():.4f}")

    # ── 日志表格 ──
    render_log_table(filtered_log, f"hist_table_{entity}")

    # ── 导出 ──
    st.markdown("<div class='card card-sm'>", unsafe_allow_html=True)
    export_col1, export_col2 = st.columns([3, 1])
    with export_col1:
        st.markdown(
            "<span style='color:#1E293B;font-weight:600;'>📥 导出审核日志</span>"
            "<span style='color:#64748B;font-size:12px;margin-left:8px;'>下载 CSV 文件</span>",
            unsafe_allow_html=True,
        )
    with export_col2:
        csv = filtered_log.to_csv(index=False, encoding="utf-8-sig")
        st.download_button(
            "⬇ 下载 CSV",
            csv,
            f"{entity}_review_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv",
            key=f"download_log_{entity}",
            use_container_width=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 页面：黄金记录
# ═══════════════════════════════════════════════════════════════════════════════

def render_golden_records(entity: str) -> None:
    """渲染黄金记录 Tab"""
    st.markdown(
        """<div class='section-header'>
          <div class='section-header-icon amber'>🏆</div>
          <div>
            <div class='section-header-title'>黄金主数据记录</div>
            <div class='section-header-sub'>最终实体 · 去重指标 · 数据导出</div>
          </div>
        </div>""",
        unsafe_allow_html=True,
    )

    golden_df = load_golden(entity)
    final_metrics = load_json(FINAL_METRICS_FILES[entity])
    metrics = load_json(METRICS_FILES[entity])

    if golden_df.empty:
        st.warning("未找到黄金记录文件。请先运行 finalize 脚本。")
        return

    # ── 指标卡片 ──
    source_total = sum(
        [1600, 1400, 1200] if entity == "customer" else [1000, 1000]
    )

    _md(f"""\
    <div class='grid-3'>
      <div class='card card-sm card-emerald' style='margin-top:22px;'>
        <div class='card-icon-circle emerald'>🏆</div>
        <div class='metric-val' style='color:var(--quaternary);'>{fmt_num(final_metrics.get("final_entity_count", len(golden_df)))}</div>
        <div class='metric-lbl'>黄金实体数</div>
        <div class='metric-sub'>从 {fmt_num(source_total)} 条源记录合并</div>
      </div>
      <div class='card card-sm card-amber' style='margin-top:22px;'>
        <div class='card-icon-circle amber'>🔗</div>
        <div class='metric-val'>{fmt_num(final_metrics.get("total_merge_pairs", 0))}</div>
        <div class='metric-lbl'>总合并对</div>
        <div class='metric-sub'>自动 {fmt_num(final_metrics.get("auto_merge_pairs", 0))} · 人工 {fmt_num(final_metrics.get("manual_merge_pairs", 0))}</div>
      </div>
      <div class='card card-sm card-accent' style='margin-top:22px;'>
        <div class='card-icon-circle violet'>📉</div>
        <div class='metric-val' style='color:var(--accent);'>
          {((source_total - final_metrics.get("final_entity_count", source_total)) / source_total * 100):.1f}%
        </div>
        <div class='metric-lbl'>去重率</div>
        <div class='metric-sub'>消除 {fmt_num(source_total - final_metrics.get("final_entity_count", 0))} 条重复</div>
      </div>
    </div>\
    """)

    # ── 精确率 / 召回率（基于管道指标估算） ──
    auto_merge_true = metrics.get("auto_merge_true_ratio", 0)
    review_true = metrics.get("review_true_ratio", 0)
    auto_merge_count = metrics.get("auto_merge_count", 0)
    review_count = metrics.get("review_count", 0)
    total_true_pairs = metrics.get("customer_duplicate_pairs" if entity == "customer" else "product_duplicate_pairs", 0)

    # 估计：自动合并中真匹配 + 审核队列中真匹配
    estimated_tp = auto_merge_count * auto_merge_true + review_count * review_true
    estimated_precision = (estimated_tp / (auto_merge_count + review_count) * 100) if (auto_merge_count + review_count) > 0 else 0
    estimated_recall = (estimated_tp / total_true_pairs * 100) if total_true_pairs > 0 else 0

    _md(f"""\
    <div class='grid-2'>
      <div class='card card-sm' style='margin-top:22px;'>
        <div class='card-icon-circle violet'>🎯</div>
        <div class='metric-lbl'>估计匹配准确率 (Precision)</div>
        <div class='metric-val sm' style='color:var(--accent);'>{estimated_precision:.1f}%</div>
        <div class='metric-sub'>高置信自动合并准确率: {auto_merge_true*100:.1f}%</div>
      </div>
      <div class='card card-sm' style='margin-top:22px;'>
        <div class='card-icon-circle amber'>🔍</div>
        <div class='metric-lbl'>估计匹配召回率 (Recall)</div>
        <div class='metric-val sm' style='color:var(--tertiary);'>{estimated_recall:.1f}%</div>
        <div class='metric-sub'>总真匹配对: {total_true_pairs}</div>
      </div>
    </div>\
    """)

    # ── 黄金记录预览表格 ──
    with st.expander(f"📋 黄金记录预览（共 {len(golden_df)} 条）", expanded=False):
        st.dataframe(
            golden_df.head(100),
            use_container_width=True,
            hide_index=True,
            column_config={
                col: st.column_config.TextColumn(col, width="small")
                for col in golden_df.columns[:15]
            },
        )

    # ── 下载按钮 ──
    st.markdown("<div class='card card-sm'>", unsafe_allow_html=True)
    dl_col1, dl_col2, dl_col3 = st.columns([2, 1, 1])
    with dl_col1:
        st.markdown(
            "<span style='color:#1E293B;font-weight:600;'>📥 数据导出</span>",
            unsafe_allow_html=True,
        )
    with dl_col2:
        st.download_button(
            "⬇ 黄金记录 CSV",
            golden_df.to_csv(index=False, encoding="utf-8-sig"),
            f"golden_{entity}s_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv",
            key=f"download_golden_{entity}",
            use_container_width=True,
        )
    with dl_col3:
        st.download_button(
            "⬇ 指标 JSON",
            json.dumps(final_metrics, ensure_ascii=False, indent=2),
            f"{entity}_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "application/json",
            key=f"download_metrics_{entity}",
            use_container_width=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # ── 页面配置 ──
    st.set_page_config(
        page_title="MDM 主数据审核系统",
        page_icon="🧩",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # ── 自定义 CSS ──
    st.markdown(CSS, unsafe_allow_html=True)

    # ── 装饰性 Confetti 背景 ──
    st.markdown(
        """<div class='confetti-container'>
          <div class='confetti-piece'></div>
          <div class='confetti-piece'></div>
          <div class='confetti-piece'></div>
          <div class='confetti-piece'></div>
          <div class='confetti-piece'></div>
          <div class='confetti-piece'></div>
          <div class='confetti-piece'></div>
          <div class='confetti-piece'></div>
        </div>""",
        unsafe_allow_html=True,
    )

    # ── 侧边栏 ──
    with st.sidebar:
        st.markdown(
            "<div style='font-size:20px;font-weight:800;color:#1E293B;margin-bottom:2px;"
            "font-family:Outfit,system-ui,sans-serif;display:flex;align-items:center;gap:8px;'>"
            "<span style='display:inline-flex;align-items:center;justify-content:center;"
            "width:32px;height:32px;border-radius:10px;background:var(--accent);"
            "border:2px solid var(--border-dark);font-size:16px;'>🧩</span>"
            "MDM 系统</div>"
            "<div style='font-size:11px;color:#64748B;margin-bottom:14px;padding-left:40px;'>"
            "主数据管理智能清洗</div>"
            "<div class='squiggle-divider'></div>",
            unsafe_allow_html=True,
        )

        entity = st.selectbox(
            "选择实体类型",
            ["customer", "product"],
            format_func=lambda x: "👤 客户主数据" if x == "customer" else "📦 商品主数据",
            key="entity_select",
        )

        st.markdown("<div class='squiggle-divider'></div>", unsafe_allow_html=True)

        # ── 竖直导航菜单 ──
        TAB_LABELS = ["📊 仪表板", "🔍 审核队列", "📝 详情审核", "📋 审核历史", "🏆 黄金记录"]
        if "active_tab" not in st.session_state:
            st.session_state["active_tab"] = 0
        st.session_state["active_tab"] = max(0, min(st.session_state["active_tab"], len(TAB_LABELS) - 1))

        active_idx = st.radio(
            "导航",
            TAB_LABELS,
            index=st.session_state["active_tab"],
            label_visibility="collapsed",
            key="tab_selector",
        )
        st.session_state["active_tab"] = TAB_LABELS.index(active_idx)

        st.markdown("<div class='squiggle-divider'></div>", unsafe_allow_html=True)

        # 快速统计 — 卡片式布局
        queue_df = load_review_queue(entity)
        reviewed_pairs = get_reviewed_pairs(entity)
        total_queue = len(queue_df)
        total_reviewed = len(reviewed_pairs)
        remaining = max(0, total_queue - total_reviewed)
        progress = total_reviewed / max(total_queue, 1)

        # 加载审核日志获取决策统计
        log_df = load_review_log(entity)
        merge_count = len(log_df[log_df["decision"] == "合并"]) if not log_df.empty else 0
        not_merge_count = len(log_df[log_df["decision"] == "不合并"]) if not log_df.empty else 0
        pending_count = len(log_df[log_df["decision"] == "保留待定"]) if not log_df.empty else 0
        total_log = len(log_df)
        merge_rate = (merge_count / total_log * 100) if total_log > 0 else 0

        _md(f"""\
        <div class='sidebar-card sidebar-card-accent'>
          <div class='sidebar-icon violet'>📥</div>
          <div class='sidebar-metric-val'>{fmt_num(total_queue)}</div>
          <div class='sidebar-metric-lbl'>待审核候选对</div>
          <div class='sidebar-metric-sub'>剩余 {remaining} 条未审核</div>
        </div>
        <div class='sidebar-card sidebar-card-emerald'>
          <div class='sidebar-icon emerald'>✅</div>
          <div class='sidebar-metric-val'>{fmt_num(total_reviewed)}</div>
          <div class='sidebar-metric-lbl'>已审核记录</div>
          <div class='sidebar-metric-sub'>进度 {progress*100:.1f}%</div>
        </div>
        <div class='sidebar-card sidebar-card-emerald'>
          <div class='sidebar-icon emerald'>🔗</div>
          <div class='sidebar-metric-val' style='color:var(--quaternary);'>{merge_count}</div>
          <div class='sidebar-metric-lbl'>合并决策</div>
          <div class='sidebar-metric-sub'>合并率 {merge_rate:.1f}%</div>
        </div>
        <div class='sidebar-card sidebar-card-pink'>
          <div class='sidebar-icon pink'>❌</div>
          <div class='sidebar-metric-val' style='color:var(--secondary);'>{not_merge_count}</div>
          <div class='sidebar-metric-lbl'>不合并决策</div>
          <div class='sidebar-metric-sub'>待定 {pending_count} 条</div>
        </div>\
        """)

        st.markdown("<div class='squiggle-divider'></div>", unsafe_allow_html=True)
        st.markdown(
            "<div style='font-size:11px;color:#94A3B8;'>"
            "<span class='dot-accent pink'></span>v2.0 · MDM 智能审核系统<br>"
            "基于规则+Embedding的多策略匹配"
            "</div>",
            unsafe_allow_html=True,
        )

    # ── 页头 ──
    _md(f"""\
    <div class='card card-blob bg-stripes' style='display:flex;justify-content:space-between;align-items:center;position:relative;overflow:hidden;'>
      <div class='deco-circle' style='width:60px;height:60px;top:-20px;right:80px;background:rgba(244,114,182,0.12);'></div>
      <div class='deco-circle' style='width:40px;height:40px;bottom:-10px;right:160px;background:rgba(251,191,36,0.15);'></div>
      <div style='position:relative;z-index:1;'>
        <div class='page-title'>🧩 主数据匹配审核界面</div>
        <div class='page-subtitle'>
          当前实体：{"👤 客户主数据" if entity == "customer" else "📦 商品主数据"}
          · 审核队列 {len(queue_df)} 条 · 已审核 {len(reviewed_pairs)} 条
        </div>
        <div class='hero-stats'>
          <div class='hero-stat'>
            <div class='hero-stat-val' style='color:var(--accent);'>{fmt_num(len(queue_df))}</div>
            <div class='hero-stat-lbl'>待审核</div>
          </div>
          <div class='hero-stat'>
            <div class='hero-stat-val' style='color:var(--quaternary);'>{fmt_num(len(reviewed_pairs))}</div>
            <div class='hero-stat-lbl'>已审核</div>
          </div>
          <div class='hero-stat'>
            <div class='hero-stat-val' style='color:var(--tertiary);'>{len(load_review_log(entity))/max(len(queue_df),1)*100:.0f}%</div>
            <div class='hero-stat-lbl'>完成率</div>
          </div>
        </div>
      </div>
      <div style='text-align:right;position:relative;z-index:1;'>
        <span class='entity-tag {"customer" if entity == "customer" else "product"}'>
          {"CRM + ERP + 电商" if entity == "customer" else "ERP + 电商"}
        </span>
        <div style='margin-top:8px;font-size:11px;color:var(--muted-fg);'>v2.0 · MDM 智能审核系统</div>
      </div>
    </div>
    <div class='squiggle-divider'></div>\
    """)

    if active_idx == "📊 仪表板":
        render_dashboard(entity)
    elif active_idx == "🔍 审核队列":
        render_review_queue(entity)
    elif active_idx == "📝 详情审核":
        render_detail_review(entity)
    elif active_idx == "📋 审核历史":
        render_history(entity)
    else:
        render_golden_records(entity)


if __name__ == "__main__":
    main()

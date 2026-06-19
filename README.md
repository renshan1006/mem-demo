# MDM 主数据管理智能清洗系统

基于 **Dify + Embedding + 规则引擎** 的跨源客户/商品主数据自动匹配系统。

## 项目简介

企业不同系统（CRM、ERP、电商）存储的客户与商品数据存在大量重复与不一致，例如"北京阿尔法科技有限公司"与"北京 Alpha 科技公司"实为同一实体。本系统实现：

- **跨源数据接入**：CRM / ERP / ECommerce 三源客户 + 双源商品
- **多策略实体匹配**：阻塞规则 + bge-large-zh 语义向量 + 规则加权打分
- **智能审核流转**：高置信度自动合并，中等置信度人工审核 + Dify LLM 辅助判断
- **黄金记录生成**：UnionFind 聚类合并，自动选择最完整记录
- **阈值反馈回训**：基于审核日志 + 真值标签自动优化匹配阈值

## 快速开始（从小白到跑通，跟着做就行）

### 你电脑上需要有的东西

| 需要 | 怎么检查 | 没有的话去哪装 |
|------|---------|-------------|
| Python 3.9 或更新 | 终端输入 `python --version` | https://python.org 下载 |
| 网络（首次运行） | 能打开网页就行 | — |
| 约 3 GB 磁盘空间 | 模型 1.3GB + 数据文件 ~100MB | — |
| Dify 账号（可选） | 见下文 | https://cloud.dify.ai 免费注册 |

### 第一步：下载项目 & 安装依赖

```bash
git clone <repo-url>
cd mdm-demo
pip install -r requirements.txt
```

这一步会安装 pandas、numpy、streamlit、sentence-transformers 等 Python 包。如果报错，检查 pip 是否最新：`pip install --upgrade pip`

### 第二步：生成数据集

```bash
cd scripts
python generate_dataset.py
```

运行后 `data/` 目录下会多出 5 个 CSV 文件（共 6,200 条模拟数据）。这些数据都是代码生成的假数据，不涉及真实隐私。

### 第三步：运行匹配管道（关键步骤 ⚠️）

```bash
python mdm_pipeline.py
python product_pipeline.py
python finalize.py customer
python finalize.py product
```

**首次运行时**，`mdm_pipeline.py` 会自动从网上下载 AI 模型（`bge-large-zh-v1.5`，约 1.3 GB），下载需要几分钟。下载完后会缓存到 `.model_cache/` 目录，以后就不用再下了。

**模型有两种运行模式：**

| 模式 | 触发条件 | 效果 | 审核队列大小 |
|------|---------|------|:----------:|
| 🟢 **Embedding 模式** | 模型下载成功 | AI 语义理解名称，能识别"Alpha" = "阿尔法" | 约 240 条 |
| 🟡 **回退模式** | 模型下载失败 / 网络不通 | 用编辑距离做名称比对，准确率稍低 | 约 2,000 条 |

> 💡 **怎么知道我在哪种模式下？** 看终端输出：如果看到 `模型加载完成` 就是 Embedding 模式；如果看到 `模型加载失败，回退到编辑距离匹配` 就是回退模式。两种都能跑，只是效果有差异。网络恢复后重新运行 `python mdm_pipeline.py` 就会自动重新下载模型切换到 Embedding 模式。

### 第四步：启动审核界面

```bash
cd ..   # 回到项目根目录
python run_streamlit.py run streamlit_app.py
```

浏览器打开 **http://localhost:8501**，你会看到一个 5 个标签页的审核系统。

### 3. 启动审核界面

```bash
# 回到项目根目录
cd ..
python run_streamlit.py run streamlit_app.py
```

浏览器打开 http://localhost:8501

### 4.（可选）阈值优化

```bash
cd scripts

# 基于真值标签 + 审核日志自动优化阈值
python tune_thresholds.py

# 用新阈值重新运行管道
python mdm_pipeline.py
python product_pipeline.py
python finalize.py customer
python finalize.py product
```

### 5.（可选）Dify 网页版 LLM 增强

Dify 是一个可视化的 LLM 工作流平台，本系统用它来做**匹配候选对的二次判断**——规则打分放入审核队列的模糊对，交给 LLM 做更精准的语义推理。

#### 5.1 注册 & 导入工作流

1. 打开 https://cloud.dify.ai → 注册/登录
2. 右上角头像 →「设置」→「模型供应商」→ 添加 **DeepSeek**
   - 去 https://platform.deepseek.com 注册，创建 API Key（新用户送 500 万 tokens）
   - 把 DeepSeek API Key 粘贴到 Dify 里
3. 回到 Dify 首页 →「创建应用」→ 选择「**工作流**」→「导入 DSL 文件」
4. 上传 `dify/mdm_workflow.yml`

导入后你会看到 4 个节点连成一条线：

```
[开始] → [LLM 语义匹配] → [解析 JSON] → [结束]
```

#### 5.2 配置节点

**LLM 节点**（双击打开）：
- 模型下拉框选 `deepseek-chat`
- System Prompt 和 User Prompt 已预填好，无需修改
- 检查完毕后点右上角关闭

**Code 节点**：确认输入变量 `llm_output` 的来源是 `LLM 语义匹配 → text`

#### 5.3 测试工作流

点击右上角「**运行**」→ 填入测试数据：

| 变量 | 值 |
|------|-----|
| entity_type | customer |
| source_a | CRM |
| source_b | ERP |
| company_name_a | 北京阿尔法科技有限公司 |
| company_name_b | 北京Alpha科技公司 |
| region_a | 北京市 |
| region_b | 北京市 |

点击运行，应返回：

```json
{ "decision": "merge", "confidence": 0.95, "reasoning": "税号一致,名称语义高度相似..." }
```

#### 5.4 发布 & 接入项目

1. 测试通过后 → 右上角「**发布**」
2. 左侧「API 访问」→ 复制 API Key（格式 `app-xxxxx`）
3. 编辑 `dify/config.json`：

```json
{ "api_key": "app-你的Key", "base_url": "https://api.dify.ai" }
```

#### 5.5 运行增强

```bash
cd scripts

# 测试连接
python dify_client.py

# 对审核队列做 LLM 增强（20 条采样）
python dify_enrich.py customer 20
python dify_enrich.py product
```

增强结果写入 `data/customer_review_enriched.csv`，包含 Dify 的决策、置信度和推理过程，可与规则评分对比。

> 详细图文教程：[dify/README.md](dify/README.md)

### 6.数据迁移到数据库

```bash
# SQLite
python migrate_to_db.py

# PostgreSQL（需要先配置 sql/pg_config.json）
python migrate_to_db.py --pg
```

## 管道流程

```
数据生成                    匹配管道                    审核 & 合并
───────                    ────────                    ──────────
generate_dataset.py  →     mdm_pipeline.py        →   streamlit_app.py
(6,200 条, 5 个源)          product_pipeline.py       (人工审核)

                              │                          │
                              ├─ 文本规范化               ├─ 决策写入 review_log
                              ├─ Embedding 预计算          │
                              ├─ 阻塞分桶 (region/category) │
                              ├─ 多策略打分                ▼
                              └─ 决策分类              finalize.py
                                                       (UnionFind 聚类)
                              tune_thresholds.py  ←── review_log
                              (阈值优化反馈)         黄金记录输出
```

## 匹配算法

### 客户匹配

```
总分 = Embedding名称 × 0.45 + 税号匹配 × 0.30 + 电话匹配 × 0.15 + 地址 × 0.10
     + 税号加分(0.10) + 电话加分(0.05)
```

### 商品匹配

```
总分 = Embedding名称×0.20 + Embedding品牌×0.15 + Embedding型号×0.10
     + SKU匹配×0.25 + UPC匹配×0.15 + 规格×0.05
     - SKU/UPC 冲突惩罚（上限 0.55）
```

### 决策阈值

| 区间 | 客户阈值 | 商品阈值 | 动作 |
|------|:------:|:------:|------|
| auto_merge | ≥ 0.53 | ≥ 0.54 | 自动合并 |
| review | 0.50–0.53 | 0.51–0.54 | 人工审核 |
| no_match | < 0.50 | < 0.51 | 判定不同 |

## 项目结构

```
mdm-demo/
├── run_streamlit.py              # Streamlit 启动器
├── streamlit_app.py              # 审核界面（5-Tab）
├── requirements.txt
├── .gitignore
│
├── scripts/
│   ├── generate_dataset.py       # 数据生成
│   ├── embedder.py               # bge-large-zh 语义向量模块
│   ├── mdm_pipeline.py           # 客户匹配管道
│   ├── product_pipeline.py       # 商品匹配管道
│   ├── finalize.py               # 黄金记录生成
│   ├── tune_thresholds.py        # 阈值优化回训
│   ├── dify_client.py            # Dify API 客户端
│   ├── dify_enrich.py            # Dify 审核增强
│   ├── db_manager.py             # 数据库管理（SQLite/PostgreSQL）
│   ├── cache_manager.py          # 缓存管理（Redis/本地）
│   └── migrate_to_db.py          # CSV → 数据库迁移
│
├── data/
│   ├── crm_customers.csv         # 源数据：CRM 客户 (1600)
│   ├── erp_customers.csv         # 源数据：ERP 客户 (1400)
│   ├── ecommerce_customers.csv   # 源数据：电商客户 (1200)
│   ├── erp_products.csv          # 源数据：ERP 商品 (1000)
│   ├── ecommerce_products.csv    # 源数据：电商商品 (1000)
│   ├── 数据集说明.md               # 数据集说明
│   ├── dataset_summary.json      # 数据统计
│   ├── optimized_thresholds.json # 优化后的阈值
│   ├── final_golden_customers.csv    # 最终黄金客户
│   ├── final_golden_products.csv     # 最终黄金商品
│   ├── final_customer_metrics.json   # 客户匹配指标
│   ├── final_product_metrics.json    # 商品匹配指标
│   ├── customer_review_log.csv       # 客户审核日志
│   └── product_review_log.csv        # 商品审核日志
│
├── dify/
│   ├── mdm_workflow.yml          # Dify 工作流 DSL
│   ├── config.example.json       # 配置模板
│   ├── README.md                 # Dify 部署指南
│   └── prompts/
│       ├── customer_match_prompt.md
│       └── product_match_prompt.md
│
├── sql/
│   ├── schema.sql                # PostgreSQL 建表脚本
│   ├── pg_config.json            # PG 连接配置
│   └── redis_config.json         # Redis 连接配置
│
└── docs/
    ├── 智能体设计文档.md            # 工作流编排、Prompt 设计、架构
    ├── 调试报告.md                 # 开发过程中的问题与解决方案
    └── 答辩PPT大纲.md              # 答辩演示文稿大纲
```

## 技术栈

| 层 | 技术 |
|----|------|
| 数据生成 | Python + Faker + NumPy |
| 语义匹配 | bge-large-zh-v1.5 (Sentence-Transformers) |
| 规则引擎 | pandas + 自定义加权评分 |
| 智能体平台 | Dify Cloud（LLM 工作流） |
| 审核界面 | Streamlit（5-Tab 暗色主题） |
| 数据库 | SQLite（开发）/ PostgreSQL（生产） |
| 缓存 | 本地文件（开发）/ Redis（生产） |
| 模型 | DeepSeek-Chat / GPT-4o-mini |

## 评估指标

| 指标 | V1 (difflib) | V2 (+Embedding) | V3 (+回训) |
|------|:----------:|:------------:|:---------:|
| 客户 auto_merge 精确率 | 100% | 100% | 100% |
| 客户 auto_merge 召回率 | 77% | 81% | 98.2% |
| 客户审核队列大小 | 2,156 | 386 | **241** |
| 商品审核队列大小 | 1,317 | 80 | **31** |
| 总人工审核量 | 3,473 | 466 | **272** |
| 人工审核减少 | — | 87% | **92%** |

## 常见问题（小白必看）

**Q: 运行 `mdm_pipeline.py` 很慢 / 卡住了？**
A: 正常现象。第一次运行需要下载 AI 模型（约 1.3 GB），需要几分钟。进度条会显示 `Batches: 30%|███ | 10/33` 之类的。下载完就行了，以后不用再下。

**Q: 显示"模型加载失败，回退到编辑距离匹配"？**
A: 网络连不上模型下载地址。不影响运行，但匹配效果会打折扣（审核队列会多 8 倍）。网络恢复后重新跑一次 `python mdm_pipeline.py` 就好。

**Q: 我不想等模型下载，能用吗？**
A: 能。系统会自动用"回退模式"跑，所有功能都正常，只是审核队列里有更多待审核记录（约 2000 条而不是 240 条）。

**Q: 怎么确认模型下载成功？**
A: 看终端输出——出现 `模型加载完成` 并且看到 `Embedding 完成，耗时 XX 秒` 就是成功了。同时 `data/` 下会生成 `customer_review_queue.csv` 大约 240 行（而不是 2000 行）。

**Q: Streamlit 启动报 SSL 错误？**
A: 不要直接运行 `streamlit run`，用项目自带的 `python run_streamlit.py run streamlit_app.py` 启动，已内置修复。

**Q: 审核界面里没有数据？**
A: 先确保运行了第三步的 4 个命令（`mdm_pipeline.py`、`product_pipeline.py`、`finalize.py customer`、`finalize.py product`）。如果都跑了还没有，检查 `data/` 下是否有 `customer_review_queue.csv`。

**Q: 想用 PostgreSQL 代替 SQLite？**
A: 先装好 PostgreSQL → 配好 `sql/pg_config.json` → 运行 `python migrate_to_db.py --pg`。不配的话默认用 SQLite，什么都不用装。

**Q: Dify API 返回 404？**
A: 去 Dify 后台检查工作流是否点了「发布」。确认 `config.json` 里 `base_url` 写的是 `https://api.dify.ai`（不是 `cloud.dify.ai`）。

**Q: 为什么我看不到 Dify LLM 建议？**
A: 因为还没有运行 Dify 增强。运行 `python dify_enrich.py customer 20` 给前 20 条记录加上 LLM 建议，然后刷新 Streamlit 就能看到。

## License

MIT

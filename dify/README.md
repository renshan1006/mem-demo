# Dify 智能体部署指南

## 1. 前置准备：添加模型供应商

Dify 右上角头像 →「设置」→「模型供应商」→ 添加一个模型：

**推荐 DeepSeek**（便宜、中文好、注册送 500 万 tokens）：
1. 打开 https://platform.deepseek.com → 注册 → 创建 API Key
2. 在 Dify 设置 → 模型供应商 → DeepSeek → 粘贴 API Key

**或用 OpenAI**：
1. https://platform.openai.com → 创建 API Key
2. 在 Dify 设置 → 模型供应商 → OpenAI → 粘贴

## 2. 导入工作流

本项目提供**两个独立的 Dify 工作流**（一个 Dify app = 一个工作流 = 一个 api_key，需分别导入与发布）：

| 工作流 DSL | 用途 | 配置键 | 模块 |
|-----------|------|--------|------|
| `dify/mdm_workflow.yml` | 实体匹配（客户/商品是否同一实体） | `api_key` | M1 |
| `dify/quality_workflow.yml` | 数据质量评估（生成中文整改建议） | `quality_api_key` | M2 |

**导入步骤（每个工作流都做一遍）：**

1. Dify 控制台 →「创建应用」→「导入 DSL 文件」
2. 选择对应的 `dify/*.yml` → 导入

## 3. 导入后配置

导入成功后会自动打开工作流画布，你会看到 4 个节点：

```
[开始] → [LLM 语义匹配] → [解析 JSON] → [结束]
```

### 3.1 配置 LLM 节点

1. 双击「LLM 语义匹配」节点
2. 在「模型」下拉框中选择 `deepseek-chat`（或你添加的模型）
3. 检查 System Prompt 和 User Prompt 内容是否完整
4. 点击节点外部保存

### 3.2 检查 Code 节点

1. 双击「解析 JSON」节点
2. 确认「输入变量」里 `llm_output` 的值是 `{{#LLM 语义匹配.text#}}`
3. （如果提示变量来源缺失，手动选择：LLM 语义匹配 → text）
4. 确认 4 个输出变量都在：`decision`, `confidence`, `reasoning`, `signals`

### 3.3 检查 End 节点

1. 双击「结束」节点
2. 确认 4 个输出都关联到了 Code 节点的对应变量

## 4. 测试工作流

1. 点击右上角「运行」按钮
2. 填入测试数据：

```
entity_type: customer
source_a: CRM
source_b: ERP
company_name_a: 北京阿尔法科技有限公司
company_name_b: 北京Alpha科技公司
region_a: 北京市
region_b: 北京市
city_a: 北京市
city_b: 北京市
address_a: 北京市海淀区中关村大街1号
address_b: 北京市海淀区中关村大街1号楼
phone_a: 010-12345678
phone_b: 01012345678
tax_id_a: 91110108MA01XXXXX
tax_id_b: 91110108MA01XXXXX
```

3. 点击运行 → 应返回类似：
```json
{
  "decision": "merge",
  "confidence": 0.92,
  "reasoning": "税号一致且名称语义相似，判定为同一实体",
  "signals": { ... }
}
```

## 5. 发布 & 获取 API Key

1. 测试通过后 → 点击右上角「**发布**」
2. 左侧菜单 →「**API 访问**」
3. 复制 API Key（格式：`app-xxxxxxxxxxxxx`）
4. 填入项目配置文件：

编辑 `dify/config.json`：
```json
{
  "api_key": "app-你的APIKey",
  "base_url": "https://api.dify.ai"
}
```

> ⚠️ `base_url` 必须是 `api.dify.ai`（不是 `cloud.dify.ai`，后者调用会 404）。

### 质量评估工作流（M2，可选）

重复上述步骤导入 `dify/quality_workflow.yml` 并发布，拿到第二个 API Key，填入同一 `dify/config.json`：

```json
{
  "api_key": "app-实体匹配Key",
  "base_url": "https://api.dify.ai",
  "quality_api_key": "app-质量评估Key",
  "quality_base_url": "https://api.dify.ai"
}
```

配置后 `python quality_agent.py --source crm_customers.csv` 的报告会自动填充「LLM 整改建议」章节；未配置则降级为纯统计报告，不报错。

## 6. 测试 Python 客户端

```bash
cd d:/demo/scripts
python dify_client.py
```

输出示例：
```
Dify API 可用: True
{
  "decision": "merge",
  "confidence": 0.92,
  "reasoning": "税号一致判定为同一实体",
  "signals": {
    "tax_id_match": true,
    "phone_match": false,
    ...
  }
}
```

## 7. 增强审核队列（可选）

用 Dify LLM 对审核队列中的模糊对做二次判断：

```bash
cd d:/demo/scripts
python dify_enrich.py customer    # 处理客户审核队列
python dify_enrich.py product 10  # 处理前 10 条商品审核队列
```

输出文件：`data/customer_review_enriched.csv`、`data/product_review_enriched.csv`

## 8. 重新运行匹配管道

Dify 配置好后，运行管道时会自动尝试调用 Dify 做辅助判断：

```bash
cd d:/demo/scripts
python mdm_pipeline.py
python product_pipeline.py
python finalize_customers.py
python finalize_products.py
```

---

## 常见问题

**Q: 导入后 LLM 节点报错「模型不可用」？**
A: 说明模型供应商还没配。去设置 → 模型供应商 → 添加 DeepSeek/OpenAI API Key。

**Q: Code 节点输入变量是空的？**
A: 手动点一下下拉框，选 LLM 语义匹配 → text。

**Q: 运行时 User Prompt 里的 {{#start.xxx#}} 没被替换？**
A: 运行前要在左侧输入面板填好对应的变量值。不想填的字段留空即可。

**Q: API 调用报 401？**
A: config.json 里的 api_key 写错了，去 Dify → API 访问 → 重新复制。

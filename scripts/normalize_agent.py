"""
数据标准化智能体 (M3)
功能：地址标准化、公司名规范化、电话格式统一
"""
import os
from dotenv import load_dotenv
load_dotenv()
import pandas as pd
import re
from typing import Dict, Any, List, Optional
from pathlib import Path

# 导入项目中的基类
from scripts.base_agent import BaseAgent

# 导入 Dify 客户端（用于 LLM 增强）
try:
    from scripts.dify_client import DifyClient
except ImportError:
    DifyClient = None
    print("警告: dify_client 未找到，将只使用 Baseline 模式")


class NormalizeAgent(BaseAgent):
    """数据标准化智能体"""
    
    name = "data_normalize"
    description = "数据标准化智能体 - 地址/公司名/电话格式统一"

    def __init__(self, dify_client: Optional[Any] = None):
        # 如果没有传入客户端，自己创建一个
        if dify_client is None:
            try:
                from scripts.dify_client import DifyClient
                self.dify_client = DifyClient()
            except ImportError:
                print("⚠️ 无法导入 DifyClient，将只使用 Baseline 模式")
                self.dify_client = None
        else:
            self.dify_client = dify_client
        self.data_dir = Path(__file__).parent.parent / "data"

    def run(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """执行标准化任务（必须实现的方法）"""
        action = task.get("action", "normalize_all")
        params = task.get("params", {})
        data = task.get("data")
        
        try:
            # 如果没有传入数据，从 CSV 读取
            if data is None:
                source = params.get("source")
                if not source:
                    return self._error("请指定 source 参数或传入 data")
                file_path = self.data_dir / source
                if not file_path.exists():
                    return self._error(f"文件不存在: {file_path}")
                data = pd.read_csv(file_path)
            
            # 根据动作分发
            if action == "normalize_address":
                result = self._normalize_address(data, params)
            elif action == "normalize_company":
                result = self._normalize_company(data, params)
            elif action == "normalize_phone":
                result = self._normalize_phone(data, params)
            elif action == "normalize_all":
                result = self._normalize_all(data, params)
            else:
                return self._error(f"不支持的动作: {action}")
            
            # 保存日志
            self._save_log(result)
            
            return {
                "success": True,
                "data": result["data"],
                "summary": result["summary"],
                "details": result["details"],
                "error": None,
            }
        except Exception as e:
            return self._error(str(e))

    def _error(self, msg: str) -> Dict[str, Any]:
        return {
            "success": False,
            "data": None,
            "summary": f"失败: {msg}",
            "details": {},
            "error": msg,
        }

    # ==================== 以下是各个标准化功能 ====================

    def _normalize_address(self, df: pd.DataFrame, params: dict) -> dict:
        """地址标准化（Baseline：正则）"""
        field = params.get("field", "address")
        df_copy = df.copy()
        details = {"total": len(df_copy), "normalized": 0}
        
        def clean(text):
            if pd.isna(text) or not text:
                return text
            text = str(text)
            # 去除多余空格
            text = re.sub(r'\s+', '', text)
            # 统一后缀
            text = re.sub(r'号楼', '号', text)
            text = re.sub(r'北京市', '北京', text)
            text = re.sub(r'上海市', '上海', text)
            text = re.sub(r'海淀区', '海淀', text)   
            text = re.sub(r'朝阳区', '朝阳', text)   
            text = re.sub(r'广东省', '广东', text) 
            # ===== 省级行政区划清洗 =====
            text = re.sub(r'北京市?', '北京', text)
            text = re.sub(r'上海市?', '上海', text)
            text = re.sub(r'天津市?', '天津', text)
            text = re.sub(r'重庆市?', '重庆', text)
            text = re.sub(r'河北省', '河北', text)
            text = re.sub(r'山西省', '山西', text)
            text = re.sub(r'辽宁省', '辽宁', text)
            text = re.sub(r'吉林省', '吉林', text)
            text = re.sub(r'黑龙江省', '黑龙江', text)
            text = re.sub(r'江苏省', '江苏', text)
            text = re.sub(r'浙江省', '浙江', text)
            text = re.sub(r'安徽省', '安徽', text)
            text = re.sub(r'福建省', '福建', text)
            text = re.sub(r'江西省', '江西', text)
            text = re.sub(r'山东省', '山东', text)
            text = re.sub(r'河南省', '河南', text)
            text = re.sub(r'湖北省', '湖北', text)
            text = re.sub(r'湖南省', '湖南', text)
            text = re.sub(r'广东省', '广东', text)
            text = re.sub(r'海南省', '海南', text)
            text = re.sub(r'四川省', '四川', text)
            text = re.sub(r'贵州省', '贵州', text)
            text = re.sub(r'云南省', '云南', text)
            text = re.sub(r'陕西省', '陕西', text)
            text = re.sub(r'甘肃省', '甘肃', text)
            text = re.sub(r'青海省', '青海', text)
            text = re.sub(r'台湾省', '台湾', text)
            text = re.sub(r'内蒙古自治区', '内蒙古', text)
            text = re.sub(r'广西壮族自治区', '广西', text)
            text = re.sub(r'西藏自治区', '西藏', text)
            text = re.sub(r'宁夏回族自治区', '宁夏', text)
            text = re.sub(r'新疆维吾尔自治区', '新疆', text)

            # ===== 市级行政区划清洗（示例，可按需扩充） =====
            text = re.sub(r'广州市', '广州', text)
            text = re.sub(r'深圳市', '深圳', text)
            text = re.sub(r'杭州市', '杭州', text)
            text = re.sub(r'成都市', '成都', text)
            text = re.sub(r'武汉市', '武汉', text)
            text = re.sub(r'南京市', '南京', text)
            text = re.sub(r'西安市', '西安', text)

            # ===== 区级/县级清洗 =====
            text = re.sub(r'海淀区', '海淀', text)
            text = re.sub(r'朝阳区', '朝阳', text)
            text = re.sub(r'浦东新区', '浦东', text)
            text = re.sub(r'天河区', '天河', text)
            text = re.sub(r'福田区', '福田', text)

            # ===== 冗余词去除 =====
            text = re.sub(r'号楼', '号', text)
            text = re.sub(r'单元', '', text)
            text = re.sub(r'室$', '', text)
            text = re.sub(r'栋', '号', text)
            text = re.sub(r'大厦', '', text)
            text = re.sub(r'广场', '', text)
            text = re.sub(r'中心', '', text)
            text = re.sub(r'路', '路', text)  # 保留"路"
            text = re.sub(r'街', '街', text)  # 保留"街"
            text = re.sub(r'巷', '巷', text)  # 保留"巷"

            # ===== 去除多余空格和特殊符号 =====
            text = re.sub(r'\s+', '', text)
            text = re.sub(r'[，,、]', '', text)  # 去除中文/英文逗号、顿号
            return text
        
        df_copy[f"{field}_normalized"] = df_copy[field].apply(clean)
        details["normalized"] = (df_copy[field] != df_copy[f"{field}_normalized"]).sum()
        
        return {
            "data": df_copy,
            "summary": f"地址标准化完成，共 {details['total']} 条，修改 {details['normalized']} 条",
            "details": details,
        }

    def _normalize_company(self, df: pd.DataFrame, params: dict) -> dict:
        """公司名规范化（Baseline：正则 + LLM 增强）"""
        field = params.get("field", "company_name")
        use_llm = params.get("use_llm", True)  # 默认开启 LLM
        
        df_copy = df.copy()
        details = {"total": len(df_copy), "normalized": 0, "llm_used": 0}
        
        def clean(text):
            if pd.isna(text) or not text:
                return text
            text = str(text)
            original = text  # 保存原始值，供 LLM 使用
            
            # ===== 第一步：Baseline 正则清洗 =====
            # 去除公司后缀
            text = re.sub(r'有限公司$', '', text)
            text = re.sub(r'有限责任公司$', '', text)
            text = re.sub(r'股份有限公司$', '', text)
            text = re.sub(r'集团公司$', '', text)
            text = re.sub(r'集团$', '', text)
            text = re.sub(r'工厂$', '', text)
            text = re.sub(r'厂$', '', text)
            text = re.sub(r'公司$', '', text)

            # 去除英文后缀（为 LLM 翻译做准备）
            text = re.sub(r'Co\.?$', '', text, flags=re.IGNORECASE)
            text = re.sub(r'Ltd\.?$', '', text, flags=re.IGNORECASE)
            text = re.sub(r'Inc\.?$', '', text, flags=re.IGNORECASE)
            text = re.sub(r'Corp\.?$', '', text, flags=re.IGNORECASE)
            text = re.sub(r'LLC$', '', text, flags=re.IGNORECASE)

            # 去除冗余行业词
            text = re.sub(r'科技$', '', text)
            text = re.sub(r'技术$', '', text)
            text = re.sub(r'信息$', '', text)
            text = re.sub(r'网络$', '', text)
            text = re.sub(r'软件$', '', text)
            text = re.sub(r'电子$', '', text)
            text = re.sub(r'商贸$', '', text)
            text = re.sub(r'贸易$', '', text)
            text = re.sub(r'咨询$', '', text)
            text = re.sub(r'服务$', '', text)

            # 去除括号内容
            text = re.sub(r'[（(][^）)]*[）)]', '', text)

            # 去除多余空格
            text = re.sub(r'\s+', '', text)
            
            # ===== 第二步：如果开启 LLM 且包含英文，调用大模型翻译 =====
            if use_llm and self.dify_client and self.dify_client.available:
                # 检查是否包含英文字母
                if re.search(r'[a-zA-Z]', original):
                    try:
                        llm_result = self._call_llm_for_company(original)
                        if llm_result and len(llm_result) > 0:
                            details["llm_used"] = details.get("llm_used", 0) + 1
                            # 再次清理 LLM 返回的结果（去空格等）
                            llm_result = re.sub(r'\s+', '', llm_result)
                            return llm_result
                    except Exception as e:
                        print(f"⚠️ LLM 调用失败，回退到 Baseline: {e}")
            
            return text
        
        df_copy[f"{field}_normalized"] = df_copy[field].apply(clean)
        details["normalized"] = (df_copy[field] != df_copy[f"{field}_normalized"]).sum()
        
        return {
            "data": df_copy,
            "summary": f"公司名规范化完成，共 {details['total']} 条，其中 LLM 处理 {details.get('llm_used', 0)} 条",
            "details": details,
        }

    def _call_llm_for_company(self, text: str) -> Optional[str]:
        """调用大模型规范公司名"""
        if not self.dify_client or not self.dify_client.available:
            return None
        
        prompt = f"""
你是一个专业的数据清洗专家。请将以下公司名规范化为标准中文名称：

要求：
1. 如果有英文，翻译成标准中文（如 Alpha → 阿尔法，Tech → 科技）
2. 如果包含缩写，展开为完整中文（如 Co → 公司）
3. 去除冗余词汇，保留核心名称
4. 只返回规范化后的名称，不要任何解释，不要加引号

公司名：{text}
规范化结果："""
        
        try:
            response = self.dify_client.chat(prompt)
            if response and response.get("success"):
                result = response.get("answer", "").strip()
                if result and len(result) > 0:
                    return result
            return None
        except Exception as e:
            print(f"❌ LLM 调用异常: {e}")
            return None

    def _normalize_phone(self, df: pd.DataFrame, params: dict) -> dict:
        """电话标准化（Baseline：正则）"""
        field = params.get("field", "phone")
        df_copy = df.copy()
        details = {"total": len(df_copy), "normalized": 0}
        
        def clean(text):
            if pd.isna(text) or not text:
                return text
            text = str(text)
            # 去除非数字字符
            text = re.sub(r'[^0-9]', '', text)
            return text
        
        df_copy[f"{field}_normalized"] = df_copy[field].apply(clean)
        details["normalized"] = (df_copy[field] != df_copy[f"{field}_normalized"]).sum()
        
        return {
            "data": df_copy,
            "summary": f"电话标准化完成，共 {details['total']} 条，修改 {details['normalized']} 条",
            "details": details,
        }

    def _normalize_all(self, df: pd.DataFrame, params: dict) -> dict:
        """一键标准化所有字段"""
        fields = params.get("fields", ["address", "company_name", "phone"])
        results = {}
        combined_df = df.copy()
        total_modified = 0
        
        for field in fields:
            if field == "address":
                res = self._normalize_address(combined_df, {"field": field})
            elif field == "company_name":
                res = self._normalize_company(combined_df, {"field": field})
            elif field == "phone":
                res = self._normalize_phone(combined_df, {"field": field})
            else:
                continue
            
            results[field] = res["details"]
            # 把标准化后的列合并到主 DataFrame
            norm_col = f"{field}_normalized"
            if norm_col in res["data"].columns:
                combined_df[norm_col] = res["data"][norm_col]
                total_modified += res["details"].get("normalized", 0)
        
        return {
            "data": combined_df,
            "summary": f"一键标准化完成，共处理 {len(combined_df)} 条记录，修改 {total_modified} 处",
            "details": {"fields": results, "total_modified": total_modified},
        }

    def _save_log(self, result: dict):
        """保存标准化日志到 data/normalization_log.csv"""
        df = result.get("data")
        if df is None:
            return
        
        log_path = self.data_dir / "normalization_log.csv"
        norm_cols = [c for c in df.columns if c.endswith("_normalized")]
        if not norm_cols:
            print("没有发现标准化列，跳过日志保存。")
            return
        
        # 重置索引，防止 "index" 找不到
        save_df = df.reset_index().rename(columns={'index': 'row_id'})
        
        # 提取要保存的列
        cols_to_save = ['row_id'] + norm_cols
        cols_to_save = [c for c in cols_to_save if c in save_df.columns]
        
        save_df[cols_to_save].to_csv(log_path, index=False)
        print(f"✅ 日志已成功保存到 {log_path}")

    def health_check(self) -> bool:
        return True

    def get_actions(self) -> List[str]:
        return ["normalize_address", "normalize_company", "normalize_phone", "normalize_all"]


# ==================== 独立运行入口 ====================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="数据源文件名，如 crm_customers.csv")
    parser.add_argument("--action", default="normalize_all", 
                       choices=["normalize_address", "normalize_company", "normalize_phone", "normalize_all"])
    args = parser.parse_args()
    
    agent = NormalizeAgent()
    
    # 读取全部数据
    data = pd.read_csv(agent.data_dir / args.source)
    
    # ===== 测试模式开关 =====
    # 取消下面一行的注释，即可只处理前 10 条数据（用于快速测试）
    # data = data.head(10)
    # ======================
    
    print(f"📊 共读取 {len(data)} 条数据")
    
    task = {
        "action": args.action,
        "params": {"source": args.source},
        "data": data
    }
    result = agent.run(task)
    print("=" * 50)
    print("执行结果:", result["summary"])
    print("详细信息:", result["details"])
    if result["data"] is not None:
        print("标准化后数据预览:")
        print(result["data"].head())
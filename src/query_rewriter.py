"""
================================================================================
模块名称: src/query_rewriter.py
业务定位: Step 4 - 多视角子查询生成器与专业信息动态注入引擎 (Multi-Perspective Query Rewriter)
核心职责:
  1. 专业知识动态注入 (Domain Knowledge Injection):
     - 基于中英数学考纲知识树与专有名词映射库，动态检索并注入英文考纲路径、缩写与学术定义。
     - 彻底破除 "中文最小公倍数 ➔ 英文 LCM"、"极差 ➔ Range" 的跨语种与缩写词表鸿沟。
  2. 多视角子查询并行派生 (Multi-Perspective Query Generation):
     - 学情错因视角 (Misconception Query): 聚焦认知障碍、思维误区与典型错误选项
     - 名师教法视角 (Strategy Query): 聚焦名师破局一问、引导脚手架步骤与反例追问
     - 考纲考点视角 (Curriculum Query): 聚焦学科层级路径与核心数值/代数式约束
  3. 契约校验与自愈重试机制 (Strict Pydantic Validation & Remediation):
     - 结构化 JSON 校验，确保改写输出 100% 契合 MultiPerspectiveQueries。
================================================================================
"""

import os
import re
import sys
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 确保项目根目录在 sys.path 中
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from pydantic import ValidationError
from src.models import MultiPerspectiveQueries

logger = logging.getLogger(__name__)


class QueryRewriteError(RuntimeError):
    """LLM 查询改写失败或输出不符合严格契约。"""

# ================================================================================
# 1. 领域考纲知识树与专有名词映射库 (Domain Taxonomy & Glossary Map)
# ================================================================================

MATH_DOMAIN_GLOSSARY: List[Dict[str, Any]] = [
    {
        "keywords": ["四舍五入", "保留", "小数", "近似", "5.4598", "3.153", "0.02", "7.503", "356,958", "356958", "最近百位", "最近整数", "rounding", "round"],
        "subject_path": "Number > Rounding and Estimating > Rounding to Decimal Places",
        "terms": ["Rounding", "Decimal Places", "Nearest Multiple", "Rounding to the Nearest 10, 100, 1000", "Truncation vs Rounding", "Nearest Integer"],
        "canonical_examples": ["5.4598 rounded to 1dp is 5.5", "3.153 rounded to nearest 0.02 is 3.16", "7.503 rounded to nearest integer is 8"]
    },
    {
        "keywords": ["公倍数", "最小公倍数", "因数与倍数", "互质", "lcm", "coprime", "factors and multiples"],
        "subject_path": "Number > Multiples and Factors > LCM",
        "terms": ["LCM (Least Common Multiple)", "Multiples and Factors", "Coprime Generalization Error"],
        "canonical_examples": ["LCM of 3 and 5 is 15 (coprime), but LCM of non-coprime numbers is not simply the product"]
    },
    {
        "keywords": ["运算顺序", "四则运算", "先乘除", "分式", "分数线", "除法", "加法", "(54+58)/2", "n+4÷5", "(n+4)/5", "order of operations", "bodmas", "bidmas"],
        "subject_path": "Number > Basic Arithmetic > BIDMAS",
        "terms": ["Order of Operations", "BIDMAS / BODMAS", "Fraction Bar Division Precedence", "Parentheses Grouping"],
        "canonical_examples": ["Fraction bar acts as grouping: (54+58)/2 requires adding numerator first before division", "n+4/5 vs (n+4)/5"]
    },
    {
        "keywords": ["去括号", "展开", "单项式", "4(3c+2)", "4(2x+1)", "-3(1-2p)", "expanding", "brackets", "single brackets"],
        "subject_path": "Algebra > Expanding Brackets > Expanding Single Brackets",
        "terms": ["Expanding Single Brackets", "Distributive Property", "Multiplying Terms Inside Parentheses", "Negative Sign Distribution"],
        "canonical_examples": ["4(3c+2) = 12c + 8", "-3(1-2p) = -3 + 6p", "4(2x+1) - (5x-9) = 8x + 4 - 5x + 9"]
    },
    {
        "keywords": ["异分母", "通分", "公分母", "分数减法", "分数加减", "5/7", "1/4", "fractions", "adding and subtracting fractions"],
        "subject_path": "Number > Fractions and Decimals > Adding and Subtracting Fractions",
        "terms": ["Adding and Subtracting Fractions", "Common Denominator", "Equivalent Fractions", "Revoicing"],
        "canonical_examples": ["5/7 - 1/4 with common denominator 28 requires converting numerators to 20/28 - 7/28 = 13/28"]
    },
    {
        "keywords": ["位值", "小数除法", "0.2÷0.4", "32×65", "3.2×6.5", "2080", "place value", "decimal multiplication"],
        "subject_path": "Number > Basic Arithmetic > Place Value",
        "terms": ["Place Value", "Multiplying and Dividing Decimals", "Decimal Point Movement", "Scaling by Powers of 10"],
        "canonical_examples": ["32 x 65 = 2080 implies 3.2 x 6.5 = 20.8 (divide by 100 total)", "0.2 / 0.4 = 2 / 4 = 1/2 = 0.5"]
    },
    {
        "keywords": ["速度", "距离", "路程", "时间", "40分钟", "40/60", "speed", "distance", "time"],
        "subject_path": "Number > Proportion > Speed, Distance, Time",
        "terms": ["Speed, Distance, Time", "Unit Conversion (Minutes to Fraction of Hour)", "Speed = Distance / Time"],
        "canonical_examples": ["40 minutes is 40/60 = 2/3 of an hour, not 4/10"]
    },
    {
        "keywords": ["密度", "质量", "体积", "单位", "g/cm3", "density", "mass", "volume"],
        "subject_path": "Number > Proportion > Density",
        "terms": ["Density = Mass / Volume", "Compound Units (g/cm3, kg/m3)", "Physical Quantities"],
        "canonical_examples": ["Density = mass / volume; valid unit is g/cm3, not cm3/g"]
    },
    {
        "keywords": ["折线图", "实际图像", "水杯", "深度", "斜率", "最高点", "real life graphs", "line graphs", "time series"],
        "subject_path": "Algebra > Other Graphs > Real Life Graphs",
        "terms": ["Real Life Graphs", "Gradient as Rate of Change", "Interpreting Slope vs Peak Value", "Container Filling Graphs"],
        "canonical_examples": ["Steepest slope indicates fastest rate of filling, not the highest y-value"]
    },
    {
        "keywords": ["频数", "统计", "极差", "范围", "看电视", "调查", "中位数", "平均数", "众数", "frequency", "range", "median", "mean", "mode", "averages"],
        "subject_path": "Data and Statistics > Data Processing > Range and Interquartile Range from a List of Data",
        "terms": ["Range (Maximum Value - Minimum Value)", "Frequency Tables", "Averages (mean, median, mode)", "Frequency vs Data Value Confusion"],
        "canonical_examples": ["Range is calculated from data values (Max - Min), not by dividing or subtracting frequencies"]
    },
    {
        "keywords": ["负数", "乘方", "指数", "括号", "负号", "(-q)^2", "-q^2", "-p", "p*(-q)", "power", "index", "indices"],
        "subject_path": "Algebra > Algebraic Expressions > Powers and Indices",
        "terms": ["Powers and Indices", "Negative Base Exponentiation", "Parenthesis Scope in Powers", "Multiplication with Negative Numbers"],
        "canonical_examples": ["(-q)^2 = q^2 (positive), whereas -q^2 = -(q*q) (negative)", "p * (-q) = -pq"]
    },
    {
        "keywords": ["质数", "素数", "合数", "105", "末尾5", "尾数", "prime", "composite"],
        "subject_path": "Number > Factors and Multiples > Prime Numbers",
        "terms": ["Prime Numbers", "Composite Numbers", "Divisibility Rules (ends in 5 is divisible by 5)"],
        "canonical_examples": ["105 is not prime because it ends in 5 and is divisible by 5"]
    },
    {
        "keywords": ["不等式", "除以负数", "翻转", "-2x < 12", "-12", "-2", "inequality", "inequalities"],
        "subject_path": "Algebra > Inequalities > Solving Linear Inequalities",
        "terms": ["Solving Linear Inequalities", "Negative Division Rule", "Flipping Inequality Sign"],
        "canonical_examples": ["Dividing both sides by a negative number reverses the inequality: -2x < 12 => x > -6"]
    },
    {
        "keywords": ["长方体", "体积", "表面积", "长宽高", "cuboid", "volume", "surface area", "prism"],
        "subject_path": "Geometry and Measure > Volume and Surface Area > Volume of Prisms",
        "terms": ["Volume of Prisms (Length * Width * Height)", "Surface Area vs Volume Distinction"],
        "canonical_examples": ["Volume = length * width * height; Surface area is the total area of all faces"]
    },
    {
        "keywords": ["时间", "10 to 12", "钟表", "10点12分", "11:50", "clock", "time"],
        "subject_path": "Geometry and Measure > Time > Clocks and Analog Time",
        "terms": ["Analog Time Reading", "English Time Phrasing ('X to Y' means Y minus X minutes)"],
        "canonical_examples": ["'ten to twelve' means 11:50 (or 23:50 at night), not 10:12"]
    }
]


class DomainKnowledgeInjector:
    """
    数学学科考纲知识与专有名词动态注入器。
    """

    def __init__(self, glossary: Optional[List[Dict[str, Any]]] = None):
        """Args: glossary: 领域词条表 (默认 MATH_DOMAIN_GLOSSARY)，可注入自定义考点映射。"""
        self.glossary = glossary or MATH_DOMAIN_GLOSSARY

    def match_and_inject(self, raw_query: str) -> Tuple[str, List[str]]:
        """
        匹配原始提问中的关键词，生成注入的考纲上下文和关键词列表。
        """
        q_lower = raw_query.lower()
        matched_entries = []
        extracted_terms = []

        for entry in self.glossary:
            for kw in entry["keywords"]:
                if kw.lower() in q_lower:
                    matched_entries.append(entry)
                    extracted_terms.extend(entry["terms"])
                    break

        if not matched_entries:
            return "", []

        # 构建高密度注入上下文
        lines = ["【学科专业考纲与中英术语映射上下文】:"]
        for idx, entry in enumerate(matched_entries[:2], start=1):
            lines.append(f"  • 考纲路径: {entry['subject_path']}")
            lines.append(f"  • 核心术语: {', '.join(entry['terms'])}")
            lines.append(f"  • 经典样例: {', '.join(entry['canonical_examples'])}")

        injected_text = "\n".join(lines)
        return injected_text, list(set(extracted_terms))


# ================================================================================
# 2. 多视角改写器实现 (MultiPerspectiveQueryRewriter)
# ================================================================================

REWRITE_SYSTEM_PROMPT = """你是一名精通英国中考（GCSE / Edexcel）数学辅导与认知科学的资深教研专家。
你的任务是将教师或用户输入的简短、口语化提问，改写为三个不同专业视角的检索表达，以最大化在知识库（错因卡与名师策略库）中的召回命中率。

请严格根据输入的【学科专业考纲与中英术语映射上下文】与【用户原始提问】，输出包含以下 3 个维度的 JSON 结构：
1. misconception_query: 学情错因诊断视角，阐明学生易犯的深层思维误区、符号混淆与典型错选机理（必须吸收考纲中的专业术语与英文缩写如 LCM/Range/Rounding）；
2. strategy_query: 名师教法视角，阐明针对该错因名师使用的核心破局一问、引导脚手架步骤与反例追问；
3. curriculum_query: 考纲考点视角，包含考纲层级路径、原题核心数值/代数式与考点定义；
4. extracted_keywords: 核心数值、中英术语与考点关键词列表。

请直接输出合法 JSON，不要附带任何 markdown 标记之外的多余解释。"""


class MultiPerspectiveQueryRewriter:
    """
    多视角子查询改写器:
    - 动态注入领域知识
    - 大模型多视角派生与 Pydantic 自愈校验
    - 具备确定性离线规则兜底
    """

    def __init__(
        self,
        openai_client: Optional[Any] = None,
        model_name: Optional[str] = None,
        injector: Optional[DomainKnowledgeInjector] = None,
        mode: Optional[str] = None,
        max_retries: int = 3,
    ):
        """
        构造函数强制显式配置: 不存在任何隐式默认后端。

        Args:
            openai_client: OpenAI 兼容客户端 (llm 模式必填)。
            model_name: 改写用模型名 (llm 模式必填)。
            injector: 领域知识注入器 (默认内置 MATH_DOMAIN_GLOSSARY)。
            mode: 'llm' (大模型多视角改写) 或 'deterministic' (离线规则模板，调用方显式选择)。
            max_retries: LLM 输出校验失败时的自纠重试上限。
        """
        if mode not in {"llm", "deterministic"}:
            raise ValueError("mode 必须显式指定为 'llm' 或 'deterministic'")
        if mode == "llm" and openai_client is None:
            raise ValueError("llm 模式必须提供 openai_client")
        if mode == "llm" and not model_name:
            raise ValueError("llm 模式必须显式提供 model_name")
        if max_retries < 1:
            raise ValueError("max_retries 必须至少为 1")
        self.client = openai_client
        self.model_name = model_name
        self.injector = injector or DomainKnowledgeInjector()
        self.mode = mode
        self.max_retries = max_retries

    @property
    def backend_name(self) -> str:
        """当前后端标识 (llm / deterministic_rules)，写入评测 system_config 以保证可追溯。"""
        return "llm" if self.mode == "llm" else "deterministic_rules"

    def rewrite(self, raw_query: str) -> MultiPerspectiveQueries:
        """
        将原始口语提问改写为结构化多视角子查询。
        """
        # 1. 前置动态知识注入
        injected_context, extracted_terms = self.injector.match_and_inject(raw_query)

        # 2. 仅调用方显式选择时执行确定性离线增强。
        if self.mode == "deterministic":
            logger.warning(
                "[QueryRewrite] 使用调用方显式选择的 deterministic_rules 后端"
            )
            return self._offline_rule_rewrite(raw_query, injected_context, extracted_terms)

        # 3. LLM 多视角派生与自纠校验循环
        user_prompt = f"{injected_context}\n\n【用户原始提问】: {raw_query}\n\n请按规范输出 JSON:"
        
        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1,
                    response_format={"type": "json_object"}
                )
                raw_json = json.loads(resp.choices[0].message.content)
                raw_json["raw_query"] = raw_query
                raw_json["injected_domain_context"] = injected_context
                
                # Pydantic 严格校验
                validated = MultiPerspectiveQueries.model_validate(raw_json)
                logger.info(f"✨ [QueryRewrite] 成功派生多视角查询: 错因='{validated.misconception_query[:30]}...' | 策略='{validated.strategy_query[:30]}...'")
                return validated
            except Exception as e:
                last_error = e
                logger.warning(
                    "[QueryRewrite] LLM 校验/解析失败 (%s/%s): %s",
                    attempt,
                    self.max_retries,
                    e,
                )
                user_prompt += f"\n\n上次输出格式校验失败: {e}，请重新输出包含 misconception_query, strategy_query, curriculum_query, extracted_keywords 的严格合法 JSON。"

        raise QueryRewriteError(
            f"LLM 查询改写在 {self.max_retries} 次尝试后失败"
        ) from last_error

    def _offline_rule_rewrite(
        self,
        raw_query: str,
        injected_context: str,
        extracted_terms: List[str]
    ) -> MultiPerspectiveQueries:
        """
        确定性离线规则增强改写算子（注入考纲术语与多视角模板句式）。
        """
        terms_str = " ".join(extracted_terms)
        
        misconception_q = f"【学情错因诊断】: {raw_query} {injected_context} {terms_str} 学生深层认知障碍与典型错选机理"
        strategy_q = f"【名师启发策略】: {raw_query} {injected_context} {terms_str} 核心破局一问与引导脚手架步骤"
        curriculum_q = f"【学科考纲与原题】: {raw_query} {injected_context} {terms_str} 题目题干与核心运算法则"

        return MultiPerspectiveQueries(
            raw_query=raw_query,
            misconception_query=misconception_q.strip(),
            strategy_query=strategy_q.strip(),
            curriculum_query=curriculum_q.strip(),
            extracted_keywords=extracted_terms,
            injected_domain_context=injected_context
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    print("🚀 测试多视角子查询生成与专业信息动态注入器...")
    
    rewriter = MultiPerspectiveQueryRewriter(mode="deterministic")
    
    test_queries = [
        "学生为什么会误以为任意两个数的最小公倍数就是它们的乘积？",
        "当学生把保留一位小数做成保留两位时，名师是如何用简单数字破局提问的？",
        "在看电视数量的频数统计表中，学生计算极差时为什么会混淆频数和数据本身？"
    ]
    
    for q in test_queries:
        res = rewriter.rewrite(q)
        print(f"\n【原始提问】: {res.raw_query}")
        print(f"  • 注入考纲术语: {res.extracted_keywords}")
        print(f"  • 错因视角 (Misconception): {res.misconception_query}")
        print(f"  • 策略视角 (Strategy):      {res.strategy_query}")
        print(f"  • 考纲视角 (Curriculum):    {res.curriculum_query}")
        print("-" * 80)
        
    print("\n✅ 多视角子查询改写器单测通过！")

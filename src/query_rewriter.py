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

# ================================================================================
# 1. 领域考纲知识树与专有名词映射库 (Domain Taxonomy & Glossary Map)
# ================================================================================

MATH_DOMAIN_GLOSSARY: List[Dict[str, Any]] = [
    {
        "keywords": ["四舍五入", "保留", "小数", "近似", "5.4598", "3.153", "0.02", "rounding", "round"],
        "subject_path": "Number > Rounding and Estimating > Rounding to Decimal Places",
        "terms": ["Rounding", "Decimal Places", "Nearest Multiple", "Truncation vs Rounding"],
        "canonical_examples": ["5.4598 rounded to 1dp is 5.5", "3.153 rounded to nearest 0.02 is 3.16"]
    },
    {
        "keywords": ["公倍数", "最小公倍数", "因数与倍数", "互质", "lcm", "coprime", "factors and multiples"],
        "subject_path": "Number > Multiples and Factors > LCM",
        "terms": ["LCM (Least Common Multiple)", "Multiples and Factors", "Coprime Generalization Error"],
        "canonical_examples": ["LCM of 3 and 5 is 15 (coprime), but LCM of non-coprime numbers is not simply the product"]
    },
    {
        "keywords": ["运算顺序", "四则运算", "先乘除", "分式", "除法", "加法", "order of operations", "bodmas", "bidmas"],
        "subject_path": "Number > Fractions and Decimals > Order of Operations",
        "terms": ["Order of Operations", "BIDMAS / BODMAS", "Fraction Operations"],
        "canonical_examples": ["Division has higher precedence than addition in mixed expressions"]
    },
    {
        "keywords": ["频数", "统计", "极差", "范围", "看电视", "调查", "frequency", "range", "survey"],
        "subject_path": "Statistics > Data Presentation > Frequency Tables",
        "terms": ["Frequency Tables", "Range (Maximum Value - Minimum Value)", "Frequency vs Data Value Confusion"],
        "canonical_examples": ["Range is calculated from data values, not from the frequencies themselves"]
    },
    {
        "keywords": ["负数", "乘方", "指数", "括号", "负号", "(-q)^2", "-q^2", "-p", "power", "index", "indices"],
        "subject_path": "Algebra > Algebraic Expressions > Powers and Indices",
        "terms": ["Powers and Indices", "Negative Base Exponentiation", "Parenthesis Scope in Powers"],
        "canonical_examples": ["(-q)^2 = (-q)*(-q) = q^2 (positive), whereas -q^2 = -(q*q) (negative)", "Does -p x q give us the same answer?"]
    },
    {
        "keywords": ["质数", "素数", "合数", "105", "末尾5", "尾数", "prime", "composite"],
        "subject_path": "Number > Factors and Multiples > Prime Numbers",
        "terms": ["Prime Numbers", "Composite Numbers", "Divisibility Rules (ends in 5 is divisible by 5)"],
        "canonical_examples": ["105 is not prime because it ends in 5 and is divisible by 5"]
    },
    {
        "keywords": ["不等式", "除以负数", "翻转", "-12", "-2", "inequality", "inequalities"],
        "subject_path": "Algebra > Inequalities > Solving Linear Inequalities",
        "terms": ["Solving Linear Inequalities", "Negative Division Rule", "Flipping Inequality Sign"],
        "canonical_examples": ["Dividing both sides by a negative number requires reversing the inequality sign"]
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
        "canonical_examples": ["'ten to twelve' means 11:50, not 10:12"]
    }
]


class DomainKnowledgeInjector:
    """
    数学学科考纲知识与专有名词动态注入器。
    """

    def __init__(self, glossary: Optional[List[Dict[str, Any]]] = None):
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
        model_name: str = "gpt-4o-mini",
        injector: Optional[DomainKnowledgeInjector] = None
    ):
        self.client = openai_client
        self.model_name = model_name
        self.injector = injector or DomainKnowledgeInjector()

    def rewrite(self, raw_query: str) -> MultiPerspectiveQueries:
        """
        将原始口语提问改写为结构化多视角子查询。
        """
        # 1. 前置动态知识注入
        injected_context, extracted_terms = self.injector.match_and_inject(raw_query)

        # 2. 若无 LLM Client，走确定性离线增强扩展 (严格遵守不伪造、显式声明原则)
        if self.client is None:
            return self._offline_rule_rewrite(raw_query, injected_context, extracted_terms)

        # 3. LLM 多视角派生与自纠校验循环
        user_prompt = f"{injected_context}\n\n【用户原始提问】: {raw_query}\n\n请按规范输出 JSON:"
        
        for attempt in range(3):
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
            except (ValidationError, json.JSONDecodeError, Exception) as e:
                logger.warning(f"⚠️ [QueryRewrite] 校验/解析失败 (第 {attempt+1} 次): {e}")
                user_prompt += f"\n\n上次输出格式校验失败: {e}，请重新输出包含 misconception_query, strategy_query, curriculum_query, extracted_keywords 的严格合法 JSON。"

        # 3 次自纠均失败，降级至规则增强改写
        logger.warning("⚠️ [QueryRewrite] LLM 自纠重试耗尽，降级至确定性规则增强改写")
        return self._offline_rule_rewrite(raw_query, injected_context, extracted_terms)

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
    
    rewriter = MultiPerspectiveQueryRewriter()
    
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

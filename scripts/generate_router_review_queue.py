"""
================================================================================
脚本名称: scripts/generate_router_review_queue.py
业务定位: 生成 Query Router 的 AI 草拟人工审核队列 (Draft review queue)

本脚本只生成待审核候选, 不生成正式 Gold:
  1. 预定义覆盖 9 个业务意图的候选问题与人工待审期望字段;
  2. 所有输出强制标记 label_source="draft";
  3. 生成 135 条 JSONL, 供人逐条确认/修改后再签署为正式验收集;
  4. 不导入、不调用 BusinessQueryRouter, 防止被测系统反向决定期望标签。

真实性边界:
  - 本脚本是"审核准备工具", 不是评测工具;
  - 生成物不能直接喂给 evaluate_business_router.py 的正式 release gate;
  - 正式评测前必须由人把审核通过的行复制到 query-router-v1.jsonl,
    并把 label_source 逐条改为 human。
================================================================================
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Sequence


# ---------------------------------------------------------------------------
# 考纲筛选模板: 使用 query_rewriter.py 已定义的稳定 subject_path 值
# ---------------------------------------------------------------------------


SUBJECTS: tuple[tuple[str, str, str], ...] = (
    (
        "fractions",
        "分数加减",
        "Number > Fractions and Decimals > Adding and Subtracting Fractions",
    ),
    (
        "rounding",
        "rounding",
        "Number > Rounding and Estimating > Rounding to Decimal Places",
    ),
    (
        "LCM",
        "最小公倍数 LCM",
        "Number > Multiples and Factors > LCM",
    ),
    (
        "inequality",
        "不等式",
        "Algebra > Inequalities > Solving Linear Inequalities",
    ),
    (
        "prime number",
        "质数 prime number",
        "Number > Factors and Multiples > Prime Numbers",
    ),
)


def _draft_case(
    *,
    case_index: int,
    query: str,
    primary_intent: str,
    scope: str,
    computation_scope: str,
    evidence_perspectives: list[str],
    subject_filters: list[str] | None = None,
    protected_tokens: list[str] | None = None,
    requested_top_k: int = 3,
    secondary_intents: list[str] | None = None,
    split: str,
) -> dict[str, Any]:
    """构造单条待人工审核记录; label_source 永远不能由调用方覆盖。"""

    return {
        "case_id": f"router-{case_index:03d}",
        "query": query,
        "primary_intent": primary_intent,
        "secondary_intents": secondary_intents or [],
        "scope": scope,
        "computation_scope": computation_scope,
        "evidence_perspectives": evidence_perspectives,
        "subject_filters": subject_filters or [],
        "protected_tokens": protected_tokens or [],
        "requested_top_k": requested_top_k,
        "label_source": "draft",
        "split": split,
    }


def _add_corpus_and_subject_group(
    *,
    rows: list[dict[str, Any]],
    start_index: int,
    primary_intent: str,
    perspectives: list[str],
    corpus_queries: Iterable[tuple[str, int, list[str]]],
    subject_query_template: str,
    secondary_intents: list[str] | None = None,
) -> int:
    """为宏观意图追加 10 条 CORPUS + 5 条 SUBJECT 草案, 返回下一个 ID。"""

    index = start_index
    for position, (query, top_k, tokens) in enumerate(corpus_queries):
        rows.append(
            _draft_case(
                case_index=index,
                query=query,
                primary_intent=primary_intent,
                scope="CORPUS",
                computation_scope="HYBRID",
                evidence_perspectives=perspectives,
                protected_tokens=tokens,
                requested_top_k=top_k,
                secondary_intents=secondary_intents if position == 0 else [],
                split="dev" if position < 7 else "holdout",
            )
        )
        index += 1
    for position, (english_term, chinese_term, subject_path) in enumerate(SUBJECTS):
        # 中英交替, 保证审核队列包含真实的跨语言考纲表达。
        topic = chinese_term if position % 2 == 0 else english_term
        rows.append(
            _draft_case(
                case_index=index,
                query=subject_query_template.format(topic=topic),
                primary_intent=primary_intent,
                scope="SUBJECT",
                computation_scope="HYBRID",
                evidence_perspectives=perspectives,
                subject_filters=[subject_path],
                secondary_intents=secondary_intents if position == 0 else [],
                split="dev" if position < 2 else "holdout",
            )
        )
        index += 1
    return index


def _add_cohort_group(rows: list[dict[str, Any]], start_index: int) -> int:
    """群体分析候选始终是 CORPUS/HYBRID/STUDENT, 后续由能力门禁决定是否执行。"""

    queries = (
        "重考者最常见的思维卡点是什么？",
        "多次重考的学生有哪些共性模式？",
        "基础薄弱学生常见哪些认知盲区？",
        "低基础学生在辅导中有什么共同表现？",
        "重考者和基础薄弱学生是否有相同模式？",
        "哪些学生群体需要更多脚手架支持？",
        "重考学生最常问什么？",
        "基础薄弱学生最容易卡在哪些概念？",
        "哪些模式在重考者身上反复出现？",
        "What patterns appear among retakers?",
        "What misconceptions do students with a weak foundation share?",
        "多次重考者在分数加减中有什么共同困难？",
        "重考者最容易犯哪些错误？",
        "低基础群体的高频问题是什么？",
        "Retakers: Top 5 shared thinking blocks",
    )
    index = start_index
    for position, query in enumerate(queries):
        top_k = 5 if "Top 5" in query else 3
        tokens = ["5"] if top_k == 5 else []
        subject_filters = (
            ["Number > Fractions and Decimals > Adding and Subtracting Fractions"]
            if "分数加减" in query
            else []
        )
        rows.append(
            _draft_case(
                case_index=index,
                query=query,
                primary_intent="STUDENT_COHORT_PATTERNS",
                scope="CORPUS",
                computation_scope="HYBRID",
                evidence_perspectives=["STUDENT"],
                subject_filters=subject_filters,
                protected_tokens=tokens,
                requested_top_k=top_k,
                split="dev" if position < 10 else "holdout",
            )
        )
        index += 1
    return index


def _add_question_group(rows: list[dict[str, Any]], start_index: int) -> int:
    """微观单题候选: 固定 QUESTION/MICRO/双视角, 公式或数字必须原样受保护。"""

    prompts: tuple[tuple[str, list[str], list[str]], ...] = (
        ("为什么 (-q)^2 和 -q^2 不同？导师怎么引导？", ["(-q)^2", "-q^2"], ["Algebra > Algebraic Expressions > Powers and Indices"]),
        ("How do I solve 5/7 - 1/4?", ["5/7 - 1/4"], ["Number > Fractions and Decimals > Adding and Subtracting Fractions"]),
        ("为什么 0.2÷0.4 等于 0.5？", ["0.2", "0.4", "0.5"], ["Number > Basic Arithmetic > Place Value"]),
        ("3.153 四舍五入到最近 0.02 应该怎么理解？", ["3.153", "0.02"], ["Number > Rounding and Estimating > Rounding to Decimal Places"]),
        ("为什么 -2x < 12 除以 -2 要翻转符号？", ["-2", "12"], ["Algebra > Inequalities > Solving Linear Inequalities"]),
        ("How can a tutor explain why 105 is not prime?", ["105"], ["Number > Factors and Multiples > Prime Numbers"]),
        ("学生为什么把 10 to 12 读成 10:12？", ["10", "12", "10:12"], ["Geometry and Measure > Time > Clocks and Analog Time"]),
        ("为什么 32×65 变成 3.2×6.5 后结果不是 2080？", ["32", "65", "3.2", "6.5", "2080"], ["Number > Basic Arithmetic > Place Value"]),
        ("What is the first step for 4(3c+2)?", ["4", "(3c+2)"], ["Algebra > Expanding Brackets > Expanding Single Brackets"]),
        ("学生把 5.4598 保留一位小数写成 5.45，错在哪里？", ["5.4598", "5.45"], ["Number > Rounding and Estimating > Rounding to Decimal Places"]),
        ("为什么 40 分钟要写成 40/60 小时？", ["40", "40/60"], ["Number > Proportion > Speed, Distance, Time"]),
        ("How should a tutor distinguish volume from surface area for a cuboid?", [], ["Geometry and Measure > Volume and Surface Area > Volume of Prisms"]),
        ("判断 105 是否为质数时，学生应该先看什么？", ["105"], ["Number > Factors and Multiples > Prime Numbers"]),
        ("为什么 (54+58)/2 不能先算 58/2？", ["(54+58)/2", "58/2"], ["Number > Basic Arithmetic > BIDMAS"]),
        ("How do I compare option A and option B for 5/7-1/4?", ["option A", "option B", "5/7-1/4"], ["Number > Fractions and Decimals > Adding and Subtracting Fractions"]),
    )
    index = start_index
    for position, (query, tokens, subject_filters) in enumerate(prompts):
        rows.append(
            _draft_case(
                case_index=index,
                query=query,
                primary_intent="QUESTION_TUTORING",
                scope="QUESTION",
                computation_scope="MICRO",
                evidence_perspectives=["STUDENT", "TUTOR"],
                subject_filters=subject_filters,
                protected_tokens=tokens,
                split="dev" if position < 8 else "holdout",
            )
        )
        index += 1
    return index


def _add_unsupported_group(rows: list[dict[str, Any]], start_index: int) -> int:
    """业务边界明确不支持的候选: 必须是 REJECT, 不应向相邻 RAG 能力滑落。"""

    queries = (
        "谁是最优秀的导师？",
        "哪个导师是最好的导师？",
        "哪位导师属于最佳导师？",
        "这个策略能提升多少分？",
        "该教学策略可以提升多少分？",
        "使用脚手架能提高多少分？",
        "这种方法的因果效果是什么？",
        "NBCOT 最难的内容是什么？",
        "NBCOT 学生最常犯什么错误？",
        "Which NBCOT topic is hardest?",
        "Which NBCOT strategy improves scores?",
        "请比较最优秀的导师和普通导师。",
        "最好的导师一定使用哪些策略？",
        "NBCOT 的通过率可以提升多少分？",
        "谁是当前最优秀的导师？",
    )
    index = start_index
    for position, query in enumerate(queries):
        rows.append(
            _draft_case(
                case_index=index,
                query=query,
                primary_intent="UNSUPPORTED",
                scope="CORPUS",
                computation_scope="REJECT",
                evidence_perspectives=["STUDENT"],
                split="dev" if position < 8 else "holdout",
            )
        )
        index += 1
    return index


def build_review_cases() -> list[dict[str, Any]]:
    """生成固定 135 条 AI 草拟候选; 该函数不读取也不调用被测 Router。"""

    rows: list[dict[str, Any]] = []
    index = 1

    # 1) 学生高频问题: 宏观统计 + 学生视角证据。
    index = _add_corpus_and_subject_group(
        rows=rows,
        start_index=index,
        primary_intent="STUDENT_FREQUENT_QUESTIONS",
        perspectives=["STUDENT"],
        corpus_queries=(
            ("学生在辅导中最常提出的问题是什么？", 3, []),
            ("学生最常问什么？", 3, []),
            ("列出学生高频问题 Top 5", 5, ["5"]),
            ("总体来看学生最常提出哪些困惑？", 3, []),
            ("学生 FAQ 包含哪些高频问题？", 3, []),
            ("学生反复提问的高频问题有哪些？", 3, []),
            ("全库学生最常问哪些问题？", 3, []),
            ("What do students ask most often?", 3, []),
            ("What are the most common student questions?", 3, []),
            ("Students frequently ask which kinds of questions?", 3, []),
        ),
        subject_query_template="{topic} 中学生最常问什么？",
    )

    # 2) 困难概念: 宏观统计 + 学生视角; 导师视角可由人工审核时追加。
    index = _add_corpus_and_subject_group(
        rows=rows,
        start_index=index,
        primary_intent="DIFFICULT_CONCEPTS",
        perspectives=["STUDENT"],
        corpus_queries=(
            ("哪些概念或主题最难理解？", 3, []),
            ("学生在哪些主题上最容易卡壳？", 3, []),
            ("哪些困难主题耗时最长？", 3, []),
            ("哪些概念让学生最费时间？", 3, []),
            ("学生最难攻克的主题有哪些？", 3, []),
            ("总体最难理解的概念是什么？", 3, []),
            ("Top 5 difficult concepts", 5, ["5"]),
            ("What are the most difficult concepts?", 3, []),
            ("What do students struggle with most overall?", 3, []),
            ("Which concepts are hardest for students?", 3, []),
        ),
        subject_query_template="{topic} 中最难理解的概念是什么？",
    )

    # 3) 共性误区: 跨会话学生错因聚合。
    index = _add_corpus_and_subject_group(
        rows=rows,
        start_index=index,
        primary_intent="RECURRING_MISCONCEPTIONS",
        perspectives=["STUDENT"],
        corpus_queries=(
            ("有哪些共性误区反复出现？", 3, []),
            ("学生常见的认知错误有哪些？", 3, []),
            ("哪些错误理解在不同学生身上反复出现？", 3, []),
            ("学生有哪些共性认知盲区？", 3, []),
            ("最常见的误解是什么？", 3, []),
            ("哪些概念最容易混淆？", 3, []),
            ("Top 5 recurring misconceptions", 5, ["5"]),
            ("What recurring misconceptions appear across students?", 3, []),
            ("What common misunderstandings do students have?", 3, []),
            ("Which misconceptions recur most often?", 3, []),
        ),
        subject_query_template="{topic} 有哪些共性误区？",
    )

    # 4) 考试错误: 从学生端错误表现做模式归纳。
    index = _add_corpus_and_subject_group(
        rows=rows,
        start_index=index,
        primary_intent="EXAM_ERROR_PATTERNS",
        perspectives=["STUDENT"],
        corpus_queries=(
            ("学生在考试中最常犯哪些错误？", 3, []),
            ("有哪些稳定的考试错误模式？", 3, []),
            ("学生常见的误选模式是什么？", 3, []),
            ("学生做选择题时有哪些答题错误？", 3, []),
            ("考试中审题错误有哪些模式？", 3, []),
            ("哪些程序性答题错误反复出现？", 3, []),
            ("学生考试时最容易遗漏哪些条件？", 3, []),
            ("What exam errors recur most often?", 3, []),
            ("What test mistakes do students make?", 3, []),
            ("Top 5 wrong answer patterns", 5, ["5"]),
        ),
        subject_query_template="{topic} 题中常见的错误模式有哪些？",
    )

    # 5) 群体模式: 只识别意图, 是否执行由 DataCapabilityRegistry 决定。
    index = _add_cohort_group(rows, index)

    # 6) 导师策略: 宏观策略模式以 tutor 证据为主。
    index = _add_corpus_and_subject_group(
        rows=rows,
        start_index=index,
        primary_intent="TUTOR_STRATEGY_PATTERNS",
        perspectives=["TUTOR"],
        corpus_queries=(
            ("导师反复使用哪些引导策略？", 3, []),
            ("导师通常怎么引导学生？", 3, []),
            ("金牌导师常传授哪些教学策略？", 3, []),
            ("导师最常使用哪些脚手架？", 3, []),
            ("导师话术中反复出现的引导方式是什么？", 3, []),
            ("导师常用哪些策略帮助学生理解？", 3, []),
            ("Top 5 tutor strategies", 5, ["5"]),
            ("Tutor strategies that recur most often", 3, []),
            ("What teaching strategies do tutors use most often?", 3, []),
            ("How do tutors guide students?", 3, []),
        ),
        subject_query_template="{topic} 中导师通常使用什么策略？",
    )

    # 7) 内容改进: 同时需要学生需求与导师经验两个证据视角。
    index = _add_corpus_and_subject_group(
        rows=rows,
        start_index=index,
        primary_intent="CONTENT_IMPROVEMENT",
        perspectives=["STUDENT", "TUTOR"],
        corpus_queries=(
            ("哪些教材内容应该改进？", 3, []),
            ("根据辅导记录应该补哪些微课？", 3, []),
            ("题库解析应该如何改进？", 3, []),
            ("哪些教育内容最值得优先开发？", 3, []),
            ("课程改进应该关注什么？", 3, []),
            ("基于数据应该新增哪些教学内容？", 3, []),
            ("Top 5 content improvement opportunities", 5, ["5"]),
            ("What micro lessons should we create?", 3, []),
            ("What content improvements do the tutoring records suggest?", 3, []),
            ("How should the curriculum improve?", 3, []),
        ),
        subject_query_template="{topic} 教材内容应该如何改进？",
    )

    # 8) 单题辅导: 公式/数字均保留为 protected tokens, 双视角只用于内部证据链。
    index = _add_question_group(rows, index)

    # 9) 不支持: 人员排名、因果分数和 NBCOT 跨领域结论必须拒绝。
    index = _add_unsupported_group(rows, index)

    if len(rows) != 135 or index != 136:
        raise RuntimeError(f"review queue construction error: expected 135 rows, got {len(rows)}")
    return rows


def write_review_queue(output: Path) -> int:
    """把草案写入一个此前不存在的 JSONL 文件, 返回实际条数。"""

    resolved = output.resolve()
    if resolved.exists():
        raise FileExistsError(f"refusing to overwrite draft review queue: {resolved}")
    rows = build_review_cases()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口: 只创建 draft 审阅队列, 绝不生成正式 human Gold。"""

    parser = argparse.ArgumentParser(
        description="Generate a draft-only review queue for business query router labels"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evals/datasets/query-router-v1.review.jsonl"),
    )
    args = parser.parse_args(argv)
    try:
        count = write_review_queue(args.output)
    except (FileExistsError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "case_count": count,
                "label_source": "draft",
                "release_ready": False,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

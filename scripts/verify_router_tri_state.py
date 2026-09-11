"""
验证 BusinessQueryRouter 路由结果是否严格仅包含三种状态:
1. RESOLVED
2. NEED_LLM
3. UNSUPPORTED
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.business_models import BusinessQueryRequest, QueryIntent
from src.query_intent import BusinessQueryRouter

# 第一批: 10 个标准测试用例 (正例、反例、冲突待消歧例)
FIRST_BATCH = [
    {
        "id": 1,
        "category": "正例 (RESOLVED)",
        "query": "学生在辅导中最常提出的问题是什么？列出10条",
        "expected_status": "RESOLVED",
        "expected_intent": QueryIntent.STUDENT_FREQUENT_QUESTIONS,
    },
    {
        "id": 2,
        "category": "正例 (RESOLVED)",
        "query": "导师反复使用哪些引导策略？",
        "expected_status": "RESOLVED",
        "expected_intent": QueryIntent.TUTOR_STRATEGY_PATTERNS,
    },
    {
        "id": 3,
        "category": "正例 (RESOLVED)",
        "query": "哪些概念让学生最难理解？",
        "expected_status": "RESOLVED",
        "expected_intent": QueryIntent.DIFFICULT_CONCEPTS,
    },
    {
        "id": 4,
        "category": "正例 (RESOLVED)",
        "query": "学生做选择题时最常犯的审题错误有哪些？",
        "expected_status": "RESOLVED",
        "expected_intent": QueryIntent.EXAM_ERROR_PATTERNS,
    },
    {
        "id": 5,
        "category": "正例 (RESOLVED)",
        "query": "为什么 5/7-1/4 不能直接减分母？",
        "expected_status": "RESOLVED",
        "expected_intent": QueryIntent.QUESTION_TUTORING,
    },
    {
        "id": 6,
        "category": "正例 (RESOLVED)",
        "query": "基于数据分析应该优先改进教材哪些内容？",
        "expected_status": "RESOLVED",
        "expected_intent": QueryIntent.CONTENT_IMPROVEMENT,
    },
    {
        "id": 7,
        "category": "反例 (UNSUPPORTED - 最佳评价禁令)",
        "query": "机构里谁是最优秀的导师？请给我排名",
        "expected_status": "UNSUPPORTED",
        "expected_intent": QueryIntent.UNSUPPORTED,
    },
    {
        "id": 8,
        "category": "反例 (UNSUPPORTED - 因果提分禁令)",
        "query": "使用苏格拉底提问法能帮学生提升多少分？",
        "expected_status": "UNSUPPORTED",
        "expected_intent": QueryIntent.UNSUPPORTED,
    },
    {
        "id": 9,
        "category": "反例 (UNSUPPORTED - 跨域语料禁令)",
        "query": "NBCOT 考试中最难的知识点是哪一部分？",
        "expected_status": "UNSUPPORTED",
        "expected_intent": QueryIntent.UNSUPPORTED,
    },
    {
        "id": 10,
        "category": "冲突/待消歧例 (NEED_LLM - 多意图冲突)",
        "query": "重考学生最常问什么？",
        "expected_status": "NEED_LLM",
        "expected_intent": QueryIntent.UNRESOLVED,
    },
]

# 第二批: 15 个扩展深入测试用例 (英文、公式符号、对抗、多意图、空泛模糊)
SECOND_BATCH = [
    {
        "id": 11,
        "category": "英文正例 (RESOLVED)",
        "query": "What are the most common student questions in fractions?",
        "expected_status": "RESOLVED",
        "expected_intent": QueryIntent.STUDENT_FREQUENT_QUESTIONS,
    },
    {
        "id": 12,
        "category": "英文正例 (RESOLVED)",
        "query": "What tutor strategies recur most often in algebra?",
        "expected_status": "RESOLVED",
        "expected_intent": QueryIntent.TUTOR_STRATEGY_PATTERNS,
    },
    {
        "id": 13,
        "category": "英文几何单题 (RESOLVED)",
        "query": "How should a tutor distinguish volume from surface area for a cuboid?",
        "expected_status": "RESOLVED",
        "expected_intent": QueryIntent.QUESTION_TUTORING,
    },
    {
        "id": 14,
        "category": "公式符号单题 (RESOLVED)",
        "query": "(-2)^2 和 -2^2 为什么学生容易卡壳？",
        "expected_status": "RESOLVED",
        "expected_intent": QueryIntent.QUESTION_TUTORING,
    },
    {
        "id": 15,
        "category": "反例 (UNSUPPORTED - 最优教师)",
        "query": "统计哪个老师教学效果最好？",
        "expected_status": "UNSUPPORTED",
        "expected_intent": QueryIntent.UNSUPPORTED,
    },
    {
        "id": 16,
        "category": "反例 (UNSUPPORTED - 因果效果)",
        "query": "这个名师策略与学生成绩提高之间有没有因果效果？",
        "expected_status": "UNSUPPORTED",
        "expected_intent": QueryIntent.UNSUPPORTED,
    },
    {
        "id": 17,
        "category": "反例 (UNSUPPORTED - 最佳评价)",
        "query": "怎么才能评选最佳导师？",
        "expected_status": "UNSUPPORTED",
        "expected_intent": QueryIntent.UNSUPPORTED,
    },
    {
        "id": 18,
        "category": "三意图冲突 (NEED_LLM)",
        "query": "多次重考的学生在遇到分数最难理解的概念时，导师有什么策略？",
        "expected_status": "NEED_LLM",
        "expected_intent": QueryIntent.UNRESOLVED,
    },
    {
        "id": 19,
        "category": "泛词无强信号 (NEED_LLM)",
        "query": "这道题反映了什么？",
        "expected_status": "NEED_LLM",
        "expected_intent": QueryIntent.UNRESOLVED,
    },
    {
        "id": 20,
        "category": "模糊多视角提问 (NEED_LLM)",
        "query": "学生和导师在课堂上讨论了什么？",
        "expected_status": "NEED_LLM",
        "expected_intent": QueryIntent.UNRESOLVED,
    },
    {
        "id": 21,
        "category": "无强信号多主题 (NEED_LLM)",
        "query": "关于小数和负数，都有哪些模式？",
        "expected_status": "NEED_LLM",
        "expected_intent": QueryIntent.UNRESOLVED,
    },
    {
        "id": 22,
        "category": "共现跨课堂业务 (RESOLVED)",
        "query": "共现误区与教学策略共同出现的规律是什么？",
        "expected_status": "RESOLVED",
        "expected_intent": QueryIntent.CONTENT_IMPROVEMENT,
    },
    {
        "id": 23,
        "category": "干预业务 (RESOLVED)",
        "query": "围绕学生错因和导师策略如何干预？",
        "expected_status": "RESOLVED",
        "expected_intent": QueryIntent.CONTENT_IMPROVEMENT,
    },
    {
        "id": 24,
        "category": "具体数值单题 (RESOLVED)",
        "query": "学生在做关于 3.14159 的题目时如何引导？",
        "expected_status": "RESOLVED",
        "expected_intent": QueryIntent.QUESTION_TUTORING,
    },
    {
        "id": 25,
        "category": "反例变体 (UNSUPPORTED)",
        "query": "请告诉我谁教得最好",
        "expected_status": "UNSUPPORTED",
        "expected_intent": QueryIntent.UNSUPPORTED,
    },
]

VALID_STATUSES = {"RESOLVED", "NEED_LLM", "UNSUPPORTED"}

def run_suite(suite_name: str, cases: list) -> bool:
    print(f"\n{'='*70}")
    print(f"正在执行: {suite_name} (共 {len(cases)} 题)")
    print(f"{'='*70}")
    router = BusinessQueryRouter()
    all_passed = True
    observed_statuses = set()
    for case in cases:
        req = BusinessQueryRequest(query=case["query"])
        res = router.route(req)
        status = res.routing_status
        observed_statuses.add(status)
        intent = res.primary_intent
        candidates = [c.value for c in res.candidate_intents]

        is_valid_status = status in VALID_STATUSES
        status_match = (status == case["expected_status"])
        intent_match = (intent == case["expected_intent"])
        passed = is_valid_status and status_match and intent_match

        if not passed:
            all_passed = False
            mark = "FAIL"
        else:
            mark = "PASS"

        print(f"[{mark}] Case {case['id']:02d} | 类别: {case['category']}")
        print(f"       Query:    \"{case['query']}\"")
        print(f"       Status:   {status} (期望: {case['expected_status']})")
        print(f"       Intent:   {intent.value} (期望: {case['expected_intent'].value})")
        if candidates:
            print(f"       Candidates: {candidates}")
        if not is_valid_status:
            print(f"       [ERROR] 违背三态约束! 出现了非法状态: {status}")

    invalid_statuses = observed_statuses - VALID_STATUSES
    if invalid_statuses:
        print(f"\n[ALERT] 观察到了超出三种有效状态的值: {invalid_statuses}")
        return False
    else:
        print(f"\n[CONFIRM] 本批次实际观测到的状态全集为: {observed_statuses}，完全被约束在 3 种有效路由状态集合中！")

    return all_passed

if __name__ == "__main__":
    print("开始执行路由结果严格三态 (RESOLVED / NEED_LLM / UNSUPPORTED) 注入验证...")
    p1 = run_suite("第一阶段: 10 个典型正反例注入验证", FIRST_BATCH)
    if not p1:
        print("\n第一阶段验证未全部通过，终止执行后续注入。")
        sys.exit(1)
    print("\n第一阶段 10 个用例 100% 通过！且全部严格处于三种状态之内！")

    print("\n继续执行第二阶段扩展注入测试...")
    p2 = run_suite("第二阶段: 15 个多样化/对抗/多语言扩展注入验证", SECOND_BATCH)
    if not p2:
        print("\n第二阶段扩展注入未全部通过！")
        sys.exit(1)
    
    print("\n" + "="*70)
    print("全部 25 个注入测试（包含正例、反例、冲突待消歧例）100% 全部通过！")
    print("系统所有用例的 routing_status 严格只属于 {'RESOLVED', 'NEED_LLM', 'UNSUPPORTED'} 三种状态！")
    print("="*70)

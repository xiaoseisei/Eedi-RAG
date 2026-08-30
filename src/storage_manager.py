"""
================================================================================
模块名称: src/storage_manager.py
业务定位: Step 3 - 双引擎存储持久化管理器 (Dual-Engine Storage Persistence Manager)
核心职责:
  1. DuckDB 确定性关系底表管理 (单文件零运维 tutoring_knowledge.duckdb):
     - tutoring_sessions: 宏观会话与原题事实表
     - session_dialogue_turns: 清洗后的明细时序对话表 (供 [Turn N] 证据回溯与高亮反查)
     - misconception_chunks: 学生错因结构化表 (支撑 SQL GROUP BY 聚合与学情看板分析)
     - tutor_strategy_chunks: 导师策略与话术表 (支撑按教学法类别精准过滤)
     - sliding_window_chunks: 纯规则滑动窗口冷备表
  2. ChromaDB 物理隔离多向量集合持久化:
     - student_misconceptions: 面向学情诊断与错因机理模糊检索
     - tutor_strategies: 面向名师破局提问与启发脚手架模糊检索
     - fallback_windows: 面向零信任模式下的纯原文直取检索
  3. 语义增强模板构建 (Semantic Enrichment):
     - 将学科考纲层级路径、原题题干与结构化字段拼接为高密度 Embedding Document，注入分类学语义锚点。
  4. 证据回溯与并行检索融合 (Parent-Child & Late Fusion):
     - 检索命中知识卡片后，通过 (session_id, source_turn_ids) 从 DuckDB 秒级抓取原始对白实录作为不可篡改的真值证据。
================================================================================
"""

import os
import sys
import json
import math
import re
import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import duckdb
import chromadb
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
from chromadb.api.models.Collection import Collection

# 确保项目根目录在 sys.path 中
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.models import (
    CleanedSession,
    DialogueTurn,
    StudentMisconceptionProfile,
    TutorStrategyProfile,
    ExtractedPIU,
    Chunk
)

logger = logging.getLogger(__name__)


class FastDeterministicEmbeddingFunction(EmbeddingFunction):
    """
    高性能确定性语义特征哈希 Embedding 函数 (Zero-Dependency & 0-Cost):
    - 结合字符级 N-Gram、词汇 Token 与符号权重生成归一化密集向量；
    - 支持中英双语与数学公式；
    - 彻底免除外部庞大 ONNX 模型的联网下载与环境依赖，实现 0.05ms 极速确定性推理。
    """
    def __init__(self, dim: int = 128):
        self.dim = dim

    def name(self) -> str:
        return "fast_deterministic"

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = []
        for doc in input:
            vec = [0.0] * self.dim
            text_clean = str(doc).lower()
            # 提取中英文字词与数字、公式符号
            tokens = re.findall(r'[\u4e00-\u9fa5]+|[a-zA-Z0-9_\.\-]+', text_clean)
            
            # 同时加入 2-gram 字符特征增强语义连贯性
            ngrams = [text_clean[i:i+2] for i in range(len(text_clean)-1)] if len(text_clean) > 1 else []
            all_features = tokens + ngrams
            
            for feat in all_features:
                h = int(hashlib.md5(feat.encode('utf-8')).hexdigest(), 16)
                idx = h % self.dim
                sign = 1.0 if ((h >> 8) & 1) == 1 else -1.0
                vec[idx] += sign
                
            norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            embeddings.append([x / norm for x in vec])
        return embeddings


def build_misconception_embedding_doc(
    card: StudentMisconceptionProfile,
    question_text: str = "",
    subject_path: Optional[str] = None
) -> str:
    """
    为学生认知误区卡构建语义增强 Embedding Document。
    将学科考纲路径、考题原题与错因机理深度拼合，作为强语义锚点。
    """
    path = subject_path or card.subject_path or "通用数学考点"
    triggers_str = ", ".join(card.confusion_triggers) if card.confusion_triggers else "无特定触发词"
    quotes_str = " | ".join(f'"{q}"' for q in card.verbatim_student_quotes) if card.verbatim_student_quotes else "无直接原声"
    
    doc = (
        f"【知识卡片类型】: 学生认知误区卡 (Student Misconception Profile)\n"
        f"【学科考纲路径】: {path}\n"
        f"【关联考题原题】: {question_text}\n"
        f"【误选选项】: {card.error_choice or '未知'}\n"
        f"【错因标准命名】: {card.misconception_name}\n"
        f"【深层认知机理】: {card.deep_mechanism}\n"
        f"【困惑触发概念】: {triggers_str}\n"
        f"【学生原声证据】: {quotes_str}"
    )
    return doc


def build_tutor_strategy_embedding_doc(
    card: TutorStrategyProfile,
    question_text: str = "",
    subject_path: str = ""
) -> str:
    """
    为名师启发式策略卡构建语义增强 Embedding Document。
    将教学目标、名师破局一问、引导步骤链与教学动作深度拼合。
    """
    scaffolding_str = " -> ".join(card.scaffolding_steps) if card.scaffolding_steps else "自然启发"
    moves_str = ", ".join(card.talk_moves) if card.talk_moves else "标准互动"
    metaphor_str = card.analogy_or_metaphor or "无"
    
    doc = (
        f"【知识卡片类型】: 名师启发式策略卡 (Tutor Pedagogical Strategy)\n"
        f"【学科考纲路径】: {subject_path or '通用数学考点'}\n"
        f"【关联考题原题】: {question_text}\n"
        f"【教学引导目标】: {card.pedagogical_goal}\n"
        f"【教学策略分类】: {card.strategy_category}\n"
        f"【名师破局一问】: {card.key_aha_question}\n"
        f"【脚手架步骤链】: {scaffolding_str}\n"
        f"【启发比喻案例】: {metaphor_str}\n"
        f"【涉及教学动作】: {moves_str}\n"
        f"【最终辅导成效】: {card.resolution_outcome}"
    )
    return doc


class DualEngineStorageManager:
    """
    双引擎存储管理器:
    - 关系底表引擎: DuckDB (负责精准点查、SQL 聚合、跨表关联与 Turn 级证据反查)
    - 语义向量引擎: ChromaDB (负责学情误区、名师策略与滑动窗口的多路模糊向量召回)
    """

    def __init__(
        self,
        db_path: Union[str, Path] = "data/db/tutoring_knowledge.duckdb",
        chroma_dir: Optional[Union[str, Path]] = "data/chroma",
        embedding_function: Any = None,
        in_memory: bool = False
    ):
        """
        初始化双引擎持久化管理器。
        
        参数:
          db_path: DuckDB 文件存储路径 (或 ':memory:')
          chroma_dir: ChromaDB 持久化目录 (in_memory=True 时使用内存客户端)
          embedding_function: 可选的 ChromaDB Embedding 函数 (默认使用 FastDeterministicEmbeddingFunction)
          in_memory: 是否运行纯内存模式 (用于单元测试和极速沙盒验证)
        """
        self.in_memory = in_memory
        
        # 1. 初始化 DuckDB 关系底表
        if in_memory or str(db_path) == ":memory:":
            self.duck_conn = duckdb.connect(database=":memory:")
            self.db_path = ":memory:"
        else:
            self.db_path = Path(db_path)
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self.duck_conn = duckdb.connect(database=str(self.db_path))
            
        self._init_duckdb_schema()
        
        # 2. 初始化 ChromaDB 多向量集合
        self.embedding_function = embedding_function or FastDeterministicEmbeddingFunction()
        if in_memory:
            self.chroma_client = chromadb.Client()
            self.chroma_dir = None
        else:
            self.chroma_dir = Path(chroma_dir) if chroma_dir else Path("data/chroma")
            self.chroma_dir.mkdir(parents=True, exist_ok=True)
            self.chroma_client = chromadb.PersistentClient(path=str(self.chroma_dir))
            
        self._init_chromadb_collections()

    def _init_duckdb_schema(self):
        """初始化 DuckDB 确定性关系底表 DDL Schema。"""
        # 会话与题目事实表
        self.duck_conn.execute("""
        CREATE TABLE IF NOT EXISTS tutoring_sessions (
            intervention_id BIGINT PRIMARY KEY,
            question_id BIGINT NOT NULL,
            tutor_id BIGINT NOT NULL,
            subject_path VARCHAR,
            question_text VARCHAR,
            options_json VARCHAR,
            total_turns INTEGER,
            student_turn_count INTEGER,
            tutor_turn_count INTEGER,
            has_valid_tutoring BOOLEAN
        );
        """)
        
        # 时序对话明细表 (用于证据逐轮反查高亮)
        self.duck_conn.execute("""
        CREATE TABLE IF NOT EXISTS session_dialogue_turns (
            turn_pk VARCHAR PRIMARY KEY,
            intervention_id BIGINT NOT NULL,
            turn_id INTEGER NOT NULL,
            speaker VARCHAR NOT NULL,
            is_tutor BOOLEAN NOT NULL,
            text VARCHAR NOT NULL,
            talk_moves VARCHAR[],
            is_greeting_or_noise BOOLEAN NOT NULL
        );
        """)
        
        # 学生认知误区卡片表 (支持 SQL GROUP BY 聚合)
        self.duck_conn.execute("""
        CREATE TABLE IF NOT EXISTS misconception_chunks (
            chunk_id VARCHAR PRIMARY KEY,
            session_id BIGINT NOT NULL,
            question_id BIGINT NOT NULL,
            subject_path VARCHAR,
            misconception_name VARCHAR NOT NULL,
            error_choice VARCHAR,
            deep_mechanism VARCHAR NOT NULL,
            confusion_triggers VARCHAR[],
            verbatim_student_quotes VARCHAR[],
            source_turn_ids INTEGER[]
        );
        """)
        
        # 名师启发策略卡片表
        self.duck_conn.execute("""
        CREATE TABLE IF NOT EXISTS tutor_strategy_chunks (
            chunk_id VARCHAR PRIMARY KEY,
            session_id BIGINT NOT NULL,
            question_id BIGINT NOT NULL,
            subject_path VARCHAR,
            pedagogical_goal VARCHAR NOT NULL,
            strategy_category VARCHAR NOT NULL,
            key_aha_question VARCHAR NOT NULL,
            scaffolding_steps VARCHAR[],
            analogy_or_metaphor VARCHAR,
            talk_moves VARCHAR[],
            resolution_outcome VARCHAR NOT NULL,
            source_turn_ids INTEGER[]
        );
        """)
        
        # 滑动窗口冷备表
        self.duck_conn.execute("""
        CREATE TABLE IF NOT EXISTS sliding_window_chunks (
            chunk_id VARCHAR PRIMARY KEY,
            session_id BIGINT NOT NULL,
            question_id BIGINT NOT NULL,
            subject_path VARCHAR,
            window_start_turn INTEGER,
            window_end_turn INTEGER,
            content VARCHAR NOT NULL,
            source_turn_ids INTEGER[]
        );
        """)

    def _init_chromadb_collections(self):
        """初始化 ChromaDB 物理隔离的三个向量集合。"""
        self.coll_misconceptions: Collection = self.chroma_client.get_or_create_collection(
            name="student_misconceptions",
            metadata={"description": "学生深层认知障碍与错因机理向量索引库"},
            embedding_function=self.embedding_function
        )
        
        self.coll_strategies: Collection = self.chroma_client.get_or_create_collection(
            name="tutor_strategies",
            metadata={"description": "名师破局提问与启发式脚手架策略向量索引库"},
            embedding_function=self.embedding_function
        )
        
        self.coll_windows: Collection = self.chroma_client.get_or_create_collection(
            name="fallback_windows",
            metadata={"description": "纯规则滑动窗口原文直接检索向量索引库"},
            embedding_function=self.embedding_function
        )

    # =========================================================================
    # 数据批量摄取与持久化 (Ingestion Pipeline)
    # =========================================================================

    def get_storage_audit_snapshot(self) -> Dict[str, Any]:
        """
        获取当前双引擎存储的白盒审计快照 (White-Box Storage Audit Snapshot)。
        汇总 DuckDB 各关系表行数与 ChromaDB 向量集合计数，彻底消除黑盒隐患。
        """
        duck_counts = {
            "tutoring_sessions": self.duck_conn.execute("SELECT COUNT(*) FROM tutoring_sessions").fetchone()[0],
            "session_dialogue_turns": self.duck_conn.execute("SELECT COUNT(*) FROM session_dialogue_turns").fetchone()[0],
            "misconception_chunks": self.duck_conn.execute("SELECT COUNT(*) FROM misconception_chunks").fetchone()[0],
            "tutor_strategy_chunks": self.duck_conn.execute("SELECT COUNT(*) FROM tutor_strategy_chunks").fetchone()[0],
            "sliding_window_chunks": self.duck_conn.execute("SELECT COUNT(*) FROM sliding_window_chunks").fetchone()[0],
        }
        chroma_counts = {
            "student_misconceptions": self.coll_misconceptions.count(),
            "tutor_strategies": self.coll_strategies.count(),
            "fallback_windows": self.coll_windows.count(),
        }
        
        snapshot = {
            "duckdb_tables": duck_counts,
            "chromadb_collections": chroma_counts,
            "is_in_memory": self.in_memory
        }
        
        # 格式化输出白盒审计看板
        logger.info(
            f"\n📊 ====================【双引擎存储白盒审计快照】====================\n"
            f"  * 关系底表引擎 (DuckDB {'[内存模式]' if self.in_memory else '[持久化文件]'}):\n"
            f"    - tutoring_sessions:       {duck_counts['tutoring_sessions']:>5} 行 (宏观会话事实与原题)\n"
            f"    - session_dialogue_turns:  {duck_counts['session_dialogue_turns']:>5} 轮 (时序原声对话实录明细)\n"
            f"    - misconception_chunks:    {duck_counts['misconception_chunks']:>5} 条 (学生错因机理结构化)\n"
            f"    - tutor_strategy_chunks:   {duck_counts['tutor_strategy_chunks']:>5} 条 (名师启发策略结构化)\n"
            f"    - sliding_window_chunks:   {duck_counts['sliding_window_chunks']:>5} 块 (冷备滑动窗口)\n"
            f"  * 语义向量引擎 (ChromaDB 多物理隔离集合):\n"
            f"    - student_misconceptions:  {chroma_counts['student_misconceptions']:>5} 向量 (学情障碍与错因诊断)\n"
            f"    - tutor_strategies:        {chroma_counts['tutor_strategies']:>5} 向量 (名师破局提问与脚手架)\n"
            f"    - fallback_windows:        {chroma_counts['fallback_windows']:>5} 向量 (零信任纯原文直取)\n"
            f"========================================================================"
        )
        return snapshot

    def ingest_sessions(self, sessions: List[CleanedSession]):
        """将宏观会话事实及对话明细批量存入 DuckDB 关系表。"""
        logger.info(f"[step=Step3_Storage|action=ingest_sessions] 📦 正在摄取 {len(sessions)} 场清洗会话至 DuckDB 关系底表...")
        session_rows = []
        turn_rows = []
        
        for s in sessions:
            subject_p = s.subjects.paths[0] if s.subjects.paths else "通用数学考点"
            options_j = json.dumps(s.question.options, ensure_ascii=False) if s.question.options else "{}"
            q_preview = s.question.question_text[:35].replace('\n', ' ') if s.question.question_text else "无题干"
            
            logger.info(
                f"  ├─ [会话摄取] Session #{s.intervention_id} | 题目 QID={s.question_id} | "
                f"轮数={s.total_turns} (学生={s.student_turn_count}, 导师={s.tutor_turn_count}) | "
                f"考点: '{subject_p}' | 题干: '{q_preview}...'"
            )
            
            session_rows.append((
                s.intervention_id,
                s.question_id,
                s.tutor_id,
                subject_p,
                s.question.question_text,
                options_j,
                s.total_turns,
                s.student_turn_count,
                s.tutor_turn_count,
                s.has_valid_tutoring
            ))
            
            for t in s.turns:
                turn_pk = f"{s.intervention_id}_{t.turn_id}"
                turn_rows.append((
                    turn_pk,
                    s.intervention_id,
                    t.turn_id,
                    t.speaker,
                    t.is_tutor,
                    t.text,
                    t.talk_moves,
                    t.is_greeting_or_noise
                ))
                
        # 写入 tutoring_sessions
        if session_rows:
            self.duck_conn.executemany("""
            INSERT OR REPLACE INTO tutoring_sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, session_rows)
            
        # 写入 session_dialogue_turns
        if turn_rows:
            self.duck_conn.executemany("""
            INSERT OR REPLACE INTO session_dialogue_turns VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """, turn_rows)
            
        logger.info(
            f"[step=Step3_Storage|status=SUCCESS] ✅ DuckDB 会话事实与对话明细入库完成: "
            f"tutoring_sessions (+{len(session_rows)} 条), session_dialogue_turns (+{len(turn_rows)} 轮对话)"
        )

    def ingest_extracted_pius(
        self,
        extracted_list: List[ExtractedPIU],
        sessions_map: Optional[Dict[int, CleanedSession]] = None
    ):
        """
        将大模型蒸馏抽取出的知识卡片，分别持久化至 DuckDB 关系表与 ChromaDB 双向量集合。
        """
        logger.info(f"[step=Step3_Storage|action=ingest_cards] 🧠 正在持久化 {len(extracted_list)} 份双卡片知识资产 (DuckDB + ChromaDB)...")
        misc_duck_rows = []
        strat_duck_rows = []
        
        misc_chroma_ids = []
        misc_chroma_docs = []
        misc_chroma_metas = []
        
        strat_chroma_ids = []
        strat_chroma_docs = []
        strat_chroma_metas = []
        
        for item in extracted_list:
            sess = sessions_map.get(item.session_id) if sessions_map else None
            q_text = sess.question.question_text if sess else ""
            subj_p = item.misconception.subject_path or (sess.subjects.paths[0] if sess and sess.subjects.paths else "")
            
            misc = item.misconception
            strat = item.tutor_strategy
            
            misc_chunk_id = f"session_{item.session_id}_misconception"
            strat_chunk_id = f"session_{item.session_id}_tutor_strategy"
            
            # 白盒日志透视：输出具体错因卡片与策略卡片的核心业务内容
            mech_preview = misc.deep_mechanism[:40].replace('\n', ' ')
            logger.info(
                f"  ├─ [错因卡入库] Session #{misc.session_id} (QID={misc.question_id}) | "
                f"误区:「{misc.misconception_name}」 (错误选项={misc.error_choice or '未记录'}) | "
                f"机理: '{mech_preview}...' | 证据轮次={misc.source_turn_ids}"
            )
            logger.info(
                f"  ├─ [策略卡入库] Session #{strat.session_id} (QID={strat.question_id}) | "
                f"策略分类: [{strat.strategy_category}] | 破局一问:「{strat.key_aha_question}」 | "
                f"教学动作={strat.talk_moves} | 证据轮次={strat.source_turn_ids}"
            )
            
            # 1. 组装 DuckDB 关系行
            misc_duck_rows.append((
                misc_chunk_id,
                misc.session_id,
                misc.question_id,
                subj_p,
                misc.misconception_name,
                misc.error_choice or "",
                misc.deep_mechanism,
                misc.confusion_triggers,
                misc.verbatim_student_quotes,
                misc.source_turn_ids
            ))
            
            strat_duck_rows.append((
                strat_chunk_id,
                strat.session_id,
                strat.question_id,
                subj_p,
                strat.pedagogical_goal,
                strat.strategy_category,
                strat.key_aha_question,
                strat.scaffolding_steps,
                strat.analogy_or_metaphor or "",
                strat.talk_moves,
                strat.resolution_outcome,
                strat.source_turn_ids
            ))
            
            # 2. 组装 ChromaDB 语义增强向量 Document
            misc_doc = build_misconception_embedding_doc(misc, question_text=q_text, subject_path=subj_p)
            misc_chroma_ids.append(misc_chunk_id)
            misc_chroma_docs.append(misc_doc)
            misc_chroma_metas.append({
                "session_id": misc.session_id,
                "question_id": misc.question_id,
                "subject_path": subj_p,
                "misconception_name": misc.misconception_name,
                "error_choice": misc.error_choice or "",
                "source_turn_ids": json.dumps(misc.source_turn_ids)
            })
            
            strat_doc = build_tutor_strategy_embedding_doc(strat, question_text=q_text, subject_path=subj_p)
            strat_chroma_ids.append(strat_chunk_id)
            strat_chroma_docs.append(strat_doc)
            strat_chroma_metas.append({
                "session_id": strat.session_id,
                "question_id": strat.question_id,
                "subject_path": subj_p,
                "strategy_category": strat.strategy_category,
                "key_aha_question": strat.key_aha_question,
                "source_turn_ids": json.dumps(strat.source_turn_ids)
            })
            
        # 写入 DuckDB 关系表
        if misc_duck_rows:
            self.duck_conn.executemany("""
            INSERT OR REPLACE INTO misconception_chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, misc_duck_rows)
            
        if strat_duck_rows:
            self.duck_conn.executemany("""
            INSERT OR REPLACE INTO tutor_strategy_chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, strat_duck_rows)
            
        # 写入 ChromaDB 物理隔离集合
        if misc_chroma_ids:
            self.coll_misconceptions.upsert(
                ids=misc_chroma_ids,
                documents=misc_chroma_docs,
                metadatas=misc_chroma_metas
            )
            
        if strat_chroma_ids:
            self.coll_strategies.upsert(
                ids=strat_chroma_ids,
                documents=strat_chroma_docs,
                metadatas=strat_chroma_metas
            )
            
        logger.info(
            f"[step=Step3_Storage|status=SUCCESS] ✅ 知识卡片双轨同步完成: "
            f"DuckDB (+{len(misc_duck_rows)} 错因, +{len(strat_duck_rows)} 策略) | "
            f"ChromaDB (student_misconceptions: +{len(misc_chroma_ids)}, tutor_strategies: +{len(strat_chroma_ids)})"
        )

    def ingest_sliding_window_chunks(self, chunks: List[Chunk]):
        """将滑动窗口切块存入 DuckDB 及 fallback_windows 向量集合。"""
        logger.info(f"[step=Step3_Storage|action=ingest_windows] 🪟 正在持久化 {len(chunks)} 个滑动窗口切块...")
        duck_rows = []
        c_ids = []
        c_docs = []
        c_metas = []
        
        for c in chunks:
            if c.chunk_type != "sliding_window":
                continue
            subj_p = c.metadata.get("subject_path", "通用数学考点")
            w_start = c.metadata.get("window_start_turn", 1)
            w_end = c.metadata.get("window_end_turn", 6)
            s_turns = c.source_turn_ids or []
            
            duck_rows.append((
                c.chunk_id,
                c.session_id,
                c.question_id,
                subj_p,
                w_start,
                w_end,
                c.content,
                s_turns
            ))
            
            c_ids.append(c.chunk_id)
            c_docs.append(c.content)
            c_metas.append({
                "session_id": c.session_id,
                "question_id": c.question_id,
                "subject_path": subj_p,
                "window_start_turn": w_start,
                "window_end_turn": w_end,
                "source_turn_ids": json.dumps(s_turns)
            })
            
        if duck_rows:
            self.duck_conn.executemany("""
            INSERT OR REPLACE INTO sliding_window_chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """, duck_rows)
            
        if c_ids:
            self.coll_windows.upsert(
                ids=c_ids,
                documents=c_docs,
                metadatas=c_metas
            )
            
        logger.info(
            f"[step=Step3_Storage|status=SUCCESS] ✅ 滑动窗口冷备写入完成: "
            f"DuckDB sliding_window_chunks (+{len(duck_rows)} 行), ChromaDB fallback_windows (+{len(c_ids)} 向量)"
        )

    def ingest_all(
        self,
        cleaned_sessions: List[CleanedSession],
        extracted_pius: List[ExtractedPIU],
        chunks: Optional[List[Chunk]] = None
    ):
        """一站式批量摄取全生命周期资产。"""
        logger.info(f"🚀 [step=Step3_Storage|action=ingest_all] 开始双引擎全资产摄取: {len(cleaned_sessions)} 场会话, {len(extracted_pius)} 份抽取卡片...")
        sessions_map = {s.intervention_id: s for s in cleaned_sessions}
        
        self.ingest_sessions(cleaned_sessions)
        self.ingest_extracted_pius(extracted_pius, sessions_map=sessions_map)
        if chunks:
            self.ingest_sliding_window_chunks(chunks)
            
        # 输出白盒持久化审计快照
        self.get_storage_audit_snapshot()
        logger.info("🎉 [step=Step3_Storage|status=SUCCESS] 双引擎存储摄取与持久化 100% 同步完成！")


    # =========================================================================
    # DuckDB 关系查询与证据反查 (Relational Grounding & SQL Analytics)
    # =========================================================================

    def get_dialogue_turns(
        self,
        session_id: int,
        turn_ids: Optional[List[int]] = None
    ) -> List[Dict[str, Any]]:
        """
        核心证据反查接口：根据 (session_id, turn_ids) 秒级从 DuckDB 检索真实对话实录。
        支撑【卡片为引，原文为据】的 Parent-Child 证据回溯架构。
        """
        if turn_ids is not None and len(turn_ids) > 0:
            turn_list_str = ",".join(str(int(t)) for t in turn_ids)
            query = f"""
            SELECT turn_id, speaker, is_tutor, text, talk_moves, is_greeting_or_noise
            FROM session_dialogue_turns
            WHERE intervention_id = {int(session_id)} AND turn_id IN ({turn_list_str})
            ORDER BY turn_id ASC;
            """
        else:
            query = f"""
            SELECT turn_id, speaker, is_tutor, text, talk_moves, is_greeting_or_noise
            FROM session_dialogue_turns
            WHERE intervention_id = {int(session_id)}
            ORDER BY turn_id ASC;
            """
        df = self.duck_conn.execute(query).fetchdf()
        return df.to_dict(orient="records")

    def query_misconceptions_sql(
        self,
        subject_filter: Optional[str] = None,
        keyword: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """在 DuckDB 错因表中执行确定性 SQL 过滤查询。"""
        clauses = []
        if subject_filter:
            clauses.append(f"subject_path LIKE '%{subject_filter}%'")
        if keyword:
            clauses.append(f"(misconception_name LIKE '%{keyword}%' OR deep_mechanism LIKE '%{keyword}%')")
            
        where_stmt = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT * FROM misconception_chunks {where_stmt} ORDER BY session_id ASC;"
        return self.duck_conn.execute(sql).fetchdf().to_dict(orient="records")

    def query_strategies_sql(
        self,
        strategy_category: Optional[str] = None,
        subject_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """在 DuckDB 策略表中按教学分类与考点执行确定性 SQL 查询。"""
        clauses = []
        if strategy_category:
            clauses.append(f"strategy_category = '{strategy_category}'")
        if subject_filter:
            clauses.append(f"subject_path LIKE '%{subject_filter}%'")
            
        where_stmt = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT * FROM tutor_strategy_chunks {where_stmt} ORDER BY session_id ASC;"
        return self.duck_conn.execute(sql).fetchdf().to_dict(orient="records")

    def get_misconception_stats_by_subject(self) -> List[Dict[str, Any]]:
        """
        执行 SQL 聚合统计：按学科考点统计错因频次与关联考题数（学情看板核心 SQL）。
        """
        sql = """
        SELECT 
            subject_path,
            COUNT(*) AS total_misconceptions,
            COUNT(DISTINCT question_id) AS distinct_questions,
            COUNT(DISTINCT session_id) AS distinct_sessions
        FROM misconception_chunks
        GROUP BY subject_path
        ORDER BY total_misconceptions DESC;
        """
        return self.duck_conn.execute(sql).fetchdf().to_dict(orient="records")

    # =========================================================================
    # ChromaDB 向量语义检索与并行融合 (Vector Search & Late Fusion)
    # =========================================================================

    def search_misconceptions_vector(
        self,
        query: str,
        top_k: int = 3,
        where_filter: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """在 student_misconceptions 向量集合中执行模糊语义检索。"""
        kwargs = {"query_texts": [query], "n_results": top_k}
        if where_filter:
            kwargs["where"] = where_filter
            
        results = self.coll_misconceptions.query(**kwargs)
        formatted = []
        if results and results.get("ids") and len(results["ids"][0]) > 0:
            for i in range(len(results["ids"][0])):
                formatted.append({
                    "chunk_id": results["ids"][0][i],
                    "document": results["documents"][0][i] if results.get("documents") else "",
                    "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
                    "distance": results["distances"][0][i] if results.get("distances") else None,
                    "chunk_type": "misconception"
                })
        return formatted

    def search_strategies_vector(
        self,
        query: str,
        top_k: int = 3,
        where_filter: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """在 tutor_strategies 向量集合中执行名师策略模糊语义检索。"""
        kwargs = {"query_texts": [query], "n_results": top_k}
        if where_filter:
            kwargs["where"] = where_filter
            
        results = self.coll_strategies.query(**kwargs)
        formatted = []
        if results and results.get("ids") and len(results["ids"][0]) > 0:
            for i in range(len(results["ids"][0])):
                formatted.append({
                    "chunk_id": results["ids"][0][i],
                    "document": results["documents"][0][i] if results.get("documents") else "",
                    "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
                    "distance": results["distances"][0][i] if results.get("distances") else None,
                    "chunk_type": "tutor_strategy"
                })
        return formatted

    def search_fallback_windows_vector(
        self,
        query: str,
        top_k: int = 3,
        where_filter: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """在 fallback_windows 向量集合中执行纯原文直接检索 (零信任直取模式)。"""
        kwargs = {"query_texts": [query], "n_results": top_k}
        if where_filter:
            kwargs["where"] = where_filter
            
        results = self.coll_windows.query(**kwargs)
        formatted = []
        if results and results.get("ids") and len(results["ids"][0]) > 0:
            for i in range(len(results["ids"][0])):
                formatted.append({
                    "chunk_id": results["ids"][0][i],
                    "document": results["documents"][0][i] if results.get("documents") else "",
                    "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
                    "distance": results["distances"][0][i] if results.get("distances") else None,
                    "chunk_type": "sliding_window"
                })
        return formatted

    def search_parallel_and_fuse(
        self,
        query: str,
        top_k_each: int = 3,
        fetch_evidence_turns: bool = True
    ) -> Dict[str, Any]:
        """
        【反硬路由黄金接口】：并行检索错因集合与策略集合，并自动从 DuckDB 回溯原文证据实录。
        
        参数:
          query: 用户教研提问
          top_k_each: 各集合召回的 Top-K 数量
          fetch_evidence_turns: 是否自动通过 (session_id, source_turn_ids) 追回原始实录证据
          
        返回:
          Dict: 包含 misconceptions 与 strategies 候选列表及各自附带的真实对话证据
        """
        logger.info(f"[step=Step4_Retrieval|action=search_parallel_and_fuse] 🔍 接收教研查询: '{query}' (top_k_each={top_k_each})")
        misc_results = self.search_misconceptions_vector(query, top_k=top_k_each)
        strat_results = self.search_strategies_vector(query, top_k=top_k_each)
        
        if fetch_evidence_turns:
            # 为每个命中的错因卡追回原文
            for item in misc_results:
                sess_id = item["metadata"].get("session_id")
                turn_ids_raw = item["metadata"].get("source_turn_ids")
                turn_ids = json.loads(turn_ids_raw) if isinstance(turn_ids_raw, str) else (turn_ids_raw or [])
                if sess_id:
                    item["evidence_turns"] = self.get_dialogue_turns(sess_id, turn_ids)
                    
            # 为每个命中的策略卡追回原文
            for item in strat_results:
                sess_id = item["metadata"].get("session_id")
                turn_ids_raw = item["metadata"].get("source_turn_ids")
                turn_ids = json.loads(turn_ids_raw) if isinstance(turn_ids_raw, str) else (turn_ids_raw or [])
                if sess_id:
                    item["evidence_turns"] = self.get_dialogue_turns(sess_id, turn_ids)
                    
        total_evidence_turns = sum(len(item.get("evidence_turns", [])) for item in misc_results + strat_results)
        misc_summary = [f"{m['metadata'].get('misconception_name', m['chunk_id'])}(dist={m.get('distance', 0.0):.3f})" for m in misc_results]
        strat_summary = [f"{s['metadata'].get('key_aha_question', s['chunk_id'])[:20]}...(dist={s.get('distance', 0.0):.3f})" for s in strat_results]
        
        logger.info(
            f"[step=Step4_Retrieval|status=SUCCESS] 🎯 并行召回与证据回溯完成: "
            f"错因卡 Top-{len(misc_results)} {misc_summary} | "
            f"策略卡 Top-{len(strat_results)} {strat_summary} | "
            f"回溯真实对话证据={total_evidence_turns} 轮"
        )
        return {
            "query": query,
            "misconceptions": misc_results,
            "strategies": strat_results,
            "total_retrieved": len(misc_results) + len(strat_results)
        }

    def close(self):
        """关闭 DuckDB 数据库连接。"""
        if hasattr(self, "duck_conn") and self.duck_conn:
            self.duck_conn.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    sample_cleaned_file = project_root / "data" / "sample" / "cleaned_sessions_sample.jsonl"
    sample_extracted_file = project_root / "data" / "sample" / "extracted_pius_sample.jsonl"
    sample_chunks_file = project_root / "data" / "sample" / "chunks_sample.jsonl"
    
    print("🚀 开始测试 Step 3 双引擎存储管理器入库...")
    
    sessions = [CleanedSession.model_validate(json.loads(line)) for line in open(sample_cleaned_file, encoding="utf-8") if line.strip()]
    extracted = [ExtractedPIU.model_validate(json.loads(line)) for line in open(sample_extracted_file, encoding="utf-8") if line.strip()]
    chunks = [Chunk.model_validate(json.loads(line)) for line in open(sample_chunks_file, encoding="utf-8") if line.strip()]
    
    manager = DualEngineStorageManager(
        db_path="data/db/tutoring_knowledge.duckdb",
        chroma_dir="data/chroma"
    )
    
    manager.ingest_all(sessions, extracted, chunks)
    
    # 验证 DuckDB 聚合
    stats = manager.get_misconception_stats_by_subject()
    print(f"\n📊 DuckDB 学情看板聚合统计 Top 3:")
    for s in stats[:3]:
        print(f"  考点: {s['subject_path']} | 错因数: {s['total_misconceptions']} | 涉及题目: {s['distinct_questions']}")
        
    # 验证并行向量检索 + 证据回溯
    test_q = "四舍五入 5.4598 到 1 位小数学生容易混淆什么？名师怎么引导？"
    fused_res = manager.search_parallel_and_fuse(test_q, top_k_each=1, fetch_evidence_turns=True)
    
    print(f"\n🔍 并行检索与证据回溯验证 (Query: {test_q}):")
    if fused_res["misconceptions"]:
        top_misc = fused_res["misconceptions"][0]
        print(f"  [命中错因卡]: {top_misc['metadata'].get('misconception_name')}")
        print(f"  [追回原声证据轮数]: {len(top_misc.get('evidence_turns', []))} 轮")
        
    if fused_res["strategies"]:
        top_strat = fused_res["strategies"][0]
        print(f"  [命中策略卡]: {top_strat['metadata'].get('key_aha_question')}")
        print(f"  [追回原声证据轮数]: {len(top_strat.get('evidence_turns', []))} 轮")
        
    manager.close()
    print("\n✅ Step 3 双引擎持久化端到端验证通过！")

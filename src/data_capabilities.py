"""
================================================================================
模块名称: src/data_capabilities.py
业务定位: Eedi-RAG 底层数据能力探测与查询前置门禁 (Data capability gates)

本文件不负责创建业务表、填充分析数据或推断缺失字段, 只回答一个严格问题:
"当前实际打开的数据资产, 是否具备支撑某类业务查询的结构条件?"

能力边界:
  1. DIALOGUE_EVIDENCE —— 权威 DuckDB 是否包含可回查的 Session/Turn 原文;
  2. BUSINESS_ANALYTICS —— analytics sidecar 是否存在版本化分析资产;
  3. RETAKE_COHORT —— 是否存在脱敏学生键与明确的重考次数;
  4. ABILITY_COHORT —— 是否存在脱敏学生键、历史题量与正确率指标。

核心设计原则:
  - 只读探测: 仅查询 information_schema, 绝不为缺失能力自动建表或写默认值;
  - 显式失败: require() 遇到缺失能力抛出带 capability / missing_requirements 的异常;
  - 真实状态: available、coverage 与缺失字段保持一致, 不把空结构伪装成可用业务数据;
  - 生产与测试隔离: synthetic_snapshot() 仅供单元测试构造快照, 不读取或修改生产资产。
================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from src.business_models import DataCapabilitySnapshot


# ---------------------------------------------------------------------------
# 能力需求目录: 每项能力声明数据归属与最小 schema 要求
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CapabilityRequirement:
    """单项数据能力的最小结构要求, 供 inspect() 逐项执行只读检查。"""

    capability: str                         # 对外稳定能力名
    connection_name: Literal["source", "analytics"]  # 应检查的连接
    table: str                              # 必须存在的表名
    columns: frozenset[str]                 # 必须存在的字段集合


REQUIREMENTS: dict[str, CapabilityRequirement] = {
    # 权威原始对话能力: 来自生产源 DuckDB, 不依赖分析 sidecar。
    "DIALOGUE_EVIDENCE": CapabilityRequirement(
        capability="DIALOGUE_EVIDENCE",
        connection_name="source",
        table="session_dialogue_turns",
        columns=frozenset({"intervention_id", "turn_id", "speaker", "text"}),
    ),
    # 宏观业务分析能力: 由 Part 2 生成的版本化 sidecar 提供。
    "BUSINESS_ANALYTICS": CapabilityRequirement(
        capability="BUSINESS_ANALYTICS",
        connection_name="analytics",
        table="business_analytics_manifest",
        columns=frozenset({"source_artifact_hash", "analytics_version", "build_status"}),
    ),
    # 重考群体能力: 没有稳定学生键与尝试次数字段就不能执行。
    "RETAKE_COHORT": CapabilityRequirement(
        capability="RETAKE_COHORT",
        connection_name="analytics",
        table="student_session_features",
        columns=frozenset({"session_id", "student_key", "attempt_number", "measured_at"}),
    ),
    # 基础能力分层: 没有历史题量与正确率就不能声称"基础薄弱"。
    "ABILITY_COHORT": CapabilityRequirement(
        capability="ABILITY_COHORT",
        connection_name="analytics",
        table="student_session_features",
        columns=frozenset(
            {
                "session_id",
                "student_key",
                "historic_question_count",
                "historic_correctness",
                "measured_at",
            }
        ),
    ),
}

# Part 1 计划曾使用 ``status`` 命名, Part 2 sidecar DDL 使用更明确的
# ``build_status`` 命名。两者只允许作为同一个 manifest 状态字段的兼容别名;
# 其他分析版本字段仍必须全部存在, 不能借别名绕过 schema 门禁。
_COLUMN_ALIASES: dict[str, frozenset[str]] = {
    "build_status": frozenset({"build_status", "status"}),
}


class MissingDataCapabilityError(RuntimeError):
    """调用方要求的业务数据能力不存在时的显式门禁异常。"""

    def __init__(self, capability: str, missing_requirements: list[str]) -> None:
        self.capability = capability                         # 缺失的稳定能力名
        self.missing_requirements = list(missing_requirements)  # 缺失表/字段/连接
        details = ", ".join(self.missing_requirements) or "unknown requirement"
        super().__init__(f"missing data capability {capability}: {details}")


class DataCapabilityRegistry:
    """
    数据能力注册表。

    source_conn 必须是权威源 DuckDB 连接; analytics_conn 是可选的分析 sidecar
    连接。构造函数不执行写操作, inspect() 每次按当前 schema 重新读取真实状态。
    """

    def __init__(
        self,
        *,
        source_conn: Any,
        analytics_conn: Any | None = None,
        source_artifact: str = "source-duckdb",
        analytics_artifact: str = "analytics-sidecar",
    ) -> None:
        if source_conn is None:
            raise ValueError("source_conn must be provided for capability inspection")
        if not str(source_artifact).strip():
            raise ValueError("source_artifact must be non-blank")
        if not str(analytics_artifact).strip():
            raise ValueError("analytics_artifact must be non-blank")
        self.source_conn = source_conn
        self.analytics_conn = analytics_conn
        self.source_artifact = str(source_artifact).strip()
        self.analytics_artifact = str(analytics_artifact).strip()

    @staticmethod
    def _table_columns(connection: Any, table: str) -> set[str]:
        """只读查询 information_schema, 返回指定表的标准化小写字段名集合。"""

        rows = connection.execute(
            """
            SELECT lower(column_name)
            FROM information_schema.columns
            WHERE lower(table_name) = lower(?)
            """,
            [table],
        ).fetchall()
        return {str(row[0]).casefold() for row in rows if row and row[0] is not None}

    @staticmethod
    def _table_exists(connection: Any, table: str) -> bool:
        """只读查询 information_schema.tables, 不允许以 CREATE TABLE 修补能力。"""

        row = connection.execute(
            """
            SELECT 1
            FROM information_schema.tables
            WHERE lower(table_name) = lower(?)
            LIMIT 1
            """,
            [table],
        ).fetchone()
        return row is not None

    def _connection_for(
        self, requirement: CapabilityRequirement
    ) -> tuple[Any | None, str, list[str]]:
        """解析能力对应的连接与来源标识; analytics 连接缺失时显式暴露。"""

        if requirement.connection_name == "source":
            return self.source_conn, self.source_artifact, []
        if self.analytics_conn is None:
            return (
                None,
                self.analytics_artifact,
                ["analytics_conn", requirement.table, *sorted(requirement.columns)],
            )
        return self.analytics_conn, self.analytics_artifact, []

    def _inspect_requirement(
        self, requirement: CapabilityRequirement
    ) -> DataCapabilitySnapshot:
        """检查一项能力的连接、表和字段, 生成不自相矛盾的快照。"""

        connection, artifact, connection_missing = self._connection_for(requirement)
        if connection is None:
            return DataCapabilitySnapshot(
                capability=requirement.capability,
                available=False,
                coverage=0.0,
                missing_requirements=connection_missing,
                source_artifact=artifact,
            )

        if not self._table_exists(connection, requirement.table):
            return DataCapabilitySnapshot(
                capability=requirement.capability,
                available=False,
                coverage=0.0,
                missing_requirements=[requirement.table, *sorted(requirement.columns)],
                source_artifact=artifact,
            )

        actual_columns = self._table_columns(connection, requirement.table)
        missing_columns = sorted(
            required_column
            for required_column in requirement.columns
            if not (
                actual_columns.intersection(
                    _COLUMN_ALIASES.get(required_column, frozenset({required_column}))
                )
            )
        )
        if missing_columns:
            return DataCapabilitySnapshot(
                capability=requirement.capability,
                available=False,
                coverage=0.0,
                missing_requirements=missing_columns,
                source_artifact=artifact,
            )

        # Schema 存在不等于有可用数据。空表必须显式暴露为不可用，不能伪装成
        # coverage=1.0；READY manifest 等更高层契约由调用方继续校验。
        row_count = int(
            connection.execute(
                f'SELECT COUNT(*) FROM "{requirement.table}"'
            ).fetchone()[0]
        )
        if row_count == 0:
            return DataCapabilitySnapshot(
                capability=requirement.capability,
                available=False,
                coverage=0.0,
                missing_requirements=[f"{requirement.table}:non_empty"],
                source_artifact=artifact,
            )

        return DataCapabilitySnapshot(
            capability=requirement.capability,
            available=True,
            coverage=1.0,
            missing_requirements=[],
            source_artifact=artifact,
        )

    def inspect(self) -> dict[str, DataCapabilitySnapshot]:
        """按 REQUIREMENTS 固定顺序只读探测全部能力, 返回稳定映射。"""

        return {
            capability: self._inspect_requirement(requirement)
            for capability, requirement in REQUIREMENTS.items()
        }

    def require(self, names: list[str]) -> list[DataCapabilitySnapshot]:
        """要求一组能力全部可用; 按调用顺序返回, 首个缺失项显式抛错。"""

        if not isinstance(names, list):
            raise TypeError("names must be a list of capability names")
        snapshots = self.inspect()
        required: list[DataCapabilitySnapshot] = []
        for name in names:
            if name not in REQUIREMENTS:
                raise ValueError(f"unknown data capability: {name}")
            snapshot = snapshots[name]
            if not snapshot.available:
                raise MissingDataCapabilityError(
                    capability=name,
                    missing_requirements=snapshot.missing_requirements,
                )
            required.append(snapshot)
        return required

    @staticmethod
    def synthetic_snapshot(
        *,
        dialogue: bool,
        analytics: bool,
        retake: bool,
        ability: bool,
    ) -> dict[str, DataCapabilitySnapshot]:
        """生成仅供测试的能力快照, 生产代码不得用它替代 inspect()。"""

        return {
            "DIALOGUE_EVIDENCE": DataCapabilitySnapshot(
                capability="DIALOGUE_EVIDENCE",
                available=dialogue,
                coverage=1.0 if dialogue else 0.0,
                missing_requirements=[] if dialogue else ["session_dialogue_turns"],
                source_artifact="synthetic",
            ),
            "BUSINESS_ANALYTICS": DataCapabilitySnapshot(
                capability="BUSINESS_ANALYTICS",
                available=analytics,
                coverage=1.0 if analytics else 0.0,
                missing_requirements=[] if analytics else ["business_analytics_manifest"],
                source_artifact="synthetic",
            ),
            "RETAKE_COHORT": DataCapabilitySnapshot(
                capability="RETAKE_COHORT",
                available=retake,
                coverage=1.0 if retake else 0.0,
                missing_requirements=[]
                if retake
                else ["student_key", "attempt_number"],
                source_artifact="synthetic",
            ),
            "ABILITY_COHORT": DataCapabilitySnapshot(
                capability="ABILITY_COHORT",
                available=ability,
                coverage=1.0 if ability else 0.0,
                missing_requirements=[]
                if ability
                else ["student_key", "historic_question_count", "historic_correctness"],
                source_artifact="synthetic",
            ),
        }


__all__ = [
    "CapabilityRequirement",
    "DataCapabilityRegistry",
    "MissingDataCapabilityError",
    "REQUIREMENTS",
]

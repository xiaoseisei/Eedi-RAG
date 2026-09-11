"""图投影的严格业务契约。"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from src.business_models import StrictBusinessModel


NodeType = Literal[
    "SESSION", "QUESTION", "SUBJECT", "MISCONCEPTION_EVENT",
    "MISCONCEPTION_CONCEPT", "TUTOR_STRATEGY_EVENT", "STRATEGY_CONCEPT",
    "STUDENT_QUESTION_EVENT", "QUESTION_CLUSTER", "DIFFICULTY_EVENT", "TURN", "TALK_MOVE",
]
EdgeType = Literal[
    "ANCHORS", "ABOUT_SUBJECT", "HAS_MISCONCEPTION", "HAS_STRATEGY",
    "HAS_STUDENT_QUESTION", "HAS_DIFFICULTY_SIGNAL", "HAS_TURN", "TYPE_OF",
    "EVIDENCED_BY", "USES_TALK_MOVE", "CO_OCCURS_WITH", "FOLLOWED_BY",
]
ReviewStatus = Literal["SOURCE", "AUTO_ASSIGNED", "HUMAN_APPROVED"]


class GraphNode(StrictBusinessModel):
    """图节点：保留来源会话、卡片、轮次和派生版本，支持精确回溯。"""

    node_id: str = Field(min_length=1)
    node_type: NodeType
    label: str = Field(min_length=1)
    properties: dict[str, Any] = Field(default_factory=dict)
    source_session_id: int | None = Field(default=None, ge=1)
    source_card_id: str | None = None
    source_turn_ids: list[int] = Field(default_factory=list)
    derivation_version: str = Field(min_length=1)
    review_status: ReviewStatus

    @field_validator("source_turn_ids")
    @classmethod
    def validate_turn_ids(cls, value: list[int]) -> list[int]:
        if any(turn_id < 1 for turn_id in value):
            raise ValueError("source turn IDs must be positive")
        return value


class GraphEdge(StrictBusinessModel):
    """图边：所有关系必须绑定真实会话和至少一条来源轮次。"""

    edge_id: str = Field(min_length=1)
    source_node_id: str = Field(min_length=1)
    target_node_id: str = Field(min_length=1)
    edge_type: EdgeType
    source_session_id: int = Field(ge=1)
    source_turn_ids: list[int] = Field(min_length=1)
    derivation_version: str = Field(min_length=1)
    review_status: ReviewStatus

    @field_validator("source_turn_ids")
    @classmethod
    def validate_turn_ids(cls, value: list[int]) -> list[int]:
        if any(turn_id < 1 for turn_id in value):
            raise ValueError("source turn IDs must be positive")
        return value


class GraphProjectionManifest(StrictBusinessModel):
    """图 sidecar 的构建清单；READY 只代表哈希绑定且计数完整。"""

    graph_version: str = Field(min_length=1)
    source_artifact_hash: str = Field(min_length=64, max_length=64)
    node_count: int = Field(ge=0)
    edge_count: int = Field(ge=0)
    build_status: Literal["STAGING", "READY", "FAILED"]
    built_at_utc: datetime
    failure_reason: str | None = None

    @field_validator("built_at_utc")
    @classmethod
    def validate_utc_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None or value.utcoffset().total_seconds() != 0:
            raise ValueError("built_at_utc must be timezone-aware UTC")
        return value

    @field_validator("source_artifact_hash")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        if any(char not in "0123456789abcdefABCDEF" for char in value):
            raise ValueError("source artifact hash must be SHA-256 hex")
        return value.lower()

    @model_validator(mode="after")
    def validate_status(self) -> "GraphProjectionManifest":
        if self.build_status == "FAILED" and not self.failure_reason:
            raise ValueError("failed graph projection requires failure_reason")
        if self.build_status != "FAILED" and self.failure_reason is not None:
            raise ValueError("failure_reason is only valid for FAILED projections")
        return self

"""受控跨会话标签：独立于 Card embedding 的确定性分类契约。"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from src.business_models import StrictBusinessModel


LabelType = Literal[
    "QUESTION_FUNCTION", "MISCONCEPTION", "EXAM_ERROR",
    "TUTOR_STRATEGY", "DIFFICULTY_SIGNAL", "CONTENT_OPPORTUNITY",
]


class Taxonomy(StrictBusinessModel):
    taxonomy_version: str = Field(min_length=1)
    label_type: LabelType
    labels: dict[str, str] = Field(min_length=1)
    signals: dict[str, list[str]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def signals_reference_labels(self) -> "Taxonomy":
        unknown = set(self.signals) - set(self.labels)
        if unknown:
            raise ValueError(f"taxonomy signals reference unknown labels: {sorted(unknown)}")
        if "UNCLASSIFIED" not in self.labels:
            raise ValueError("taxonomy must define UNCLASSIFIED")
        return self


class LabelAssignment(StrictBusinessModel):
    source_node_id: str = Field(min_length=1)
    label_type: str = Field(min_length=1)
    canonical_label: str = Field(min_length=1)
    assignment_source: Literal["deterministic_rules", "llm_suggestion"]
    review_status: Literal["AUTO_ASSIGNED", "HUMAN_APPROVED", "UNCLASSIFIED"]
    taxonomy_version: str = Field(min_length=1)
    source_session_id: int = Field(ge=1)
    source_turn_ids: list[int] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_review_gate(self) -> "LabelAssignment":
        if self.assignment_source == "llm_suggestion" and self.review_status == "HUMAN_APPROVED":
            raise ValueError("llm suggestions cannot be marked HUMAN_APPROVED")
        return self


class DifficultySignal(StrictBusinessModel):
    source_node_id: str = Field(min_length=1)
    signal: str = Field(min_length=1)
    review_status: Literal["AUTO_ASSIGNED", "HUMAN_APPROVED", "UNCLASSIFIED"]
    taxonomy_version: str = Field(min_length=1)
    source_session_id: int = Field(ge=1)
    source_turn_ids: list[int] = Field(min_length=1)


_ROOT = Path(__file__).resolve().parents[1]
_ARTIFACT = _ROOT / "data" / "business" / "graph-taxonomy-v1.json"


def load_taxonomies(path: str | Path = _ARTIFACT) -> dict[str, Taxonomy]:
    """读取受控 artifact；数组顺序就是确定性规则顺序。"""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    version = raw.get("taxonomy_version", "graph-taxonomy-v1")
    result: dict[str, Taxonomy] = {}
    type_map = {
        "question_function": "QUESTION_FUNCTION", "misconception": "MISCONCEPTION",
        "exam_error": "EXAM_ERROR", "tutor_strategy": "TUTOR_STRATEGY",
        "difficulty_signal": "DIFFICULTY_SIGNAL", "content_opportunity": "CONTENT_OPPORTUNITY",
    }
    for key, label_type in type_map.items():
        names = raw.get(key, raw.get("misconceptions", [])) if key == "misconception" else raw.get(key, [])
        labels = {name: name.replace("_", " ").title() for name in names}
        labels.setdefault("UNCLASSIFIED", "Unclassified")
        signals = {name: raw.get("matching_signals", {}).get(name, []) for name in labels}
        result[label_type] = Taxonomy(taxonomy_version=version, label_type=label_type,
                                      labels=labels, signals=signals)
    return result


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value.casefold()
    if isinstance(value, (list, tuple, set)):
        return " ".join(_text(item) for item in value)
    if isinstance(value, dict):
        return " ".join(_text(item) for item in value.values())
    return "" if value is None else str(value).casefold()


def _assignment(node_id: str, label_type: str, canonical: str, taxonomy: Taxonomy,
                session_id: Any, turn_ids: Any) -> LabelAssignment:
    if session_id is None or turn_ids is None:
        raise ValueError("label assignment requires source session and turns")
    ids = [int(item) for item in turn_ids]
    sid = int(session_id)
    if sid < 1 or not ids or any(item < 1 for item in ids):
        raise ValueError("label assignment requires positive source provenance")
    return LabelAssignment(source_node_id=node_id, label_type=label_type, canonical_label=canonical,
                           assignment_source="deterministic_rules", review_status="AUTO_ASSIGNED",
                           taxonomy_version=taxonomy.taxonomy_version, source_session_id=sid,
                           source_turn_ids=ids)


def _match(text: str, taxonomy: Taxonomy, preferred: list[str]) -> str:
    for label in preferred:
        if label not in taxonomy.labels:
            continue
        signals = taxonomy.signals.get(label, [])
        if signals and any(signal.casefold() in text for signal in signals):
            return label
    for label in taxonomy.labels:
        signals = taxonomy.signals.get(label, [])
        if label != "UNCLASSIFIED" and signals and any(signal.casefold() in text for signal in signals):
            return label
    return "UNCLASSIFIED"


def assign_question_function(student_turn_text: str, *, source_node_id: str = "student-question",
                             source_session_id: int = 1, source_turn_ids: list[int] | None = None) -> LabelAssignment:
    taxonomy = load_taxonomies()["QUESTION_FUNCTION"]
    text = _text(student_turn_text)
    rules = [("REQUEST_EXPLANATION", ["why", "explain"]), ("REQUEST_CONFIRMATION", ["is this", "right"]),
             ("ASK_NEXT_STEP", ["next", "what do i do"]), ("COMPARE_OPTIONS", ["which", "option"]),
             ("RECALL_RULE", ["rule", "formula"]), ("ASK_DEFINITION", ["what is"]),
             ("EXPRESS_UNCERTAINTY", ["not sure", "don't know", "uncertain"])]
    canonical = next((name for name, signals in rules if any(signal in text for signal in signals)), "OTHER")
    if canonical not in taxonomy.labels:
        canonical = "UNCLASSIFIED"
    return _assignment(source_node_id, "QUESTION_FUNCTION", canonical, taxonomy, source_session_id,
                       source_turn_ids if source_turn_ids is not None else [1])


def assign_misconception_label(card_row: dict[str, Any], taxonomy: Taxonomy, *, source_node_id: str | None = None) -> LabelAssignment:
    text = _text({key: card_row.get(key) for key in ("misconception_name", "deep_mechanism", "confusion_triggers")})
    canonical = _match(text, taxonomy, ["PLACE_VALUE_CONFUSION", "ROUNDING_RULE_CONFUSION",
                                        "ORDER_OF_OPERATIONS_CONFUSION", "FRACTION_DENOMINATOR_CONFUSION",
                                        "NEGATIVE_SIGN_SCOPE_CONFUSION", "FORMULA_SELECTION_ERROR",
                                        "LANGUAGE_INTERPRETATION_ERROR", "OPTION_COMPARISON_ERROR"])
    return _assignment(source_node_id or str(card_row.get("chunk_id", "misconception-event")), "MISCONCEPTION", canonical, taxonomy,
                       card_row.get("session_id"), card_row.get("source_turn_ids"))


def assign_exam_error_label(card_row: dict[str, Any], taxonomy: Taxonomy, *, source_node_id: str | None = None) -> LabelAssignment:
    text = _text(card_row)
    canonical = _match(text, taxonomy, ["CONCEPTUAL_ERROR", "PROCEDURAL_ERROR", "REPRESENTATION_ERROR",
                                        "LANGUAGE_INTERPRETATION_ERROR", "CONDITION_OMISSION",
                                        "OVERGENERALIZATION_ERROR", "OPTION_COMPARISON_ERROR"])
    return _assignment(source_node_id or str(card_row.get("chunk_id", "exam-error-event")), "EXAM_ERROR", canonical, taxonomy,
                       card_row.get("session_id"), card_row.get("source_turn_ids"))


def assign_strategy_labels(strategy_row: dict[str, Any], taxonomy: Taxonomy, *, source_node_id: str | None = None) -> list[LabelAssignment]:
    text = _text({"strategy_category": strategy_row.get("strategy_category"), "talk_moves": strategy_row.get("talk_moves"),
                  "key_aha_question": strategy_row.get("key_aha_question"), "scaffolding_steps": strategy_row.get("scaffolding_steps")})
    preferred = ["ASK_FOR_REASONING", "PRESS_FOR_ACCURACY", "USE_COUNTER_EXAMPLE", "RESTATE_STUDENT_IDEA",
                 "BREAK_INTO_STEPS", "CONNECT_TO_PRIOR_KNOWLEDGE", "COMPARE_OPTIONS", "CHECK_UNDERSTANDING"]
    matches = [label for label in preferred if label in taxonomy.labels and any(signal.casefold() in text for signal in taxonomy.signals.get(label, []))]
    if not matches:
        matches = ["UNCLASSIFIED"]
    return [_assignment(source_node_id or str(strategy_row.get("chunk_id", "tutor-strategy-event")), "TUTOR_STRATEGY", label, taxonomy,
                        strategy_row.get("session_id"), strategy_row.get("source_turn_ids")) for label in matches]


def assign_content_opportunity(card_row: dict[str, Any], taxonomy: Taxonomy, *, source_node_id: str | None = None) -> LabelAssignment:
    """将真实的错因/策略缺口标记为内容机会，不把建议冒充事实。"""
    text = _text(card_row)
    if "option" in text or "选择" in text:
        preferred = "ADD_OPTION_COMPARISON"
    elif "language" in text or "英语" in text:
        preferred = "CLARIFY_LANGUAGE"
    elif card_row.get("deep_mechanism") or card_row.get("scaffolding_steps"):
        preferred = "ADD worked_example"
    else:
        preferred = "UNCLASSIFIED"
    canonical = preferred if preferred in taxonomy.labels else "UNCLASSIFIED"
    return _assignment(source_node_id or str(card_row.get("chunk_id", "content-opportunity")), "CONTENT_OPPORTUNITY", canonical, taxonomy,
                       card_row.get("session_id"), card_row.get("source_turn_ids"))


def derive_difficulty_signals(session_row: dict[str, Any], turns: list[dict[str, Any]]) -> list[DifficultySignal]:
    taxonomy = load_taxonomies()["DIFFICULTY_SIGNAL"]
    if session_row.get("intervention_id") is None:
        raise ValueError("difficulty signals require intervention_id")
    ids = [int(turn["turn_id"]) for turn in turns]
    if any(turn_id < 1 for turn_id in ids):
        raise ValueError("difficulty signals require positive turn_id values")
    text = _text(turns)
    labels = [label for label in taxonomy.labels if label != "UNCLASSIFIED"
              and any(signal.casefold() in text for signal in taxonomy.signals.get(label, []))]
    if not labels:
        labels = ["UNCLASSIFIED"]
    return [DifficultySignal(source_node_id=f"session:{int(session_row['intervention_id'])}", signal=label,
                             review_status="AUTO_ASSIGNED", taxonomy_version=taxonomy.taxonomy_version,
                             source_session_id=int(session_row["intervention_id"]), source_turn_ids=ids) for label in labels]

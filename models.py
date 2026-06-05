"""Shared data models for the agricultural reasoning pipeline.

Every module imports ONLY from this file. No module imports another module.
"""

from enum import Enum
from typing import List, Optional, Protocol, Literal


class Evidence:
    """A single retrieved passage from RAG."""

    def __init__(
        self,
        content: str,
        source: str,
        relevance_score: float,
        metadata: Optional[dict] = None,
    ):
        self.content = content
        self.source = source
        self.relevance_score = relevance_score
        self.metadata = metadata or {}

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "source": self.source,
            "relevance_score": self.relevance_score,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Evidence":
        return cls(
            content=d["content"],
            source=d["source"],
            relevance_score=d["relevance_score"],
            metadata=d.get("metadata", {}),
        )


class ReasoningStep:
    """One step in a structured reasoning chain."""

    VALID_TYPES = (
        "context_setup",
        "knowledge_application",
        "causal_reasoning",
        "comparison",
        "condition_analysis",
        "evidence_integration",
        "conclusion",
    )
    VALID_CONFIDENCES = ("high", "medium", "low")

    def __init__(
        self,
        step: int,
        type: str,
        content: str,
        evidence: Optional[str] = None,
        confidence: str = "medium",
    ):
        if type not in self.VALID_TYPES:
            raise ValueError(f"Invalid step type: {type}. Must be one of {self.VALID_TYPES}")
        if confidence not in self.VALID_CONFIDENCES:
            raise ValueError(f"Invalid confidence: {confidence}. Must be one of {self.VALID_CONFIDENCES}")
        self.step = step
        self.type = type
        self.content = content
        self.evidence = evidence
        self.confidence = confidence

    def to_dict(self) -> dict:
        d = {"step": self.step, "type": self.type, "content": self.content, "confidence": self.confidence}
        if self.evidence:
            d["evidence"] = self.evidence
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ReasoningStep":
        return cls(
            step=d["step"],
            type=d["type"],
            content=d["content"],
            evidence=d.get("evidence"),
            confidence=d.get("confidence", "medium"),
        )


class ReasoningChain:
    """A complete structured reasoning chain."""

    def __init__(
        self,
        steps: List[ReasoningStep],
        react_rounds: int = 0,
        self_consistency_selected: int = 0,
    ):
        self.steps = steps
        self.react_rounds = react_rounds
        self.self_consistency_selected = self_consistency_selected

    def to_dict(self) -> dict:
        return {
            "steps": [s.to_dict() for s in self.steps],
            "react_rounds": self.react_rounds,
            "self_consistency_selected": self.self_consistency_selected,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ReasoningChain":
        return cls(
            steps=[ReasoningStep.from_dict(s) for s in d["steps"]],
            react_rounds=d.get("react_rounds", 0),
            self_consistency_selected=d.get("self_consistency_selected", 0),
        )

    def to_text(self) -> str:
        """Serialize chain to readable text for LLM prompts."""
        lines = []
        for s in self.steps:
            line = f"Step {s.step} [{s.type}]: {s.content}"
            if s.evidence:
                line += f"\n  Evidence: {s.evidence}"
            line += f" (confidence: {s.confidence})"
            lines.append(line)
        return "\n".join(lines)


class DifficultyLevel(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class ClassificationResult:
    """Output of the difficulty classifier."""

    def __init__(self, difficulty: DifficultyLevel, raw_response: str):
        self.difficulty = difficulty
        self.raw_response = raw_response

    def to_dict(self) -> dict:
        return {"difficulty": self.difficulty.value, "raw_response": self.raw_response}


class ReviewAction:
    """Atomic modification action from Reviewer."""

    VALID_ACTIONS = (
        "add_evidence",
        "revise_step",
        "insert_step",
        "remove_step",
        "merge_steps",
        "adjust_confidence",
    )
    VALID_PRIORITIES = ("P0", "P1", "P2")

    def __init__(
        self,
        action: str,
        target_step: int,
        priority: str,
        params: dict,
        reason: str,
    ):
        if action not in self.VALID_ACTIONS:
            raise ValueError(f"Invalid action: {action}. Must be one of {self.VALID_ACTIONS}")
        if priority not in self.VALID_PRIORITIES:
            raise ValueError(f"Invalid priority: {priority}. Must be one of {self.VALID_PRIORITIES}")
        self.action = action
        self.target_step = target_step
        self.priority = priority
        self.params = params
        self.reason = reason

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "target_step": self.target_step,
            "priority": self.priority,
            "params": self.params,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ReviewAction":
        return cls(
            action=d["action"],
            target_step=d["target_step"],
            priority=d["priority"],
            params=d.get("params", {}),
            reason=d.get("reason", ""),
        )


class UnifiedActions:
    """Output of Reviewer Phase C."""

    def __init__(
        self,
        priority_actions: List[ReviewAction],
        optional_improvements: Optional[List[ReviewAction]] = None,
        conflicts_resolved: Optional[List[str]] = None,
    ):
        self.priority_actions = priority_actions
        self.optional_improvements = optional_improvements or []
        self.conflicts_resolved = conflicts_resolved or []

    def to_dict(self) -> dict:
        return {
            "priority_actions": [a.to_dict() for a in self.priority_actions],
            "optional_improvements": [a.to_dict() for a in self.optional_improvements],
            "conflicts_resolved": self.conflicts_resolved,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "UnifiedActions":
        return cls(
            priority_actions=[ReviewAction.from_dict(a) for a in d.get("priority_actions", [])],
            optional_improvements=[ReviewAction.from_dict(a) for a in d.get("optional_improvements", [])],
            conflicts_resolved=d.get("conflicts_resolved", []),
        )


class ReviewCritique:
    """Output of a single review phase."""

    VALID_PHASES = ("A_logic", "B_external", "C_integration")

    def __init__(self, phase: str, issues: List[dict]):
        if phase not in self.VALID_PHASES:
            raise ValueError(f"Invalid phase: {phase}. Must be one of {self.VALID_PHASES}")
        self.phase = phase
        self.issues = issues

    def to_dict(self) -> dict:
        return {"phase": self.phase, "issues": self.issues}

    @classmethod
    def from_dict(cls, d: dict) -> "ReviewCritique":
        return cls(phase=d["phase"], issues=d["issues"])


class QualityScores:
    """Evaluator output with quality dimensions."""

    def __init__(
        self,
        faithfulness: float,
        structure: float,
        information_density: float,
        logical_completeness: float,
        traceability: float,
        overall: float,
        step_order: float = 0.0,
        ppl: Optional[float] = None,
    ):
        self.faithfulness = faithfulness
        self.structure = structure
        self.information_density = information_density
        self.logical_completeness = logical_completeness
        self.traceability = traceability
        self.overall = overall
        self.step_order = step_order
        self.ppl = ppl

    def to_dict(self) -> dict:
        d = {
            "faithfulness": self.faithfulness,
            "structure": self.structure,
            "information_density": self.information_density,
            "logical_completeness": self.logical_completeness,
            "traceability": self.traceability,
            "step_order": self.step_order,
            "overall": self.overall,
        }
        if self.ppl is not None:
            d["ppl"] = self.ppl
        return d


class PipelineItem:
    """A single item flowing through the pipeline."""

    def __init__(
        self,
        id: str,
        question: str,
        answer: str,
        question_type: str,
        difficulty: Optional[DifficultyLevel] = None,
        draft_chain: Optional[ReasoningChain] = None,
        unified_actions: Optional[UnifiedActions] = None,
        revised_chain: Optional[ReasoningChain] = None,
        quality_scores: Optional[QualityScores] = None,
        critique_history: Optional[List[ReviewCritique]] = None,
        metadata: Optional[dict] = None,
    ):
        self.id = id
        self.question = question
        self.answer = answer
        self.question_type = question_type
        self.difficulty = difficulty
        self.draft_chain = draft_chain
        self.unified_actions = unified_actions
        self.revised_chain = revised_chain
        self.quality_scores = quality_scores
        self.critique_history = critique_history or []
        self.metadata = metadata or {}

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "question": self.question,
            "answer": self.answer,
            "question_type": self.question_type,
        }
        if self.difficulty:
            d["difficulty"] = self.difficulty.value
        if self.draft_chain:
            d["draft_chain"] = self.draft_chain.to_dict()
        if self.unified_actions:
            d["unified_actions"] = self.unified_actions.to_dict()
        if self.revised_chain:
            d["revised_chain"] = self.revised_chain.to_dict()
        if self.quality_scores:
            d["quality_scores"] = self.quality_scores.to_dict()
        if self.critique_history:
            d["critique_history"] = [c.to_dict() for c in self.critique_history]
        if self.metadata:
            d["metadata"] = self.metadata
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "PipelineItem":
        return cls(
            id=d["id"],
            question=d["question"],
            answer=d["answer"],
            question_type=d["question_type"],
            difficulty=DifficultyLevel(d["difficulty"]) if d.get("difficulty") else None,
            draft_chain=ReasoningChain.from_dict(d["draft_chain"]) if d.get("draft_chain") else None,
            unified_actions=UnifiedActions.from_dict(d["unified_actions"]) if d.get("unified_actions") else None,
            revised_chain=ReasoningChain.from_dict(d["revised_chain"]) if d.get("revised_chain") else None,
            quality_scores=None,  # reconstructed from dict if needed
            critique_history=[ReviewCritique.from_dict(c) for c in d.get("critique_history", [])],
            metadata=d.get("metadata", {}),
        )


class LLMCallFn(Protocol):
    """Protocol for the injected LLM call function."""

    def __call__(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.2,
        max_tokens: int = 4096,
        model: str = None,
    ) -> str: ...

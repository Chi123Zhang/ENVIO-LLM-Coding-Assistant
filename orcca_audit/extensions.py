"""Explicit contracts for future validated implementation/hypothesis modules."""
from typing import Protocol


class ImplementationEvidenceAnalysis(Protocol):
    def analyze(self, records, approved_codebook, time_link_policy) -> dict: ...


class HypothesisEvaluation(Protocol):
    def evaluate(self, validated_measures, prespecified_estimand, authorization) -> dict: ...


EXTENSIONS = {"Implementation Evidence Analysis": "not_implemented_requires_codebook_and_real_labels",
              "Hypothesis Evaluation": "not_implemented_requires_specification_and_explicit_analysis_authorization"}

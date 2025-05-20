from dataclasses import dataclass


@dataclass
class PromptPair:
    """One (defect, filter) prompt with optional weight."""

    target: str
    background: str
    weight: float = 1.0  # future use for weighted voting


@dataclass
class ObjectPrompt:
    """Structured replacement for the old English sentence."""

    name: str
    count: int  # N
    max_anomalies: int  # k_mask
    anomaly_area_ratio: float  # e.g. 0.3

    @property
    def object_max_area(self) -> float:
        return 1.0 / self.count

from dataclasses import dataclass


@dataclass
class PromptPair:
    """One (defect, filter) prompt with optional weight."""

    target: str
    background: str
    weight: float = 1.0  # future use for weighted voting


@dataclass
class ObjectPrompt:
    """Prompt for a specific object instance."""

    name: str
    max_anomalies: int  # k_mask
    anomaly_area_ratio: float  # e.g. 0.3
    count: int = 1  # N
    proposed_object_min_area: float = 0.3  # e.g. 0.3
    proposed_object_max_area: float = 1.0

    @property
    def object_max_area(self) -> float:
        return max(1.0 / self.count, self.proposed_object_max_area)

    @property
    def object_min_area(self) -> float:
        return self.proposed_object_min_area

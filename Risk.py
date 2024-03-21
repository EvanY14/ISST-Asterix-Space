from dataclasses import dataclass, field
@dataclass
class Risk:
    name: str = field(init=True)
    baseline_likelihood: float = field(init=True)

    # Schedule Risk Parameterers
    schedule_risk_minimum_value: float = field(init=True, default=0.0)
    schedule_risk_maximum_value: float = field(init=True, default=1.0)
    schedule_risk_most_likely_value: float = field(init=True, default=0.5)

    # Cost Risk Parameterers
    cost_risk_minimum_value: float = field(init=True, default=0.0)
    cost_risk_maximum_value: float = field(init=True, default=1.0)
    cost_risk_most_likely_value: float = field(init=True, default=0.5)

    # Technical Risk Parameters
    technical_risk_minimum_values: list[float] = field(init=True, default_factory=lambda: [])
    technical_risk_maximum_values: list[float] = field(init=True, default_factory=lambda: [])
    technical_risk_most_likely_values: list[float] = field(init=True, default_factory=lambda: [])
"""
Three-Tier Symbolic Conflict Resolution Engine.

Implements Algorithm 4 from the paper.

18 rules evaluated over 12 concept activations {cₖ}₁₂ₖ₌₁:
    Tier 1: Concurrent physio + facial + (opt) acoustic → high confidence
    Tier 2: Consensus from ≥2 indicator domains → moderate confidence
    Tier 3: Conflicting signals → escalate σ² by +0.15, trigger alert

Tier-3 escalation (+0.15) raises σ² > τ* = 0.35, ensuring all
Tier-3 conflicts trigger a clinician alert via the same operating
point used by the UQ layer (Algorithm 3).

The complete rule set is in rules/symbolic_rules.json.

NOTE: The sensitivity of the MSE improvement attributable to this module
(ΔMSE = 0.11; Table 9) to the specific 18-rule instantiation has not
been formally evaluated. See Section 3.6 (Limitations) of the paper.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

TIER3_ESCALATION = 0.15   # raises σ² to 0.5 > τ* = 0.35 → always triggers alert

# Concept activation indices
CONCEPT_NAMES = [
    "c_hrv_up",       # 0: HRV elevation
    "c_hrv_down",     # 1: HRV suppression (opioid effect)
    "c_spo2_down",    # 2: SpO₂ decrease
    "c_resp_up",      # 3: Respiratory rate increase
    "c_au4",          # 4: AU4 brow lowerer
    "c_au6",          # 5: AU6 cheek raiser
    "c_au9",          # 6: AU9 nose wrinkler
    "c_au10",         # 7: AU10 upper lip raiser
    "c_au25",         # 8: AU25 lips part
    "c_cry",          # 9: Cry / vocalisation (optional acoustic)
    "c_grimace",      # 10: General grimace
    "c_neutral",      # 11: Neutral expression (no visible pain markers)
]
N_CONCEPTS = len(CONCEPT_NAMES)
CONCEPT_IDX = {name: i for i, name in enumerate(CONCEPT_NAMES)}


@dataclass
class Rule:
    """A single rule in the symbolic engine."""
    tier: int
    conditions: List[str]          # concept names that must be active
    output_pain_level: float       # NRS value if rule fires
    weight: float = 1.0
    explanation: str = ""
    conflicts_with: List[str] = field(default_factory=list)


def load_rules(rules_path: str) -> List[Rule]:
    """Load rules from JSON file."""
    with open(rules_path) as f:
        raw = json.load(f)
    return [Rule(**r) for r in raw["rules"]]


class SymbolicConflictEngine:
    """
    Three-tier symbolic rule engine.

    Translates uncertainty-weighted concept activations into
    human-readable pain labels and explanations.

    Args:
        rules_path:  Path to symbolic_rules.json.
        threshold:   Alert threshold τ* (must match UQ layer, default 0.35).
    """

    def __init__(
        self,
        rules_path: str = "rules/symbolic_rules.json",
        alert_threshold: float = 0.35,
    ) -> None:
        self.alert_threshold = alert_threshold
        if Path(rules_path).exists():
            self.rules = load_rules(rules_path)
        else:
            self.rules = self._default_rules()

        self.tier1_rules = [r for r in self.rules if r.tier == 1]
        self.tier2_rules = [r for r in self.rules if r.tier == 2]
        self.tier3_rules = [r for r in self.rules if r.tier == 3]

    def _default_rules(self) -> List[Rule]:
        """
        Default 18-rule set based on CPOT/FLACC behavioural indicators.
        Designed with clinical domain experts (Section 3.6).
        """
        rules = []

        # ── Tier 1: High confidence (physio + facial + optional acoustic) ──
        rules += [
            Rule(1, ["c_hrv_up", "c_au4", "c_cry"], 8.0, 1.2,
                 "High pain: HRV↑ + AU4 + cry"),
            Rule(1, ["c_hrv_up", "c_au6", "c_grimace"], 7.5, 1.1,
                 "High pain: HRV↑ + orbital tighten + grimace"),
            Rule(1, ["c_resp_up", "c_au9", "c_au25"], 7.0, 1.1,
                 "High pain: resp↑ + nose wrinkler + lips part"),
            Rule(1, ["c_hrv_up", "c_resp_up", "c_au4", "c_au10"], 9.0, 1.3,
                 "Severe pain: multi-physio + multi-AU"),
            Rule(1, ["c_spo2_down", "c_au6", "c_au9"], 7.5, 1.0,
                 "High pain: SpO₂↓ + facial AUs"),
            Rule(1, ["c_hrv_up", "c_grimace", "c_cry"], 8.5, 1.2,
                 "High pain: HRV + grimace + vocalisation"),
        ]

        # ── Tier 2: Moderate confidence (≥2 indicator domains) ──────────
        rules += [
            Rule(2, ["c_hrv_up", "c_au4"], 5.5, 1.0,
                 "Moderate pain: HRV↑ + brow lower"),
            Rule(2, ["c_spo2_down", "c_au6"], 5.0, 1.0,
                 "Moderate pain: SpO₂↓ + cheek raise"),
            Rule(2, ["c_resp_up", "c_grimace"], 5.5, 1.0,
                 "Moderate pain: resp↑ + grimace"),
            Rule(2, ["c_au4", "c_au25"], 4.0, 0.9,
                 "Mild-moderate pain: facial AUs only"),
            Rule(2, ["c_hrv_up", "c_au9"], 5.0, 1.0,
                 "Moderate pain: HRV + nose wrinkler"),
            Rule(2, ["c_resp_up", "c_au4"], 5.0, 1.0,
                 "Moderate pain: resp + brow"),
        ]

        # ── Tier 3: Conflicting signals → escalate uncertainty ─────────
        rules += [
            Rule(3, ["c_hrv_up", "c_neutral"], 0.0, 1.0,
                 "Conflict: HRV↑ but neutral face",
                 conflicts_with=["c_neutral"]),
            Rule(3, ["c_au4", "c_hrv_down"], 0.0, 1.0,
                 "Conflict: pain AU but HRV suppressed (opioid?)",
                 conflicts_with=["c_hrv_down"]),
            Rule(3, ["c_spo2_down", "c_neutral"], 0.0, 1.0,
                 "Conflict: SpO₂↓ but neutral face"),
            Rule(3, ["c_grimace", "c_hrv_down"], 0.0, 1.0,
                 "Conflict: grimace but cardiac suppression"),
            Rule(3, ["c_resp_up", "c_neutral"], 0.0, 1.0,
                 "Conflict: resp↑ but neutral face"),
            Rule(3, ["c_au6", "c_hrv_down"], 0.0, 1.0,
                 "Conflict: orbital AU but suppressed HRV"),
        ]

        assert len(rules) == 18, f"Expected 18 rules, got {len(rules)}"
        return rules

    def evaluate(
        self,
        concept_activations: torch.Tensor,
        current_var: float = 0.0,
    ) -> Tuple[float, int, str, float]:
        """
        Implements Algorithm 4.

        Args:
            concept_activations: (N_CONCEPTS,) float tensor ∈ [0,1].
            current_var:         Current σ² from UQ layer.

        Returns:
            y_hat:      Predicted NRS score (0 if Tier-3 ABSTAIN).
            tier:       Winning tier (1, 2, or 3).
            explanation: Human-readable explanation string.
            updated_var: σ² after Tier-3 escalation (if applicable).
        """
        acts = concept_activations.detach().cpu()
        activation_threshold = 0.5

        def is_active(concept_name: str) -> bool:
            idx = CONCEPT_IDX.get(concept_name, -1)
            if idx < 0 or idx >= len(acts):
                return False
            return bool(acts[idx] > activation_threshold)

        def rule_fires(rule: Rule) -> bool:
            return all(is_active(c) for c in rule.conditions)

        updated_var = current_var

        # Step 2: Check Tier 1
        t1_firing = [r for r in self.tier1_rules if rule_fires(r)]
        if t1_firing:
            best = max(t1_firing, key=lambda r: r.weight * r.output_pain_level)
            return best.output_pain_level, 1, best.explanation, updated_var

        # Step 4: Check Tier 2
        t2_firing = [r for r in self.tier2_rules if rule_fires(r)]
        if t2_firing:
            total_w = sum(r.weight for r in t2_firing)
            y_hat = sum(r.weight * r.output_pain_level for r in t2_firing) / total_w
            explanations = "; ".join(r.explanation for r in t2_firing)
            return y_hat, 2, f"Consensus: {explanations}", updated_var

        # Step 6: Tier 3 — conflicting signals
        t3_firing = [r for r in self.tier3_rules if rule_fires(r)]
        if t3_firing:
            # Escalate σ² by +0.15 (raises to 0.5 > τ* = 0.35 → always alerts)
            updated_var = current_var + TIER3_ESCALATION
            explanations = "; ".join(r.explanation for r in t3_firing)

            if updated_var > 0.5:
                return 0.0, 3, f"ABSTAIN: {explanations}", updated_var
            else:
                return 0.0, 3, f"Uncertain: {explanations}", updated_var

        # No rules fired
        return 0.0, 3, "No rule fired; default to uncertain", updated_var

    def batch_evaluate(
        self,
        concept_activations: torch.Tensor,
        current_vars: torch.Tensor,
    ) -> Dict[str, list]:
        """
        Evaluate rules for a batch of samples.

        Args:
            concept_activations: (B, N_CONCEPTS)
            current_vars:        (B,)

        Returns:
            Dict with lists of y_hat, tier, explanation, updated_var.
        """
        results = {"y_hat": [], "tier": [], "explanation": [], "updated_var": []}
        for i in range(concept_activations.size(0)):
            y, t, e, v = self.evaluate(
                concept_activations[i], float(current_vars[i])
            )
            results["y_hat"].append(y)
            results["tier"].append(t)
            results["explanation"].append(e)
            results["updated_var"].append(v)
        return results

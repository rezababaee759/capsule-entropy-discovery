"""The Hub/Specialist taxonomy rule and the role names.

A capsule is "high" for a class when its importance reaches that class's 90th
percentile over capsules, and "moderate" when it reaches the 50th.
"""

from __future__ import annotations

import numpy as np

#: Role keys in the order the rule tests them; also the key order of hub_taxonomy.json.
ROLES = ("primary_hubs", "secondary_hubs", "generalists", "specialists", "non_critical")

#: Role keys from most to least important: the order of tables and of the
#: beta-ordering hypothesis.
ROLE_ORDER = (
    "primary_hubs",
    "secondary_hubs",
    "specialists",
    "generalists",
    "non_critical",
)

ROLE_NAMES = {
    "primary_hubs": "Primary Hub",
    "secondary_hubs": "Secondary Hub",
    "generalists": "Generalist",
    "specialists": "Specialist",
    "non_critical": "Non-critical",
}

HUB_PCT = 90
MODERATE_PCT = 50
PRIMARY_MIN = 7
SECONDARY_MIN = 4
SPECIALIST_MAX = 2


def threshold_counts(
    importance_matrix: np.ndarray, hub_pct=HUB_PCT, moderate_pct=MODERATE_PCT
):
    """For a [n_classes, n_capsules] matrix: per capsule, the number of classes for which
    it is high and moderate, and the per-class high and moderate thresholds."""
    high = np.percentile(importance_matrix, hub_pct, axis=1, keepdims=True)
    moderate = np.percentile(importance_matrix, moderate_pct, axis=1, keepdims=True)
    return (
        (importance_matrix >= high).sum(0),
        (importance_matrix >= moderate).sum(0),
        high.ravel(),
        moderate.ravel(),
    )


def assign_role(
    high_count,
    moderate_count,
    primary_min=PRIMARY_MIN,
    secondary_min=SECONDARY_MIN,
    specialist_max=SPECIALIST_MAX,
) -> str:
    """The rule. Branch order matters: Generalists are tested before Specialists, and
    Non-critical is the residual (it includes capsules moderate for 7 or more classes).
    """
    if high_count >= primary_min:
        return "primary_hubs"
    if high_count >= secondary_min:
        return "secondary_hubs"
    if 3 <= moderate_count <= 6:
        return "generalists"
    if 1 <= high_count <= specialist_max:
        return "specialists"
    return "non_critical"


def classify(
    importance_matrix: np.ndarray,
    hub_pct=HUB_PCT,
    moderate_pct=MODERATE_PCT,
    primary_min=PRIMARY_MIN,
    secondary_min=SECONDARY_MIN,
    specialist_max=SPECIALIST_MAX,
) -> dict:
    """Role key -> ascending capsule indices, keys in ROLES order."""
    high_count, moderate_count, _, _ = threshold_counts(
        importance_matrix, hub_pct, moderate_pct
    )
    roles = {role: [] for role in ROLES}
    for i, (h, m) in enumerate(zip(high_count, moderate_count)):
        roles[assign_role(h, m, primary_min, secondary_min, specialist_max)].append(i)
    return roles


#: Role names in ROLE_ORDER.
ORDERED_ROLE_NAMES = tuple(ROLE_NAMES[key] for key in ROLE_ORDER)


def capsule_roles(tax: dict) -> dict:
    """Capsule index -> role name, from a hub_taxonomy.json dict. Insertion order is
    ROLE_ORDER, then each role's list order."""
    m = {}
    for key in ROLE_ORDER:
        for c in tax.get(key, []):
            m[int(c)] = ROLE_NAMES[key]
    return m


class RoleCoverageError(RuntimeError):
    """The role map does not give every capsule index exactly one of the five roles."""


def assert_role_coverage(capsule_to_role: dict, num_capsules: int) -> None:
    """Every capsule index must carry exactly one of the five contract roles."""
    missing = [i for i in range(num_capsules) if i not in capsule_to_role]
    unexpected = sorted(
        {r for r in capsule_to_role.values() if r not in ORDERED_ROLE_NAMES}
    )
    if missing or unexpected:
        raise RoleCoverageError(
            f"role map does not cover the capsule index space: "
            f"{len(missing)} of {num_capsules} indices unmapped"
            + (f" (first: {missing[:8]})" if missing else "")
            + (f"; unexpected role name(s): {unexpected}" if unexpected else "")
            + ". Check that the taxonomy emits the keys "
            f"{sorted(ROLE_NAMES)} for this model."
        )


def load_importance_matrix(tax_json) -> np.ndarray:
    """The [n_classes, n_capsules] importance_matrix.npy beside a hub_taxonomy.json."""
    return np.load(str(tax_json).replace("hub_taxonomy.json", "importance_matrix.npy"))

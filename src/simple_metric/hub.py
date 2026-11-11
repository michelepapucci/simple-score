"""Names of the released SIMPLE checkpoints on the Hugging Face Hub."""

from __future__ import annotations

HUB_NAMESPACE = "mpapucci"

#: Short aliases accepted by :meth:`simple_metric.SIMPLE.from_pretrained`.
MODEL_ALIASES = {
    "small": f"{HUB_NAMESPACE}/simple-metric-small",
    "base": f"{HUB_NAMESPACE}/simple-metric-base",
    "large": f"{HUB_NAMESPACE}/simple-metric-large",
}

DEFAULT_MODEL = "large"


def resolve_model_name(name_or_path: str) -> str:
    """Map a short alias (``"small"``, ``"base"``, ``"large"``) to its Hub repo id."""
    key = name_or_path.lower()
    for prefix in ("simple-", "simple_"):
        if key.startswith(prefix) and key[len(prefix):] in MODEL_ALIASES:
            key = key[len(prefix):]
    return MODEL_ALIASES.get(key, name_or_path)

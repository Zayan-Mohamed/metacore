"""Module 4 Semantic Translation Package — Abductive Causal Logging.

ZERO ML DEPENDENCIES.
"""

from .abductive.attribution import AbductiveAttributor
from .templates.causal_logger import TemplateCausalLogger
from .types import CausalLog, Generator

__all__ = [
    "AbductiveAttributor",
    "TemplateCausalLogger",
    "CausalLog",
    "Generator",
]

"""rellm — REA-style reverse engineering of LLM internals (logit lens, attention, patching, DLA, SAEs) over MCP."""
from ._version import __version__
from .evidence import EvidenceStore, EvidenceError
from .session import ModelSession, SessionError, Budget
from . import analysis

__all__ = ["ModelSession", "SessionError", "Budget", "EvidenceStore", "EvidenceError", "analysis", "__version__"]

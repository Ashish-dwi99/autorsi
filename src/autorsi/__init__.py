"""AutoRSI: bounded, evaluation-gated weight editing. No network I/O on import."""
from .backends import InMemoryBackend, ScalarBackend, StaleRevisionError
from .contracts import (Backend, EditRecipe, Evaluator, Goal, Guard, ModelState,
                        ProposalContext, Proposer, RunLimits, RunResult, TensorEdit, ValidationError)
from .engine import AutoRSI, RollbackJournalError
from .proposers import JSONProposer, SequenceProposer, parse_recipe
from .torch_backend import TorchBackend

__version__ = "0.1.0"
__all__ = ["AutoRSI", "Backend", "EditRecipe", "Evaluator", "Goal", "Guard", "InMemoryBackend",
           "JSONProposer", "ModelState", "ProposalContext", "Proposer", "RunLimits", "RunResult",
           "ScalarBackend", "SequenceProposer", "StaleRevisionError", "TensorEdit", "TorchBackend", "parse_recipe",
           "ValidationError", "RollbackJournalError"]

"""Wesker — in-process AST mutation testing for Python.

Zero dependencies beyond the test framework. Categorical mutant stratification
(VALUE, BOUNDARY, SWAP, STATE, TYPE, ARITHMETIC, LOGICAL) with Monty Hall
filtering, 3-layer test discovery, equivalent mutant detection, and MC/DC
verification.
"""

# THE one owner of this number. `pyproject.toml` declares `dynamic = ["version"]` and
# `[tool.hatch.version] path = "Wesker/__init__.py"`, so the build reads it from HERE — bump this
# and nothing else. (This comment used to say "keep in lockstep with pyproject's version — this is
# restated... bump both, or neither", which was true until the number moved here and stopped being
# restated. Following it now would put a second copy back in pyproject and recreate precisely the
# drift going dynamic removed: 0.6.0 shipped to PyPI announcing itself as 0.5.1.)
__version__ = "0.9.2"

from .engine import (
    BoundaryInput,
    CategoryResult,
    Mutant,
    MutantResult,
    MutationCategory,
    ProfilingResult,
    SamplingResult,
    check_equivalent,
    estimate_universe_size,
    evaluate_mutant,
    extract_boundary_inputs,
    generate_mutants,
    run_function_converged,
    run_function_profiling,
    run_function_sampling,
)
from .filter import CategoryPrior, filter_categories, prioritize_categories

__all__ = [
    # Enums
    "MutationCategory",
    # Result types
    "BoundaryInput",
    "CategoryPrior",
    "CategoryResult",
    "Mutant",
    "MutantResult",
    "ProfilingResult",
    "SamplingResult",
    # Engine functions
    "check_equivalent",
    "estimate_universe_size",
    "evaluate_mutant",
    "extract_boundary_inputs",
    "generate_mutants",
    "run_function_converged",
    "run_function_profiling",
    "run_function_sampling",
    # Filter functions
    "filter_categories",
    "prioritize_categories",
]

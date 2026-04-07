"""Wesker — in-process AST mutation testing for Python.

Zero dependencies beyond the test framework. Categorical mutant stratification
(VALUE, BOUNDARY, SWAP, STATE, TYPE, ARITHMETIC, LOGICAL) with Monty Hall
filtering, 3-layer test discovery, equivalent mutant detection, and MC/DC
verification.
"""

__version__ = "0.1.0"

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

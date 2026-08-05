"""The versioned mutation policy (issue #8): identity, derivation, enforcement.

Two goldens carry the enforcement story. The policy-id golden makes "a policy
change without a version bump" a test failure instead of a silent drift; the
fingerprint-counts golden is the same fact in a readable shape, so the failure
names the category that moved instead of handing you a hash. Everything else
pins that the manifest is DERIVED from the engine's own tables — the manifest
lying about the engine would be the same second-predicate disease the counters
had (issue #9), one level up.
"""

from __future__ import annotations

import ast
import json
from dataclasses import asdict

from Wesker.engine import (
    _EXCEPTION_SUB_MODES,
    _STATE_SUB_MODES,
    MutationCategory,
    _SwapMutator,
    estimate_universe_size,
    generate_mutants,
)
from Wesker.policy import (
    _FINGERPRINT_CORPUS,
    POLICY_VERSION,
    mutation_policy,
)

# ── The goldens ──────────────────────────────────────────────────

# THE policy identity. When this fails, one of two things is true:
#   * you changed the mutation universe ON PURPOSE — bump POLICY_VERSION in
#     Wesker/policy.py, re-run `python -c "from Wesker import mutation_policy;
#     print(mutation_policy().policy_id)"`, and update this golden (and the
#     counts below, which will name what moved); or
#   * you changed eligibility semantics BY ACCIDENT — a mutator, a counter, a
#     skip rule — and this failure is the entire point: find the drift before
#     touching the golden.
GOLDEN_POLICY_ID = "5.751e8e9f4f11"  # 5: type-impossible arithmetic leaves the universe

# The same fact, readable: the engine's target counts over the fingerprint
# corpus. A failure here names the category and function that moved.
GOLDEN_FINGERPRINT = {
    "fp_value": {"VALUE": 9, "STATE": 1, "STMT": 1},
    "fp_boundary": {"BOUNDARY": 17, "STATE": 1},
    "fp_arith_logic": {
        "ARITHMETIC": 4,
        "LOGICAL": 2,
        "STATE": 1,
        "STMT": 1,
        "VALUE": 2,
    },
    "fp_swap": {"ARITHMETIC": 1, "STATE": 1, "SWAP": 9, "VALUE": 4},
    "fp_state_type_exc": {
        "ARITHMETIC": 1,
        "BOUNDARY": 3,
        "EXCEPTION": 3,
        "LOGICAL": 1,
        "STATE": 6,
        "STMT": 3,
        "SWAP": 4,
        "TYPE": 1,
        "VALUE": 4,
    },
    "fp_stmt": {"DATAFLOW": 2, "STATE": 1, "STMT": 3, "SWAP": 1, "VALUE": 7},
    "fp_async": {"ARITHMETIC": 1, "STATE": 1, "VALUE": 2},
    # The enrolled slice: return total over both visible candidates. (name_sub
    # would add x→y/y→x at the binding statement — implemented, not enrolled;
    # see _DATAFLOW_SUB_MODES for the measured reason.)
    "fp_dataflow": {"ARITHMETIC": 1, "DATAFLOW": 2, "STATE": 1},
    # Policy 3: one STATE target per self-write SPELLING (plain, annotated,
    # augmented); the valueless declaration is not a write. STMT's overlap on
    # attribute writes extends symmetrically; ARITHMETIC is the += itself.
    # Policy 4: the two-target write adds one dimension PER attribute (left,
    # right), each with its own per-target mutant.
    "fp_state_spellings": {"ARITHMETIC": 1, "STATE": 5, "STMT": 4},
    # Policy 5: only the one-sided f-string Add, `n + 1`, and the name-name
    # return Add survive as arithmetic; the two-str Adds and `'-' * 3` are
    # type-impossible. VALUE 9 = four str literals + two ints (two dims each)
    # + the f-string's '!' fragment.
    "fp_str_arith": {"ARITHMETIC": 3, "STATE": 1, "SWAP": 1, "VALUE": 9},
}


def test_policy_id_is_the_golden():
    assert mutation_policy().policy_id == GOLDEN_POLICY_ID, (
        "The mutation-policy fingerprint moved. Intentional universe change → "
        "bump POLICY_VERSION and update GOLDEN_POLICY_ID (+ GOLDEN_FINGERPRINT, "
        "which names what moved). Unintentional → you changed eligibility "
        "semantics by accident; find the drift before touching the golden."
    )


def test_fingerprint_counts_are_the_goldens():
    got = mutation_policy().fingerprint_counts
    want = {
        fn: {cat.value: by_cat.get(cat.value, 0) for cat in MutationCategory}
        for fn, by_cat in GOLDEN_FINGERPRINT.items()
    }
    assert got == want


def test_policy_id_embeds_the_version():
    p = mutation_policy()
    assert p.policy_id.startswith(f"{POLICY_VERSION}.")
    assert p.policy_version == POLICY_VERSION


# ── Manifest ↔ engine derivation (no second predicate, one level up) ──


def test_manifest_covers_exactly_the_enum():
    assert set(mutation_policy().categories) == {c.value for c in MutationCategory}


def test_manifest_sub_modes_are_the_engine_tuples():
    cats = mutation_policy().categories
    assert cats["STATE"]["sub_modes"] == dict(_STATE_SUB_MODES)
    assert cats["EXCEPTION"]["sub_modes"] == dict(_EXCEPTION_SUB_MODES)


def test_manifest_duals_are_the_engine_table():
    assert mutation_policy().categories["SWAP"]["duals"] == _SwapMutator._DUALS


def test_first_order_semantics_declared():
    # Issue #11: the completeness claim names its concept class — first-order
    # mutants generated from the original AST, never operator compositions.
    p = mutation_policy()
    assert p.order == 1
    assert p.generation == "original-ast"
    assert any("first-order" in e for e in p.exclusions)


def test_to_json_is_canonical_and_round_trips():
    p = mutation_policy()
    blob = p.to_json()
    assert json.loads(blob) == json.loads(json.dumps(asdict(p), sort_keys=True))
    assert "\n" not in blob


# ── Universe accounting over the corpus (the counter IS the generator's
#    index space — the invariant the record-mode migration must keep) ──


def _corpus_nodes() -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    nodes = []
    for src in _FINGERPRINT_CORPUS:
        node = ast.parse(src).body[0]
        assert isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        nodes.append(node)
    return nodes


def test_every_generated_mutant_indexes_inside_the_counted_universe():
    for node in _corpus_nodes():
        for cat in MutationCategory:
            count = estimate_universe_size(node, {cat})
            mutants = generate_mutants(node, {cat}, max_per_category=0)
            assert len(mutants) <= count, (node.name, cat.value)
            assert all(m.target_index < count for m in mutants), (
                node.name,
                cat.value,
            )
            if count == 0:
                assert mutants == []


def test_state_dof_mode_covers_one_mutant_per_live_dimension():
    # fp_state_type_exc's STATE dimensions: remove_assign collapses three
    # self.total sites into ONE attribute dimension; return_none is one;
    # loop_flow carries break and continue as two. DOF mode (budget None)
    # spends exactly one mutant per live dimension.
    node = next(n for n in _corpus_nodes() if n.name == "fp_state_type_exc")
    mutants = generate_mutants(node, {MutationCategory.STATE}, max_per_category=None)
    assert len(mutants) == 4
    assert {m.dimension for m in mutants} == {
        "STATE:remove_assign:total",
        "STATE:return_none",
        "STATE:loop_flow:break",
        "STATE:loop_flow:continue",
    }


def test_state_spelling_dimensions_exact():
    # Direct record-mode pin (the golden tests reach these visits only through
    # the @cache'd mutation_policy(), which hides mutants from the suite):
    # exact labels, exact order, valueless declaration excluded.
    from Wesker.engine import _record_state_dimensions

    node = next(n for n in _corpus_nodes() if n.name == "fp_state_spellings")
    assert _record_state_dimensions(node, "remove_assign") == [
        "STATE:remove_assign:plain",
        "STATE:remove_assign:annotated",
        "STATE:remove_assign:augmented",
        "STATE:remove_assign:left",
        "STATE:remove_assign:right",
    ]


def test_every_write_spelling_index_is_reachable():
    # Exhaustive generation must produce ONE mutant per write, exactly — a
    # dead index increment in any visit silently drops every later same-mode
    # site (the ≤-count invariant tolerates that; this equality does not).
    # The non-self annotated write must contribute nothing.
    fn = ast.parse(
        "def m(self, v):\n"
        "    self.a = v\n"
        "    self.b: int = v\n"
        "    self.c += v\n"
        "    self.d = v\n"
        "    w: int = v\n"
    ).body[0]
    assert isinstance(fn, ast.FunctionDef)
    mutants = generate_mutants(fn, {MutationCategory.STATE}, max_per_category=0)
    assert sorted(m.dimension for m in mutants) == [
        "STATE:remove_assign:a",
        "STATE:remove_assign:b",
        "STATE:remove_assign:c",
        "STATE:remove_assign:d",
    ]


def test_multi_target_write_unbinds_per_target():
    """Policy 4's regression: `self.a = self.b = x` is two questions, two
    DISTINCT mutants — drop a keeping b, drop b keeping a — never two
    dimensions sharing one byte-identical statement->pass mutant. The
    single-target statement still collapses to pass, byte-identical to the
    policy-3 universe."""
    fn = ast.parse(
        "def m(self, x):\n    self.a = self.b = x\n    self.solo = x\n"
    ).body[0]
    assert isinstance(fn, ast.FunctionDef)
    mutants = generate_mutants(fn, {MutationCategory.STATE}, max_per_category=0)
    by_dim = {m.dimension: m for m in mutants}
    assert set(by_dim) == {
        "STATE:remove_assign:a",
        "STATE:remove_assign:b",
        "STATE:remove_assign:solo",
    }
    ids = [m.mutant_id for m in mutants]
    assert len(set(ids)) == len(ids), "byte-identical mutants are back"

    def line(m, i):
        return ast.unparse(m.mutated_node).splitlines()[i].strip()

    assert line(by_dim["STATE:remove_assign:a"], 1) == "self.b = x"
    assert line(by_dim["STATE:remove_assign:b"], 1) == "self.a = x"
    assert line(by_dim["STATE:remove_assign:solo"], 2) == "pass"


def test_single_statement_body_removal_stays_compilable():
    # `pass`-replacement, never statement DELETION: for a body of exactly one
    # write, deletion would leave an empty body — an unparseable mutant. The
    # generated mutant must exist and compile.
    for body in ("self.x += v", "self.x = v"):
        fn = ast.parse(f"def m(self, v):\n    {body}\n").body[0]
        assert isinstance(fn, ast.FunctionDef)
        mutants = generate_mutants(fn, {MutationCategory.STATE}, max_per_category=0)
        assert [m.dimension for m in mutants] == ["STATE:remove_assign:x"], body
        compile(
            ast.fix_missing_locations(ast.Module([mutants[0].mutated_node], [])),
            "<m>",
            "exec",
        )


def test_annassign_removal_drops_the_write():
    # The annotated spelling's mutant replaces exactly its own statement.
    node = next(n for n in _corpus_nodes() if n.name == "fp_state_spellings")
    mutants = generate_mutants(node, {MutationCategory.STATE}, max_per_category=0)
    (annotated,) = [
        m for m in mutants if m.dimension == "STATE:remove_assign:annotated"
    ]
    body = ast.unparse(annotated.mutated_node).splitlines()
    assert "self.plain = v" in body[1]
    assert body[2].strip() == "pass"
    assert "self.augmented += v" in body[3]


def test_augassign_removal_keeps_the_prior_value():
    # The policy-3 semantic choice, executed: dropping `self.x += v` leaves
    # the attribute at its prior value (the STMT rebinding rationale), so the
    # mutant is killable exactly by a test that observes the update.
    src = (
        "class C:\n"
        "    def __init__(self):\n"
        "        self.total = 10\n"
        "    def bump(self, v):\n"
        "        self.total += v\n"
        "        return self.total\n"
    )
    bump = ast.parse(src).body[0].body[1]
    assert isinstance(bump, ast.FunctionDef)
    (mutant,) = [
        m
        for m in generate_mutants(bump, {MutationCategory.STATE}, max_per_category=0)
        if m.dimension == "STATE:remove_assign:total"
    ]

    ns_orig: dict = {}
    ns_mut: dict = {}
    exec(src, ns_orig)
    mutated_cls = ast.parse(src)
    mutated_cls.body[0].body[1] = mutant.mutated_node
    exec(compile(ast.fix_missing_locations(mutated_cls), "<m>", "exec"), ns_mut)
    c_orig, c_mut = ns_orig["C"](), ns_mut["C"]()
    c_orig.bump(5)
    c_mut.bump(5)
    assert c_orig.total == 15
    assert c_mut.total == 10  # prior value survives the dropped update


def test_state_non_greedy_generation_keeps_the_empty_dimension_contract():
    # The legacy non-greedy path never recorded dimensions; the record-mode
    # counting migration must not silently start stamping them.
    node = next(n for n in _corpus_nodes() if n.name == "fp_state_type_exc")
    mutants = generate_mutants(
        node, {MutationCategory.STATE}, max_per_category=0, greedy=False
    )
    assert len(mutants) == 6  # every STATE target applies: 3 + 1 + 2
    assert all(m.dimension == "" for m in mutants)

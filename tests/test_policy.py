"""The versioned mutation policy (issue #8): identity, derivation, enforcement.

Two goldens carry the enforcement story. The policy-id golden makes an unreviewed
policy change a test failure instead of a silent drift; the
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
    _OUTPUT_CALL_SUB_MODES,
    _OUTPUT_SUB_MODES,
    _OUTPUT_TYPE_CONDITIONAL,
    _OUTPUT_YIELD_SUB_MODES,
    _STATE_SUB_MODES,
    MutationCategory,
    _SwapMutator,
    estimate_universe_size,
    generate_mutants,
)
from Wesker.mu_minus import MU_MINUS_CATALOG
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
#   * you repaired observation semantics without changing the universe — retain
#     POLICY_VERSION, verify categories/counts unchanged, and update the id with
#     the repair receipt so older observation results cannot share its basis; or
#   * you changed eligibility semantics BY ACCIDENT — a mutator, a counter, a
#     skip rule — and this failure is the entire point: find the drift before
#     touching the golden.
# W#47: the same policy-8 universe, with single-invocation observation semantics.
# Policy 9: an existing `return None` is no longer a STATE return_none target (the rewrite was the
# original program); the corpus gained fp_return_none.
# W#27: the same policy-9 universe; each mutant observation is bounded by the calibrated loop
# budget (32x the original's count on that test, plus 10,000), whose time is never charged to the
# mutant's allowance, and the equivalence probes are budgeted too.
# Detective #105: the same policy-9 universe; each test run against a mutant writes only inside its
# write authority, and a prevented write is the unscored outcome write_outside_authority, never a kill
# (categories and counts verified unchanged: GOLDEN_FINGERPRINT holds).
# Wesker #55: the same policy-9 universe; every evaluation receives the test's arguments as authored
# (fresh copies, either mode), and a failure the ORIGINAL reproduces in the same process state is the
# unscored outcome original_also_fails, never a kill (categories and counts verified unchanged:
# GOLDEN_FINGERPRINT holds). One-sign id 9.261d14e34119 -> 9.415d3755221e -> 9.008828e5f6a9 (the copy is
# a measured precondition: an argument the original fails on as a copy stays shared).
GOLDEN_POLICY_ID = "9.008828e5f6a9"

# The same fact, readable: the engine's target counts over the fingerprint
# corpus. A failure here names the category and function that moved.
GOLDEN_FINGERPRINT = {
    "fp_annotations": {"ARITHMETIC": 1, "DATAFLOW": 1, "STATE": 1, "VALUE": 4},
    "fp_value": {"STATE": 1, "STMT": 1, "VALUE": 9},
    "fp_boundary": {"BOUNDARY": 17, "STATE": 1},
    "fp_arith_logic": {
        "ARITHMETIC": 4,
        "LOGICAL": 2,
        "STATE": 1,
        "STMT": 1,
        "VALUE": 2,
    },
    "fp_bitwise": {"ARITHMETIC": 8, "STATE": 1, "STMT": 1, "VALUE": 2},
    "fp_swap": {"ARITHMETIC": 1, "STATE": 1, "SWAP": 9, "VALUE": 4},
    # Policy 7's witness. Every OTHER corpus call has at most two positional arguments, where
    # "every pair" and "every adjacent pair" coincide — so this row is the only one that moves
    # when the argument-order vocabulary does. SWAP 15 = g(a,b,c)'s three pairs + its unwrap,
    # plus h(...)'s ten ASKED pairs (of C(6,2)=15, five withheld) + its unwrap.
    "fp_swap_wide": {"ARITHMETIC": 1, "STATE": 1, "SWAP": 15},
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
    "fp_dataflow": {"ARITHMETIC": 1, "DATAFLOW": 2, "STATE": 1},
    "fp_str_arith": {"ARITHMETIC": 3, "STATE": 1, "SWAP": 1, "VALUE": 9},
    "fp_state_spellings": {"ARITHMETIC": 1, "STATE": 5, "STMT": 4},
    "fp_return_none": {"BOUNDARY": 3, "STATE": 1},
}


# The TWO-SIGN fingerprint's OUTPUT column, readable (no hash golden: the two-sign id also moves
# with every observation-text repair the one-sign golden above records). Fork 1 per return site
# (none, const, identity; identity needs a parameter, and a no-op is not a target), plus Form B
# (#44): return_stale on every non-generator, and raise_swallow where the function's own body
# raises (fp_state_type_exc). The corpus holds no generator, so no yield mode appears here.
# #44 on policy 9: each row is one more than before (two more for fp_state_type_exc), and the
# one-sign id is unchanged. Two-sign id: 9+neg.aaff90df3dbf -> 9+neg.e273b30c9a36 (Form A's Fork 2
# table enters the manifest) -> 9+neg.8144c37b1638 (Form B and the unbuilt catalog) ->
# 9+neg.f5a15364c11d (integrated with W#27's loop budget and #105's write authority, both
# observation texts the one-sign golden records) -> 9+neg.001902a9950c (W#55's fresh-argument and
# original-control observation text) -> 9+neg.89326e7f549c (W#49: the stale wrapper's memory ends at
# a test boundary; the one-sign id is unchanged by it). The identity-sensitivity rule above then moved
# both ids: two-sign 9+neg.89326e7f549c -> 9+neg.1a13fddd1d5f.
GOLDEN_TWO_SIGN_OUTPUT = {
    "fp_annotations": 4,
    "fp_value": 3,
    "fp_boundary": 4,
    "fp_arith_logic": 4,
    "fp_bitwise": 4,
    "fp_swap": 4,
    "fp_swap_wide": 4,
    "fp_state_type_exc": 5,
    "fp_stmt": 4,
    "fp_async": 4,
    "fp_dataflow": 4,
    "fp_str_arith": 4,
    "fp_state_spellings": 1,
    # `return None` -> const, identity; `return x` -> none, const (identity is a no-op); + stale.
    "fp_return_none": 5,
}


def test_policy_id_is_the_golden():
    assert mutation_policy().policy_id == GOLDEN_POLICY_ID, (
        "The mutation-policy fingerprint moved. Intentional universe change → "
        "bump POLICY_VERSION and update GOLDEN_POLICY_ID (+ GOLDEN_FINGERPRINT, "
        "which names what moved). Observation-only repair → verify unchanged "
        "categories/counts and record the new observation id. Unintentional → "
        "you changed eligibility "
        "semantics by accident; find the drift before touching the golden."
    )


def test_fingerprint_counts_are_the_goldens():
    got = mutation_policy().fingerprint_counts
    # The DEFAULT policy's fingerprint covers the one-sign categories only — OUTPUT (μ⁻) is
    # withheld, which is exactly what keeps the default id byte-identical. Build `want` over the
    # same set, so this golden pins the one-sign fingerprint and not a spurious OUTPUT:0 column.
    want = {
        fn: {
            cat.value: by_cat.get(cat.value, 0)
            for cat in MutationCategory
            if cat is not MutationCategory.OUTPUT
        }
        for fn, by_cat in GOLDEN_FINGERPRINT.items()
    }
    assert got == want


def test_two_sign_output_column_is_the_golden():
    two = mutation_policy(two_sign=True).fingerprint_counts
    assert {fn: cats["OUTPUT"] for fn, cats in two.items()} == GOLDEN_TWO_SIGN_OUTPUT
    # Every other column is the one-sign fingerprint's: μ⁻ adds questions, it moves none.
    one = mutation_policy().fingerprint_counts
    assert all(
        {cat: n for cat, n in two[fn].items() if cat != "OUTPUT"} == one[fn]
        for fn in one
    )


def test_policy_id_embeds_the_version():
    p = mutation_policy()
    assert p.policy_id.startswith(f"{POLICY_VERSION}.")
    assert p.policy_version == POLICY_VERSION


# ── Manifest ↔ engine derivation (no second predicate, one level up) ──


def test_manifest_covers_exactly_the_enum():
    # The DEFAULT one-sign policy covers every category EXCEPT OUTPUT (μ⁻), which is withheld
    # unless the two-sign policy is requested; the two-sign policy covers the full enum.
    assert set(mutation_policy().categories) == {
        c.value for c in MutationCategory if c is not MutationCategory.OUTPUT
    }
    assert set(mutation_policy(two_sign=True).categories) == {
        c.value for c in MutationCategory
    }


def test_two_sign_policy_is_a_distinct_declared_policy():
    # σ(P, μ ∪ μ⁻) is a genuinely different universe, so it earns its own id — while the
    # DEFAULT one-sign id stays byte-identical (the OUTPUT enum member is present but withheld).
    one = mutation_policy()
    two = mutation_policy(two_sign=True)
    assert one.policy_id == GOLDEN_POLICY_ID
    assert two.policy_id != one.policy_id
    assert two.policy_id.startswith(f"{POLICY_VERSION}+neg.")
    assert "OUTPUT" in two.categories and "OUTPUT" not in one.categories


def test_manifest_sub_modes_are_the_engine_tuples():
    cats = mutation_policy().categories
    assert cats["STATE"]["sub_modes"] == dict(_STATE_SUB_MODES)
    assert cats["EXCEPTION"]["sub_modes"] == dict(_EXCEPTION_SUB_MODES)
    assert mutation_policy(two_sign=True).categories["OUTPUT"]["sub_modes"] == dict(
        _OUTPUT_SUB_MODES
    )


def test_two_sign_manifest_declares_the_type_conditional_table():
    # #44: Fork 2's table is the question set a two-sign run asks of an observed codomain. It was
    # left out of the manifest ("type-conditional perturbations are out of this policy"), so adding
    # 23 operators to it would not have moved the two-sign id, and a receipt measured before them
    # would have read as current. Derived from the engine table, never restated.
    two = mutation_policy(two_sign=True)
    assert two.categories["OUTPUT"]["type_conditional"] == {
        mode: {"question": desc, "types": sorted(types)}
        for mode, desc, types in _OUTPUT_TYPE_CONDITIONAL
    }
    # The id before #44 (Wesker 204a760, policy 9). Its universe lacked the new operators.
    assert two.policy_id != "9+neg.aaff90df3dbf"


def test_two_sign_manifest_declares_form_b_and_what_it_does_not_measure():
    # Form B is static (no observed type decides it), so it is in the census; the manifest names
    # its modes from the engine tables and, from the catalog, every Def. 11.8 operator a two-sign
    # certificate does NOT measure, so a receipt cannot be read as covering the whole family.
    two = mutation_policy(two_sign=True)
    form_b = two.categories["OUTPUT"]["form_b"]
    assert form_b["yield"] == dict(_OUTPUT_YIELD_SUB_MODES)
    assert form_b["call"] == dict(_OUTPUT_CALL_SUB_MODES)
    unbuilt = {
        op.key for op in MU_MINUS_CATALOG if op.status in ("partial", "unsupported")
    }
    assert unbuilt == {"dict", "field", "class", "clamp", "raise", "state-extra"}


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
    # S102: executing the original and mutated class is the observation under test
    exec(src, ns_orig)  # noqa: S102
    mutated_cls = ast.parse(src)
    mutated_cls.body[0].body[1] = mutant.mutated_node
    # S102: as above, for the mutant
    exec(compile(ast.fix_missing_locations(mutated_cls), "<m>", "exec"), ns_mut)  # noqa: S102
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

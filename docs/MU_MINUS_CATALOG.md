# μ⁻ operator catalog

The negative operator μ⁻ of the two-sign contract σ(P, μ ∪ μ⁻) perturbs a function's **output**. A
perturbation that survives the suite is an output invariant the suite does not pin. Def. 11.8 of
Detective's `docs/theory/NEGATIVE_SPECIFICATION.md` enumerates the family Π: 38 operators in eight
codomain families, counting the string and container `empty` once.

This file states, for each operator:
- whether Wesker generates it, and how;
- where it applies;
- the evidence that it is sound.

The source of truth is `Wesker.mu_minus.MU_MINUS_CATALOG`, together with the engine's mode tables.
`tests/test_mu_minus_catalog_doc_intent.py` checks this file against both, so the suite fails on any
change this file does not follow. That covers an operator's form, status or sub-modes, a Fork 2
mode's applicable types, and a cited test that no longer exists.

μ⁻ is opt-in: Detective's `--two-sign`, Wesker's `filter_categories(node, two_sign=True)`. The
default one-sign policy generates no `OUTPUT` mutant, and its id does not move with anything here.

## Summary

| standing | operators |
|---|---|
| built | 30 |
| partial | 2 (dict, raise) |
| realized by another operator | 2 (0, state-noop) |
| unsupported | 4 (field, class, clamp, state-extra) |

The two-sign policy manifest names every partial and unsupported operator with its reason
(`negative_policy.unbuilt`), so a two-sign certificate cannot be read as covering all 38.

## How an operator is delivered

**Form A (Def. 11.4): a return-site rewrite.** `return <expr>` becomes `return p(<expr>)`, an
ordinary AST mutant. It is compiled, evaluated, TCE-checked and boundary-probed like every other.
Two forks decide where a Form A operator applies (Def. 11.10):

- **Fork 1, always applicable.** none, const and identity are total on any value. They apply to
  every `return <expr>`; identity needs a parameter other than `self`/`cls`.
- **Fork 2, type-conditional.** The mode is emitted only where EVERY observed return type is in its
  applicable types: `output_mode_applies`, the subset condition observed ⊆ types, never mere
  intersection. Form A rewrites the return site statically, so a mode one branch cannot take would
  raise on that branch. A heterogeneous return (`int | str`) therefore forgoes it. The observed
  codomain is Detective's harvest of `type(return).__name__` over the tests the run consults.

The Fork 2 rewrites added by #44 are expression templates (`_OUTPUT_TEMPLATES`):
- Each evaluates the return expression **exactly once**, so a perturbation is p ∘ f. The
  expression's own side effects happen as they do in f. A constant (→True) does not skip them, and
  a value used twice (duplicate) does not repeat them.
- Builtins and stdlib modules are reached through `__import__`, so a parameter or global named
  `type`, `min` or `round` cannot redirect them.
- A set element is chosen by least `repr`, never by iteration order, which PYTHONHASHSEED moves
  between processes.
- The modes that predate #44 keep their original AST, because their ids and diffs are what `flag`
  records.

**Form B (Def. 11.6): what the return-site rewrite cannot reach.** This is a generator's yielded
sequence, a return across calls, or a raise. A Form B mutant is **source**: the target's own
definition with `@__import__("Wesker.mu_minus", fromlist=["form_b"]).form_b("<mode>")` applied
INNERMOST.
- Every evaluator builds it the same way, the isolated worker included, since it receives nothing
  but source.
- The target's own decorators wrap the perturbation as they wrap the function: `classmethod` binds
  `cls` once, and `property` and `lru_cache` see it.
- The four yield modes keep their legacy marker as their identity, and the isolated worker gets
  their decorated twin (`isolated_mutant_node`).

The pure decision `form_b_output_modes(is_generator, has_own_raise)` gives Form B's applicability,
read off the function's OWN body (nested scopes excluded):
- a generator gets the yield modes;
- any other function gets `return_stale`, plus `raise_swallow` where its own body raises.

## The four states

- **built**: generated; the sub-modes listed realize it.
- **partial**: some variants generated; the rest are named, with the reason they are not.
- **realized by**: an existing operator already asks exactly this question, so a second mutant would
  duplicate it.
- **unsupported**: not generated, because it cannot be built soundly; the reason is stated.

## Evidence

| property | why it matters | established by |
|---|---|---|
| never raises on an applicable type (Prop. 11.11) | a raise would be a crash kill the suite never earned | `tests/test_mu_minus_families_intent.py::test_never_raises_on_an_edge_value_of_an_applicable_type`, over NaN, ±inf, −0.0, 5e-324, a 5000-digit int, a lone surrogate, the extreme dates (naive and aware) and the empty containers |
| changes some value of every applicable type | an operator that changes nothing is a built-in survivor | `tests/test_mu_minus_families_intent.py::test_changes_some_value_of_every_applicable_type` |
| evaluates the return expression once | evaluating it twice, or not at all, perturbs more than the output | `tests/test_mu_minus_families_intent.py::test_the_return_expression_is_evaluated_once_under_every_template` |
| a shadowed builtin cannot redirect it | a parameter named `type` must not turn the rewrite into another function | `tests/test_mu_minus_families_intent.py::test_a_shadowed_builtin_cannot_change_the_perturbation` |
| deterministic across processes | converge's reproducibility check re-observes in another process | `tests/test_mu_minus_families_intent.py::test_a_set_element_is_chosen_the_same_way_under_every_hash_seed` |
| a heterogeneous return forgoes what a branch cannot take | Fork 2's subset condition | `tests/test_mu_minus_families_intent.py::test_a_heterogeneous_return_forgoes_every_mode_one_branch_cannot_take` |
| count == generation (#9) | the denominator must count what is generated | `tests/test_mu_minus_families_intent.py::test_count_equals_generation_under_every_observed_type`, `tests/test_mu_minus_form_b_intent.py::test_count_equals_generation_with_form_b` |
| killed by assertion on a pinning test | the kill is a value distinction, not a perturbation crash | `tests/test_mu_minus_families_intent.py::test_each_family_dies_by_assertion_on_a_pinning_test`, `tests/test_mu_minus_form_b_intent.py::test_stale_is_killed_by_a_second_call_and_survives_a_single_one`, `tests/test_mu_minus_form_b_intent.py::test_swallow_is_killed_by_a_test_that_pins_the_raise` |
| the isolated worker measures the mutant, not the original | isolated verdicts back the certificate | `tests/test_mu_minus_form_b_intent.py::test_an_isolated_worker_measures_the_yield_mutant_not_the_original`, `tests/test_mu_minus_form_b_intent.py::test_an_isolated_worker_measures_stale_by_two_calls_in_one_test` |
| a Form B mutant is never bytecode-equivalent | TCE would otherwise mark every Form B survivor equivalent unrun | `tests/test_mu_minus_form_b_intent.py::test_a_form_b_mutant_is_never_bytecode_equivalent_to_the_original` |
| the engine's interrupts are never swallowed | Abandoned, MeasurementRefused and StepBudgetExceeded are BaseException | `tests/test_mu_minus_form_b_intent.py::test_swallow_returns_none_for_an_ordinary_raise_only` |
| the call shape survives | sync, async, method, classmethod, staticmethod, property, async generator | `tests/test_mu_minus_form_b_intent.py::test_methods_and_descriptors_bind_once_under_the_innermost_decorator`, `tests/test_mu_minus_form_b_intent.py::test_async_functions_stay_awaitable`, `tests/test_mu_minus_form_b_intent.py::test_an_async_generator_stays_an_async_generator` |
| the nine operators built before #44 keep their ids | `flag` keys on a survivor's id and diff | `tests/test_mu_minus_families_intent.py::test_the_nine_built_operators_keep_their_ids` |

Measured end to end through Detective's CLI (`converge --two-sign` on fresh fixtures, Wesker 204a760,
policy 9 -> #44). Each row is killed/scored, before -> after. "unproven" is "modulo 1
unproven-equivalent"; crash-only gaps are omitted.

| fixture | before | after |
|---|---|---|
| `scale(x, k)` (int) | ✓ COMPLETE 10/10 | ✓ COMPLETE 15/15 |
| `double(x)`, one test | ✓ COMPLETE 9/9 | ✓ COMPLETE 14/14 |
| `mean(xs)` (float) | COMPLETE 9/10, unproven (abs) | COMPLETE 19/20, the same one |
| `is_even(n)` (bool) | ✓ COMPLETE 12/12 | ✓ COMPLETE 17/17 |
| `parse_pos(s)` (raises) | COMPLETE 17/18, unproven (abs) | COMPLETE 23/24, the same one; swallow value-pinned |
| `greet(name)` (str), one test | ✓ COMPLETE 9/9 | COMPLETE 14/15, unproven: stale |
| `evens_list(xs)` (list), one test | ✓ COMPLETE 14/14 | COMPLETE 19/20, unproven: stale |
| `uniq_words(s)` (set), one test | ✓ COMPLETE 6/6 | COMPLETE 9/10, unproven: stale |
| `tally(xs)` (dict), one test | ✓ COMPLETE 14/14 | COMPLETE 16/17, unproven: stale |
| `next_day(d)` (date), one test | ✓ COMPLETE 7/7 | COMPLETE 8/9, unproven: stale |
| `evens(n)` (generator) under `--isolated` | 10/14: four yield mutants falsely survive | 14/14 |

Every new Form A operator was killed in every fixture. The stale rows are the open item described
under "What is not closed".

## Families

### (i) Universal — Def. 11.8 (i)

These apply to every codomain. none, const and id are the independence pair plus existence. Rem.
11.9 (a): no positive operator and no other perturbation catches a function that ignores its
arguments or passes one through.

| operator | form | status | sub-modes | applicability |
|---|---|---|---|---|
| none | A | built | `return_none` | every `return <expr>` (Fork 1) |
| const | A | built | `return_const` | every `return <expr>` (Fork 1) |
| id | A | built | `return_identity` | every `return <expr>` with a non-receiver parameter (Fork 1) |
| zero-of | A | built | `return_zero_of` | Fork 2 |
| type | A | built | `return_type` | Fork 2 |
| stale | B | built | `return_stale` | every non-generator |

- **zero-of.** The canonical inhabitant. For an int that is 0 (return_const) and for a sized type its
  empty (return_empty), so return_zero_of covers what those miss: float, complex, Decimal, Fraction.
- **type.** The same value as another type: str -> bytes (utf-8, surrogatepass), bytes -> str
  (latin-1), bool -> int, int/float -> Decimal. The coercion preserves `==` for the numeric cases,
  so only a type pin kills it. Detective writes one (`assert type(result) is int`,
  NEGATIVE_SPECIFICATION Prop. 11.14).
- **stale.** Each call returns the previous call's return within one test; the first call of a test
  returns its own, and the memory ends at the test boundary (W#49), so the verdict does not depend on
  test order or on which subset ran. It is killed only by a second call with a different result in
  one test, credited to that test. Two tests that call once each do not kill it. Where every call
  returns the same value it is a true no-op, which `check_equivalent`'s sequential boundary calls
  detect. It is not an output operator in the adequacy sense (it reads a prior call). It remains a μ⁻
  fence for leaked state and accidental memoization, an axis the positive policy misses.

### (ii) Boolean — Def. 11.8 (ii)

bool only. bool is not int: no numeric operator applies to a predicate.

| operator | form | status | sub-modes | applicability |
|---|---|---|---|---|
| not | A | built | `return_not` | Fork 2 |
| true | A | built | `return_true` | Fork 2; not a target where the return is already `True` |
| false | A | built | `return_false` | Fork 2; not a target where the return is already `False` |

Rem. 11.9 (b): `not` is near-redundant with a positive LOGICAL mutant on the return. `true` and
`false` are distinct fences: the always-true and the always-false classifier are different failures.

### (iii) Numeric — Def. 11.8 (iii)

int and float, bool excluded.

| operator | form | status | sub-modes | applicability |
|---|---|---|---|---|
| neg | A | built | `return_negate` | Fork 2 |
| abs | A | built | `return_abs` | Fork 2 |
| ±1 | A | built | `return_plus_one`, `return_minus_one` | Fork 2 |
| ±ε | A | built | `return_next_up`, `return_next_down` | Fork 2 |
| scale | A | built | `return_scale` | Fork 2 |
| round | A | built | `return_round` | Fork 2 |
| NaN | A | built | `return_nan` | Fork 2 |
| inf | A | built | `return_inf` | Fork 2 |
| 0 | A | realized_by | `return_const` | — |

- ±ε is `math.nextafter`, the adjacent float.
- scale is k = 2.
- round is to a whole number, `round(v, 0)`, which is total on NaN and ±inf.
- inf is +inf; −inf asks the same finiteness question that NaN asks.
- Rem. 11.9 (b): ±1 is near-redundant with a positive BOUNDARY/ARITHMETIC mutant. It is kept because
  it fences the OUTPUT, whatever computed it.

### (iv) String — Def. 11.8 (iv)

| operator | form | status | sub-modes | applicability |
|---|---|---|---|---|
| empty | A | built | `return_empty` | Fork 2 (also (v)'s container empty) |
| trunc | A | built | `return_truncate` | Fork 2 |
| case | A | built | `return_swapcase` | Fork 2 |
| ws | A | built | `return_pad` | Fork 2 |
| rev | A | built | `return_reverse` | Fork 2 |

- trunc drops the last character, and case is swapcase; both apply to str and bytes.
- ws pads with a trailing space. Strip is the other spelling, but it is a no-op on most values, so pad
  is the member that changes every string.

### (v) Container — Def. 11.8 (v)

| operator | form | status | sub-modes | applicability |
|---|---|---|---|---|
| perm | A | built | `return_reorder` | Fork 2 |
| drop | A | built | `return_drop` | Fork 2 |
| dup | A | built | `return_duplicate` | Fork 2 |
| singleton | A | built | `return_singleton` | Fork 2 |
| ctype | A | built | `return_ctype` | Fork 2 |
| dict | A | partial | `return_drop`, `return_field_default` | Fork 2 |

- **perm.** Reversal of a list or tuple. It is a no-op under set and dict-key semantics, which is why
  it is not emitted there.
- **drop.** Removes the last element of a list or tuple, the last key of a dict (dict's drop-field),
  or the least-repr element of a set or frozenset.
- **singleton.** Keeps the first element, or the least-repr one for a set.
- **ctype.** Maps list <-> tuple and set <-> frozenset. list -> set is not total, because elements
  may be unhashable. set -> list has no deterministic order.
- **dict, partial.** drop-field (`return_drop`) and default-field (`return_field_default`: the last
  key's value becomes None) are built. The key<->value swap is NOT built: it raises on an unhashable
  value, so it is not total on dict, and a perturbation that raises is a crash kill (Prop. 11.11).

### (vi) Structured — Def. 11.8 (vi)

| operator | form | status | sub-modes | applicability |
|---|---|---|---|---|
| field | A | unsupported | — | — |
| class | A/B | unsupported | — | — |

- **field.** Applies the field type's own family to one field. The observed codomain is a bare type
  name, so it cannot show that a return is a record, nor its fields' types. Neither applicability
  nor the family to apply is known before generation. A runtime dispatch would be a no-op wherever
  the premise fails, a built-in survivor. It would also have to rebuild the record, which re-runs
  its validation, and validation can raise.
- **class.** Returns a super- or sub-type instance. That instance has to be constructed, and no
  total construction follows from a type name.

### (vii) Ordered — Def. 11.8 (vii)

| operator | form | status | sub-modes | applicability |
|---|---|---|---|---|
| adj | A | built | `return_adjacent` | Fork 2 |
| clamp | A | unsupported | — | — |
| sortbreak | A | built | `return_sortbreak` | Fork 2 |

- **adj.** The adjacent date (±1 day), datetime or timedelta (±1 µs). It steps the other way at the
  type's maximum. An aware datetime is compared on its naive fields, so the guard never compares
  naive with aware. For an int or float, adjacency is ±1 and ±ε.
- **clamp.** `clip(v, l, u)` needs the range the invariant names. No premise for it is observed, and
  inventing one would make the fence the engine's guess.
- **sortbreak.** Transposes the first two elements of a list or tuple, a local break, distinct from
  perm's reversal on anything longer than two.

### (viii) Effect — Def. 11.8 (viii)

Form B only. Prop. 11.7: Form B is load-bearing for the completeness claim, not optional coverage.

| operator | form | status | sub-modes | applicability |
|---|---|---|---|---|
| yield | B | built | `yield_truncate`, `yield_drop_first`, `yield_duplicate_last`, `yield_empty` | generators and async generators |
| raise | B | partial | `raise_swallow` | non-generators whose own body raises |
| state-noop | B | realized_by | `STATE:remove_assign`, `STMT` | — |
| state-extra | B | unsupported | — | — |

- **yield.** Materializes at most 10 000 items before perturbing, so an infinite generator cannot
  hang the evaluator.
- **raise, partial.** swallow returns None for an `Exception` and lets a `BaseException` through,
  so the engine's interrupts are never eaten. A `pytest.raises` test kills it as `exception`, a value
  kill. Injecting a raise is NOT built: every test that reaches the call fails it, so its kill
  measures reach, not specification.
- **state-noop.** Omitting a side effect is exactly the positive STATE `remove_assign` and STMT
  deletion.
- **state-extra.** An extra mutation must choose which object and attribute to change. The function
  determines no such target, and inventing one would make the fence the engine's guess.

## Fork 2 applicable types

The engine table (`_OUTPUT_TYPE_CONDITIONAL`) as it stands. A mode is emitted only where every
observed return type is in its set.

| sub-mode | applicable types |
|---|---|
| `return_negate` | float, int |
| `return_abs` | float, int |
| `return_nan` | float |
| `return_empty` | bytes, dict, frozenset, list, set, str, tuple |
| `return_reorder` | list, tuple |
| `return_zero_of` | Decimal, Fraction, complex, float |
| `return_type` | bool, bytes, float, int, str |
| `return_not` | bool |
| `return_true` | bool |
| `return_false` | bool |
| `return_plus_one` | float, int |
| `return_minus_one` | float, int |
| `return_next_up` | float |
| `return_next_down` | float |
| `return_scale` | float, int |
| `return_round` | float |
| `return_inf` | float |
| `return_truncate` | bytes, str |
| `return_swapcase` | bytes, str |
| `return_pad` | str |
| `return_reverse` | bytes, str |
| `return_drop` | dict, frozenset, list, set, tuple |
| `return_duplicate` | list, tuple |
| `return_singleton` | frozenset, list, set, tuple |
| `return_ctype` | frozenset, list, set, tuple |
| `return_field_default` | dict |
| `return_sortbreak` | list, tuple |
| `return_adjacent` | date, datetime, timedelta |

## What is not closed

- **stale needs two calls with different results in ONE test.** Detective's witness pool may reach the
  body with only one input. The measured case is an un-annotated str parameter: one harvested value,
  then integer probes that all raise. There stale reads "candidate-equivalent". Before W#49 a second
  `--input` with a different result killed it, written as a second test; the wrapper's memory now ends
  with each test, so that no longer does: the pin must call the function twice in one test. Measured
  on five one-test fixtures (str, list, set, dict, date),
  each reading "COMPLETE modulo 1 unproven-equivalent". `next_day` takes a date, which has no
  literal form, so its remedy is a second test rather than an `--input`. The fix belongs to
  Detective: the witness search needs a second exercising input before it can see a history-
  dependent perturbation.
- **A decorated generator in process.** The in-process yield factory wraps the live, decorated
  original, while the isolated twin wraps the raw function inside its decorators. For an undecorated
  generator the two coincide.
- **Typing is by bare type name.** A user class named `list` would be treated as the builtin.
- **A set element's repr.** The set element choice calls `repr`, which raises for an element whose
  `__repr__` raises or for a >4300-digit int.
- **The modes that predate #44 drop the return expression.** return_none, return_const and
  return_nan replace the expression rather than evaluate it, and they keep that shape for flag
  identity.
- **No generator in the corpus.** The fingerprint corpus holds none, so the yield modes do not move
  the policy id. Their tables are in the manifest.

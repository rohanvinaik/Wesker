"""Sharding the whole-codebase measurement: plan, record, merge (Wesker #29, decision B1).

THE DEFECT. The badge run measured all 24 files of Wesker in ONE job: 146 minutes when it finished
(run 43102482321, `engine.py` alone 88 of them) and past the 180-minute limit when it did not. The law
(Detective ARCHITECTURE section 11) is that the unit of measurement is one function; anything that scales with
the codebase is split. A longer timeout is not the fix.

THE SHAPE. A plan splits the target files into groups (`partition`), each group is measured by the
ordinary Action with `--shard i/N` and writes a *shard record* (`shard_record`), and a merge reads the
records back into the one report the unsharded run would have written (`merge_reports`).

THE RULES, each a pure decision with named codes (never a bool):

* `shard_setting`      what a `--shard` argument asks for.
* `plan_standing`      whether a target list can be planned at all.
* `partition`          the groups: deterministic, every file in exactly one group, none empty.
* `completeness_code`  the refusal a cut or unresolved run earns (the Action's gates, by code).
* `record_standing`    whether a file read back is a shard record.
* `merge_standing`     whether a set of records may be merged; a shard's own refusal is returned AS ITSELF.

A refusal is never averaged away: "measured" and "could not measure" do not share a sign. Totals are
SUMS of the shards' counts and every percentage is computed once, from the sums.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

RECORD_KIND = "wesker-shard-report"
PLAN_KIND = "wesker-shard-plan"
SCHEMA = 1

# The refusal codes a shard record may carry (the Action's own refusals, by name). `merge_standing`
# returns one of these when a shard ended in it; the structural codes below are disjoint from them.
SHARD_REFUSALS = (
    "worker_lost",
    "no_live_session",
    "execution_mode",
    "suite_health",
    "truncated",
    "unresolved",
)


def shard_setting(raw: str) -> tuple[str, int, int]:
    """What a `--shard` argument asks for, as ``(code, index, count)`` (#29, pure — pinned).

    Absent and malformed are different facts, and a typo must not read as "the whole run":

      "whole"    blank: the unsharded run, which is shard 1 of 1 so every run writes the same record
      "shard"    ``i/N`` with 1 <= i <= N, plain decimal digits
      "invalid"  anything else (zero, a sign, a fraction, i > N, text)
    """
    text = raw.strip()
    if not text:
        return "whole", 1, 1
    left, sep, right = text.partition("/")
    if not sep or not left.isdecimal() or not right.isdecimal():
        return "invalid", 0, 0
    index, count = int(left), int(right)
    if index < 1 or count < 1 or index > count:
        return "invalid", 0, 0
    return "shard", index, count


def plan_standing(files: list[str], shards: int) -> str:
    """Whether a target list can be planned into ``shards`` groups (#29, pure — pinned).

    Each code has its own remedy:

      "bad_shard_count"   ``shards`` < 1
      "no_files"          nothing to measure (an empty group would measure the WHOLE codebase when
                          its targets reach the Action empty, so none is ever planned)
      "unsplittable_path" a path that is blank or holds whitespace: the Action's `targets` input is a
                          space-separated list, so it could not be handed to a shard intact
      "duplicate_file"    the same path twice: it would be measured twice and summed twice
      "planned"           plannable
    """
    if shards < 1:
        return "bad_shard_count"
    if not files:
        return "no_files"
    if any((not f) or f != f.strip() or any(c.isspace() for c in f) for f in files):
        return "unsplittable_path"
    if len(set(files)) != len(files):
        return "duplicate_file"
    return "planned"


def partition(weights: dict[str, int], shards: int) -> list[list[str]]:
    """The groups of a plan (#29, pure — pinned).

    Longest-processing-time first: files in descending weight (ties by path), each to the group with the
    least weight so far (ties to the lowest index). Same inputs, same groups, on every machine and
    Python version: the only order consulted is an explicit sort on (weight, str), never a hash or a
    directory listing. Every file lands in exactly one group; a group is its files sorted by path; an
    empty group is never returned, so ``min(shards, len(weights))`` groups come back (none for no files
    or ``shards`` < 1 — `plan_standing` is the decision that names why).
    """
    if shards < 1 or not weights:
        return []
    count = min(shards, len(weights))
    loads = [0] * count
    groups: list[list[str]] = [[] for _ in range(count)]
    for path in sorted(weights, key=lambda p: (-weights[p], p)):
        target = min(range(count), key=lambda i: (loads[i], i))
        loads[target] += weights[path]
        groups[target].append(path)
    return [sorted(g) for g in groups]


def completeness_code(truncated: int, unresolved: int, allow_sample: bool) -> str:
    """The refusal a run's cut/unresolved counts earn (#29, pure — pinned).

    The Action's gates (`gate_truncation`, `gate_unresolved`) by code, in their order, so a shard record
    can say WHICH refusal ended it. ``allow_sample`` turns both into warnings, as ``--allow-truncation``
    does, and the run then has no refusal.

      "truncated"   functions were only partially evaluated (cut)
      "unresolved"  evaluated in full, but some mutants reached no verdict
      "none"        neither, or the sample was allowed
    """
    if allow_sample:
        return "none"
    if truncated:
        return "truncated"
    if unresolved:
        return "unresolved"
    return "none"


def record_standing(record: dict) -> str:
    """Whether a decoded JSON value is a shard record this code can merge (#29, pure — pinned).

    "not_a_shard_report"  it is not an object of this kind
    "wrong_schema"        a record of another schema version
    "malformed"           a field the merge reads is absent or the wrong type
    "ok"
    """
    if record.get("kind") != RECORD_KIND:
        return "not_a_shard_report"
    if record.get("schema") != SCHEMA:
        return "wrong_schema"
    text_fields = ("commit", "policy_id", "wesker_version", "refusal", "refusal_detail")
    if any(not isinstance(record.get(f), str) for f in text_fields):
        return "malformed"
    for field in ("shard_index", "shard_count"):
        value = record.get(field)
        if not isinstance(value, int) or isinstance(value, bool):
            return "malformed"
    targets = record.get("targets")
    if not isinstance(targets, list) or any(not isinstance(t, str) for t in targets):
        return "malformed"
    report = record.get("report")
    refusal = record.get("refusal")
    if report is not None and not isinstance(report, dict):
        return "malformed"
    # A record with no refusal must carry the report it claims; one that refused may or may not.
    if report is None and refusal == "":
        return "malformed"
    return "ok"


def _provenance(field: str, values: list[str]) -> str:
    """``<field>_missing`` / ``<field>_disagree`` / "" for one provenance field across shards."""
    if any(not v for v in values):
        return f"{field}_missing"
    if len(set(values)) > 1:
        return f"{field}_disagree"
    return ""


def merge_standing(planned: list[list[str]], shards: list[dict]) -> str:
    """Whether a set of shard summaries may be merged, and if not, why (#29, pure — pinned).

    ``planned`` is the plan's groups (shard i is ``planned[i-1]``). Each summary is a dict of
    ``index``, ``count``, ``targets``, ``stray`` (measured files outside ``targets``), ``commit``,
    ``policy_id``, ``wesker_version`` and ``refusal``. The first rule that fails names the result, in
    this order:

      "plan_empty"             the plan holds no group (or an empty one)
      "no_reports"             no summary at all
      "shard_count_differs"    a summary was cut for a different shard count than the plan
      "duplicate_shard"        two summaries claim the same shard
      "shard_unplanned"        a summary's index is outside the plan
      "report_missing"         a planned shard has no summary
      "commit_missing" / "commit_disagree", "policy_id_*", "wesker_version_*"
                               shards must name, and agree on, what they measured with
      "file_in_two_shards"     one file measured by two shards (it would be summed twice)
      "shard_files_differ"     a shard's files are not its planned group, or it measured a file it was
                               not asked for
      "measured_set_differs"   the union of the shards' files is not the planned set
      <a SHARD_REFUSALS code>  the first shard (by index) that ended in a refusal, AS ITSELF
      "merged"                 every rule holds: the totals may be summed
    """
    if not planned or any(not group for group in planned):
        return "plan_empty"
    if not shards:
        return "no_reports"
    n = len(planned)
    if any(s["count"] != n for s in shards):
        return "shard_count_differs"
    indices = [s["index"] for s in shards]
    if len(set(indices)) != len(indices):
        return "duplicate_shard"
    if any(i < 1 or i > n for i in indices):
        return "shard_unplanned"
    if set(indices) != set(range(1, n + 1)):
        return "report_missing"
    for field in ("commit", "policy_id", "wesker_version"):
        bad = _provenance(field, [str(s[field]) for s in shards])
        if bad:
            return bad
    measured = [f for s in shards for f in s["targets"]]
    if len(set(measured)) != len(measured):
        return "file_in_two_shards"
    for s in shards:
        if sorted(s["targets"]) != sorted(planned[s["index"] - 1]) or s["stray"]:
            return "shard_files_differ"
    if set(measured) != {f for group in planned for f in group}:
        return "measured_set_differs"
    for s in sorted(shards, key=lambda s: s["index"]):
        if s["refusal"]:
            return str(s["refusal"])
    return "merged"


def describe_merge(code: str, planned: list[list[str]], shards: list[dict]) -> str:
    """The sentence that names a `merge_standing` code's cause and the shards/files it concerns (pure)."""
    n = len(planned)
    have = sorted(s["index"] for s in shards)
    if code == "plan_empty":
        return "the plan holds no shard (or an empty one): there is nothing to merge against."
    if code == "no_reports":
        return f"no shard report was given; the plan has {n} shard(s)."
    if code == "report_missing":
        missing = [i for i in range(1, n + 1) if i not in have]
        named = "; ".join(
            f"shard {i}/{n} ({' '.join(planned[i - 1])})" for i in missing
        )
        return f"no report for planned shard(s): {named}. A shard that did not report was not measured."
    if code == "shard_count_differs":
        return f"a report was cut for another shard count than the plan's {n}."
    if code == "duplicate_shard":
        return f"two reports claim the same shard (indices {have})."
    if code == "shard_unplanned":
        return f"a report's shard index is outside the plan's 1..{n} (indices {have})."
    for field in ("commit", "policy_id", "wesker_version"):
        if code in (f"{field}_missing", f"{field}_disagree"):
            seen = ", ".join(
                f"shard {s['index']}/{n}: {s[field] or '(none)'}"
                for s in sorted(shards, key=lambda s: s["index"])
            )
            verb = "does not name" if code.endswith("_missing") else "disagree on"
            return f"the shards {verb} {field.replace('_', ' ')} ({seen}); they cannot be shown to have measured the same thing."
    if code == "file_in_two_shards":
        seen: dict[str, list[int]] = {}
        for s in shards:
            for f in s["targets"]:
                seen.setdefault(f, []).append(s["index"])
        dup = "; ".join(
            f"{f} in shards {sorted(ix)}"
            for f, ix in sorted(seen.items())
            if len(ix) > 1
        )
        return f"a file was measured by more than one shard ({dup}); its counts would be summed twice."
    if code == "shard_files_differ":
        bad = [
            f"shard {s['index']}/{n} measured {' '.join(sorted(s['targets'])) or '(nothing)'}"
            + (f" and stray {' '.join(sorted(s['stray']))}" if s["stray"] else "")
            + f", planned {' '.join(planned[s['index'] - 1])}"
            for s in sorted(shards, key=lambda s: s["index"])
            if sorted(s["targets"]) != sorted(planned[s["index"] - 1]) or s["stray"]
        ]
        return "a shard's files are not its planned group: " + "; ".join(bad) + "."
    if code == "measured_set_differs":
        return "the files the shards measured are not the planned set."
    if code == "merged":
        return "every shard reported the planned files."
    refused = [s for s in sorted(shards, key=lambda s: s["index"]) if s["refusal"]]
    return "; ".join(
        f"shard {s['index']}/{n} ({' '.join(sorted(s['targets']))}) ended in {s['refusal']}: "
        f"{s.get('detail', '')}".rstrip(": ")
        for s in refused
    )


# ── The impure boundary: files and reports ───────────────────────


def build_plan(targets: list[str], shards: int) -> dict:
    """The plan document for ``targets``: weight = source bytes (stable, free, and a fair proxy).

    Raises ``ValueError`` naming the `plan_standing` code, or the target that is not a readable file.
    """
    standing = plan_standing(targets, shards)
    if standing != "planned":
        raise ValueError(standing)
    weights: dict[str, int] = {}
    for path in targets:
        if not Path(path).is_file():
            raise ValueError(f"not_a_file: {path}")
        weights[path] = Path(path).stat().st_size
    groups = partition(weights, shards)
    return {
        "kind": PLAN_KIND,
        "schema": SCHEMA,
        "shard_count": len(groups),
        "weight": "source bytes",
        "groups": groups,
        "include": [
            {
                "shard": f"{i}/{len(groups)}",
                "artifact": f"shard-{i}-of-{len(groups)}",
                "targets": " ".join(g),
            }
            for i, g in enumerate(groups, 1)
        ],
    }


def matrix_json(plan: dict) -> str:
    """The GitHub matrix for a plan: ``{"include": [...]}`` and nothing else, as one line."""
    return json.dumps({"include": plan["include"]}, separators=(",", ":"))


def shard_record(
    *,
    index: int,
    count: int,
    targets: list[str],
    commit: str,
    policy_id: str,
    version: str,
    refusal: str,
    detail: str,
    report: dict | None,
) -> dict:
    """What one Action run writes so a merge can verify and sum it."""
    return {
        "kind": RECORD_KIND,
        "schema": SCHEMA,
        "shard_index": index,
        "shard_count": count,
        "targets": list(targets),
        "commit": commit,
        "policy_id": policy_id,
        "wesker_version": version,
        "refusal": refusal,
        "refusal_detail": detail,
        "report": report,
    }


def summarize(record: dict) -> dict:
    """The primitive summary `merge_standing` decides on, read from a validated record."""
    report = record["report"] or {}
    asked = set(record["targets"])
    return {
        "index": record["shard_index"],
        "count": record["shard_count"],
        "targets": list(record["targets"]),
        "stray": sorted(set(report.get("per_file") or {}) - asked),
        "commit": record["commit"],
        "policy_id": record["policy_id"],
        "wesker_version": record["wesker_version"],
        "refusal": record["refusal"],
        "detail": record["refusal_detail"],
    }


def _dict_sum(reports: list[dict], key: str) -> int:
    return sum(int(r.get(key, 0) or 0) for r in reports)


def merge_reports(reports: list[dict]) -> dict:
    """The one report the unsharded run would have written, from the shards' reports (in shard order).

    Counts are summed; every percentage is recomputed from the sums with the expressions
    `ci.profile_codebase` uses; the per-function lists are concatenated, so the Action's gates and
    renderers consume the same per-function facts they consume on a single run.
    """
    from Wesker.ci import _harness_totals, memory_rollup_standing

    def cat(key: str) -> list:
        return [x for r in reports for x in (r.get(key) or [])]

    truncated = cat("truncated_functions")
    unresolved = cat("unresolved_functions")
    harness = cat("harness_functions")
    memory_counts: dict[str, int] = {}
    for r in reports:
        for name, n in (r.get("memory_standings") or {}).items():
            memory_counts[name] = memory_counts.get(name, 0) + int(n)
    standings = [s for s in sorted(memory_counts) for _ in range(memory_counts[s])]
    categories: dict[str, dict] = {}
    for r in reports:
        for c in r.get("per_category") or []:
            agg = categories.setdefault(
                c["category"],
                {
                    "category": c["category"],
                    "total": 0,
                    "killed": 0,
                    "survived": 0,
                    "equivalent": 0,
                },
            )
            for field in ("total", "killed", "survived", "equivalent"):
                agg[field] += c.get(field, 0)
    per_file: dict[str, dict] = {}
    for r in reports:
        per_file.update(r.get("per_file") or {})
    killed = _dict_sum(reports, "total_killed")
    mutants = _dict_sum(reports, "total_mutants")
    equivalent = _dict_sum(reports, "total_equivalent")
    dof = _dict_sum(reports, "total_dof")
    modes = {str(r.get("execution_mode", "unknown")) for r in reports}
    isolated = all(bool(r.get("isolated")) for r in reports)
    limit = max((int(r.get("memory_limit_mb") or 0) for r in reports), default=0)
    return {
        "total_killed": killed,
        "total_mutants": mutants,
        "total_equivalent": equivalent,
        "total_universe": _dict_sum(reports, "total_universe"),
        "total_dof": dof,
        "total_dof_covered": _dict_sum(reports, "total_dof_covered"),
        "dof_pct": round(100 * _dict_sum(reports, "total_dof_covered") / max(dof, 1)),
        "total_dof_pinned": _dict_sum(reports, "total_dof_pinned"),
        "spec_pct": round(100 * _dict_sum(reports, "total_dof_pinned") / max(dof, 1)),
        "kill_pct": round(100 * killed / max(mutants - equivalent, 1)),
        "total_functions": _dict_sum(reports, "total_functions"),
        "total_truncated": len(truncated),
        "truncated_functions": truncated,
        "total_unresolved": len({u["function_key"] for u in unresolved}),
        "unresolved_functions": unresolved,
        "total_harness_mutants": sum(e["count"] for e in harness),
        "harness_mutants": _harness_totals(harness),
        "total_harness": len({e["function_key"] for e in harness}),
        "harness_functions": harness,
        "unreached_functions": cat("unreached_functions"),
        "isolated": isolated,
        "memory_limit_mb": limit,
        "memory_standing": memory_rollup_standing(isolated, limit, standings),
        "memory_standings": {s: memory_counts[s] for s in sorted(memory_counts)},
        "passes": max((int(r.get("passes") or 1) for r in reports), default=1),
        "elapsed_ms": _dict_sum(reports, "elapsed_ms"),
        "progress_lines_dropped": _dict_sum(reports, "progress_lines_dropped"),
        "per_file": per_file,
        "per_category": list(categories.values()),
        "survivors": cat("survivors"),
        "execution_mode": modes.pop() if len(modes) == 1 else "mixed",
    }


def read_json(path: str) -> object:
    """Decode a JSON file; raises ``ValueError`` (named by the caller) on anything unreadable."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: {exc}") from exc


def environment_commit() -> str:
    """The commit a run measures: ``GITHUB_SHA`` when CI names it, else git's HEAD, else ""."""
    named = os.environ.get("GITHUB_SHA", "").strip()
    if named:
        return named
    from Wesker.action import _git

    head = _git(["rev-parse", "HEAD"], ".")
    return head.strip() if head else ""

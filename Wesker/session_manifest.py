"""What pytest ACTUALLY did, recorded by the session that did it (Detective #58).

The regime a proof was measured under is reconstructed in at least five places — config
precedence mirrored from files on disk, the selected config re-parsed, the suite's import path
rebuilt from one pyproject dialect, `norecursedirs` read from a single table, a dotted module
name derived from a path. Each is a PREDICTION of what pytest would do, made from the same
inputs but not by the same code, and a certificate cannot rest on a prediction: the failure
mode is exactly the case where prediction and runner disagree.

The one existing live check reads pytest's own answer by REGEX-SCRAPING `--collect-only`
stdout for a single field, at the cost of a subprocess — which is why it runs only at
`--migrate` and never on the measurement path.

This is captured from the `pytest_collection_modifyitems` hook, which already receives
`session`, `config` and `items` from the live collection Wesker performs anyway. Nothing extra
runs; the hook was discarding two of its three arguments.

FILE ORIGIN IS CANONICAL; A DOTTED MODULE NAME IS AN ALIAS. Two files under one name, or one
file under two names, is the shadowing condition a verdict must refuse over — a measurement
attributed to the wrong copy of a function is a measurement of something else. `module_origins`
records the mapping so that conflict is detectable rather than assumed absent.

NOT A PROOF OF EXECUTION. This describes COLLECTION: what pytest selected, under what regime,
from which files. Per-item outcomes belong to the run and are not here.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
import contextlib
from dataclasses import dataclass, field
from typing import Any


def _digest(path: str) -> str:
    """A file's content digest, or "" when it cannot be read.

    Empty rather than a sentinel hash: "we did not read it" and "it hashed to X" are different
    facts, and a placeholder that compares equal across two unreadable files would report them
    as identical content.
    """
    try:
        with open(path, "rb") as fh:
            return hashlib.sha256(fh.read(), usedforsecurity=False).hexdigest()[:16]
    except OSError:
        return ""


@dataclass(frozen=True)
class CollectedItem:
    """One test pytest selected, addressed the way a proof basis must address it."""

    node_id: str
    origin: str
    origin_digest: str
    module: str


def conflicting_module_names(
    module_origins: dict[str, tuple[str, ...]],
) -> tuple[str, ...]:
    """Module names that do not identify exactly ONE file, sorted (#58, pure — pinned).

    The shadowing condition, stated in identities rather than paths: if one dotted name maps to
    two files, a measurement attributed to that name could have come from either, and a
    certificate over it is a certificate about something unspecified.

    Split out of the property it serves so it can be pinned at all. A property has an implicit
    receiver, so input synthesis cannot construct a call — measured, the property alone pinned
    0 of 12 mutants. Taking the mapping directly puts the whole decision inside a literal
    grammar, and the property below keeps the accessor and holds no decision of its own.

    Three cases that are NOT conflicts, each a way a naive check goes wrong: the same file
    listed twice is one file observed twice, not two (origins accumulate per sighting); a name
    with no origins never participated, and flagging it would refuse verdicts over nothing; and
    a clean name alongside a dirty one must not drag the clean one into the refusal.
    """
    return tuple(
        sorted(name for name, files in module_origins.items() if len(set(files)) > 1)
    )


def collection_identity_standing(
    observed: bool, conflicting_modules: tuple[str, ...]
) -> str:
    """What the LIVE collection established about module identity (#58, pure — pinned).

    The manifest exists so a certificate can name the regime it was measured under instead of
    PREDICTING it. Until now nothing read it: `last_session_manifest()` had zero consumers in
    either repo — not even a test — so the capture ran every session and informed no decision.
    This is the seam that makes it load-bearing.

    Three states, because "we did not look" and "we looked and it was clean" are different facts,
    and collapsing them is how an absent check comes to read as a passed one:

    ``unobserved`` — no manifest for this run. The pre-flight prediction (``shadowed_target``)
    stands alone, exactly as before; a missing observation must never manufacture a refusal.
    ``ambiguous`` — a dotted name resolved to MORE THAN ONE FILE in the very run that measured.
    A measurement attributed to the wrong copy of a function is a measurement of something else,
    so nothing computed over this collection can be gated on.
    ``confirmed`` — the runner itself reports every name identifying exactly one file. Positive
    evidence, and strictly stronger than the prediction, because it is what pytest DID rather
    than what we reconstructed it would do.

    Broader than ``shadowed_target``, which resolves ONE target's own module name in a subprocess.
    This sees every module the session imported, so it also catches a shadowed DEPENDENCY — the
    case where the target itself resolves fine and the code beneath it does not.
    """
    if not observed:
        return "unobserved"
    if conflicting_modules:
        return "ambiguous"
    return "confirmed"


def manifest_admissibility(manifest_scope: int, current_scope: int) -> str:
    """Whether a captured manifest may serve as THIS run's proof basis (#26, pure — pinned).

    Split from the consumer so the whole decision sits inside a literal grammar and can be
    pinned: the consumer holds live ContextVar objects and a frozen dataclass, neither of which
    input synthesis can construct, while this takes two ints and returns a named code.

    Scopes are positive session ids; ``0`` means absent — no live session in scope, or a
    manifest that predates #26 / came from a collect-only discovery. Admissible only when both
    are present and identical, i.e. the manifest was minted by the exact session now consuming
    it. Every other case is ``refuse``, and the three ways it can arise are kept distinct in the
    reasoning even though they share a verdict:

    * ``current_scope <= 0`` — nothing is measuring; a manifest lingering in the ContextVar is
      not evidence for a run that is not happening.
    * ``manifest_scope <= 0`` — the manifest was never stamped by a live session (collect-only,
      or pre-#26); it describes a collection, not this measurement.
    * ``manifest_scope != current_scope`` — a DIFFERENT session captured it. This is the leak:
      two projects can share rootpath, module names, and config shape, so only the id separates
      "our collection" from "the last collection", and matching fields after the fact is exactly
      the substitution the manifest exists to end.
    """
    if current_scope <= 0:
        return "refuse"
    if manifest_scope <= 0:
        return "refuse"
    if manifest_scope != current_scope:
        return "refuse"
    return "admit"


@dataclass(frozen=True)
class PytestSessionManifest:
    """The collection regime and its selected items, as the runner reports them."""

    pytest_version: str = ""
    python_version: str = ""
    rootpath: str = ""
    inipath: str = ""
    # The config file's CONTENT digest (§2.2 a-1), captured at build. `inipath` alone keys the
    # PATH; an in-place edit to a config at the same path is a different regime this binds.
    inicontent_digest: str = ""
    import_mode: str = ""
    invocation_args: tuple[str, ...] = ()
    plugins: tuple[str, ...] = ()
    collection_errors: tuple[str, ...] = ()
    items: tuple[CollectedItem, ...] = ()
    # Dotted name -> every distinct file seen under it. A well-formed session yields exactly
    # one file per name; more is the shadow/alias conflict a certificate must refuse over.
    module_origins: dict[str, tuple[str, ...]] = field(default_factory=dict)
    # The live measurement session that captured this manifest (#26). 0 means "no session" — a
    # collect-only discovery, or a pre-#26 manifest. A proof-facing consumer admits this manifest
    # only when its scope matches the session now consuming it, so a prior project's collection
    # left in the ContextVar cannot be read as this run's proof basis.
    scope: int = 0

    @property
    def conflicting_modules(self) -> tuple[str, ...]:
        """Module names that do not identify exactly one file, sorted.

        Derived rather than stored: a recorded conflict list could disagree with the mapping it
        summarises, and the two drifting apart is how a session reports clean while its own
        origins say otherwise.
        """
        return conflicting_module_names(self.module_origins)

    @property
    def regime_digest(self) -> str:
        """A digest of the pytest EXECUTION REGIME (#63) — the config that determines HOW collected
        tests run, so a warm verdict measured under one regime is not served under another.

        Covers the runner identity a manifest can see: pytest and python version, rootdir, ini file
        (path AND content, §2.2 a-1), import mode, and the loaded plugin set (sorted — load order is
        not a regime change).
        Distribution plugins bind their version; unversioned/local plugins (including conftests)
        bind canonical source path + content digest. Per-item fixture definitions are additionally
        bound by ``trace_cache.test_fingerprint``. Excludes ``items`` / ``scope`` /
        ``collection_errors`` / ``invocation_args`` — those vary per collection, per session, and
        per target-path selection, so folding them in would churn the key with no regime change.
        """
        import hashlib

        # An unobserved/unreadable plugin has no cross-session identity. Keep the MANIFEST itself
        # reproducible, but refuse to mint a cache key from incomplete regime evidence; consumers
        # interpret "" as uncacheable/unknown rather than equating two unknown plugin contexts.
        if any(
            "@<unobserved>" in p or ":<unreadable>" in p or p.startswith("<unnamed>:")
            for p in self.plugins
        ):
            return ""
        payload = "\x00".join(
            (
                self.pytest_version,
                self.python_version,
                self.rootpath,
                self.inipath,
                # §2.2 a-1: bind the config CONTENT, not just its path — an in-place edit to
                # addopts / markers / options at the same path is a different regime. Captured at
                # build by `capture_manifest` (via `_digest`), so this stays a PURE hash of the
                # frozen snapshot rather than reading the filesystem from a property.
                self.inicontent_digest,
                self.import_mode,
                *sorted(self.plugins),
            )
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def capture_manifest(
    session: Any, config: Any, items: list[Any]
) -> PytestSessionManifest:
    """Build the manifest from a LIVE `Config`/`Session` and the items it selected.

    Every read is defensive. This runs inside a collection hook, and a manifest that raises
    turns a working collection into a failed one — the measurement is the product, the
    description of it is not. Anything unavailable is recorded as empty, which reads as "not
    observed" rather than as a value.

    Origins are `realpath`'d because pytest canonicalises and callers routinely do not; comparing
    the two spellings of one file as different files is a defect this repo has already paid for
    twice (Wesker #15, #20).
    """
    items = items or []
    origins: dict[str, list[str]] = {}
    collected: list[CollectedItem] = []
    for item in items:
        raw = getattr(item, "path", None) or getattr(item, "fspath", None)
        origin = os.path.realpath(str(raw)) if raw else ""
        module = getattr(getattr(item, "module", None), "__name__", "") or ""
        if module and origin:
            origins.setdefault(module, [])
            if origin not in origins[module]:
                origins[module].append(origin)
        collected.append(
            CollectedItem(
                node_id=str(getattr(item, "nodeid", "")),
                origin=origin,
                origin_digest=_digest(origin) if origin else "",
                module=module,
            )
        )

    def _opt(name: str, default: str = "") -> str:
        try:
            return str(config.getoption(name))
        except Exception:  # noqa: BLE001 — an absent option is not an error here
            return default

    try:
        import pytest as _pytest

        pytest_version = getattr(_pytest, "__version__", "")
    except Exception:  # noqa: BLE001
        pytest_version = ""

    # PLUGIN IDENTITY MUST BE REPRODUCIBLE, or the field defeats its own purpose. #58 wants the
    # regime comparable across runs, and `list_name_plugin` names an ANONYMOUS plugin — one
    # registered as an instance, which includes Wesker's own collector — by stringifying its
    # `id()`. Recorded raw, two identical sessions produced different manifests (measured:
    # '4372631232' then '4407692992'), so any digest over this field was unstable and any replay
    # check on it could only ever fail or be ignored. Neither is a regime check.
    plugins: tuple[str, ...] = ()
    try:
        # Distribution-backed plugins carry a real version; that is the identity worth having,
        # and `list_plugin_distinfo` is the authoritative source for it.
        versioned: dict[int, str] = {}
        for _plugin, _dist in config.pluginmanager.list_plugin_distinfo():
            _pname = str(getattr(_dist, "project_name", "") or "")
            _pver = str(getattr(_dist, "version", "") or "")
            if _pname:
                versioned[id(_plugin)] = f"{_pname}=={_pver}" if _pver else _pname
        named: list[str] = []
        anonymous = 0
        for name, plugin in config.pluginmanager.list_name_plugin():
            entry = versioned.get(id(plugin))
            if entry:
                named.append(entry)
                continue

            stable_name = bool(name and not name.startswith("_") and not name.isdigit())
            owner_name = getattr(plugin, "__module__", "") or getattr(
                type(plugin), "__module__", ""
            )
            owner_qualname = getattr(type(plugin), "__qualname__", "")
            label = (
                str(name)
                if stable_name
                else ".".join(p for p in (owner_name, owner_qualname) if p)
            )

            origin = getattr(plugin, "__file__", None)
            if not origin and owner_name:
                owner = sys.modules.get(owner_name)
                origin = getattr(owner, "__file__", None)
            if not origin and stable_name:
                # Pytest registers several builtin plugins twice (``cacheprovider`` and
                # ``pytest_cacheprovider``) with a sentinel rather than the module object.
                # Resolve the loaded builtin module by its canonical namespace; this is an
                # observation of sys.modules, not a guessed file path.
                base = str(name).removeprefix("pytest_")
                owner = sys.modules.get(f"_pytest.{base}")
                origin = getattr(owner, "__file__", None)
                if not origin:
                    with contextlib.suppress(ImportError, AttributeError, ValueError):
                        plugin_spec = importlib.util.find_spec(f"_pytest.{base}")
                        origin = getattr(plugin_spec, "origin", None)

            if origin and label:
                # An unversioned plugin is commonly a local conftest/module. Its NAME stays stable
                # across an edit while its hooks can begin reaching a target, so name-only regime
                # identity is not enough for proof-grade cached routing (#15). Bind the canonical
                # source path and digest when observable; unreadable still widens by changing the
                # entry from a falsely authoritative bare name to an explicit unknown-content one.
                canonical = os.path.realpath(str(origin))
                digest = _digest(canonical) or "<unreadable>"
                named.append(f"{label}@{canonical}:{digest}")
            elif stable_name:
                # Without an observable plugin identity there is nothing safe to compare across
                # sessions. Make the regime deliberately uncacheable rather than treating two
                # unknown plugins as the same execution context.
                named.append(f"{name}@<unobserved>")
            else:
                anonymous += 1
        # The COUNT is kept even though the identities cannot be: it is stable across runs and
        # still detects "a plugin appeared that was not here before", which dropping them
        # silently would not. An identity we cannot state honestly is not invented.
        plugins = tuple(sorted(set(named))) + (
            (f"<unnamed>:{anonymous}",) if anonymous else ()
        )
    except Exception:  # noqa: BLE001
        plugins = ()

    # `collection_errors` is DELIBERATELY LEFT EMPTY here, not fabricated. A file that failed
    # to import never reaches `pytest_collection_modifyitems` — it is reported through
    # `pytest_collectreport`, a different hook — so this capture point genuinely cannot see
    # them. The field exists because the distinction matters enormously ("no test reaches this
    # target" versus "we could not look"), and filling it from the wrong source would make an
    # always-empty list read as a clean session. Wiring the second hook is the next step.
    errors: tuple[str, ...] = ()
    _ = session  # accepted for the hook's signature; nothing here is derivable from it yet

    # Bind ONCE, then read. This used to dereference one `getattr(config, "invocation_params")`
    # call while guarding a SECOND, separate call — so the value checked was never the value
    # used. On a plain attribute the two agree; on a property, a mock, or a plugin-wrapped
    # config they need not, and this module's whole contract is that describing a collection
    # must never break one. `.args` is read defensively for the same reason: an object that
    # exists WITHOUT that attribute would raise from inside a hook that promises not to.
    _invocation_args = tuple(
        str(a)
        for a in getattr(getattr(config, "invocation_params", None), "args", ()) or ()
    )

    # Stamp the live session that captured this (#26). Read here, at capture, so the id belongs
    # to whatever session is actually collecting — a live measurement stamps its own scope, a
    # collect-only discovery stamps 0 (no scope), and neither can later be mistaken for the
    # other. Local import: `pytest_discovery` runtime-imports this module, so a module-level
    # back-edge would be a cycle.
    from .pytest_discovery import current_measurement_scope

    _inipath = str(getattr(config, "inipath", "") or "")
    return PytestSessionManifest(
        pytest_version=pytest_version,
        python_version=sys.version.split()[0],
        rootpath=str(getattr(config, "rootpath", "") or ""),
        inipath=_inipath,
        inicontent_digest=_digest(_inipath) if _inipath else "",
        import_mode=_opt("importmode"),
        invocation_args=_invocation_args,
        plugins=plugins,
        collection_errors=errors,
        items=tuple(collected),
        module_origins={k: tuple(v) for k, v in origins.items()},
        scope=current_measurement_scope() or 0,
    )

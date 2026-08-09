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
import os
import sys
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


@dataclass(frozen=True)
class PytestSessionManifest:
    """The collection regime and its selected items, as the runner reports them."""

    pytest_version: str = ""
    python_version: str = ""
    rootpath: str = ""
    inipath: str = ""
    import_mode: str = ""
    invocation_args: tuple[str, ...] = ()
    plugins: tuple[str, ...] = ()
    collection_errors: tuple[str, ...] = ()
    items: tuple[CollectedItem, ...] = ()
    # Dotted name -> every distinct file seen under it. A well-formed session yields exactly
    # one file per name; more is the shadow/alias conflict a certificate must refuse over.
    module_origins: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def conflicting_modules(self) -> tuple[str, ...]:
        """Module names that do not identify exactly one file, sorted.

        Derived rather than stored: a recorded conflict list could disagree with the mapping it
        summarises, and the two drifting apart is how a session reports clean while its own
        origins say otherwise.
        """
        return conflicting_module_names(self.module_origins)


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
            elif name and not name.startswith("_") and not name.isdigit():
                named.append(name)
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

    return PytestSessionManifest(
        pytest_version=pytest_version,
        python_version=sys.version.split()[0],
        rootpath=str(getattr(config, "rootpath", "") or ""),
        inipath=str(getattr(config, "inipath", "") or ""),
        import_mode=_opt("importmode"),
        invocation_args=_invocation_args,
        plugins=plugins,
        collection_errors=errors,
        items=tuple(collected),
        module_origins={k: tuple(v) for k, v in origins.items()},
    )

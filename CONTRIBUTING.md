# Contributing

Issues and pull requests are welcome. For anything bigger than a small fix, open an issue first, so the
approach is agreed before the work is done.

Security problems go through the private channel in [SECURITY.md](./SECURITY.md), not an issue.

## Setup

```bash
uv sync
```

The engine has no runtime dependencies; pytest is an optional extra. Keep it that way: a dependency in the
engine needs a strong reason in the pull request.

## Checks

CI runs these on every push; running them first saves a round trip.

```bash
uv run ruff check Wesker tests
uv run ruff format --check Wesker tests
uv run pytest
uvx zizmor@1.30.1 .     # when you touch .github/ or action.yml
```

The suite treats warnings as errors. A new warning is something to fix, not to silence.

`uvx pre-commit install` runs ruff and zizmor on every commit (see `.pre-commit-config.yaml`).

## Wesker and Detective

[Detective](https://github.com/rohanvinaik/Detective) is built on Wesker and exercises it far more heavily
than Wesker's own suite does. A change to what `Wesker.engine`, `Wesker.ci` or `Wesker.filter` expose can
pass here and fail there. If you change that surface, also run Detective's suite against your checkout,
from a Detective checkout:

```bash
PYTHONPATH=/path/to/Detective:/path/to/Wesker uv run pytest
```

## Tests

Files named `tests/*_synth.py` are written by Detective. They are characterizations: they pin what a
function currently does, bugs included. Every behaviour change also gets a hand-written test that states
the intended behaviour.

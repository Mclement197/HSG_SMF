# HSG_SMF

Python project managed with [`uv`](https://docs.astral.sh/uv/) and formatted/linted with [`ruff`](https://docs.astral.sh/ruff/).

## Setup

Install `uv` if it is not already installed:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Install and pin Python 3.13:

```bash
uv python install 3.13
uv python pin 3.13
```

Create or update the virtual environment and install all dependencies:

```bash
uv sync
```

Run Python inside the managed environment:

```bash
uv run python --version
uv run python your_script.py
```

## Dependencies

Add runtime dependencies:

```bash
uv add yfinance
uv add package-name
```

Add development-only dependencies:

```bash
uv add --dev ruff
uv add --dev package-name
```

Remove dependencies:

```bash
uv remove package-name
uv remove --dev package-name
```

Update the lockfile and environment:

```bash
uv lock --upgrade
uv sync
```

Show installed packages:

```bash
uv pip list
uv tree
```

## Ruff

Check for lint issues:

```bash
uv run ruff check .
```

Automatically fix lint issues, including unused imports and sorted imports:

```bash
uv run ruff check . --fix
```

Apply fixes that may change behavior more aggressively, such as removing unused variables:

```bash
uv run ruff check . --fix --unsafe-fixes
```

Format all Python files:

```bash
uv run ruff format .
```

This is the Ruff command that automatically splits long Python expressions,
function calls, and function definitions across multiple lines where possible.
Ruff does not automatically rewrite every long string or prose docstring, so
those may still need manual shortening.

Run the usual cleanup command:

```bash
uv run ruff check . --fix --unsafe-fixes
uv run ruff format .
```

Check formatting without changing files:

```bash
uv run ruff format . --check
```

Show what Ruff would change:

```bash
uv run ruff check . --diff
uv run ruff format . --diff
```

Explain a Ruff rule:

```bash
uv run ruff rule F401
uv run ruff rule I001
```

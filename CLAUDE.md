# Claude Code rules for DCBA-data-set

## Project overview

Generates synthetic multilayer graphs via ABCD/mABCD (Julia-backed) and provides primitive data
loading primitives for further training tasks (see the `../DCBA/` repository).

## Environment

- Package manager: **uv**. Use `uv add <pkg>` to install packages and `uv run <cmd>` to execute
  project scripts. Never use pip directly.
- Python entry point: `uv run dcba-data-set <config.yaml>`

## Language

- Use **British English** in all text: comments, docstrings, commit messages, and documentation.

## Code style

- Line length: **100 characters**.
- Type hints are required on every function signature (arguments and return type).
- Docstring style: **reStructuredText** using standard Sphinx field directives only: `:param x:`,
  `:type x:`, `:returns:`, `:rtype:`, `:raises ExcType:`. For class attribute documentation use
  plain prose — do **not** use `:ivar:`.
- Prefer `pathlib.Path` over bare strings for all file-system paths.
- Prefer placing text in new line (i.e., \n after """) if docstring cannot fit in one line.

## Version management

- When starting work on a new branch, bump `version` in `pyproject.toml` before making other
  changes.

## Git workflow

- When moving files use `git mv`, not bare `mv`.
- Test code before every commit — do not commit changes that are known to be broken.
- If pre-commit is installed (`pre-commit` in PATH or `.pre-commit-config.yaml` exists), run
  `pre-commit run --files <changed files>` before committing.
- Commit messages: short imperative subject line, no co-authorship trailers.

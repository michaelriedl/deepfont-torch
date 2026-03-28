# Development Environment

The project uses a `.venv` virtual environment managed by `uv`. Run all tools through the venv:

```bash
.venv/bin/pytest tests/ -v
.venv/bin/ruff format --check deepfont
.venv/bin/ruff check deepfont
.venv/bin/ty check deepfont
```

## CI/CD Checks

All three checks run on every pull request and must pass before merging:

1. **Tests** (`pytest`): `pytest tests/ -v --tb=short --cov=deepfont` (Python 3.12 and 3.13)
2. **Lint and format** (`ruff`): `ruff format --check deepfont` and `ruff check deepfont`
3. **Type checking** (`ty`): `ty check deepfont`

Run all three locally before pushing.

# Coding Standards

## Language

Use American English in all code, comments, docstrings, and documentation.

Common corrections:
- "normalisation" -> "normalization"
- "normalised" -> "normalized"
- "parameterises" -> "parameterizes"
- "optimiser" -> "optimizer"
- "initialise" -> "initialize"
- "serialise" -> "serialize"
- "colour" -> "color"
- "behaviour" -> "behavior"
- "favour" -> "favor"
- "modelling" -> "modeling"
- "labelled" -> "labeled"
- "catalogue" -> "catalog"
- "centre" -> "center"
- "analysing" -> "analyzing"

## Comments

- Keep comments plain text without special formatting characters.
- Do not use Sphinx/RST markup such as `:class:`, `:func:`, `:param:`, or `:type:` in comments or docstrings.
- Do not use bold (`**text**`), italic (`*text*`), or double-backtick inline code (` ``code`` `) formatting inside docstrings or comments.
- Do not use extended characters to create visual separators (e.g., `# -- Section -----` or `# Section =====`). Use a plain comment instead (e.g., `# Section`).

## Docstrings

Follow the Google Python docstring style:

- Use a one-line summary on the first line of the docstring.
- Separate the summary from the body with a blank line.
- Use Google-style section headers: `Args:`, `Returns:`, `Raises:`, `Example:`, `Note:`, `Attributes:`, etc.
- Format examples with `>>>` prompts so they are compatible with doctest:

```python
def add(a: int, b: int) -> int:
    """Return the sum of two integers.

    Args:
        a: First operand.
        b: Second operand.

    Returns:
        The sum of a and b.

    Example:
        >>> add(2, 3)
        5
    """
    return a + b
```

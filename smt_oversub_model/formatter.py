"""
Composable plain-text formatting primitives for terminal output.

All formatting functions return plain text using box-drawing characters (no ANSI).
Color is applied separately via colorize() as a post-processing step for terminal display.

No external dependencies — pure Python.
"""

import os
import re
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


def supports_color() -> bool:
    """Detect whether the terminal supports ANSI color output.

    Respects NO_COLOR (https://no-color.org/) and FORCE_COLOR env vars.
    """
    if os.environ.get('NO_COLOR') is not None:
        return False
    if os.environ.get('FORCE_COLOR') is not None:
        return True
    if not hasattr(sys.stdout, 'isatty'):
        return False
    return sys.stdout.isatty()


# ── Box-drawing characters ──────────────────────────────────────────

_HEAVY_H = '═'
_LIGHT_H = '─'

_TL = '┌'
_TR = '┐'
_BL = '└'
_BR = '┘'
_VL = '│'
_TJ = '┬'
_BJ = '┴'
_CJ = '┼'
_LJ = '├'
_RJ = '┤'


# ── Formatting primitives ──────────────────────────────────────────

def title(text: str, width: int = 60) -> str:
    """Prominent title with heavy-line borders.

    Example::

        ═══════════════ My Title ═══════════════
    """
    padding = width - len(text) - 2  # 2 for spaces around text
    if padding < 4:
        padding = 4
    left = padding // 2
    right = padding - left
    return f"{_HEAVY_H * left} {text} {_HEAVY_H * right}"


def heading(text: str) -> str:
    """Section heading with light-line underline.

    Example::

        Section Name
        ────────────
    """
    line = _LIGHT_H * len(text)
    return f"  {text}\n  {line}"


def kv_block(items: Sequence[Tuple[str, str]], indent: int = 2) -> str:
    """Aligned key-value pairs with dot leaders.

    Example::

        Baseline ········· SMT @ R=1.0
        Reference ········ SMT @ R=1.3
    """
    if not items:
        return ""
    max_key = max(len(k) for k, _ in items)
    leader_char = '·'
    lines = []
    prefix = ' ' * indent
    for key, value in items:
        dots = leader_char * (max_key - len(key) + 2)
        lines.append(f"{prefix}{key} {dots} {value}")
    return "\n".join(lines)


def table(
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
    aligns: Optional[Sequence[str]] = None,
) -> str:
    """Box-drawing bordered table.

    Args:
        headers: Column header strings.
        rows: List of row data (each row is a sequence of values).
        aligns: Per-column alignment: 'l' (left), 'r' (right), 'c' (center).
                Defaults to left for all columns.

    Example::

        ┌──────────┬──────────┬──────────┐
        │ Param    │ Carbon % │ TCO %    │
        ├──────────┼──────────┼──────────┤
        │ 1.000    │   -14.0% │    -8.2% │
        │ 1.500    │    +2.3% │    +5.1% │
        └──────────┴──────────┴──────────┘
    """
    if not headers:
        return ""

    n_cols = len(headers)
    if aligns is None:
        aligns = ['l'] * n_cols

    # Compute column widths
    widths = [len(str(h)) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < n_cols:
                widths[i] = max(widths[i], len(str(cell)))

    # Add padding
    widths = [w + 2 for w in widths]  # 1 space each side

    def _align_cell(text: str, width: int, align: str) -> str:
        inner = width - 2  # subtract padding spaces
        s = str(text)
        if align == 'r':
            return ' ' + s.rjust(inner) + ' '
        elif align == 'c':
            return ' ' + s.center(inner) + ' '
        else:
            return ' ' + s.ljust(inner) + ' '

    def _h_line(left: str, mid: str, right: str) -> str:
        segments = [_LIGHT_H * w for w in widths]
        return left + mid.join(segments) + right

    lines = []
    # Top border
    lines.append(_h_line(_TL, _TJ, _TR))
    # Header row
    cells = [_align_cell(str(h), widths[i], aligns[i]) for i, h in enumerate(headers)]
    lines.append(_VL + _VL.join(cells) + _VL)
    # Header separator
    lines.append(_h_line(_LJ, _CJ, _RJ))
    # Data rows
    for row in rows:
        cells = []
        for i in range(n_cols):
            val = row[i] if i < len(row) else ''
            cells.append(_align_cell(str(val), widths[i], aligns[i]))
        lines.append(_VL + _VL.join(cells) + _VL)
    # Bottom border
    lines.append(_h_line(_BL, _BJ, _BR))

    return "\n".join(lines)


def info_line(text: str, indent: int = 2) -> str:
    """Indented informational text with diamond marker.

    Example::

        ◆ Breakeven value: 0.85
    """
    prefix = ' ' * indent
    return f"{prefix}◆ {text}"


def badge(label: str, value: str, indent: int = 2) -> str:
    """Highlighted key result with arrow marker.

    Example::

        ▸ Carbon: -14.02%
    """
    prefix = ' ' * indent
    return f"{prefix}▸ {label}: {value}"


def note_block(lines_list: Sequence[str], indent: int = 2) -> str:
    """Indented note section with dot markers.

    Example::

        · Negative % = savings vs baseline
        · Positive % = increase vs baseline
    """
    prefix = ' ' * indent
    return "\n".join(f"{prefix}· {line}" for line in lines_list)


def separator(width: int = 60) -> str:
    """Light horizontal rule."""
    return _LIGHT_H * width


# ── ANSI Color Post-Processing ─────────────────────────────────────

# ANSI escape codes
_RESET = '\033[0m'
_BOLD = '\033[1m'
_DIM = '\033[2m'
_GREEN = '\033[32m'
_RED = '\033[31m'
_YELLOW = '\033[33m'
_CYAN = '\033[36m'
_BLUE = '\033[34m'
_MAGENTA = '\033[35m'


def colorize(text: str) -> str:
    """Apply ANSI colors to formatted text via regex pattern matching.

    Detects and colorizes:
    - Title lines (═══) → bold cyan
    - Heading underlines (───) → dim
    - Positive percentages (+N%) → red
    - Negative percentages (-N%) → green
    - Key result markers (◆, ▸) → yellow
    - Note markers (·) → dim
    - Table borders (box-drawing chars) → dim
    - "Yes"/"No" values → green/red
    - "N/A" → dim yellow
    """
    lines = text.split('\n')
    result = []

    for line in lines:
        colored = _colorize_line(line)
        result.append(colored)

    return '\n'.join(result)


def _colorize_line(line: str) -> str:
    """Apply color to a single line."""
    stripped = line.strip()

    # Title lines with ═══
    if _HEAVY_H in line and not line.strip().startswith(_VL):
        return f"{_BOLD}{_CYAN}{line}{_RESET}"

    # Heading underlines (lines of ─── only)
    if stripped and all(c == _LIGHT_H for c in stripped):
        return f"{_DIM}{line}{_RESET}"

    # Table rows (contain │)
    if _VL in line:
        return _colorize_table_row(line)

    # Table borders (┌┐└┘├┤┬┴┼─)
    if stripped and stripped[0] in (_TL, _BL, _LJ):
        return f"{_DIM}{line}{_RESET}"

    # Info lines with ◆
    if '◆' in line:
        return _colorize_info_line(line)

    # Badge lines with ▸
    if '▸' in line:
        return _colorize_badge_line(line)

    # Note lines with ·  (but not dot leaders)
    if stripped.startswith('·'):
        return f"{_DIM}{line}{_RESET}"

    # Lines with percentages
    line = _colorize_percentages(line)

    return line


def _colorize_table_row(line: str) -> str:
    """Colorize a table data row, applying colors to percentage values."""
    # Split by │, colorize each cell, rejoin
    parts = line.split(_VL)
    colored_parts = []
    for part in parts:
        colored_parts.append(_colorize_cell(part))
    return f"{_DIM}{_VL}{_RESET}".join(colored_parts)


def _colorize_cell(cell: str) -> str:
    """Colorize the content of a single table cell."""
    stripped = cell.strip()
    if not stripped:
        return cell

    # Percentage values
    if re.match(r'^[+-]\d+\.?\d*%$', stripped):
        if stripped.startswith('+'):
            return cell.replace(stripped, f"{_RED}{stripped}{_RESET}")
        else:
            return cell.replace(stripped, f"{_GREEN}{stripped}{_RESET}")

    # Yes/No
    if stripped == 'Yes':
        return cell.replace(stripped, f"{_GREEN}{stripped}{_RESET}")
    if stripped == 'No':
        return cell.replace(stripped, f"{_RED}{stripped}{_RESET}")

    # N/A
    if stripped == 'N/A':
        return cell.replace(stripped, f"{_DIM}{_YELLOW}{stripped}{_RESET}")

    return cell


def _colorize_info_line(line: str) -> str:
    """Colorize an info line (◆ marker)."""
    return line.replace('◆', f"{_YELLOW}◆{_RESET}")


def _colorize_badge_line(line: str) -> str:
    """Colorize a badge line (▸ marker), with value colored by sign."""
    line = line.replace('▸', f"{_YELLOW}▸{_RESET}")
    line = _colorize_percentages(line)
    return line


def _colorize_percentages(line: str) -> str:
    """Colorize +N% and -N% patterns in a line."""
    def _repl(m: re.Match) -> str:
        s = m.group(0)
        if s.startswith('+'):
            return f"{_RED}{s}{_RESET}"
        else:
            return f"{_GREEN}{s}{_RESET}"

    return re.sub(r'[+-]\d+\.?\d*%', _repl, line)

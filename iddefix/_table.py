"""Shared resonator table layout for terminal and notebook output."""

from html import escape
from typing import Sequence

HEADERS = ("Rs [Ohm/m or Ohm]", "Q", "fres [Hz]")
COLORS = {
    "blue": ("\033[34m", "#3b82f6"),
    "red": ("\033[31m", "#ef4444"),
}
RESET = "\033[0m"


def display_resonator_table(
    rows: Sequence[Sequence[str]],
    *,
    to_markdown: bool = False,
    minimum_widths: tuple[int, int, int] = (24, 18, 25),
    colors: Sequence[Sequence[str | None]] | None = None,
) -> None:
    """Display three preformatted parameter columns with numbered rows."""
    if to_markdown:
        print("\n")
        print("| Resonator | Rs [Ohm/m or Ohm] | Q | fres [Hz] |")
        print("|-----------|------------------|---|-----------|")
        for i, row in enumerate(rows, start=1):
            print(f"| {i} | {row[0]} | {row[1]} | {row[2]} |")
        return

    widths = [
        max(minimum_widths[j], len(HEADERS[j]), *(len(row[j]) + 2 for row in rows))
        for j in range(3)
    ]
    header = (
        f"{'Resonator':^10}|{HEADERS[0]:^{widths[0]}}|"
        f"{HEADERS[1]:^{widths[1]}}|{HEADERS[2]:^{widths[2]}}"
    )
    separator = "-" * len(header)
    lines = [separator, header, separator]

    for i, row in enumerate(rows):
        cells = [f"{row[j]:^{widths[j]}}" for j in range(3)]
        if colors is not None:
            for j, color in enumerate(colors[i]):
                if color is not None:
                    cells[j] = f"{COLORS[color][0]}{cells[j]}{RESET}"
        lines.append(f"{i + 1:^10d}|{'|'.join(cells)}")
    lines.append(separator)

    try:
        from IPython import get_ipython
        from IPython.display import HTML, display
    except ImportError:
        shell = None
    else:
        shell = get_ipython()

    if shell is not None and getattr(shell, "kernel", None) is not None:
        table_html = escape("\n".join(lines))
        for ansi, html_color in COLORS.values():
            table_html = table_html.replace(ansi, f'<span style="color: {html_color}">')
        table_html = table_html.replace(RESET, "</span>")
        display(
            HTML(
                '<pre style="font-family: monospace; white-space: pre; '
                f'overflow-x: auto">{table_html}</pre>'
            )
        )
    else:
        print("\n")
        print("\n".join(lines))

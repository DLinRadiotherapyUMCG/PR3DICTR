from rich.console import Console
from rich.text import Text


def log_table(rich_table):
    console = Console()
    with console.capture() as capture:
        console.print(rich_table)
    return Text.from_ansi(capture.get())
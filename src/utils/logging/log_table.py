from rich.console import Console
from rich.text import Text


def log_table(rich_table):    # NOTE: DANIEL: what does this do?
    console = Console()
    with console.capture() as capture:
        console.print(rich_table)
    return Text.from_ansi(capture.get())
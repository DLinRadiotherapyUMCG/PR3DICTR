import logging

import torch
from rich.console import Console
from rich.logging import RichHandler


def setup_logging(log_level ='INFO'):
    # initialises the logger
    
    logging.basicConfig(
        level=log_level.upper(),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(markup=True, rich_tracebacks=True, tracebacks_suppress=[torch])]
    )

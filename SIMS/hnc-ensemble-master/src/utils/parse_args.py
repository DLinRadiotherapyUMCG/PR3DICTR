import argparse
import logging

from src.constants import TOXICITIES


def parse_args():
    """
    Parse the command line arguments.
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--toxicity', type=str, default=TOXICITIES[0], nargs='?')
    parser.add_argument('--log', type=str, default='INFO', nargs='?')
    args = parser.parse_args()

    # Load the toxicity
    tox = args.toxicity
    log_level = args.log

    # Check if the toxicity exists
    if args.toxicity not in TOXICITIES:
        raise ValueError('Toxicity not found. Please choose from the following: ' + ', '.join(TOXICITIES))

    return tox, log_level

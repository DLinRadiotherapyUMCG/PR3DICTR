import argparse

def parse_args():
    """
    Parse the command line arguments.
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', type=str, default='INFO', nargs='?')
    args = parser.parse_args()

    # Load the toxicity
    log_level = args.log

    return log_level
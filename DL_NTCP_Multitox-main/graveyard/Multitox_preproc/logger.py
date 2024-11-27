import logging

class Logger:
    def __init__(self, output_filename=None):
        logging.basicConfig(filename=output_filename, format='%(asctime)s - %(message)s', level=logging.INFO,
                            filemode='w')

    def my_print(self, message, level='info'):
        """
        Manual print operation.

        Args:
            message: input string.
            level: level of logging.

        Returns:

        """
        if level == 'info':
            print_message = 'INFO: {}'.format(message)
            logging.info(print_message)
        elif level == 'exception':
            print_message = 'EXCEPTION: {}'.format(message)
            logging.exception(print_message)
        elif level == 'warning':
            print_message = 'WARNING: {}'.format(message)
            logging.warning(print_message)
        else:
            print_message = 'INFO: {}'.format(message)
            logging.info(print_message)
        print(print_message)

    def close(self):
        logging.shutdown()
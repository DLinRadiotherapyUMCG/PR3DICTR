"""
This script contains general classes and functions used for the data preparation step. The functions are sorted in
alphabetical order.
"""
import os
import re
import shutil
import logging
import numpy as np
from pydicom.multival import MultiValue
from tabulate import tabulate


class Logger(object):
    def __init__(self, output_filename=None, suppress_CLI_prints=False):

        # initialise the logger itself
        time_format = "%Y-%m-%d %H:%M:%S"
        print_format = '%(asctime)s - %(message)s'

        formatter = logging.Formatter(fmt=print_format, datefmt=time_format)
        l = logging.getLogger("my_logger")
        # a reduced logger that only writes to file (for during HP tuning, where CLI prints are not needed)
        if suppress_CLI_prints: 
            l.setLevel(logging.WARNING)
            if output_filename:
                fileHandler = logging.FileHandler(output_filename, mode='w') # for writing to file
                fileHandler.setFormatter(formatter)
            streamHandler = logging.StreamHandler()  # for CLI prints
            streamHandler.setFormatter(formatter)
            l.setLevel(logging.INFO)
            if output_filename:
                l.addHandler(fileHandler)
            self.logger = l

        # logging as normal (how Hung had it)
        else: 
            try: # make sure that theres no existing handlers
                l.handlers.pop()
            except:
                pass
            l.setLevel(logging.INFO)
            # make handler to print to CLI, and add that to the logger
            streamHandler = logging.StreamHandler()
            streamHandler.setFormatter(formatter)
            l.addHandler(streamHandler)
            if output_filename:
                # now the same for the file handler (to write the log.txt file)
                fileHandler = logging.FileHandler(output_filename, mode='w')            
                fileHandler.setFormatter(formatter)
                l.addHandler(fileHandler)

            self.logger = l
            

        self.dict_names = []
        self.auc_dicts_list = []

        self.best_auc_dict = None
        self.best_epoch = None
        self.current_epoch = None
              

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
            self.logger.info(print_message)
        elif level == 'exception':
            print_message = 'EXCEPTION: {}'.format(message)
            self.logger.exception(print_message)
        elif level == 'warning':
            print_message = 'WARNING: {}'.format(message)
            self.logger.warning(print_message)
        else:
            print_message = 'INFO: {}'.format(message)
            self.logger.info(print_message)

    def close(self):
        try:
            self.logger.handlers.pop()
        except:
            pass
        logging.shutdown()
        del self.logger


    def add_epoch_dict(self, name, auc_dict, loss, avg_auc, epoch_n):
        self.current_epoch = epoch_n
        self.dict_names.append(name)
        #auc_dict = auc_dict.copy()
        auc_dict['Average AUC'] = avg_auc
        auc_dict['Loss'] = loss
        self.auc_dicts_list.append(auc_dict)

    def set_best_epoch_dict(self, best_auc_dict, loss, avg_auc, epoch_n):
        self.best_epoch = epoch_n
        best_auc_dict['Average AUC'] = avg_auc
        best_auc_dict['Loss'] = loss
        self.best_auc_dict = best_auc_dict

    def print_epoch_table(self, mode='epoch'):
        #decimals = config.nr_of_decimals
        auc_dicts_list = self.auc_dicts_list
        if mode == 'epoch':
            auc_dicts_list.append(self.best_auc_dict)
            self.dict_names.append('Best Validation')
            tablefmt = 'presto'
        else:
            tablefmt = 'pretty'
        names = self.dict_names
        keys = list(set().union(*auc_dicts_list))
        
        # Move "Loss" and "Average AUC" to the start of the list
        keys.remove("Loss")
        keys.remove("Average AUC")
        keys.sort()
        keys = ["Average AUC", "Loss"] + keys

        header = [''] + keys
        header = [h.replace('_', ' ').title() for h in header] # replace _ with space and capitalize
        rows = []
        
        for name, d in zip(names, auc_dicts_list):
            row_values = []
            for key in keys:
                row_values.append(f"{d[key]:.3f}" if ((key in d) and (type(d[key]) != str)) else "N/A")
            row = [name] + row_values
            rows.append(row)

        # pad the strings to be equal length
        # first find the max length of each column
        """
        max_length = max([len(h) for h in header])

        # pad all of the items in the rows and header to be the same length as the longest item
        header = [h.center(max_length) for h in header]
        rows = [[value.rjust(max_length) for value in row] for row in rows]
        """

        table = tabulate(rows, headers=header, tablefmt=tablefmt)
        table_split = table.split('\n')
        #table = table.split('\n')
        self.my_print(f"Current epoch: {self.current_epoch}.  Best validation epoch: {self.best_epoch}")
        
        for table_row in table_split:
            self.my_print(table_row)
        
        # delete the epochs we've just printed
        self.dict_names = []
        self.auc_dicts_list = []

def copy_folder(src, dst):
    """
    Copy contents of source (src) folder to destination (dst) folder.

    Note that the contents of the src folder are copied, but not the src folder itself! Therefore, it is recommended
    to specify a new folder name in the 'dst' argument. The destination folder does NOT need to exist, because of
    'dirs_exist_ok' argument in shutil.copytree().

    For example, if we want to copy the folder C:/foo to folder D:/bar, then use:
        src = 'C:/foo'
        dst = 'D:/bar/foo' (instead of 'D:/bar')

    Args:
        src: source folder, containing contents to be copied to the destination folder
        dst: destination folder, i.e. folder that will get copies of the folder+files from the source folder

    Returns:

    """
    # Recursively copy an the content of a directory tree rooted at src to a directory named dst. The 'dirs_exist_ok'
    # argument dictates whether to raise an exception in case dst or any missing parent directory already exists.
    shutil.copytree(src, dst, dirs_exist_ok=True)


def copy_file(src, dst):
    """
    Copy source (src) file to destination (dst) file.

    Note that renaming is possible.

    Args:
        src: source file to be copied to the destination file
        dst: destination file (potentially renamed) from source file.

    Returns:

    """
    shutil.copy(src, dst)


def create_folders_if_not_exist(folders, logger=None):
    for folder in folders:
        create_folder_if_not_exists(folder, logger=logger)


def create_folder_if_not_exists(folder, logger=None):
    """
    Create folder if it does not exist yet.

    It is also possible to create subfolders. For example, if path D:/foo exists and we want to create D:/foo/bar/baz,
    but D:/foo/bar does not exist, then it is still possible to directly create D:/foo/bar/baz by
    create_folder_if_not_exists('D:/foo/bar/baz').

    Args:
        folder:
        logger:

    Returns:

    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    if logger is not None:
        logger.my_print('Creating folder: {}'.format(folder))


def get_all_folders(path):
    """
    Get all folders, including sub-folders, in path.

    Args:
        path (str):

    Returns:
        list of folders in path.
    """
    return [x[0] for x in os.walk(path)]


def get_folders_diff_or_intersection(list_1, list_2, mode):
    """
    mode='diff': Get list of folders (as strings) that are present in list_1, but not in list_2.
    mode='intersection': Get list of folders (as strings) that are present in both list_1 and list_2.
    Important: no subfolders are considered!

    Args:
        list_1 (list): folders (as strings) in list_1
        list_2 (list): folders (as strings) in list_2
        mode (str): either 'diff' (difference) or 'intersection'

    Returns:

    """
    if mode == 'diff':
        output = [x for x in list_1 if x not in list_2]
    elif mode == 'intersection':
        output = [x for x in list_1 if x in list_2]
    else:
        raise ValueError('Mode {} is not valid.'.format(mode))

    # Sort list
    output = sort_human(output)

    return output


def list_to_txt_str(l, sep='\n'):
    """
    Convert a list to string. For example, if l = ['a', 'b', 'c', 1] and sep='\n', then the output will be:
    txt = 'a\nb\nc\n1\n'

    Args:
        l: list with elements
        sep: separator

    Returns:

    """
    # Initiate txt
    txt = ''

    for i in l:
        txt += str(i) + sep

    return txt


def move_folder_or_file(src, dst):
    """
    Move source (src) folder/file to destination (dst) folder.

    Args:
        src: source file to be moved to the destination folder
        dst: destination folder

    Returns:

    """
    shutil.move(src, dst)


def round_nths_of_list(in_list, n, decimal):
    """
    Round every n^{th} element of `in_list` to decimcal.

    Args:
        in_list (list):
        n (int):
        decimal (int):

    Returns:

    """
    return [round(x, decimal) if (i + 1) % n == 0 else x for i, x in enumerate(in_list)]


def set_default(obj):
    """
    Set JSON defaults: determine alternative datatype for datatypes that are invalid for JSONs. Otherwise,
    Python will raise an error.

    Args:
        obj:

    Returns:

    """
    if isinstance(obj, set):
        return list(obj)
    if isinstance(obj, MultiValue):
        return list(obj)
    if np.issubdtype(obj, np.unsignedinteger):
        return int(obj)
    raise TypeError


def sort_human(l):
    """
    Sort the input list. However normally with l.sort(), e.g., l = ['1', '2', '10', '4'] would be sorted as
    l = ['1', '10', '2', '4']. The sort_human() function makes sure that l will be sorted properly,
    i.e.: l = ['1', '2', '4', '10'].
    
    Source: https://stackoverflow.com/questions/3426108/how-to-sort-a-list-of-strings-numerically

    Args:
        l: to-be-sorted list

    Returns:
        l: properly sorted list
    """
    convert = lambda text: float(text) if text.isdigit() else text
    alphanum = lambda key: [convert(c) for c in re.split('([-+]?[0-9]*\.?[0-9]*)', key)]
    l.sort(key=alphanum)
    return l



def get_all_combinations(toxicity, timepoint, spacer="_"):
    """
    This function generates column names for all of the combinations of toxicities and timepoints
    e.g. input ["Xerostomia","Taste"] and ["M06", "M12"] will return:
    ["Xerostomia_M06", "Xerostomia_M12", "Taste_M06", "Taste_M12"]
    """
    # get all the possible pairs for toxicity and timepoint
    pairs = [(toxicity[i], timepoint[j]) for i in range(len(toxicity))
            for j in range(len(timepoint))]
    
    # join each pair into a string, with the 'spacer' in-between
    keys = [spacer.join([i,j]) for i,j in pairs]

    return keys
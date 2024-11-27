import glob
import os
import config





if __name__ == '__main__':     

    folder_path = config.exp_root_dir

    # Define the list of filenames you want to delete
    filenames_to_delete = [config.filename_best_model_pth, config.filename_best_scheduler_pth, config.filename_best_optimizer_pth]

    # Use glob to find all files with the specified filenames in all subdirectories
    file_paths = []
    for filename in filenames_to_delete:
        file_paths.extend(glob.glob(os.path.join(folder_path, '**', filename), recursive=True))

    # Delete the files
    for file_path in file_paths:
        os.remove(file_path)
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import time

import data_preproc_config as data_prc_cfg
from data_preproc_functions import Logger, create_folders_if_not_exist

import data_folder_types
import data_collection
import data_preproc_ct_rtdose
import data_preproc_ct_segmentation_map_citor
import data_preproc_ct_segmentation_map_dlc
import data_preproc_ct_segmentation_map
import data_preproc
import data_preproc_exclude_patients
import checks.check_folder_structure as check_folder_structure
import checks.check_rtstruct as check_rtstruct
import checks.check_rtdose as check_rtdose
import checks.check_data as check_data
import checks.check_data_preproc_ct_rtdose as check_data_preproc_ct_rtdose
import checks.check_data_preproc_ct_segmentation_map as check_data_preproc_ct_segmentation_map
import checks.check_data_preproc as check_data_preproc
from data_preproc_functions import create_folder_if_not_exists

def create_preproc_folders():
    create_folder_if_not_exists(data_prc_cfg.save_root_dir)
    create_folder_if_not_exists(data_prc_cfg.save_dir_citor)
    create_folder_if_not_exists(data_prc_cfg.save_root_dir_2)
    create_folder_if_not_exists(data_prc_cfg.save_dir)
    create_folder_if_not_exists(data_prc_cfg.save_dir_final_dataset)

    
    create_folder_if_not_exists(data_prc_cfg.save_dir_segmentation_map_citor)
    create_folder_if_not_exists(data_prc_cfg.save_dir_segmentation_map_dlc)
    create_folder_if_not_exists(data_prc_cfg.save_dir_segmentation_map)
    create_folder_if_not_exists(data_prc_cfg.save_dir_dataset_full)
    #create_folder_if_not_exists(save_dir_dataset)
    create_folder_if_not_exists(data_prc_cfg.save_dir_figures)
    create_folder_if_not_exists(data_prc_cfg.save_dir_figures_all)

if __name__ == '__main__':
    create_preproc_folders()
    # Initialize variables
    save_root_dir = data_prc_cfg.save_root_dir
    main_logger = Logger(os.path.join(save_root_dir, data_prc_cfg.filename_main_logging_txt))
    start = time.time()

    # """
    # Run the separate modules
    # main_logger.my_print('Running data_folder_types...')
    # data_folder_types.main()
    
    
    
    # main_logger.my_print('Running check_folder_structure...')
    # check_folder_structure.main()
    
    
    # main_logger.my_print('Running data_collection...')
    # data_collection.main()
    
    """
    # The following checks can be skipped
    # main_logger.my_print('Running check_rtstruct...')
    # check_rtstruct.main()  # may be skipped
    
    # main_logger.my_print('Running check_rtdose...')
    # check_rtdose.main()  # can be skipped
    
    # main_logger.my_print('Running check_data...')
    # check_data.main()  # can be skipped
    """
    
    
    # main_logger.my_print('Running data_preproc_ct_rtdose...')
    # data_preproc_ct_rtdose.main()
    
    # main_logger.my_print('Running check_data_preproc_ct_rtdose...')
    # check_data_preproc_ct_rtdose.main()
    
    
    # main_logger.my_print('Running data_preproc_segmentation_map_citor...')
    # data_preproc_ct_segmentation_map_citor.main()
    
    
    # main_logger.my_print('Running data_preproc_ct_segmentation_map_dlc...')
    # data_preproc_ct_segmentation_map_dlc.main()
    
    
    # main_logger.my_print('Running data_preproc_ct_segmentation_map...')
    # data_preproc_ct_segmentation_map.main()
    
    # main_logger.my_print('Running check_data_preproc_ct_segmentation_map...')
    # check_data_preproc_ct_segmentation_map.main()
    
    

    # main_logger.my_print('Running data_preproc...')
    # data_preproc.main()

    main_logger.my_print('Running check_data_preproc...')
    check_data_preproc.main()
    
    
    main_logger.my_print('Running data_preproc_exclude_patients...')
    # # # Note: if len(test_patients_list) > 0, but we need exclude_patients.csv for all patients, then Search for:
    # # # `# TODO: temporary: consider all patients, for exclude_patients.csv`, and uncomment/enable the code lines.
    #data_preproc_exclude_patients.main()   
    
    
    
    
    
    end = time.time()
    main_logger.my_print('Elapsed time: {time} seconds'.format(time=round(end - start, 3)))
    main_logger.my_print('DONE!')
    

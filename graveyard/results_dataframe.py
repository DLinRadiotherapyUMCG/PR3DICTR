import os
from datetime import datetime
from typing import Optional, TypedDict

import pandas as pd

# Define the columns
COLUMNS = ["PatientID", "label", "pred_lr_0", "pred_lr_1", "pred_lr_2", "pred_lr_3", "pred_lr_4",
           "pred_dl_0", "pred_dl_1", "pred_dl_2", "pred_dl_3", "pred_dl_4",
           "mean_lr", "mean_dl", "entropy_lr", "entropy_dl", "std_lr", "std_dl", "cv_lr", "cv_dl",
           "updated_at", "predictive_variance_lr", "predictive_variance_dl", "predictive_entropy_lr",
           "predictive_entropy_dl", "aleatoric_uncertainty_lr", "aleatoric_uncertainty_dl",
             "epistemic_uncertainty_lr", "epistemic_uncertainty_dl"]


class Row(TypedDict, total=False):
    PatientID: str
    label: int
    pred_lr_0: Optional[float]
    pred_lr_1: Optional[float]
    pred_lr_2: Optional[float]
    pred_lr_3: Optional[float]
    pred_lr_4: Optional[float]
    pred_dl_0: Optional[float]
    pred_dl_1: Optional[float]
    pred_dl_2: Optional[float]
    pred_dl_3: Optional[float]
    pred_dl_4: Optional[float]
    mean_lr: Optional[float]
    mean_dl: Optional[float]
    entropy_lr: Optional[float]
    entropy_dl: Optional[float]
    std_lr: Optional[float]
    std_dl: Optional[float]
    cv_lr: Optional[float]
    cv_dl: Optional[float]
    updated_at: datetime


def load_results(file_path):
    """
    Load a dataframe from a file. If the file does not exist or a column is missing,
    create a new dataframe with the specified columns.

    :param file_path: Path to the file.
    :return: Loaded or created dataframe.
    """
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, dtype={'PatientID': str})

        # Check if any column is missing
        missing_columns = set(COLUMNS) - set(df.columns)
        for column in missing_columns:
            df[column] = None  # Add the missing column with None values
    else:
        # Create a new dataframe with the specified columns
        df = pd.DataFrame(columns=COLUMNS)

    return df


def save_results(df, file_path):
    """
    Save a dataframe to a file. Validate if the dataframe has the correct columns.

    :param df: Dataframe to save.
    :param file_path: Path to the file.
    """
    # Validate the dataframe columns
    if set(df.columns) != set(COLUMNS):
        raise ValueError("The dataframe does not have the correct columns.")

    df.to_csv(file_path, index=False)


def update_patient_result(df, patient_id, update_dict):
    """
    Update the patient's results in the dataframe. If the patient does not exist, create a new row for them.

    :param df: The dataframe containing the results.
    :param patient_id: The ID of the patient to update.
    :param update_dict: A dictionary where the keys are column names and the values are the new values.
    :return: The updated dataframe.
    """
    # Find the index of the patient in the dataframe
    patient_index = df[df['PatientID'] == patient_id].index

    if len(patient_index) == 0:
        # If the patient is not in the dataframe, create a new row for them
        new_row: Row = {column: None for column in df.columns}
        new_row['PatientID'] = patient_id
        new_row.update(update_dict)
        new_row['updated_at'] = datetime.now()
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        # Update the specified columns with their new values
        for column, new_value in update_dict.items():
            if column in df.columns:
                df.at[patient_index[0], column] = new_value  # Use the first item in patient_index
            else:
                raise ValueError(f"Column {column} not found in the dataframe.")
        df.at[patient_index[0], 'updated_at'] = datetime.now()  # Use the first item in patient_index

    return df

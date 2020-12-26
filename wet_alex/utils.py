import pandas as pd
import numpy as np

import sklearn
from sklearn.pipeline import Pipeline
import os


class customPipeline(Pipeline):
    """Extension to the existing pipeline class. We add a function to simply apply transforms one after another"""

    def __init__(self, steps):
        super().__init__(steps)

    def apply_transforms(self, df_dut: pd.DataFrame()):

        for _, name, transform in self._iter(with_final=True):
            df_dut = transform.transform(df_dut)

        return df_dut


class CustomFeatureHandler():
    """Apply transformation on the some feature/s

    Any custom transform function has to include:
    the "transform" function which takes and returns dataframe
    and "fit" function which won't be used in our case.

    """
    def __init__(self):
        pass

    def fit(self):
        # no fit required
        return self

    def transform(self, df_dut : pd.DataFrame()):
        return df_dut



def split_the_data(df: pd.DataFrame(), split_list):

    train_ds_part, valid_ds_part, test_ds_part = split_list
    total_ds_rows_n = df.shape[0]

    train_ds_part_int = round(total_ds_rows_n * train_ds_part)
    valid_ds_part_int = round(total_ds_rows_n * valid_ds_part)
    test_ds_part_int = total_ds_rows_n - valid_ds_part_int - train_ds_part_int

    train_end_idx = train_ds_part_int
    valid_end_idx = train_ds_part_int + valid_ds_part_int

    split_data = np.split(df, [train_end_idx, valid_end_idx], axis=0)

    print(f"Total size: {total_ds_rows_n}\n\
    Train df: {split_data[0].shape[0]}\n\
    Valid df: {split_data[1].shape[0]}\n\
    Test df: {split_data[2].shape[0]}")

    # return indexes to 0
    split_data[1].index -= split_data[1].index.start
    split_data[2].index -= split_data[2].index.start

    ds_dict = {}

    ds_dict["train"] = split_data[0]
    ds_dict["train_original"] = split_data[0]

    ds_dict["valid"] = split_data[1]
    ds_dict["valid_original"] = split_data[1]

    ds_dict["test"] = split_data[2]
    ds_dict["test_original"] = split_data[2]

    return ds_dict



def save_csv_files(ds_dict : dict, output_folder_name : str):

    """Save all 6 pd.Dataframe objects in 6 csv files"""

    curdir = os.getcwd()
    output_dir = os.path.abspath(os.path.join(curdir, output_folder_name))

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)


    ds_dict["train"].to_csv(os.path.join(output_dir, "train.csv"), index=False)
    ds_dict["train_original"].to_csv(os.path.join(output_dir, "train_original.csv"), index=False)
    ds_dict["valid"].to_csv(os.path.join(output_dir, "valid.csv"), index=False)
    ds_dict["valid_original"].to_csv(os.path.join(output_dir, "valid_original.csv"), index=False)
    ds_dict["test"].to_csv(os.path.join(output_dir, "test.csv"), index=False)
    ds_dict["test_original"].to_csv(os.path.join(output_dir, "test_original.csv"), index=False)


    return

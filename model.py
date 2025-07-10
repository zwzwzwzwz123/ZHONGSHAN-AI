import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split

from Model_Manager.model_manager import Model_Manager


class DataCenter_Dataset(Dataset):
    def __init__(
        self, df, seq_len, pred_len, input_columns, output_columns, flag="train"
    ):
        self.df = df
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.seq_len = seq_len
        self.pred_len = pred_len

        assert flag in ["train", "test"]
        type_map = {"train": 0, "test": 1}
        self.set_type = type_map[flag]
        self.__read_data__()

    def __read_data__(self):
        # obtain the mean and std, and store it.
        df_x = self.df[self.input_columns]
        df_y = self.df[self.output_columns]
        # x_mean, x_std = df_x.mean(), df_x.std()
        y_mean, y_std = df_y.mean(), df_y.std()
        if not os.path.exists("model"):
            os.makedirs("model")
        # np.save("model/x_scale.npy", np.array([x_mean, x_std]))
        np.save("model/y_scale.npy", np.array([y_mean, y_std]))

        # df_x = (df_x - x_mean) / (1e-9 + x_std)
        df_y = (df_y - y_mean) / (1e-9 + y_std)
        num_train = int(len(self.df) * 0.8)
        border1s = [0, num_train]
        border2s = [num_train, len(self.df)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        self.data_x = df_x.values[border1:border2]
        self.data_y = df_y.values[border1:border2]

        self.input_shape = self.data_x.shape[1]
        self.output_shape = self.data_y.shape[1]

    def __getitem__(self, index):
        insample = np.zeros((self.seq_len, self.input_shape))
        outsample = np.zeros((self.pred_len, self.output_shape))
        s_begin = index
        s_end = min(len(self.data_x), s_begin + self.seq_len)
        r_begin = max(0, s_end)
        r_end = min(len(self.data_y), r_begin + self.pred_len)
        insample_window = self.data_x[s_begin:s_end, :]
        outsample_window = self.data_y[r_begin:r_end, :]
        insample[-len(insample_window) :, :] = insample_window
        outsample[: len(outsample_window), :] = outsample_window
        return insample, outsample

    def __len__(self):
        return len(self.data_x)


def train(
    df,
    manager,
    model_name,
    model_path,
    input_columns,
    output_columns,
    args,
    batch_size=512,
    num_workers=0,
    init=1,
    device="cpu",
):
    args["input_size"], args["output_size"] = len(input_columns), len(output_columns)
    train_dataset = DataCenter_Dataset(
        df,
        args["sequence_length"],
        args["predict_length"],
        input_columns,
        output_columns,
        flag="train",
    )
    trainDl = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    if init == 1:
        save_model = manager.create_init_model(model_name, args, device=device)
        save_model = manager.pretrained_model(
            model_name, trainDl, model_path=model_path
        )
    if init == 0:
        save_model = manager.fine_tune_existed_model(
            model_name, trainDl, model_path=model_path
        )
    torch.save(save_model, model_path)
    return save_model

from bayes_opt import BayesianOptimization
import numpy as np
import torch
import pandas as pd


class Bayes_Optimization:
    def __init__(self, configs, model, new_data: pd.DataFrame) -> None:
        self.model = model
        self.ac_num_params = configs["ac_num_params"]
        # self.converter_freq_setting = configs["converter_freq_setting"]
        self.input_optimization_uid_list = configs["uid"]["input_optimization_uid"]
        self.status = new_data
        self.ac_onoff_status_setting_list = self.input_optimization_uid_list[
            "ac_onoff_status_setting"
        ]
        self.ac_temp_setting_list = self.input_optimization_uid_list[
            "ac_temperature_setting"
        ]
        self.temp_set_change_step = 0.5

    def init_pbounds(self):
        ac_onoff_status_setting_bounds = (0, 1)
        ac_temp_setting_bounds = (17, 29)
        # ac_freq_setting_bounds = (35, 50)
        pbounds = {
            f"{self.ac_onoff_status_setting_list[i]}": ac_onoff_status_setting_bounds
            for i in range(self.ac_num_params)
        }
        pbounds.update(
            {
                f"{self.ac_temp_setting_list[i]}": ac_temp_setting_bounds
                for i in range(self.ac_num_params)
            }
        )
        # pbounds.update(
        #     {
        #         f"freq{i+1}": ac_freq_setting_bounds
        #         for i in range(self.converter_freq_setting)
        #     }
        # )
        return pbounds

    def rebuilt_input(self, params: pd.DataFrame) -> pd.DataFrame:
        # 空调回风温度和室温是状态变量，params是可调参数构成的list
        # 读进未归一化的数据
        ac_temperature = self.status[
            self.input_optimization_uid_list["ac_temperature"]
        ]  # .values.tolist()[0]
        room_temperature = self.status[
            self.input_optimization_uid_list["room_temperature"]
        ]
        result = pd.concat([ac_temperature, room_temperature, params], axis=1)
        return result
    
    def index_step_change(self, data:pd.DataFrame):
        # 参数重新排序
        index_list = self.ac_onoff_status_setting_list + self.ac_temp_setting_list
        reindex_data = data.reindex(columns=index_list)
        # 贝叶斯给出的参数取到0/1/0.5步长
        reindex_data.iloc[:, : self.ac_num_params] = reindex_data.iloc[
            :, : self.ac_num_params
        ].round()
        reindex_data.iloc[:, self.ac_num_params : self.ac_num_params * 2] = (
            np.round(
                (reindex_data.iloc[:, self.ac_num_params : self.ac_num_params * 2])
                / self.temp_set_change_step
            )
            * self.temp_set_change_step
        )
        return reindex_data

    def target_function(self, input, model):
        temp_safe = 0
        # input 归一化
        # [mean, std] = np.load("model/x_scale.npy")
        # input = (input - mean) / (std + 1e-9)
        with torch.no_grad():
            input = input.values
            y = model.pred(input)  # 这里y是numpy array   1*4*7
        col_index = self.ac_num_params
        [mean, std] = np.load("model/y_scale.npy")
        if type(y) is torch.Tensor:
            y = y.cpu().detach().numpy() * std + mean
        else:
            y = y * std + mean
        ac_temp_column = y[0][:, :col_index]  # 空调回风温度
        temperature_high = ac_temp_column.max()
        # result = y.detach().numpy().reshape(-1)

        # 判断是否超过安全边界
        if temperature_high >= 29:
            temp_safe = -100
        # 超过安全边界就加一个惩罚
        result = temperature_high + temp_safe
        return result.item()

    def discrete_target_function(self, **kwargs):
        # 拆分输入参数
        pd_kwargs = pd.DataFrame([kwargs])
        reindex_pd_kwargs = self.index_step_change(pd_kwargs)
        # # 参数重新排序
        # index_list = self.ac_onoff_status_setting_list + self.ac_temp_setting_list
        # reindex_pd_kwargs = pd_kwargs.reindex(columns=index_list)
        # # 贝叶斯给出的参数取到0/1/0.5步长
        # reindex_pd_kwargs.iloc[:, : self.ac_num_params] = reindex_pd_kwargs.iloc[
        #     :, : self.ac_num_params
        # ].round()
        # reindex_pd_kwargs.iloc[:, self.ac_num_params : self.ac_num_params * 2] = (
        #     np.round(
        #         (reindex_pd_kwargs.iloc[:, self.ac_num_params : self.ac_num_params * 2])
        #         / self.temp_set_change_step
        #     )
        #     * self.temp_set_change_step
        # )
        # 贝叶斯给出的参数反归一化
        # [mean, std] = np.load("model/x_scale.npy")
        # mean_kwargs = mean[-self.ac_num_params * 2 :]
        # std_kwargs = std[-self.ac_num_params * 2 :]
        # new_kwargs = reindex_pd_kwargs * std_kwargs + mean_kwargs
        model_kwargs = pd.concat([reindex_pd_kwargs] * 8, ignore_index=True)
        params_list = self.ac_onoff_status_setting_list + self.ac_temp_setting_list
        params = model_kwargs[params_list]
        # 加上状态变量的总输入
        input = self.rebuilt_input(params)
        return self.target_function(input=input, model=self.model)

    def bayes_optimization(self) -> dict:
        pbounds = self.init_pbounds()
        bo_result = BayesianOptimization(
            f=self.discrete_target_function,
            pbounds=pbounds,
            verbose=0,  # verbose = 2 时打印全部，verbose = 1 时打印运行中发现的最大值，verbose = 0 将什么都不打印
            random_state=10,
        )
        bo_result.maximize(init_points=10, n_iter=50, acq="ucb")
        result = bo_result.max
        params = result["params"]
        pd_params = pd.DataFrame([params])
        output_params = self.index_step_change(pd_params)
        return output_params

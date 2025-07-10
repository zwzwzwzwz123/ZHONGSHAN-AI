import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from .model_basic import Basic_model


class RandomForestModel(Basic_model):
    def __init__(self, args=None,device='cpu'):
        super().__init__()  # 调用父类的初始化方法
        self.args = args

        self.input_size = args["input_size"]
        self.output_size = args["output_size"]
        self.sequence_length = args["sequence_length"]
        self.predict_length = args["predict_length"]

        # args for RF
        self.n_estimators = args["n_estimators"]
        self.max_depth = args["max_depth"]

        # 创建一个随机森林回归器
        if args is not None:
            self.model = RandomForestRegressor(
                n_estimators=self.n_estimators, max_depth=self.max_depth
            )
        else:
            self.model = RandomForestRegressor()

    def pretrain_and_save(self, trainDl, model_path):
        train_x_list, train_y_list = [], []
        for train_x, train_y in trainDl:
            train_x_list.append(train_x.numpy())
            train_y_list.append(train_y.numpy())
        train_x = np.concatenate(train_x_list, axis=0)
        train_y = np.concatenate(train_y_list, axis=0)
        train_x = train_x.reshape(train_x.shape[0], -1)
        train_y = train_y.reshape(train_y.shape[0], -1)
        self.model.fit(train_x, train_y)
        joblib.dump(self.model, model_path)
        return self.model

    def fine_tune_and_save(self, trainDl, model_path):
        self.model = joblib.load(model_path)
        self.pretrain_and_save(trainDl, model_path)
        return self.model

    def pred(self, input):
        input = input.reshape(1, -1)
        output = self.model.predict(input).reshape(1, self.predict_length, self.output_size)
        return output

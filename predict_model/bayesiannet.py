import joblib
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import ConstantKernel as C

from .model_basic import Basic_model


class BayesianModel(Basic_model):
    def __init__(self, args=None,device='cpu'):
        super().__init__()  # 调用父类的初始化方法
        self.args = args

        self.input_size = args["input_size"]
        self.output_size = args["output_size"]
        self.sequence_length = args["sequence_length"]
        self.predict_length = args["predict_length"]

        self.n_restarts_optimizer = args["n_restarts_optimizer"]

        # 创建一个高斯过程回归模型
        if args is not None:
            self.model = GaussianProcessRegressor(
                n_restarts_optimizer=self.n_restarts_optimizer
            )
        else:
            kernel = C(1.0, (1e-4, 1e1)) * RBF(1.0, (1e-4, 1e1))
            self.model = GaussianProcessRegressor(
                kernel=kernel, n_restarts_optimizer=10
            )

    def pretrain_and_save(self, trainDl, model_path):
        train_x_list, train_y_list = [], []
        for train_x, train_y in trainDl:
            train_x_list.append(train_x.numpy())
            train_y_list.append(train_y.numpy())
        train_x = np.concatenate(train_x_list, axis=0)
        train_y = np.concatenate(train_y_list, axis=0)
        train_x = train_x.reshape(train_x.shape[0], -1)
        train_y = train_y.reshape(train_y.shape[0], -1)
        # 训练模型
        self.model.fit(train_x, train_y)
        joblib.dump(self.model, model_path)
        return self.model

    def fine_tune_and_save(self, trainDl, model_path):
        self.model = joblib.load(model_path)
        self.pretrain_and_save(trainDl, model_path)
        return self.model
        # TODO: 没有发现partial_fit方法
        # self.model.partial_fit(train_x, train_y)

    def pred(self, input):
        input = input.reshape(1, -1)
        output = self.model.predict(input).reshape(
            1, self.predict_length, self.output_size
        )
        return output

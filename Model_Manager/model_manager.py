import numpy as np

from predict_model import (
    BayesianModel,
    GRUModel,
    LSTMModel,
    MLPModel,
    RandomForestModel,
    TransformerModel,
)


class Model_Manager:
    def  __init__(self) -> None:
        self.model_dict = {'MLP':MLPModel,
                           'RF':RandomForestModel,
                           'Bayes':BayesianModel,
                           'LSTM':LSTMModel,
                           'GRU':GRUModel,
                           'Transformer':TransformerModel,
                           }
        self.exist_model_repository = {}

    ### 获得模型
    def _get_model(self,model_name):
        return self.exist_model_repository[model_name]
    ### 存入模型
    def _set_model(self,model,model_name):
        self.exist_model_repository[model_name] = model

    ### 创建初始模型
    def create_init_model(self, model_name, args,device):
        model = self.model_dict[model_name](args,device) # Instantiate model
        self._set_model(model, model_name)
        return model

    ### 预训练模型
    def pretrained_model(self,model_name,trainDl,model_path):
        model = self._get_model(model_name)
        model.pretrain_and_save(trainDl,model_path)
        self._set_model(model,model_name)
        return model

    ### 微调已保存的模型
    def fine_tune_existed_model(self,model_name,trainDl,model_path):
        model = self._get_model(model_name)
        model.fine_tune_and_save(trainDl,model_path)
        self._set_model(model,model_name)
        return model

    ### 预测
    def pred_by_model(self,model_name,input):
        model = self._get_model(model_name)
        # scale should be used when predict
        [mean, std] = np.load("model/scale.npy")
        output = model.pred((input- mean) / std)
        return output * std + mean

    # 计算模型预测的准确性
    def calculate_accuracy(self, model_name, test_x, test_y):
        model = self._get_model(model_name)
        mse_result = model.mse(test_x, test_y)
        # TODO: 需要修改
        return mse_result




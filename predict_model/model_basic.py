import torch.nn as  nn

class Basic_model(nn.Module):
    def __init__(self) -> None:
        super(Basic_model, self).__init__()

    
    def pretrain_and_save(self, train_x, train_y, model_path):
        pass
    
    def fine_tune_and_save(self, train_x, train_y, model_path):
        pass
     
    def pred(self):
        pass
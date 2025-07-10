import torch.nn as nn 
from .model_basic import Basic_model

class Cusomized_model(Basic_model):
    def __init__(self) -> None:
        super().__init__()
        
    def fit(self,input,label):
        '''
        The fit function of Customized model
        '''
        pass


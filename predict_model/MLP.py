from typing import Any

import torch
import torch.nn as nn

from .model_basic import Basic_model


class MLPModel(Basic_model):
    def __init__(self, args,device):
        super(MLPModel, self).__init__()
        self.args = args

        self.input_size = args["input_size"]
        self.output_size = args["output_size"]
        self.sequence_length = args["sequence_length"]
        self.predict_length = args["predict_length"]
        self.epochs = args["epochs"]
        self.lr = args["learning_rate"]
        self.device=device
        self.model = MLP(
            in_dim=self.input_size * self.sequence_length,
            out_dim=self.output_size * self.predict_length,
            hidden_dims=args["hidden_dims"],
        ).to(self.device)

    def pretrain_and_save(self, trainDl, model_path):
        self.model.train()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        for _ in range(self.epochs):
            for train_x, train_y in trainDl:
                # TODO: add to device funcs
                train_x = train_x.float().to(self.device)
                train_y = train_y.float().to(self.device)
                self.batch_size = train_x.shape[0]
                optimizer.zero_grad()
                y_pred = self.model(train_x.reshape(self.batch_size, -1))
                y_pred = y_pred.reshape(self.batch_size, self.predict_length, self.output_size)
                loss = criterion(y_pred, train_y)
                loss.backward()
                optimizer.step()
        torch.save(self.model.state_dict(), model_path)
        return self.model

    def fine_tune_and_save(self, trainDl, model_path):
        self.model.load_state_dict(torch.load(model_path))
        self.pretrain_and_save(trainDl, model_path)
        return self.model

    def pred(self, input):
        input = torch.Tensor(input).float().reshape(1, -1).to(self.device)
        output = self.model(input)
        return output.reshape(1, self.predict_length, self.output_size)

class MLP(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        hidden_dims,
        dropout=0,
        l1_lambda=0,
        l2_lambda=0,
        use_batchnorm=False,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dims = hidden_dims
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.use_batchnorm = use_batchnorm
        self.dropout = dropout
        if self.hidden_dims != []:
            # Define input layer
            input_layers = [
                nn.Linear(self.in_dim, self.hidden_dims[0]),
                nn.ReLU(),
                nn.Dropout(p=self.dropout),
            ]
            self.input_layer = nn.Sequential(*input_layers)

            # Define hidden layers
            hidden_layers = []
            for i in range(1, len(self.hidden_dims)):
                hidden_layers.extend(
                    [
                        nn.Linear(self.hidden_dims[i - 1], self.hidden_dims[i]),
                        nn.ReLU(),
                        nn.Dropout(p=self.dropout),
                    ]
                )
            self.hidden = nn.Sequential(*hidden_layers)

            # Define output layer
            self.output_layer = nn.Linear(self.hidden_dims[-1], self.out_dim)
        else:  # hidden = []
            layers = [
                nn.Linear(self.in_dim, self.out_dim),
                nn.ReLU(),
                nn.Dropout(p=self.dropout),
            ]
            self.layer = nn.Sequential(*layers)

    def forward(self, x):
        if self.hidden_dims != []:
            # Pass input through input layer
            x = self.input_layer(x)

            # Pass input through hidden layers
            x = self.hidden(x)

            # Pass input through output layer
            x = self.output_layer(x)
        else:
            x = self.layer(x)

        return x

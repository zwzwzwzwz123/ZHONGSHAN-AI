import torch
import torch.nn as nn

from .model_basic import Basic_model


class LSTMModel(Basic_model):
    def __init__(self, args=None,device='cpu'):
        super(LSTMModel, self).__init__()
        self.args = args

        self.input_size = args["input_size"]
        self.hidden_size = args["hidden_size"]
        self.num_layers = args["num_layers"]
        self.output_size = args["output_size"]
        self.epochs = args["epochs"]
        self.lr = args["learning_rate"]
        self.predict_length = args["predict_length"]
        # batch_first=True表示输入数据的形状是(batch_size, sequence_length, input_size)
        # 而不是默认的(sequence_length, batch_size, input_size)。
        # batch_size是指每个训练批次中包含的样本数量, sequence_length是指输入序列的长度
        self.device=device
        self.model = LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            output_size=self.output_size,
            predict_length=self.predict_length,
        ).to(self.device)

    def pretrain_and_save(self, trainDl, model_path):
        self.model.train()
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        epochs = self.epochs
        for i in range(epochs):
            for train_x, train_y in trainDl:
                train_x = train_x.float().to(self.device)
                train_y = train_y.float().to(self.device)
                optimizer.zero_grad()
                y_pred = self.model(train_x)
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
        input = torch.Tensor(input).float().unsqueeze(0).to(self.device)
        return self.model(input).squeeze(0)


class LSTM(nn.Module):
    def __init__(
        self, input_size, hidden_size, num_layers, output_size, predict_length
    ):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.predict_length = predict_length

        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # Fully connected layer for mapping hidden states to the output
        self.fc_inner = nn.Linear(hidden_size, input_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden state and cell state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate through LSTM using the input sequence
        out, _ = self.lstm(x, (h0, c0))

        # Take the hidden state from the last time step
        last_hidden = out[:, -1, :]  # Shape: (batch_size, hidden_size)

        # Initialize the output sequence tensor
        outputs = []

        # Generate each time step of the prediction sequence
        for _ in range(self.predict_length):
            # Pass the last hidden state through the fully connected layer
            input = self.fc_inner(last_hidden)
            output = self.fc(last_hidden)
            outputs.append(output)

            # Update last hidden state and cell state using LSTM (reshape to add batch and seq dimensions)
            last_hidden, (h0, c0) = self.lstm(input.unsqueeze(1), (h0, c0))
            last_hidden = last_hidden.squeeze(1)  # Remove seq dimension

        # Stack the outputs to form the final output sequence
        outputs = torch.stack(
            outputs, dim=1
        )  # Shape: (batch_size, predict_length, output_size)

        return outputs

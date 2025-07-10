import torch
import torch.nn as nn

from .model_basic import Basic_model


class TransformerModel(Basic_model):
    def __init__(self, args=None,device='cpu'):
        super(TransformerModel, self).__init__()
        self.args = args

        self.input_size = args["input_size"]
        self.output_size = args["output_size"]

        self.hidden_size = args["hidden_size"]
        self.num_layers = args["num_layers"]
        self.epochs = args["epochs"]
        self.lr = args["learning_rate"]
        self.predict_length = args["predict_length"]
        self.num_heads = args["num_heads"]
        self.dropout = args["dropout"]
        self.device=device
        self.model = Transformer(
            input_dim=self.input_size,
            output_dim=self.output_size,
            pred_length=self.predict_length,
            num_heads=self.num_heads,
            num_encoder_layers=self.num_layers,
            num_decoder_layers=self.num_layers,
            dim_feedforward=self.hidden_size,
            dropout=self.dropout,
            device=self.device
        )

    def pretrain_and_save(self, trainDl, model_path):
        self.model.train()
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        for _ in range(self.epochs):
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


class Transformer(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        pred_length,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        dim_feedforward,
        dropout,
        device
    ):
        super(Transformer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.pred_len = pred_length
        self.num_heads = num_heads
        self.device=device
        self.encoder = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        ).to(self.device)
        self.decoder = nn.TransformerDecoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        ).to(self.device)
        self.encoder_seq = nn.TransformerEncoder(self.encoder, num_encoder_layers).to(self.device)
        self.decoder_seq = nn.TransformerDecoder(self.decoder, num_decoder_layers).to(self.device)
        self.positional_encoding = self._generate_positional_encoding(input_dim).to(self.device)
        self.fc_in = nn.Linear(input_dim, input_dim).to(self.device)
        self.fc_out = nn.Linear(input_dim, output_dim).to(self.device)

    def forward(self, x):
        # x -> [seq_len, batch_size, input_dim]
        x = x.permute(1, 0, 2)
        src = x + self.positional_encoding[: x.size(0), :]
        mask = self._generate_square_subsequent_mask(src.size(0))
        output = self.fc_in(src)
        enc_output = self.encoder_seq(output, mask)
        dec_output = self.decoder_seq(enc_output, enc_output, mask)
        return self.fc_out(dec_output[: self.pred_len, :, :]).permute(1, 0, 2)

    def _generate_positional_encoding(self, dim, max_len=5000):
        """
        生成位置编码
        """
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        return pe

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

import torch
from torch import nn
import numpy as np

from Prediction.predictor import TrajectoryPredictor
from Prediction.detector import xywh_to_centroid, xywh_to_xyxy
import math


class REDPredictor(TrajectoryPredictor):
    def __init__(self, model_path, history_len, forecast_horizon, **kwargs):
        super().__init__(forecast_horizon)
        self.history_len = history_len

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = LSTMdense(4, 2, forecast_horizon, **kwargs).double().to(self.device)
        self.net.load_state_dict(torch.load(model_path))
        self.net.eval()

    def init_trajectory(self, detection):
        pass

    def update_and_predict(self, history):
        """
        Receive an updated bbox history and generate and return a forecast
        trajectory.
        """
        if history.shape[0] >= self.history_len:
            with torch.no_grad():
                bboxes = history[-self.history_len :]
                centroids = xywh_to_centroid(bboxes)
                inp = torch.from_numpy(centroids.astype(np.double)).to(self.device)
                torch_bboxes = torch.from_numpy(bboxes[:, 2:].astype(np.double)).to(
                    self.device
                )
                inp = torch.cat([inp, torch_bboxes], dim=1)
                inp = inp.unsqueeze(0).double()
                forecast = self.net(inp)
                return forecast.squeeze().cpu().numpy()
        else:
            return None


class Seq2SeqPredictor(TrajectoryPredictor):
    def __init__(self, model, model_path, history_len, forecast_horizon):
        super().__init__(forecast_horizon)
        self.history_len = history_len

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).double()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def init_trajectory(self, detection):
        pass

    def update_and_predict(self, history):
        """
        Receive an updated bbox history and generate and return a forecast
        trajectory.
        """
        if history.shape[0] >= self.history_len:
            with torch.no_grad():
                bboxes = history[-self.history_len :]
                centroids = xywh_to_centroid(bboxes)
                inp = torch.from_numpy(centroids).to(self.device)

                torch_bboxes = torch.from_numpy(bboxes[:, 2:].astype(np.double)).to(
                    self.device
                )
                inp = torch.cat([inp, torch_bboxes], dim=1)

                inp = inp.unsqueeze(0).double()
                forecast = self.model(inp)
                return forecast.squeeze().cpu().numpy()
        else:
            return None


class LSTMdense(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        output_seq_size,
        embedding_size=16,
        hidden_size=256,
        LSTM_layers=2,
        dropout=0.0,
    ):
        super(LSTMdense, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.LSTM_layers = LSTM_layers
        self.output_size = output_size
        self.output_seq_size = output_seq_size
        self.output_dim = output_size * output_seq_size

        if embedding_size is not None:
            self.embedding_encoder = nn.Linear(
                in_features=input_size, out_features=embedding_size
            )
        else:
            embedding_size = input_size
            self.embedding_encoder = None

        self.LSTM = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=LSTM_layers,
            dropout=dropout,
        )

        self.out_dense = nn.Linear(
            in_features=hidden_size, out_features=self.output_dim
        )

    def forward(self, input_seq):
        offset = (
            input_seq[:, -1, :2].repeat(self.output_seq_size, 1, 1).transpose(0, 1)
        )  # TODO: use batch_first in lstm
        diffs = input_seq[:, 1:, :2] - input_seq[:, :-1, :2]

        inp = torch.cat([diffs, input_seq[:, 1:, 2:]], dim=2)
        inp = inp.transpose(0, 1)
        if self.embedding_encoder is not None:
            inp = self.embedding_encoder(inp)

        # ignore output (0) and cell (1,1)
        _, (h_out, _) = self.LSTM(inp)

        output = self.out_dense(h_out[-1])  # take hidden state of last layer
        output_mat = output.view(-1, self.output_seq_size, self.output_size)

        return offset + output_mat


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class GRUEncDec(nn.Module):
    def __init__(
        self,
        input_size=2,
        output_size=2,
        output_seq_size=20,
        hidden_size=64,
        GRU_layers=1,
        dropout=0.0,
        tie_enc_dec=False,
    ):
        super(GRUEncDec, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.input_size = input_size
        self.output_size = output_size
        self.output_seq_size = output_seq_size
        self.hidden_size = hidden_size
        self.tie_enc_dec = tie_enc_dec

        self.dropout_layer = torch.nn.Dropout(dropout)

        self.encoderGRU = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=GRU_layers,
            batch_first=True,
        )

        if not tie_enc_dec:
            self.decoderGRU = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=GRU_layers,
                batch_first=True,
            )
        else:
            self.decoderGRU = self.encoderGRU

        self.linear = nn.Linear(in_features=hidden_size, out_features=output_size)

    def forward(self, input_seq):
        offset = input_seq[:, -1, :2].repeat(self.output_seq_size, 1, 1).transpose(0, 1)
        diffs = input_seq[:, 1:, :2] - input_seq[:, :-1, :2]

        diffs = self.dropout_layer(diffs)

        _, hn = self.encoderGRU(diffs)
        out_list = []

        # prev_x = input_seq[:, -1]
        prev_x = diffs[:, -1]

        for i in range(self.output_seq_size):
            _, hn = self.decoderGRU(prev_x.unsqueeze(1), hn)
            lin = self.linear(hn[-1])
            x = lin + prev_x
            out_list.append(x.unsqueeze(1))
            prev_x = x

        out = torch.cat(out_list, dim=1)
        # return out
        return out + offset


class GRUPositionEncDec(nn.Module):
    def __init__(
        self,
        output_size=2,
        output_seq_size=20,
        hidden_size=64,
        embedding_size=8,
        GRU_layers=1,
        dropout=0.0,
        tie_enc_dec=False,
    ):
        super(GRUPositionEncDec, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.output_size = output_size
        self.output_seq_size = output_seq_size
        self.hidden_size = hidden_size
        self.tie_enc_dec = tie_enc_dec

        self.vel_embed = torch.nn.Linear(in_features=2, out_features=embedding_size)
        # self.pos_embed = torch.nn.Linear(in_features=2, out_features=embedding_size)
        self.pos_embed = PositionalEncoding(embedding_size)

        # self.dropout_layer = torch.nn.Dropout(dropout)

        self.encoderGRU = nn.GRU(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=GRU_layers,
            batch_first=True,
        )

        if not tie_enc_dec:
            self.decoderGRU = nn.GRU(
                input_size=embedding_size,
                hidden_size=hidden_size,
                num_layers=GRU_layers,
                batch_first=True,
            )
        else:
            self.decoderGRU = self.encoderGRU

        self.linear = nn.Linear(in_features=hidden_size, out_features=output_size)

    def forward(self, input_seq):
        offset = input_seq[:, -1, :2].repeat(self.output_seq_size, 1, 1).transpose(0, 1)
        diffs = input_seq[:, 1:, :2] - input_seq[:, :-1, :2]

        vel_embedding = self.vel_embed(diffs)
        # pos_embedding = self.pos_embed(input_seq[:, :-1])
        embedding = self.pos_embed(vel_embedding)

        _, hn = self.encoderGRU(embedding)
        out_list = []

        prev_x = diffs[:, -1]

        for i in range(self.output_seq_size):
            prev_x_embedding = self.vel_embed(prev_x)
            _, hn = self.decoderGRU(prev_x_embedding.unsqueeze(1), hn)
            lin = self.linear(hn[-1])
            x = lin + prev_x
            out_list.append(x.unsqueeze(1))
            prev_x = x

        out = torch.cat(out_list, dim=1)
        # return out
        return out + offset


class VelLinear(nn.Module):
    def __init__(
        self,
        input_size=2,
        output_size=2,
        input_seq_size=20,
        output_seq_size=20,
        hidden_size=64,
        dropout=0.0,
    ):
        super(VelLinear, self).__init__()

        self.output_size = output_size
        self.output_seq_size = output_seq_size
        self.input_seq_size = output_seq_size

        self.dropout_layer = torch.nn.Dropout(dropout)
        self.encoder = torch.nn.Linear(
            in_features=input_size * (input_seq_size - 1), out_features=hidden_size
        )
        self.decoder = torch.nn.Linear(
            in_features=hidden_size, out_features=output_size * output_seq_size
        )

    def forward(self, input_seq):
        offset = input_seq[:, -1, :2].repeat(self.output_seq_size, 1, 1).transpose(0, 1)
        diffs = input_seq[:, 1:] - input_seq[:, :-1]

        x = self.dropout_layer(diffs)
        x = self.encoder(x.view(x.shape[0], -1))
        x = torch.nn.functional.relu(x)
        x = self.decoder(x)

        out = x.view(x.shape[0], self.input_seq_size, self.output_size)

        return out + offset

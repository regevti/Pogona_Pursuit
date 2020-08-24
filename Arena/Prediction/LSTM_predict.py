import torch
from torch import nn
import numpy as np

from Prediction.predictor import TrajectoryPredictor
from Prediction.detector import xywh_to_centroid


class REDPredictor(TrajectoryPredictor):
    def __init__(self, model_path, history_len, forecast_horizon, **kwargs):
        super().__init__(forecast_horizon)
        self.history_len = history_len

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = LSTMdense(2, 2, forecast_horizon, **kwargs).double().to(self.device)
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
                centroids = xywh_to_centroid(history[-self.history_len :])
                inp = torch.from_numpy(centroids.astype(np.double)).to(self.device)
                inp = inp.unsqueeze(0).double()
                forecast = self.net(inp)
                return forecast.squeeze().cpu().numpy()
        else:
            return None


class LSTMdense(nn.Module):
    def __init__(
        self, input_size, output_size, output_seq_size, hidden_size=256, LSTM_layers=2
    ):
        super(LSTMdense, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.LSTM_layers = LSTM_layers
        self.output_size = output_size
        self.output_seq_size = output_seq_size
        self.output_dim = output_size * output_seq_size

        self.LSTM = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, num_layers=LSTM_layers
        )

        self.dense = nn.Linear(in_features=hidden_size, out_features=self.output_dim)

    def forward(self, input_seq):
        offset = input_seq[:, -1].repeat(self.output_seq_size, 1, 1).transpose(0, 1)
        diffs = input_seq[:, 1:, :] - input_seq[:, :-1, :]

        # ignore output (0) and cell (1,1)
        _, (h_out, _) = self.LSTM(diffs.transpose(0, 1))
        # _, (h_out, _) = self.LSTM(input_seq.transpose(0, 1))

        output = self.dense(h_out[1])  # take hidden state of second layer
        return offset + output.view(-1, self.output_seq_size, self.output_size)

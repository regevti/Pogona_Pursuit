"""
This module implements a trjaectory predictor based on sequence-to-sequence
neural network models.

Seq2SeqPredictor - A TrajectoryPredictor subclass that uses a pytorch model for prediction.
TrajEncDec - A pytorch model that uses an encoder-decoder seq2seq architechture to predict trjaectories based on past trajectories.
"""

import torch
from torch import nn
from Prediction.predictor import TrajectoryPredictor


class Seq2SeqPredictor(TrajectoryPredictor):
    """
    A TrajectoryPredictor subclass that uses a pytorch model to generate trajectory forecasts.
    """

    def __init__(self, model, weights_path, input_len, forecast_horizon):
        """
        The model is sent to the available device, and weights are loaded from the file.

        :param model: An initialized PyTorch seq2seq model (torch.nn.Module subclass).
        :param weights_path: Path to a pickled state dictionary containing trained weights for the model.
        :param input_len: Number of timesteps to look back in order to generate a forecast.
        :param forecast_horizon: The forecast size in timesteps into the future.
        """
        super().__init__(input_len, forecast_horizon)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).float()
        self.model.load_state_dict(torch.load(weights_path))
        self.model.eval()

    def init_trajectory(self, detection):
        """
        No need to initialize trjaectory as this model is only concerned with the past input_len detections.
        """
        pass

    def _update_and_predict(self, past_input):
        """
        Receive an updated bbox trajectory of the last input_len time steps, and generate and return a forecast
        trajectory by passing it as input to the model.
        """
        with torch.no_grad():
            inp = torch.from_numpy(past_input).to(self.device)
            inp = inp.unsqueeze(0).float()  # Adds an additional dimension for a batch size of 1.
            forecast = self.model(inp)
            return forecast.squeeze().cpu().numpy()


class TrajEncDec(nn.Module):
    """
    Pytorch module implementing a sequence to sequence decoder-encoder architecture for bounding box trajectories.
    The module expects a bounding box trajectory as input.

    The input trajectory is encoded as a tensor with shape (batch_size, trajectory length, 4) where trajectory
    length can be any size and is not constrained by the architecture.

    The output of the module is a tensor with shape (batch_size, output_seq_size, 4) representing a predicted
    trajectory of bounding boxes.

    See the init function for details about customizing the module architecture.
    """

    def __init__(
            self,
            output_seq_size=20,
            hidden_size=64,
            rnn_layers=1,
            dropout=0.0,
            decoder_type='RNN',
            rnn_type='GRU',
            use_abs_pos=False,
            tie_enc_dec=False,
            use_rnn_cell=False,
    ):
        """
        Initialize the module.

        The init parameters allow for customizing the encoder-decoder architecture.
        The RNN variant for the encoder and decoder can be set using the rnn_type parameter, and can be either
        an LSTM or GRU network. The decoder can be either an RNN of the same type as the encoder or
        a simple (and much faster) linear layer by changing the decoder_type parameter.
        When both decoder and encoder are RNNs, it's possible to share weights between them by setting tie_enc_dec to
        True. This however requires that both encoder and decoder are full RNN layers, while using a single RNN cell
        for the decoder is faster. This can be set using the use_rnn_cell parameter. Additionally, in this case the
        decoder will ignore the rnn_layers parameter, and will consist of a single layer.

        Even though the input to the module is a trajectory of absolute bounding box position, the module normally
        ignores the absolute position of the input trajectory and uses only the velocity vectors of the trajectory
        as input features. When setting use_abs_pos to True, the module will also use the absolute positions as
        input features.

        There are two dropout layers, the first is on the input vector of the encoder, and the second is used on
        the output of the RNN decoder, before running through a linear layer that converts the output to bbox
        coordinates.

        The module also supports schedule sampling based training (see Bengio et al., Scheduled Sampling for Sequence
        Prediction with Recurrent Neural Networks, 2015). To use scheduled sampling set the sampling_eps attribute
        to the desired probability for sampling from the target batch, and set the target attribute to the prediction
        target tensor. These attributes should be updated before running each training batch.

        :param output_seq_size: forecast trajectory length
        :param hidden_size: dimension of hidden state in RNNs
        :param rnn_layers: number of RNN layers in encoder and decoder
        :param dropout: dropout probablity in dropout layers
        :param decoder_type: Architecture of decoder. 'RNN' or 'Linear'.
        :param rnn_type: RNN architecture. 'GRU' or 'LSTM'.
        :param use_abs_pos: Boolean, whether to use absolute position as an additional encoder feature.
        :param tie_enc_dec: Boolean, whether to use the same parameters in the encoder and decoder.
        :param use_rnn_cell: Boolean, whether to use a pytorch RNN cell or an RNN layer for the decoder.
        """
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.output_seq_size = output_seq_size
        self.hidden_size = hidden_size
        self.use_rnn_cell = use_rnn_cell
        self.use_abs_pos = use_abs_pos
        self.decoder_type = decoder_type
        self.rnn_type = rnn_type

        self.sampling_eps = 0  # scheduled sampling probability for training
        self.target = None

        self.dropout_layer = torch.nn.Dropout(dropout)

        if use_abs_pos:
            input_size = 8
        else:
            input_size = 4

        # initialize encoder
        if rnn_type == 'GRU':
            self.encoder = nn.GRU(input_size=input_size,
                                  hidden_size=hidden_size,
                                  num_layers=rnn_layers,
                                  batch_first=True,
                                  )
        else:
            self.encoder = nn.LSTM(input_size=input_size,
                                   hidden_size=hidden_size,
                                   num_layers=rnn_layers,
                                   batch_first=True,
                                   )

        # initialize decoder
        if decoder_type == 'RNN':
            if not tie_enc_dec:
                if use_rnn_cell:
                    if rnn_type == 'GRU':
                        self.decoder = nn.GRUCell(input_size=input_size,
                                                  hidden_size=hidden_size,
                                                  )
                    else:
                        self.decoder = nn.LSTMCell(input_size=input_size,
                                                   hidden_size=hidden_size,
                                                   )
                else:
                    if rnn_type == 'GRU':
                        self.decoder = nn.GRU(input_size=input_size,
                                              hidden_size=hidden_size,
                                              num_layers=rnn_layers,
                                              batch_first=True,
                                              )
                    else:
                        self.decoder = nn.LSTM(input_size=input_size,
                                               hidden_size=hidden_size,
                                               num_layers=rnn_layers,
                                               batch_first=True,
                                               )

            else:
                self.decoder = self.encoder
            self.linear = nn.Linear(in_features=hidden_size, out_features=4)
        else:
            # linear decoder - maps hidden directly to velocities
            self.decoder = nn.Linear(in_features=hidden_size, out_features=4 * self.output_seq_size)

    def decoder_rnn(self, hn, cn, input_seq, input_vels):
        """
        Forward function for an RNN based decoder.

        :param hn: last hidden state
        :param cn: last cell state (LSTM only)
        :param input_seq: module input sequence (used for batch size and last position)
        :param input_vels: first difference of the input_seq (used for last input velocity)
        """
        out_list = []

        if self.use_rnn_cell:
            # reduce hidden state dimension to pass into RNN single cell.
            hn = hn[0]
            if self.rnn_type == "LSTM":
                cn = cn[0]

        if self.sampling_eps:
            # generate the scheduled sampling target encoding.
            pos_target = self.target[:, :-1]
            vel_target = self.target[:, 1:] - self.target[:, :-1]
            if self.use_abs_pos:
                target_en = torch.cat((pos_target, vel_target), dim=-1)
            else:
                target_en = vel_target

        # start iterating from the last position and velocity in the input.
        x = input_seq[:, -1]
        vel = input_vels[:, -1]

        # iterate over the encoder to generate the output sequence.
        for i in range(self.output_seq_size):

            vel = self.dropout_layer(vel)

            if self.sampling_eps and i > 0:
                # scheduled sampling. select random time steps to sample from the target sequence.
                coins = torch.rand(input_seq.shape[0])
                take_true = (coins < self.sampling_eps).unsqueeze(1).to(self.device)
                truths = take_true * target_en[:, i - 1]
                if self.use_abs_pos:
                    x = truths[:, :4] + (~take_true) * x
                    vel = truths[:, -4:] + (~take_true) * vel
                else:
                    vel = truths + (~take_true) * vel

            if self.use_abs_pos:
                decoder_input = torch.cat((x, vel), dim=-1)
            else:
                decoder_input = vel

            if self.use_rnn_cell:
                if self.rnn_type == "GRU":
                    hn = self.decoder(decoder_input, hn)
                else:

                    hn, cn = self.decoder(decoder_input, (hn, cn))
                vel = self.linear(hn)
            else:
                if self.rnn_type == "GRU":
                    _, hn = self.decoder(decoder_input.unsqueeze(1), hn)
                else:
                    _, (hn, cn) = self.decoder(decoder_input.unsqueeze(1), (hn, cn))

                vel = self.linear(hn[-1])

            if self.use_abs_pos:
                x = vel + x

            out_list.append(vel.unsqueeze(1))

        # concatenate the output sequence and return an output sequence of velocity vectors.
        out = torch.cat(out_list, dim=1)
        return out

    def decoder_linear(self, hn):
        """
        Forward function for a linear layer decoder. Use a single linear layer to transform
        the encoder's hidden state into a sequence of positions relative to the last position
        of the input trajectory sequence.

        :param hn: the encoder's hidden state vector
        """
        # take hidden state of last encoder layer
        h_out = hn[-1]

        # run through linear layer
        output = self.decoder(h_out)

        # reshape the output into a sequence of bbox coordinates.
        return output.view(-1, self.output_seq_size, 4)

    def forward(self, input_seq):
        """
        Main forward function for the encoder-decoder module.

        :param input_seq: An input trajectory sequence tensor of size (batch size, sequence length, 4)
        """

        # switch from absolute positions to velocity vectors.
        offset = input_seq[:, -1][:, None, :]
        input_vels = input_seq[:, 1:] - input_seq[:, :-1]

        if self.use_abs_pos:
            # encode both positions and velocities features.
            input_en = torch.cat((input_seq[:, :-1], input_vels), dim=-1)
        else:
            # use only velocity vectors as input features.
            input_en = input_vels

        if self.decoder_type == "RNN":
            input_en = input_en[:, :-1]  # up until and including v_{m-2}, v_{m-1} is the decoder's first input
        input_en = self.dropout_layer(input_en)

        # encode input
        if self.rnn_type == "GRU":
            _, hn = self.encoder(input_en)
            cn = None
        else:
            _, (hn, cn) = self.encoder(input_en)

        # decode output
        if self.decoder_type == 'RNN':
            out = self.decoder_rnn(hn, cn, input_seq, input_vels)
        else:
            out = self.decoder_linear(hn)

        return out.cumsum(dim=1) + offset

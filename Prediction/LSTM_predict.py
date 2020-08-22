import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

import time

from Prediction.predictor import TrajectoryPredictor


class RedPredictor(TrajectoryPredictor):
    def __init__(self, forecast_horizon):
        super().__init__(forecast_horizon)
        pass

    def init_trajectory(self, detection):
        pass

    def update_and_predict(self, history):
        """
        Receive an updated bbox history and generate and return a forecast
        trajectory.
        """
        pass


class LSTMdense(nn.Module):
    def __init__(self, input_size, output_size, output_seq_size,
                 hidden_size=256, LSTM_layers=2):
        super(LSTMdense, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.LSTM_layers = LSTM_layers
        self.LSTM = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=LSTM_layers)
        self.output_size = output_size
        self.output_seq_size = output_seq_size
        self.output_dim = output_size * output_seq_size
        self.dense = nn.Linear(in_features=hidden_size,
                             out_features=self.output_dim)

    def forward(self, input_seq):
        _, (h_out, _) = self.LSTM(input_seq.transpose(0, 1))  # ignore output (0) and cell (1,1)
        #_, (h_out, _) = self.LSTM(input_seq)  # ignore output (0) and cell (1,1)

        output = self.dense(h_out[1])  # take hidden state of second layer
        return output


class TrajectoriesData(Dataset):
    def __init__(self, X, Y):
        assert X.shape[0] == Y.shape[0]

        self.X = X
        self.Y = Y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, item):
        return self.X[item], self.Y[item]


def create_dataloader(X, Y, train_test_ratio = 0.8, shuffle=True, batch_size=256, num_workers=0):
    dataset = TrajectoriesData(X, Y)
    train_size = round(train_test_ratio * len(dataset)) - 1
    test_size = len(dataset) - train_size
    train, test = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return train_loader, test_loader


def calc_ADE(y_hat, y):
    """
    Calculate the mean average displacement error (ADE) of a test set of predictions
    displacement is the L2 norm between the prediction and the truth
    :param y_hat: torch tensor dims (number of sample, out seq size, out size)
    :param y: same
    :return: mean ADE over batch
    """
    if len(y_hat.shape) == 2:
        y_hat = y_hat.unsqueeze(0)
        y = y.unsqueeze(0)

    return torch.mean(torch.norm(y_hat - y, dim=2), dim=1).mean()


def train_RED(model, dataloader_train, dataloader_test, epochs, path, eval_freq=25):
    optimizer = Adam(model.parameters(),amsgrad=True)  # TODO maybe use other optimizer params
    criterion = nn.MSELoss()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)

    for epoch in range(epochs):
        model.train()

        epoch_loss = 0.0
        epoch_start = time.time()
        for i, (x, y) in enumerate(dataloader_train):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            flattened_y = y.view(y.shape[0], y.shape[1] * y.shape[2], -1).squeeze()
            """
            print(f'x shape: {x.shape}, y shape {y.shape}')
            print(f'pred shape: {output.shape}')
            print(f'flat y shape: {flattened_y.shape}')
            """

            loss = criterion(output, flattened_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        epoch_time = time.time() - epoch_start
        if epoch % 20 == 0:
            print(f'epoch: {epoch}, loss: {epoch_loss:.3f}, epoch time: {epoch_time:.3f}')

        if epoch % eval_freq == 0:
            model.eval()
            total_ADE = 0
            count = 0
            for (x, y) in dataloader_test:
                x, y = x.to(device), y.to(device)
                with torch.no_grad():
                    pred = model(x)
                pred = pred.view(-1, model.output_seq_size, model.output_size)
                total_ADE += calc_ADE(pred, y)
                count+=1
            print(f'++++++++++++++++')
            print(f'Eval epoch: {epoch}, Test set mean ADE {total_ADE/count:.3f}')
            torch.save(model.state_dict(), path + f'/RED_model_{epoch}.pth')

    torch.save(model.state_dict(), path + '/RED_model_final.pth')
    print(f'Finished training')

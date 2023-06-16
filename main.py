import torch
import numpy as np
from numpy.fft import fft, ifft
import fft_nn_settings
from fft_nn_model import fft_model
import utils
import torch.nn as nn

if __name__ == '__main__':
    # FFT reference: https://pythonnumericalmethods.berkeley.edu/notebooks/chapter24.04-FFT-in-Python.html
    # https://en.wikipedia.org/wiki/DFT_matrix

    print('Running FFT learning')

    DEVICE = utils.get_device()

    model = fft_model(fft_nn_settings.FFT_SIZE).to(DEVICE)
    optimz = torch.optim.Adam(model.parameters(), lr=fft_nn_settings.LEARNING_RATE)
    criterion = nn.MSELoss()
    for epoch in range(fft_nn_settings.EPOCHS):

        print(f'epoch = {epoch}')

        # Create the dataset
        x = np.random.rand(fft_nn_settings.BATCH_SIZE, fft_nn_settings.FFT_SIZE)
        X = fft(x)

        y_list = []
        for batch in range(fft_nn_settings.BATCH_SIZE):
            y = []
            for i in range(fft_nn_settings.FFT_SIZE):
                y.append(X[batch, i].real)
                y.append(X[batch, i].imag)

            y_list.append(y)

        ground_truth = np.array(y_list)

        x = torch.tensor(x, dtype=torch.float32).to(DEVICE)
        X = torch.tensor(ground_truth, dtype=torch.float32).to(DEVICE)

        # Train Model ---------------------------
        model.train()

        optimz.zero_grad()

        preds = model(x)
        loss = criterion(preds, X)

        print(f'loss = {loss.item()}')

        loss.backward()
        optimz.step()


    print(model.model[0].weight)


    # Create the complex matrix
    weight_matrix = model.model[0].weight.detach().to('cpu').numpy()

    fft_matrix = []
    for i in range(0, weight_matrix.shape[0], 2):
        row = []
        for j in range(weight_matrix.shape[1]):
            row.append( complex(weight_matrix[i,j], weight_matrix[i+1,j]) )

        fft_matrix.append(row)

    fft_matrix = np.array(fft_matrix)

    print(f'fft matrix shape = {fft_matrix.shape}')
    print(fft_matrix)

    '''
    Sample output
    [[ 9.99996841e-01+1.12286046e-40j  1.00000119e+00-2.52808256e-41j   1.00000131e+00-9.40453438e-41j   1.00000119e+00-2.42816998e-41j]
     [ 9.99999940e-01-5.83203850e-08j  -4.07864569e-08-9.99999762e-01j  -9.99999821e-01-4.29513491e-08j  -4.63540708e-08+9.99999881e-01j]
     [ 1.00000000e+00-1.50667611e-41j  -1.00000024e+00-5.68520800e-41j  9.99999940e-01+9.89316716e-42j   -9.99999762e-01+3.21457868e-41j]
     [ 9.99999881e-01-3.07347960e-08j  6.93777327e-11+9.99999762e-01j   -9.99999881e-01-3.13096855e-08j  -1.51074808e-09-9.99999702e-01j]]
    '''




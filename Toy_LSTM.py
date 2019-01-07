# imports
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def temporal_softmax_loss(x, y, mask, verbose=False):
    """
    A temporal version of softmax loss for use in RNNs. We assume that we are
    making predictions over a vocabulary of size V for each timestep of a
    timeseries of length T, over a minibatch of size N. The input x gives scores
    for all vocabulary elements at all timesteps, and y gives the indices of the
    ground-truth element at each timestep. We use a cross-entropy loss at each
    timestep, summing the loss over all timesteps and averaging across the
    minibatch.

    As an additional complication, we may want to ignore the model output at some
    timesteps, since sequences of different length may have been combined into a
    minibatch and padded with NULL tokens. The optional mask argument tells us
    which elements should contribute to the loss.

    Inputs:
    - x: Input scores, of shape (N, T, V)
    - y: Ground-truth indices, of shape (N, T) where each element is in the range
         0 <= y[i, t] < V
    - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
      the scores at x[i, t] should contribute to the loss.

    Returns a tuple of:
    - loss: Scalar giving loss
    - dx: Gradient of loss with respect to scores x.
    """

    N, T, V = x.shape

    x_flat = x.reshape(N * T, V).detach().numpy()
    y_flat = y.reshape(N * T).detach().numpy()
    mask_flat = mask.reshape(N * T).detach().numpy()



    probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N
    dx_flat = probs.copy()
    dx_flat[np.arange(N * T), y_flat] -= 1
    dx_flat /= N
    dx_flat *= mask_flat[:, None]

    if verbose: print('dx_flat: ', dx_flat.shape)

    dx = dx_flat.reshape(N, T, V)

    return loss, dx


class Model_toy_lstm(nn.Module):

    def __init__(self, embed_size, hidden_size, word_2_idx, num_layers, max_seq_length=20):
        """Set the hyper-parameters and build the layers."""
        super(Model_toy_lstm, self).__init__()

        self.num_layers = num_layers  # 1
        self.vocab_size = len(word_2_idx)  # V
        self.word_2_idx = word_2_idx
        self.hidden_size = hidden_size
        self.max_seq_length = max_seq_length
        self.embed_size = embed_size

        self.word_embed = nn.Embedding(self.vocab_size, self.embed_size)  # V, D
        self.lstm = nn.LSTM(self.embed_size, self.hidden_size, self.num_layers, batch_first=True)  # D, H, 1
        # output layer which projects back to tag space
        self.hidden_to_tag = nn.Linear(self.hidden_size, self.vocab_size)

        # some useful things
        self._null = self.word_2_idx.get('<NULL>', None)
        self._start = self.word_2_idx.get('<START>', None)
        self._end = self.word_2_idx.get('<END>', None)


    def initialize_lstm(self, X):
        batch_size, D = X.shape
        # the weights are of the form (nb_layers, batch_size, nb_lstm_units)
        hidden_h = torch.randn(self.num_layers, batch_size, self.hidden_size).to(device)
        hidden_c = torch.randn(self.num_layers, batch_size, self.hidden_size).to(device)
        return hidden_h, hidden_c

    def forward(self, captions):
        # captions shape is 10 x 17, where B = 10 is the batch size, and 17 is the feature size
        B, D = captions.shape


        lstm_h, lstm_c = self.initialize_lstm(captions)
        embedded_captions = self.word_embed(captions)  # output B x E where E is the embedding size

        X, (lstm_h, lstm_c) = self.lstm(embedded_captions, (lstm_h, lstm_c))

        # Dim transformation: (batch_size, seq_len, nb_lstm_units) -> (batch_size * seq_len, nb_lstm_units)
        # this one is a bit tricky as well. First we need to reshape the data so it goes into the linear layer
        X = X.contiguous() #  it actually makes a copy of tensor so the order of elements would be same as if tensor of same shape created from scratch.
        X = X.view(-1, X.shape[2])
        X = self.hidden_to_tag(X)
        X = F.log_softmax(X, dim=1)

        Y_hat = X
        return Y_hat

    def loss(self, captions_out, captions_gt):
        """ gets the computed captions and ground truth ones"""
        B, L = captions_gt.shape
        captions_out = captions_out.view(B, L, self.vocab_size) #N, T, V
        mask = (captions_gt != self._null)

        loss, dscores = temporal_softmax_loss(captions_out, captions_gt, mask)
        return loss






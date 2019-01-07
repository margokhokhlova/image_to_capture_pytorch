import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


class Model_text_lstm(nn.Module):

    def __init__(self, embed_size, hidden_size, word_2_idx, num_layers, max_seq_length=20):
        """Set the hyper-parameters and build the layers."""
        super(Model_text_lstm, self).__init__()

        self.num_layers = num_layers  # 1
        self.nb_vocab_words = len(word_2_idx)  # V
        self.word_2_idx = word_2_idx
        self.hidden_size = hidden_size
        self.max_seq_length = max_seq_length
        self.embedding_dim = embed_size


        # some useful things
        self._null = self.word_2_idx.get('<NULL>', None)
        self._start = self.word_2_idx.get('<START>', None)
        self._end = self.word_2_idx.get('<END>', None)

        #layers
        self.word_embedding = nn.Embedding(
            num_embeddings=self.nb_vocab_words,
            embedding_dim=self.embedding_dim,
            padding_idx=self._null)

        # design LSTM
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )

        # output layer which projects back to tag space
        self.hidden_to_vocab = nn.Linear(self.hidden_size, self.nb_vocab_words)


    def init_hidden(self, X):
        batch_size, D = X.shape
        # the weights are of the form (nb_layers, batch_size, nb_lstm_units)
        hidden_h = torch.randn(self.num_layers, batch_size, self.hidden_size).to(device)
        hidden_c = torch.randn(self.num_layers, batch_size, self.hidden_size).to(device)
        return (hidden_h, hidden_c)

    def forward(self, X):
        # reset the LSTM hidden state. Must be done before you run a new batch. Otherwise the LSTM will treat
        # a new batch as a continuation of a sequence
        self.hidden = self.init_hidden(X)

        batch_size, seq_len = X.shape
        #seq_len = X_lengths[0] # take the longest sentence

        # ---------------------
        # 1. embed the input
        # Dim transformation: (batch_size, seq_len, 1) -> (batch_size, seq_len, embedding_dim)
        X = self.word_embedding(X)

        # ---------------------
        # 2. Run through RNN
        # TRICK 2 ********************************
        # Dim transformation: (batch_size, seq_len, embedding_dim) -> (batch_size, seq_len, nb_lstm_units)

        # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
        #X = torch.nn.utils.rnn.pack_padded_sequence(X, X_lengths, batch_first=True)

        # now run through LSTM
        X, self.hidden = self.lstm(X, self.hidden)

        # undo the packing operation
        #X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True)

        # ---------------------
        # 3. Project back to vocab space
        # Dim transformation: (batch_size, seq_len, nb_lstm_units) -> (batch_size * seq_len, nb_lstm_units)

        # this one is a bit tricky as well. First we need to reshape the data so it goes into the linear layer
        X = X.contiguous()
        X = X.view(-1, X.shape[2])

        # run through actual linear layer
        X = self.hidden_to_vocab(X)

        # ---------------------
        # 4. Create softmax activations bc we're doing classification
        # Dim transformation: (batch_size * seq_len, nb_lstm_units) -> (batch_size, seq_len, nb_tags)
        X = F.log_softmax(X, dim=1)

        # I like to reshape for mental sanity so we're back to (batch_size, seq_len, words)
        X = X.view(batch_size, seq_len, self.nb_vocab_words)

        Y_hat = X
        return Y_hat


    def loss(self, Y_hat, Y): # , X_lengths maybe also this
        # TRICK 3 ********************************
        # before we calculate the negative log likelihood, we need to mask out the activations
        # this means we don't want to take into account padded items in the output vector
        # simplest way to think about this is to flatten ALL sequences into a REALLY long sequence
        # and calculate the loss on that.

        # flatten all the labels

        Y = Y.view(-1)

        # flatten all predictions
        Y_hat = Y_hat.view(-1, self.nb_vocab_words)

        # create a mask by filtering out all tokens that ARE NOT the padding token
        tag_pad_token = self._null
        mask = (Y > tag_pad_token).float()

        # count how many tokens we have
        nb_tokens = int(torch.sum(mask).data[0])

        # pick the values for the label and zero out the rest with the mask
        Y_hat = Y_hat[range(Y_hat.shape[0]), Y] * mask

        # compute cross entropy loss which ignores all <PAD> tokens
        ce_loss = -torch.sum(Y_hat) / nb_tokens

        return ce_loss




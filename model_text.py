import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from coco_utils import  sample_coco_minibatch
from bleu_score import evaluate_model




class Model_text_lstm(nn.Module):

    def __init__(self, embed_size, img_feat_size, hidden_size, word_2_idx, num_layers, max_seq_length=17, device = 'cpu'):
        """Set the hyper-parameters and build the layers."""
        super(Model_text_lstm, self).__init__()

        self.num_layers = num_layers  # 1
        self.nb_vocab_words = len(word_2_idx)  # V
        self.word_2_idx = word_2_idx
        self.hidden_size = hidden_size
        self.max_seq_length = max_seq_length
        self.embedding_dim = embed_size
        self.image_input_size = img_feat_size


        # some useful things
        self._null = self.word_2_idx.get('<NULL>', None)
        self._start = self.word_2_idx.get('<START>', None)
        self._end = self.word_2_idx.get('<END>', None)


        #layers
        self.image_embedding = nn.Linear(self.image_input_size, self.hidden_size) # image - hidden size dimension

        self.word_embedding = nn.Embedding(
            num_embeddings=self.nb_vocab_words,
            embedding_dim=self.embedding_dim)

        # design LSTM
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )

        # output layer which projects back to tag space
        self.hidden_to_vocab = nn.Linear(self.hidden_size, self.nb_vocab_words)
        self.device = device

    def init_hidden(self, X, image_feat = None):
        batch_size, D, _ = X.shape
        # the weights are of the form (nb_layers, batch_size, nb_lstm_units)
        hidden_h = torch.randn(self.num_layers, batch_size, self.hidden_size).to(self.device)
        hidden_c = torch.randn(self.num_layers, batch_size, self.hidden_size).to(self.device)

        if image_feat is not None:
            # then use the embedded image as a hidden layer
            hidden_h = image_feat.unsqueeze(0)#image_feat.view(1, -1, -1)
        return hidden_h, hidden_c

    def _reset(self):
        """
        Set up some book-keeping variables for optimization. Not sure I need it
        """
        # Set up some variables for book-keeping
        self.epoch = 0
        self.best_val_acc = 0
        self.best_params = {}
        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

    def temporal_softmax_loss(self, x, y, mask, verbose=False):
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

        Returns :
        - loss: Scalar giving loss
        """

        N, T, V = x.shape

        x_flat = x.reshape(N * T, V)
        y_flat = y.reshape(N * T)
        mask_flat = mask.reshape(N * T)
        maximums, _ = x_flat.max(1)
        probs = torch.exp(x_flat.transpose(-1,0) - maximums).transpose(-1,0) # softmax
        probs = torch.div(probs, torch.sum(probs, 0)) # softmax
        val = torch.log(probs[torch.arange(N * T), y_flat])
        loss = -torch.sum(mask_flat * val ) / N
        return loss

    def forward(self, features, X):

        batch_size, seq_len = X.shape
        #seq_len = X_lengths[0] # take the longest sentence

        # ---------------------
        # 1. embed the input
        # Dim transformation: (batch_size, seq_len, 1) -> (batch_size, seq_len, embedding_dim)
        X = self.word_embedding(X)

        # ---------------------
        # pass image features through the net
        imf2hid = self.image_embedding(features) # initial hidden state: [N, H]

        # reset the LSTM hidden state. Must be done before you run a new batch.
        self.hidden_h, self.hidden_c = self.init_hidden(X, imf2hid)

        # now run through LSTM
        X, (self.hidden_h, self.hidden_c) = self.lstm(X, (self.hidden_h, self.hidden_c))

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
        # 4. Create softmax activations
        # Dim transformation: (batch_size * seq_len, nb_lstm_units) -> (batch_size, seq_len, nb_tags)
        #X = F.log_softmax(X, dim=1)

        # I like to reshape for mental sanity so we're back to (batch_size, seq_len, words)
        X = X.view(batch_size, seq_len, self.nb_vocab_words)

        Y_hat = X

        return Y_hat


    def loss(self, Y_hat, Y): # , X_lengths maybe also this

        # #
        # # before we calculate the negative log likelihood, we need to mask out the activations
        # # this means we don't want to take into account padded items in the output vector
        # # flatten all the labels
        # Y = Y.view(-1)
        #
        # # flatten all predictions
        # #Y_hat = Y_hat.view(-1, self.nb_vocab_words)
        # #Y_hat = Y_hat.view(-1)
        # # create a mask by filtering out all tokens that ARE NOT the padding token
        tag_pad_token = self._null
        mask = (Y > tag_pad_token).float()

        # # count how many actual words is there
        # nb_tokens = int(torch.sum(mask).item())
        #
        # # pick the values for the label and zero out the rest with the mask
        # Y_hat = Y_hat[range(Y_hat.shape[0]), Y] * mask

        # compute cross entropy loss which ignores all <PAD> tokens
        ce_loss = self.temporal_softmax_loss(Y_hat, Y, mask)

        return ce_loss

    def sample(self, features):
        """ function which samples the captions from a pre-trained model
        features - image features """
        features = torch.from_numpy(features).to(self.device)  # make a tensor here

        N = features.shape[0] # Batch size
        captions = self._null * torch.zeros([N, self.max_seq_length], dtype=torch.int64).to(self.device) # prepare the output
        captions[:, 0] = self._start # start with start

        imf2hid = self.image_embedding(features)  # initial hidden state: [N, H]
        self.hidden_h = imf2hid.unsqueeze(0)#
        self.hidden_c = torch.zeros_like(self.hidden_h)

        iteration = 1
        while iteration < self.max_seq_length:
            # for all the words
            onehots = captions[:,iteration-1]
            word_vectors = self.word_embedding(onehots)
            inputs = word_vectors.unsqueeze(1)
            X, (self.hidden_h, self.hidden_c) = self.lstm(inputs, (self.hidden_h, self.hidden_c))

            X = X.contiguous() # get the last hidden layer
            X = X.view(-1, X.shape[2])

            # run through actual linear layer
            X = self.hidden_to_vocab(X)
            # Create softmax activations bc we're doing classification
            # Dim transformation: (batch_size * seq_len, nb_lstm_units) -> (batch_size, seq_len, nb_wirds)
            #X = F.log_softmax(X, dim=1) # for some reason, no softmax during sampling?
            _, x = X.max(1) # so predict here

            captions[:, iteration] = x
            iteration += 1
            #print(captions) #show the iteration change of caption vector


        return np.array(captions) # cast back to numpy


    def train(self, data, num_epochs, batch_size, optimizer):

        self._reset() # reset the model
        optimizer.zero_grad()  # where shall I make it?
        loss = 0.0

        print('Training...')
        num_train = data['train_captions'].shape[0]
        iterations_per_epoch = np.int(num_train // batch_size)
        num_iterations = num_epochs * iterations_per_epoch



        for i in range(num_iterations):
            minibatch = sample_coco_minibatch(data, batch_size=batch_size, split='train')
            captions, features, urls = minibatch
            captions = torch.LongTensor(captions).to(self.device)
            features = torch.from_numpy(features).to(self.device)
            Y_hat = self.forward(features, captions)
            loss = self.loss(Y_hat, captions)
            self.loss_history.append(loss)  # save loss
            loss.backward()
            optimizer.step()  # Updates all the weights of the network
            # At the end of every epoch, increment the epoch counter and decay the
            # learning rate.
            epoch_end = (i + 1) % iterations_per_epoch == 0
            if epoch_end:
                self.epoch += 1
                optimizer.zero_grad()  # where shall I make it? maybe for every batch? or for every epoch?

                if (self.epoch) % 1 == 0:
                    print('Epoch:  %d | Current Loss: %.4f' % (self.epoch, self.loss_history[-1]))
                    if self.epoch % 10 == 0:
                        val_accuracy = evaluate_model(self, data, data['idx_to_word'], batch_size=10) # evaluate the BLEU score from time to in a small batch time...
                        self.val_acc_history.append(val_accuracy)
                        if val_accuracy > self.best_val_acc:
                            self.best_val_acc = val_accuracy
                            torch.save(self, 'models/best_validation.pytorch')

        torch.save(self, 'models/current_model.pytorch') # maybe to save the model giving the best BLEU score?
        return self.loss_history



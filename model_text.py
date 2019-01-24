import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from coco_utils import  sample_coco_minibatch, decode_captions
from bleu_score import evaluate_model
# import EarlyStopping
from pytorchtools import EarlyStopping
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam
from torchvision.utils import save_image
from image_utils import image_from_url
from Visdom_grapher import VisdomGrapher
import matplotlib.pyplot as plt
def maximums(T, N):
    """returns the N higher values in tensor T, with their positions,
        along dimension 2

    inputs :
        - T : (1, 1, E) tensor
        - N : int
    outputs :
        (max_val, max_ind) list
            max_val : float
            max_ind : (1) LongTensor 	"""

    T_copy = T.clone()  # les valeurs de T_copy seront modifiées, pas celles du tenseur original
    infinity = (T.abs()).sum(dim=(0,1,2)).item()+1
    # everytime a maximum has been found, the tensor is modified so that we dont find it again
    result = []

    for i in range(N):
        max_val, max_ind = T_copy.max(dim=2)
        max_val = max_val.item()

        T_copy[0, 0, max_ind.item()] -= infinity  # Cette valeur-là ne sera plus jamais prise

        result.append((max_val, max_ind.view(1)))

    return result


class Model_text_lstm(nn.Module):

    def __init__(self, embed_size, img_feat_size, hidden_size, word_2_idx, num_layers, max_seq_length=17, device = 'cpu',embedding_matrix = None):
        """Set the hyper-parameters and build the layers."""
        super(Model_text_lstm, self).__init__()

        self.device = device

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
        # if I use pre-trained embeddings and fine-tune them
        if embedding_matrix is not None:
            self.word_embedding = self.create_emb_layer(embedding_matrix, non_trainable=False)
            print('Using a pre-trained embedding matrix, but fine-tuning it')

        # design LSTM
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )

        self.droplayer  =nn.Dropout(0.4) # to add some regularization
        # output layer which projects back to tag space
        self.hidden_to_vocab = nn.Linear(self.hidden_size, self.nb_vocab_words)


    def init_hidden(self, X, image_feat = None):
        ''' initializes the hidden layer of LSTM
        if image_feat are given - initialize with them hidden_h '''
        batch_size, D, _ = X.shape
        # the weights are of the form (nb_layers, batch_size, nb_lstm_units)
        hidden_h = torch.randn(self.num_layers, batch_size, self.hidden_size).to(self.device)/(D*D)
        hidden_c = torch.randn(self.num_layers, batch_size, self.hidden_size).to(self.device)/(D*D)

        if image_feat is not None:
            # then use the embedded image as a hidden layer
            hidden_h = image_feat.unsqueeze(0)#image_feat.view(1, -1, -1)
        return hidden_h, hidden_c

    def _reset(self):
        """
        Set up some book-keeping variables for optimization. Not sure I need it
        """
        # Set up some variables for book-keeping
        self.best_val_acc = 0
        self.best_params = {}
        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

    def create_emb_layer(self, weights_matrix, non_trainable=False):
        '''
        Function which creates the embedding layer for the vocabulary
        :param weights_matrix: the embedding weights
        :param non_trainable: if I don't want to retrain the matrix
        :return:
        the embedding layer
        '''

        num_embeddings, embedding_dim = weights_matrix.shape
        weights_matrix = torch.from_numpy(weights_matrix).float().to(self.device)
        emb_layer = nn.Embedding.from_pretrained(weights_matrix)
        # emb_layer = nn.Embedding(num_embeddings, embedding_dim)
        # emb_layer.load_state_dict({'weight': weights_matrix})
        if non_trainable:
            emb_layer.weight.requires_grad = False

        return emb_layer

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

        ''' forward pass:
        features - CNN weights
        X - gt captures '''

        batch_size, seq_len = X.shape

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

        X = self.droplayer(X)

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
        ''' Computes cross entory loss:
        Y - true labels (scores from the last output layer
        Y_hat - predicted labels '''

        mask = (Y != self._null).float()
        ce_loss = self.temporal_softmax_loss(Y_hat, Y, mask)

        return ce_loss

    def sample(self, features):
        """ function which samples the captions from a pre-trained model
        features - image features

        """
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


        return np.array(captions.cpu()) # cast back to numpy


    def train_val_step(self, data, batch_size, optimizer, train_mode = True):
        optimizer.zero_grad()
        if train_mode:
            minibatch = sample_coco_minibatch(data, batch_size=batch_size, split='train')
        else:
            minibatch = sample_coco_minibatch(data, batch_size=batch_size, split='val')
        captions, features, urls = minibatch
        captions = torch.LongTensor(captions).to(self.device)
        features = torch.from_numpy(features).to(self.device)
        captions_in = captions[:, :-1]
        captions_out = captions[:, 1:]
        Y_hat = self.forward(features, captions_in)
        loss = self.loss(Y_hat, captions_out)
        if train_mode:
            loss.backward()
            optimizer.step()
        return loss


    def train(self, data, num_epochs, batch_size, optimizer, env_name, use_visdom = None):

        self._reset() # reset the model
        scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True) #to reduce lr
        early_stopping = EarlyStopping(patience = 100, verbose = True) #criteria of stopping
        if use_visdom is not None:
            vis = VisdomGrapher(env_name=env_name, server=use_visdom)

        print('Training...')
        num_train = data['train_captions'].shape[0]
        iterations_per_epoch = np.int(num_train // batch_size)
        num_iterations = num_epochs * iterations_per_epoch
        for i in range(num_iterations):
            loss_tr = self.train_val_step(data, batch_size, optimizer, True)
            loss_val = self.train_val_step(data, batch_size, optimizer, False)
            scheduler.step(loss_val)
            if use_visdom:
                # to visualize the thing
                vis.add_scalar(plot_name='Training loss', idtag='train', y=loss_tr.item(), x=i)
                vis.add_scalar(plot_name='Validating loss', idtag='val', y=loss_val.item(), x=i)

            if (i) % 10 == 0:
                print('Epoch:  %d | Current Loss: %.4f' % (i, loss_tr.item()))
                val_accuracy, train_accuracy = evaluate_model(self, data, data['idx_to_word'], batch_size=50) # evaluate the BLEU score from time to in a small batch time...
                self.val_acc_history.append(val_accuracy)
                if use_visdom:
                    # to visualize the thing
                    vis.add_scalar(plot_name='Bleu Score Validation', idtag='val_bleu', y=val_accuracy, x=i)
                    vis.add_scalar(plot_name='Bleu Score Train', idtag='train_bleu', y=train_accuracy, x=i)
                    test_im, test_cap = self.getAnnotatedImage(data, 'train')
                    vis.add_image(plot_name=test_cap, idtag='train_img', image=test_im)
                    val_im, val_cap = self.getAnnotatedImage(data, 'val')
                    vis.add_image(plot_name=val_cap, idtag='val_img', image=val_im)

                early_stopping(loss_val.cpu().detach().numpy(), self)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
        torch.save(self, 'models/trained_model.pytorch') # maybe to save the model giving the best BLEU score?
        return self.loss_history


    def beam_decode(self, features, beam_size = 3):
       ''' version with a single sample and the beam_search
       features = 1 single image
       Quick explanation :
            S = beam size
            all the kept beams are in a list (best_beams). At first, it starts with only 1 element, the <BOS> tag.
            For every letter we add :
                For every current_beam in the list :
                    compute the best S beams starting from current_beam
                    add them to the next_best_beams list
                next_best_beams now has S² elements
                put the S best beams of next_best_beams in the best_beams list
            return best_beams[0]

            One beam is a 5-tuple (conditional probability, hidden_states_top, hidden_states_bot, previous letters, previous letter probabilities)
        """
        '''
       features = torch.from_numpy(features).to(self.device)  # make a tensor here
       imf2hid = self.image_embedding(features)  # initial hidden state: [N, H]
       hidden_h = imf2hid.unsqueeze(0)  #
       hidden_c = torch.zeros_like(hidden_h)

       wordlist = torch.tensor([self._start]).long().to(self.device)  # On initialise la phrase avec un <BOS>
       out = torch.zeros(1, 1, self.nb_vocab_words).to(self.device) # for the probability
       out[0, 0, self._start] = 1 # because the probability is 1 for start word

       best_beams = [(1, (hidden_h, hidden_c), wordlist, out)]
       iteration = 1
       while iteration < self.max_seq_length:
           next_best_beams = []
           # for all the words
           for (p_cond, (hidden_h, hidden_c), wordlist, out) in best_beams:
               last_word = wordlist[-1].view(1, 1)
               if last_word.item() == self._end:  # 2 is the <EOS> tag index
                   # do not change a thing
                   next_best_beams.append((p_cond, (hidden_h, hidden_c), wordlist, out))
               else:
                   last_word_embedded = self.word_embedding(last_word)
                   inputs = last_word_embedded
                   X, (hidden_h, hidden_c) = self.lstm(inputs, (hidden_h, hidden_c))
                   X = X.contiguous()  # get the last hidden layer
                   X = X.view(-1, X.shape[2])
                   # run through actual linear layer
                   X = self.hidden_to_vocab(X)
                   X = X.unsqueeze(0)
                   out = torch.cat([out, X], dim=0) # save the scores
                   maxs_list = maximums(F.softmax(X, dim=2), beam_size)
                   next_best_beams += [(p_cond * value, (hidden_h, hidden_c),
                                        torch.cat([wordlist, lastword], dim=0), out) for (value, lastword) in
                                       maxs_list]

                   # end for {beam in best_beams}

           # restrict to a fixed number
           next_best_beams.sort(key=lambda x: -x[0])  # tri par probabilité **décroissante** (d'où le "-")
           next_best_beams = next_best_beams[:beam_size]  # restriction

           best_beams = next_best_beams
           iteration += 1
           # end while {t<T}

       wordlist = best_beams[0][2]
       out = best_beams[0][3]
       out = out.view(out.shape[0], 1, -1)

       return out, np.array(wordlist.cpu())



    def getAnnotatedImage(self, data, split):
        ''' samples image and returns it with GT and generated capture'''
        minibatch = sample_coco_minibatch(data, batch_size=1, split=split)
        captions, features, urls = minibatch
         # sample some captions given image features
        gt_captions = decode_captions(captions,  data['idx_to_word'] )
        #_, sample_captions = self.beam_decode(features)
        captions_out = self.sample(features)
        sample_captions = decode_captions(captions_out, data['idx_to_word'])
        for gt_caption, sample_caption, url in zip(gt_captions, sample_captions, urls):
             img = image_from_url(url)
             img = np.asarray(img)
             try:
                img = np.swapaxes(img, 0, 2).transpose(0,2,1)
             except ValueError:
                 img = np.random.rand(3, 256, 256)
             caption = ('%s:%s.\nGT:%s' % (split, sample_caption, gt_caption))

        return img, caption












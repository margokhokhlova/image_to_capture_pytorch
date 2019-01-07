from coco_utils import load_coco_data, sample_coco_minibatch, decode_captions
from image_utils import image_from_url
import numpy as np
import matplotlib
from Toy_LSTM import Model_toy_lstm, temporal_softmax_loss
import torch
from torch.optim import Adam
from model_text import Model_text_lstm
import bleu_score

def get_cap_lenght(captions, PAD):
    ''' return lenght of sequences without padding'''
    #for sentence in captions:
    X_lengths = [len([x for x in w if x != PAD]) for w in captions]
    #captions_sorted = [x for _, x in sorted(zip(X_lengths, captions), reverse=True)]
    captions_sorted = [x for _, x in sorted(zip(X_lengths, captions), key=lambda pair: pair[0], reverse=True)]
    return sorted(X_lengths, reverse=True), captions_sorted


def evaluate_model(model):
    """
    model: CaptioningRNN model
    Prints unigram BLEU score averaged over 1000 training and val examples.
    """
    for split in ['train', 'val']:
        minibatch = sample_coco_minibatch(med_data, split=split, batch_size=1000)
        gt_captions, features, urls = minibatch
        gt_captions = decode_captions(gt_captions, data['idx_to_word'])

        sample_captions = model.sample(features)
        sample_captions = decode_captions(sample_captions, data['idx_to_word'])

        total_score = 0.0
        for gt_caption, sample_caption, url in zip(gt_captions, sample_captions, urls):
            total_score += BLEU_score(gt_caption, sample_caption)

        BLEUscores[split] = total_score / len(sample_captions)

    for split in BLEUscores:
        print('Average BLEU score for %s: %f' % (split, BLEUscores[split]))


# Load COCO data from disk; this returns a dictionary
# We'll work with dimensionality-reduced features for this notebook, but feel
# free to experiment with the original features by changing the flag below.
data = load_coco_data(pca_features=True)

# Print out all the keys and values from the data dictionary
for k, v in data.items():
    if type(v) == np.ndarray:
        print(k, type(v), v.shape, v.dtype)
    else:
        print(k, type(v), len(v))

small_data = load_coco_data(max_train=50)
minibatch = sample_coco_minibatch(small_data, batch_size=10, split='train')
captions, features, urls = minibatch
print(captions.shape)
print(features.shape)


word2idx = data['word_to_idx']
PAD = word2idx.get('<NULL>', None)
# ## toy model test
# captions = torch.LongTensor(captions)
# model = Model_toy_lstm(embed_size=50, hidden_size=256, word_2_idx=word2idx, num_layers=1, max_seq_length=20)
# optimizer = Adam(model.parameters(), lr=0.001)
# output = model(captions)
# loss = model.loss(output, captions)
# loss.backward()

num_epochs = 10

#captions_lengths, captions = get_cap_lenght(captions, PAD)

model = Model_text_lstm(embed_size=50, hidden_size=256, word_2_idx=word2idx, num_layers=1, max_seq_length=20)
optimizer = Adam(model.parameters(), lr=0.001)



for i in range(num_epochs):
    # training loop
    train_running_loss = 0.0
    train_acc = 0.0
    model.train()

    for batches in range(10):
        small_data = load_coco_data(max_train=500)
        minibatch = sample_coco_minibatch(small_data, batch_size=50, split='train')
        captions, features, urls = minibatch
        captions = torch.LongTensor(captions)

        # zero the parameter gradients
        optimizer.zero_grad()
        Y_hat = model(captions)
        loss = model.loss(Y_hat, captions)
        loss.backward()
        optimizer.step()
        train_running_loss += loss.detach().item()
        #print('Epoch:  %d | Loss: %.4f' % (i,  train_running_loss))
    model.eval()
    print('Epoch:  %d | Average Loss: %.4f' % (i, train_running_loss / (i+1)))



from coco_utils import load_coco_data, sample_coco_minibatch, decode_captions
from image_utils import image_from_url
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.optim import Adam
from model_text import Model_text_lstm
from  bleu_score import evaluate_model

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
device = (torch.device('cuda:0') if torch.cuda.is_available() else 'cpu')
print("device  :",device)


###### DATA LOAD

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

# load a small samle of data and let's go!
small_data = load_coco_data(max_train=50)
word2idx = data['word_to_idx']
num_epochs = 100
batch_size = 25

model = Model_text_lstm(embed_size=256, hidden_size=512, img_feat_size=512, word_2_idx=word2idx, num_layers=1, max_seq_length=17, device=device)
optimizer = Adam(model.parameters(), lr=0.001)

####### DEVICE
model.to(device)


####### TRAIN

loss_history = model.train(small_data, num_epochs, batch_size, optimizer)

# different things with plots, no need for the online version

# # Plot the training losses ==> TO CHANGE FOR VISDOM LATER
# plt.plot(loss_history)
# plt.xlabel('Iteration')
# plt.ylabel('Loss')
# plt.title('Training loss history')
# plt.show()


# #######  TEST

# show some examples
print("print some examples...")
for split in ['train', 'val']:
    minibatch = sample_coco_minibatch(small_data, batch_size=3, split=split)
    captions, features, urls = minibatch
    # sample some captions given image features
    captions_out = model.sample(features)

    gt_captions = decode_captions(captions,  data['idx_to_word'] )
    sample_captions = decode_captions(captions_out,  data['idx_to_word'])
    for gt_caption, sample_caption, url in zip(gt_captions, sample_captions, urls):
        # plt.imshow(image_from_url(url))
        # plt.title('%s\n%s\nGT:%s' % (split, sample_caption, gt_caption))
        # plt.axis('off')
        # plt.show()
        print('%s\n%s\nGT:%s' % (split, sample_caption, gt_caption))

# calculate the BLEU score

print("evaluating the model using the BLEU score after the training is finished: ")
evaluate_model(model, small_data, data['idx_to_word'], batch_size=1)  # evaluate the BLEU score on 50 samples...





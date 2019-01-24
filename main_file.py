from coco_utils import load_coco_data, sample_coco_minibatch, decode_captions
from image_utils import image_from_url
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.optim import Adam
from model_text import Model_text_lstm
from  bleu_score import evaluate_model
import argparse
from pre_trained_embeddings import readGloveFile, get_embedding_matrix


import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
device = (torch.device('cuda:0') if torch.cuda.is_available() else 'cpu')
print("device  :",device)



###### PARSING THE ARGUEMENTS

parser = argparse.ArgumentParser(description='image2caption example')

# Task parameters
parser.add_argument('--uid', type=str, default='image2caption',
                    help='Staging identifier (default:image2caption)')
# Model parameters
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='Hidden layer  size (default: 256')
parser.add_argument('--pre_trained_wordemd', type=bool, default=False,
                    help='Enable pre-trained word embeddings')
parser.add_argument('--wordembed_size', type=int, default=256, metavar='N',
                    help='Word Embedding Size layer  size (default: 256) Attention!  chenge to the other one if embde vectors are used')


# Optimizer
parser.add_argument('--optimizer', type=str, default="Adam",
                    help='Optimizer (default: Adam')
parser.add_argument('--epochs', type=int, default=25, metavar='N',
                    help='number of training epochs')
parser.add_argument('--lr', type=float, default= 0.001,
                    help='Set Learning rate')
parser.add_argument('--batchsize', type=int, default= 50,
                    help='Set Batch size')



# Visdom / tensorboard
parser.add_argument('--visdom-url', type=str, default=None,
                    help='visdom url, needs http, e.g. http://localhost (default: None)')
parser.add_argument('--visdom-port', type=int, default=8097,
                    help='visdom server port (default: 8097')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='batch interval for logging (default: 1')
parser.add_argument('--visdomenv',type=str, default='margotest',
                    help='environment name for Visdom')



## START OF THE PROGRAM
args = parser.parse_args()
use_pretrainedemd = args.pre_trained_wordemd is not False
use_visdom = args.visdom_url if args.visdom_url is not None else None
lr = args.lr # the learning rate
num_epochs = args.epochs
batch_size = args.batchsize
hidden_size = args.hidden_size
env_name = args.visdomenv
w_emb = args.wordembed_size


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

######### EMBEDDINGS FOR THE TEXT
embedding_weights = None
if use_pretrainedemd:
# Load Embeddings
    print('Load Embeddings')
    GLOVE_DIR = '/data/khokhlov/embeddings/'
    embeddings_index = readGloveFile(GLOVE_DIR)
    embedding_weights = get_embedding_matrix(data['word_to_idx'], data['idx_to_word'], embeddings_index)


# load a small samle of data and let's go!
small_data = load_coco_data(max_train=1000)
word2idx = data['word_to_idx']


model = Model_text_lstm(embed_size=w_emb, hidden_size=hidden_size, img_feat_size=512, word_2_idx=word2idx,
                        num_layers=1, max_seq_length=17, embedding_matrix=embedding_weights, device=device)
optimizer = Adam(model.parameters(), lr=lr)

####### DEVICE
model.to(device)

####### TRAIN

loss_history = model.train(small_data, num_epochs, batch_size, optimizer,  env_name, use_visdom)
#
# # # Plot the training losses ==> TO CHANGE FOR VISDOM LATER
# # plt.plot(loss_history)
# # plt.xlabel('Iteration')
# # plt.ylabel('Loss')
# # plt.title('Training loss history')
# # plt.show()
#
#
# #######  TEST
#
# # show some examples
# for split in ['train', 'val']:
#     minibatch = sample_coco_minibatch(small_data, batch_size=2, split=split)
#     captions, features, urls = minibatch
#     # sample some captions given image features
#     captions_out = model.sample(features)
#
#     gt_captions = decode_captions(captions,  data['idx_to_word'] )
#     sample_captions = decode_captions(captions_out,  data['idx_to_word'])
#     for gt_caption, sample_caption, url in zip(gt_captions, sample_captions, urls):
#         plt.imshow(image_from_url(url))
#         plt.title('%s\n%s\nGT:%s' % (split, sample_caption, gt_caption))
#         plt.axis('off')
#         plt.show()


# load best model
model = torch.load("models/best_validation.pytorch")
model.to(device)
# calculate the BLEU score
print('Final best model score on 100 samples: ')
evaluate_model(model, small_data, data['idx_to_word'], batch_size=100, beam_size=5)  # evaluate the BLEU score on 50 samples...









from typing import Dict, List, Any

import numpy as np
import os
import re


''' I use the pre-trained word embeddings from GLOVE'''
def readGloveFile(GLOVE_DIR):
    embeddings_index = {}

    f = open(os.path.join(GLOVE_DIR, 'glove.6B.50d.txt'), encoding="utf8")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))
    return embeddings_index



def get_embedding_matrix(word2ind, ind2word, glove, EMBEDDING_DIM = 50):
    matrix_len = len(ind2word)

    weights_matrix = np.zeros((matrix_len, 50))
    words_found = 0

    for index in range(matrix_len):
        word = ind2word[index]
        try:
            weights_matrix[index] = glove[word]
            words_found += 1
        except KeyError:
            weights_matrix[index] = np.random.normal(scale=0.6, size=(EMBEDDING_DIM,))
    # embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    # for word, i in word_index.items():
    #     embedding_vector = embeddings_index.get(word)
    #     if embedding_vector is not None:
    #     # words not found in embedding index will be all-zeros.
    #         embedding_matrix[i] = embedding_vector
    #     else:
    #         print('Missing word detected %s' % word)
    #        # word_emb = embeddings_index.get('x') # let's get this word for the moment
    #        # embedding_matrix[i] = word_emb
    print("The number of words in pre-trained embeddings matrix is: %d" % words_found)
    return weights_matrix


def get_indexed_sentences(data, dct):
    # append our dictionary with new words    dct.add_documents["EOL", "PAD", "BOF", "UNK"]
    list_of_sentences = []
    list_of_ids = []
    for caption in data:
        id_sentence = caption['id']
        sentences = (caption['caption']) # list of all sentences
        for sent in sentences:
            sent = sent.rstrip('\n')
            single_sentence = re.findall(r"[\w']+|[.,!?;[\n]", sent.lower(), re.DOTALL)
            # small loop to remove the words which are not in the dct
            for idx, word in enumerate(single_sentence):
                try:
                    single_sentence[idx] = dct.token2id[(word)]
                except KeyError:
                    single_sentence[idx] = dct.token2id["UNK"]
            list_of_sentences.append(single_sentence)
            list_of_ids.append(id_sentence)
    return list_of_sentences, list_of_ids



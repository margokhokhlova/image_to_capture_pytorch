from coco_utils import sample_coco_minibatch, decode_captions
import nltk
import numpy as np
import torch

def BLEU_score(gt_caption, sample_caption):
    """
    gt_caption: string, ground-truth caption
    sample_caption: string, your model's predicted caption
    Returns unigram BLEU score.
    """
    reference = [x for x in gt_caption.split(' ')
                 if ('<END>' not in x and '<START>' not in x and '<UNK>' not in x)]
    hypothesis = [x for x in sample_caption.split(' ')
                  if ('<END>' not in x and '<START>' not in x and '<UNK>' not in x)]
    BLEUscore = nltk.translate.bleu_score.sentence_bleu([reference], hypothesis, weights = [1])
    return BLEUscore

def evaluate_model(model, med_data, idx_to_word, batch_size = 1000, beam_size = None):
    """
    model: CaptioningRNN model
    Prints unigram BLEU score averaged over 1000 training and val examples.

    """
    BLEUscores = {}
    if beam_size is None: # no beam search
        for split in ['train', 'val']:
            minibatch = sample_coco_minibatch(med_data, split=split, batch_size=batch_size)
            gt_captions, features, urls = minibatch
            gt_captions = decode_captions(gt_captions, med_data['idx_to_word'])

            sample_captions = model.sample(features)

            sample_captions = decode_captions(sample_captions, med_data['idx_to_word'])

            total_score = 0.0
            for gt_caption, sample_caption, url in zip(gt_captions, sample_captions, urls):
                total_score += BLEU_score(gt_caption, sample_caption)

            BLEUscores[split] = total_score / len(sample_captions)

        for split in BLEUscores:
            print('Average BLEU score for %s: %f' % (split, BLEUscores[split]))
    else: # with beam search
        for split in ['train', 'val']:
            sample_captions = []  # empty list for the sample captures
            gt_captions = [] # empty list for GT
            urls = []
            for batch in range(batch_size):
                minibatch = sample_coco_minibatch(med_data, split=split, batch_size=1) # each time only one sample
                gt_caption, features, url = minibatch
                gt_caption = decode_captions(gt_caption, med_data['idx_to_word'])

                _, sample_caption = model.beam_decode(features, beam_size=beam_size)

                sample_caption = decode_captions(sample_caption, med_data['idx_to_word'])

                sample_captions.append(str(sample_caption))
                gt_captions.append(str(gt_caption))
                urls.append(url)

            total_score = 0.0
            for gt_caption, sample_caption, url in zip(gt_captions, sample_captions, urls):
                total_score += BLEU_score(gt_caption, sample_caption)

            BLEUscores[split] = total_score / len(sample_captions) # divide by the lenght of words
        for split in BLEUscores:
            print('Average BLEU score for %s: %f' % (split, BLEUscores[split]))

    return BLEUscores['val']
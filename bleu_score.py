from coco_utils import sample_coco_minibatch, decode_captions
import nltk
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

def evaluate_model(model, med_data, idx_to_word, batch_size = 1000):
    """
    model: CaptioningRNN model
    Prints unigram BLEU score averaged over 1000 training and val examples.

    """
    BLEUscores = {}
    for split in ['train', 'val']:
        minibatch = sample_coco_minibatch(med_data, split=split, batch_size=batch_size)
        gt_captions, features, urls = minibatch
        gt_captions = decode_captions(gt_captions, med_data['idx_to_word'])

        features = torch.from_numpy(features) # make a tensor here

        sample_captions = model.sample(features)
        sample_captions = decode_captions(sample_captions, med_data['idx_to_word'])

        total_score = 0.0
        for gt_caption, sample_caption, url in zip(gt_captions, sample_captions, urls):
            total_score += BLEU_score(gt_caption, sample_caption)

        BLEUscores[split] = total_score / len(sample_captions)

    for split in BLEUscores:
        print('Average BLEU score for %s: %f' % (split, BLEUscores[split]))
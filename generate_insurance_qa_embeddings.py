#!/usr/bin/env python

"""
Command-line script for generating embeddings
Useful if you want to generate larger embeddings for some models

TODO: Log a few words to check similarities
TODO: Run 10 epochs with method in notebook (with decreasing learning rate)
"""

import os
import sys
import random; random.seed(42)
import pickle
import argparse
import logging


def revert(vocab, indices):
    """Convert word indices into words
    """
    return [vocab.get(i, 'X') for i in indices]

try:
    data_path = os.environ['INSURANCE_QA']
except KeyError:
    print('INSURANCE_QA is not set. Set it to your clone of https://github.com/codekansas/insurance_qa_python')
    sys.exit(1)

# parse arguments
parser = argparse.ArgumentParser(description='Generate embeddings for the InsuranceQA dataset')
parser.add_argument('--iter', metavar='N', type=int, default=10, help='number of times to run')
parser.add_argument('--size', metavar='D', type=int, default=100, help='dimensions in embedding')
args = parser.parse_args()

# configure logging
logger = logging.getLogger(os.path.basename(sys.argv[0]))
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)
logger.info('running %s' % ' '.join(sys.argv))

# imports go down here because they are time-consuming
from gensim.models import Word2Vec
import gensim.models.word2vec
from keras_models import *
from util import load
import multiprocessing

cores = multiprocessing.cpu_count()
assert gensim.models.word2vec.FAST_VERSION > -1, "this will be painfully slow otherwise"

vocab = load(data_path, 'vocabulary')

answers = load(data_path, 'answers')
sentences = [revert(vocab, txt) for txt in answers.values()]
sentences += [revert(vocab, q['question']) for q in load(data_path, 'train')]

# run model
# sg=0 uses CBOW. Read somewhere that cbow better for our use case
model = Word2Vec(size=args.size, min_count=1, window=5, sg=0)
model.build_vocab(sentences)
for epoch in range(args.iter):
    logging.info("Epoch %d" % epoch)
    model.train(sentences)
    model.alpha -= 0.002
    model.min_alpha = model.alpha
    random.shuffle(sentences)

weights = model.wv.syn0
d = dict([(k, v.index) for k, v in model.wv.vocab.items()])
emb = np.zeros(shape=(len(vocab)+1, args.size), dtype='float32')

for i, w in vocab.items():
    if w not in d: continue
    emb[i, :] = weights[d[w], :]

np.save(open('word2vec_%d_dim.embeddings' % args.size, 'wb'), emb)
logger.info('saved to "word2vec_%d_dim.embeddings"' % args.size)
np.save(open('word2vec_%d_dim.model' % args.size, 'wb'), model)
logger.info('saved to "word2vec_%d_dim.model"' % args.size)

logger.info("Most similar word vectors")
test_words = ['£amount100', 'pin', 'bug', 'limit', 'froze', 'monzo', 'ios',
    'monzome', 'address', 'number', 'queue', 'topup', '€number', 'unable']
for tw in test_words:
    logger.info("Most similar word to %s: \n %s" % 
        (tw, ', '.join([tt[0] for tt in model.most_similar(tw)]))
    )


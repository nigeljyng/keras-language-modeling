import os
import pickle

def load(path, name):
    return pickle.load(open(os.path.join(path, name), 'rb'))


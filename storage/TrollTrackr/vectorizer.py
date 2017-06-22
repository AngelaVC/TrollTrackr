from sklearn.feature_extraction.text import from sklearn.feature_extraction.text import TfidfVectorizer
import re
import os
import pickle

cur_dir = os.path.dirname(os.path.abspath('__file__'))

with open('forest_model.pkl', 'rb') as fin:
  vectorizer, forest = pickle.load(fin)

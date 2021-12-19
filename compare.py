# use this file to play around with fastText vectors.
# * note that AnnA is faster than this implementation because it uses
#   a memoize call to fastText

import fasttext as fastText
import fasttext.util
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
from nltk.corpus import stopwords
import numpy as np
import re

# SETTINGS : ##############################################
lang_2L = "fr"  # < language in 2 letter format, example: fr, en
lang_full = "french"  # < language in full word, example french, english

# USAGE : #################################################
# a low output score means that the two sentence are similar :
# compare("learn how to walk", "understand how to move"

stops = stopwords.words(lang_full) + [""]
fastText.util.download_model(lang_2L, if_exists='ignore')
ft = fastText.load_model(f"cc.{lang_2L}.300.bin")
alphanum = re.compile(r"[^ _\w]|\d|_")


def preprocessor(string):
    """
    prepare string of text to be vectorized by fastText
    * makes lowercase
    * removes all non letters
    * removes extra spaces
    * outputs each words in a list
    """
    return re.sub(alphanum, " ", string.lower()).split()


def vec(string):
    return normalize(np.sum([ft.get_word_vector(x)
                             for x in preprocessor(string)
                             if x not in stops],
                            axis=0).reshape(1, -1),
                     norm='l2')


def compare(s1, s2):
    "compare two sentence"
    dist = pairwise_distances(vec(s1), vec(s2), metric="cosine")
    return np.round(float(dist), 4)

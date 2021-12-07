# use this file to play around with fastText vectors.
# * note that AnnA is faster than this implementation because it uses
#   a memoize call to fastText
# * a low comparison score means that the two sentence are similar

import fasttext as fastText
import fasttext.util
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
import numpy as np
import re

fastText_lang = "fr"
fastText.util.download_model(fastText_lang, if_exists='ignore')
ft = fastText.load_model(f"cc.{fastText_lang[0:2]}.300.bin")
alphanum = re.compile(r"[^ _\w]|\d|_")
apostrophes = re.compile("[a-zA-Z]\'")

def preprocessor(string):
    """
    prepare string of text to be vectorized by fastText
    * makes lowercase
    * removes every letter+apostrophe like " t'aime "
    * removes all non letters
    """
    return re.sub(alphanum,
                  "",
                  re.sub(apostrophes, "", string.lower())
                  )

def vec(string):
    return normalize(np.sum([ft.get_word_vector(preprocessor(x)) for x in string.split(" ") if x != ""],
                         axis=0
                         ).reshape(1, -1),
                     norm='l2')

def compare(s1, s2):
    "used to simply compare two sentence"
    return np.round(float(pairwise_distances(vec(s1),
                              vec(s2),
                              metric="cosine")), 4)

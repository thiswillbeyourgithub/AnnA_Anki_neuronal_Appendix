import fasttext as fastText
import fasttext.util
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
import numpy as np

fastText_lang = "fr"
fastText.util.download_model(fastText_lang, if_exists='ignore')
ft = fastText.load_model(f"cc.{fastText_lang[0:2]}.300.bin")

def get_vec(word):
    # uncomment to print which words are identified:
    #print(f"Identified word: {word}")
    return ft.get_word_vector(word)

def vec(string):
    return normalize(np.sum([get_vec(x) for x in string.split(" ") if x != ""],
                         axis=0
                         ).reshape(1, -1),
                     norm='l1')

def compare(s1, s2):
    "used to simply compare two sentence"
    print("(lower score = more similar)")
    return np.round(float(pairwise_distances(vec(s1),
                              vec(s2),
                              metric="cosine")), 4)

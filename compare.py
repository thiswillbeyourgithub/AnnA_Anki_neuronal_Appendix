from sentence_transformers import SentenceTransformer, util
# multilingual model:
#sBERT = SentenceTransformer('distiluse-base-multilingual-cased-v1')
# glove model:
sBERT = SentenceTransformer('sentence-transformers/average_word_embeddings_glove.6B.300d')
# english only model:
#sBERT = SentenceTransformer('all-MiniLM-L6-v2')

def compare(s1, s2):
    "used to simply compare two sentence"
    return util.cos_sim(sBERT.encode(s1), sBERT.encode(s2))

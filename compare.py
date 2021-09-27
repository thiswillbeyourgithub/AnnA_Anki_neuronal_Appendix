
def compare(s1, s2):
    "used to simply compare two sentence"
    from sentence_transformers import SentenceTransformer, util

    # multilingual model:
    sBERT = SentenceTransformer('distiluse-base-multilingual-cased-v1')

    # english only model:
    #sBERT = SentenceTransformer('all-MiniLM-L6-v2')

    return util.cos_sim(sBERT.encode(s1),
                        sBERT.encode(s2))

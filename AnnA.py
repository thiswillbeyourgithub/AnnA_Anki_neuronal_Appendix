import pdb
import signal
import os
import argparse
import json
import urllib.request
import pyfiglet
import pandas as pd
from pprint import pprint
from tqdm import tqdm
import re
from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter

from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
#import numpy as np
#import transformers
#from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.decomposition import TruncatedSVD
#from sklearn.metrics import
from sklearn.decomposition import PCA
import plotly.express as px
import umap.umap_

os.environ["TOKENIZERS_PARALLELISM"] = "true"

# makes the script interuptible, resume it using c+enter
signal.signal(signal.SIGINT, (lambda signal, frame : pdb.set_trace()))


##################################################
# arguments
parser = argparse.ArgumentParser()
parser.add_argument("-d",
                    "--deck",
                    help="the name of the deck that you wish to use",
                    dest="deck",
                    metavar="DECKNAME")
parser.add_argument("-v",
                    "--verbose",
                    help="increase verbosity, for debugging",
                    action='store_true',
                    dest="verbosity")
parser.add_argument("--replace_greek",
                    help="replace greek chacacter letters to its written\
equivalent. See file greek_alphabet_mapping.",
                    action='store_true',
                    dest="replace_greek")
parser.add_argument("--replace_acronym",
                    help="to automatically replace acronyms by their written\
equivalent. The user has to supply the list as a dictionary in the file\
user_acronym_list.",
                    action='store_true',
                    dest="replace_acronym")
parser.add_argument("--stop_word_lang",
                    help="language in which to fetch the stop words. Can be\
used multiple times. For example 'english' and 'french'. Default is none.",
                    action='append',
                    dest="stop_word_lang")
parser.add_argument("--keep_ocr",
                    help="to keep ocr text or not (see related addons)",
                    action='store_true',
                    dest="keep_ocr")
args = parser.parse_args().__dict__


##################################################
# anki related functions
def ankiconnect_invoke(action, **params):
    "send requests to ankiconnect addon"

    def request_wrapper(action, **params):
        return {'action': action, 'params': params, 'version': 6}

    requestJson = json.dumps(request_wrapper(action, **params)).encode('utf-8')
    if verb is True:
        pprint(requestJson)
    try:
        response = json.load(urllib.request.urlopen(
                                urllib.request.Request('http://localhost:8765',
                                                       requestJson)))
    except (ConnectionRefusedError, urllib.error.URLError) as e:
        print(f"{e}: is Anki open and ankiconnect enabled?")
        raise SystemExit()
    if verb is True:
        pprint(response)
    if len(response) != 2:
        raise Exception('response has an unexpected number of fields')
    if 'error' not in response:
        raise Exception('response is missing required error field')
    if 'result' not in response:
        raise Exception('response is missing required result field')
    if response['error'] is not None:
        raise Exception(response['error'])
    return response['result']


def get_deckname_list():
    "get the list of decks"
    return ankiconnect_invoke(action="deckNames")


def get_card_id_from_query(query):
    "get notes from query"
    return ankiconnect_invoke(action="findCards", query=query)


def get_cards_info_from_card_id(card_id):
    "get cardinfo from card id, works with either int of list of int"
    if isinstance(card_id, list):
        r_list = []
        for card in tqdm(card_id):
            r_list.extend(ankiconnect_invoke(action="cardsInfo", cards=[card]))
        return r_list
    if isinstance(card_id, int):
        return ankiconnect_invoke(action="cardsInfo", cards=[card_id])


def sanitize_text(text):
    text = str(text)
    if args["keep_ocr"] is True:
        # extract title of images (usually OCRed text) before html is removed
        text = re.sub("title=(\".*?\")", "> OCR \\1 <", text)
    if args["replace_greek"] is True:
        # https://gist.github.com/beniwohli/765262
        import greek_alphabet_mapping
        for a, b in greek_alphabet_mapping.greek_alphabet.items():
            text = re.sub(a, b, text)
    if args['replace_acronym'] is True:
        import user_acronym_list
        for a, b in user_acronym_list.acronym_list.items():
            text = re.sub(a, b, text)
    text = re.sub(r'[a-zA-Z0-9-]+\....', " ", text)  # removes mediafile
    text = re.sub('\\n|<div>|</div>|<br>|<span>|</span>|<li>|</li>|<ul>|</ul>',
            " ", text)  # removes newline
    text = re.sub("<a href.*?</a>", " ", text)  # removes links
    text = re.sub(r'http[s]?://\S*', " ", text)  # removes plaintext links
    text = re.sub("<.*?>", " ", text)  # removes html tags
    text = re.sub('\u001F', " ", text)  # removes \x1F
    text = re.sub(r"{{c\d+?::", "", text)
    text = re.sub("}}|::", "", text)
    text = text.replace("&nbsp;", " ")
    text = text.replace("&gt;", ">")
    text = text.replace("&l;;", "<")
    return text


def sanitize_fields(field_dic):
    to_return = ""
    for side in field_dic.keys():
        for a, b in field_dic[side].items():
            if a in rlvt_fields:
                to_return += sanitize_text(b)
    return to_return


alphaNumeric = re.compile("[a-z0-9A-Z]{1,25}")
def remove_stopwords(text):
    text = str(text)
    text = [w for w in alphaNumeric.findall(text) if w not in stops]
    text = ' '.join(text).strip()
    return text


##################################################
# machine learning related functions



##################################################
# main loop
if __name__ == "__main__":

    # printing banner
    ascii_banner = pyfiglet.figlet_format("AnnA")
    print(ascii_banner)
    print("Anki neuronal Appendix\n\n")

    # loading args
    deck = args["deck"]
    verb = args["verbosity"]
    lang_list = args["stop_word_lang"]
    if verb is True:
        pprint(args)

    # getting correct deck name
    decklist = get_deckname_list()
    if deck is not None:
        if deck not in decklist:
            print("Couldn't find this deck.")
            deck = None
    if deck is None:
        auto_complete = WordCompleter(decklist,
                                      match_middle=True,
                                      ignore_case=True)
        deck = ""
        while deck not in decklist:
            deck = prompt("Enter the name of the deck to use:\n>", completer=auto_complete)


    # extracting card list
    print("Getting due card from this deck...")
    due_cards = get_card_id_from_query(f"deck:{deck} is:due is:review -is:learn -is:suspended -is:buried")

    print("Getting cards that where rated in the last week from this deck...")
    rated_cards = get_card_id_from_query(f"deck:{deck} rated:7 -is:suspended")

    for i in due_cards:  # removes overlap
        if i in rated_cards:
            rated_cards.remove(i)

    # extracting card information
    all_rlvt_cards = list(rated_cards + due_cards)
    print(f"Fetching information from relevant {len(all_rlvt_cards)} cards...")
    list_cardsInfo = get_cards_info_from_card_id(all_rlvt_cards)

    # creating pandas dataframe
    df = pd.DataFrame(columns=["cardId"])
    df = df.set_index("cardId")
    for i in list_cardsInfo:
        i = dict(i)
        df = df.append(i, ignore_index=True)
    df.drop(columns=["answer", "question", "css", "fieldOrder", "ord"], inplace=True)
    df.sort_index()

    # cleaning up text and keeping only relevant fields
    print("Cleaning up text...")
    rlvt_fields = ["Body", "More", "Source", "Header", "value",
                   "Spanish word with article", "English",
                   "Simple example sentences",
                   "Image", "Header", "Extra 1", "Front", "Back"]
    df["cleaned_text"] = [sanitize_fields(x) for x in tqdm(df["fields"])]
    df.sort_index()

    # getting list of stop words
    stops = []
    if lang_list != []:
        for i in lang_list:
            try:
                stops = stops + stopwords.words(i)
            except Exception as e:
                print(f"{e}: nltk doesn't seem to have a stopwords list for\
language '{i}'")
    else:
        stops = []

    print("Removing stop words...")
    df["cleaned_text_wo_sw"] = [remove_stopwords(x) for x in tqdm(df["cleaned_text"])]
    df.sort_index()


# removed, tfidf is super fast even on very large matrices BUT sentence bert
# includes the wordpiece algorithm

#    print("Loading BERT tokenizer...")
#    tokenizer2 = transformers.BertTokenizerFast.from_pretrained('bert-base-multilingual-uncased')
#
#    print("Tokenizing text using BERT...")
#    df["tkns"] = [' '.join(
#        tokenizer2.convert_ids_to_tokens(tokenizer2.encode(
#            i,
#            add_special_tokens=False,
#            truncation=True))) for i in tqdm(df["cleaned_text_wo_sw"])]
#
#    print("Loading Tfidf...")
#    tfidf = TfidfVectorizer(tokenizer=None)
#    print("Computing Tfidf vectors...")
#    tfs = tfidf.fit_transform(tqdm(df['tkns']))
#
#    print("Reducing Tfidf vectors to 100 dimensions using SVD...")
#    tfs_svd = TruncatedSVD(n_components=512,
#                           random_state=42).fit_transform(tfs)
#    df["tfs_svd"] = [list(x) for x in tfs_svd]

    print("Computing vectors from sentence-bert...")
    model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
    df["sbert_vec"] = model.encode(df["cleaned_text"],
                                    normalize_embeddings=True,
                                    output_value="token_embeddings",
                                    show_progress_bar=True)

    breakpoint()
#    print("Combining vectors from tfidf and sentence-BERT into\
#the same matrix...")
#    df["combo_vec"] = [list(df.loc[i, "sbert_vec"]) + df.loc[i, "tfs_svd"]
#                       for i in df.index]

    print("Reducing sentence-BERT vectors to 50 dimensions only using PCA...")
    vec_n = [int(x) for x in range(len(df.loc[0,"combo_vec"]))]
    df_c_vec = pd.DataFrame(columns=["cardId"] + vec_n)
    df_c_vec["cardId"] = df.index
    df_c_vec = df_c_vec.set_index("cardId")

    for i in df.index:
        df_c_vec.loc[i] = df.loc[i, "combo_vec"]

    pca_fitted = PCA(n_components=2, random_state=42).fit(df_c_vec)
    df_c_vec = pca_fitted.transform(df_c_vec)
    df["combo_vec_pca"] = [list(x) for x in df_c_vec]
#    print("Reducing sentence-BERT vectors to 50 dimension using PCA...")
#    # PCA seemes to only work when given a full dataframe with each column
#    # a vector so  I had to create it
#    vec_n = [int(x) for x in range(0, 512)]
#    df_s_vec = pd.DataFrame(columns=["cardId"] + vec_n)
#    df_s_vec["cardId"] = df.index
#    df_s_vec = df_s_vec.set_index("cardId")
#
#    for i in df.index:
#        df_s_vec.loc[i] = df.loc[i, "sbert_vec"]
#
#    pca_fitted = PCA(n_components=50, random_state=42).fit(df_s_vec)
#    df_s_vec = pca_fitted.transform(df_s_vec)
#    df["sbert_PCA"] = [list(x) for x in df_s_vec]

    print("Clustering using KMeans...")
    kmeans = KMeans(n_clusters=50)
    df['cluster_kmeans'] = kmeans.fit_predict(df_c_vec)

    print("Computing UMAP embeddings in 2D...")
    umap_embed = umap.UMAP(
            n_jobs=-1,
            verbose=3,
            n_components=2,
            metric="euclidean",
            init="random",
            random_state=42,
            transform_seed=42,
            n_neighbors=5,
            min_dist=1,
            n_epochs=500,
            target_n_neighbors=20).fit_transform(df_c_vec)


    print("Plotting results...")
    fig = px.scatter(df,
                     title="AnnA Anki neuronal Appendix",
                     #x=[x[0] for x in df["combo_vec_pca"]],
                     #y=[x[1] for x in df["combo_vec_pca"]],
                     x=umap_embed[:,0],
                     y=umap_embed[:,1],
                     color=df["cluster_kmeans"],
                     hover_data=["cleaned_text"])
    fig.show()
    breakpoint()


##################################################
# TODO
def add_tag_to_card_id(card_id, tag):
    "add tag to card id"
    # first gets note it from card id
    note_id = ankiconnect_invoke(action="cardsToNote", cards=card_id)
    return ankiconnect_invoke(action="addTags",
                              notes=note_id,
                              tags=tag)


def add_vectorTags(note_dic, vectorizer, vectors):
    "adds a tag containing the vector values to each note"
    if vectorizer not in ["sentence_bert", "tf-idf"]:
        print("Wrong vectorTags vectorizer")
        raise SystemExit()
    return ankiconnect_invoke(action="addTags",
                              notes=note_id,
                              tags=vectors_tag)


def add_cluterTags(note_dic):
    "add a tag containing the cluster number and name to each note"
    pass


def add_actionTags(note_dic):
    """
    add a tag containing the action that should be taken regarding cards.
    The action can be "bury" or "study_today"
    Currently, you then have to manually bury them or study them into anki
    """

def get_cluster_topic(cluster_note_dic, all_note_dic, cluster_nb):
    "given notes, outputs the topic that is likely the subject of the cards"
    pass


def find_similar_notes(note):
    "given a note, find similar other notes with highest cosine similarity"
    pass


def find_notes_similar_to_input(user_input, nlimit):
    "given a text input, find notes with highest cosine similarity"
    pass


def show_latent_space(query):
    """
    given a query, will open plotly and show a 2d scatterplot of your cards
    semantically arranged.
    """
    # TODO test if the cards contain the relevant tags otherwise exit

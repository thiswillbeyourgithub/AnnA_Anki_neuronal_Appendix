import pdb
import signal
import os
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
import transformers
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_distances
from sklearn.decomposition import PCA
import plotly.express as px
import umap.umap_
import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "true"

# makes the script interuptible, resume it using c+enter
signal.signal(signal.SIGINT, (lambda signal, frame : pdb.set_trace()))


# I put those three lines here because they are long to run
# hence I don't want to rerun them at each new instance of AnnA
print("Loading BERT tokenizer...")
tokenizer2 = transformers.BertTokenizerFast.from_pretrained('bert-base-multilingual-uncased')
tfidf = TfidfVectorizer(tokenizer=None)
sbert = SentenceTransformer('distiluse-base-multilingual-cased-v1')

class AnnA:
    def __init__(self,
                 deckname="",
                 verbose=False,
                 replace_greek=True,
                 replace_acronym=False,
                 stop_word_lang=["en"],
                 keep_ocr=False,
                 rated_last_X_days=4,
                 skip_banner=False):
        # printing banner
        if skip_banner is not True:
            ascii_banner = pyfiglet.figlet_format("AnnA")
            print(ascii_banner)
            print("Anki neuronal Appendix\n\n")

        # loading args etc
        self.deckname = deckname
        self.verbose = verbose
        self.replace_greek = replace_greek
        self.replace_acronym = replace_acronym
        self.stop_word_lang = stop_word_lang
        self.keep_ocr = keep_ocr
        self.rated_last_X_days = rated_last_X_days
        self.word_finder = re.compile("[a-z0-9A-Z]{1,25}")


        # actual execution
        self.stop_words = self._gathering_stopwords()
        self.deckname = self._check_deck(deckname)
        self._create_and_fill_df()
        self._clean_text()
        self._remove_stop_words()
        self._vectors()
        self.Kmeans_clustering()
        #self.UMAP_embeddings()
        self.plot_map()

    def _vectors(self):
        df = self.df

        print("Tokenizing text using BERT...")
        df["tkns"] = [' '.join(
            tokenizer2.convert_ids_to_tokens(tokenizer2.encode(
                i,
                add_special_tokens=False,
                truncation=True))) for i in tqdm(df["cleaned_text_wo_sw"])]

        print("Computing Tfidf vectors...")
        tfs = tfidf.fit_transform(tqdm(df['tkns']))
        df["tfs"] = [x for x in tfs]

        print("Reducing Tfidf vectors to 100 dimensions using SVD...")
        tfs_svd = TruncatedSVD(n_components=100,
                               random_state=42).fit_transform(tfs)
        df["tfs_svd"] = [x for x in tfs_svd]

        print("Computing vectors from sentence-bert...")
        df["sbert"] = [sbert.encode(x,
                                        normalize_embeddings=True,
                                        convert_to_numpy=True,
                                        show_progress_bar=False)
                           for x in tqdm(df["cleaned_text"])]

        print("Reducing sentence-BERT vectors to 50 dimensions maximum using PCA...")
        df_temp = pd.DataFrame(
                columns=["V"+str(x) for x in range(512)], data=[x[0:]
                    for x in df["sbert"]])
        pca = PCA(n_components=100, random_state=42)
        res = pca.fit_transform(df_temp)
        df["sbert_PCA"] = [list(x[0:]) for x in res]
        # the index dtype gets somehow turned into float so I turn it back into int :
        temp = df.reset_index()
        temp["cardId"] = temp["cardId"].astype(int)
        df = temp.set_index("cardId")

        print("Combining vectors from tfidf and sentence-BERT into \
the same matrix...")
        df["combo_vec"] = [list(df.loc[x, "sbert_PCA"]) + list(df.loc[x, "tfs_svd"]) for x in df.index]
        self.df = df

    def Kmeans_clustering(self, col = "combo_vec"):
        df = self.df
        print("Clustering using KMeans...")
        kmeans = KMeans(n_clusters=50)
        df_temp = pd.DataFrame( columns=["V"+str(x) for x in range(len(df.loc[df.index[0], col]))], data=[x[0:] for x in df["combo_vec"]])
        df['cluster_kmeans'] = kmeans.fit_predict(df_temp)
        self.df = df.sort_index()

#    def UMAP_embeddings(self):
#        df = self.df
#        df_temp = pd.DataFrame( columns=["V"+str(x) for x in range(len(df.loc[df.index[0], "combo_vec"]))], data=[x[0:] for x in df["combo_vec"]])
#        print("Computing UMAP embeddings in 2D...")
#        df["umap_embed"] = umap.UMAP(
#                n_jobs=-1,
#                verbose=3,
#                n_components=2,
#                metric="euclidean",
#                init="random",
#                random_state=42,
#                transform_seed=42,
#                n_neighbors=5,
#                min_dist=1,
#                n_epochs=500,
#                target_n_neighbors=20).fit_transform(df_temp)
#        self.df = df.sort_index()


    def plot_map(self):
        df = self.df
        print("Reduce to 2 dimensions via PCA before plotting...")
        df_temp = pd.DataFrame( columns=["V"+str(x) for x in range(len(df.loc[df.index[0], "combo_vec"]))], data=[x[0:] for x in df["combo_vec"]])
        pca = PCA(n_components=2, random_state=42)
        res = pca.fit_transform(df_temp).T
        print("Plotting results...")
        fig = px.scatter(df, title="AnnA Anki neuronal Appendix", x = res[0], y = res[1], color=df["cluster_kmeans"], hover_data=["cleaned_text"])
#        fig = px.scatter(df,
#                         title="AnnA Anki neuronal Appendix",
##                         x=[x[0] for x in self.df["umap_embed"]],
##                         y=[x[1] for x in self.df["umap_embed"]],
##                         x=umap_embed[:, 0],
##                         y=umap_embed[:, 1],
#                         color=df["cluster_kmeans"],
#                         hover_data=["cleaned_text"])
        fig.show()

    def _gathering_stopwords(self):
        stops = []
        if self.stop_word_lang != []:
            for lang in self.stop_word_lang:
                try:
                    stops = stops + stopwords.words(lang)
                except Exception as e:
                    print(f"{e}: nltk doesn't seem to have a stopwords list for\
    language '{lang}'")
        else:
            stops = []
        self.stops = stops


    def _clean_text(self):
        "cleaning up text and keeping only relevant fields"
        df = self.df
        rlvt_fields = ["Body", "More", "Header", "value",
                       "Spanish word with article", "English",
                       "Simple example sentences",
                       "Image", "Header", "Extra 1", "Front", "Back"]
        def filterf(field_dic, field_list):
            to_return = ""
            for side in field_dic.keys():
                for a, b in field_dic[side].items():
                    if a in field_list:
                        to_return += self._sanitize_text(b)
            return to_return

        df["cleaned_text"] = [filterf(x, rlvt_fields) for x in tqdm(df["fields"])]
        self.df = df.sort_index()

    def _remove_stop_words(self):
        df = self.df
        def remover(text):
            text = str(text)
            text = [w for w in self.word_finder.findall(text)
                    if w not in self.stops]
            text = ' '.join(text).strip()
            return text

        df["cleaned_text_wo_sw"] = [remover(x) for x in tqdm(df["cleaned_text"])]
        self.df = df.sort_index()

    def _create_and_fill_df(self):
        "fill the dataframe with due cards and rated cards" 

        print("Getting due card from this deck...")
        n_rated_days = int(self.rated_last_X_days)
        due_cards = self._get_card_id_from_query(
                f"deck:{self.deckname} is:due is:review -is:learn -is:suspended -is:buried"
                )

        print(f"Getting cards that where rated in the last {n_rated_days} days from this deck...")
        rated_cards = self._get_card_id_from_query(
                f"deck:{self.deckname} rated:{n_rated_days} -is:suspended"
                )

        combined_card_list = list(rated_cards + due_cards)
        if len(combined_card_list) <50:
            print("You don't have enough due and rated cards!\nExiting.")
            raise SystemExit()

        # removes overlap if found
        for i in due_cards:
            if i in rated_cards:
                rated_cards.remove(i)

        list_cardInfo = []
        df = pd.DataFrame()
        for x in tqdm(combined_card_list, desc="Extracting cards info", unit="Card"):
            list_cardInfo.extend(self._ankiconnect_invoke(action="cardsInfo", cards=[x]))
            if x in due_cards:
                list_cardInfo[-1]["status"] = "due"
            elif x in rated_cards:
                list_cardInfo[-1]["status"] = "rated"
            else:
                list_cardInfo[-1]["status"] = "ERROR"
                print(f"Error processing card with ID {x}")
        for x in list_cardInfo:
            df = df.append(x, ignore_index=True)
        df = df.set_index("cardId")
        self.df = df.sort_index()


    def _request_wrapper(self, action, **params):
        return {'action': action, 'params': params, 'version': 6}

    def _ankiconnect_invoke(self, action, **params):
        "send requests to ankiconnect addon"

        requestJson = json.dumps(self._request_wrapper(action, **params)
                                 ).encode('utf-8')
        if self.verbose is True:
            pprint(requestJson)
        try:
            response = json.load(urllib.request.urlopen(
                                    urllib.request.Request(
                                        'http://localhost:8765',
                                        requestJson)))
        except (ConnectionRefusedError, urllib.error.URLError) as e:
            print(f"{e}: is Anki open and ankiconnect enabled?")
            raise SystemExit()
        if self.verbose is True:
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

    def _get_card_id_from_query(self, query):
        "get notes from query"
        return self._ankiconnect_invoke(action="findCards", query=query)

    def _get_cards_info_from_card_id(self, card_id):
        "get cardinfo from card id, works with either int of list of int"
        if isinstance(card_id, list):
            r_list = []
            for card in tqdm(card_id):
                r_list.extend(self._ankiconnect_invoke(action="cardsInfo",
                              cards=[card]))
            return r_list
        if isinstance(card_id, int):
            return self._ankiconnect_invoke(action="cardsInfo",
                                            cards=[card_id])


    def _sanitize_text(self, text):
        text = str(text)
        if self.keep_ocr is True:
            # extract title of images (usually OCRed text) before html is removed
            text = re.sub("title=(\".*?\")", "> OCR \\1 <", text)
        if self.replace_greek is True:
            # https://gist.github.com/beniwohli/765262
            import greek_alphabet_mapping
            for a, b in greek_alphabet_mapping.greek_alphabet.items():
                text = re.sub(a, b, text)
        if self.replace_acronym is True:
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

    def _check_deck(self, deckname=None):
        """
        getting correct deck name
        """
        decklist = self._ankiconnect_invoke(action="deckNames")
        if deckname is not None:
            if deckname not in decklist:
                print("Couldn't find this deck.")
                deck = None
        if deckname is None:
            auto_complete = WordCompleter(decklist,
                                          match_middle=True,
                                          ignore_case=True)
            deckname = ""
            while deck not in decklist:
                deckname = prompt("Enter the name of the deck to use:\n>",
                                  completer=auto_complete)
        return deckname





##################################################
# TODO
#def add_tag_to_card_id(card_id, tag):
#    "add tag to card id"
#    # first gets note it from card id
#    note_id = self._ankiconnect_invoke(action="cardsToNote", cards=card_id)
#    return self._ankiconnect_invoke(action="addTags",
#                              notes=note_id,
#                              tags=tag)
#
#
#def add_vectorTags(note_dic, vectorizer, vectors):
#    "adds a tag containing the vector values to each note"
#    if vectorizer not in ["sentence_bert", "tf-idf"]:
#        print("Wrong vectorTags vectorizer")
#        raise SystemExit()
#    return self._ankiconnect_invoke(action="addTags",
#                              notes=note_id,
#                              tags=vectors_tag)
#
#
#def add_cluterTags(note_dic):
#    "add a tag containing the cluster number and name to each note"
#    pass
#
#
#def add_actionTags(note_dic):
#    """
#    add a tag containing the action that should be taken regarding cards.
#    The action can be "bury" or "study_today"
#    Currently, you then have to manually bury them or study them into anki
#    """
#
#def get_cluster_topic(cluster_note_dic, all_note_dic, cluster_nb):
#    "given notes, outputs the topic that is likely the subject of the cards"
#    pass
#
#
#def find_similar_notes(note):
#    "given a note, find similar other notes with highest cosine similarity"
#    pass
#
#
#def find_notes_similar_to_input(user_input, nlimit):
#    "given a text input, find notes with highest cosine similarity"
#    pass
#
#
#def show_latent_space(query):
#    """
#    given a query, will open plotly and show a 2d scatterplot of your cards
#    semantically arranged.
#    """
#    # TODO test if the cards contain the relevant tags otherwise exit

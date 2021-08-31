import random
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
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
import transformers
import scipy.sparse as sp
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import paired_cosine_distances
from sklearn.decomposition import PCA
import plotly.express as px
import umap.umap_
import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "true"

# makes the script interuptible, resume it using c+enter
signal.signal(signal.SIGINT, (lambda signal, frame: pdb.set_trace()))


# I put those three lines here because they are long to run
# hence I don't want to rerun them at each new instance of AnnA
print("Loading tokenizers...")
tokenizer_bert = transformers.BertTokenizerFast.from_pretrained(
        'bert-base-multilingual-uncased'
        )
sbert = SentenceTransformer('distiluse-base-multilingual-cased-v1')
# this variable contains as key a notetype model name and as value the
# name of the fields to keep, in order
# ex : "basic": ["Front", "Back"]
field_dic = {"clozolkor": ["Header", "Body"],
             "occlusion": ["Header", "Image"]
             }


class AnnA:
    def __init__(self,
                 deckname="",
                 verbose=False,
                 replace_greek=True,
                 replace_acronym=False,
                 stop_word_lang=["en"],
                 keep_ocr=False,
                 rated_last_X_days=4,
                 show_banner=True,
                 SVD_dim=10,
                 PCA_dim=10,
                 n_cluster=8):
        # printing banner
        if show_banner is True:
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
        self.SVD_dim = SVD_dim
        self.PCA_dim = PCA_dim
        self.n_cluster = n_cluster

        # loading backend stuf
        self.stop_words = self._gathering_stopwords()
        self.tfidf = TfidfVectorizer(
                                tokenizer=self._tokenizer_wrap,
                                analyzer="word",
                                norm="l2",
                                strip_accents="ascii",
                                lowercase=True,
                                stop_words=self.stop_words
                                )
        self.TSVD = TruncatedSVD(n_components=self.SVD_dim, random_state=42)
        self.pca = PCA(n_components=self.PCA_dim, random_state=42)
        self.pca_disp = PCA(n_components=2, random_state=42)

        # actual execution
        self.deckname = self._check_deck(deckname)
        self._create_and_fill_df()
        self._reset_index_dtype()
        self._format_card()
        self._reset_index_dtype()
        self._vectors()
        self._reset_index_dtype()
        #self._distances()
        self.do_clustering()
        self.show_latent_space()

    def _gathering_stopwords(self):
        "store the completed list of stopwords to self.stops"
        stops = []
        if self.stop_word_lang != []:
            for lang in self.stop_word_lang:
                try:
                    stops = stops + stopwords.words(lang)
                except Exception as e:
                    print(f"{e}: nltk doesn't seem to have a stopwords list for language '{lang}'")
        else:
            stops = []
        self.stops = list(set(stops))

    def _reset_index_dtype(self):
        """
        the index dtype (cardId) somehow gets turned into float so I turn it back into int
        """
        temp = self.df.reset_index()
        temp["cardId"] = temp["cardId"].astype(int)
        self.df = temp.set_index("cardId")

    def _ankiconnect_invoke(self, action, **params):
        "send requests to ankiconnect addon"

        def request_wrapper(action, **params):
            return {'action': action, 'params': params, 'version': 6}

        requestJson = json.dumps(request_wrapper(action, **params)
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

    def _create_and_fill_df(self):
        "fill the dataframe with due cards and rated cards"

        print("Getting due card from this deck...")
        n_rated_days = int(self.rated_last_X_days)
        query = f"deck:{self.deckname} is:due is:review -is:learn -is:suspended -is:buried"
        due_cards = self._get_card_id_from_query(query)

        print(f"Getting cards that where rated in the last {n_rated_days} days from this deck...")
        query = f"deck:{self.deckname} rated:{n_rated_days} -is:suspended"
        rated_cards = self._get_card_id_from_query(query)

        combined_card_list = list(rated_cards + due_cards)
        if len(combined_card_list) < 50:
            print("You don't have enough due and rated cards!\nExiting.")
            raise SystemExit()

        # removes overlap if found
        for i in due_cards:
            if i in rated_cards:
                rated_cards.remove(i)

        list_cardInfo = []
        df = pd.DataFrame()

        n = len(due_cards + rated_cards)
        print(f"Query Anki for information about {n} cards (takes ~{n//500}s) ...")
        list_cardInfo.extend(self._ankiconnect_invoke(action="cardsInfo",
                             cards=combined_card_list))

        for i, card in enumerate(list_cardInfo):
            # removing large fields:
            list_cardInfo[i].pop("question")
            list_cardInfo[i].pop("answer")
            list_cardInfo[i].pop("css")
            list_cardInfo[i].pop("fields_no_html")
            if card["cardId"] in due_cards:
                list_cardInfo[i]["status"] = "due"
            elif card["cardId"] in rated_cards:
                list_cardInfo[i]["status"] = "rated"
            else:
                list_cardInfo[i]["status"] = "ERROR"
                print(f"Error processing card with ID {card['cardId']}")

        for x in list_cardInfo:
            df = df.append(x, ignore_index=True)
        # removing the largest and useless columns
        df = df.set_index("cardId")
        self.df = df.sort_index()

    def _format_text(self, text):
        "text preprocessor"
        text = str(text)
        if self.keep_ocr is True:
            # keep image title (usually OCR)
            text = re.sub("title=(\".*?\")", "> Image: \\1. <", text)
        if self.replace_greek is True:
            # https://gist.github.com/beniwohli/765262
            import greek_alphabet_mapping
            for a, b in greek_alphabet_mapping.greek_alphabet.items():
                text = re.sub(a, b, text)
        if self.replace_acronym is True:
            import user_acronym_list
            for a, b in user_acronym_list.acronym_list.items():
                text = re.sub(a, b, text)
        text = re.sub(r'[a-zA-Z0-9-]+\....', " ", text)  # media file name
        text = re.sub("\d", "", text)
        text = re.sub('\\n|<div>|</div>|<br>|<span>|</span>|<li>|</li>|<ul>|</ul>',
                      " ", text)  # newline
        text = re.sub("<a href.*?</a>", " ", text)  # html links
        text = re.sub(r'http[s]?://\S*', " ", text)  # plaintext links
        text = re.sub("<.*?>", " ", text)  # remaining html tags
        text = re.sub('\u001F|&nbsp;', " ", text)
        text = re.sub(r"{{c\d+?::", "", text)
        text = re.sub("}}|::", "", text)
        text = text.replace("&gt;", ">")
        text = text.replace("&l;;", "<")
        return text.strip()

    def _format_card(self):
        "keep only relevant field of card then clean text"
        df = self.df

        for index in tqdm(df.index, desc="Processing text", unit="card"):
            card_model = df.loc[index, "modelName"]
            take_first_field = False
            fields_to_keep = []

            cnt=0
            for user_model in field_dic.keys():
                if user_model.lower() in card_model.lower():
                    cnt+=1
                    target_model = user_model
            if cnt == 0:
                take_first_field = True
            elif cnt == 1:
                fields_to_keep = field_dic[target_model]
            elif cnt > 1:
                tqdm.write(f"Several corresponding model found!\
Edit the variable 'field_dic' to use {card_model}")
                take_first_field = True

            if take_first_field is True:
                field_list = list(df.loc[index, "fields"])
                for f in field_list:
                    order = df.loc[index,"fields"][f]["order"]
                    if order == 0:
                        break
                fields_to_keep = [f]

            comb_text = ""
            for f in fields_to_keep:
                comb_text = comb_text + df.loc[index, "fields"][f]["value"] + ": "
            comb_text = comb_text.strip()[:-2].replace(" : ", ": ")
            if comb_text.startswith(": "):
                comb_text = comb_text[2:]
            df.loc[index, "comb_text"] = comb_text
        df["text"] = [self._format_text(x) for x in tqdm(df["comb_text"])]
        self.df = df.sort_index()

    def _tokenizer_wrap(self, string):
        "just a wrapper to pass arguments to the tokenizer"
        return tokenizer_bert.tokenize(string,
                                       add_special_tokens=False,
                                       truncation=True)

    def _vectors(self):
        """
        Assigne vectors to each card
        df["tfs"] contains tf-idf vectors
        df["tfs_svd"] contains tf-idf vectors after SVD dim reduction
        df["sbert"] contains sentencebert vectors
        df["sbert_pca"] contains sentencebert vectors after PCA dim reduction
        df["combo_vec"] contains sbert_pca next to tfs_svd
        """
        df = self.df

        print("Computing Tfidf vectors...")
        #tfs = self.tfidf.fit_transform(tqdm(df['tkns']))
        tfs = self.tfidf.fit_transform(tqdm(df['text']))
        print(f"Reducing Tfidf vectors to {self.SVD_dim} dimensions using SVD...")
        tfs2 = self.TSVD.fit_transform(tfs)

        df["tfs"] = [x for x in tfs]
        df["tfs_svd"] = [x for x in tfs2]



        print("Computing vectors from sentence-bert...")
        sentence_list = [x for x in df["text"]]
        df["sbert"] = [x for x in sbert.encode(sentence_list,
                                               normalize_embeddings=True,
                                               show_progress_bar=True)]

        print(f"Reducing sentence-BERT vectors to {self.PCA_dim} dimensions using PCA...")
        # temporary datarame that holds the vectors of sbert to feed it to the PCA engine
        df_temp = pd.DataFrame(
                               columns=[f"V{x}" for x in range(len(df.loc[df.index[0], "sbert"]))],
                               data=[x[0:] for x in df["sbert"]]
                               )
        df["sbert_pca"] = [list(x[0:]) for x in self.pca.fit_transform(df_temp)]


        print("Concatenating vectors from tfidf and sentence-BERT...")
        df["combo_vec"] = [ list(df.loc[x, "sbert_pca"]) + list(df.loc[x, "tfs_svd"])
                           for x in df.index]
        self.df = df.sort_index()

    def _distances(self):
        "compute distance matrix between cards"
        df = self.df
        self.df = df

    def do_clustering(self,
            method="DBSCAN",
            input_col = "combo_vec",
            output_col="clusters",
            **kwargs):
        "perform clustering over some column"
        df = self.df
        if method == "kmeans":
            clust = KMeans(n_clusters=self.n_cluster,
                    **kwargs)
        elif method == "DBSCAN":
            clust = DBSCAN(eps=0.5,
                    min_samples=3,
                    n_jobs=-1,
                    **kwargs
                    )
        elif method.lower() in "agglomerative":
            clust = AgglomerativeClustering(
                        n_clusters=self.n_cluster,
                        #distance_threshold=0.1,
                        affinity="cosine",
                        memory="/tmp/",
                        linkage="average",
                        **kwargs)

        print(f"Clustering using {method}...")
        df_temp = pd.DataFrame(
            columns=["V"+str(x) for x in range(len(df.loc[df.index[0], input_col]))],
            data=[x[0:] for x in df[input_col]])
        df[output_col] = clust.fit_predict(df_temp)

        cluster_list = list(set(list(df[output_col])))
        cluster_nb = len(cluster_list)
        print(f"Getting cluster topics for {cluster_nb} clusters...")
        
        df_by_cluster = df.groupby(["clusters"], as_index=False).agg({'text': ' '.join})
        count = CountVectorizer().fit_transform(df_by_cluster.text)
        ctfidf = CTFIDFVectorizer().fit_transform(count, n_samples=len(df_by_cluster.index))
        count_vectorizer = CountVectorizer().fit(df_by_cluster.text)
        count = count_vectorizer.transform(df_by_cluster.text)
        words = count_vectorizer.get_feature_names()
        ctfidf = CTFIDFVectorizer().fit_transform(count, n_samples=len(df_by_cluster.index)).toarray()
        words_per_class = {str(label): [
                                words[index] for index in ctfidf[label].argsort()[-5:]
                                ] for label in df_by_cluster.clusters}
        df["cluster_topic"] = ""
        for i in df.index:
            df.loc[i, "cluster_topic"] = " ".join([x for x in words_per_class[str(df.loc[i, "clusters"])]])

        self.df = df.sort_index()


    def show_latent_space(self,
                 reduce_dim="umap",
                 color_col="clusters",
                 coordinate_col="combo_vec"):
        "display a graph showing the cards spread out into 2 dimensions"
        df = self.df

        if reduce_dim is not None:
            df_temp = pd.DataFrame(
                columns=["V"+str(x) for x in range(len(df.loc[df.index[0], coordinate_col]))],
                data=[x[0:] for x in df[coordinate_col]])
            print(f"Reduce to 2 dimensions via {reduce_dim} before plotting...")
        if reduce_dim.lower() in "pca":
            res = self.pca_disp.fit_transform(df_temp).T
            x_coor = res[0]
            y_coor = res[1]
        elif reduce_dim.lower() in "umap":
                res = umap.UMAP(n_jobs=-1,
                                verbose=2,
                                n_components=2,
                                metric="cosine",
                                init="random",
                                random_state=42,
                                transform_seed=42,
                                n_neighbors=5,
                                min_dist=1,
                                n_epochs=500,
                                target_n_neighbors=5).fit_transform(df_temp)
                x_coor = res.T[0]
                y_coor = res.T[1]
        elif reduce_dim is None:
            x_coor = [x[0] for x in df[coordinate_col]],
            x_coor = list(x_coor)[0]
            y_coor = [x[1] for x in df[coordinate_col]],
            y_coor = list(y_coor)[0]
        print("Plotting results...")
        fig = px.scatter(df,
                         title="AnnA Anki neuronal Appendix",
                         x = x_coor,
                         y = y_coor,
                         color=color_col,
                         hover_data=["text", "cluster_topic"])
        fig.show()

    def find_notes_similar_to_input(self, user_input, nlimit=100):
        "given a text input, find notes with highest cosine similarity"
        embed = sbert.encode(user_input)
        df = self.df
        dist = {}
        for i in df.index:
            dist.update({i: paired_cosine_distances(embed.reshape(1, -1), df.loc[i, "sbert"].reshape(1, -1))})
        if len(dist.keys()) == 0:
            print("No cards found")
        else:
            print(f"Found {len(dist.keys())} cards:")
            index = dist.keys()
            index = sorted(index, key=lambda row: int(index[row]), reverse=True)
            print(df.loc[index, "text"])
        return True


    def find_similar_card(self, card_id, field_name):
        "given a card_id, find similar other notes with highest cosine similarity"
        info = self._get_cards_info_from_card_id(card_id)
        text = info[0]['fields_no_html'][field_name]
        self.find_notes_similar_to_input(text)


class CTFIDFVectorizer(TfidfTransformer):
    "source: https://towardsdatascience.com/creating-a-class-based-tf-idf-with-scikit-learn-caea7b15b858"
    def __init__(self, *args, **kwargs):
        super(CTFIDFVectorizer, self).__init__(*args, **kwargs)

    def fit(self, X: sp.csr_matrix, n_samples: int):
        """Learn the idf vector (global term weights) """
        _, n_features = X.shape
        df = np.squeeze(np.asarray(X.sum(axis=0)))
        idf = np.log(n_samples / df)
        self._idf_diag = sp.diags(idf, offsets=0,
                                  shape=(n_features, n_features),
                                  format='csr',
                                  dtype=np.float64)
        return self

    def transform(self, X: sp.csr_matrix) -> sp.csr_matrix:
        """Transform a count-based matrix to c-TF-IDF """
        X = X * self._idf_diag
        X = normalize(X, axis=1, norm='l1', copy=False)
        return X



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

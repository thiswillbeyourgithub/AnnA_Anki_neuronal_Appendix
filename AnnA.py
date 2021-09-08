import time
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
from pathlib import Path
import threading
from sklearn.feature_extraction.text import TfidfTransformer
import scipy.sparse as sp
import numpy as np

# avoids annoying warning
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# makes the script interuptible, resume it using c+enter
signal.signal(signal.SIGINT, (lambda signal, frame: pdb.set_trace()))


# This variable contains as key a notetype model name and as value the
# name of the fields to keep, in order
# ex : "basic": ["Front", "Back"]
field_dic = {
             "clozolkor": ["Header", "Body"],
             "occlusion": ["Header", "Image"],
             "spanish cards": ["Spanish", "English"]
             }

def asynchronous_importer():
    "used to asynchroneously import the modules, speeds up launch time"
    global stopwords, SentenceTransformer, KMeans, DBSCAN, \
        AgglomerativeClustering, transformers, sp, normalize, TfidfVectorizer,\
        CountVectorizer, TruncatedSVD, StandardScaler, \
        pairwise_distances, PCA, px, umap, np, tokenizer_bert, sbert
    print("Importing modules.")
    from nltk.corpus import stopwords
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sentence_transformers import SentenceTransformer
    sbert = SentenceTransformer('distiluse-base-multilingual-cased-v1')
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    import plotly.express as px
    import umap.umap_
    from sklearn.metrics import pairwise_distances
    from sklearn.preprocessing import normalize
    from sklearn.decomposition import TruncatedSVD
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    print("Finished importing modules")



class AnnA:
    def __init__(self,
                 deckname=None,
                 verbose=False,
                 replace_greek=True,
                 replace_acronym=True,
                 keep_ocr=False,
                 desired_deck_size=500,
                 rated_last_X_days=7,
                 show_banner=True,
                 card_limit=None,
                 n_clusters=None,
                 pca_sbert_dim=300,  # should keep about 99% of variance
                 ):
        # printing banner
        if show_banner is True:
            ascii_banner = pyfiglet.figlet_format("AnnA")
            print(ascii_banner)
            print("(Anki neuronal Appendix)\n\n")

        # loading args etc
        if deckname is None:
            deckname = "*"
        self.deckname = deckname
        self.verbose = verbose
        self.replace_acronym = replace_acronym
        self.replace_greek = replace_greek
        self.keep_ocr = keep_ocr
        self.desired_deck_size = desired_deck_size
        self.rated_last_X_days = rated_last_X_days
        self.card_limit = card_limit
        self.n_clusters = n_clusters
        self.pca_sbert_dim = pca_sbert_dim

        # loading backend stuf
        if self.replace_acronym is True:
            from user_acronym_list import acronym_list
            self.acronym_list = acronym_list
        if self.replace_greek is True:
            from greek_alphabet_mapping import greek_alphabet
            self.greek_alphabet = greek_alphabet
        import_thread.join()  # asynchroneous importing of large module
        time.sleep(1)
        self.pca_sbert = PCA(n_components=self.pca_sbert_dim, random_state=42)
        self.pca_2D = PCA(n_components=2, random_state=42)
        self.scaler = StandardScaler()

        # actual execution
        self.deckname = self._check_deck(deckname)
        self._create_and_fill_df()
        self.df = self._reset_index_dtype(self.df)
        self._format_card()
        self._vectors()
        self._compute_distance_matrix()
        self.assign_scoring()

    def _reset_index_dtype(self, df):
        """
        the index dtype (cardId) somehow gets turned into float so I turn it back into int
        """
        temp = df.reset_index()
        temp["cardId"] = temp["cardId"].astype(int)
        df = temp.set_index("cardId")
        return df

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
            if len(card_id) < 300:
                r_list = []
                for card in tqdm(card_id):
                    r_list.extend(self._ankiconnect_invoke(action="cardsInfo",
                                  cards=[card]))
                return r_list

            else:
                lock = threading.Lock()
                threads = []
                cnt = 0
                r_list = []
                target_thread_n = 10
                batchsize = len(card_id)//target_thread_n+3
                print(f"Large number of cards to retrieve, creating 10 threads of {batchsize} cards...")

                def retrieve_cards(card_list, lock, cnt, r_list):
                    "for multithreaded card retrieval"
                    out_list = self._ankiconnect_invoke(action="cardsInfo",
                                  cards=card_list)
                    lock.acquire()
                    r_list.extend(out_list)
                    #tqdm.write(f"Thread #{cnt} finished in {int(time.time()-start)}s")
                    pbar.update(1)
                    lock.release()
                    return True


                with tqdm(total=target_thread_n,
                          unit="thread",
                          dynamic_ncols=True,
                          desc="Finished threads",
                          delay=2,
                          smoothing=0) as pbar:
                    for nb in range(0, len(card_id), batchsize):
                        cnt += 1
                        temp_card_id = card_id[nb: nb+batchsize]
                        thread = threading.Thread(target=retrieve_cards,
                                                  args=(temp_card_id, lock, cnt, r_list),
                                                  daemon=True)
                        thread.start()
                        threads.append(thread)
                        time.sleep(0.1)
                    print("")
                    for t in threads:
                        t.join()
                assert len(r_list) == len(card_id)
                r_list = sorted(r_list, key= lambda x: x["cardId"], reverse=False)
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
        "create and fill the dataframe with due cards and rated cards"

        print("Getting due card from this deck...")
        n_rated_days = int(self.rated_last_X_days)
        query = f"deck:{self.deckname} is:due is:review -is:learn -is:suspended -is:buried -is:new -rated:1"
        print(query)
        due_cards = self._get_card_id_from_query(query)

        print(f"Getting cards that where rated in the last {n_rated_days} days from this deck...")
        query = f"deck:{self.deckname} rated:{n_rated_days} -is:suspended"
        print(query)
        rated_cards = self._get_card_id_from_query(query)

        # removes overlap if found
        rated_cards = [x for x in rated_cards if x not in due_cards]

        if self.card_limit is None:
            combined_card_list = list(rated_cards + due_cards)
        else:
            combined_card_list = list(rated_cards + due_cards)[0:self.card_limit]
        if len(combined_card_list) < 50:
            print("You don't have enough due and rated cards!\nExiting.")
            raise SystemExit()


        list_cardInfo = []

        n = len(combined_card_list)
        print(f"Asking Anki for information about {n} cards...")
        start = time.time()
        list_cardInfo.extend(self._get_cards_info_from_card_id(card_id=combined_card_list))
        print(f"Extracted information in {int(time.time()-start)} seconds.")

        for i, card in enumerate(tqdm(list_cardInfo, desc="Filtering only relevant fields...", unit="card")):
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

        if len(list_cardInfo) != len(list(set(combined_card_list))):
            print("Duplicate elements!\nExiting.")
            pdb.set_trace()


        self.df = pd.DataFrame().append(list_cardInfo,
                                   ignore_index=True,
                                   sort=True).set_index("cardId").sort_index()

    def _format_text(self, text):
        "text preprocessor, called by _format_card on each card content"
        text = str(text)
        if self.keep_ocr is True:
            # keep image title (usually OCR)
            text = re.sub("title=(\".*?\")", "> Image: \\1. <", text)
        if self.replace_greek is True:
            global greek_alphabet
            for a, b in self.greek_alphabet.items():
                text = re.sub(a, b, text)
        if self.replace_acronym is True:
            global acronym_list
            for a, b in self.acronym_list.items():
                text = re.sub(rf"\b{a}\b", b, text)  # \b matches beginning and end of a word
        text = re.sub(r'[a-zA-Z0-9-]+\....', " ", text)  # media file name
        text = re.sub('\\n|<div>|</div>|<br>|<span>|</span>|<li>|</li>|<ul>|</ul>',
                      " ", text)  # newline
        text = re.sub("<a href.*?</a>", " ", text)  # html links
        text = re.sub(r'http[s]?://\S*', " ", text)  # plaintext links
        text = re.sub("<.*?>", " ", text)  # remaining html tags
        text = re.sub('\u001F|&nbsp;', " ", text)
        text = re.sub(r"{{c\d+?::", "", text)
        text = re.sub("{{c|{{|}}|::", " ", text)
        text = re.sub("\d", " ", text)
        text = text.replace("&gt;", ">")
        text = text.replace("&l;;", "<")
        text = " ".join(text.split())  # multiple spaces
        text = text.strip()
        return text

    def _format_card(self):
        "keep only relevant field of card then clean text"
        df = self.df

        for index in tqdm(df.index, desc="Parsing text content", unit="card"):
            card_model = df.loc[index, "modelName"]
            take_first_field = False
            fields_to_keep = []

            # determines which is the corresponding model described in field_dic
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

            # concatenates the corresponding fields into one string:
            if take_first_field is True:  # case where no corresponding model 
                # found in field_dic
                field_list = list(df.loc[index, "fields"])
                for f in field_list:
                    order = df.loc[index, "fields"][f]["order"]
                    if order == 0:
                        break
                fields_to_keep = [f]

            comb_text = ""
            for f in fields_to_keep:
                to_add = df.loc[index, "fields"][f]["value"].strip()
                if to_add != "":
                    comb_text = comb_text + to_add + " "
            df.loc[index, "comb_text"] = comb_text.strip().replace(": :", "").strip()
        df["text"] = [self._format_text(x) for x in tqdm(df["comb_text"], desc="Formating text")]
        print("\n\n5 random samples of your formated text, to help troubleshoot formating issues:")
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)
        for i in range(5):
            print(df.sample(1)["text"], end="\n\n")
        pd.reset_option("display.max_rows")
        pd.reset_option('display.max_columns')
        pd.reset_option('display.width')
        pd.reset_option('display.max_colwidth')
        print("\n\n")
        self.df = df.sort_index()

    def _vectors(self, df=None, use_sbert_cache=True):
        """
        Assigne vectors to each card
        df["sbert"] contains sentencebert vectors, eventually after PCA
        df["sbert_before_pca"] if exists, then it's the full 512 vectors
        """
        if df is None:
            df = self.df

        if use_sbert_cache is True:
            print("\nLooking for cached sentence-bert pickle file...", end="")
            sbert_file = Path("./sbert_cache.pickle")
            df["sbert"] = 0*len(df.index)
            df["sbert"] = df["sbert"].astype("object")
            loaded_sbert = 0
            index_to_recompute = []

            # reloads sbert vectors and only recomputes the new one:
            if not sbert_file.exists():
                print(" sentence-bert cache not found, will create it.")
                df_cache = pd.DataFrame(columns=["cardId", "mod", "text", "sbert"]).set_index("cardId")
                index_to_recompute = df.index
            else:
                print(" Found sentence-bert cache.")
                df_cache = pd.read_pickle(sbert_file)
                df_cache["sbert"] = df_cache["sbert"].astype("object")
                df_cache["mod"] = df_cache["mod"].astype("object")
                df_cache["text"] = df_cache["text"]
                df_cache = self._reset_index_dtype(df_cache)
                for i in df.index:
                    if i in df_cache.index and \
                            (str(df_cache.loc[i, "mod"]) == str(df.loc[i, "mod"])) and \
                            (str(df_cache.loc[i, "text"]) == str(df.loc[i, "text"])):
                        df.at[i, "sbert"] = df_cache.loc[i, "sbert"].astype("object")
                        loaded_sbert += 1
                    else:
                        index_to_recompute.append(i)

            print(f"Loaded {loaded_sbert} vectors from cache, will compute {len(index_to_recompute)} others...")
            if len(index_to_recompute) != 0:
                sentence_list = [df.loc[x, "text"]
                        for x in df.index if x in index_to_recompute]
                sentence_embeddings = sbert.encode(sentence_list,
                                                   normalize_embeddings=True,
                                                   show_progress_bar=True)

                for i, ind in enumerate(tqdm(index_to_recompute)):
                    df.at[ind, "sbert"] = sentence_embeddings[i]

            # stores newly computed sbert vectors in a file:
            df_cache = self._reset_index_dtype(df_cache)
            for i in [x for x in index_to_recompute if x not in df_cache.index]:
                df_cache.loc[i, "sbert"] = df.loc[i, "sbert"].astype("object")
                df_cache.loc[i, "mod"] = df.loc[i, "mod"].astype("object")
                df_cache.loc[i, "text"] = df.loc[i, "text"]
            for i in [x for x in index_to_recompute if x in df_cache.index]:
                df_cache.loc[i, "sbert"] = df.loc[i, "sbert"].astype("object")
                df_cache.loc[i, "mod"] = df.loc[i, "mod"].astype("object")
                df_cache.loc[i, "text"] = df.loc[i, "text"]
            df_cache = self._reset_index_dtype(df_cache)
            df_cache.to_pickle(f"sbert_cache.pickle")

        if self.pca_sbert_dim is not None:
            print(f"Reducing dimension of sbert to {self.pca_sbert_dim} using PCA...")
            df_temp = pd.DataFrame(
                columns=["V"+str(x+1) for x in range(len(df.loc[df.index[0], "sbert"]))],
                data=[x[0:] for x in df["sbert"]])
            out = self.pca_sbert.fit_transform(df_temp)
            print(f"Explained variance ratio after PCA on SBERT: {round(sum(self.pca_sbert.explained_variance_ratio_)*100,1)}%")
            df["sbert_before_pca"] = df["sbert"]
            df["sbert"] = [x for x in out]

    def _compute_distance_matrix(self, method="cosine", input_col="sbert"):
        """
        compute distance matrix between cards, given that L2 norm is used throughout the 
        script, cosine is not used but np.dot is used instead.
        """
        print("Computing the distance matrix...", end="")
        df = self.df

        df_temp = pd.DataFrame(
            columns=["V"+str(x+1) for x in range(len(df.loc[df.index[0], input_col]))],
            data=[x[0:] for x in df[input_col]])
        df_dist = pairwise_distances(df_temp, n_jobs=-1, metric=method)

        print(" Done.")
        self.df_dist = df_dist
        self.df = df


    def assign_scoring(self, reference_order="lowest_interval"):
        """
        assign scoring to each card, the score reflects the order in which 
        they should be reviewed to minimize useless reviews.
        The score is computed according to formula:
            score = interval + median of the proximity to each card of the queue
        Reference_order can either be "lower_interval" or "relative_overdueness"
        """
        print("Assigning scores...")
        df = self.df
        df_dist = self.df_dist
        queue_size_goal = self.desired_deck_size
        df["ivl_std"] = self.scaler.fit_transform(df["interval"].to_numpy().reshape(-1, 1))

        if reference_order != "lowest_interval":
            print("Using another reference than lowest interval is not yet supproted")
            reference_order = "lowest_interval"

        rated = [x for x in df.index if df.loc[x, "status"] == "rated"]
        queue = []
        print(f"Already rated in the past relevant days: {len(rated)}")


        if len(rated) == 0:
            if reference_order == "lowest_interval":
                queue.append(df["ivl_std"].idxmin())

        cnt = 0
        df_temp = pd.DataFrame(columns=rated, index=df.index)
        with tqdm(desc="Finding optimal review order", unit="Card", smoothing=0, total=queue_size_goal) as pbar:
            while len(queue) < queue_size_goal:
                cnt += 1
                if cnt > 500000:
                    print("Detected potential endeless loop. Exiting.")
                    break

                for q in rated + queue:
                    df_temp[q] = df_dist[df.index.get_loc(q)]
                scaled_median = self.scaler.fit_transform(
                        np.median(df_temp, axis=1).reshape(-1, 1))
                df["scaled_median_dist_to_queue"] = scaled_median
                df["score"] = df["ivl_std"] - df["scaled_median_dist_to_queue"]
                chosen_one = df.drop(labels=queue)["score"].idxmin()
                queue.append(chosen_one)
                df_temp[chosen_one] = df_dist[df.index.get_loc(chosen_one)]
                pbar.update(1)

        print("Done. Now all that is left is to send all of this to anki.\n")
        self.best_review_order = queue

    def to_anki(self, template_name=f"AnnA - Optimal Review Order"):
        """
        add a tag to the queue cards then
        orders the creation of a filtered deck filtering by this tag
        then manually alter the order of the review in the deck to 
        match self.best_review_order
        """

        filtered_deck_name = str(template_name + f" - {self.deckname}").replace("::", "_")
        if filtered_deck_name in self._ankiconnect_invoke(action="deckNames"):
            input(f"Deck '{filtered_deck_name}' already exists. Make sure to delete this deck before continuing.\nDone? (y/n) >")
        tag_name = f"AnnA_Optimal_review_order::{self.deckname.replace('::', '_')}::session_{'_'.join(time.asctime().split()[0:3])}"

        # first remove the tag if present:
        tag_list = self._ankiconnect_invoke(action="getTags")
        if tag_name in tag_list:
            note_list = self._ankiconnect_invoke(action="findNotes", query=f"\"tag:{tag_name}f\"")
            self._ankiconnect_invoke(action="removeTags", notes=note_list, tags=tag_name)
            self._ankiconnect_invoke(action="clearUnusedTags")
            print(f"Removed tag that was already present: {tag_name}")

        note_list = list(set([int(self.df.loc[x, "note"]) for x in self.best_review_order]))
        self._ankiconnect_invoke(action="addTags", notes=note_list, tags=tag_name)
        print(f"Added tag: {tag_name} to all the notes from best_review_order.")

        self._ankiconnect_invoke(action="createFilteredDeck",
                                 newDeckName=filtered_deck_name,
                                 searchQuery=f'tag:{tag_name}',
                                 gatherCount=len(self.best_review_order),
                                 reschedule=True,
                                 sortOrder=0,
                                 createEmpty=False)
        # sortOrder = 0 is "oldest seen first", this way I can see if something is fishy if the deck rebuilded itself
        print(f"Created deck {filtered_deck_name}")
        self.filtered_deck_name = filtered_deck_name

        # checks that the content of filtered deck name is the same as best_review_order
        currently_in_deck = self._ankiconnect_invoke(action="findCards",
                query=f"\"deck:{filtered_deck_name}\"")
        diff = [x for x in self.best_review_order + currently_in_deck if x not in self.best_review_order or x not in currently_in_deck]
        if len(diff) != 0:
            print("Inconsistency! The deck does not contain the same cards as best_review_order!") 
            pdb.set_trace()

        incrementer = -10*len(self.best_review_order)
        for c in tqdm(self.best_review_order, desc="Altering due order", unit="card"):
            incrementer += 1
            self._ankiconnect_invoke(action="setSpecificValueOfCard",
                                     card=int(c),
                                     keys=["due"],
                                     newValues=[incrementer])
        print("All done!\n\n")


    def rechange_due_order(self):
        "reassign the due order without having the re-run the whole script"
        # checks that the content of filtered deck name is the same as best_review_order
        currently_in_deck = self._ankiconnect_invoke(action="findCards",
                query=f"\"deck:{self.filtered_deck_name}\"")
        diff = [x for x in self.best_review_order + currently_in_deck if x not in self.best_review_order or x not in currently_in_deck]
        if len(diff) != 0:
            print("Inconsistency! The deck does not contain the same cards as best_review_order!") 
            pdb.set_trace()
        incrementer = -len(self.best_review_order)-1
        for c in tqdm(self.best_review_order, desc="Altering due order", unit="card"):
            incrementer += 1
            self._ankiconnect_invoke(action="setSpecificValueOfCard",
                                     card=int(c),
                                     keys=["due"],
                                     newValues=[incrementer])
        print("All done!\n\n")

    def compute_clusters(self,
                      method="kmeans",
                      input_col="sbert",
                      output_col="clusters",
                      **kwargs):
        "perform clustering over a given column"
        df = self.df
        if self.n_clusters is None:
            self.n_clusters = len(df.index)//10
        if method == "kmeans":
            clust = KMeans(n_clusters=min(self.n_clusters, 100),
                    **kwargs)
        elif method == "DBSCAN":
            clust = DBSCAN(eps=0.75,
                           min_samples=3,
                           n_jobs=-1,
                           **kwargs
                           )
        elif method.lower() in "agglomerative":
            clust = AgglomerativeClustering(
                        n_clusters=self.n_clusters,
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
                 coordinate_col="sbert"):
        "display a graph showing the cards spread out into 2 dimensions"
        df = self.df
        if "clusters" not in df.columns:
            df["clusters"] = 0
        if "cluster_topic" not in df.columns:
            df["cluster_topic"] = 0

        if reduce_dim is not None:
            df_temp = pd.DataFrame(
                columns=["V"+str(x) for x in range(len(df.loc[df.index[0], coordinate_col]))],
                data=[x[0:] for x in df[coordinate_col]])
            print(f"Reduce to 2 dimensions via {reduce_dim} before plotting...")
        if reduce_dim.lower() in "pca":
            res = self.pca_2D.fit_transform(df_temp).T
            x_coor = res[0]
            y_coor = res[1]
        elif reduce_dim.lower() in "umap":
                res = umap.UMAP(n_jobs=-1,
                                verbose=0,
                                n_components=2,
                                metric="cosine",
                                init='spectral',
                                random_state=42,
                                transform_seed=42,
                                n_neighbors=100,
                                min_dist=0.1).fit_transform(df_temp)
                x_coor = res.T[0]
                y_coor = res.T[1]
        elif reduce_dim is None:
            x_coor = [x[0] for x in df[coordinate_col]],
            x_coor = list(x_coor)[0]
            y_coor = [x[1] for x in df[coordinate_col]],
            y_coor = list(y_coor)[0]
        print("Plotting results...")
        df["cropped_text"] = df["text"].str[0:75]
        fig = px.scatter(df,
                         title="AnnA Anki neuronal Appendix",
                         x = x_coor,
                         y = y_coor,
                         color=color_col,
                         hover_data=["cropped_text", "cluster_topic"])
        fig.show()

    def find_notes_similar_to_input(self,
                                    user_input,
                                    nlimit=5,
                                    user_col="sbert",
                                    dist="cosine", reverse=False):
        "given a text input, find notes with highest cosine similarity"
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)
        df = self.df

        embed = sbert.encode(user_input, normalize_embeddings=True)
        print("")
        tqdm.pandas(desc="Searching")
        df["distance"] = df[user_col].progress_apply(
                lambda x : pairwise_distances(embed.reshape(1, -1),
                                              x.reshape(1, -1),
                                              metric=dist))
        index = df.index
        good_order = sorted(index, key=lambda row: df.loc[row, "distance"], reverse=reverse)
        cnt = 0
        ans = "y"
        while True:
            cnt += 1
            if ans != "n":
                print(df.loc[good_order[nlimit*cnt:nlimit*(cnt+1)], ["text", "distance"]])
            else:
                break
            ans = input("Show more?\n(y/n)>")
        pd.reset_option("display.max_rows")
        pd.reset_option('display.max_columns')
        pd.reset_option('display.width')
        pd.reset_option('display.max_colwidth')
        return True


    def save_df(self, df=None, out_name=None):
        "export dataframe as pickle format"
        if df is None:
            df = self.df
        name = f"{out_name}_{self.deckname}_{int(time.time())}.pickle"
        df.to_pickle(name)
        print(f"Dataframe exported to {name}")



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
        X = normalize(X, axis=1, norm='l2', copy=False)
        return X

import_thread = threading.Thread(target=asynchronous_importer, daemon=True)
import_thread.start()

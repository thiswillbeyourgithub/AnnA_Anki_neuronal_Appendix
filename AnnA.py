import sys
import pickle
import time
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
import importlib
from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter
from pathlib import Path
import threading
from sklearn.feature_extraction.text import TfidfTransformer
import scipy.sparse as sp
import logging

# avoids annoying warning
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# makes the script interuptible, resume it using c+enter
signal.signal(signal.SIGINT, (lambda signal, frame: pdb.set_trace()))

# adds logger
log = logging.getLogger()
out_hdlr = logging.StreamHandler(sys.stdout)
out_hdlr.setFormatter(logging.Formatter('%(message)s'))
out_hdlr.setLevel(logging.INFO)
log.addHandler(out_hdlr)
log.setLevel(logging.ERROR)

def war(string):
    coloured_log(string, "war")
def inf(string):
    coloured_log(string, "inf")
def err(string):
    coloured_log(string, "err")

def coloured_log(string, mode):
    col_red = "\033[91m"
    col_yel = "\033[93m"
    col_rst = "\033[0m"

    if mode == "inf":
        log.info(col_rst + string + col_rst)
    elif mode == "war":
        log.warn(col_yel + string + col_rst)
    elif mode == "err":
        log.error(col_red + string + col_rst)


def asynchronous_importer(TFIDF_enable):
    """
    used to asynchronously import large modules, this way between
    importing AnnA and creating the instance of the class, the language model
    have some more time to load
    """
    global np, KMeans, DBSCAN, tokenizer, \
        AgglomerativeClustering, transformers, normalize, TfidfVectorizer,\
        CountVectorizer, TruncatedSVD, StandardScaler, \
        pairwise_distances, PCA, px, umap, np, tokenizer_bert, \
        MiniBatchKMeans, interpolate
    if "sentence_transformers" not in sys.modules:
        inf("Began importing modules...\n")
        print_when_ends = True
    else:
        print_when_ends = False
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.feature_extraction.text import CountVectorizer
    if TFIDF_enable is False:
        global sBERT
        from sentence_transformers import SentenceTransformer
        sBERT = SentenceTransformer('distiluse-base-multilingual-cased-v1')
    else:
        global stopwords
        from nltk.corpus import stopwords

    from transformers import BertTokenizerFast
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-multilingual-uncased")
    from sklearn.metrics import pairwise_distances
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    from sklearn.cluster import MiniBatchKMeans
    import plotly.express as px
    import umap.umap_
    from sklearn.preprocessing import normalize
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import StandardScaler
    from scipy import interpolate
    if print_when_ends:
        inf("Finished importing modules.\n\n")


class AnnA:
    """
    main class: used to centralize everything
    just instantiating the class does most of the job, as you can see
    in self.__init__
    """
    def __init__(self, show_banner=True,
                 # main settings
                 deckname=None,
                 reference_order="lowest_interval",
                 desired_deck_size="80%",
                 rated_last_X_days=4,
                 stride=2500,
                 scoring_weights=(1, 1),
                 log_level=0,
                 replace_greek=True,
                 keep_ocr=True,
                 field_mappings="field_mappings.py",
                 acronym_list="acronym_list.py",

                 # steps:
                 clustering_enable=True,
                 clustering_nb_clust="auto",
                 compute_opti_rev_order=True,
                 check_database=False,

                 # tasks:
                 task_filtered_deck=True,
                 task_bury_learning=False,
                 task_index_deck=False,

                 # vectorization:
                 sBERT_dim=None,
                 TFIDF_enable=True,
                 TFIDF_dim=1000,
                 TFIDF_stopw_lang=["english", "french"],

                 # misc:
                 debug_card_limit=None,
                 debug_force_score_formula=None,
                 prefer_similar_card=False,
                 ):
        if log_level == 0:
            log.setLevel(logging.ERROR)
        elif log_level == 1:
            log.setLevel(logging.WARNING)
        elif log_level >= 2:
            log.setLevel(logging.DEBUG)
        else:
            log.setLevel(logging.INFO)


        if show_banner is True:
            print(pyfiglet.figlet_format("AnnA"))
            print("(Anki neuronal Appendix)\n\n")


        # start importing large modules
        import_thread = threading.Thread(target=asynchronous_importer,
                args=(TFIDF_enable,))
        import_thread.start()

        # loading args
        self.replace_greek = replace_greek
        self.keep_ocr = keep_ocr
        self.desired_deck_size = desired_deck_size
        self.rated_last_X_days = rated_last_X_days
        self.debug_card_limit = debug_card_limit
        self.clustering_nb_clust = clustering_nb_clust
        self.sBERT_dim = sBERT_dim
        self.stride = stride
        self.prefer_similar_card = prefer_similar_card
        self.scoring_weights = scoring_weights
        self.reference_order = reference_order
        self.field_mappings = field_mappings
        self.acronym_list = acronym_list
        self.debug_force_score_formula = debug_force_score_formula
        self.TFIDF_enable  = TFIDF_enable
        self.TFIDF_dim = TFIDF_dim
        self.TFIDF_stopw_lang = TFIDF_stopw_lang

        assert stride > 0
        assert reference_order in ["lowest_interval", "relative_overdueness"]

        if self.acronym_list is not None:
            file = Path(acronym_list)
            if not file.exists():
                raise Exception(f"Acronym file was not found: {acronym_list}")
            else:
                imp = importlib.import_module(
                        acronym_list.replace(".py", ""))
                self.acronym_dict = imp.acronym_dict
        if self.field_mappings is not None:
            file = Path(self.field_mappings)
            try:
                assert file.exists()
                imp = importlib.import_module(
                        self.field_mappings.replace(".py", ""))
                self.field_dic = imp.field_dic
            except Exception as e:
                err(f"Error with field mapping file, will use default \
values. {e}")
                self.field_dic = {"dummyvalue": "dummyvalue"}

        # actual execution
        self.deckname = self._check_deck(deckname, import_thread)
        if task_index_deck is True:
            print(f"Task : cache vectors of deck: {self.deckname}")
            self.rated_last_X_days = None
            self._create_and_fill_df(task_index_deck=task_index_deck)
            self.df = self._reset_index_dtype(self.df)
            self._format_card()
            self.show_acronyms()
            self._compute_sBERT_vec(import_thread=import_thread)
            if clustering_enable is True:
                self.compute_clusters(minibatchk_kwargs={"verbose": 0})
        elif task_bury_learning is True:
            # bypasses most of the code to bury learning cards
            # directly in the deck without creating filtered decks
            print("Task : bury some learning cards.")
            inf(f"Burying similar learning cards from deck {self.deckname}..\
.")
            inf("Forcing 'reference_order' to 'lowest_interval'.")
            self.reference_order = "lowest_interval"
            inf("Forcing rated_last_X_days to None.")
            self.rated_last_X_days = None

            self._create_and_fill_df(just_learning=True)
            self.df = self._reset_index_dtype(self.df)
            self._format_card()
            self.show_acronyms()
            self._compute_sBERT_vec(import_thread=import_thread)
            self._compute_distance_matrix()
            self._compute_opti_rev_order()
            self.task_filtered_deck(just_bury=True)
        else:
            self._create_and_fill_df()
            self.df = self._reset_index_dtype(self.df)
            self._format_card()
            self.show_acronyms()
            self._compute_sBERT_vec(import_thread=import_thread)
            if clustering_enable is True:
                self.compute_clusters(minibatchk_kwargs={"verbose": 0})
            self._compute_distance_matrix()
            if compute_opti_rev_order is True:
                self._compute_opti_rev_order()
                if task_filtered_deck is True:
                    self.task_filtered_deck()

        # pickle itself
        print("\nSaving instance as 'last_run.pickle'...")
        if Path("last_run.pickle").exists():
            Path("last_run.pickle").unlink()
        with open("last_run.pickle", "wb") as f:
            try:
                pickle.dump(self, f)
                inf("Done! You can now restore this instance of AnnA without having to \
execute the code using:\n'import pickle ; a = pickle.load(open(\"last_run.pickle\
\", \"rb\"))'")
            except TypeError as e:
                err(f"Error when saving instance as pickle file: {e}")

        if check_database is True:
            inf("Re-optimizing Anki database")
            self._ankiconnect(action="guiCheckDatabase")

        print(f"Done with {self.deckname}")

    def _reset_index_dtype(self, df):
        """
        the index dtype (cardId) somehow gets turned into float so I
        occasionally turn it back into int
        """
        temp = df.reset_index()
        temp["cardId"] = temp["cardId"].astype(int)
        df = temp.set_index("cardId")
        return df

    @classmethod
    def _ankiconnect(self, action, **params):
        """
        used to send request to anki using the addon anki-connect
        """
        def request_wrapper(action, **params):
            return {'action': action, 'params': params, 'version': 6}

        requestJson = json.dumps(request_wrapper(action, **params)
                                 ).encode('utf-8')
        try:
            response = json.load(urllib.request.urlopen(
                                    urllib.request.Request(
                                        'http://localhost:8765',
                                        requestJson)))
        except (ConnectionRefusedError, urllib.error.URLError) as e:
            raise Exception(f"{e}: is Anki open and ankiconnect enabled?")

        if len(response) != 2:
            raise Exception('response has an unexpected number of fields')
        if 'error' not in response:
            raise Exception('response is missing required error field')
        if 'result' not in response:
            raise Exception('response is missing required result field')
        if response['error'] is not None:
            raise Exception(response['error'])
        return response['result']

    def _get_cards_info_from_card_id(self, card_id):
        """
        get all information from a card using its card id, works with
        either int of list of int

        * Due to the time it takes to get thousands of cards, I decided
            to used Threading extensively to speed it up.
        """
        if isinstance(card_id, list):
            if len(card_id) < 50:
                r_list = []
                for card in tqdm(card_id):
                    r_list.extend(self._ankiconnect(action="cardsInfo",
                                  cards=[card]))
                return r_list

            else:
                lock = threading.Lock()
                threads = []
                cnt = 0
                r_list = []
                target_thread_n = 5
                batchsize = len(card_id)//target_thread_n+3
                inf(f"Large number of cards to retrieve: creating 10 \
threads of size {batchsize} (total: {len(card_id)} cards)...")

                def retrieve_cards(card_list, lock, cnt, r_list):
                    "for multithreaded card retrieval"
                    out_list = self._ankiconnect(action="cardsInfo",
                                                        cards=card_list)
                    with lock:
                        r_list.extend(out_list)
                        pbar.update(1)
                    return True

                with tqdm(total=target_thread_n,
                          unit="thread",
                          dynamic_ncols=True,
                          desc="Done threads",
                          delay=2,
                          smoothing=0) as pbar:
                    for nb in range(0, len(card_id), batchsize):
                        cnt += 1
                        temp_card_id = card_id[nb: nb+batchsize]
                        thread = threading.Thread(target=retrieve_cards,
                                                  args=(temp_card_id,
                                                        lock,
                                                        cnt,
                                                        r_list),
                                                  daemon=False)
                        thread.start()
                        threads.append(thread)
                        time.sleep(0.1)
                        while sum([t.is_alive() for t in threads]) >= 15:
                            time.sleep(0.5)
                    print("")
                    [t.join() for t in threads]
                assert len(r_list) == len(card_id)
                r_list = sorted(r_list,
                                key=lambda x: x["cardId"],
                                reverse=False)
                return r_list

        if isinstance(card_id, int):
            return self._ankiconnect(action="cardsInfo",
                                            cards=[card_id])

    def _check_deck(self, deckname, import_thread):
        """
        used to check if the deckname is correct
        if incorrect, user is asked to enter the name, using autocompletion
        """
        decklist = self._ankiconnect(action="deckNames") + ["*"]
        if deckname is not None:
            if deckname not in decklist:
                err("Couldn't find this deck.")
                deckname = None
        if deckname is None:
            auto_complete = WordCompleter(decklist,
                                          match_middle=True,
                                          ignore_case=True)
            deckname = ""
            import_thread.join()  # otherwise some message can appear
            # in the middle of the prompt
            time.sleep(0.5)
            while deckname not in decklist:
                deckname = prompt("Enter the name of the deck:\n>",
                                  completer=auto_complete)
        print(f"Selected deck: {deckname}")
        return deckname

    def _create_and_fill_df(self, just_learning=False, task_index_deck=False):
        """
        create a pandas DataFrame, fill it with the information gathered from
        anki connect like card content, intervals, etc
        """

        if just_learning is False:
            print("Getting due card list...")
            query = f"deck:{self.deckname} is:due is:review -is:learn \
-is:suspended -is:buried -is:new -rated:1"
            inf(" >  '" + query + "'\n\n")
            due_cards = self._ankiconnect(action="findCards", query=query)
        elif just_learning is True:
            print("Getting is:learn card list...")
            query = f"deck:{self.deckname} is:learn -is:suspended is:due -rated:1"
            inf(" >  '" + query + "'\n\n")
            due_cards = self._ankiconnect(action="findCards", query=query)
            print(f"Found {len(due_cards)} learning cards...")
        elif task_index_deck is True:
            print("Getting all cards from collection...")
            query = f"deck:{self.deckname}"
            inf(" >  '" + query + "'\n\n")
            due_cards = self._ankiconnect(action="findCards", query=query)
            print(f"Found {len(due_cards)} cards...")

        n_rated_days = self.rated_last_X_days
        if n_rated_days is not None:
            if int(n_rated_days) != 0:
                print(f"Getting cards that where rated in the last \
{n_rated_days} days  ...")
                query = f"deck:{self.deckname} rated:{n_rated_days} \
-is:suspended"
                inf(" >  '" + query + "'\n\n")
                r_cards = self._ankiconnect(action="findCards", query=query)

                # removes overlap if found
                rated_cards = [x for x in r_cards if x not in due_cards]
                print(f"Rated cards contained {len(rated_cards)} relevant cards \
(out of {len(r_cards)}).")
        else:
            print("Will not look for cards rated in past days.")
            rated_cards = []
        self.due_cards = due_cards
        self.rated_cards = rated_cards

        limit = self.debug_card_limit if self.debug_card_limit else None
        combined_card_list = list(rated_cards + due_cards)[:limit]

        if len(combined_card_list) < 20:
            err("You don't have enough cards!\nExiting.")
            raise SystemExit()

        list_cardInfo = []

        n = len(combined_card_list)
        print(f"\nAsking Anki for information about {n} cards...")
        start = time.time()
        list_cardInfo.extend(
                self._get_cards_info_from_card_id(
                    card_id=combined_card_list))
        inf(f"Extracted information in {int(time.time()-start)} seconds.\n\n")

        for i, card in enumerate(list_cardInfo):
            # removing large fields:
            list_cardInfo[i].pop("question")
            list_cardInfo[i].pop("answer")
            list_cardInfo[i].pop("css")
            list_cardInfo[i].pop("fields_no_html")
            list_cardInfo[i]["tags"] = " ".join(list_cardInfo[i]["tags"])
            if card["cardId"] in due_cards:
                list_cardInfo[i]["status"] = "due"
            elif card["cardId"] in rated_cards:
                list_cardInfo[i]["status"] = "rated"
            else:
                list_cardInfo[i]["status"] = "ERROR"
                err(f"Error processing card with ID {card['cardId']}")

        if len(list_cardInfo) != len(list(set(combined_card_list))):
            err("Error: duplicate cards in DataFrame!\nExiting.")
            pdb.set_trace()

        self.df = pd.DataFrame().append(list_cardInfo,
                                        ignore_index=True,
                                        sort=True
                                        ).set_index("cardId").sort_index()
        return True

    def _format_text(self, text):
        """
        take text and output processed and formatted text
        Acronyms will be replaced if the corresponding arguments is passed
            when instantiating AnnA
        Greek letters can also be replaced on the fly
        """
        orig = text

        s = re.sub

        text = str(text)
        text = s("\n+", " ", text)
        text = text.replace("+", " ")  # sbert does not work well with that
        text = text.replace("-", " ")
        if self.keep_ocr is True:
            # keep image title (usually OCR)
            text = s("title=(\".*?\")", "> Caption: '\\1' <", text)
            text = text.replace('Caption: \'""\'', "")
        if self.replace_greek is True:
            for a, b in greek_alphabet_mapping.items():
                text = s(a, b, text)
        if self.acronym_list is True:
            global acronym_dict
            for a, b in self.acronym_dict.items():
                text = s(rf"{a}", f"{a} ({b})", text, flags=re.IGNORECASE)
                # \b matches beginning and end of a word
        text = s(r'[a-zA-Z0-9-]+\....', " ", text)  # media file name
        text = s("<blockquote(.*?)</blockquote>",
                 lambda x: x.group(0).replace("<br>", " ; "), text)
        text = s('\\n|<div>|</div>|<br>|<span>|</span>|<li>|</li>|<ul>|</ul>',
                 " ", text)  # newlines
        text = s("<a href.*?</a>", " ", text)  # html links
        text = s(r'http[s]?://\S*', " ", text)  # plaintext links
        text = s("<.*?>", " ", text)  # remaining html tags
        text = s('\u001F|&nbsp;', " ", text)
        text = s(r"{{c\d+?::", "", text)
        text = s("{{c|{{|}}|::", " ", text)
        text = s(r"\b(\d+)e\b", "\\1 euros", text)
        text = s(r"\b(\d+)j\b", "\\1 jours", text)
        text = s(r"\b(\d+)h\b", "\\1 heures", text)
        text = s(r"\b(\d+)m\b", "\\1 minutes", text)
        text = s(r"\b(\d+)s\b", "\\1 secondes", text)
        text = s(r"\[\d*\]", "", text)  # wiki citation
        text = text.replace("&amp;", "&")
        text = text.replace("/", " / ")
        text = s(r"\w{1,5}>", " ", text)  # missed html tags
        text = s("&gt;|&lt;|<|>", "", text)
        text = s("[.?!] ([a-zA-Z])", lambda x: x.group(0).upper(), text)
        # adds capital letter after punctuation

        text = text.replace(" : ", ": ")
        text = " ".join(text.split())  # multiple spaces
        if len(text) < 10:
            if "src=" in orig:
                text = text + " " + " ".join(re.findall('src="(.*?)">', orig))
        if len(text) > 2:
            text = text[0].upper() + text[1:]
            if text[-1] not in ["?", ".", "!"]:
                text += "."
        text = text.replace(" :.", ".")
        text = text.replace(":.", ".")
        return text

    def _format_card(self):
        """
        filter the fields of each card and keep only the relevant fields
        a "relevant field" is one that is mentionned in the variable field_dic
        which can be found at the top of the file. If not relevant field are
        found then only the first field is kept.
        """
        def _threaded_field_filter(df, index_list, lock, pbar):
            """
            threaded implementaation to speed up execution
            """
            for index in index_list:
                card_model = df.loc[index, "modelName"]
                take_first_field = False
                fields_to_keep = []

                # determines which is the corresponding model described
                # in field_dic
                cnt = 0
                field_dic = self.field_dic
                for user_model in field_dic.keys():
                    if user_model.lower() in card_model.lower():
                        cnt += 1
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
                if take_first_field is True:  # if no corresponding model
                    # was found in field_dic
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
                        comb_text = comb_text + to_add + ": "
                final_text = comb_text.strip().replace(": :", "")

                with lock:
                    self.df.at[index, "comb_text"] = final_text
                    pbar.update(1)

        df = self.df.copy()
        n = len(df.index)
        batchsize = n//5+1
        lock = threading.Lock()
        threads = []
        with tqdm(total=n,
                  desc="Keeping only relevant fields",
                  smoothing=0,
                  unit=" card") as pbar:
            for nb in range(0, n, batchsize):
                    sub_card_list = df.index[nb: nb+batchsize]
                    thread = threading.Thread(target=_threaded_field_filter,
                                              args=(df,
                                                    sub_card_list,
                                                    lock,
                                                    pbar),
                                              daemon=False)
                    thread.start()
                    threads.append(thread)
                    while sum([t.is_alive() for t in threads]) >= 15:
                        time.sleep(0.5)
            [t.join() for t in threads]

        df = self.df.copy()
        df["text"] = [self._format_text(x)
                      for x in tqdm(
                      df["comb_text"],
                      desc="Formating text", smoothing=0, unit=" card")]
        print("\n\nPrinting 5 random samples of your formated text, to help \
adjust formating issues:")
        pd.set_option('display.max_colwidth', 80)
        sub_index = random.choices(df.index.tolist(), k=5)
        for i in sub_index:
            print(f"{i}: {df.loc[i, 'text']}\n")
        pd.reset_option('display.max_colwidth')
        print("\n")
        self.df = df.sort_index()
        return True

    def _compute_sBERT_vec(self, df=None, use_sBERT_cache=True, import_thread=None):
        """
        Assigne sBERT vectors to each card
        df["VEC_FULL"], contains the vectors
        df["VEC"] contains either all the vectors or less if you
            enabled dimensionality reduction
        * given how long it is to compute the vectors I decided to store
            all already computed sBERT to a pickled DataFrame at each run
        """
        if df is None:
            df = self.df

        if self.TFIDF_enable is not True:
            if use_sBERT_cache is True:
                print("\nLooking for cached sBERT pickle file...", end="")
                sBERT_file = Path("./sBERT_cache.pickle")
                df["VEC"] = 0*len(df.index)
                df["VEC"] = df["VEC"].astype("object")
                loaded_sBERT = 0
                id_to_recompute = []

                # reloads sBERT vectors and only recomputes the new one:
                if not sBERT_file.exists():
                    inf(" sBERT cache not found, will create it.")
                    df_cache = pd.DataFrame(
                            columns=["cardId", "mod", "text", "VEC"]
                            ).set_index("cardId")
                    id_to_recompute = df.index
                else:
                    print(" Found sBERT cache.")
                    df_cache = pd.read_pickle(sBERT_file)
                    df_cache["VEC"] = df_cache["VEC"].astype("object")
                    df_cache["mod"] = df_cache["mod"].astype("object")
                    df_cache["text"] = df_cache["text"]
                    df_cache = self._reset_index_dtype(df_cache)
                    for i in df.index:
                        if i in df_cache.index and \
                                (str(df_cache.loc[i, "text"]) ==
                                    str(df.loc[i, "text"])):
                            df.at[i, "VEC"] = df_cache.loc[i, "VEC"]
                            loaded_sBERT += 1
                        else:
                            id_to_recompute.append(i)

                print(f"Loaded {loaded_sBERT} vectors from cache, will compute \
{len(id_to_recompute)} others...")
                if import_thread is not None:
                    import_thread.join()
                    time.sleep(0.5)
                if len(id_to_recompute) != 0:
                    sentence_list = [df.loc[x, "text"]
                                     for x in df.index if x in id_to_recompute]
                    sentence_embeddings = sBERT.encode(sentence_list,
                                                       normalize_embeddings=True,
                                                       show_progress_bar=True)

                    for i, ind in enumerate(tqdm(id_to_recompute)):
                        df.at[ind, "VEC"] = sentence_embeddings[i]

                # stores newly computed sBERT vectors in a file:
                df_cache = self._reset_index_dtype(df_cache)
                for i in [x for x in id_to_recompute if x not in df_cache.index]:
                    df_cache.loc[i, "VEC"] = df.loc[i, "VEC"].astype("object")
                    df_cache.loc[i, "mod"] = df.loc[i, "mod"].astype("object")
                    df_cache.loc[i, "text"] = df.loc[i, "text"]
                for i in [x for x in id_to_recompute if x in df_cache.index]:
                    df_cache.loc[i, "VEC"] = df.loc[i, "VEC"].astype("object")
                    df_cache.loc[i, "mod"] = df.loc[i, "mod"].astype("object")
                    df_cache.loc[i, "text"] = df.loc[i, "text"]
                df_cache = self._reset_index_dtype(df_cache)
                try:
                    Path("sBERT_cache.pickle_temp").unlink()
                except FileNotFoundError:
                    pass
                df_cache.to_pickle("sBERT_cache.pickle_temp")
                sBERT_file.unlink()
                Path("sBERT_cache.pickle_temp").rename("sBERT_cache.pickle")

            df["VEC_FULL"] = df["VEC"]
            if self.sBERT_dim is not None:
                print(f"Reducing sBERT to {self.sBERT_dim} dimensions \
using PCA...")
                pca_sBERT = PCA(n_components=self.sBERT_dim, random_state=42)
                df_temp = pd.DataFrame(
                    columns=["V"+str(x+1)
                             for x in range(len(df.loc[df.index[0], "VEC"]))],
                    data=[x[0:] for x in df["VEC"]])
                out = pca_sBERT.fit_transform(df_temp)
                inf(f"Explained variance ratio after PCA on sBERT: \
{round(sum(pca_sBERT.explained_variance_ratio_)*100,1)}%")
                df["VEC"] = [x for x in out]

        else:  # use TFIDF instead of sBERT
            if import_thread is not None:
                import_thread.join()
                time.sleep(0.5)
            print("Creating stop words list...")
            try:
                stops = []
                for lang in self.TFIDF_stopw_lang:
                    [stops.extend(
                        tokenizer.tokenize(x)
                        ) for x in stopwords.words(lang)]
                stops = list(set(stops))
            except Exception as e:
                err(f"Error when extracting stop words: {e}")
                err("Setting stop words list to None.")
                stops = None

            vectorizer = TfidfVectorizer(strip_accents="unicode",
                                         lowercase=True,
                                         tokenizer=lambda x: tokenizer.tokenize(x),
                                         stop_words=stops,
                                         ngram_range=(1, 10),
                                         max_features=10_000,
                                         norm="l2")
            t_vec = vectorizer.fit_transform(tqdm(df["text"],
                                             desc="Vectorizing text using TFIDF"))
            df["VEC_FULL"] = ""
            df["VEC"] = ""
            if self.TFIDF_dim is None:
                df["VEC_FULL"] = [x for x in t_vec]
                df["VEC"] = [x for x in t_vec]
                self.t_vec = [x for x in t_vec]
                self.t_red = None
            else:
                print(f"Reducing dimensions to {self.TFIDF_dim}")
                svd = TruncatedSVD(n_components = min(self.TFIDF_dim,
                                                      t_vec.shape[1]))
                t_red = svd.fit_transform(t_vec)
                inf(f"Explained variance ratio after SVD on Tf_idf: \
{round(sum(svd.explained_variance_ratio_)*100,1)}%")
                df["VEC_FULL"] = [x for x in t_vec]
                df["VEC"] = [x for x in t_red]
                self.t_vec = [x for x in t_vec]
                self.t_red = [x for x in t_red]

        self.df = df
        return True

    def _compute_distance_matrix(self, method="cosine", input_col="VEC"):
        """
        compute distance matrix between all the cards
        * the distance matrix can be parallelised by scikit learn so I didn't
            bother saving and reusing the matrix
        """
        df = self.df

        df_temp = pd.DataFrame(
            columns=["V"+str(x+1)
                     for x in range(len(df.loc[df.index[0], input_col]))],
            data=[x[0:] for x in df[input_col]])

        print("\nComputing distance matrix on all available cores...")
        df_dist = pairwise_distances(df_temp, n_jobs=-1, metric=method)
#        print("Interpolating matrix between 0 and 1...")
#        df_dist = np.interp(df_dist, (df_dist.min(), df_dist.max()), (0, 1))

        self.df_dist = pd.DataFrame(columns=df.index,
                                    index=df.index,
                                    data=df_dist)
        return True

    def _compute_opti_rev_order(self):
        """
        a loop that assigns a score to each card, the lowest score at each
            turn is added to the queue, each new turn compares the cards to
            the present queue

        * this score reflects the order in which they should be reviewed
        * the intuition is that anki doesn't know before hand if some cards
            are semantically close and can have you review them the same day
        * The score is computed according to the formula:
           score = ref - min of (similarity to each card of the big_queue)
           (big_queue here referes to the recently rated cards concatenated
            with the queue cards)
        * ref is either the interval or the negative relative overdueness),
            it is centered and scaled. In both case, a lower ref is indicating
            that reviewing the card is urgent.
        * the_chosen_one is the card with the lowest score at each round
        * the queue starts empty. At the end of each turn, the_chosen_one is
            added to it
        * some values are clipped,cented, scaled. Relative
            overdueness is weird so I'm trying interpolating if > 500 o
            < -500
        * I clipped the distance value below 0.3 as they were messing with the
            scaling afterwards

        CAREFUL: this docstring might not be up to date as I am constantly
            trying to improve the code
        """
        # getting args
        reference_order = self.reference_order
        df = self.df.copy()
        df_dist = self.df_dist
        desired_deck_size = self.desired_deck_size
        rated = self.rated_cards
        due = self.due_cards
        queue = []
        if self.prefer_similar_card is True:
            sign = 1
        else:
            sign = -1
        w1 = self.scoring_weights[0]
        w2 = self.scoring_weights[1]*sign

        # alter the value from rated cards as they will not be useful
        df.loc[rated, "due"] = np.median(df.loc[due, "due"].values)
        df.loc[rated, "interval"] = np.median(df.loc[due, "interval"].values)

        # preparing interval column
        ivl = df['interval'].to_numpy().reshape(-1, 1)
        df["interval_cs"] = StandardScaler().fit_transform(ivl)

        # lowest interval is still centered and scaled
        if reference_order == "lowest_interval":
            df["ref"] = df["interval_cs"]

        # computes relative_overdueness
        if reference_order == "relative_overdueness":
            print("Computing relative overdueness...")
            # getting due date
            for i in df.index:
                df.at[i, "ref_due"] = df.loc[i, "odue"]
                if df.loc[i, "ref_due"] == 0:
                    df.at[i, "ref_due"] = df.at[i, "due"]

            # computing overdue
            anki_col_time = int(self._ankiconnect(
                action="getCollectionCreationTime"))
            time_offset = int((time.time() - anki_col_time) // 86400)
            overdue = (df["ref_due"] - time_offset).to_numpy().reshape(-1, 1)

            # computing relative overdueness
            ro = -1 * (df["interval"].values + 0.001) / (overdue.T + 0.001)

            # either: clipping abnormal values
#            ro_intp = np.clip(ro, -500, 500)

            # or: interpolating values
            n_pos = sum(sum(ro > 500))
            n_neg = sum(sum(ro < -500))
            ro_intp = ro
            if n_pos != 0:
                f = interpolate.interp1d(x=ro[ro > 500],
                                         y=[500+x for x in range(n_pos)])
                ro_intp[ro > 500] = f(ro[ro > 500])
            if n_neg != 0:
                f = interpolate.interp1d(x=ro[ro < -500],
                                         y=[-500-x for x in range(n_neg)])
                ro_intp[ro < -500] = f(ro[ro < -500])

            # center and scale
            ro_cs = StandardScaler().fit_transform(ro_intp.T)
            df["ref"] = ro_cs

        # centering and scaling df_dist after clipping
        print("Centering and scaling distance matrix...")
        df_dist.loc[:, :] = StandardScaler().fit_transform(df_dist)

        # adjusting with weights
        df["ref"] = df["ref"]*w1
        df_dist = df_dist*w2

        assert len([x for x in rated if df.loc[x, "status"] != "rated"]) == 0
        print(f"\nCards rated in the past relevant days: {len(rated)}")

        if isinstance(desired_deck_size, float):
            if desired_deck_size < 1.0:
                desired_deck_size = str(desired_deck_size*100) + "%"
        if isinstance(desired_deck_size, str):
            if desired_deck_size in ["all", "100%"]:
                print("Taking the whole deck.")
                desired_deck_size = len(df.index) - len(rated)
            elif desired_deck_size.endswith("%"):
                print(f"Taking {desired_deck_size} of the deck.")
                desired_deck_size = 0.01*int(desired_deck_size[:-1])*(
                            len(df.index)-len(rated)
                            )
        desired_deck_size = int(desired_deck_size)

        if desired_deck_size > int(len(df.index)-len(rated)):
            err(f"You wanted to create a deck with \
{desired_deck_size} in it but the deck only contains \
{len(df.index)-len(rated)} cards. Taking the lowest value.")
        queue_size_goal = min(desired_deck_size,
                              len(df.index)-len(rated))

        if len(rated) < 1:  # can't start with an empty queue
            # so picking 1 urgent cards
            pool = df.loc[df["status"] == "due", "ref"].nsmallest(
                    n=min(50, len(self.due_cards))
                    ).index
            queue.extend(random.choices(pool, k=1))

        if self.debug_force_score_formula is not None:
            if self.debug_force_score_formula == "only_different":
                df["ref"] = 0
            if self.debug_force_score_formula == "only_similar":
                df["ref"] = 0
                df_dist.loc[:, :] = np.ones_like(df_dist.values)

        inf("\nReference score stats:")
        inf(f"mean: {df['ref'].describe()}\n")
        inf(f"max: {pd.DataFrame(data=df_dist.values.flatten(), columns=['distance matrix']).describe()}\n\n")

        with tqdm(desc="Computing optimal review order",
                  unit=" card",
                  initial=len(rated),
                  smoothing=0,
                  total=queue_size_goal+len(rated)) as pbar:
            indTODO = df.drop(index=rated+queue).index.tolist()
            indQUEUE = (rated+queue)
            while len(queue) < queue_size_goal:
                queue.append(indTODO[
                        (df.loc[indTODO, "ref"].values + np.mean([
                            np.max(
                                df_dist.loc[indQUEUE[-self.stride:], indTODO].values,
                                axis=0),
                            np.mean(
                                df_dist.loc[indQUEUE[-self.stride:], indTODO].values,
                                axis=0)
                            ], axis=0)
                         ).argmin()])
                indQUEUE.append(indTODO.pop(indTODO.index(queue[-1])))

                # I had some trouble with implementing this loop
                # so I am keeping legacy code as fallback:
                # canonical and check version give the same results
                # canonical version:
#                queue.append(indTODO[
#                        (df.loc[indTODO, "ref"].values + np.min(
#                            df_dist.loc[indQUEUE[-self.stride:], indTODO].values,
#                            axis=0)
#                         ).argmin()])

                # check version:
#                queue2 = [x for x in queue[:-1]]
#                df_temp = pd.DataFrame(columns=rated, index=df.index)
#                for q in (rated+queue2)[-self.stride:]:
#                    df_temp[q] = df_dist.values[df.index.get_loc(q)]
#                df["score"] = df["ref"].values + np.min(df_temp, axis=1)
#                chosen_one2 = df.drop(index=(rated+queue2))["score"].idxmin()
#                queue2.append(chosen_one2)
#                df_temp[chosen_one2] = df_dist.values[df.index.get_loc(chosen_one2)]
#                if queue[-1] != queue2[-1]:
#                    tqdm.write(f">   NO")
#                else:
#                    tqdm.write("YES")

                pbar.update(1)
        assert len(queue) != 0
        print("Done.\n")
        self.opti_rev_order = queue
        return True

    def display_opti_rev_order(self, display_limit=50):
        """
        display the cards in the best optimal order. Useful to see if something
        went wrong before creating the filtered deck
        """
        order = self.opti_rev_order[:display_limit]
        print(self.df.loc[order, "text"])
        return True

    def task_filtered_deck(self,
                     deck_template="AnnA - Optimal Review Order",
                     just_bury=False):
        """
        create a filtered deck containing the cards to review in optimal order

        * When first creating the filtered deck, I chose 'sortOrder = 0'
            ("oldest seen first") this way I will notice if the deck
            somehow got rebuild and lost the right order
        * To speed up the process, I decided to create a threaded function call
        * I do a few sanity check to see if the filtered deck
            does indeed contain the right number of cards and the right cards
        * -100 000 seems to be the starting value for due order in filtered
            decks
        * if just_bury is True, then no filtered deck will be created and
            AnnA will just bury the cards that are too similar
        """
        if just_bury is True:
            to_keep = self.opti_rev_order
            to_bury = [x for x in self.due_cards if x not in to_keep]
            assert len(to_bury) < len(self.due_cards)
            print(f"Burying {len(to_bury)} cards out of {len(self.due_cards)}")
            self._ankiconnect(action="bury",
                              cards=to_bury)
            print("Done.")
            return True

        filtered_deck_name = str(deck_template + f" - {self.deckname}")
        filtered_deck_name = filtered_deck_name.replace("::", "_")
        self.filtered_deck_name = filtered_deck_name

        while filtered_deck_name in self._ankiconnect(action="deckNames"):
            print(f"\nFound existing filtered deck: {filtered_deck_name} \
You have to delete it manually, the cards will be returned to their original \
deck.")
            input("Done? >")

        def _threaded_value_setter(card_list, tqdm_desc, keys, newValues):
            """
            create threads to edit card values quickly
            """
            def do_action(card_list,
                          sub_card_list,
                          keys,
                          newValues,
                          lock,
                          pbar):
                for c in sub_card_list:
                    if keys == ["due"]:
                        newValues = [-100000 + card_list.index(c)]
                    self._ankiconnect(action="setSpecificValueOfCard",
                                      card=int(c),
                                      keys=keys,
                                      newValues=newValues)
                    pbar.update(1)
                return True

            with tqdm(desc=tqdm_desc,
                      unit=" card",
                      total=len(card_list),
                      dynamic_ncols=True,
                      smoothing=0) as pbar:
                lock = threading.Lock()
                threads = []
                batchsize = len(card_list)//3+1
                for nb in range(0, len(card_list), batchsize):
                    sub_card_list = card_list[nb: nb+batchsize]
                    thread = threading.Thread(target=do_action,
                                              args=(card_list,
                                                    sub_card_list,
                                                    keys,
                                                    newValues,
                                                    lock,
                                                    pbar),
                                              daemon=False)
                    thread.start()
                    threads.append(thread)
                    while sum([t.is_alive() for t in threads]) >= 5:
                        time.sleep(0.5)
                [t.join() for t in threads]
            return True

        print(f"Creating deck containing the cards to review: \
{filtered_deck_name}")
        query = "cid:" + ','.join([str(x) for x in self.opti_rev_order])
        self._ankiconnect(action="createFilteredDeck",
                          newDeckName=filtered_deck_name,
                          searchQuery=query,
                          gatherCount=len(self.opti_rev_order)+1,
                          reschedule=True,
                          sortOrder=0,
                          createEmpty=False)

        print("Checking that the content of filtered deck name is the same as \
 the order inferred by AnnA...", end="")
        cur_in_deck = self._ankiconnect(action="findCards",
                                        query=f"\"deck:{filtered_deck_name}\"")
        diff = [x for x in self.opti_rev_order + cur_in_deck
                if x not in self.opti_rev_order or x not in cur_in_deck]
        if len(diff) != 0:
            err("Inconsistency! The deck does not contain the same cards \
as opti_rev_order!")
            pprint(diff)
            err(f"\nNumber of inconsistent cards: {len(diff)}")
        else:
            print(" Done.")

        _threaded_value_setter(card_list=self.opti_rev_order,
                               tqdm_desc="Altering due order",
                               keys=["due"],
                               newValues=None)
        print("All done!\n\n")
        return True

    def compute_clusters(self,
                         method="minibatch-kmeans",
                         input_col="VEC",
                         output_col="clusters",
                         n_topics=5,
                         minibatchk_kwargs=None,
                         kmeans_kwargs=None,
                         agglo_kwargs=None,
                         dbscan_kwargs=None,
                         add_as_tags=True,
                         tokenize_tags=True):
        """
        finds cluster of cards and their respective topics
        * this is not mandatory to create the filtered deck but it's rather
            fast
        * Several algorithm are supported for clustering: kmeans, DBSCAN,
            agglomerative clustering
        * n_topics is the number of topics (=word) to get for each cluster
        * To find the topic of each cluster, ctf-idf is used
        """
        df = self.df
        if self.clustering_nb_clust is None or self.clustering_nb_clust == "auto":
            self.clustering_nb_clust = len(df.index)//20
            print(f"No number of clusters supplied, will try with \
{self.clustering_nb_clust}.")
        kmeans_kwargs_deploy = {"n_clusters": self.clustering_nb_clust}
        dbscan_kwargs_deploy = {"eps": 0.75,
                                "min_samples": 3,
                                "n_jobs": -1}
        agglo_kwargs_deploy = {"n_clusters": self.clustering_nb_clust,
                               # "distance_threshold": 0.6,
                               "affinity": "cosine",
                               "memory": "/tmp/",
                               "linkage": "average"}
        minibatchk_kwargs_deploy = {"n_clusters": self.clustering_nb_clust,
                                    "max_iter": 100,
                                    "batch_size": 100,
                                    "verbose": 1,
                                    }
        if minibatchk_kwargs is not None:
            minibatchk_kwargs_deploy.update(minibatchk_kwargs)
        if kmeans_kwargs is not None:
            kmeans_kwargs_deploy.update(kmeans_kwargs)
        if dbscan_kwargs is not None:
            dbscan_kwargs_deploy.update(dbscan_kwargs)
        if agglo_kwargs is not None:
            agglo_kwargs_deploy.update(agglo_kwargs)

        if method.lower() in "minibatch-kmeans":
            clust = MiniBatchKMeans(**minibatchk_kwargs_deploy)
            method = "minibatch-K-Means"
        elif method.lower() in "kmeans":
            clust = KMeans(**kmeans_kwargs_deploy)
            method = "K-Means"
        elif method.lower() in "DBSCAN":
            clust = DBSCAN(**dbscan_kwargs_deploy)
            method = "DBSCAN"
        elif method.lower() in "agglomerative":
            clust = AgglomerativeClustering(**agglo_kwargs_deploy)
            method = "Agglomerative Clustering"
        print(f"Clustering using {method}...")

        df_temp = pd.DataFrame(
            columns=["V"+str(x)
                     for x in range(len(df.loc[df.index[0], input_col]))],
            data=[x[0:] for x in df[input_col]])
        df[output_col] = clust.fit_predict(df_temp)


        cluster_list = sorted(list(set(list(df[output_col]))))
        cluster_nb = len(cluster_list)
        print(f"Getting cluster topics for {cluster_nb} clusters...")

        # reordering cluster number, as they are sometimes offset
        for i, clust_nb in enumerate(cluster_list):
            df.loc[ df[output_col] == clust_nb, "output_col"] = i

        if tokenize_tags is True:
            df["tokenized"] = df.apply(
                    lambda row: " ".join(
                        tokenizer.tokenize(row["text"])),
                    axis=1)
        else:
            df["tokenized"] = df["text"]

        df_by_cluster = df.groupby(["clusters"],
                                   as_index=False).agg({'tokenized': ' '.join})
        count = CountVectorizer().fit_transform(df_by_cluster.tokenized)
        ctfidf = CTFIDFVectorizer().fit_transform(count,
                                                  n_samples=len(
                                                      df_by_cluster.index))
        count_vectorizer = CountVectorizer().fit(df_by_cluster.tokenized)
        count = count_vectorizer.transform(df_by_cluster.tokenized)
        words = count_vectorizer.get_feature_names()
        ctfidf = CTFIDFVectorizer().fit_transform(count,
                                                  n_samples=len(
                                                      df_by_cluster.index
                                                      )).toarray()
        try:
            w_by_class = {str(clust): [
                                       words[index]
                                       for index in
                                       ctfidf[ind].argsort()[-n_topics:]
                                       ] for clust, ind in enumerate(
                                           df_by_cluster.clusters)}
            df["cluster_topic"] = ""
            for i in df.index:
                clst_tpc = " ".join([x for x in w_by_class[
                                                      str(df.loc[i, "clusters"])]])
                df.loc[i, "cluster_topic"] = clst_tpc

            self.w_by_class = w_by_class
            self.df = df.sort_index()

            if add_as_tags is True:
                # creates two threads, the first one removes old tags and the
                # other one adds the new one
                threads = []
                full_note_list = list(set(df["note"].tolist()))
                all_tags =  df["tags"].tolist()
                present_tags = []
                for t in all_tags:
                    if " " in t:
                        present_tags.extend(t.split(" "))
                    else:
                        present_tags.append(t)
                present_tags = list(set(present_tags))
                if "" in present_tags:
                    present_tags.remove("")
                to_remove = list(set([x for x in filter(
                                      lambda x: "AnnA::cluster_topic::" in x,
                                      present_tags)]))
                if len(to_remove) != 0:
                    def _threaded_remove_tags(tags, pbar_r):
                        for tag in tags:
                            self._ankiconnect(action="removeTags",
                                              notes=full_note_list,
                                              tags=str(tag))
                            pbar_r.update(1)
                    pbar_r = tqdm(desc="Removing old cluster tags...",
                              unit="cluster",
                              position=1,
                              total=len(to_remove))
                    thread = threading.Thread(target=_threaded_remove_tags,
                                              args=(to_remove, pbar_r),
                                              daemon=False)
                    thread.start()
                    threads.append(thread)

                df["cluster_topic"] = df["cluster_topic"].str.replace(" ", "_")

                def _threaded_add_cluster_tags(list_i, pbar_a):
                    for i in list_i:
                        cur_time = "_".join(time.asctime().split()[0:4]).replace(
                                ":", "h")[0:-3]
                        newTag = f"AnnA::cluster_topic::{cur_time}::cluster_#{str(i)}"
                        newTag += f"::{df[df['clusters']==i]['cluster_topic'].iloc[0]}"
                        note_list = list(set(df[df["clusters"] == i]["note"].tolist()))
                        self._ankiconnect(action="addTags",
                                          notes=note_list,
                                          tags=newTag)
                        pbar_a.update(1)
                    return True
                pbar_a = tqdm(total=len(cluster_list),
                          desc="Adding new cluster tags",
                          position=0,
                          unit="cluster")
                thread = threading.Thread(
                                    target=_threaded_add_cluster_tags,
                                    daemon=False,
                                    args=(cluster_list, pbar_a))
                thread.start()
                threads.append(thread)

                [t.join() for t in threads]
                pbar_a.close()
                if len(to_remove) != 0:
                    pbar_r.close()

                self._ankiconnect(action="clearUnusedTags")
        except IndexError as e:
            err(f"Index Error when finding cluster topic! {e}")
        return True

    def plot_latent_space(self,
                          specific_index=None,
                          reduce_dim="umap",
                          color_col="tags",
                          hover_cols=["cropped_text",
                                      "tags",
                                      "clusters",
                                      "cluster_topic"],
                          coordinate_col="VEC",
                          disable_legend=True,
                          umap_kwargs=None,
                          plotly_kwargs=None,
                          pca_kwargs=None,
                          ):
        """
        open a browser tab with a 2D plot showing your cards and their relations
        """
        df = self.df.copy()
        pca_kwargs_deploy = {"n_components": 2, "random_state": 42}
        umap_kwargs_deploy = {"n_jobs": -1,
                              "verbose": 1,
                              "n_components": 2,
                              "metric": "cosine",
                              "init": 'spectral',
                              "random_state": 42,
                              "transform_seed": 42,
                              "n_neighbors": 50,
                              "min_dist": 0.1}
        plotly_kwargs_deploy = {"data_frame": df,
                                "title": "AnnA Anki neuronal Appendix",
                                "x": "X",
                                "y": "Y",
                                "hover_data": hover_cols,
                                "color": color_col}
        if umap_kwargs is not None:
            umap_kwargs_deploy.update(umap_kwargs)
        if plotly_kwargs is not None:
            plotly_kwargs_deploy.update(plotly_kwargs)
        if pca_kwargs is not None:
            pca_kwargs_deploy.update(pca_kwargs)

        if reduce_dim is not None:
            if specific_index is not None:
                data = [x[0:] for x in df.loc[specific_index, coordinate_col]]
            else:
                data = [x[0:] for x in df[coordinate_col]]
            df_temp = pd.DataFrame(
                columns=["V"+str(x)
                         for x in
                         range(len(df.loc[df.index[0],
                                   coordinate_col]))
                         ], data=data)
            print(f"Reduce to 2 dimensions using {reduce_dim} before \
plotting...")
        if reduce_dim.lower() in "pca":
            pca_2D = PCA(**pca_kwargs_deploy)
            res = pca_2D.fit_transform(df_temp).T
            x_coor = res[0]
            y_coor = res[1]
        elif reduce_dim.lower() in "umap":
            res = umap.UMAP(**umap_kwargs_deploy).fit_transform(df_temp).T
            x_coor = res[0]
            y_coor = res[1]
        elif reduce_dim is None:
            x_coor = [x[0] for x in df[coordinate_col]],
            x_coor = list(x_coor)[0]
            y_coor = [x[1] for x in df[coordinate_col]],
            y_coor = list(y_coor)[0]

        df["X"] = x_coor
        df["Y"] = y_coor
        print("Plotting results...")
        df["cropped_text"] = df["text"].str[0:75]
        if "clusters" not in df.columns:
            df["clusters"] = 0
        if "cluster_topic" not in df.columns:
            df["cluster_topic"] = 0
        fig = px.scatter(**plotly_kwargs_deploy)
        if disable_legend is True:
            fig = fig.update_layout(showlegend=False)
        fig.show()
        return True

    @classmethod
    def search_for_notes(self,
                         user_input,
                         nlimit=10,
                         user_col="VEC_FULL",
                         do_format_input=False,
                         anki_or_print="anki",
                         dist="cosine",
                         reverse=False):
        """
        given a text input, find notes with highest cosine similarity
        * note that you cannot use the pca version of the sBERT vectors
            otherwise you'd have to re run the whole PCA, so it's quicker
            to just use the full vectors
        * note that if the results are displayed in the browser, the order
            cannot be maintained. So you will get the nlimit best match
            randomly displayed, then the nlimit+1 to 2*nlimit best match
            randomly displayed etc
        * note that this obviously only works using sBERT vectors and not TFIDF
            (they would have to be recomputed each time)
        """
        if self.TFIDF_enable is True:
            err("Cannot search for note using TFIDF vectors, only sBERT can \
be used.")
            return False
        pd.set_option('display.max_colwidth', None)
        df = self.df.copy()

        if do_format_input is True:
            user_input = self._format_text(user_input)
        embed = sBERT.encode(user_input, normalize_embeddings=True)
        print("")
        tqdm.pandas(desc="Searching")
        try:
            df["distance"] = 0.0
            df["distance"] = df[user_col].progress_apply(
                    lambda x: pairwise_distances(embed.reshape(1, -1),
                                                 x.reshape(1, -1),
                                                 metric=dist))
            df["distance"] = df["distance"].astype("float")
        except ValueError as e:
            err(f"Error {e}: did you select 'VEC' instead of \
'VEC_FULL'?")
            return False
        index = df.index
        good_order = sorted(index,
                            key=lambda row: df.loc[row, "distance"],
                            reverse=reverse)
        cnt = 0
        ans = "y"
        while True:
            cnt += 1
            if ans != "n":
                if anki_or_print == "print":
                    print(df.loc[
                                 good_order[
                                     nlimit*cnt:nlimit*(cnt+1)
                                     ], ["text", "distance"]])
                elif anki_or_print == "anki":
                    query = "cid:" + ",".join(
                            [str(x)
                                for x in good_order[
                                    nlimit*cnt:nlimit*(cnt+1)]]
                            )
                    self._ankiconnect(
                            action="guiBrowse",
                            query=query)
            else:
                break
            ans = input("Show more?\n(y/n)>")
        pd.reset_option('display.max_colwidth')
        return True

    def save_df(self, df=None, out_name=None):
        """
        export dataframe as pickle format a DF_backups
        """
        if df is None:
            df = self.df.copy()
        if out_name is None:
            out_name = "AnnA_Saved_DataFrame"
        cur_time = "_".join(time.asctime().split()[0:4]).replace(
                ":", "h")[0:-3]
        name = f"{out_name}_{self.deckname}_{cur_time}.pickle"
        df.to_pickle("./DataFrame_backups/" + name)
        print(f"Dataframe exported to {name}.")
        return True

    def show_acronyms(self, exclude_OCR_text=True):
        """
        shows acronym present in your collection that were not found in
        the file supplied by the argument `acronym_list`
        * acronyms found in OCR caption are removed by default
        """
        full_text = " ".join(self.df["text"].tolist())
        if exclude_OCR_text is True:
            full_text = re.sub("> Caption: '.*?' <", " ", full_text)
        matched = set(re.findall("[A-Z]{3,}", full_text))
        acro_count = {}
        for acr in matched:
            acro_count.update({acr: full_text.count(acr)})
        sorted_by_count = sorted([x for x in matched], key= lambda x: acro_count[x], reverse=True)
        relevant = random.choices(sorted_by_count[0:50],
                k=min(len(sorted_by_count), 10))

        if self.acronym_list is None:
            err("\nYou did not supply an acronym list, printing all acronym \
found...")
            pprint(relevant)
        else:
            acro_list = sorted(self.acronym_dict.keys())
#            print("\nList of acronyms that were already replaced:")
#            print(acro_list)
#            acro_list = [x.lower() for x in acro_list]

            print("List of some acronyms still found:")
            if exclude_OCR_text is True:
                print("(Excluding OCR text)")
            out = {acr for acr in filter(lambda x: x.lower() not in acro_list,
                                         relevant)
                   }
            pprint(sorted(out))
        print("")
        return True


class CTFIDFVectorizer(TfidfTransformer):
    """
    this class is just used to allow class Tf-idf, I took it from:
    https://towardsdatascience.com/creating-a-class-based-tf-idf-with-scikit-learn-caea7b15b858
    """
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


# from https://gist.github.com/beniwohli/765262
greek_alphabet_mapping = {
    u'\u0391': 'Alpha',
    u'\u0392': 'Beta',
    u'\u0393': 'Gamma',
    u'\u0394': 'Delta',
    u'\u0395': 'Epsilon',
    u'\u0396': 'Zeta',
    u'\u0397': 'Eta',
    u'\u0398': 'Theta',
    u'\u0399': 'Iota',
    u'\u039A': 'Kappa',
    u'\u039B': 'Lamda',
    u'\u039C': 'Mu',
    u'\u039D': 'Nu',
    u'\u039E': 'Xi',
    u'\u039F': 'Omicron',
    u'\u03A0': 'Pi',
    u'\u03A1': 'Rho',
    u'\u03A3': 'Sigma',
    u'\u03A4': 'Tau',
    u'\u03A5': 'Upsilon',
    u'\u03A6': 'Phi',
    u'\u03A7': 'Chi',
    u'\u03A8': 'Psi',
    u'\u03A9': 'Omega',
    u'\u03B1': 'alpha',
    u'\u03B2': 'beta',
    u'\u03B3': 'gamma',
    u'\u03B4': 'delta',
    u'\u03B5': 'epsilon',
    u'\u03B6': 'zeta',
    u'\u03B7': 'eta',
    u'\u03B8': 'theta',
    u'\u03B9': 'iota',
    u'\u03BA': 'kappa',
    u'\u03BB': 'lamda',
    u'\u03BC': 'mu',
    u'\u03BD': 'nu',
    u'\u03BE': 'xi',
    u'\u03BF': 'omicron',
    u'\u03C0': 'pi',
    u'\u03C1': 'rho',
    u'\u03C3': 'sigma',
    u'\u03C4': 'tau',
    u'\u03C5': 'upsilon',
    u'\u03C6': 'phi',
    u'\u03C7': 'chi',
    u'\u03C8': 'psi',
    u'\u03C9': 'omega',
}

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
import numpy as np

# avoids annoying warning
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# makes the script interuptible, resume it using c+enter
signal.signal(signal.SIGINT, (lambda signal, frame: pdb.set_trace()))



def asynchronous_importer():
    """
    used to asynchronously import large modules, this way between
    importing AnnA and creating the instance of the class, the language model
    have some more time to load
    """
    global stopwords, SentenceTransformer, KMeans, DBSCAN, \
        AgglomerativeClustering, transformers, sp, normalize, TfidfVectorizer,\
        CountVectorizer, TruncatedSVD, StandardScaler, \
        pairwise_distances, PCA, px, umap, np, tokenizer_bert, sBERT, \
        MiniBatchKMeans
    print("Importing modules.")
    from nltk.corpus import stopwords
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.feature_extraction.text import CountVectorizer
    from sentence_transformers import SentenceTransformer
    sBERT = SentenceTransformer('distiluse-base-multilingual-cased-v1')
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    from sklearn.cluster import MiniBatchKMeans
    import plotly.express as px
    import umap.umap_
    from sklearn.metrics import pairwise_distances
    from sklearn.preprocessing import normalize
    from sklearn.decomposition import TruncatedSVD
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    print("Finished importing modules")
import_thread = threading.Thread(target=asynchronous_importer, daemon=True)
import_thread.start()


class AnnA:
    """
    main class: used to centralize everything
    just instantiating the class does most of the job, as you can see
    in self.__init__
    """
    def __init__(self,
                 deckname=None,
                 replace_greek=True,
                 field_mapping="field_mapping.py",
                 optional_acronym_list="acronym_list.py",
                 keep_ocr=True,
                 desired_deck_size="75%",
                 rated_last_X_days=2,
                 rated_last_X_cards=2000,
                 show_banner=True,
                 debug_card_limit=None,
                 n_clusters="auto",
                 pca_sBERT_dim=300,
                 stride=2500,
                 prefer_similar_card=False,
                 scoring_weights=(1, 1.3),
                 reference_order="relative_overdueness",
                 compute_opti_rev_order=True,
                 send_to_anki=False,
                 ):
        # printing banner
        if show_banner is True:
            print(pyfiglet.figlet_format("AnnA"))
            print("(Anki neuronal Appendix)\n\n")

        # loading args etc
        self.deckname = self._check_deck(deckname)
        self.replace_greek = replace_greek
        self.keep_ocr = keep_ocr
        self.desired_deck_size = desired_deck_size
        self.rated_last_X_days = rated_last_X_days
        self.rated_last_X_cards = rated_last_X_cards
        self.debug_card_limit = debug_card_limit
        self.n_clusters = n_clusters
        self.pca_sBERT_dim = pca_sBERT_dim
        self.stride = stride
        self.prefer_similar_card = prefer_similar_card
        self.scoring_weights = scoring_weights
        assert reference_order in ["lowest_interval", "relative_overdueness"]
        self.reference_order = reference_order
        self.field_mapping = field_mapping
        self.optional_acronym_list = optional_acronym_list

        if self.optional_acronym_list is not None:
            file = Path(optional_acronym_list)
            if not file.exists():
                raise Exception(f"Acronym file was not found: {optional_acronym_list}")
            else:
                imp = importlib.import_module(
                        optional_acronym_list.replace(".py", ""))
                self.acronym_list = imp.acronym_list
        if self.field_mapping is not None:
            file = Path(self.field_mapping)
            try:
                assert file.exists()
                imp = importlib.import_module(
                        self.field_mapping.replace(".py", ""))
                self.field_dic = imp.field_dic
            except Exception:
                print("Error with field mapping file, will use default \
values.")
                self.field_dic = {"dummyvalue": "dummyvalue"}

        # actual execution
        import_thread.join()  # asynchronous importing of large module
        time.sleep(0.5)  # sometimes import_thread takes too long apparently
        self._create_and_fill_df()
        self.scaler = StandardScaler()
        self.df = self._reset_index_dtype(self.df)
        self._format_card()
        self._compute_sBERT_vec()
        self._compute_distance_matrix()
        if compute_opti_rev_order is True:
            self._compute_opti_rev_order()
            if send_to_anki is True:
                self.send_to_anki()

    def _reset_index_dtype(self, df):
        """
        the index dtype (cardId) somehow gets turned into float so I
        occasionally turn it back into int
        """
        temp = df.reset_index()
        temp["cardId"] = temp["cardId"].astype(int)
        df = temp.set_index("cardId")
        return df

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
                target_thread_n = 10
                batchsize = len(card_id)//target_thread_n+3
                print(f"Large number of cards to retrieve, creating 10 \
threads of {batchsize} cards to fetch {len(card_id)} cards...")

                def retrieve_cards(card_list, lock, cnt, r_list):
                    "for multithreaded card retrieval"
                    out_list = self._ankiconnect(action="cardsInfo",
                                                        cards=card_list)
                    lock.acquire()
                    r_list.extend(out_list)
                    pbar.update(1)
                    lock.release()
                    return True

                with tqdm(total=target_thread_n,
                          unit="thread",
                          dynamic_ncols=True,
                          desc="Finished threads",
                          delay=1,
                          smoothing=0) as pbar:
                    for nb in range(0, len(card_id), batchsize):
                        cnt += 1
                        temp_card_id = card_id[nb: nb+batchsize]
                        thread = threading.Thread(target=retrieve_cards,
                                                  args=(temp_card_id,
                                                        lock,
                                                        cnt,
                                                        r_list),
                                                  daemon=True)
                        thread.start()
                        threads.append(thread)
                        time.sleep(0.1)
                    print("")
                    for t in threads:
                        t.join()
                assert len(r_list) == len(card_id)
                r_list = sorted(r_list,
                                key=lambda x: x["cardId"],
                                reverse=False)
                return r_list

        if isinstance(card_id, int):
            return self._ankiconnect(action="cardsInfo",
                                            cards=[card_id])

    def _check_deck(self, deckname):
        """
        used to check if the deckname is correct
        if incorrect, user is asked to enter the name, using autocompletion
        """
        decklist = self._ankiconnect(action="deckNames") + ["*"]
        if deckname is not None:
            if deckname not in decklist:
                print("Couldn't find this deck.", end=" ")
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
        return deckname

    def _create_and_fill_df(self):
        """
        create a pandas DataFrame, fill it with the information gathered from
        anki connect like card content, intervals, etc
        """

        print("Getting due card from this deck...")
        n_rated_days = int(self.rated_last_X_days)
        query = f"deck:{self.deckname} is:due is:review -is:learn \
-is:suspended -is:buried -is:new -rated:1"
        print(" >  '" + query + "'")
        due_cards = self._ankiconnect(action="findCards", query=query)

        if n_rated_days is not None and n_rated_days != 0:
            print(f"Getting cards that where rated in the last {n_rated_days} days \
from this deck...")
            query = f"deck:{self.deckname} rated:{n_rated_days} -is:suspended"
            print(" >  '" + query + "'")
            r_cards = self._ankiconnect(action="findCards", query=query)

            # removes overlap if found
            rated_cards = [x for x in r_cards if x not in due_cards]
            print(f"Rated cards contained {len(rated_cards)} relevant cards \
(from {len(r_cards)})")

            if self.rated_last_X_cards is not None and \
                    len(rated_cards) > self.rated_last_X_cards:
                print(f"Found {len(rated_cards)} cards rated in the last few days, but \
will only keep {self.rated_last_X_cards} to ease calculation.")
                rated_cards = rated_cards[:self.rated_last_X_cards]
        else:
            print("Will not look for cards rated in past days.")
            rated_cards = []
        self.due_cards_list = due_cards
        self.rated_cards_list = rated_cards

        limit = self.debug_card_limit if self.debug_card_limit else -1
        combined_card_list = list(rated_cards + due_cards)[:limit]

        if len(combined_card_list) < 50:
            raise Exception("You don't have enough cards!\nExiting.")

        list_cardInfo = []

        n = len(combined_card_list)
        print(f"Asking Anki for information about {n} cards...")
        start = time.time()
        list_cardInfo.extend(
                self._get_cards_info_from_card_id(
                    card_id=combined_card_list))
        print(f"Extracted information in {int(time.time()-start)} seconds.")

        for i, card in enumerate(tqdm(list_cardInfo,
                                      desc="Filtering only relevant fields...",
                                      unit="card")):
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
                print(f"Error processing card with ID {card['cardId']}")

        if len(list_cardInfo) != len(list(set(combined_card_list))):
            print("Duplicate elements!\nExiting.")
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
        text = str(text).replace("\n", " ")
        if self.keep_ocr is True:
            # keep image title (usually OCR)
            text = re.sub("title=(\".*?\")", "> Caption: '\\1' <", text)
        if self.replace_greek is True:
            for a, b in greek_alphabet_mapping.items():
                text = re.sub(a, b, text)
        if self.optional_acronym_list is True:
            global acronym_list
            for a, b in self.acronym_list.items():
                text = re.sub(rf"\b{a}\b", f"{a} ({b})", text)
                # \b matches beginning and end of a word
        text = re.sub(r'[a-zA-Z0-9-]+\....', " ", text)  # media file name
        text = re.sub('\\n|<div>|</div>|<br>|<span>|</span>|<li>|</li>|<ul>|\
</ul>',
                      " ", text)  # newline
        text = re.sub("<a href.*?</a>", " ", text)  # html links
        text = re.sub(r'http[s]?://\S*', " ", text)  # plaintext links
        text = re.sub("<.*?>", " ", text)  # remaining html tags
        text = re.sub('\u001F|&nbsp;', " ", text)
        text = re.sub(r"{{c\d+?::", "", text)
        text = re.sub("{{c|{{|}}|::", " ", text)
        text = re.sub(r"\d", " ", text)
        text = text.replace("&amp;", "&")
        text = text.replace("&gt;", ">")
        text = text.replace("&lt;", "<")
        text = " ".join(text.split())  # multiple spaces
        text = text.strip()
        return text

    def _format_card(self):
        """
        filter the fields of each card and keep only the relevant fields
        a "relevant field" is one that is mentionned in the variable field_dic
        which can be found at the top of the file. If not relevant field are
        found then only the first field is kept.
        """
        df = self.df

        for index in tqdm(df.index, desc="Parsing text content", unit="card"):
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
            df.loc[index, "comb_text"] = comb_text.strip().replace(": :", "")
        df["text"] = [self._format_text(x)
                      for x in tqdm(
                      df["comb_text"],
                      desc="Formating text")]
        print("\n\n5 random samples of your formated text, to help \
troubleshoot formating issues:")
        pd.set_option('display.max_colwidth', 100)
        print(df.sample(5)["text"], end="\n")
        pd.reset_option('display.max_colwidth')
        print("\n\n")
        self.df = df.sort_index()
        return True

    def _compute_sBERT_vec(self, df=None, use_sBERT_cache=True):
        """
        Assigne sBERT vectors to each card
        df["sBERT_before_pca"] if exists, contains the vectors from sBERT
        df["sBERT"] contains either all the vectors from sBERT or less if you
            enabled pca reduction
        * given how long it is to compute the vectors I decided to store
            all already computed sBERT to a pickled DataFrame at each run
        """
        if df is None:
            df = self.df

        if use_sBERT_cache is True:
            print("\nLooking for cached sBERT pickle file...", end="")
            sBERT_file = Path("./sBERT_cache.pickle")
            df["sBERT"] = 0*len(df.index)
            df["sBERT"] = df["sBERT"].astype("object")
            loaded_sBERT = 0
            id_to_recompute = []

            # reloads sBERT vectors and only recomputes the new one:
            if not sBERT_file.exists():
                print(" sBERT cache not found, will create it.")
                df_cache = pd.DataFrame(
                        columns=["cardId", "mod", "text", "sBERT"]
                        ).set_index("cardId")
                id_to_recompute = df.index
            else:
                print(" Found sBERT cache.")
                df_cache = pd.read_pickle(sBERT_file)
                df_cache["sBERT"] = df_cache["sBERT"].astype("object")
                df_cache["mod"] = df_cache["mod"].astype("object")
                df_cache["text"] = df_cache["text"]
                df_cache = self._reset_index_dtype(df_cache)
                for i in df.index:
                    if i in df_cache.index and \
                            (str(df_cache.loc[i, "text"]) ==
                                str(df.loc[i, "text"])):
                        df.at[i, "sBERT"] = df_cache.loc[i, "sBERT"]
                        loaded_sBERT += 1
                    else:
                        id_to_recompute.append(i)

            print(f"Loaded {loaded_sBERT} vectors from cache, will compute \
{len(id_to_recompute)} others...")
            if len(id_to_recompute) != 0:
                sentence_list = [df.loc[x, "text"]
                                 for x in df.index if x in id_to_recompute]
                sentence_embeddings = sBERT.encode(sentence_list,
                                                   normalize_embeddings=True,
                                                   show_progress_bar=True)

                for i, ind in enumerate(tqdm(id_to_recompute)):
                    df.at[ind, "sBERT"] = sentence_embeddings[i]

            # stores newly computed sBERT vectors in a file:
            df_cache = self._reset_index_dtype(df_cache)
            for i in [x for x in id_to_recompute if x not in df_cache.index]:
                df_cache.loc[i, "sBERT"] = df.loc[i, "sBERT"].astype("object")
                df_cache.loc[i, "mod"] = df.loc[i, "mod"].astype("object")
                df_cache.loc[i, "text"] = df.loc[i, "text"]
            for i in [x for x in id_to_recompute if x in df_cache.index]:
                df_cache.loc[i, "sBERT"] = df.loc[i, "sBERT"].astype("object")
                df_cache.loc[i, "mod"] = df.loc[i, "mod"].astype("object")
                df_cache.loc[i, "text"] = df.loc[i, "text"]
            df_cache = self._reset_index_dtype(df_cache)
            df_cache.to_pickle("sBERT_cache.pickle")

        if self.pca_sBERT_dim is not None:
            print(f"Reducing dimension of sBERT to {self.pca_sBERT_dim} \
using PCA...")
            pca_sBERT = PCA(n_components=self.pca_sBERT_dim, random_state=42)
            df_temp = pd.DataFrame(
                columns=["V"+str(x+1)
                         for x in range(len(df.loc[df.index[0], "sBERT"]))],
                data=[x[0:] for x in df["sBERT"]])
            out = pca_sBERT.fit_transform(df_temp)
            print(f"Explained variance ratio after PCA on sBERT: \
{round(sum(pca_sBERT.explained_variance_ratio_)*100,1)}%")
            df["sBERT_before_pca"] = df["sBERT"]
            df["sBERT"] = [x for x in out]
            return True

    def _compute_distance_matrix(self, method="cosine", input_col="sBERT"):
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

        print("\nComputing the distance matrix...", end="")
        df_dist = pairwise_distances(df_temp, n_jobs=-1, metric=method)
        print(" Done.\n")

        self.df_dist = df_dist
        return True

    def _compute_opti_rev_order(self):
        """
        assign score to each card
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
        * the queue starts empty. At each turn, the_chosen_one is added to it
        """
        print("Computing similarity scores...")
        reference_order = self.reference_order
        df = self.df
        df_dist = self.df_dist
        if self.prefer_similar_card is True:
            sign = 1
        else:
            sign = -1
        w1 = self.scoring_weights[0]
        w2 = self.scoring_weights[1]*sign
        desired_deck_size = self.desired_deck_size

        ivl = df['interval'].to_numpy().reshape(-1, 1)
        df["interval_cs"] = self.scaler.fit_transform(ivl)

        if reference_order == "relative_overdueness":
            for i in df.index:
                df.at[i, "ref_due"] = df.loc[i, "odue"]
                if df.loc[i, "ref_due"] == 0:
                    df.at[i, "ref_due"] = df.at[i, "due"]
            anki_col_time = int(self._ankiconnect(
                action="runConsoleCmd",
                cmd="print(aqt.mw.col.crt)",
                token="I understand that calling this is a security risk!"
                ).strip())

            time_offset = int((time.time() - anki_col_time) // 86400)
            overdue = (df["ref_due"] - time_offset).to_numpy().reshape(-1, 1)

            ro = -1 * (df["interval"].values + 0.001) / (overdue.T + 0.001)
            df["ref"] = self.scaler.fit_transform(ro.T)
        else:  # then is "lowest interval"
            df["ref"] = df["interval_cs"]

        rated = self.rated_cards_list
        assert len([x for x in rated if df.loc[x, "status"] != "rated"]) == 0
        queue = []
        print(f"Cards rated in the past relevant days: {len(rated)}")

        if isinstance(desired_deck_size, float):
            if desired_deck_size < 1.0:
                desired_deck_size = str(desired_deck_size*100) + "%"
        if isinstance(desired_deck_size, str):
            if desired_deck_size in ["all", "100%"]:
                print("Taking the whole deck.")
                desired_deck_size = len(self.due_cards_list)
            elif desired_deck_size.endswith("%"):
                print(f"Taking {desired_deck_size[:-1]}% of the deck.")
                desired_deck_size = 0.01*int(desired_deck_size[:-1])*(
                            len(df.index)-len(rated)
                            )
        desired_deck_size = int(desired_deck_size)

        if desired_deck_size > int(len(df.index)-len(rated)):
            print(f"You wanted to create a deck with \
{desired_deck_size} in it but the deck only contains \
{len(df.index)-len(rated)} cards. Taking the lowest value.")
        queue_size_goal = min(desired_deck_size,
                              len(df.index)-len(rated))

        if len(rated) < 3:
            pool = df["ref"].nsmallest(
                    n=min(50, len(self.due_cards_list))
                    ).index
            queue.extend(random.choices(pool, k=3))

        df_temp = pd.DataFrame(columns=rated, index=df.index)
        with tqdm(desc="Computing optimal review order",
                  unit=" card",
                  total=queue_size_goal) as pbar:
            while len(queue) < queue_size_goal:
                for q in list(rated + queue)[-self.stride:-1]:
                    df_temp[q] = df_dist[df.index.get_loc(q)]
                df["score"] = w1*df["ref"] + w2*np.min(df_temp, axis=1)
                chosen_one = df.drop(index=list(rated+queue))["score"].idxmin()
                queue.append(chosen_one)
                df_temp[chosen_one] = df_dist[df.index.get_loc(chosen_one)]
                pbar.update(1)

        print("Done. Now all that is left is to send all of this to anki.\n")
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

    def send_to_anki(self, deck_template="AnnA - Optimal Review Order"):
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
        """

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
            create 10 threads to edit card values quickly
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
                    lock.acquire()
                    pbar.update(1)
                    lock.release()
                return True

            with tqdm(desc=tqdm_desc,
                      unit="card",
                      total=len(card_list),
                      dynamic_ncols=True,
                      smoothing=1) as pbar:
                lock = threading.Lock()
                threads = []
                batchsize = len(card_list)//10+1
                for nb in range(0, len(card_list), batchsize):
                    sub_card_list = card_list[nb: nb+batchsize]
                    thread = threading.Thread(target=do_action,
                                              args=(card_list,
                                                    sub_card_list,
                                                    keys,
                                                    newValues,
                                                    lock,
                                                    pbar),
                                              daemon=True)
                    thread.start()
                    threads.append(thread)
                for t in threads:
                    t.join()
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
            print("Inconsistency! The deck does not contain the same cards \
as opti_rev_order!")
            pprint(diff)
            print(f"\nNumber of inconsistent cards: {len(diff)}")
        else:
            print(" Done.")

        _threaded_value_setter(card_list=self.opti_rev_order,
                               tqdm_desc="Altering due order",
                               keys=["due"],
                               newValues=None)
        print("Re-optimizing Anki database")
        self._ankiconnect(action="guiCheckDatabase")
        print("All done!\n\n")
        return True

    def compute_clusters(self,
                         method="minibatch-kmeans",
                         input_col="sBERT",
                         output_col="clusters",
                         n_topics=5,
                         minibatchk_kwargs=None,
                         kmeans_kwargs=None,
                         agglo_kwargs=None,
                         dbscan_kwargs=None,
                         add_as_tags=True):
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
        if self.n_clusters is None or self.n_clusters == "auto":
            self.n_clusters = len(df.index)//100
            print(f"No number of clustrs supplied, will try {self.n_clusters}")
        kmeans_kwargs_deploy = {"n_clusters": self.n_clusters}
        dbscan_kwargs_deploy = {"eps": 0.75,
                                "min_samples": 3,
                                "n_jobs": -1}
        agglo_kwargs_deploy = {"n_clusters": self.n_clusters,
                               # "distance_threshold": 0.6,
                               "affinity": "cosine",
                               "memory": "/tmp/",
                               "linkage": "average"}
        minibatchk_kwargs_deploy = {"n_clusters": self.n_clusters,
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

        cluster_list = list(set(list(df[output_col])))
        cluster_nb = len(cluster_list)
        print(f"Getting cluster topics for {cluster_nb} clusters...")
        df_by_cluster = df.groupby(["clusters"],
                                   as_index=False).agg({'text': ' '.join})
        count = CountVectorizer().fit_transform(df_by_cluster.text)
        ctfidf = CTFIDFVectorizer().fit_transform(count,
                                                  n_samples=len(
                                                      df_by_cluster.index))
        count_vectorizer = CountVectorizer().fit(df_by_cluster.text)
        count = count_vectorizer.transform(df_by_cluster.text)
        words = count_vectorizer.get_feature_names()
        ctfidf = CTFIDFVectorizer().fit_transform(count,
                                                  n_samples=len(
                                                      df_by_cluster.index
                                                      )).toarray()
        w_by_class = {str(label): [
                                   words[index]
                                   for index in
                                   ctfidf[label].argsort()[-n_topics:]
                                   ] for label in df_by_cluster.clusters}
        df["cluster_topic"] = ""
        for i in df.index:
            clst_tpc = " ".join([x for x in w_by_class[
                                                  str(df.loc[i, "clusters"])]])
            df.loc[i, "cluster_topic"] = clst_tpc

        self.w_by_class = w_by_class
        self.df = df.sort_index()

        if add_as_tags is True:
            df["cluster_topic"] = df["cluster_topic"].str.replace(" ", "_")
            cluster_list = list(set(list(df["clusters"])))
            for i in tqdm(cluster_list, desc="Adding cluster tags",
                          unit="cluster"):
                cur_time = "_".join(time.asctime().split()[0:4]).replace(
                        ":", "h")[0:-3]
                newTag = f"AnnA::cluster_topic::{cur_time}::cluster_#{str(i)}"
                newTag += f"::{df[df['clusters']==i]['cluster_topic'].iloc[0]}"
                note_list = list(df[df["clusters"] == i]["note"])
                self._ankiconnect(action="addTags",
                                  notes=note_list,
                                  tags=newTag)
        return True

    def plot_latent_space(self,
                          specific_index=None,
                          reduce_dim="umap",
                          color_col="tags",
                          hover_cols=["cropped_text",
                                      "tags",
                                      "clusters",
                                      "cluster_topic"],
                          coordinate_col="sBERT",
                          disable_legend=True,
                          umap_kwargs=None,
                          plotly_kwargs=None,
                          pca_kwargs=None,
                          ):
        """
        open a browser tab with a 2D plot showing your cards and their relations
        """
        df = self.df
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

    def search_for_notes(self,
                         user_input,
                         nlimit=10,
                         user_col="sBERT_before_pca",
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
        """
        pd.set_option('display.max_colwidth', None)
        df = self.df

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
            print(f"Error {e}: did you select 'sBERT' instead of \
'sBERT_before_pca'?")
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
            df = self.df
        if out_name is None:
            out_name = "AnnA_Saved_DF"
        cur_time = "_".join(time.asctime().split()[0:4]).replace(
                ":", "h")[0:-3]
        name = f"{out_name}_{self.deckname}_{cur_time}.pickle"
        df.to_pickle("./DataFrame_backups/" + name)
        print(f"Dataframe exported to {name}")
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

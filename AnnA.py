import logging
import gc
import pickle
import time
import random
import pdb
import signal
import os
import json
import urllib.request
import pyfiglet
from pprint import pprint
from tqdm import tqdm
import re
import importlib
from pathlib import Path
import threading
from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter

import pandas as pd
import numpy as np
import Levenshtein as lev
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-uncased")

from scipy import interpolate, sparse
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer, CountVectorizer
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, MiniBatchKMeans
from sklearn.preprocessing import normalize, StandardScaler
import plotly.express as px

# avoids annoying warning
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# makes the script interuptible, resume it using c+enter
signal.signal(signal.SIGINT, (lambda signal, frame: pdb.set_trace()))

# adds logger, restrict it to 5000 lines
Path("logs.txt").write_text(
    "\n".join(
        Path("logs.txt").read_text().split("\n")[-5000:]))
logging.basicConfig(filename="logs.txt",
                    filemode='a',
                    format=f"{time.asctime()}: %(message)s")
log = logging.getLogger()
log.setLevel(logging.INFO)


def set_global_logging_level(level=logging.ERROR, prefices=[""]):
    """
    https://github.com/huggingface/transformers/issues/3050#issuecomment-682167272
    To control logging level for various modules used in the application:
    """
    prefix_re = re.compile(fr'^(?:{ "|".join(prefices) })')
    for name in logging.root.manager.loggerDict:
        if re.match(prefix_re, name):
            logging.getLogger(name).setLevel(level)


def coloured_log(color_asked):
    col_red = "\033[91m"
    col_yel = "\033[93m"
    col_rst = "\033[0m"

    if color_asked == "white":
        def printer(string):
            string = str(string)
            log.info(string)
            tqdm.write(col_rst + string + col_rst)
    elif color_asked == "yellow":
        def printer(string):
            string = str(string)
            log.warn(string)
            tqdm.write(col_yel + string + col_rst)
    elif color_asked == "red":
        def printer(string):
            string = str(string)
            log.error(string)
            tqdm.write(col_red + string + col_rst)
    return printer


whi = coloured_log("white")
yel = coloured_log("yellow")
red = coloured_log("red")

set_global_logging_level(logging.ERROR,
                         ["transformers", "nlp", "torch",
                          "tensorflow", "sklearn", "nltk",
                          "fastText"])


def asynchronous_importer(vectorizer, task, fastText_lang, fastText_model_name):
    """
    used to asynchronously import large modules, this way between
    importing AnnA and creating the instance of the class, the language model
    have some more time to load
    """
    if vectorizer == "fastText" or task == "index":
        if "ft" not in globals():
            global fastText, ft
            import fasttext as fastText
            import fasttext.util
            try:
                fasttext.util.download_model(fastText_lang, if_exists='ignore')
                if fastText_model_name is None:
                    ft = fastText.load_model(f"cc.{fastText_lang[0:2]}.300.bin")
                else:
                    ft = fastText.load_model(fastText_model_name)
            except Exception as e:
                red(f"Couldn't load fastText model: {e}")
                raise SystemExit()


class AnnA:
    """
    main class: used to centralize everything
    just instantiating the class does most of the job, as you can see
    in self.__init__
    """
    def __init__(self, show_banner=True,
                 # main settings
                 deckname=None,
                 reference_order="relative_overdueness",
                 # can be "lowest_interval", "relative overdueness",
                 # "order_added"
                 target_deck_size="80%",
                 rated_last_X_days=4,
                 due_threshold=30,
                 highjack_due_query=None,
                 highjack_rated_query=None,
                 queue_stride=10_000,
                 score_adjustment_factor=(1, 5),
                 log_level=2,
                 replace_greek=True,
                 keep_ocr=True,
                 field_mappings="field_mappings.py",
                 acronym_list="acronym_list.py",

                 # steps:
                 clustering_enable=False,
                 clustering_nb_clust="auto",
                 compute_opti_rev_order=True,
                 task="filter_review_cards",
                 # can be "filter_review_cards",
                 # "bury_excess_review_cards",
                 # "bury_excess_learning_cards",
                 # "index"
                 check_database=False,

                 # vectorization:
                 vectorizer="TFIDF",  # can be "TFIDF" or "fastText"
                 fastText_dim=None,
                 fastText_model_name=None,
                 fastText_lang="en",
                 TFIDF_dim=100,
                 TFIDF_stopw_lang=["english", "french"],
                 TFIDF_stem=False,
                 TFIDF_tokenize=True,

                 # misc:
                 debug_card_limit=None,
                 save_instance_as_pickle=False,
                 ):

        if show_banner:
            red(pyfiglet.figlet_format("AnnA"))
            red("(Anki neuronal Appendix)\n\n")

        # miscellaneous
        if log_level == 0:
            log.setLevel(logging.ERROR)
        elif log_level == 1:
            log.setLevel(logging.WARNING)
        elif log_level >= 2:
            log.setLevel(logging.DEBUG)
        else:
            log.setLevel(logging.INFO)

        if vectorizer == "fasttext":
            vectorizer = "fastText"

        # start importing large modules
        import_thread = threading.Thread(target=asynchronous_importer,
                                         args=(vectorizer,
                                               task,
                                               fastText_lang,
                                               fastText_model_name))
        import_thread.start()

        # loading args
        self.replace_greek = replace_greek
        self.keep_ocr = keep_ocr
        self.target_deck_size = target_deck_size
        self.rated_last_X_days = rated_last_X_days
        self.due_threshold = due_threshold
        self.debug_card_limit = debug_card_limit
        self.clustering_nb_clust = clustering_nb_clust
        self.highjack_due_query = highjack_due_query
        self.highjack_rated_query = highjack_rated_query
        self.queue_stride = queue_stride
        self.score_adjustment_factor = score_adjustment_factor
        self.reference_order = reference_order
        self.field_mappings = field_mappings
        self.acronym_list = acronym_list
        self.vectorizer = vectorizer
        self.fastText_lang = fastText_lang
        self.fastText_dim = fastText_dim
        self.fastText_model_name = fastText_model_name
        self.TFIDF_dim = TFIDF_dim
        self.TFIDF_stopw_lang = TFIDF_stopw_lang
        self.TFIDF_stem = TFIDF_stem
        self.TFIDF_tokenize = TFIDF_tokenize
        self.task = task
        self.save_instance_as_pickle = save_instance_as_pickle

        # args sanity checks
        if isinstance(self.target_deck_size, int):
            self.target_deck_size = str(self.target_deck_size)
        assert TFIDF_stem + TFIDF_tokenize in [0, 1]
        assert queue_stride > 0
        assert reference_order in ["lowest_interval", "relative_overdueness",
                                   "order_added"]
        assert task in ["filter_review_cards", "index",
                        "bury_excess_learning_cards",
                        "bury_excess_review_cards"]
        assert vectorizer in ["TFIDF", "fastText"]

        if self.acronym_list is not None:
            file = Path(acronym_list)
            if not file.exists():
                raise Exception(f"Acronym file was not found: {acronym_list}")
            else:
                imp = importlib.import_module(
                    acronym_list.replace(".py", ""))
                acronym_dict = imp.acronym_dict
                compiled_dic = {}
                for ac in acronym_dict:
                    if ac.lower() == ac:
                        compiled = re.compile(r"\b" + ac + r"\b",
                                              flags=re.IGNORECASE |
                                              re.MULTILINE |
                                              re.DOTALL)
                    else:
                        compiled = re.compile(r"\b" + ac + r"\b",
                                              flags=re.MULTILINE | re.DOTALL)
                    compiled_dic[compiled] = acronym_dict[ac]
                self.acronym_dict = compiled_dic

        if self.field_mappings is not None:
            f = Path(self.field_mappings)
            try:
                assert f.exists()
                imp = importlib.import_module(
                    self.field_mappings.replace(".py", ""))
                self.field_dic = imp.field_dic
            except Exception as e:
                red(f"Error with field mapping file, will use default \
values. {e}")
                self.field_dic = {"dummyvalue": "dummyvalue"}

        try:
            stops = []
            for lang in self.TFIDF_stopw_lang:
                stops += stopwords.words(lang)
            if self.TFIDF_tokenize:
                temp = []
                [temp.extend(tokenizer.tokenize(x)) for x in stops]
                stops.extend(temp)
            elif self.TFIDF_stem:
                ps = PorterStemmer()
                stops += [ps.stem(x) for x in stops]
            self.stops = list(set(stops))
        except Exception as e:
            red(f"Error when extracting stop words: {e}")
            red("Setting stop words list to None.")
            self.stops = None

        # actual execution
        self.deckname = self._check_deck(deckname, import_thread)
        yel(f"Selected deck: {self.deckname}\n")
        if task == "index":
            yel(f"Task : cache vectors of deck: {self.deckname}\n")
            self.vectorizer = "fastText"
            self.fastText_dim = None
            self.rated_last_X_days = 0
            self._create_and_fill_df()
            if self.not_enough_cards is True:
                return
            self.df = self._reset_index_dtype(self.df)
            self._format_card()
            self.show_acronyms()
            self._compute_card_vectors(import_thread=import_thread)
            if clustering_enable:
                self.compute_clusters(minibatchk_kwargs={"verbose": 0})

        elif task in ["bury_excess_learning_cards",
                      "bury_excess_review_cards"]:
            # bypasses most of the code to bury learning cards
            # directly in the deck without creating filtered decks
            if task == "bury_excess_learning_cards":
                yel("Task : bury some learning cards")
            elif task == "bury_excess_review_cards":
                yel("Task : bury some reviews\n")
            self._create_and_fill_df()
            if self.not_enough_cards is True:
                return
            self.df = self._reset_index_dtype(self.df)
            self._format_card()
            self.show_acronyms()
            self._compute_card_vectors(import_thread=import_thread)
            if clustering_enable:
                self.compute_clusters(minibatchk_kwargs={"verbose": 0})
            self._compute_distance_matrix()
            self._compute_opti_rev_order()
            self.task_filtered_deck(task=task)
        else:
            yel("Task : created filtered deck containing review cards")
            self._create_and_fill_df()
            if self.not_enough_cards is True:
                return
            self.df = self._reset_index_dtype(self.df)
            self._format_card()
            self.show_acronyms()
            self._compute_card_vectors(import_thread=import_thread)
            if clustering_enable:
                self.compute_clusters(minibatchk_kwargs={"verbose": 0})
            self._compute_distance_matrix()
            if compute_opti_rev_order:
                self._compute_opti_rev_order()
                if task == "filter_review_cards":
                    self.task_filtered_deck()

        # pickle itself
        self._collect_memory()
        if save_instance_as_pickle:
            yel("\nSaving instance as 'last_run.pickle'...")
            if Path("./last_run.pickle").exists():
                Path("./last_run.pickle").unlink()
            with open("last_run.pickle", "wb") as f:
                try:
                    pickle.dump(self, f)
                    whi("Done! You can now restore this instance of AnnA without having to \
execute the code using:\n'import pickle ; a = pickle.load(open(\"last_run.pickle\
\", \"rb\"))'")
                except TypeError as e:
                    red(f"Error when saving instance as pickle file: {e}")

        if check_database:
            whi("Re-optimizing Anki database")
            self._ankiconnect(action="guiCheckDatabase")

        yel(f"Done with '{self.task}' on deck {self.deckname}")

    def _collect_memory(self):
        gc.collect()

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
                batchsize = len(card_id) // target_thread_n + 3
                whi(f"(Large number of cards to retrieve: creating 10 \
threads of size {batchsize})")

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
                        temp_card_id = card_id[nb: nb + batchsize]
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
                red("Couldn't find this deck.")
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
        self._collect_memory()

        if self.highjack_due_query is not None:
            red("Highjacking due card list:")
            query = self.highjack_due_query
            red(" >  '" + query)
            due_cards = self._ankiconnect(action="findCards", query=query)
            whi(f"Found {len(due_cards)} cards...\n")

        elif self.task == "filter_review_cards":
            yel("Getting due card list...")
            query = f"\"deck:{self.deckname}\" is:due is:review -is:learn \
-is:suspended -is:buried -is:new -rated:1"
            whi(" >  '" + query)
            due_cards = self._ankiconnect(action="findCards", query=query)
            whi(f"Found {len(due_cards)} due cards...\n")

        elif self.task == "bury_excess_review_cards":
            yel("Getting due card list...")
            query = f"\"deck:{self.deckname}\" is:due is:review -is:learn \
-is:suspended -is:buried -is:new -rated:1"
            whi(" >  '" + query)
            due_cards = self._ankiconnect(action="findCards", query=query)
            whi(f"Found {len(due_cards)} reviews...\n")

        elif self.task == "bury_excess_learning_cards":
            yel("Getting is:learn card list...")
            query = f"\"deck:{self.deckname}\" is:learn -is:suspended is:due \
-rated:1 -rated:2:1"
            whi(" >  '" + query)
            due_cards = self._ankiconnect(action="findCards", query=query)
            whi(f"Found {len(due_cards)} learning cards...\n")

        elif self.task == "index":
            yel("Getting all cards from deck...")
            query = f"\"deck:{self.deckname}\" -is:suspended"
            whi(" >  '" + query)
            due_cards = self._ankiconnect(action="findCards", query=query)
            whi(f"Found {len(due_cards)} cards...\n")

        rated_cards = []
        if self.highjack_rated_query is not None:
            red("Highjacking rated card list:")
            query = self.highjack_rated_query
            red(" >  '" + query)
            rated_cards = self._ankiconnect(action="findCards", query=query)
            red(f"Found {len(rated_cards)} cards...\n")
        elif self.rated_last_X_days != 0:
            yel(f"Getting cards that where rated in the last \
{self.rated_last_X_days} days...")
            query = f"\"deck:{self.deckname}\" rated:{self.rated_last_X_days} \
-is:suspended -is:buried"
            whi(" >  '" + query)
            rated_cards = self._ankiconnect(action="findCards",
                                            query=query)
            whi(f"Found {len(rated_cards)} cards...\n")
        else:
            yel("Will not look for cards rated in past days.")
            rated_cards = []

        if rated_cards != []:
            temp = [x for x in rated_cards if x not in due_cards]
            diff = len(rated_cards) - len(temp)
            if diff != 0:
                red(f"Removed overlap between rated cards and due cards: \
{diff} cards removed. Keeping {len(temp)} cards.\n")
                rated_cards = temp
        self.due_cards = due_cards
        self.rated_cards = rated_cards

        if len(self.due_cards) < self.due_threshold:
            red("Number of due cards is less than threshold.\nStopping.")
            self.not_enough_cards = True
            return
        else:
            self.not_enough_cards = False

        limit = self.debug_card_limit if self.debug_card_limit else None
        combined_card_list = list(rated_cards + due_cards)[:limit]

        list_cardInfo = []

        n = len(combined_card_list)
        yel(f"\nAsking Anki for information about {n} cards...")
        start = time.time()
        list_cardInfo.extend(
            self._get_cards_info_from_card_id(
                card_id=combined_card_list))
        whi(f"Got all infos in {int(time.time()-start)} seconds.\n\n")

        for i, card in enumerate(list_cardInfo):
            # removing large fields:
            list_cardInfo[i].pop("question")
            list_cardInfo[i].pop("answer")
            list_cardInfo[i].pop("css")
            list_cardInfo[i].pop("fields_no_html")
            list_cardInfo[i]["fields"] = dict(
                (k.lower(), v)
                for k, v in list_cardInfo[i]["fields"].items())
            list_cardInfo[i]["tags"] = " ".join(list_cardInfo[i]["tags"])
            if card["cardId"] in due_cards:
                list_cardInfo[i]["status"] = "due"
            elif card["cardId"] in rated_cards:
                list_cardInfo[i]["status"] = "rated"
            else:
                list_cardInfo[i]["status"] = "ERROR"
                red(f"Error processing card with ID {card['cardId']}")

        if len(list_cardInfo) != len(list(set(combined_card_list))):
            red("Error: duplicate cards in DataFrame!\nExiting.")
            pdb.set_trace()

        self.df = pd.DataFrame().append(list_cardInfo,
                                        ignore_index=True,
                                        sort=True
                                        ).set_index("cardId").sort_index()
        return True

    def _smart_acronym_replacer(self, string, compiled, new_w):
        """
        acronym replacement using regex needs this function to replace
            match groups. For example: replacing 'IL(\\d+)' by "Interleukin 2"
        """
        if len(string.groups()):
            for i in range(len(string.groups())):
                if string.group(i + 1) is not None:
                    new_w = new_w.replace('\\' + str(i + 1),
                                          string.group(i + 1))
        out = string.group(0) + f" ({new_w})"
        return out

    def _format_text(self, text):
        """
        take text and output processed and formatted text
        Acronyms will be replaced if the corresponding arguments is passed
            when instantiating AnnA
        Greek letters can also be replaced on the fly
        """
        orig = text
        s = re.sub


        text = s('\u001F|&nbsp;', " ", text)
        text = text.replace("&amp;", "&").replace("/", " / ")
        text = text.replace("+++", " important ")

        # spaces
        text = s("<blockquote(.*?)</blockquote>",
                 lambda x: x.group(0).replace("<br>", " ; "), text)
        text = s('\\n|<div>|</div>|<br>|<span>|</span>|<li>|</li>|<ul>|</ul>',
                 " ", text)  # newlines
        text = " ".join(text.split())

        # OCR
        if self.keep_ocr:
            # keep image title (usually OCR)
            text = s("title=(\".*?\")", "> Caption: '\\1' <",
                     text,
                     flags=re.MULTILINE | re.DOTALL)
            text = text.replace('Caption: \'""\'', "")

        # cloze
        text = s(r"{{c\d+?::", "", text)
        text = s("{{c|{{|}}|::", " ", text)

        # misc
        text = s(r'[a-zA-Z0-9-]+\....', " ", text)  # media file name
        text = s("<a href.*?</a>", " ", text)  # html links
        text = s(r'http[s]?://\S*', " ", text)  # plaintext links
        text = s("<.*?>", " ", text)  # remaining html tags
        text = s(r"\[\d*\]", "", text)  # wiki citation
        text = s(r"\b\w{1,5}>", " ", text)  # missed html tags
        text = s("&gt;|&lt;|<|>", "", text)

        # adds capital letter after punctuation
        text = s(r"[.?!]\s+?([a-zA-Z])", lambda x: x.group(0).upper(), text)

        # replace greek letter
        if self.replace_greek:
            for a, b in greek_alphabet_mapping.items():
                text = s(a, b, text)

        # replace acronyms
        if self.acronym_list is not None:
            for compiled, new_word in self.acronym_dict.items():
                text = s(compiled,
                         lambda string:
                         self._smart_acronym_replacer(string,
                                                      compiled,
                                                      new_word),
                         text)

        # misc
        text = text.replace(" : ", ": ")
        text = " ".join(text.split())  # multiple spaces

        # if text too short, include image name if present
        if len(text) < 10:
            if "src=" in orig:
                text = text + " " + " ".join(re.findall('src="(.*?)">', orig,
                    flags=re.MULTILINE | re.DOTALL))

        # add punctuation
#        if len(text) > 2:
#            text = text[0].upper() + text[1:]
#            if text[-1] not in ["?", ".", "!"]:
#                text += "."

        # misc
        text = text.replace(" :.", ".")
        text = text.replace(":.", ".")

        # optionnal stemmer
        if self.vectorizer == "TFIDF":
            if self.TFIDF_stem is True:
                text = " ".join([ps.stem(x) for x in text.split()])
            text += " " + " ".join(re.findall(r'src="(.*?\..{2,3})" ', orig,
                flags=re.MULTILINE | re.DOTALL))

        return text

    def _format_card(self):
        """
        filter the fields of each card and keep only the relevant fields
        a "relevant field" is one that is mentioned in the variable field_dic
        which can be found at the top of the file. If not relevant field are
        found then only the first field is kept.
        """
        self._collect_memory()
        def _threaded_field_filter(df, index_list, lock, pbar, stopw_compiled):
            """
            threaded implementation to speed up execution
            """
            for index in index_list:
                card_model = df.loc[index, "modelName"]
                fields_to_keep = []

                # determines which is the corresponding model described
                # in field_dic
                field_dic = self.field_dic
                target_model = []
                for user_model in field_dic:
                    if user_model.lower() in card_model.lower():
                        target_model.append(user_model)
                if len(target_model) == 0:
                    fields_to_keep = "take_first_fields"
                elif len(target_model) == 1:
                    fields_to_keep = field_dic[target_model[0]]
                elif len(target_model) > 1:
                    target_model = sorted(target_model,
                                          key=lambda x: lev.ratio(
                                              x.lower(), user_model.lower()))
                    fields_to_keep = field_dic[target_model[0]]
                    with lock:
                        to_notify.append(f"Several notetypes match \
{card_model}. Chose to notetype {target_model[0]}")

                # concatenates the corresponding fields into one string:
                if fields_to_keep == "take_first_fields":
                    fields_to_keep = ["", ""]
                    field_list = list(df.loc[index, "fields"])
                    for f in field_list:
                        order = df.loc[index, "fields"][f.lower()]["order"]
                        if order == 0:
                            fields_to_keep[0] = f
                        if order == 1:
                            fields_to_keep[1] = f
                    with lock:
                        to_notify.append(f"No matching notetype found for \
{card_model}. Chose first 2 fields: {', '.join(fields_to_keep)}")

                comb_text = ""
                field_counter = {}
                for f in fields_to_keep:
                    if f in field_counter:
                        field_counter[f] += 1
                    else:
                        field_counter[f] = 1
                    try:
                        next_field = re.sub(stopw_compiled,
                                            "",
                                df.loc[index, "fields"][f.lower()]["value"].strip())
                        if next_field != "":
                            comb_text = comb_text + next_field + ": "
                    except KeyError as e:
                        with lock:
                            to_notify.append(f"Error when looking for field {e} in card \
{df.loc[index, 'modelName']} identified as notetype {target_model}")
                if comb_text[-2:] == ": ":
                    comb_text = comb_text[:-2]

                # add tags to comb_text
                tags = self.df.loc[index, "tags"].split(" ")
                spacers_reg = re.compile("::|_|-|/")
                for t in tags:
                    if "AnnA" not in t:
                        comb_text += " " + re.sub(spacers_reg, " ", t)

                with lock:
                    self.df.at[index, "comb_text"] = comb_text
                    pbar.update(1)

        n = len(self.df.index)
        batchsize = n // 4 + 1
        lock = threading.Lock()

        threads = []
        to_notify = []
        stopw_compiled = re.compile("\b" + "\b|\b".join(self.stops) + "\b",
                                    flags=re.MULTILINE | re.IGNORECASE | re.DOTALL)

        with tqdm(total=n,
                  desc="Combining relevant fields",
                  smoothing=0,
                  unit=" card") as pbar:

            for nb in range(0, n, batchsize):
                sub_card_list = self.df.index[nb: nb + batchsize]
                thread = threading.Thread(target=_threaded_field_filter,
                                          args=(self.df,
                                                sub_card_list,
                                                lock,
                                                pbar,
                                                stopw_compiled),
                                          daemon=False)
                thread.start()
                threads.append(thread)

            [t.join() for t in threads]

            cnt = 0
            while sum(self.df.isna()["comb_text"]) != 0:
                cnt += 1
                na_list = [x for x in self.df.index[self.df.isna()["comb_text"]].tolist()]
                pbar.update(-len(na_list))
                red(f"Found {sum(self.df.isna()['comb_text'])} null values in comb_text: retrying")
                thread = threading.Thread(target=_threaded_field_filter,
                                          args=(self.df,
                                                na_list,
                                                lock,
                                                pbar,
                                                stopw_compiled),
                                          daemon=False)
                thread.start()
                thread.join()
                if cnt > 10:
                    red(f"Error: restart anki then rerun AnnA.")
                    raise SystemExit()
            if cnt > 0:
                yel(f"Succesfully corrected null combined texts on #{cnt} trial.")


        for notification in list(set(to_notify)):
            red(notification)

        tqdm.pandas(desc="Formating text", smoothing=0, unit=" card")
        self.df["text"] = self.df["comb_text"].progress_apply(lambda x: self._format_text(x))
        print("\n\nPrinting 5 random samples of your formated text, to help \
adjust formating issues:")
        pd.set_option('display.max_colwidth', 80)
        max_length = 100
        sub_index = random.choices(self.df.index.tolist(), k=5)
        for i in sub_index:
            print(f" *  {i}: {str(self.df.loc[i, 'text'])[0:max_length]}...")
        pd.reset_option('display.max_colwidth')
        print("\n")
        self.df = self.df.sort_index()
        return True

    def _compute_card_vectors(self,
                              df=None,
                              store_vectors=False,
                              import_thread=None):
        """
        Assigne fastText vectors to each card
        df["VEC_FULL"], contains the vectors
        df["VEC"] contains either all the vectors or less if you
            enabled dimensionality reduction
        """
        self._collect_memory()
        if df is None:
            df = self.df

        if import_thread is not None:
            import_thread.join()
            time.sleep(0.5)

        if self.vectorizer == "fastText":

            def preprocessor(string):
                """
                prepare string of text to be vectorized by fastText
                * makes lowercase
                * removes all non letters
                * removes extra spaces
                * outputs each words in a list
                """
                return re.sub(alphanum, " ", string.lower()).split()

            def memoize(f):
                """
                store previous value to speed up vector retrieval
                (sped up by about x40)
                """
                memo = {}

                def helper(x):
                    if x not in memo:
                        memo[x] = f(x)
                    return memo[x]
                return helper

            def vec(string):
                return normalize(np.sum([mvec(x)
                                         for x in preprocessor(string)
                                         if x != ""],
                                        axis=0).reshape(1, -1),
                                 norm='l2')

            alphanum = re.compile(r"[^ _\w]|\d|_")
            mvec = memoize(ft.get_word_vector)
            ft_vec = np.empty(shape=(len(df.index), ft.get_dimension()),
                              dtype=float)

            for i, x in enumerate(
                    tqdm(df.index, desc="Vectorizing using fastText")):
                ft_vec[i] = vec(str(df.loc[x, "text"]))

            if self.fastText_dim is None:
                df["VEC"] = [x for x in ft_vec]
                df["VEC_FULL"] = [x for x in ft_vec]
            else:
                print(f"Reducing dimensions to {self.fastText_dim} using UMAP")
                red("(WARNING: EXPERIMENTAL FEATURE)")
                import umap.umap_
                umap_kwargs = {"n_jobs": -1,
                               "verbose": 1,
                               "n_components": min(self.fastText_dim,
                                                   len(df.index) - 1),
                               "metric": "cosine",
                               "init": 'spectral',
                               "random_state": 42,
                               "transform_seed": 42,
                               "n_neighbors": 5,
                               "min_dist": 0.01}
                try:
                    ft_vec_red = umap.UMAP(**umap_kwargs).fit_transform(ft_vec)
                    df["VEC"] = [x for x in ft_vec_red]
                except Exception as e:
                    red(f"Error when computing UMAP reduction, using all vectors: {e}")
                    df["VEC"] = [x for x in ft_vec]
                finally:
                    df["VEC_FULL"] = [x for x in ft_vec]

            if store_vectors:
                def storing_vectors(df):
                    yel("Storing computed vectors")
                    fastText_store = Path("./stored_vectors.pickle")
                    if not fastText_store.exists():
                        df_store = pd.DataFrame(columns=["cardId", "mod", "text", "VEC_FULL"]).set_index("cardId")
                    else:
                        df_store = pd.read_pickle(fastText_store)
                    for i in df.index:
                        df_store.loc[i, :] = df.loc[i, ["mod", "text", "VEC_FULL"]]
                    df_store.to_pickle("stored_vectors.pickle_temp")
                    if fastText_store.exists():
                        fastText_store.unlink()
                    Path("stored_vectors.pickle_temp").rename("stored_vectors.pickle")
                    yel("Finished storing computed vectors")
                stored_df = df.copy()
                thread = threading.Thread(target=storing_vectors,
                                          args=(stored_df,),
                                          daemon=True)
                thread.start()




        elif self.vectorizer == "TFIDF":
            if self.TFIDF_tokenize:
                def tknzer(x):
                    return tokenizer.tokenize(x)
            else:
                def tknzer(x):
                    return x

            n_features = len(" ".join([x for x in df["text"]]).split(" ")) // 100
            n_features = max(n_features, 500)
            n_features = min(n_features, 10_000)
            red(f"Max number of features for TF_IDF: {n_features}")

            vectorizer = TfidfVectorizer(strip_accents="ascii",
                                         lowercase=True,
                                         tokenizer=tknzer,
                                         stop_words=None,  # already removed
                                         ngram_range=(1, 5),
                                         max_features=n_features,
                                         norm="l2")
            t_vec = vectorizer.fit_transform(tqdm(df["text"],
                                             desc="Vectorizing text using \
TFIDF"))
            if self.TFIDF_dim is None:
                df["VEC_FULL"] = [x for x in t_vec]
                df["VEC"] = [x for x in t_vec]
            else:
                while True:
                    print(f"Reducing dimensions to {self.TFIDF_dim} using SVD")
                    svd = TruncatedSVD(n_components=min(self.TFIDF_dim,
                                                        t_vec.shape[1]))
                    t_red = svd.fit_transform(t_vec)
                    evr = round(sum(svd.explained_variance_ratio_)*100,1)
                    if evr >= 80:
                        break
                    else:
                        if self.TFIDF_dim >= 2000:
                            break
                        if evr <= 40:
                            self.TFIDF_dim *= 4
                        elif evr <= 60:
                            self.TFIDF_dim *= 2
                        else:
                            self.TFIDF_dim += int(self.TFIDF_dim*0.5)
                        self.TFIDF_dim = min(self.TFIDF_dim, 2000)
                        yel(f"Explained variance ratio is only {evr}% (\
retrying until above 80% or 2000 dimensions)")
                        continue
                whi(f"Explained variance ratio after SVD on Tf_idf: \
{round(sum(svd.explained_variance_ratio_)*100,1)}%")


                df["VEC_FULL"] = [x for x in t_vec]
                df["VEC"] = [x for x in t_red]

        self.df = df
        return True

    def _compute_distance_matrix(self, input_col="VEC"):
        """
        compute distance matrix between all the cards
        * the distance matrix can be parallelised by scikit learn so I didn't
            bother saving and reusing the matrix
        """
        self._collect_memory()
        df = self.df

        print("\nComputing distance matrix on all available cores...")
        df_temp = pd.DataFrame(
            columns=["vec_"+str(x+1)
                     for x in range(len(df.loc[df.index[0], input_col]))],
            data=[x[0:] for x in df[input_col].values])

        self.df_dist = pd.DataFrame(columns=df.index,
                                    index=df.index,
                                    data=pairwise_distances(
                                        df_temp,
                                        n_jobs=-1,
                                        metric="cosine"))

        # showing to user which cards are similar and different,
        # for troubleshooting
        red("Printing the most semantically different cards:")
        pd.set_option('display.max_colwidth', 80)
        max_length = 100
        maxs = np.where(self.df_dist.values == np.max(self.df_dist.values))
        maxs = [x for x in zip(maxs[0], maxs[1])]
        yel(f"* {str(df.loc[df.index[maxs[0][0]]].text)[0:max_length]}...")
        yel(f"* {str(df.loc[df.index[maxs[0][1]]].text)[0:max_length]}...")
        print("")

        printed = False
        lowest_values = [np.max(np.diag(self.df_dist))]
        start_time = time.time()
        for i in range(9999):
            if printed is True:
                break
            if time.time() - start_time >= 60:
                red("Taking too long to show similar cards, skipping")
                break
            lowest_values.append(self.df_dist.values[self.df_dist.values > max(
                        lowest_values)].min())
            low = lowest_values[-1]
            mins = np.where(self.df_dist.values == low)
            mins = [x for x in zip(mins[0], mins[1]) if x[0] != x[1]]
            random.shuffle(mins)
            for x in range(len(mins)):
                pair = mins[x]
                text_1 = str(df.loc[df.index[pair[0]]].text)
                text_2 = str(df.loc[df.index[pair[1]]].text)
                if text_1 != text_2:
                    red("Printing among most semantically similar cards:")
                    yel(f"* {text_1[0:max_length]}...")
                    yel(f"* {text_2[0:max_length]}...")
                    printed = True
                    break
        if printed is False:
            red("Couldn't find lowest values to print!")
        print("")
        pd.reset_option('display.max_colwidth')
        print("\n")
        return True

    def _compute_opti_rev_order(self):
        """
        a loop that assigns a score to each card, the lowest score at each
            turn is added to the queue, each new turn compares the cards to
            the present queue

        * this score reflects the order in which they should be reviewed
        * the intuition is that anki doesn't know before hand if some cards
            are semantically close and can have you review them the same day

        # TODO: this a changed a lot:
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
        * Index score: combining two scores was hard (they had really different
            distribution, so I added this flag to tell wether to use the scores
            or the argsort of the scores as score

        CAREFUL: this docstring might not be up to date as I am constantly
            trying to improve the code
        """
        self._collect_memory()
        # getting args
        reference_order = self.reference_order
        df = self.df.copy()
        df_dist = self.df_dist
        target_deck_size = self.target_deck_size
        rated = self.rated_cards
        due = self.due_cards
        queue = []
        w1 = self.score_adjustment_factor[0]
        w2 = self.score_adjustment_factor[1]
        use_index_of_score = False
        display_stats = True

        if w1 == 0:
            yel("Ignoring reference order because the first score adjustment \
factor is 0.")
            df["ref"] = 0

        elif reference_order == "lowest_interval":
            # alter the value from rated cards as they will not be useful
            df.loc[rated, "due"] = np.median(df.loc[due, "due"].values)
            df.loc[rated, "interval"] = np.median(df.loc[due, "interval"].values)

            ivl = df['interval'].to_numpy().reshape(-1, 1)
            df["interval_cs"] = StandardScaler().fit_transform(ivl)
            df["ref"] = df["interval_cs"]

        elif reference_order == "order_added":
            order_added = df.index.to_numpy().reshape(-1, 1)
            df["ref"] = StandardScaler().fit_transform(order_added)
            df.loc[rated, "due"] = np.median(df.loc[due, "due"].values)
            df.loc[rated, "interval"] = np.median(df.loc[due, "interval"].values)

        elif reference_order == "relative_overdueness":
            print("Computing relative overdueness...")
            anki_col_time = int(self._ankiconnect(
                action="getCollectionCreationTime"))
            # getting due date
            for i in df.index:
                df.at[i, "ref_due"] = df.loc[i, "odue"]
                if df.loc[i, "ref_due"] == 0:
                    df.at[i, "ref_due"] = df.at[i, "due"]
                if df.loc[i, "ref_due"] >= 100_000:  # timestamp instead of days
                    df.at[i, "ref_due"] = (df.at[i, "ref_due"]-anki_col_time) / 86400
            df.loc[rated, "ref_due"] = np.median(df.loc[due, "ref_due"].values)
            df.loc[rated, "interval"] = np.median(df.loc[due, "interval"].values)

            # computing overdue
            time_offset = int((time.time() - anki_col_time) / 86400)
            overdue = (df["ref_due"] - time_offset).to_numpy().reshape(-1, 1)

            # computing relative overdueness
            ro = -1 * (df["interval"].values + 0.5) / (overdue.T + 0.5)

            # center and scale
            ro_cs = StandardScaler().fit_transform(ro.T)
            df["ref"] = ro_cs

        assert len([x for x in rated if df.loc[x, "status"] != "rated"]) == 0
        red(f"\nCards identified as rated in the past {self.rated_last_X_days} days: \
{len(rated)}")

        if isinstance(target_deck_size, float):
            if target_deck_size < 1.0:
                target_deck_size = str(target_deck_size*100) + "%"
        if isinstance(target_deck_size, str):
            if target_deck_size in ["all", "100%"]:
                red("Taking the whole deck.")
                target_deck_size = len(df.index) - len(rated)
            elif target_deck_size.endswith("%"):
                red(f"Taking {target_deck_size} of the deck.")
                target_deck_size = 0.01*int(target_deck_size[:-1])*(
                            len(df.index)-len(rated)
                            )
        target_deck_size = int(target_deck_size)

        if len(rated+queue) < 1:  # can't start with an empty queue
            # so picking 1 urgent cards
            pool = df.loc[df["status"] == "due", "ref"].nsmallest(
                    n=min(50, len(self.due_cards))
                    ).index
            queue.extend(random.choices(pool, k=1))

        indTODO = df.drop(index=rated+queue).index.tolist()
        indQUEUE = (rated+queue)

        # remove siblings of indTODO:
        noteCard = {}
        for card, note in {df.loc[x].name: df.loc[x, "note"]
                           for x in indTODO}.items():
            if note not in noteCard:
                noteCard[note] = card
            else:
                if float(df.loc[noteCard[note], "ref"]
                         ) > float(df.loc[card, "ref"]):
                    noteCard[note] = card
        previous_len = len(indTODO)
        [indTODO.remove(x) for x in indTODO if x not in noteCard.values()]
        cur_len = len(indTODO)
        if previous_len - cur_len == 0:
            yel("No siblings found.")
        else:
            red(f"Removed {previous_len-cur_len} siblings cards out of \
{previous_len}.")

        if target_deck_size > cur_len:
            red(f"You wanted to create a deck with \
{target_deck_size} in it but only {cur_len} cards remain, taking the \
lowest value.")
        queue_size_goal = min(target_deck_size, cur_len)

        if display_stats:
            pd.set_option('display.float_format', lambda x: '%.5f' % x)
            try:
                whi("\nScore stats (adjusted):")
                if w1 != 0:
                    whi(f"Reference: {(w1*df['ref']).describe()}\n")
                val = pd.DataFrame(data=w2*df_dist.values.flatten(),
                                   columns=['distance matrix']).describe(include='all')
                whi(f"Distance: {val}\n\n")
            except Exception as e:
                red(f"Exception: {e}")
            pd.reset_option('display.float_format')

        def combinator(array):
            return 0.9*np.min(array, axis=0) + 0.1*np.mean(array, axis=0)

        with tqdm(desc="Computing optimal review order",
                  unit=" card",
                  initial=len(rated),
                  smoothing=0,
                  total=queue_size_goal+len(rated)) as pbar:
            while len(queue) < queue_size_goal:
                if use_index_of_score:
                    # NOT YET USABLE:
                    queue.append(indTODO[
                            (w1*df.loc[indTODO, "ref"].values.argsort() -\
                             w2*(
                             np.min(
                                 df_dist.loc[indQUEUE[-self.queue_stride:], indTODO].values,
                                 axis=0).argsort() +\
                             np.mean(
                                 df_dist.loc[indQUEUE[-self.queue_stride:], indTODO].values,
                                 axis=0).argsort()
                             )).argmin()])
                    indQUEUE.append(indTODO.pop(indTODO.index(queue[-1])))
                else:
                    queue.append(indTODO[
                            (w1*df.loc[indTODO, "ref"].values -\
                             w2*combinator(df_dist.loc[indQUEUE[-self.queue_stride:], indTODO ].values)
                             ).argmin()])
                    indQUEUE.append(indTODO.pop(indTODO.index(queue[-1])))

                pbar.update(1)
        assert len(queue) != 0

        try:
            red("Sum distance among the optimized queue:")
            spread_queue = np.sum(self.df_dist_unscaled.loc[queue, queue].values)
            yel(spread_queue)

            red("Sum distance if you had not used AnnA:")
            woAnnA = [x
                      for x in df.sort_values(
                          "ref", ascending=True).index.tolist()
                      if x in self.due_cards][0:len(queue)]
            spread_else = np.sum(self.df_dist_unscaled.loc[woAnnA, woAnnA].values)
            yel(spread_else)

            ratio = round(spread_queue / spread_else, 3)
            red("Improvement ratio:")
            red(pyfiglet.figlet_format(str(ratio)))

            red(f"Cards in common: {len(set(queue)&set(woAnnA))} in a queue of {len(queue)} cards.")

        except Exception as e:
            red(f"\nException: {e}")

        yel("Done computing order.\n")
        self.opti_rev_order = queue
        self.df = df
        return True

    def display_opti_rev_order(self, display_limit=50):
        """
        display the cards in the best optimal order. Useful to see if something
        went wrong before creating the filtered deck
        """
        self._collect_memory()
        order = self.opti_rev_order[:display_limit]
        print(self.df.loc[order, "text"])
        return True

    def task_filtered_deck(self,
                           deck_template=None,
                           task=None):
        """
        create a filtered deck containing the cards to review in optimal order

        * When first creating the filtered deck, I chose 'sortOrder = 0'
            ("oldest seen first") this way I will notice if the deck
            somehow got rebuild and lost the right order
        * deck_template can be used to group your filtered decks together.
            Leaving it to None will make the filtered deck appear alongside
            the original deck
        * To speed up the process, I decided to create a threaded function call
        * I do a few sanity check to see if the filtered deck
            does indeed contain the right number of cards and the right cards
        * -100 000 seems to be the starting value for due order in filtered
            decks
        * if task is set to bury_excess_learning_cards ot bury_excess_review_cards, then no filtered
            deck will be created and AnnA will just bury the cards that are
            too similar
        """
        self._collect_memory()
        if task in ["bury_excess_learning_cards", "bury_excess_review_cards"]:
            to_keep = self.opti_rev_order
            to_bury = [x for x in self.due_cards if x not in to_keep]
            assert len(to_bury) < len(self.due_cards)
            red(f"Burying {len(to_bury)} cards out of {len(self.due_cards)}.")
            red("This will not affect the due order.")
            self._ankiconnect(action="bury",
                              cards=to_bury)
            print("Done.")
            return True
        else:
            if deck_template is not None:
                filtered_deck_name = str(deck_template + f" - {self.deckname}")
                filtered_deck_name = filtered_deck_name.replace("::", "_")
            else:
                filtered_deck_name = f"{self.deckname} - AnnA Optideck"
            self.filtered_deck_name = filtered_deck_name

            while filtered_deck_name in self._ankiconnect(action="deckNames"):
                red(f"\nFound existing filtered deck: {filtered_deck_name} \
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

        whi(f"Creating deck containing the cards to review: \
{filtered_deck_name}")
        query = "is:due -rated:1 cid:" + ','.join([str(x) for x in self.opti_rev_order])
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
            red("Inconsistency! The deck does not contain the same cards \
as opti_rev_order!")
            pprint(diff)
            red(f"\nNumber of inconsistent cards: {len(diff)}")
        else:
            print(" Done.")

        _threaded_value_setter(card_list=self.opti_rev_order,
                               tqdm_desc="Altering due order",
                               keys=["due"],
                               newValues=None)
        print("All done!\n\n")
        return True

    def compute_clusters(self,
                         algo="minibatch-kmeans",
                         input_col="VEC",
                         cluster_col="clusters",
                         n_topics=5,
                         minibatchk_kwargs=None,
                         kmeans_kwargs=None,
                         agglo_kwargs=None,
                         dbscan_kwargs=None,
                         add_as_tags=True,
                         stem_topics=True):
        """
        finds cluster of cards and their respective topics
        * this is not mandatory to create the filtered deck but it's rather
            fast
        * Several algorithm are supported for clustering: kmeans, DBSCAN,
            agglomerative clustering
        * n_topics is the number of topics (=word) to get for each cluster
        * To find the topic of each cluster, ctf-idf is used
        """
        self._collect_memory()
        df = self.df
        if self.clustering_nb_clust is None or \
                self.clustering_nb_clust == "auto":
            self.clustering_nb_clust = len(df.index)//50
            red(f"No number of clusters supplied, will try with \
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

        if algo.lower() in "minibatch-kmeans":
            clust = MiniBatchKMeans(**minibatchk_kwargs_deploy)
            algo = "minibatch-K-Means"
        elif algo.lower() in "kmeans":
            clust = KMeans(**kmeans_kwargs_deploy)
            algo = "K-Means"
        elif algo.lower() in "DBSCAN":
            clust = DBSCAN(**dbscan_kwargs_deploy)
            algo = "DBSCAN"
        elif algo.lower() in "agglomerative":
            clust = AgglomerativeClustering(**agglo_kwargs_deploy)
            algo = "Agglomerative Clustering"
        print(f"Clustering using {algo}...")

        df_temp = pd.DataFrame(
            columns=["V"+str(x)
                     for x in range(len(df.loc[df.index[0], input_col]))],
            data=[x[0:] for x in df[input_col]])
        df[cluster_col] = clust.fit_predict(df_temp)

        cluster_list = sorted(list(set(list(df[cluster_col]))))
        cluster_nb = len(cluster_list)
        print(f"Getting cluster topics for {cluster_nb} clusters...")

        # reordering cluster number, as they are sometimes offset
        for i, clust_nb in enumerate(cluster_list):
            df.loc[df[cluster_col] == clust_nb, cluster_col] = i
        cluster_list = sorted(list(set(list(df[cluster_col]))))
        cluster_nb = len(cluster_list)

        if stem_topics:
            df["stemmed"] = df.apply(
                    lambda row: " ".join([ps.stem(x) for x in row["text"].split()]),
                    axis=1)
            topics_col = "stemmed"
        else:
            topics_col = "text"

        df_by_cluster = df.groupby([cluster_col],
                                   as_index=False).agg({topics_col: ' '.join})
        count = CountVectorizer().fit_transform(df_by_cluster[topics_col])
        ctfidf = CTFIDFVectorizer().fit_transform(count,
                                                  n_samples=len(
                                                      df_by_cluster.index))
        count_vectorizer = CountVectorizer().fit(df_by_cluster[topics_col])
        count = count_vectorizer.transform(df_by_cluster[topics_col])
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
                clst_tpc = " ".join([x
                                     for x in w_by_class[str(
                                         df.loc[i, cluster_col])]])
                df.loc[i, "cluster_topic"] = clst_tpc

            self.w_by_class = w_by_class
            self.df = df.sort_index()

            if add_as_tags:
                # creates two threads, the first one removes old tags and the
                # other one adds the new one
                threads = []
                full_note_list = list(set(df["note"].tolist()))
                all_tags = df["tags"].tolist()
                present_tags = []
                self._ankiconnect(action="addTags", batchmode="open")
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
                                              tags=str(tag),
                                              batchmode="ongoing")
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
                        cur_time = "_".join(
                                time.asctime().split()[0:4]).replace(
                                ":", "h")[0:-3]
                        nb = df[df['clusters'] == i]['cluster_topic'].iloc[0]
                        newTag = "AnnA::cluster_topic::"
                        newTag += f"{cur_time}::cluster_#{str(i)}"
                        newTag += f"::{nb}"
                        note_list = list(set(
                            df[df["clusters"] == i]["note"].tolist()))
                        self._ankiconnect(action="addTags",
                                          notes=note_list,
                                          tags=newTag,
                                          batchmode="ongoing")
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
        except IndexError as e:
            red(f"Index Error when finding cluster topic! {e}")
        finally:
            self._ankiconnect(action="addTags", batchmode="close")
            self._ankiconnect(action="clearUnusedTags")
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
                          save_as_html=True,
                          ):
        """
        open a browser tab with a 2D plot showing your cards and their
        relations
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
            import umap.umap_
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
        if disable_legend:
            fig = fig.update_layout(showlegend=False)
        if save_as_html:
            fig.write_html(f"plotly_graph_{self.deckname}.html")
            print(f"Saved plot as 'plotly_graph_{self.deckname}.html'")
        print("Opening graph in browser...")
        fig.show()
        return True

    @classmethod
    def search_for_notes(self,
                         user_input,
                         nlimit=10,
                         user_col="VEC_FULL",
                         do_format_input=False,
                         anki_or_print="anki",
                         reverse=False,
                         fastText_lang="fr",
                         offline=False):
        """
        given a text input, find notes with highest cosine similarity
        * note that you cannot use the pca version of the fastText vectors
            otherwise you'd have to re run the whole PCA, so it's quicker
            to just use the full vectors
        * note that if the results are displayed in the browser, the order
            cannot be maintained. So you will get the nlimit best match
            randomly displayed, then the nlimit+1 to 2*nlimit best match
            randomly displayed etc
        * note that this obviously only works using fastText vectors and not TFIDF
            (they would have to be recomputed each time)
        """
        pd.set_option('display.max_colwidth', None)
        if offline:
            fastText_cachefile = Path("./cached_vectors.pickle")
            if fastText_cachefile.exists():
                df = pd.read_pickle(fastText_cachefile)
            else:
                red("cached vectors not found.")
                return False

            red("Loading fastText model")
            import fastText
            import fastText.util
            try:
                fastText.util.download_model(fastText_lang, if_exists='ignore')
                ft = fastText.load_model(f"cc.{fastText_lang[0:2]}.300.bin")
            except Exception as e:
                red(f"Couldn't load fastText model: {e}")
                raise SystemExit()

            from sklearn.metrics import pairwise_distances

            anki_or_print = "print"
            user_col = "VEC"

        elif self.vectorizer == "TFIDF":
            red("Cannot search for note using TFIDF vectors, only fastText can \
be used.")
            return False

        else:
            df = self.df.copy()

        if do_format_input:
            user_input = self._format_text(user_input)

        embed = np.max([ft.get_word_vector(x) for x in user_input.split(" ")], axis=0)

        print("")
        tqdm.pandas(desc="Searching")
        try:
            df["distance"] = 0.0
            df["distance"] = df[user_col].progress_apply(
                    lambda x: pairwise_distances(embed.reshape(1, -1),
                                                 x.reshape(1, -1),
                                                 metric="cosine"))
            df["distance"] = df["distance"].astype("float")
        except ValueError as e:
            red(f"Error {e}: did you select column 'VEC' instead of \
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
        df.to_pickle("./.DataFrame_backups/" + name)
        print(f"Dataframe exported to {name}.")
        return True

    def show_acronyms(self, exclude_OCR_text=True):
        """
        shows acronym present in your collection that were not found in
        the file supplied by the argument `acronym_list`
        * acronyms found in OCR caption are removed by default
        """
        full_text = " ".join(self.df["text"].tolist())
        if exclude_OCR_text:
            full_text = re.sub(" Caption: '.*?' ", " ", full_text,
                               flags=re.MULTILINE | re.DOTALL)
        matched = list(set(re.findall("[A-Z]{3,}", full_text)))

        # if exists as lowercase : it's probably not an acronym and just
        # used for emphasis
        for m in matched:
            if m.lower() in full_text:
                matched.remove(m)
        if len(matched) == 0:
            return True
        sorted_by_count = sorted([x for x in matched],
                                 key=lambda x: full_text.count(x),
                                 reverse=True)
        relevant = list(set(random.choices(sorted_by_count[0:50],
                                           k=min(len(sorted_by_count), 10))))
        if not len(matched):
            print("No acronym found in those cards.")
            return True

        if self.acronym_list is None:
            red("\nYou did not supply an acronym list, printing all acronym \
found...")
            pprint(relevant)
        else:
            acro_list = list(self.acronym_dict)

            for compiled in acro_list:
                for acr in relevant:
                    if re.match(compiled, acr) is not None:
                        relevant.remove(acr)
            print("List of some acronyms still found:")
            if exclude_OCR_text:
                print("(Excluding OCR text)")
            pprint(sorted(relevant))
        print("")
        return True


class CTFIDFVectorizer(TfidfTransformer):
    """
    this class is just used to allow class Tf-idf, I took it from:
    https://towardsdatascience.com/creating-a-class-based-tf-idf-with-scikit-learn-caea7b15b858
    """
    def __init__(self, *args, **kwargs):
        super(CTFIDFVectorizer, self).__init__(*args, **kwargs)

    def fit(self, X: sparse.csr_matrix, n_samples: int):
        """Learn the idf vector (global term weights) """
        _, n_features = X.shape
        df = np.squeeze(np.asarray(X.sum(axis=0)))
        idf = np.log(n_samples / df)
        self._idf_diag = sparse.diags(idf, offsets=0,
                                  shape=(n_features, n_features),
                                  format='csr',
                                  dtype=np.float64)
        return self

    def transform(self, X: sparse.csr_matrix) -> sparse.csr_matrix:
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

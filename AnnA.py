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

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import normalize, StandardScaler

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
        global fastText, ft
        if "ft" in globals():
            if ft.model_name in [fastText_model_name,
                                 f"cc.{fastText_lang}.300.bin"]:
                reload_ft = False
            else:
                reload_ft = True

        if ("ft" not in globals()) or (reload_ft is True):
            import fasttext as fastText
            import fasttext.util
            try:
                fasttext.util.download_model(fastText_lang, if_exists='ignore')
                if fastText_model_name is None:
                    ft = fastText.load_model(f"cc.{fastText_lang}.300.bin")
                    ft.model_name = f"cc.{fastText_lang}.300.bin"
                else:
                    ft = fastText.load_model(fastText_model_name)
                    ft.model_name = fastText_model_name
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
                 score_adjustment_factor=(1, 5),
                 log_level=2,
                 replace_greek=True,
                 keep_ocr=True,
                 field_mappings="field_mappings.py",
                 acronym_file="acronym_file.py",
                 acronym_list=None,

                 # steps:
                 compute_opti_rev_order=True,
                 task="filter_review_cards",
                 # can be "filter_review_cards",
                 # "bury_excess_review_cards",
                 # "bury_excess_learning_cards",
                 # "index"
                 deck_template=None,

                 # vectorization:
                 stopwords_lang=["english", "french"],
                 vectorizer="TFIDF",  # can be "TFIDF" or "fastText"
                 fastText_dim=100,
                 fastText_dim_algo="PCA", # can be "PCA" or "UMAP" or None
                 fastText_model_name=None,  # if you want to force a specific model
                 fastText_lang="en",
                 fastText_correction_vector=None,  # for example "medical"
                 TFIDF_dim=100,
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
        self.highjack_due_query = highjack_due_query
        self.highjack_rated_query = highjack_rated_query
        self.score_adjustment_factor = score_adjustment_factor
        self.reference_order = reference_order
        self.field_mappings = field_mappings
        self.acronym_file = acronym_file
        self.acronym_list = acronym_list
        self.vectorizer = vectorizer
        self.fastText_lang = fastText_lang
        self.fastText_dim = fastText_dim
        self.fastText_dim_algo = fastText_dim_algo.upper()
        self.fastText_model_name = fastText_model_name
        self.fastText_correction_vector = fastText_correction_vector
        self.TFIDF_dim = TFIDF_dim
        self.stopwords_lang = stopwords_lang
        self.TFIDF_stem = TFIDF_stem
        self.TFIDF_tokenize = TFIDF_tokenize
        self.task = task
        self.deck_template = deck_template
        self.save_instance_as_pickle = save_instance_as_pickle

        # args sanity checks
        if isinstance(self.target_deck_size, int):
            self.target_deck_size = str(self.target_deck_size)
        assert TFIDF_stem + TFIDF_tokenize != 2
        assert reference_order in ["lowest_interval", "relative_overdueness",
                                   "order_added"]
        assert task in ["filter_review_cards", "index",
                        "bury_excess_learning_cards",
                        "bury_excess_review_cards"]
        assert vectorizer in ["TFIDF", "fastText"]
        assert self.fastText_dim_algo in ["PCA", "UMAP", None]
        if task != "filter_review_cards":
            if self.deck_template is not None:
                red("Ignoring argument 'deck_template' because 'task' is not \
set to 'filter_review_cards'.")

        if self.acronym_file is not None and self.acronym_list is not None:
            file = Path(acronym_file)
            if not file.exists():
                raise Exception(f"Acronym file was not found: {acronym_file}")
            else:
                # importing acronym file
                if ".py" in acronym_file:
                    acr_mod = importlib.import_module(acronym_file.replace(
                        ".py", ""))
                else:
                    acr_mod = importlib.import_module(acronym_file)

                # getting acronym dictionnary list
                acr_dict_list = [x for x in dir(acr_mod)
                                 if not x.startswith("_")]

                # if empty file:
                if len(acr_dict_list) == 0:
                    red(f"No dictionnary found in {acronym_file}")
                    raise SystemExit()

                if isinstance(self.acronym_list, str):
                    self.acronym_list = [self.acronym_list]

                missing = [x for x in self.acronym_list
                           if x not in acr_dict_list]
                if missing:
                    red(f"Mising the following acronym dictionnary in \
{acronym_file}: {','.join(missing)}")
                    raise SystemExit()

                acr_dict_list = [x for x in acr_dict_list
                                 if x in self.acronym_list]

                if len(acr_dict_list) == 0:
                    red(f"No dictionnary from {self.acr_dict_list} \
found in {acronym_file}")
                    raise SystemExit()

                compiled_dic = {}
                notifs = []
                for item in acr_dict_list:
                    acronym_dict = eval(f"acr_mod.{item}")
                    for ac in acronym_dict:
                        if ac.lower() == ac:
                            compiled = re.compile(r"\b" + ac + r"\b",
                                                  flags=re.IGNORECASE |
                                                  re.MULTILINE |
                                                  re.DOTALL)
                        else:
                            compiled = re.compile(r"\b" + ac + r"\b",
                                                  flags=re.MULTILINE | re.DOTALL)
                        if compiled in compiled_dic:
                            notifs.append(f"Pattern '{compiled}' found \
multiple times in acronym dictionnary, keeping only the last one.")
                        compiled_dic[compiled] = acronym_dict[ac]
                notifs = list(set(notifs))
                if notifs:
                    for n in notifs:
                        red(n)
                self.acronym_dict = compiled_dic
        else:
            self.acronym_dict = {}

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
            for lang in self.stopwords_lang:
                stops += stopwords.words(lang)
            if self.TFIDF_tokenize:
                temp = []
                [temp.extend(tokenizer.tokenize(x)) for x in stops]
                stops.extend(temp)
            elif self.TFIDF_stem:
                global ps
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
        self.deck_config = self._ankiconnect(action="getDeckConfig",
                                             deck=self.deckname)
        if task == "index":
            yel(f"Task : cache vectors of deck: {self.deckname}\n")
            self.vectorizer = "fastText"
            self.fastText_dim = None
            self.rated_last_X_days = 0
            self._create_and_fill_df()
            if self.not_enough_cards is True:
                return
            self._format_card()
            self.show_acronyms()
            self._compute_card_vectors(import_thread=import_thread)

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
            self._format_card()
            self.show_acronyms()
            self._compute_card_vectors(import_thread=import_thread)
            self._compute_distance_matrix()
            self._compute_opti_rev_order()
            self.task_filtered_deck(task=task)
        else:
            yel("Task : created filtered deck containing review cards")
            self._create_and_fill_df()
            if self.not_enough_cards is True:
                return
            self._format_card()
            self.show_acronyms()
            self._compute_card_vectors(import_thread=import_thread)
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

        yel(f"Done with '{self.task}' on deck {self.deckname}")

    def _collect_memory(self):
        gc.collect()

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
            red(" >  '" + query + "'")
            due_cards = self._ankiconnect(action="findCards", query=query)
            whi(f"Found {len(due_cards)} cards...\n")

        elif self.task in ["filter_review_cards", "bury_excess_review_cards"]:
            yel("Getting due card list...")
            query = f"\"deck:{self.deckname}\" is:due is:review -is:learn \
-is:suspended -is:buried -is:new -rated:1"
            whi(" >  '" + query + "'")
            due_cards = self._ankiconnect(action="findCards", query=query)
            whi(f"Found {len(due_cards)} reviews...\n")

        elif self.task == "bury_excess_learning_cards":
            yel("Getting is:learn card list...")
            query = f"\"deck:{self.deckname}\" is:due is:learn -is:suspended \
-rated:1 -rated:2:1 -rated:2:2"
            whi(" >  '" + query + "'")
            due_cards = self._ankiconnect(action="findCards", query=query)
            whi(f"Found {len(due_cards)} learning cards...\n")

        elif self.task == "index":
            yel("Getting all cards from deck...")
            query = f"\"deck:{self.deckname}\" -is:suspended"
            whi(" >  '" + query + "'")
            due_cards = self._ankiconnect(action="findCards", query=query)
            whi(f"Found {len(due_cards)} cards...\n")

        rated_cards = []
        if self.highjack_rated_query is not None:
            red("Highjacking rated card list:")
            query = self.highjack_rated_query
            red(" >  '" + query + "'")
            rated_cards = self._ankiconnect(action="findCards", query=query)
            red(f"Found {len(rated_cards)} cards...\n")
        elif self.rated_last_X_days != 0:
            yel(f"Getting cards that where rated in the last \
{self.rated_last_X_days} days...")
            query = f"\"deck:{self.deckname}\" rated:{self.rated_last_X_days} \
-is:suspended -is:buried"
            whi(" >  '" + query + "'")
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
            red(f"Number of due cards is {len(self.due_cards)} which is \
less than threshold ({self.due_threshold}).\nStopping.")
            self.not_enough_cards = True
            return
        else:
            self.not_enough_cards = False

        limit = self.debug_card_limit if self.debug_card_limit else None
        combined_card_list = list(rated_cards + due_cards)[:limit]

        list_cardInfo = []

        n = len(combined_card_list)
        yel(f"Asking Anki for information about {n} cards...")
        start = time.time()
        list_cardInfo.extend(
            self._get_cards_info_from_card_id(
                card_id=combined_card_list))
        whi(f"Got all infos in {int(time.time()-start)} seconds.\n")

        for i, card in enumerate(list_cardInfo):
            # removing large fields:
            try:
                list_cardInfo[i].pop("question")
                list_cardInfo[i].pop("answer")
                list_cardInfo[i].pop("css")
                list_cardInfo[i].pop("fields_no_html")
            except KeyError as e:
                pass
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
                                        sort=True)
        self.df["cardId"] = self.df["cardId"].astype(np.int)
        self.df = self.df.set_index("cardId").sort_index()
        self.df["interval"] = self.df["interval"].astype(np.float32)
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
        text = text.replace("&amp;", "&"
                            ).replace("/", " / "
                                      ).replace("+++", " important "
                                                ).replace("&nbsp", " "
                                                          ).replace("\u001F",
                                                                    " ")
        # remove weird clozes
        text = re.sub(r"}}{{c\d+::", "", text)

        # remove sound recordings
        text = re.sub(r"\[sound:.*?\..*?\]", " ", text)

        # duplicate bold and underlined content, as well as clozes
        text = re.sub(r"\b<u>(.*?)</u>\b", r" \1 \1 ", text,
                      flags=re.M | re.DOTALL)
        text = re.sub(r"\b<b>(.*?)</b>\b", r" \1 \1 ", text,
                      flags=re.M | re.DOTALL)
        text = re.sub(r"{{c\d+::.*?}}", lambda x: 2 * (f"{x.group(0)} "),
                      text, flags=re.M | re.DOTALL)

        # if blockquote or li or ul, mention that it's a list item
        # usually indicating a harder card
        if re.match("</blockquote>|</li>|</ul|", text, flags=re.M):
            text += " list"

        # remove html spaces
        text = re.sub('\\n|</?div>|<br>|</?span>|</?li>|</?ul>', " ", text)
        text = re.sub('</?blockquote(.*?)>', " ", text)

        # OCR
        if self.keep_ocr:
            # keep image title (usually OCR)
            text = re.sub("title=(\".*?\")",
                          "> Caption:___\\1___ <",
                          text, flags=re.M | re.DOTALL)
            text = text.replace('Caption:___""___', "")

        # cloze
        text = re.sub(r"{{c\d+?::|}}", "", text)  # remove cloze brackets
        text = re.sub("::", " ", text)  # cloze hints
        text = re.sub("{{c", "", text)  # missed cloze?

        # misc
        text = re.sub(r'[a-zA-Z0-9-]+\....', " ", text)  # media file name
        text = re.sub("<a href.*?</a>", " ", text)  # html links
        text = re.sub(r'https?://\S*?', " ", text)  # plaintext links
        text = re.sub("</?su[bp]>", "", text) # exponant or indices
        text = re.sub(r"\[\d*\]", "", text)  # wiki style citation

        text = re.sub("<.*?>", "", text)  # remaining html tags
        #text = re.sub(r"\b\w{1,5}>", " ", text)  # missed html tags?
        text = re.sub("&gt;|&lt;|<|>", "", text)

        # replace greek letter
        if self.replace_greek:
            for a, b in greek_alphabet_mapping.items():
                text = re.sub(a, b, text)

        # replace acronyms
        if self.acronym_file is not None:
            for compiled, new_word in self.acronym_dict.items():
                text = re.sub(compiled,
                              lambda string:
                              self._smart_acronym_replacer(string,
                                                           compiled,
                                                           new_word),
                              text)

        # misc
        text = " ".join(text.split())  # multiple spaces
        text = re.sub(r"\b[a-zA-Z]'(\w{2,})", r"\1", text)  # french apostrophe etc

        # optionnal stemmer
        if self.vectorizer == "TFIDF":
            if self.TFIDF_stem is True:
                text = " ".join([ps.stem(x) for x in text.split()])

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
                spacers_reg = re.compile("_|-|/")
                for t in tags:
                    if "AnnA" not in t:
                        t = re.sub(spacers_reg,  # replaces _ - and /
                                   " ",  # by a space
                                   " ".join(t.split("::")[-2:]))
                        # and keep only the last 2 levels of each tags
                        comb_text += " " + t

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

        ind_short = []
        for ind in self.df.index:
            if len(self.df.loc[ind, "text"]) < 10:
                ind_short.append(ind)
        if ind_short:
            red("{len(ind_short)} cards contain less than 10 characters after \
formatting: {','.join(ind_short)}")

        print("\n\nPrinting 2 random samples of your formated text, to help \
adjust formating issues:")
        pd.set_option('display.max_colwidth', 8000)
        max_length = 1000
        sub_index = random.choices(self.df.index.tolist(), k=2)
        for i in sub_index:
            if len(self.df.loc[i, "text"]) > max_length:
                ending = "...\n"
            else:
                ending = "\n"
            print(f" * {i} : {str(self.df.loc[i, 'text'])[0:max_length]}",
                  end=ending)
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
            alphanum = re.compile(r"[^ _\w]|\d|_|\b\w\b")

            def preprocessor(string):
                """
                prepare string of text to be vectorized by fastText
                * makes lowercase
                * replaces all non letters by a space
                * outputs the sentence as a list of words
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

            memoized_vec = memoize(ft.get_word_vector)

            def vec(string):
                return np.sum([memoized_vec(x)
                               for x in preprocessor(string)
                               if x != ""
                               ], axis=0)

            ft_vec = np.zeros(shape=(len(df.index), ft.get_dimension()),
                              dtype=np.float32)
            for i, x in enumerate(
                    tqdm(df.index, desc="Vectorizing using fastText")):
                ft_vec[i] = vec(str(df.loc[x, "text"]))
            ft_vec = normalize(ft_vec, norm='l2')

            if self.fastText_correction_vector:
                red(f"Temporarily disabled the feature 'correction vecto' as \
is it not yet fully implemented.")
#                norm_vec = vec(self.fastText_correction_vector)
#                norm_vec = normalize(norm_vec.reshape(-1, 1),
#                                     norm='l1').reshape(1, -1)
#                ft_vec = ft_vec + norm_vec

            # multiplying the median of each dimension by each row's
            # corresponding dimension. The idea is to avoid penalizing
            # cards that have few words dealing with the overall context
            # of the deck for example "where is the superior colliculi ?"
            # has one third of its words being "where", and only one word
            # being strictly medical.
            ft_vec = ft_vec * normalize(np.median(ft_vec, axis=0).reshape(1, -1), norm='l2')

            ft_vec = normalize(ft_vec, norm='l2')

            if self.fastText_dim is None or self.fastText_dim_algo is None:
                yel("Not applying dimension reduction.")
                df["VEC"] = [x for x in ft_vec]
                df["VEC_FULL"] = [x for x in ft_vec]
            else:
                if self.fastText_dim_algo == "UMAP":
                    print(f"Reducing dimensions to {self.fastText_dim} using UMAP")
                    red("(WARNING: EXPERIMENTAL FEATURE)")
                    try:
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
                        ft_vec_red = umap.UMAP(**umap_kwargs).fit_transform(ft_vec)
                        df["VEC"] = [x for x in ft_vec_red]
                        df["VEC_FULL"] = [x for x in ft_vec]
                    except Exception as e:
                        red(f"Error when computing UMAP reduction, using PCA \
as fallback: {e}")
                        self.fastText_dim_algo = "PCA"
                
                if self.fastText_dim_algo == "PCA":
                    print(f"Reducing dimensions to {self.fastText_dim} using PCA")
                    if self.fastText_dim > ft_vec.shape[1]:
                        red(f"Not enough dimensions: {ft_vec.shape[1]} < {self.fastText_dim}")
                        df["VEC"] = [x for x in ft_vec]
                        df["VEC_FULL"] = [x for x in ft_vec]
                    else:
                        try:
                            pca = PCA(n_components=self.fastText_dim, random_state=42)
                            ft_vec_red = pca.fit_transform(ft_vec)
                            evr = round(sum(pca.explained_variance_ratio_) * 100, 1)
                            whi(f"Explained variance ratio after PCA on fastText: {evr}%")
                            df["VEC"] = [x for x in ft_vec_red]
                        except Exception as e:
                            red(f"Error when computing PCA reduction, using all vectors: {e}")
                            df["VEC"] = [x for x in ft_vec]
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
                                         stop_words=None,
                                         ngram_range=(1, 10),
                                         max_features=n_features,
                                         norm="l2")
            # stop words have already been removed
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
                    evr = round(sum(svd.explained_variance_ratio_) * 100, 1)
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
                            self.TFIDF_dim += int(self.TFIDF_dim * 0.5)
                        self.TFIDF_dim = min(self.TFIDF_dim, 2000)
                        yel(f"Explained variance ratio is only {evr}% (\
retrying until above 80% or 2000 dimensions)")
                        continue
                whi(f"Explained variance ratio after SVD on Tf_idf: {evr}%")

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
        self.df_dist = pd.DataFrame(columns=df.index,
                                    index=df.index,
                                    data=pairwise_distances(
                                        [x for x in df[input_col]],
                                        n_jobs=-1,
                                        metric="cosine"))

        print("Computing mean distance...")
        # ignore the diagonal of the distance matrix to get a sensible mean
        # value then scale the matrix:
        mean_dist = np.nanmean(self.df_dist[self.df_dist != 0])
        std_dist = np.nanstd(self.df_dist[self.df_dist != 0])
        yel(f"Mean distance: {mean_dist}, std: {std_dist}\n")
#        self.df_dist -= mean_dist
#        self.df_dist /= std_dist

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
        lowest_values = [0]
        start_time = time.time()
        for i in range(9999):
            if printed is True:
                break
            if time.time() - start_time >= 60:
                red("Taking too long to find nonequal similar cards, skipping")
                break
            lowest_values.append(self.df_dist.values[self.df_dist.values > max(
                        lowest_values)].min())
            mins = np.where(self.df_dist.values == lowest_values[-1])
            mins = [x for x in zip(mins[0], mins[1]) if x[0] != x[1]]
            random.shuffle(mins)
            for pair in mins: 
                text_1 = str(df.loc[df.index[pair[0]]].text)
                text_2 = str(df.loc[df.index[pair[1]]].text)
                if text_1 != text_2:
                    red("Example among most semantically similar cards:")
                    yel(f"* {text_1[0:max_length]}...")
                    yel(f"* {text_2[0:max_length]}...")
                    printed = True
                    break
        if printed is False:
            red("Couldn't find lowest values to print!")
        print("")
        pd.reset_option('display.max_colwidth')
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
        # getting args etc
        reference_order = self.reference_order
        df = self.df
        target_deck_size = self.target_deck_size
        rated = self.rated_cards
        due = self.due_cards
        w1 = self.score_adjustment_factor[0]
        w2 = self.score_adjustment_factor[1]

        # settings
        display_stats = True
        use_index_of_score = False

        # setting interval to correct value for learning and relearnings:
        steps_L = [x / 1440 for x in self.deck_config["new"]["delays"]]
        steps_RL = [x / 1440 for x in self.deck_config["lapse"]["delays"]]
        for i in df.index:
            if df.loc[i, "type"] == 1:  # learning
                df.at[i, "interval"] = steps_L[int(str(df.loc[i, "left"])[-3:])-1]
                assert df.at[i, "interval"] >= 0
            elif df.loc[i, "type"] == 3:  # relearning
                df.at[i, "interval"] = steps_RL[int(str(df.loc[i, "left"])[-3:])-1]
                assert df.at[i, "interval"] >= 0
            if df.loc[i, "interval"] < 0:  # negative values are in seconds
                yel(f"Changing interval: cid: {i}, ivl: {df.loc[i, 'interval']} => {df.loc[i, 'interval']/(-86400)}")
                df.at[i, "interval"] /= -86400

        # setting rated cards value to nan value, to avoid them
        # skewing the dataset distribution:
        df.loc[rated, "interval"] = np.nan
        df.loc[rated, "due"] = np.nan
        df["ref"] = np.nan

        # computing reference order:
        if reference_order == "lowest_interval":
            ivl = df.loc[due, 'interval'].to_numpy().reshape(-1, 1)
            interval_cs = StandardScaler().fit_transform(ivl)
            df.loc[due, "ref"] = interval_cs

        elif reference_order == "order_added":
            due_order = np.argsort(due).reshape(-1, 1)
            df[due, "ref"] = StandardScaler().fit_transform(due_order)

        elif reference_order == "relative_overdueness":
            print("Computing relative overdueness...")
            anki_col_time = int(self._ankiconnect(
                action="getCollectionCreationTime"))
            time_offset = int((time.time() - anki_col_time) / 86400)

            df["ref_due"] = np.nan
            for i in due:
                df.at[i, "ref_due"] = df.loc[i, "odue"]
                if df.loc[i, "ref_due"] == 0:
                    df.at[i, "ref_due"] = df.at[i, "due"]
                if df.loc[i, "ref_due"] >= 100_000:  # timestamp and not days
                    df.at[i, "ref_due"] -= anki_col_time
                    df.at[i, "ref_due"] /= 86400
                assert df.at[i, "ref_due"] > 0
            overdue = (df.loc[due, "ref_due"] - time_offset).to_numpy().reshape(-1, 1)
            df.drop("ref_due", axis=1, inplace=True)

            ro = -1 * (df.loc[due, "interval"].values + 0.5) / (overdue.T + 0.5)
            ro_cs = StandardScaler().fit_transform(ro.T)
            df.loc[due, "ref"] = ro_cs

        assert len([x for x in rated if df.loc[x, "status"] != "rated"]) == 0
        red(f"\nCards identified as rated in the past {self.rated_last_X_days} days: \
{len(rated)}")

        # contain the index of the cards that will be use when
        # computing optimal order
        indQUEUE = rated[:]
        indTODO = [x for x in df.index.tolist() if x not in indQUEUE]
        # at each turn of the scoring algorithm, all cards whose index is
        # in indTODO will have their distance compared to all cards whose
        # index is in indQUEUE. The lowest score card is then taken from
        # indTODO and added to indQUEUE. Rinse and repeat until queue is
        # the size desired by the user or indTODO is empty.

        # remove potential siblings of indTODO, only if the intent is not
        # to study all the backlog over a few days:
        if (target_deck_size not in ["all", "100%", 1, "1"]) and (len(indTODO) >= 500):
            noteCard = {}
            for card, note in {df.loc[x].name: df.loc[x, "note"]
                               for x in indTODO}.items():
                if note not in noteCard:
                    noteCard[note] = card
                else:
                    if float(df.loc[noteCard[note], "ref"]
                             ) > float(df.loc[card, "ref"]):
                        noteCard[note] = card  # always keep the smallest ref
                        # value, because it indicates urgency to review
            previous_len = len(indTODO)
            [indTODO.remove(x) for x in indTODO if x not in noteCard.values()]
            if previous_len - len(indTODO) == 0:
                yel("No siblings found.")
            else:
                red(f"Removed {previous_len-len(indTODO)} siblings cards out of \
{previous_len}.")
        else:
            yel("Not excluding siblings because AnnA assumes you want to keep \
all the cards and study the deck over multipl days.")

        # can't start with an empty queue so picking 1 urgent card:
        if len(indQUEUE) == 0:
            pool = df.loc[indTODO, "ref"].nsmallest(
                n=min(10,
                      len(indTODO)
                      )).index
            queue = random.choices(pool, k=1)
            indQUEUE.append(queue[-1])
            indTODO.remove(queue[-1])
        else:
            queue = []

        duewsb = indTODO[:]  # copy of indTODO, used when computing
        # improvement ratio

        # parsing desired deck size:
        if isinstance(target_deck_size, float):
            if target_deck_size < 1.0:
                target_deck_size = str(target_deck_size * 100) + "%"
        if isinstance(target_deck_size, str):
            if target_deck_size in ["all", "100%"]:
                red("Taking the whole deck.")
                target_deck_size = len(df.index) - len(rated)
            elif target_deck_size.endswith("%"):
                red(f"Taking {target_deck_size} of the deck.")
                target_deck_size = 0.01 * int(target_deck_size[:-1]) * (
                    len(df.index) - len(rated))
        target_deck_size = int(target_deck_size)

        # checking if desired deck size is feasible:
        if target_deck_size > len(indTODO):
            red(f"You wanted to create a deck with \
{target_deck_size} in it but only {len(indTODO)} cards remain, taking the \
lowest value.")
        queue_size_goal = min(target_deck_size, len(indTODO))

        # displaying stats of the reference order or the
        # distance matrix:
        if display_stats:
            pd.set_option('display.float_format', lambda x: '%.5f' % x)
            try:
                whi("\nScore stats of due cards (weight adjusted):")
                if w1 != 0:
                    whi(f"Reference score of due cards: \
{(w1*df.loc[due, 'ref']).describe()}\n")
                else:
                    whi(f"Not showing statistics of the reference score, you \
set its adjustment weight to 0")
                val = pd.DataFrame(data=w2*self.df_dist.values.flatten(),
                                   columns=['distance matrix']).describe(include='all')
                whi(f"Distance: {val}\n\n")
            except Exception as e:
                red(f"Exception: {e}")
            pd.reset_option('display.float_format')

        # final check before computing optimal order:
        for x in ["interval", "ref", "due"]:
            assert np.sum(np.isnan(df.loc[rated, x].values)) == len(rated)
            assert np.sum(np.isnan(df.loc[due, x].values)) == 0

        if use_index_of_score:
            df.loc[indTODO, "index_ref"] = df.loc[indTODO, "ref"].values.argsort()
            col_ref = "index_ref"
            def combinator(array):
                return 0.9 * np.min(array, axis=0).argsort() + 0.1 * np.mean(array, axis=0).argsort()
        else:
            col_ref = "ref"
            def combinator(array):
                return 0.9 * np.min(array, axis=0) + 0.1 * np.mean(array, axis=0)

        with tqdm(desc="Computing optimal review order",
                  unit=" card",
                  initial=len(rated),
                  smoothing=0,
                  total=queue_size_goal + len(rated)) as pbar:
            while len(queue) < queue_size_goal:
                queue.append(indTODO[
                        (w1*df.loc[indTODO, col_ref].values -\
                         w2*combinator(self.df_dist.loc[indQUEUE, indTODO].values)
                         ).argmin()])
                indQUEUE.append(indTODO.pop(indTODO.index(queue[-1])))
                pbar.update(1)

        assert indQUEUE == rated + queue

        try:
            if w1 == 0:
                yel("Not showing distance without AnnA because you set \
the adjustment weight of the reference score to 0.")
            else:
                woAnnA = [x
                          for x in df.sort_values(
                              "ref", ascending=True).index.tolist()
                          if x in duewsb][0:len(queue)]

                common = len(set(queue) & set(woAnnA))
                if common / len(queue) >= 0.95:
                    yel("Not displaying Improvement Ratio because almost \
all cards were included in the new queue.")
                else:
                    spread_queue = np.mean(self.df_dist.loc[queue, queue].values.flatten())
                    spread_else = np.mean(self.df_dist.loc[woAnnA, woAnnA].values.flatten())

                    red("Mean of distance in the new queue:")
                    yel(spread_queue)
                    red(f"Cards in common: {common} in a queue of \
{len(queue)} cards.")
                    red("Mean of distance of the queue if you had not used \
AnnA:")
                    yel(spread_else)

                    ratio = round(spread_queue / spread_else, 3)
                    red("Improvement ratio:")
                    red(pyfiglet.figlet_format(str(ratio)))

        except Exception as e:
            red(f"\nException: {e}")

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
            return True
        else:
            if self.deck_template is not None:
                deck_template = self.deck_template
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

        _threaded_value_setter(card_list=self.opti_rev_order,
                               tqdm_desc="Altering due order",
                               keys=["due"],
                               newValues=None)
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
        the file supplied by the argument `acronym_file`
        * acronyms found in OCR caption are removed by default
        """
        if not len(self.acronym_dict.keys()):
            return True

        full_text = " ".join(self.df["text"].tolist()).replace("'", " ")
        if exclude_OCR_text:
            full_text = re.sub(" Caption:___.*?___ ", " ", full_text,
                               flags=re.MULTILINE | re.DOTALL)

        matched = list({x for x in re.findall("[A-Z][A-Z0-9]{2,}", full_text)
                        if x.lower() not in full_text})
        # if exists as lowercase : probably just shouting for emphasis

        if len(matched) == 0:
            print("No acronym found in those cards.")
            return True

        for compiled in self.acronym_dict:
            for acr in matched:
                if re.match(compiled, acr) is not None:
                    matched.remove(acr)

        if not matched:
            print("All found acronyms were already replaced using \
the data in `acronym_list`.")
        else:
            print("List of some acronyms still found:")
            if exclude_OCR_text:
                print("(Excluding OCR text)")
            pprint(random.choices(matched, k=min(5, len(matched))))
        print("")
        return True


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

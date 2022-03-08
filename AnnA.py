import logging
import gc
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
from tokenizers import Tokenizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

# avoids annoying warning
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# makes the script interuptible, resume it using c+enter
signal.signal(signal.SIGINT, (lambda signal, frame: pdb.set_trace()))

# adds logger, restrict it to 5000 lines
Path("logs.txt").touch(exist_ok=True)
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
    To control logging level for various modules used in the application:
    https://github.com/huggingface/transformers/issues/3050#issuecomment-682167272
    """
    prefix_re = re.compile(fr'^(?:{ "|".join(prefices) })')
    for name in logging.root.manager.loggerDict:
        if re.match(prefix_re, name):
            logging.getLogger(name).setLevel(level)


def coloured_log(color_asked):
    """used to print color coded logs"""
    col_red = "\033[91m"
    col_yel = "\033[93m"
    col_rst = "\033[0m"

    if color_asked == "white":
        def printer(string, **args):
            if isinstance(string, list):
                string = ",".join(string)
            log.info(string)
            tqdm.write(col_rst + string + col_rst, **args)
    elif color_asked == "yellow":
        def printer(string, **args):
            if isinstance(string, list):
                string = ",".join(string)
            log.warn(string)
            tqdm.write(col_yel + string + col_rst, **args)
    elif color_asked == "red":
        def printer(string, **args):
            if isinstance(string, list):
                string = ",".join(string)
            log.error(string)
            tqdm.write(col_red + string + col_rst, **args)
    return printer


whi = coloured_log("white")
yel = coloured_log("yellow")
red = coloured_log("red")

set_global_logging_level(logging.ERROR,
                         ["transformers", "nlp", "torch",
                          "tensorflow", "sklearn", "nltk"])


class AnnA:
    """
    just instantiating the class does the job, as you can see in the
    __init__ function
    """
    def __init__(self,

                 # most important:
                 deckname=None,
                 reference_order="relative_overdueness",  # any of "lowest_interval", "relative overdueness", "order_added"
                 task="filter_review_cards", # any of "filter_review_cards", "bury_excess_review_cards", "bury_excess_learning_cards"
                 target_deck_size="deck_config",  # format: 80%, 0.8, "all", "deck_config"
                 stopwords_lang=["english", "french"],
                 rated_last_X_days=4,
                 score_adjustment_factor=(1, 2),
                 field_mappings="field_mappings.py",
                 acronym_file="acronym_file.py",
                 acronym_list=None,

                 # others:
                 minimum_due=15,
                 highjack_due_query=None,
                 highjack_rated_query=None,
                 low_power_mode=False,
                 log_level=2,  # 0, 1, 2
                 replace_greek=True,
                 keep_OCR=True,
                 tags_to_ignore=None,
                 tags_separator="::",
                 fdeckname_template=None,
                 show_banner=True,
                 skip_print_similar=False,

                 # vectorization:
                 vectorizer="TFIDF",  # can only be "TFIDF" but left for legacy reason
                 TFIDF_dim=100,
                 TFIDF_tokenize=True,
                 TFIDF_stem=False,
                 ):

        if show_banner:
            red(pyfiglet.figlet_format("AnnA"))
            red("(Anki neuronal Appendix)\n\n")

        gc.collect()

        # miscellaneous
        if log_level == 0:
            log.setLevel(logging.ERROR)
        elif log_level == 1:
            log.setLevel(logging.WARNING)
        elif log_level >= 2:
            log.setLevel(logging.DEBUG)
        else:
            log.setLevel(logging.INFO)

        assert vectorizer == "TFIDF"

        # loading args
        self.replace_greek = replace_greek
        self.keep_OCR = keep_OCR
        if keep_OCR:
            self.OCR_content = ""
        self.target_deck_size = target_deck_size
        self.rated_last_X_days = rated_last_X_days
        self.minimum_due = minimum_due
        self.highjack_due_query = highjack_due_query
        self.highjack_rated_query = highjack_rated_query
        self.score_adjustment_factor = score_adjustment_factor
        self.reference_order = reference_order
        self.field_mappings = field_mappings
        self.acronym_file = acronym_file
        self.acronym_list = acronym_list
        self.tags_to_ignore = tags_to_ignore
        self.tags_separator = tags_separator
        self.low_power_mode = low_power_mode
        self.vectorizer = vectorizer
        self.stopwords_lang = stopwords_lang
        self.TFIDF_dim = TFIDF_dim
        self.TFIDF_stem = TFIDF_stem
        self.TFIDF_tokenize = TFIDF_tokenize
        self.task = task
        self.fdeckname_template = fdeckname_template
        self.skip_print_similar = skip_print_similar

        # args sanity checks and initialization
        if isinstance(self.target_deck_size, int):
            self.target_deck_size = str(self.target_deck_size)
        assert TFIDF_stem + TFIDF_tokenize != 2
        assert reference_order in ["lowest_interval", "relative_overdueness",
                                   "order_added"]
        assert task in ["filter_review_cards",
                        "bury_excess_learning_cards",
                        "bury_excess_review_cards"]
        if self.tags_to_ignore is None:
            self.tags_to_ignore = []
        if task != "filter_review_cards" and self.fdeckname_template is not None:
            red("Ignoring argument 'fdeckname_template' because 'task' is not \
set to 'filter_review_cards'.")
        if low_power_mode:
            if TFIDF_dim > 50:
                red(f"Low power mode is activated, it is usually recommended \
to set low values of TFIDF_dim (currently set at {TFIDF_dim} dimensions)")
            if not self.skip_print_similar:
                self.skip_print_similar = True
                red("Enabling 'skip_print_similar' because 'low_power_mode' \
is set to True")

        if TFIDF_tokenize:
            # from : https://huggingface.co/bert-base-multilingual-cased/
            self.tokenizer = Tokenizer.from_file("./bert-base-multilingual-cased_tokenizer.json")
            self.tokenizer.no_truncation()
            self.tokenizer.no_padding()
            self.tokenize = lambda x : [x for x in self.tokenizer.encode(x).tokens if x not in ["[CLS]", "[SEP]"]]
        else:
            self.tokenize = lambda x : x

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
                [temp.extend(self.tokenize(x)) for x in stops]
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
        self.deckname = self._deckname_check(deckname)
        yel(f"Selected deck: {self.deckname}\n")
        self.deck_config = self._call_anki(action="getDeckConfig",
                                             deck=self.deckname)
        if self.target_deck_size == "deck_config":
            self.target_deck_size = str(self.deck_config["rev"]["perDay"])
            yel(f"Set 'target_deck_size' to deck's value: {self.target_deck_size}")

        if task in ["bury_excess_learning_cards",
                      "bury_excess_review_cards"]:
            # bypasses most of the code to bury learning cards
            # directly in the deck without creating filtered decks
            if task == "bury_excess_learning_cards":
                yel("Task : bury some learning cards")
            elif task == "bury_excess_review_cards":
                yel("Task : bury some reviews\n")
            self._init_dataFrame()
            if self.not_enough_cards is True:
                return
            self._format_card()
            if self.low_power_mode:
                red("Not printing acronyms because low_power_mode is set to 'True'")
            else:
                self._print_acronyms()
            self._compute_card_vectors()
            self._compute_distance_matrix()
            self._compute_opti_rev_order()
            self._bury_or_create_filtered(task=task)
        else:
            yel("Task : created filtered deck containing review cards")
            self._init_dataFrame()
            if self.not_enough_cards is True:
                return
            self._format_card()
            if self.low_power_mode:
                red("Not printing acronyms because low_power_mode is set to 'True'")
            else:
                self._print_acronyms()
            self._compute_card_vectors()
            self._compute_distance_matrix()
            self._compute_opti_rev_order()
            if task == "filter_review_cards":
                self._bury_or_create_filtered()

        red(f"\nDone with task '{self.task}' on deck '{self.deckname}'")
        gc.collect()

    @classmethod
    def _call_anki(self, action, **params):
        """ bridge between local python libraries and AnnA Companion addon
        (a fork from anki-connect) """
        def request_wrapper(action, **params):
            return {'action': action, 'params': params, 'version': 6}

        requestJson = json.dumps(request_wrapper(action, **params)
                                 ).encode('utf-8')
        try:
            response = json.load(urllib.request.urlopen(
                urllib.request.Request(
                    'http://localhost:8775',
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

    def _getCardsInfo(self, card_id):
        """ get all information from a card from its card id

        * Due to the time it takes to get thousands of cards, I decided
            to used Threading extensively if requesting data for more than 50
            cards
        * There doesn't seem to be a way to display progress during loading,
            so I hacked a workaround by displaying one bar for each batch
            but it's minimally informative
        """
        if isinstance(card_id, int):
            card_id = [card_id]
        if len(card_id) < 50:
            r_list = []
            for card in tqdm(card_id):
                r_list.extend(self._call_anki(action="cardsInfo",
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
                out_list = self._call_anki(action="cardsInfo",
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

    def _deckname_check(self, deckname):
        """
        check if the deck you're calling AnnA exists or not
        if not, user is asked to enter the name, suggesting autocompletion
        """
        decklist = self._call_anki(action="deckNames") + ["*"]
        if deckname is not None:
            if deckname not in decklist:
                red("Couldn't find this deck.")
                deckname = None
        if deckname is None:
            auto_complete = WordCompleter(decklist,
                                          match_middle=True,
                                          ignore_case=True)
            deckname = ""
            # in the middle of the prompt
            time.sleep(0.5)
            while deckname not in decklist:
                deckname = prompt("Enter the name of the deck:\n>",
                                  completer=auto_complete)
        return deckname

    def _init_dataFrame(self):
        """
        create a pandas DataFrame with the information gathered from
        anki (via the bridge addon) such as card fields, tags, intervals, etc
        """
        if self.highjack_due_query is not None:
            red("Highjacking due card list:")
            query = self.highjack_due_query
            red(" >  '" + query + "'")
            due_cards = self._call_anki(action="findCards", query=query)
            whi(f"Found {len(due_cards)} cards...\n")

        elif self.task in ["filter_review_cards", "bury_excess_review_cards"]:
            yel("Getting due card list...")
            query = f"\"deck:{self.deckname}\" is:due is:review -is:learn -is:suspended -is:buried -is:new -rated:1"
            whi(" >  '" + query + "'")
            due_cards = self._call_anki(action="findCards", query=query)
            whi(f"Found {len(due_cards)} reviews...\n")

        elif self.task == "bury_excess_learning_cards":
            yel("Getting is:learn card list...")
            query = f"\"deck:{self.deckname}\" is:due is:learn -is:suspended -rated:1 -rated:2:1 -rated:2:2"
            whi(" >  '" + query + "'")
            due_cards = self._call_anki(action="findCards", query=query)
            whi(f"Found {len(due_cards)} learning cards...\n")

        rated_cards = []
        if self.highjack_rated_query is not None:
            red("Highjacking rated card list:")
            query = self.highjack_rated_query
            red(" >  '" + query + "'")
            rated_cards = self._call_anki(action="findCards", query=query)
            red(f"Found {len(rated_cards)} cards...\n")
        elif self.rated_last_X_days not in [0, None]:
            yel(f"Getting cards that where rated in the last {self.rated_last_X_days} days...")
            query = f"\"deck:{self.deckname}\" rated:{self.rated_last_X_days} -is:suspended -is:buried"
            whi(" >  '" + query + "'")
            rated_cards = self._call_anki(action="findCards", query=query)
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

        if len(self.due_cards) < self.minimum_due:
            red(f"Number of due cards is {len(self.due_cards)} which is \
less than threshold ({self.minimum_due}).\nStopping.")
            self.not_enough_cards = True
            return
        else:
            self.not_enough_cards = False

        combined_card_list = list(rated_cards + due_cards)

        list_cardInfo = []

        n = len(combined_card_list)
        yel(f"Asking Anki for information about {n} cards...")
        start = time.time()
        list_cardInfo.extend(
            self._getCardsInfo(
                card_id=combined_card_list))
        whi(f"Got all infos in {int(time.time()-start)} seconds.\n")

        for i, card in enumerate(list_cardInfo):
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

    def _regexp_acronym_replacer(self, string, compiled, new_w):
        """
        this function is needed to replace acronym containing match groups
        For example: replacing 'IL2' by "IL2 Interleukin 2"
        """
        if len(string.groups()):
            for i in range(len(string.groups())):
                if string.group(i + 1) is not None:
                    new_w = new_w.replace('\\' + str(i + 1),
                                          string.group(i + 1))
        out = string.group(0) + f" {new_w} "
        return out

    def _store_OCR(self, matched):
        """storing OCR value to use later with self._print_acronyms"""
        self.OCR_content += " " + matched.group(1)
        return " " + matched.group(1) + " "

    def _text_formatter(self, text):
        """
        process and formats each card's text, including :
        * html removal
        * acronym replacement
        * greek replacement
        * OCR text extractor
        """
        text = text.replace("&amp;", "&"
                    ).replace("/", " / "
                    ).replace("+++", " important "
                    ).replace("&nbsp", " "
                    ).replace("\u001F", " ")

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
            text += " list list list list list"

        # remove html spaces
        text = re.sub('\\n|</?div>|<br>|</?span>|</?li>|</?ul>', " ", text)
        text = re.sub('</?blockquote(.*?)>', " ", text)

        # OCR
        if self.keep_OCR:
            # keep image title (usually OCR)
            text = re.sub("<img src=.*? title=\"(.*?)\".*?>",
                          lambda string: self._store_OCR(string),
                          text,
                          flags=re.M | re.DOTALL)

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
                              self._regexp_acronym_replacer(string,
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
        goes through each cards and keep only fields deemed useful in
        the file from argument 'field_mapping', then adds the field to
        a single text per card in column 'comb_text'

        * If no corresponding fields are found, only the first one is kept
        * If no corresponding notetype are found, taking the closest name
        * 2 lowest level tags will be appended at the end of the text
        * Threading is used to speed things up
        * Mentionning several times a field for a specific note_type
            in field_mapping results in the field taking more importance.
            For example you can give more importance to field "Body" of a
            cloze than to the field "More"
        """
        def _threaded_field_filter(df, index_list, lock, pbar,
                                   stopw_compiled, spacers_compiled):
            """
            threaded call to speed up execution
            """
            for index in index_list:
                card_model = df.loc[index, "modelName"]
                fields_to_keep = []

                # determines which is the corresponding model described
                # in field_dic
                field_dic = self.field_dic
                target_model = []
                if card_model in field_dic:
                    target_model = [card_model]
                    fields_to_keep = field_dic[target_model[0]]
                else:
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
'{card_model}'. Selecting '{target_model[0]}'")

                # concatenates the corresponding fields into one string:
                field_list = list(df.loc[index, "fields"])
                if fields_to_keep == "take_first_fields":
                    fields_to_keep = ["", ""]
                    for f in field_list:
                        order = df.loc[index, "fields"][f.lower()]["order"]
                        if order == 0:
                            fields_to_keep[0] = f
                        elif order == 1:
                            fields_to_keep[1] = f
                    with lock:
                        to_notify.append(f"No matching notetype found for \
{card_model}. Keeping the first 2 fields: {', '.join(fields_to_keep)}")
                elif fields_to_keep == "take_all_fields":
                    fields_to_keep = sorted(field_list,
                                            key=lambda x: int(df.loc[index, "fields"][x.lower()]["order"]))

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
                for t in tags:
                    if ("AnnA" not in t) and (t not in self.tags_to_ignore):
                        t = re.sub(spacers_compiled,  # replaces _ - and /
                                   " ",  # by a space
                                   " ".join(t.split(self.tags_separator)[-2:]))
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
        spacers_compiled = re.compile("_|-|/")

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
                                                stopw_compiled,
                                                spacers_compiled),
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
        self.df["text"] = self.df["comb_text"].progress_apply(lambda x: self._text_formatter(x))

        ind_short = []
        for ind in self.df.index:
            if len(self.df.loc[ind, "text"]) < 10:
                ind_short.append(ind)
        if ind_short:
            red("{len(ind_short)} cards contain less than 10 characters after \
formatting: {','.join(ind_short)}")

        yel("\n\nPrinting 2 random samples of your formated text, to help \
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

    def _compute_card_vectors(self):
        """
        Assigne vectors to each card's 'comb_text', using TFIDF as vectorizer.

        After calling this function df["VEC"] contains either all the vectors
            or less if you enabled dimensionality reduction
        """
        df = self.df

        if self.low_power_mode is True:
            ngram_val = (1, 1)
        else:
            ngram_val = (1, 5)
        vectorizer = TfidfVectorizer(strip_accents="ascii",
                                     lowercase=True,
                                     tokenizer=self.tokenize,
                                     stop_words=None,
                                     ngram_range=ngram_val,
                                     max_features=10_000,
                                     norm="l2")
        # stop words have already been removed
        t_vec = vectorizer.fit_transform(tqdm(df["text"],
                                         desc="Vectorizing text using \
TFIDF"))
        if self.TFIDF_dim is None:
            df["VEC"] = [x for x in t_vec]
        else:
            while True:
                yel(f"\nReducing dimensions to {self.TFIDF_dim} using SVD...", end= " ")
                svd = TruncatedSVD(n_components=min(self.TFIDF_dim,
                                                    t_vec.shape[1] - 1))
                t_red = svd.fit_transform(t_vec)
                evr = round(sum(svd.explained_variance_ratio_) * 100, 1)
                if evr >= 70:
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
                    red(f"Explained variance ratio is only {evr}% (\
retrying until above 70% or 2000 dimensions)", end= " ")
                    continue
            yel(f"Explained variance ratio after SVD on Tf_idf: {evr}%")

            df["VEC"] = [x for x in t_red]

        self.df = df
        return True

    def _compute_distance_matrix(self, input_col="VEC"):
        """
        compute distance matrix : a huge matrix containing the
            cosine distance between the vectors of each cards.
        * scikit learn allows a parallelized computation
        """
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
        mean_dist = round(np.nanmean(self.df_dist[self.df_dist != 0]), 2)
        std_dist = round(np.nanstd(self.df_dist[self.df_dist != 0]), 2)
        yel(f"Mean distance: {mean_dist}, std: {std_dist}\n")

        if self.skip_print_similar is False:
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
                    red("Taking too long to find nonequal similar cards, \
skipping")
                    break
                lowest_values.append(self.df_dist.values[
                    self.df_dist.values > max(lowest_values)].min())
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
        1. calculates the 'ref' column. The lowest the 'ref', the more urgent
            the card needs to be reviewed. The computation used depends on
            argument 'reference_order', hence picking a card according to its
            'ref' only can be the same as using a regular filtered deck with
            'reference_order' set to 'relative_overdueness' for example.
            Some of the ref columns are centered and scaled.
        2. remove siblings of the due list of found (except if the queue
            is meant to contain a lot of cards, then siblings are not removed)
        3. prints a few stats about 'ref' distribution in your deck as well
            as 'distance' distribution
        4. assigns a score to each card, the lowest score at each turn is
            added to the queue, each new turn compares the cards to
            the present queue. The algorithm is described in more details in
            the docstring of function 'combinator'.
            Here's the gist:
            At each turn, the card from indTODO with the lowest score is
                removed from indTODO and added to indQUEUE
            The score is computed according to the formula:
               score of indTODO =
               ref -
                     [
                       0.9 * min(similarity to each card of indQUEUE)
                       +
                       0.1 * mean(similarity to each card of indQUEUE)
                      ]
               (indQUEUE = recently rated cards + queue)
        5. displays improvement_ratio, a number supposed to indicate how much
            better the new queue is compared to the original queue.
        """
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
            df.loc[due, "ref"] = StandardScaler().fit_transform(np.array(due).reshape(-1, 1))

        elif reference_order == "relative_overdueness":
            print("Computing relative overdueness...")
            anki_col_time = int(self._call_anki(
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
            overdue = df.loc[due, "ref_due"] - time_offset
            df.drop("ref_due", axis=1, inplace=True)

            # the code for relative overdueness is not exactly the same as
            # in anki, as I was not able to replicate it.
            # Here's a link to one of the implementation : https://github.com/ankitects/anki/blob/afff4fc437f523a742f617c6c4ad973a4d477c15/rslib/src/storage/card/filtered.rs
            ro = -1 * (df.loc[due, "interval"].values + 0.001) / (overdue + 0.001)
            ro_clipped = np.clip(ro, -50, 50)
            ro_cs = StandardScaler().fit_transform(ro_clipped.values.reshape(-1, 1))
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
        if (target_deck_size not in ["all", "100%", "1"]) and (len(indTODO) >= 500):
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
all the cards and study the deck over multiple days.")

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

        def combinator(array):
            """
            'array' represents:
                * columns : the cards of indTODO
                * rows : the cards of indQUEUE
                * the content of each cell is the similarity between them
                    (lower value means very similar)
            Hence, for a given array:
            * if a cell of np.min is high, then the corresponding card of
                indTODO is not similar to any card of indQUEUE (i.e. its
                closest card in indQUEUE is quite different). This card is a
                good candidate to add to indQEUE.
            * if a cell of np.mean is high, then the corresponding card of
                indTODO is different from most cards of indQUEUE (i.e. it is
                quite different from most cards of indQUEUE). This cas is
                a good candidate to add to indQUEUE
            * Naturally, np.min is given more importance than np.mean

            Best candidates are cards with high combinator output.
            The outut is added to the 'ref' of each indTODO card.

            Hence, at each turn, the card from indTODO with the lowest
                'w1*ref - w2*combinator' is removed from indTODO and added
                to indQUEUE.

            The content of 'queue' is the list of card_id in best review order.
            """
            return 0.9 * np.min(array, axis=1) + 0.1 * np.mean(array, axis=1)

        with tqdm(desc="Computing optimal review order",
                  unit=" card",
                  initial=len(rated),
                  smoothing=0,
                  total=queue_size_goal + len(rated)) as pbar:
            while len(queue) < queue_size_goal:
                queue.append(indTODO[
                        (w1*df.loc[indTODO, "ref"].values -\
                         w2*combinator(self.df_dist.loc[indTODO, indQUEUE].values)
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

                    red("Mean of distance in the new queue:", end=" ")
                    yel(str(spread_queue))
                    red(f"Cards in common: {common} in a queue of \
{len(queue)} cards.")
                    red("Mean of distance of the queue if you had not used \
AnnA:", end=" ")
                    yel(str(spread_else))

                    ratio = round(spread_queue / spread_else * 100 - 100, 1)
                    red("Improvement ratio:")
                    if ratio >= 0:
                        sign = "+"
                    else:
                        sign = ""
                    red(pyfiglet.figlet_format(f"{sign}{ratio}%"))

        except Exception as e:
            red(f"\nException: {e}")

        self.opti_rev_order = [int(x) for x in queue]
        self.df = df
        return True

    def _print_acronyms(self, exclude_OCR_text=True):
        """
        Shows acronym present in the collection that were not found in
            the file supplied by the argument `acronym_file`.
            This is used to know which acronym you forgot to specify in
            `acronym_file`
        * acronyms found in OCR text are ignored by default, because they
            cause too many false positive.
        """
        yel("Looking for acronyms that perhaps should be in 'acronym_file'...")
        if not len(self.acronym_dict.keys()):
            return True

        full_text = " ".join(self.df["text"].tolist()).replace("'", " ")
        if exclude_OCR_text:
            ocr = re.findall("[A-Z][A-Z0-9]{2,}",
                    string=self.OCR_content,
                    flags=re.MULTILINE | re.DOTALL)
        else:
            ocr = []

        def exclude(word):
            """if exists as lowercase in text : probably just shouting for emphasis
               if exists in ocr : ignore"""
            if word.lower() in full_text or word in ocr:
                return False
            else:
                return True

        matched = list(set(
            [x for x in re.findall("[A-Z][A-Z0-9]{2,}", full_text)
             if exclude(x)]))

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
            pprint(", ".join(random.choices(matched, k=min(5, len(matched)))))
        print("")
        return True

    def _bury_or_create_filtered(self,
                           fdeckname_template=None,
                           task=None):
        """
        Either bury cards that are not in the optimal queue or create a
            filtered deck containing the cards to review in optimal order.

        * The filtered deck is created with setting 'sortOrder = 0', meaning
            ("oldest seen first"). This function then changes the review order
            inside this deck. That's why rebuilding this deck will keep the
            cards but lose the order.
        * fdeckname_template can be used to automatically put the filtered
            decks to a specific location in your deck hierarchy. Leaving it
            to None will make the filtered deck appear alongside the
            original deck
        * This uses a threaded call to increase speed.
        * I do a few sanity check to see if the filtered deck
            does indeed contain the right number of cards and the right cards
        * -100 000 seems to be the starting value for due order in filtered
            decks by anki : cards are review from lowest to highest "due_order".
        * if task is set to 'bury_excess_learning_cards' or
            'bury_excess_review_cards', then no filtered deck will be created
            and AnnA will just bury some cards that are too similar to cards
            that you will review.
        """
        if task in ["bury_excess_learning_cards", "bury_excess_review_cards"]:
            to_keep = self.opti_rev_order
            to_bury = [x for x in self.due_cards if x not in to_keep]
            assert len(to_bury) < len(self.due_cards)
            red(f"Burying {len(to_bury)} cards out of {len(self.due_cards)}.")
            red("This will not affect the due order.")
            self._call_anki(action="bury",
                              cards=to_bury)
            return True
        else:
            if self.fdeckname_template is not None:
                fdeckname_template = self.fdeckname_template
            if fdeckname_template is not None:
                filtered_deck_name = str(fdeckname_template + f" - {self.deckname}")
                filtered_deck_name = filtered_deck_name.replace("::", "_")
            else:
                filtered_deck_name = f"{self.deckname} - AnnA Optideck"
            self.filtered_deck_name = filtered_deck_name

            while filtered_deck_name in self._call_anki(action="deckNames"):
                red(f"\nFound existing filtered deck: {filtered_deck_name} \
You have to delete it manually, the cards will be returned to their original \
deck.")
                input("Done? >")

        whi(f"Creating deck containing the cards to review: \
{filtered_deck_name}")
        query = "is:due -rated:1 cid:" + ','.join(
                [str(x) for x in self.opti_rev_order])
        self._call_anki(action="createFilteredDeck",
                        newDeckName=filtered_deck_name,
                        searchQuery=query,
                        gatherCount=len(self.opti_rev_order) + 1,
                        reschedule=True,
                        sortOrder=0,
                        createEmpty=False)

        print("Checking that the content of filtered deck name is the same as \
 the order inferred by AnnA...", end="")
        cur_in_deck = self._call_anki(action="findCards",
                                      query=f"\"deck:{filtered_deck_name}\"")
        diff = [x for x in self.opti_rev_order + cur_in_deck
                if x not in self.opti_rev_order or x not in cur_in_deck]
        if len(diff) != 0:
            red("Inconsistency! The deck does not contain the same cards \
as opti_rev_order!")
            pprint(diff)
            red(f"\nNumber of inconsistent cards: {len(diff)}")

        yel("\nAsking anki to alter the filtered deck's due order...", end="")
        res = self._call_anki(action="setDueOrderOfFiltered",
                              cards=self.opti_rev_order)
        err = [x[1] for x in res if x[0] is False]
        if err:
            print("")
            raise(f"Error when setting due order : {err}")
        else:
            yel(" Done!")
            return True

    def display_opti_rev_order(self, display_limit=50):
        """
        instead of creating a deck or buring cards, prints the content
        of cards in the order AnnA though was best.
        Only used for debugging.
        """
        order = self.opti_rev_order[:display_limit]
        print(self.df.loc[order, "text"])
        return True

    def save_df(self, df=None, out_name=None):
        """
        export dataframe as pickle format in the folder DF_backups/
        """
        if df is None:
            df = self.df.copy()
        if out_name is None:
            out_name = "AnnA_Saved_DataFrame"
        cur_time = "_".join(time.asctime().split()[0:4]).replace(":",
                                                                 "h")[0:-3]
        name = f"{out_name}_{self.deckname}_{cur_time}.pickle"
        df.to_pickle("./.DataFrame_backups/" + name)
        print(f"Dataframe exported to {name}.")
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

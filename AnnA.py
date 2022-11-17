import textwrap
import traceback
import copy
import beepy
import argparse
import logging
import gc
from datetime import datetime
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
from tqdm_logger import TqdmLogger
import re
import importlib
from pathlib import Path
import threading
from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter
from plyer import notification

import joblib
import pandas as pd
import numpy as np
import Levenshtein as lev
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tokenizers import Tokenizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances, pairwise_kernels
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import radius_neighbors_graph

import networkx as nx
from plotly.colors import qualitative
from plotly.offline import plot as offpy
from plotly.graph_objs import (Scatter, scatter, Figure, Layout, layout)

import ankipandas as akp
import shutil

from utils.greek import greek_alphabet_mapping

# avoids annoying warning
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# makes the script interuptible, resume it using c+enter
signal.signal(signal.SIGINT, (lambda signal, frame: pdb.set_trace()))

# adds logger, restrict it to 5000 lines
Path("logs.txt").touch(exist_ok=True)
Path("logs.txt").write_text(
    "\n".join(
        Path("logs.txt").read_text().split("\n")[-10_000:]))
logging.basicConfig(filename="logs.txt",
                    filemode='a',
                    format=f"{time.ctime()}: %(message)s")
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

    # all logs are considered "errors" otherwise the datascience libs just
    # overwhelm the logs

    if color_asked == "white":
        def printer(string, **args):
            if isinstance(string, list):
                string = ",".join(string)
            log.error(string)
            tqdm.write(col_rst + string + col_rst, **args)
    elif color_asked == "yellow":
        def printer(string, **args):
            if isinstance(string, list):
                string = ",".join(string)
            log.error(string)
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


def _beep(message=None, **args):
    sound = "error"  # default sound

    if message is None:
        red("  ############")
        red("  ### BEEP ###")  # at least produce a written message
        red("  ############")
    else:
        try:
            if isinstance(message, list):
                message = "".join(message)
            elif not isinstance(message, str):
                message = str(message)
            # create notification with error
            red("NOTIF: " + message)
            notification.notify(title="AnnA",
                                message=message,
                                timeout=-1,
                                )
        except Exception as err:
            red(f"Error when creating notification: '{err}'")

    try:
        beepy.beep(sound, **args)
    except Exception:
        # retry sound if failed
        time.sleep(1)
        try:
            beepy.beep(sound, **args)
        except Exception:
            red("Failed to beep twice.")
    time.sleep(0.5)  # avoid too close beeps in a row


class AnnA:
    """
    just instantiating the class does the job, as you can see in the
    __init__ function
    """

    def __init__(self,

                 # most important arguments:
                 deckname=None,
                 reference_order="relative_overdueness",
                 # any of "lowest_interval", "relative overdueness",
                 # "order_added", "LIRO_mix", "just_add_KNN"
                 task="filter_review_cards",
                 # any of "filter_review_cards",
                 # "bury_excess_review_cards", "bury_excess_learning_cards"
                 target_deck_size="deck_config",
                 # format: 80%, "all", "deck_config"
                 max_deck_size=None,
                 stopwords_lang=["english", "french"],
                 rated_last_X_days=4,
                 score_adjustment_factor=[1, 2],
                 field_mappings="field_mappings.py",
                 acronym_file="acronym_file.py",
                 acronym_list=None,

                 # others:
                 minimum_due=5,
                 highjack_due_query=None,
                 highjack_rated_query=None,
                 low_power_mode=False,
                 log_level=0,  # 0, 1, 2
                 replace_greek=True,
                 keep_OCR=True,
                 append_tags=True,
                 tags_to_ignore=["AnnA"],
                 tags_separator="::",
                 add_KNN_to_field=False,
                 filtered_deck_name_template=None,
                 filtered_deck_by_batch=False,
                 filtered_deck_batch_size=25,
                 show_banner=True,
                 repick_task="boost",  # None, "addtag", "boost" or
                 # "boost&addtag"
                 enable_fuzz=True,
                 resort_by_dist=True,

                 # vectorization:
                 vectorizer="TFIDF",  # can only be "TFIDF" but
                 # left for legacy reason
                 TFIDF_dim="auto",
                 TFIDF_tokenize=True,
                 tokenizer_model="GPT",
                 plot_2D_embeddings=False,
                 TFIDF_stem=False,
                 dist_metric="RBF",  # 'RBF' or 'cosine'

                 whole_deck_computation=False,
                 profile_name=None,
                 ):

        if show_banner:
            red(pyfiglet.figlet_format("AnnA"))
            red("(Anki neuronal Appendix)\n\n")

        gc.collect()

        # init logging #######################################################

        self.log_level = log_level
        if log_level == 0:
            log.setLevel(logging.ERROR)
        elif log_level == 1:
            log.setLevel(logging.WARNING)
        elif log_level >= 2:
            log.setLevel(logging.DEBUG)
        else:
            log.setLevel(logging.INFO)

        # logger for tqdm progress bars
        self.t_strm = TqdmLogger("logs.txt")
        self.t_strm.reset()

        # loading arguments and proceed to check correct values ##############

        assert isinstance(
            replace_greek, bool), "Invalid type of `replace_greek`"
        self.replace_greek = replace_greek

        assert isinstance(keep_OCR, bool), "Invalid type of `keep_OCR`"
        self.keep_OCR = keep_OCR
        self.OCR_content = ""  # used to avoid looking for acronyms
        # in OCR content

        if isinstance(target_deck_size, int):
            assert target_deck_size > 0
            target_deck_size = str(target_deck_size)
        elif isinstance(target_deck_size, str):
            try:
                testint = int(target_deck_size)
            except Exception:
                pass
            assert "%" in target_deck_size or target_deck_size in [
                "all", "deck_config"] or (
                    isinstance(testint, int), (
                        "Invalid value for `target_deck_size`"))
        self.target_deck_size = target_deck_size
        if target_deck_size in ["all", 1.0, "100%"] and (
                task == "bury_excess_review_cards"):
            _beep(f"Arguments mean that all cards "
                 "will be selected and none will be buried. It makes no sense."
                 " Aborting.")
            raise Exception("Arguments mean that all cards will be selected "
                            "and none will be buried. It makes no sense."
                            " Aborting.")

        assert max_deck_size is None or max_deck_size >= 1, (
            "Invalid value for `max_deck_size`")
        self.max_deck_size = max_deck_size

        assert rated_last_X_days is None or (
            rated_last_X_days >= 0, (
                "Invalid value for `rated_last_X_days`"))
        self.rated_last_X_days = rated_last_X_days

        assert minimum_due >= 0, "Invalid value for `minimum_due`"
        self.minimum_due = minimum_due

        assert isinstance(highjack_due_query, (str, type(None))
                          ), "Invalid type of `highjack_due_query`"
        self.highjack_due_query = highjack_due_query

        assert isinstance(highjack_rated_query, (str, type(None))
                          ), "Invalid type of `highjack_rated_query`"
        self.highjack_rated_query = highjack_rated_query

        if isinstance(score_adjustment_factor, tuple):
            score_adjustment_factor = list(score_adjustment_factor)
        assert (isinstance(score_adjustment_factor, list)
                ), "Invalid type of `score_adjustment_factor`"
        assert len(
            score_adjustment_factor) == 2, (
                "Invalid length of `score_adjustment_factor`")
        for n in range(len(score_adjustment_factor)):
            if not isinstance(score_adjustment_factor[n], float):
                score_adjustment_factor[n] = float(score_adjustment_factor[n])
        self.score_adjustment_factor = score_adjustment_factor

        assert reference_order in ["lowest_interval",
                                   "relative_overdueness",
                                   "order_added",
                                   "LIRO_mix"], (
               "Invalid value for `reference_order`")
        self.reference_order = reference_order

        assert isinstance(append_tags, bool), "Invalid type of `append_tags`"
        self.append_tags = append_tags

        if tags_to_ignore is None or tags_to_ignore == "":
            tags_to_ignore = []
        if not isinstance(tags_to_ignore, list):
            assert "[" in tags_to_ignore, "missing '[' in 'tags_to_ignore'"
            tags_to_ignore = tags_to_ignore.replace("[", "").replace("]", "").strip().split(",")
        self.tags_to_ignore = [
                re.compile(".*" + t.strip() + ".*")
                for t in tags_to_ignore
                ]

        assert isinstance(add_KNN_to_field, bool), (
                "Invalid type of `add_KNN_to_field`")
        self.add_KNN_to_field = add_KNN_to_field
        assert isinstance(
            tags_separator, str), "Invalid type of `tags_separator`"
        self.tags_separator = tags_separator

        assert isinstance(
            low_power_mode, bool), "Invalid type of `low_power_mode`"
        self.low_power_mode = low_power_mode

        assert vectorizer == "TFIDF", "Invalid value for `vectorizer`"
        self.vectorizer = vectorizer

        assert isinstance(
            stopwords_lang, list), "Invalid type of var `stopwords_lang`"
        self.stopwords_lang = stopwords_lang

        assert isinstance(TFIDF_dim, (int, type(None), str)
                          ), "Invalid type of `TFIDF_dim`"
        if isinstance(TFIDF_dim, str):
            assert TFIDF_dim == "auto", "Invalid value for `TFIDF_dim`"
        self.TFIDF_dim = TFIDF_dim

        assert isinstance(plot_2D_embeddings, bool), (
            "Invalid type of `plot_2D_embeddings`")
        self.plot_2D_embeddings = plot_2D_embeddings
        assert isinstance(TFIDF_stem, bool), "Invalid type of `TFIDF_stem`"
        assert isinstance(
            TFIDF_tokenize, bool), "Invalid type of `TFIDF_tokenize`"
        assert TFIDF_stem + TFIDF_tokenize not in [0, 2], (
            "You have to enable either tokenization or stemming!")
        self.TFIDF_stem = TFIDF_stem
        self.TFIDF_tokenize = TFIDF_tokenize
        assert tokenizer_model.lower() in ["bert", "gpt", "both"], (
            "Wrong tokenizer model name!")
        self.tokenizer_model = tokenizer_model
        assert dist_metric.lower() in ["cosine", "rbf"], (
            "Invalid 'dist_metric'")
        self.dist_metric = dist_metric.lower()

        assert task in ["filter_review_cards",
                        "bury_excess_learning_cards",
                        "bury_excess_review_cards",
                        "just_add_KNN"], "Invalid value for `task`"
        self.task = task

        assert isinstance(filtered_deck_name_template, (str, type(
            None))), "Invalid type for `filtered_deck_name_template`"
        self.filtered_deck_name_template = filtered_deck_name_template

        assert isinstance(filtered_deck_by_batch,
                          bool), "Invalid type for `filtered_deck_by_batch`"
        self.filtered_deck_by_batch = filtered_deck_by_batch

        assert isinstance(filtered_deck_batch_size,
                          int), "Invalid type for `filtered_deck_batch_size`"
        self.filtered_deck_batch_size = filtered_deck_batch_size

        assert isinstance(whole_deck_computation,
                          bool), "Invalid type for `whole_deck_computation`"
        self.whole_deck_computation = whole_deck_computation

        assert isinstance(profile_name, (str, type(None))
                          ), "Invalid type for `profile_name`"
        self.profile_name = profile_name

        assert isinstance(repick_task, str), "Invalid type for `repick_task`"
        self.repick_task = repick_task

        assert isinstance(acronym_file, (str, type(None))
                          ), "Invalid type for `acronym_file`"
        self.acronym_file = acronym_file

        assert isinstance(acronym_list, (list, type(None))
                          ), "Invalid type for `acronym_list`"
        self.acronym_list = acronym_list

        assert isinstance(field_mappings, (str, type(None))
                          ), "Invalid type for `field_mappings`"
        self.field_mappings = field_mappings

        assert isinstance(enable_fuzz, bool), "Invalid type for 'enable_fuzz'"
        self.enable_fuzz = enable_fuzz

        assert isinstance(resort_by_dist, bool), (
            "Invalid type for 'resort_by_dist'")
        self.resort_by_dist = resort_by_dist

        # initialize joblib caching
        # self.mem = joblib.Memory("./cache", mmap_mode="r", verbose=0)

        # additional processing of arguments #################################

        # load or ask for deckname
        self.deckname = self._deckname_check(deckname)
        red(f"Selected deck: {self.deckname}\n")
        self.deck_config = self._call_anki(action="getDeckConfig",
                                           deck=self.deckname)

        global beep
        def beep(x):
            "simple overloading to display the deckname"
            try:
                return _beep(f"{self.deckname}: {x}")
            except Exception:
                return _beep(x)

        if task != "filter_review_cards" and (
                self.filtered_deck_name_template is not None):
            red("Ignoring argument 'filtered_deck_name_template' because "
                "'task' is not set to 'filter_review_cards'.")

        # load tokenizers
        if TFIDF_tokenize:
            if self.tokenizer_model.lower() in ["bert", "both"]:
                yel("Will use BERT as tokenizer.")
                self.tokenizer_bert = Tokenizer.from_file(
                    "./bert-base-multilingual-cased_tokenizer.json")
                self.tokenizer_bert.no_truncation()
                self.tokenizer_bert.no_padding()
                self.tokenize = self._bert_tokenize
            if self.tokenizer_model.lower() in ["gpt", "both"]:
                yel("Will use GPT as tokenizer.")
                self.tokenizer_gpt = Tokenizer.from_file(
                    "./gpt_neox_20B_tokenizer.json")
                self.tokenizer_gpt.no_truncation()
                self.tokenizer_gpt.no_padding()
                self.tokenize = self._gpt_tokenize
            if self.tokenizer_model.lower() == "both":
                yel("Using both GPT and BERT as tokenizers.")
                self.tokenize = lambda x: self._gpt_tokenize(
                        x) + self._bert_tokenize(x)
        else:
            # create dummy tokenizer
            self.tokenize = lambda x: x.replace("<NEWFIELD>", " ").strip(
                    ).split(" ")

        # load acronyms
        if self.acronym_file is not None and self.acronym_list is not None:
            file = Path(acronym_file)
            if not file.exists():
                beep(f"Acronym file was not "
                     f"found: {acronym_file}")
                raise Exception(f"Acronym file was not found: {acronym_file}")

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
                beep(f"No dictionnary found "
                     f"in {acronym_file}")
                raise SystemExit()

            if isinstance(self.acronym_list, str):
                self.acronym_list = [self.acronym_list]

            missing = [x for x in self.acronym_list
                       if x not in acr_dict_list]
            if missing:
                beep(f"Mising the following acronym "
                     "dictionnary in "
                     f"{acronym_file}: {','.join(missing)}")
                raise SystemExit()

            acr_dict_list = [x for x in acr_dict_list
                             if x in self.acronym_list]

            if len(acr_dict_list) == 0:
                beep(f"No dictionnary from "
                     f"{self.acr_dict_list} "
                     f"found in {acronym_file}")
                raise SystemExit()

            compiled_dic = {}
            notifs = []
            for item in acr_dict_list:
                acronym_dict = eval(f"acr_mod.{item}")
                for ac in acronym_dict:
                    if ac.lower() == ac:
                        compiled = re.compile(r"\b" + ac + r"\b",
                                              flags=(re.IGNORECASE |
                                                     re.MULTILINE |
                                                     re.DOTALL))
                    else:
                        compiled = re.compile(r"\b" + ac + r"\b",
                                              flags=(
                                                  re.MULTILINE |
                                                  re.DOTALL))
                    if compiled in compiled_dic:
                        notifs.append(
                                f"Pattern '{compiled}' found "
                                "multiple times in acronym dictionnary, "
                                "keeping only the last one.")
                    compiled_dic[compiled] = acronym_dict[ac]
            notifs = sorted(set(notifs))
            if notifs:
                for n in notifs:
                    beep(n)

            # checking if acronyms overlap, this can be intentionnal
            to_notify = []
            acronyms_overlapping = {}
            for compiled, value in compiled_dic.items():
                for compiled2, value2 in compiled_dic.items():
                    if compiled.pattern == compiled2.pattern:
                        continue
                    if re.match(compiled, value2):
                        if compiled.pattern not in acronyms_overlapping:
                            acronyms_overlapping[
                                    compiled.pattern] = [compiled2]
                        else:
                            acronyms_overlapping[compiled.pattern].append(compiled2)
                        to_notify.append(f"  * '{compiled.pattern}' matches "
                                         f"value of '{compiled2.pattern}'")
            to_notify = sorted(set(to_notify))
            if to_notify:
                red(f"\nFound {len(to_notify)} "
                    "overlapping "
                    "acronym patterns (this can be intentional):")
                for notif in to_notify:
                    yel(notif)
                print("\n")

            self.acronym_dict = compiled_dic
            self.acronyms_overlapping = acronyms_overlapping
        else:
            self.acronym_dict = {}
            self.acronyms_overlapping = {}

        # load field mappings
        if self.field_mappings is not None:
            f = Path(self.field_mappings)
            try:
                assert f.exists(), ("field_mappings file does not exist : "
                                    f"{self.field_mappings}")
                imp = importlib.import_module(
                    self.field_mappings.replace(".py", ""))
                self.field_dic = imp.field_dic
            except Exception as e:
                beep(f"Error with field mapping file, will use "
                     f"default values. {e}")
                self.field_dic = {"dummyvalue": "dummyvalue"}

        # load stop words
        try:
            stops = []
            stops.append("<NEWFIELD>")
            for lang in self.stopwords_lang:
                stops += stopwords.words(lang)
            if self.TFIDF_tokenize:
                temp = []
                [temp.extend(self.tokenize(x)) for x in stops]
                stops.extend(temp)
            elif self.TFIDF_stem:
                self.ps = PorterStemmer()
                stops += [self.ps.stem(x) for x in stops]
            self.stops = list(set(stops))
        except Exception as e:
            beep(f"Error when extracting stop words: {e}\n\n"
                 "Setting stop words list to None.")
            self.stops = None
        self.stopw_compiled = re.compile("\b" + "\b|\b".join(
            self.stops) + "\b", flags=(
                re.MULTILINE | re.IGNORECASE | re.DOTALL))
        assert "None" == self.repick_task or isinstance(self.repick_task, type(
            None)) or "addtag" in self.repick_task or (
                "boost" in self.repick_task), (
                    "Invalid value for `self.repick_task`")

        # actual execution ###################################################

        # load deck settings if needed
        if self.target_deck_size == "deck_config":
            self.target_deck_size = str(self.deck_config["rev"]["perDay"])
            yel("Set 'target_deck_size' to deck's value: "
                f"{self.target_deck_size}")

        if task in ["bury_excess_learning_cards",
                    "bury_excess_review_cards"]:
            # bypasses most of the code to bury learning cards
            # directly in the deck without creating filtered decks
            if task == "bury_excess_learning_cards":
                red("Task : bury some learning cards")
            elif task == "bury_excess_review_cards":
                red("Task : bury some reviews\n")
            self._init_dataFrame()
            if self.not_enough_cards is True:
                return
            self._format_card()
            if self.low_power_mode:
                red("Not printing acronyms because low_power_mode is set to "
                    "'True'")
            else:
                self._print_acronyms()
            self._compute_card_vectors()
            self._compute_distance_matrix()
            self._compute_opti_rev_order()
            self._compute_KNN()
            self._do_add_KNN_to_note()
            self._bury_or_create_filtered()

        elif task == "filter_review_cards":
            red("Task : created filtered deck containing review cards")
            self._init_dataFrame()
            if self.not_enough_cards is True:
                return
            self._format_card()
            if self.low_power_mode:
                red("Not printing acronyms because low_power_mode is set to "
                    "'True'")
            else:
                self._print_acronyms()
            self._compute_card_vectors()
            self._compute_distance_matrix()
            self._compute_opti_rev_order()
            self._compute_KNN()
            self._do_add_KNN_to_note()
            self._bury_or_create_filtered()

        elif task == "just_add_KNN":
            red("Task : find the nearest neighbour of each note and "
                "add it to a field.")
            whi("(Setting 'rated_last_X_days' to None)")
            self.rated_last_X_days = None
            self._init_dataFrame()
            if self.not_enough_cards is True:
                return
            self._format_card()
            if self.low_power_mode:
                red("Not printing acronyms because low_power_mode is set to "
                    "'True'")
            else:
                self._print_acronyms()
            self._compute_card_vectors()
            self._compute_distance_matrix()
            self._compute_KNN()
            self._do_add_KNN_to_note()

        else:
            raise ValueError(f"Invalid task value: {task}")

        if self.plot_2D_embeddings:
            try:
                self._do_2D_plots()
            except Exception as err:
                beep(f"Exception when plotting 2D embeddings: '{err}'")
                red(traceback.format_exc())
            signal.alarm(0)  # turn off timeout
        red(f"Done with task '{self.task}' on deck '{self.deckname}'")
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
            beep(f"{str(e)}: is Anki open and 'AnnA Companion addon' "
                 "enabled? Firewall issue?")
            raise Exception(f"{str(e)}: is Anki open and 'AnnA Companion "
                            "addon' enabled? Firewall issue?")

        if len(response) != 2:
            beep('response has an unexpected number of fields')
            raise Exception('response has an unexpected number of fields')
        if 'error' not in response:
            beep('response is missing required error field')
            raise Exception('response is missing required error field')
        if 'result' not in response:
            beep('response is missing required result field')
            raise Exception('response is missing required result field')
        if response['error'] is not None:
            beep(response['error'])
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
        if len(card_id) < 20:
            r_list = []
            for card in tqdm(card_id, file=self.t_strm):
                r_list.extend(self._call_anki(action="cardsInfo",
                                              cards=[card]))
            return r_list

        else:
            lock = threading.Lock()
            threads = []
            cnt = 0
            r_list = []
            target_thread_n = 5
            batchsize = max((len(card_id) // target_thread_n) + 1, 5)
            whi("(Large number of cards to retrieve: creating "
                f"{target_thread_n} threads of size {batchsize})")

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
                      file=self.t_strm,
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
            assert len(r_list) == len(card_id), "could not retrieve all cards"
            r_list = sorted(r_list,
                            key=lambda x: x["cardId"],
                            reverse=False)
            return r_list

    def _bert_tokenize(self, input_text):
        """
        This tokenizer had to be put in a specific method to remove the
        unused CLS and SEP tokens.
        """
        output = []
        for t in input_text.split("<NEWFIELD>"):
            output.extend(self.tokenizer_bert.encode(t).tokens)
        output = list(filter(lambda x: x not in ["[CLS]", "[SEP]"], output))
        return output

    def _gpt_tokenize(self, input_text):
        output = []
        for t in input_text.split("<NEWFIELD>"):
            output.extend(self.tokenizer_gpt.encode(t).tokens)
        return output

    def _deckname_check(self, deckname):
        """
        check if the deck you're calling AnnA exists or not
        if not, user is asked to enter the name, suggesting autocompletion
        """
        decklist = self._call_anki(action="deckNames") + ["*"]
        if deckname is not None:
            if deckname not in decklist:
                beep("Couldn't find this deck.")
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

    def memoize(self, f):
        """ store previous value to speed up vector retrieval
        (40x speed up) """
        memo = {}

        def helper(x):
            if x not in memo:
                memo[x] = f(x)
            return memo[x]
        return helper

    def _init_dataFrame(self):
        """
        create a pandas DataFrame with the information gathered from
        anki (via the bridge addon) such as card fields, tags, intervals, etc
        """
        if self.highjack_due_query is not None:
            beep("Highjacking due card list:")
            query = self.highjack_due_query
            red(" >  '" + query + "'")
            due_cards = self._call_anki(action="findCards", query=query)
            whi(f"Found {len(due_cards)} cards...\n")

        elif self.task in ["filter_review_cards", "bury_excess_review_cards"]:
            yel("Getting due card list...")
            query = (f"\"deck:{self.deckname}\" is:due is:review -is:learn "
                     "-is:suspended -is:buried -is:new -rated:1")
            whi(" >  '" + query + "'")
            due_cards = self._call_anki(action="findCards", query=query)
            whi(f"Found {len(due_cards)} reviews...\n")

        elif self.task == "bury_excess_learning_cards":
            yel("Getting is:learn card list...")
            query = (f"\"deck:{self.deckname}\" is:due is:learn -is:suspended "
                     "-rated:1 -rated:2:1 -rated:2:2")
            whi(" >  '" + query + "'")
            due_cards = self._call_anki(action="findCards", query=query)
            whi(f"Found {len(due_cards)} learning cards...\n")

        elif self.task == "just_add_KNN":
            yel("Getting all card list except suspended...")
            query = (f"\"deck:{self.deckname}\" -is:suspended")
            whi(" >  '" + query + "'")
            due_cards = self._call_anki(action="findCards", query=query)
            whi(f"Found {len(due_cards)} cards...\n")

        rated_cards = []
        if self.highjack_rated_query is not None:
            beep("Highjacking rated card list:")
            query = self.highjack_rated_query
            red(" >  '" + query + "'")
            rated_cards = self._call_anki(action="findCards", query=query)
            red(f"Found {len(rated_cards)} cards...\n")
        elif self.rated_last_X_days not in [0, None]:
            yel("Getting cards that where rated in the last "
                f"{self.rated_last_X_days} days...")
            query = (f"\"deck:{self.deckname}\" rated:{self.rated_last_X_days}"
                     " -is:suspended -is:buried")
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
                yel("Removed overlap between rated cards and due cards: "
                    f"{diff} cards removed. Keeping {len(temp)} cards.\n")
                rated_cards = temp
        self.due_cards = due_cards
        self.rated_cards = rated_cards

        if len(self.due_cards) < self.minimum_due:
            beep(f"Number of due cards is {len(self.due_cards)} which is "
                 f"less than threshold ({self.minimum_due}).\nStopping.")
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
            tags = []
            for t in list_cardInfo[i]["tags"]:
                skip_t = False
                for tags_to_ignore in self.tags_to_ignore:
                    if tags_to_ignore.match(t):
                        skip_t = True
                        break
                if not skip_t:
                    tags.append(t)
            list_cardInfo[i]["tags"] = " ".join(tags)
            if card["cardId"] in due_cards:
                list_cardInfo[i]["status"] = "due"
            elif card["cardId"] in rated_cards:
                list_cardInfo[i]["status"] = "rated"
            else:
                list_cardInfo[i]["status"] = "ERROR"
                beep(f"Error processing card with ID {card['cardId']}")

        if len(list_cardInfo) != len(list(set(combined_card_list))):
            beep("Error: duplicate cards in DataFrame!\nExiting.")
            pdb.set_trace()

        self.df = pd.DataFrame().append(list_cardInfo,
                                        ignore_index=True,
                                        sort=True)
        self.df["cardId"] = self.df["cardId"].astype(int)
        self.df = self.df.set_index("cardId").sort_index()
        self.df["interval"] = self.df["interval"].astype(float)
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
        out = string.group(0) + f" ({new_w})"
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
        text = text.replace(
                "&amp;", "&"
                ).replace(
                "+++", " important "
                ).replace(
                "&nbsp", " "
                ).replace(
                "\u001F", " "
                )

        # remove email adress:
        text = re.sub(r'\S+@\S+\.\S{2,3}', " ", text)

        # remove weird clozes
        text = re.sub(r"}}{{c\d+::", "", text)

        # remove sound recordings
        text = re.sub(r"\[sound:.*?\..*?\]", " ", text)

        # duplicate bold and underlined content, as well as clozes
        text = re.sub(r"\b<u>(.*?)</u>\b", r" \1 \1 ", text,
                      flags=re.M | re.DOTALL)
        text = re.sub(r"\b<b>(.*?)</b>\b", r" \1 \1 ", text,
                      flags=re.M | re.DOTALL)
        text = re.sub(r"{{c\d+::.*?}}", lambda x: f"{x.group(0)} {x.group(0)}",
                      text, flags=re.M | re.DOTALL)

        # if blockquote or li or ul, mention that it's a list item
        # usually indicating a harder card
        # the ' <br>}}{{' is something specific to AnnA's author
        if re.match(r"</?li/?>|</?ul/?>| <br>}}{{", text, flags=re.M):
            text += " list list"

        # remove html spaces
        text = re.sub(
            '\\n|</?div/?>|</?br/?>|</?span/?>|</?li/?>|</?ul/?>', " ", text)
        text = re.sub('</?blockquote(.*?)>', " ", text)

        # OCR
        if self.keep_OCR:
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
        text = re.sub("</?su[bp]>", "", text)  # exponant or indices
        text = re.sub(r"\[\d*\]", "", text)  # wiki style citation

        text = re.sub("<.*?>", "", text)  # remaining html tags
        text = text.replace("&gt", "").replace("&lt", "").replace(
            "<", "").replace(">", "").replace("'",
                                              " ")  # misc + french apostrophe

        # replace greek letter
        if self.replace_greek:
            for a, b in greek_alphabet_mapping.items():
                text = re.sub(a, b, text)

        # replace common french accents
        text = text.replace(
                "é", "e"
                ).replace(
                "è", "e"
                ).replace(
                "ê", "e"
                ).replace(
                "à", "a"
                ).replace(
                "ç", "c"
                ).replace(
                "ï", "i"
                ).replace(
                "ù", "u"
                ).replace(
                "œ", "oe")

        # replace acronyms
        already_replaced = []
        if self.acronym_file is not None:
            for regex, new_value in self.acronym_dict.items():
                if re.match(regex, text):
                    if regex not in already_replaced:
                        # only replace once but still apply the overlapping
                        # acronyms if needed
                        text = re.sub(regex,
                                      lambda in_string:
                                      self._regexp_acronym_replacer(in_string,
                                                                    regex,
                                                                    new_value),
                                      text)

                    # if overlapping patterns, apply sequentially
                    if regex.pattern in self.acronyms_overlapping:
                        for regex2 in self.acronyms_overlapping[regex.pattern]:
                            new_value2 = self.acronym_dict[regex2]
                            text = re.sub(regex2,
                                          lambda in_string:
                                          self._regexp_acronym_replacer(
                                              in_string, regex2, new_value2),
                                          text)
                            already_replaced.append(regex2)

        # misc
        text = " ".join(text.split())  # multiple spaces

        # optionnal stemmer
        if self.vectorizer == "TFIDF":
            if self.TFIDF_stem is True:
                text = " ".join([self.ps.stem(x) for x in text.split()])

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
        def _threaded_field_filter(index_list, lock, pbar,
                                   stopw_compiled, spacers_compiled):
            """
            threaded call to speed up execution
            """
            for index in index_list:
                card_model = self.df.loc[index, "modelName"]
                target_model = []
                fields_to_keep = []

                # determines which is the corresponding model described
                # in field_dic
                field_dic = self.field_dic
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
                        target_model = sorted(
                            target_model, key=lambda x: lev.ratio(
                                x.lower(), user_model.lower()))
                        fields_to_keep = field_dic[target_model[0]]
                        with lock:
                            to_notify.append(
                                f"Several notetypes match  '{card_model}'"
                                f". Selecting '{target_model[0]}'")

                # concatenates the corresponding fields into one string:
                field_list = list(self.df.loc[index, "fields"])
                if fields_to_keep == "take_first_fields":
                    fields_to_keep = ["", ""]
                    for f in field_list:
                        order = self.df.loc[index,
                                            "fields"][f.lower()]["order"]
                        if order == 0:
                            fields_to_keep[0] = f
                        elif order == 1:
                            fields_to_keep[1] = f
                    with lock:
                        to_notify.append(
                            f"No matching notetype found for {card_model}. "
                            "Keeping the first 2 fields: "
                            f"{', '.join(fields_to_keep)}")
                elif fields_to_keep == "take_all_fields":
                    fields_to_keep = sorted(
                        field_list, key=lambda x: int(
                            self.df.loc[index, "fields"][x.lower()]["order"]))
                    if "Nearest_neighbours" in field_list:
                        field_list.remove("Nearest_neighbours")

                comb_text = ""
                for f in fields_to_keep:
                    try:
                        next_field = re.sub(
                            self.stopw_compiled,
                            " ",
                            self.df.loc[index,
                                        "fields"][f.lower()]["value"].strip())
                        if next_field != "":
                            comb_text = comb_text + next_field + " <NEWFIELD> "
                    except KeyError as e:
                        with lock:
                            to_notify.append(
                                f"Error when looking for field {e} in card "
                                f"{self.df.loc[index, 'modelName']} "
                                "identified as "
                                f"notetype {target_model}")
                comb_text = comb_text.strip()
                if comb_text.endswith("<NEWFIELD>"):
                    comb_text = comb_text[:-10].strip()

                # add tags to comb_text
                if self.append_tags:
                    tags = self.df.loc[index, "tags"].split(" ")
                    for t in tags:
                        skip_tag = False
                        for tags_to_ignore in self.tags_to_ignore:
                            if tags_to_ignore.match(t):
                                skip_tag = True
                                break
                        if skip_tag:
                            continue
                        # replaces _ - and / by a space and keep only
                        # the last 2 levels of each tags:
                        t = re.sub(
                            spacers_compiled,
                            " ",
                            " ".join(t.split(self.tags_separator)[-2:]))
                        comb_text += " " + t

                with lock:
                    self.df.at[index, "comb_text"] = comb_text
                    pbar.update(1)
            return None

        n = len(self.df.index)
        batchsize = n // 4 + 1
        lock = threading.Lock()

        threads = []
        to_notify = []
        spacers_compiled = re.compile("_|-|/")

        # initialize the column to avoid race conditions
        self.df["comb_text"] = np.nan
        self.df["comb_text"] = self.df["comb_text"].astype(str)

        with tqdm(total=n,
                  desc="Combining relevant fields",
                  smoothing=0,
                  file=self.t_strm,
                  unit=" card") as pbar:
            for nb in range(0, n, batchsize):
                sub_card_list = self.df.index[nb: nb + batchsize]
                thread = threading.Thread(target=_threaded_field_filter,
                                          args=(sub_card_list,
                                                lock,
                                                pbar,
                                                self.stopw_compiled,
                                                spacers_compiled),
                                          daemon=False)
                thread.start()
                threads.append(thread)

            [t.join() for t in threads]

            # retries in case of error:
            cnt = 0
            while sum(self.df.isna()["comb_text"]) != 0:
                cnt += 1
                na_list = [x
                           for x in self.df.index[
                               self.df.isna()["comb_text"]].tolist()]
                pbar.update(-len(na_list))
                red(
                    f"Found {sum(self.df.isna()['comb_text'])} null values "
                    "in comb_text: retrying")
                thread = threading.Thread(target=_threaded_field_filter,
                                          args=(na_list,
                                                lock,
                                                pbar,
                                                self.stopw_compiled,
                                                spacers_compiled),
                                          daemon=False)
                thread.start()
                thread.join()
                if cnt > 10:
                    beep(f"Error: restart anki then"
                         "rerun AnnA.")
                    raise SystemExit()
            if cnt > 0:
                yel(f"Succesfully corrected null combined texts on #{cnt} "
                    "trial.")

        to_notify = list(set(to_notify))
        for notif in to_notify:
            beep(notif)

        # using multithreading is not faster, using multiprocess is probably
        # slower if not done by large batching
        tqdm.pandas(desc="Formating text", smoothing=0, unit=" card",
                    file=self.t_strm)
        self.df["text"] = self.df["comb_text"].progress_apply(
            lambda x: " <NEWFIELD> ".join(
                [
                    self._text_formatter(y) for y in x.split("<NEWFIELD>")
                    ]
                ).strip()
            )
        del self.df["comb_text"]

        # find short cards
        ind_short = []
        for ind in self.df.index:
            if len(self.df.loc[ind, "text"]) < 10:
                ind_short.append(ind)
        if ind_short:
            yel(f"\n{len(ind_short)} cards contain less than 10 characters "
                f"after formatting: {','.join([str(x) for x in ind_short])}")
            if self.append_tags is False:
                red("Appending tags to those cards despite setting "
                    "`append_tags` to False.")
                for ind in ind_short:
                    tags = self.df.loc[ind, "tags"].split(" ")
                    for t in tags:
                        if ("AnnA" not in t) and (
                                t not in self.tags_to_ignore):
                            t = re.sub(
                                spacers_compiled,
                                " ",
                                " ".join(t.split(self.tags_separator)[-2:]))
                            self.df.loc[ind, "text"] += " " + t

        yel("\n\nPrinting 2 random samples of your formated text, to help "
            " adjust formating issues:")
        pd.set_option('display.max_colwidth', 8000)
        max_length = 10000
        sub_index = random.sample(self.df.index.tolist(), k=2)
        for i in sub_index:
            if len(self.df.loc[i, "text"]) > max_length:
                ending = "...\n"
            else:
                ending = "\n"
            whi(f" * {i} : {str(self.df.loc[i, 'text'])[0:max_length]}",
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

        if not self.low_power_mode:
            ngram_val = (1, 5)
        else:
            ngram_val = (1, 3)

        def init_vectorizer():
            """used to make sure the same statement is used to create
            the vectorizer"""
            return TfidfVectorizer(strip_accents="ascii",
                                   lowercase=False,
                                   tokenizer=self.tokenize,
                                   token_pattern=None,
                                   stop_words=None,
                                   ngram_range=ngram_val,
                                   norm="l2",
                                   sublinear_tf=True,
                                   max_features=min(10_000,
                                                    len(df.index) // 2),
                                   # max_df=0.99,  # ignore words present in
                                   # more than X% of documents
                                   # min_df=2,  # ignore words than appear
                                   # # less than n times
                                   )
        use_fallback = False
        if self.whole_deck_computation:
            try:
                yel("\nCopying anki database to local cache file")
                original_db = akp.find_db(user=self.profile_name)
                if self.profile_name is None:
                    red("Ankipandas will use anki collection found at "
                        f"{original_db}")
                else:
                    yel("Ankipandas will use anki collection found at "
                        f"{original_db}")
                if "trash" in str(original_db).lower():
                    beep(f"Ankipandas seems to have "
                         "found a collection in "
                         "the trash folder. If that is not your intention "
                         "cancel now. Waiting 10s for you to see this "
                         "message before proceeding.")
                    time.sleep(1)
                Path.mkdir(Path("cache"), exist_ok=True)
                name = f"{self.profile_name}_{self.deckname}".replace(" ", "_")
                temp_db = shutil.copy(
                    original_db, f"./cache/{name.replace('/', '_')}")
                col = akp.Collection(path=temp_db)

                # keep only unsuspended cards from the right deck
                cards = col.cards.merge_notes()
                cards["cdeck"] = cards["cdeck"].apply(
                    lambda x: x.replace("\x1f", "::"))
                cards = cards[cards["cdeck"].str.startswith(self.deckname)]
                cards = cards[cards["cqueue"] != "suspended"]
                whi("Ankipandas db loaded successfuly.")

                if len(cards.index) == 0:
                    beep(f"Ankipandas database"
                         "is of length 0")
                    raise Exception("Ankipandas database is of length 0")

                # get only the right fields
                cards["mid"] = col.cards.mid.loc[cards.index]
                mid2fields = akp.raw.get_mid2fields(col.db)
                mod2mid = akp.raw.get_model2mid(col.db)

                if len(cards.index) == 0:
                    beep(f"Ankipandas database"
                         "is of length 0")
                    raise Exception("Ankipandas database is of length 0")

                to_notify = []

                def get_index_of_fields(mod):
                    ret = []
                    if mod in mod2mid:
                        fields = mid2fields[mod2mid[mod]]
                        if mod in self.field_dic:
                            for f in self.field_dic[mod]:
                                ret.append(fields.index(f))
                        else:
                            to_notify.append(
                                "Missing field mapping for card model "
                                f"{mod}.Taking first 2 fields.")
                            ret = [0, 1]
                    else:
                        whi(mod)
                        best_models = sorted(
                            list(mod2mid.keys()),
                            key=lambda x: lev.ratio(x.lower(), mod.lower()))
                        for m in best_models:
                            if m in self.field_dic:
                                ret = get_index_of_fields(m)
                                break
                    assert len(ret) != 0
                    return ret

                m_gIoF = self.memoize(get_index_of_fields)

                for notif in list(set(to_notify)):
                    red(notif)

                corpus = []
                spacers_compiled = re.compile("_|-|/")
                for ind in tqdm(cards.index,
                                desc=("Gathering and formating "
                                      f"{self.deckname}"),
                                file=self.t_strm):
                    indices_to_keep = m_gIoF(cards.loc[ind, "nmodel"])
                    fields_list = cards.loc[ind, "nflds"]
                    new = ""
                    for i in indices_to_keep:
                        new += fields_list[i] + " "
                    processed = self._text_formatter(re.sub(
                        self.stopw_compiled, " ", new))
                    if len(processed) < 10 or self.append_tags:
                        tags = cards.loc[ind, "ntags"]
                        for t in tags:
                            if ("AnnA" not in t) and (
                                    t not in self.tags_to_ignore):
                                t = re.sub(spacers_compiled, " ", " ".join(
                                    t.split(self.tags_separator)[-2:]))
                                processed += " " + t
                    corpus.append(processed)

                vectorizer = init_vectorizer()
                vectorizer.fit(tqdm(corpus + df["text"].tolist(),
                                    desc="Vectorizing whole deck"),
                                    file=self.t_strm)
                t_vec = vectorizer.transform(tqdm(df["text"],
                                                  desc=(
                    "Vectorizing dues cards using TFIDF"),
                                                  file=self.t_strm))
                yel("Done vectorizing over whole deck!")
            except Exception as e:
                beep(f"Exception : {e}\nUsing "
                     "fallback method...")
                use_fallback = True

        if (self.whole_deck_computation is False) or (use_fallback):
            vectorizer = init_vectorizer()
            t_vec = vectorizer.fit_transform(tqdm(df["text"],
                                                  desc=(
              "Vectorizing using TFIDF"),
                                                  file=self.t_strm))
        if self.TFIDF_dim is None:
            df["VEC"] = [x for x in t_vec]
        else:
            # explanation : trying to do a dimensions reduction on the vectors
            # but trying up to 10 times to find the right value that keeps
            # between 75% and 85% of variance. Under that the information
            # starts to get lost and over that cards can tend to
            # all be equidistant (I think).
            trial = 0
            desired_variance_kept = 50
            red("Iteratively computing dimension reduction until "
                f"{desired_variance_kept}% of variance is kept.")

            # start either from the user supplied value or from the highest
            # possible number up to 50 or from cache

            cache_dir = Path("utils")
            cache_dir.mkdir(exist_ok=True)
            cache_file = cache_dir / "latest_TFIDF_dim.cache.json"
            dim_limit = len(self.df.index) // 50  # maximum 1 dimension per 50 cards
            assert dim_limit <= (t_vec.shape[1] - 1), "Invalid number of dimension!"
            cache = {}

            if self.TFIDF_dim == "auto":
                # trying from cache
                try:
                    assert cache_file.exists(), "No 'latest_TFIDF_dim.cache.json' file, creating it."
                    cache = json.load(cache_file.open("r"))
                    assert isinstance(cache, dict), "cache is not a dict"
                    if f"{self.deckname}_{self.task}_{desired_variance_kept}" in cache:
                        self.TFIDF_dim = int(cache[f"{self.deckname}_{self.task}_{desired_variance_kept}"])
                        self.TFIDF_dim = min(self.TFIDF_dim, dim_limit)
                        whi(f"(Loaded dimension '{self.TFIDF_dim}' from cache file)")
                    else:
                        self.TFIDF_dim = dim_limit
                        yel("Cache does not contain the key to this deck, creating it.")

                except Exception as err:
                    beep(f"Exception when loading latest dimension cache: '{err}'")
                    self.TFIDF_dim = dim_limit

            already_tried = []
            while True:
                self.TFIDF_dim = min(self.TFIDF_dim, dim_limit)
                yel(f"\nReducing dimensions to {self.TFIDF_dim} using SVD...",
                    end=" ")
                svd = TruncatedSVD(n_components=self.TFIDF_dim)
                t_red = svd.fit_transform(t_vec)
                evr = round(sum(svd.explained_variance_ratio_) * 100, 1)
                trial += 1
                already_tried.append(self.TFIDF_dim)

                if abs(evr - desired_variance_kept) <= 5:
                    # success
                    break

                elif trial >= 10:
                    # failure, should not really happen
                    beep(f"Tried {trial} times to find the right number of "
                        "dimensions, carrying on.")
                    break

                else:
                    # choosing next value to try
                    offset = desired_variance_kept - evr
                    # multiply or divide by 2 every 20% of difference
                    self.TFIDF_dim *= 2**(offset/20)
                    self.TFIDF_dim = int(max(3, min(self.TFIDF_dim, dim_limit)))
                    yel(f"Explained variance ratio is only {evr}% ("
                        "retrying up to 10 times to get closer to "
                        f"{desired_variance_kept}%)", end=" ")

                    if self.TFIDF_dim in already_tried:
                        yel(f"Already tried dimension {self.TFIDF_dim}, skipping.")
                        break

                    continue

            yel(f"Explained variance ratio after SVD with {self.TFIDF_dim} "
                f"dims on Tf_idf: {evr}%")

            # store dim value into a cache file
            try:
                cache[f"{self.deckname}_{self.task}_{desired_variance_kept}"] = int(self.TFIDF_dim)
                json.dump(cache, cache_file.open("w"))
                whi("(Saved dimension to cache file.)")
            except Exception as err:
                beep(f"Exception when storing latest dimension to cache: '{err}'")

            df["VEC"] = [x for x in t_red]

        if self.plot_2D_embeddings:
            try:
                yel("Computing 2D embeddings for the plot...")

                #TODO: test with UMAP
                try:
                    whi("Trying with UMAP first.")
                    import umap.umap_
                    umap_kwargs = {"n_jobs": -1,
                                   "verbose": 1,
                                   "n_components": 2,
                                   "metric": "cosine",
                                   "init": 'spectral',  # TODO: try, 'pca'
                                   "random_state": 42,
                                   "transform_seed": 42,
                                   "n_neighbors": 20,
                                   "min_dist": 0.01,
                                   "low_memory":  False,
                                   "densmap": False,
                                   }
                    U = umap.umap_.UMAP(**umap_kwargs)
                    t_embed = U.fit_transform(t_vec)

                except Exception as err:
                    beep(f"Exception with UMAP: '{err}'")
                    red(traceback.format_exc())

                    yel("Using SVD...")
                    svd = TruncatedSVD(n_components=2)
                    t_embed = svd.fit_transform(t_vec)

                df["2D_embeddings"] = [x for x in t_embed]
            except Exception as err:
                beep(f"Error when computing 2D embeddings: '{err}'")
                red(traceback.format_exc())

        self.df = df
        return True

    def _compute_distance_matrix(self, input_col="VEC"):
        """
        compute distance matrix : a huge matrix containing the
            cosine distance between the vectors of each cards.
        * scikit learn allows a parallelized computation
        """
        df = self.df

        yel("\nComputing distance matrix on all available cores"
            "...")
        if self.dist_metric == "rbf":
            sig = np.mean(np.std([x for x in df[input_col]], axis=1))
            self.df_dist = pd.DataFrame(columns=df.index,
                                        index=df.index,
                                        data=pairwise_kernels(
                                            [x for x in df[input_col]],
                                            n_jobs=-1,
                                            metric="rbf",
                                            gamma=1/(2*sig),
                                            ))
            # turn the similarity into a distance
            # apply log to hopefully reduce the spread
            tqdm.pandas(desc="Applying log", smoothing=0, unit=" card",
                        file=self.t_strm)
            self.df_dist = self.df_dist.progress_apply(lambda x: np.log(1+x))
            max_val = np.amax(self.df_dist)
            self.df_dist /= -max_val  # normalize values then make negative
            # add 1 to get: "dist = 1 - similarity"
            self.df_dist += 1
        elif self.dist_metric == "cosine":
            self.df_dist = pd.DataFrame(columns=df.index,
                                        index=df.index,
                                        data=pairwise_distances(
                                            [x for x in df[input_col]],
                                            n_jobs=-1,
                                            metric="cosine",
                                            ))
        else:
            raise ValueError("Invalid 'dist_metric' value")

        self.df_dist = self.df_dist.sort_index()

        yel("Minmax scaling the distance matrix...")

        # computing and displaying stats before rescaling
        whi("Computing mean and std of distance matrix before minmax..."
            "\n(excluding diagonal)")
        up_triangular = np.triu_indices(self.df_dist.shape[0], 1)
        mean_dist = round(np.nanmean(self.df_dist.values[up_triangular]), 2)
        std_dist = round(np.nanstd(self.df_dist.values[up_triangular]), 2)
        yel(f"Mean distance: {mean_dist}, std: {std_dist}\n")

        # get all values
        values = self.df_dist.values[up_triangular]
        # get minimum that is not 0
        above_zero = values[values > 0].ravel().min()
        # move everything closer to 0
        values -= above_zero
        original_zero_mask = (values < 0)  # store for later
        # max scaling
        values /= values.max()
        # move everything a bit farther than 0
        values += 0.1
        # make sure the original 0 are restored (they are identical cards)
        values[original_zero_mask] = 0
        # rescale once more
        values /= values.max()
        whi("Storing values...")
        self.df_dist.values[up_triangular] = values
        lo_triangular = np.tril_indices(self.df_dist.shape[0], -1)
        self.df_dist.values[lo_triangular] = self.df_dist.values.T[lo_triangular]
        assert ((self.df_dist.values - self.df_dist.values.T) == 0).all(), (
                "Non symetric distance matrix")
        whi("Finished minmax scaling.")

        # make sure the distances are positive otherwise it might reverse
        # the sorting logic for the negative values (i.e. favoring similar
        # cards)
        assert (self.df_dist.values.ravel() < 0).sum() == 0, (
            "Negative values in the distance matrix!")

        yel("Computing mean and std of distance...\n(excluding diagonal)")
        # ignore the diagonal of the distance matrix to get a sensible mean
        # and std value:
        up_triangular = np.triu_indices(self.df_dist.shape[0], 1)
        mean_dist = round(np.nanmean(self.df_dist.values[up_triangular]), 2)
        std_dist = round(np.nanstd(self.df_dist.values[up_triangular]), 2)
        yel(f"Mean distance: {mean_dist}, std: {std_dist}\n")

        self.mean_dist = mean_dist
        self.std_dist = std_dist

        # store mean distance for the fuzz factor
        if self.enable_fuzz:
            self.mean_dist = mean_dist

        self._print_similar()
        return True

    def _compute_KNN(self):
        """
        Compute the nearest neighbours of each note of the distance matrix.
        This is used to then fill the field "Nearest_neighbours" of those notes
        so that you can quickly know the neighbour of any given card straight
        into anki without having to use AnnA.
        """
        if not ((self.add_KNN_to_field or self.plot_2D_embeddings) or (
                self.task == "just_add_KNN")):
            whi("Computing KNN matrix is not needed by those arguments.")
            return
        try:
            radius = 1 / 2 * self.mean_dist
            yel(f"Finding neighbors within {radius}...")
            self.knn = radius_neighbors_graph(
                    X=self.df_dist,
                    radius=radius,
                    #n_neighbors=n_n,
                    mode="connectivity",
                    n_jobs=-1,
                    metric="precomputed",
                    include_self=True)
        except Exception as err:
            beep(f"Error when computing KNN: '{err}'")

    def _do_add_KNN_to_note(self):
        """
        if the model card contains the field 'Nearest_neighbours', replace its
        content by a query that can be used to find the neighbour of the
        given note.
        """
        if not (self.add_KNN_to_field or self.task == "just_add_KNN"):
            whi("Not adding KNN to note field because of arguments.")
            return
        red("Adding the list of neighbours to each note.")
        nid_content = {}
        nb_of_nn = []
        try:
            for i in tqdm(
                    range(self.knn.shape[0]),
                    desc="Colecting neighbours of notes",
                    file=self.t_strm,
                    unit="card"):
                cardId = self.df.index[i]
                if "Nearest_neighbours".lower() not in self.df.loc[
                        cardId, "fields"].keys():
                    continue
                noteId = int(self.df.loc[cardId, "note"])
                if noteId in nid_content:
                    continue  # skipped because a card of this note was
                    # already processed

                knn_ar = self.knn.getcol(i).toarray().squeeze()
                neighbour_indices = list(np.where(knn_ar == 1)[0])
                nb_of_nn.append(len(neighbour_indices))
                neighbour_indices = sorted(
                        neighbour_indices,
                        key=lambda x: float(self.df_dist.loc[
                            cardId, self.df.index[x]]),
                        reverse=True)
                neighbours_nid = [str(self.df.loc[self.df.index[ind], "note"])
                                  for ind in neighbour_indices]
                nid_content[noteId] = "nid:" + ",".join(neighbours_nid)
            whi("Sending new field value to Anki...")
            self._call_anki(
                    action="update_KNN_field",
                    nid_content=nid_content,
                    )
            yel("Finished adding neighbours to notes.")
        except Exception:
            beep("Error when adding neighbour list to notes!")

        if nb_of_nn:
            yel("Number of neighbors on average:")
            whi(str(pd.DataFrame(nb_of_nn).describe()))

    def _print_similar(self):
        """ finds two cards deemed very similar (but not equal) and print
        them. This is used to make sure that the system is working correctly.
        Given that this takes time, a timeout has been implemented.
        """
        def time_watcher(signum, frame):
            "used to issue a timeout"
            raise TimeoutError("Timed out. Not showing most similar cards")
        signal.signal(signal.SIGALRM, time_watcher)
        signal.alarm(60)

        try:
            max_length = 200
            up_triangular = np.triu_indices(self.df_dist.shape[0], 1)
            pd.set_option('display.max_colwidth', 180)

            red("\nPrinting the most semantically different cards:")
            highest_value = np.amax(self.df_dist.values[up_triangular])
            coord_max = np.where(self.df_dist == highest_value)
            one = self.df.loc[self.df.index[coord_max[0][0]]].text[:max_length]
            two = self.df.loc[self.df.index[coord_max[1][0]]].text[:max_length]
            yel(f"* {one}...")
            yel(f"* {two}...")

            red("\nPrinting the most semantically but distinct similar "
                "cards:")
            # the diagonal is the minimum of distance so we are looking for
            # the distance that is just higher
            q_diagonal = (self.df_dist.shape[0] + 1) / (
                    self.df_dist.shape[0] ** 2 / 2)
            quantile_limit = np.quantile(
                    self.df_dist.values[up_triangular].ravel(), q_diagonal)
            lowest_non_zero_value = np.amin(
                    self.df_dist.values[up_triangular],
                    where=self.df_dist.values[up_triangular] > quantile_limit,
                    initial=highest_value)
            coord_min = np.where(self.df_dist == lowest_non_zero_value)
            one = self.df.loc[self.df.index[coord_min[0][0]]].text[:max_length]
            two = self.df.loc[self.df.index[coord_min[1][0]]].text[:max_length]
            yel(f"* {one}...")
            yel(f"* {two}...")
            whi(f"    (distance: {lowest_non_zero_value})")

            red("\nPrinting the median distance cards:")
            median_value = np.median(self.df_dist.values[up_triangular].ravel(
                ))
            coord_med = [[]]
            i = 1
            while len(coord_med[0]) == 0:
                if i >= 1e08:
                    break
                coord_med = np.where(np.isclose(
                    self.df_dist, median_value, atol=1e-08*i))
                i *= 1e1
            one = self.df.loc[self.df.index[coord_med[0][0]]].text[:max_length]
            two = self.df.loc[self.df.index[coord_med[1][0]]].text[:max_length]
            yel(f"* {one}...")
            yel(f"* {two}...")
        except TimeoutError:
            beep(f"Taking too long to locating similar "
                 "nonequal cards, skipping")
        except Exception as err:
            beep(f"Exception when locating similar "
                 f"cards: '{err}'")
        finally:
            signal.alarm(0)
            pd.reset_option('display.max_colwidth')
            whi("")

    def _compute_opti_rev_order(self):
        """
        1. calculates the 'ref' column. The lowest the 'ref', the more urgent
            the card needs to be reviewed. The computation used depends on
            argument 'reference_order', hence picking a card according to its
            'ref' only can be the same as using a regular filtered deck with
            'reference_order' set to 'relative_overdueness' for example.
            Some of the ref columns are minmax scaled or processed.
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
            The score is computed according to a formula like in this example:
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
        max_deck_size = self.max_deck_size
        rated = self.rated_cards
        due = self.due_cards
        w1 = self.score_adjustment_factor[0]
        w2 = self.score_adjustment_factor[1]
        if self.enable_fuzz:
            w3 = (w1 + w2) / 2 * self.mean_dist / 10
        else:
            w3 = 0
        self.urgent_dues = []

        # hardcoded settings
        display_stats = True

        # setting interval to correct value for learning and relearnings:
        steps_L = [x / 1440 for x in self.deck_config["new"]["delays"]]
        steps_RL = [x / 1440 for x in self.deck_config["lapse"]["delays"]]
        for i in df.index:
            if df.loc[i, "type"] == 1:  # learning
                df.at[i, "interval"] = steps_L[int(
                    str(df.loc[i, "left"])[-3:])-1]
                assert df.at[i,
                             "interval"] >= 0, (
                                     f"negative interval for card {i}")
            elif df.loc[i, "type"] == 3:  # relearning
                df.at[i, "interval"] = steps_RL[int(
                    str(df.loc[i, "left"])[-3:])-1]
                assert df.at[i,
                             "interval"] >= 0, (
                                     f"negative interval for card {i}")
            if df.loc[i, "interval"] < 0:  # negative values are in seconds
                yel(f"Changing interval: cid: {i}, ivl: "
                    f"{df.loc[i, 'interval']} => "
                    f"{df.loc[i, 'interval']/(-86400)}")
                df.at[i, "interval"] /= -86400

        # setting rated cards value to nan value, to avoid them
        # skewing the dataset distribution:
        df.loc[rated, "interval"] = np.nan
        df.loc[rated, "due"] = np.nan
        df["ref"] = np.nan

        # computing reference order:
        if reference_order in ["lowest_interval", "LIRO_mix"]:
            ivl = df.loc[due, 'interval'].to_numpy().reshape(-1, 1).squeeze()
            # make start at 0
            assert (ivl > 0).all(), "Negative intervals"
            ivl -= ivl.min()
            if ivl.max() != 0:
                ivl /= ivl.max()
            if not reference_order == "LIRO_mix":
                df.loc[due, "ref"] = ivl

        elif reference_order == "order_added":
            df.loc[due, "ref"] = due
            assert (due > 0).all(), "Negative due values"
            df.loc[due, "ref"] -= df.loc[due, "ref"].min()
            df.loc[due, "ref"] /= df.loc[due, "ref"].max()

        if reference_order in ["relative_overdueness", "LIRO_mix"]:
            yel("Computing relative overdueness...")

            # the code for relative overdueness is not exactly the same as
            # in anki, as I was not able to fully replicate it.
            # Here's a link to one of the original implementation :
            # https://github.com/ankitects/anki/blob/afff4fc437f523a742f617c6c4ad973a4d477c15/rslib/src/storage/card/filtered.rs

            # first, get the offset for due cards values that are timestamp
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
                assert df.at[i,
                             "ref_due"] > 0, f"negative interval for card {i}"
            df.loc[due, "overdue"] = df.loc[due, "ref_due"] - time_offset
            overdue = df.loc[due, "overdue"]

            # then, correct overdue values to make sure they are negative
            correction = max(overdue.max(), 0) + 0.01
            if correction > 1:
                beep("This should probably never happen.")
                breakpoint()

            # my implementation of relative overdueness:
            # (intervals are positive, overdue are negative for due cards
            # hence ro is positive)
            # low ro means urgent, high ro means not urgent
            assert (df.loc[due, "interval"].values > 0).all(), "Negative interval"
            assert correction > 0, "Negative correction factor"
            assert (-overdue + correction > 0).all(), "Positive overdue - correction"
            ro = (df.loc[due, "interval"].values + correction) / (-overdue + correction)
            assert (ro >= 0).all(), "wrong values of relative overdueness"
            assert ro.max() < np.inf, "Infinity is part of relative overdueness"

            # clipping extreme values, above 1 is useless anyway
            #ro = np.clip(ro, 0, 10)

            # boost cards according to how overdue they are
            boost = True if "boost" in self.repick_task else False

            # gather list of urgent dues

            p = 0.15  # all cards more than (100*p)% overdue are deemed urgent
            mask = np.argwhere(ro.values <= 1/p).squeeze()
            mask2 = np.argwhere(np.abs(overdue.values) > 1).squeeze()
            # if only one found in mask, make sure it's an iterable
            if isinstance(mask, int):
                mask = [mask]
            if isinstance(mask2, int):
                mask = [mask2]
            temp1 = [due[i] for i in np.intersect1d(mask, mask2).tolist()]
            whi(f"* Found '{len(temp1)}' cards that are more than '{int(p*100)}%' overdue.")

            ivl_limit = 5  # all cards with interval <= ivl_limit are deemed urgent
            temp2 = [ind for ind in due if df.loc[ind, "interval"] <= ivl_limit]
            whi(f"* Found '{len(temp2)}' cards that are due with 'interval <= {ivl_limit} days'.")

            ease_limit = 1750  # all cards with lower ease are deemed urgent
            temp3 = [ind for ind in due if df.loc[ind, "factor"] <= ease_limit]
            whi(f"* Found '{len(temp3)}' cards that are due with 'ease <= {ease_limit//10}%'.")

            urgent_dues = temp1 + temp2 + temp3
            urgent_dues = list(set(urgent_dues))
            whi(f"=> In total, found {len(urgent_dues)} cards to boost.")


            # reduce the increase of ro as a very high ro is not important
            while ro.max() > 1.5:
                whi("(Smoothing relative overdueness)")
                ro[ro > 1] = 1 + np.log(ro[ro > 1])

            # minmax scaling of ro
            ro -= ro.min()
            ro /= ro.max()
            ro += 0.001

            if boost:
                whi("Boosted urgent_dues cards to increase chances they are reviewed today.")
                for ind in urgent_dues:
                    ro[ind] -= 1
                assert ro.min() < 0:
                ro += abs(ro.min()) + 0.001
                assert ro.min() > 0, "Negative value in relative overdueness"
                ro /= ro.max()

            # add tag to urgent dues
            if urgent_dues:
                beep(f"{len(urgent_dues)}/{len(due)} cards "
                     "with too low "
                     "relative overdueness (i.e. on the brink of being "
                     "forgotten) where found.")
                if boost:
                    red("Those cards were boosted to make sure you review them"
                        " soon.")
                else:
                    red("Those cards were NOT boosted.")

                if "addtag" in self.repick_task:
                    d = datetime.today()
                    # time format is day/month/year
                    today_date = f"{d.day:02d}/{d.month:02d}/{d.year}"
                    notes = []
                    for card in urgent_dues:
                        notes.append(int(self.df.loc[card, "note"]))
                    notes = list(set(notes))  # remove duplicates
                    new_tag = f"AnnA::urgent_reviews::{today_date}"
                    try:
                        self._call_anki(action="addTags",
                                        notes=notes, tags=new_tag)
                        red("Appended tags 'urgent_reviews' to cards with "
                            "very low relative overdueness.")
                    except Exception as e:
                        beep(f"Error adding tags to urgent "
                             f"cards: {str(e)}")

            assert sum(ro < 0) == 0, "Negative values in relative overdueness"

            self.urgent_dues = urgent_dues  # store to use when resorting the cards

            if not reference_order == "LIRO_mix":
                df.loc[due, "ref"] = ro

        # weighted mean of lowest interval and relative overdueness
        if reference_order == "LIRO_mix":
            assert 0 not in list(
                np.isnan(df["ref"].values)), "missing ref value for some cards"
            weights = [1, 4]
            df.loc[due, "ref"] = (weights[0] * ro + weights[1] * ivl
                                  ) / sum(weights)

        assert len([x for x in rated if df.loc[x, "status"] != "rated"]
                   ) == 0, "all rated cards are not marked as rated"
        if self.rated_last_X_days is not None:
            red("\nCards identified as rated in the past "
                f"{self.rated_last_X_days} days: {len(rated)}")

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
        if (target_deck_size not in ["all", "100%"]):
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
                red(f"Removed {previous_len-len(indTODO)} siblings cards "
                    f"out of {previous_len}.")
            assert len(indTODO) >= 0, "wrong length of indTODO"
        else:
            yel("Not excluding siblings because you want to study all the "
                "cards.")

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

        duewsb = copy.deepcopy(indTODO)  # for computing improvement ratio

        # parsing desired deck size:
        if isinstance(target_deck_size, str):
            if target_deck_size in ["all", "100%"]:
                red("Taking the whole deck.")
                target_deck_size = len(indTODO) + 1
            elif target_deck_size.endswith("%"):
                red(f"Taking {target_deck_size} of the deck.")
                target_deck_size = 0.01 * int(target_deck_size[:-1]) * (
                    len(indTODO) + 1)
        target_deck_size = int(target_deck_size)

        if max_deck_size is not None:
            if target_deck_size > max_deck_size:
                diff = target_deck_size - max_deck_size
                red(f"Target deck size ({target_deck_size}) is above maximum "
                    f" threshold ({max_deck_size}), excluding {diff} cards.")
            target_deck_size = min(target_deck_size, max_deck_size)

        # checking if desired deck size is feasible:
        if target_deck_size > len(indTODO):
            yel(f"You wanted to create a deck with {target_deck_size} in it "
                f"but only {len(indTODO)} cards remain, taking the "
                "lowest value.")
        queue_size_goal = min(target_deck_size, len(indTODO))

        # displaying stats of the reference order or the
        # distance matrix:
        if display_stats:
            pd.set_option('display.float_format', lambda x: '%.5f' % x)
            try:
                whi("\nScore stats of due cards (weight adjusted):")
                if w1 != 0:
                    whi("Reference score of due cards: "
                        f" {(w1*df.loc[due, 'ref']).describe()}\n")
                else:
                    whi("Not showing statistics of the reference score, you "
                        "set its adjustment weight to 0")
                val = pd.DataFrame(data=w2*self.df_dist.values.ravel(),
                                   columns=['distance matrix']).describe(
                                           include='all')
                whi(f"Distance: {val}\n\n")
            except Exception as e:
                beep(f"Exception: {e}")
            pd.reset_option('display.float_format')

        # checking that there are no negative ref values
        assert (self.df.loc[due, "ref"].ravel() < 0).sum() == 0, (
            "Negative values in the reference score!")

        # final check before computing optimal order:
        for x in ["interval", "ref", "due"]:
            assert np.sum(np.isnan(df.loc[rated, x].values)) == len(rated), (
                    f"invalid treatment of rated cards, column : {x}")
            assert np.sum(np.isnan(df.loc[due, x].values)) == 0, (
                    f"invalid treatment of due cards, column : {x}")

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
                quite different from most cards of indQUEUE). This card is
                a good candidate to add to indQUEUE (same for np.median)
            * Naturally, np.min is given more importance than np.mean

            Best candidates are cards with high combinator output.
            The outut is substracted to the 'ref' of each indTODO card.

            Hence, at each turn, the card from indTODO with the lowest
                'w1*ref - w2*combinator' is removed from indTODO and added
                to indQUEUE.

            The content of 'queue' is the list of card_id in best review order.
            """
            minimum = 0.9 * np.min(array, axis=1)
            average = 0.05 * np.mean(array, axis=1)
            med = 0.05 * np.median(array, axis=1)
            dist_score = minimum + average + med
            # if self.log_level >= 2:
            #    avg = np.mean(dist_score) * self.score_adjustment_factor[1]
            #    tqdm.write(f"DIST_SCORE: {avg:02f}")
            return dist_score

        with tqdm(desc="Computing optimal review order",
                  unit=" card",
                  initial=len(rated),
                  smoothing=0,
                  file=self.t_strm,
                  total=queue_size_goal + len(rated)) as pbar:
            while len(queue) < queue_size_goal:
                # if self.log_level >= 2:
                #     ref_avg = np.mean(df.loc[indTODO, "ref"].values) * w1
                #     sp = " " * 22
                #     tqdm.write(f"{sp}REF_SCORE: {ref_avg:02f}")
                queue.append(indTODO[
                    (w1*df.loc[indTODO, "ref"].values -
                     w2*combinator(self.df_dist.loc[indTODO, indQUEUE].values
                                   ) + (
                     w3*np.random.rand(1, len(indTODO))
                     )
                     ).argmin()])
                indQUEUE.append(indTODO.pop(indTODO.index(queue[-1])))
                pbar.update(1)

        assert indQUEUE == rated + queue, (
                "indQUEUE is not the sum of rated and queue lists")

        self.df["action"] = "skipped_for_today"
        self.df.loc[queue, "action"] = "will_review"

        # reordering by best order
        try:
            if self.task == "filter_review_cards" and self.resort_by_dist:
                red("Reordering before creating the filtered deck "
                    "to maximize distance...")
                whi("But starts by the cards needing to be boosted)")
                new_queue = [queue[0]]
                to_process = [q for q in queue[1:]]

                # tell proportion to the user
                n = len([q for q in self.urgent_dues if q in queue])
                if n > 0:
                    proportion = int(n / len(queue) * 100)
                    yel(f"The filtered deck will contain {proportion}% of "
                        "boosted cards.")

                # create a new column like the reference score but -50 .
                # the non urgent_dues cards have their reference set at 50
                # In effect, this forces urgent_dues cards to appear
                # first in the filtered deck and only then the non urgent_dues
                # cards. But while still maximizing distance throughout the
                # reviews.
                self.df.loc[to_process, "ref_filtered"] = -50
                self.df.loc[[q
                             for q in to_process
                             if q not in self.urgent_dues
                             ], "ref_filtered"] = 50

                # keeping sorting order of user
                factor = 1 if w2 > 0 else -1

                assert set(new_queue) & set(to_process) == set(), (
                        "queue construction failed!")
                assert set(new_queue) | set(to_process) == set(queue), (
                        "queue construction failed! #2")

                pbar = tqdm(total=len(to_process),
                            file=self.t_strm,
                            desc="reordering filtered deck")
                while to_process:
                    pbar.update(1)
                    score = self.df.loc[to_process, "ref_filtered"] - (
                            factor * combinator(
                                self.df_dist.loc[to_process, new_queue].values
                            ))
                    new_queue.append(to_process.pop(score.argmin()))
                pbar.close()

                assert len(set()) == 0, "to_process is not empty!"
                assert len(set(new_queue) & set(queue)) == len(queue), (
                        "Invalid new queues!")
                queue = new_queue
        except Exception as e:
            beep(f"Exception when reordering: {e}")

        try:
            if w1 == 0:
                yel("Not showing distance without AnnA because you set "
                    "the adjustment weight of the reference score to 0.")
            else:
                woAnnA = [x
                          for x in df.sort_values(
                              "ref", ascending=True).index.tolist()
                          if x in duewsb][0:len(queue)]

                common = len(set(queue) & set(woAnnA))
                if common / len(queue) >= 0.95:
                    yel("Not displaying Improvement Ratio because almost "
                        "all cards were included in the new queue.")
                else:
                    spread_queue = np.sum(
                        self.df_dist.loc[queue, queue].values.ravel())
                    spread_else = np.sum(
                        self.df_dist.loc[woAnnA, woAnnA].values.ravel())

                    red("Sum of distance in the new queue:", end=" ")
                    yel(str(spread_queue))
                    red(f"Cards in common: {common} in a queue of "
                        f"{len(queue)} cards.")
                    red("Sum of distance of the queue if you had not used "
                        " AnnA:", end=" ")
                    yel(str(spread_else))

                    ratio = round(spread_queue / spread_else * 100 - 100, 1)
                    red("Improvement ratio:")
                    if ratio >= 0:
                        sign = "+"
                    else:
                        sign = "-"
                    red(pyfiglet.figlet_format(f"{sign}{abs(ratio)}%"))

        except Exception as e:
            beep(f"\nException: {e}")

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

        full_text = " ".join(self.df["text"].tolist()).replace(
                "'", " ").replace("<NEWFIELD>", " ")
        if exclude_OCR_text:
            ocr = re.findall("[A-Z][A-Z0-9]{2,}",
                             string=self.OCR_content,
                             flags=re.MULTILINE | re.DOTALL)
        else:
            ocr = []

        def exclude(word):
            """if exists as lowercase in text : probably just shouting for
            emphasis
            if exists in ocr : ignore"""
            if word.lower() in full_text or word in ocr:
                return False
            else:
                return True

        matched = list(set(
            [x for x in re.findall("[A-Z][A-Z0-9]{2,}", full_text)
             if exclude(x)]))

        if len(matched) == 0:
            red("No acronym found in those cards.")
            return True

        for compiled in self.acronym_dict:
            for acr in matched:
                if re.match(compiled, acr) is not None:
                    matched.remove(acr)

        if not matched:
            yel("All found acronyms were already replaced using the data "
                "in `acronym_list`.")
        else:
            yel("List of some acronyms still found:")
            if exclude_OCR_text:
                whi("(Excluding OCR text)")
            acr = random.sample(matched, k=min(5, len(matched)))
            pprint(", ".join(acr))
            beep(f"Acronyms: {acr}")

        print("")
        return True

    def _bury_or_create_filtered(self):
        """
        Either bury cards that are not in the optimal queue or create a
            filtered deck containing the cards to review in optimal order.

        * The filtered deck is created with setting 'sortOrder = 0', meaning
            ("oldest seen first"). This function then changes the review order
            inside this deck. That's why rebuilding this deck will keep the
            cards but lose the order.
        * filtered_deck_name_template can be used to automatically put the
            filtered decks to a specific location in your deck hierarchy.
            Leaving it to None will make the filtered deck appear alongside
            the original deck
        * This uses a threaded call to increase speed.
        * I do a few sanity check to see if the filtered deck
            does indeed contain the right number of cards and the right cards
        * -100 000 seems to be the starting value for due order in filtered
            decks by anki : cards are review from lowest to highest
            "due_order".
        * if task is set to 'bury_excess_learning_cards' or
            'bury_excess_review_cards', then no filtered deck will be created
            and AnnA will just bury some cards that are too similar to cards
            that you will review.
        """
        if self.task in ["bury_excess_learning_cards",
                         "bury_excess_review_cards"]:
            to_keep = self.opti_rev_order
            to_bury = [x for x in self.due_cards if x not in to_keep]
            assert len(to_bury) < len(
                self.due_cards), "trying to bury more cards than there are"
            red(f"Burying {len(to_bury)} cards out of {len(self.due_cards)}.")
            red("This will not affect the due order.")
            self._call_anki(action="bury", cards=to_bury)
            return True
        else:
            if self.filtered_deck_name_template is not None:
                filtered_deck_name = str(
                    self.filtered_deck_name_template + f" - {self.deckname}")
                filtered_deck_name = filtered_deck_name.replace("::", "_")
            else:
                filtered_deck_name = f"{self.deckname} - AnnA Optideck"
            self.filtered_deck_name = filtered_deck_name

            while filtered_deck_name in self._call_anki(action="deckNames"):
                beep(f"\nFound existing filtered "
                     f"deck: {filtered_deck_name} "
                     "You have to delete it manually, the cards will be "
                     "returned to their original deck.")
                input("Done? >")

        whi("Creating deck containing the cards to review: "
            f"{filtered_deck_name}")
        if self.filtered_deck_by_batch and (
                len(self.opti_rev_order) > self.filtered_deck_batch_size):
            yel("Creating batches of filtered decks...")
            batchsize = self.filtered_deck_batch_size
            cnt = 0
            while cnt <= 10000:
                batch_cards = self.opti_rev_order[cnt *
                                                  batchsize:(cnt+1)*batchsize]
                if not batch_cards:
                    yel(f"Done creating {cnt+1} filtered decks.")
                    break
                query = "is:due -rated:1 cid:" + ','.join(
                        [str(x) for x in batch_cards])
                self._call_anki(action="createFilteredDeck",
                                newDeckName=(
                                    f"{filtered_deck_name}_{cnt+1:02d}"),
                                searchQuery=query,
                                gatherCount=batchsize + 1,
                                reschedule=True,
                                sortOrder=5,
                                createEmpty=False)
                cnt += 1
        else:
            query = "is:due -rated:1 cid:" + ','.join(
                    [str(x) for x in self.opti_rev_order])
            self._call_anki(action="createFilteredDeck",
                            newDeckName=filtered_deck_name,
                            searchQuery=query,
                            gatherCount=len(self.opti_rev_order) + 1,
                            reschedule=True,
                            sortOrder=5,
                            createEmpty=False)

            yel("Checking that the content of filtered deck name is the "
                "same as the order inferred by AnnA...", end="")
            cur_in_deck = self._call_anki(
                    action="findCards",
                    query=f"\"deck:{filtered_deck_name}\"")
            diff = [x for x in self.opti_rev_order + cur_in_deck
                    if x not in self.opti_rev_order or x not in cur_in_deck]
            if len(diff) != 0:
                red("Inconsistency! The deck does not contain the same cards "
                    " as opti_rev_order!")
                pprint(diff)
                beep(f"\nNumber of inconsistent cards: "
                     f"{len(diff)}")

        yel("\nAsking anki to alter the due order...", end="")
        res = self._call_anki(action="setDueOrderOfFiltered",
                              cards=self.opti_rev_order)
        err = [x[1] for x in res if x[0] is False]
        if err:
            beep(f"\nError when setting due order : {err}")
            raise Exception(f"\nError when setting due order : {err}")
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
        whi(self.df.loc[order, "text"])
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
        yel(f"Dataframe exported to {name}.")
        return True

    def _do_2D_plots(self):
        """
        Create a 2D network plot of the deck.
        """
        red("Creating 2D plots...")
        assert self.plot_2D_embeddings, "invalid arguments!"
        assert hasattr(self, "knn"), "no knn in attribute!"
        assert "2D_embeddings" in self.df.columns, "no x/y columns in df!"

        # add a timeout to make sure it doesn't get stuck
        def time_watcher(signum, frame):
            "used to issue a timeout"
            raise TimeoutError("Timed out. Not finishing 2D plots.")
        signal.signal(signal.SIGALRM, time_watcher)
        signal.alarm(300)  # 5 minutes

        self.plot_dir = Path("NetworkX_plots")
        self.plot_dir.mkdir(exist_ok=True)
        G = nx.MultiGraph()
        positions = {}
        node_colours = []
        all_edges = {}

        # to bind each tags or deck to a color
        #self.df["colors"] = self.df["deckName"].factorize()[0]
        self.df["colors"] = self.df["tags"].factorize()[0]

        # add nodes
        for cid in tqdm(self.df.index, desc="Adding nodes", unit="node",
                        file=self.t_strm):
            nid = self.df.loc[cid, "note"]
            G.add_node(nid)  # duplicate nodes are ignored by networkx
            if nid not in positions:
                positions[nid] = list(self.df.loc[cid, "2D_embeddings"])
                node_colours.append(self.df.loc[cid, "colors"])

        # create a dict containing all edges and their weights
        min_w = np.inf
        max_w = -np.inf
        for i in tqdm(
                range(self.knn.shape[0]),
                desc="Computing edges",
                file=self.t_strm,
                unit="card"):
            knn_ar = self.knn.getcol(i).toarray().squeeze()
            neighbour_indices = np.where(knn_ar == 1)[0]
            # sort neighbour by distance
            neighbour_indices = sorted(
                    neighbour_indices,
                    key=lambda x: self.df_dist.loc[
                        self.df.index[i], self.df.index[x]],
                    reverse=True)
            neighbours_nid = [self.df.loc[self.df.index[ind], "note"]
                              for ind in neighbour_indices]
            noteId = self.df.loc[self.df.index[i], "note"]
            for ii, n_nid in enumerate(neighbours_nid):
                if noteId == n_nid:
                    # skip self neighbouring
                    continue
                if ii > 20:
                    # Only considering 20 first neighbours
                    break
                smallest = min(noteId, n_nid)
                largest = max(noteId, n_nid)
                # new weight is of decreasing importance
                new_w = self.df_dist.loc[
                    self.df.index[i], self.df.index[neighbour_indices[ii]]
                    ]
                if smallest not in all_edges:
                    all_edges[smallest] = {largest: new_w}
                else:
                    if largest not in all_edges[smallest]:
                        all_edges[smallest][largest] = new_w
                    else:
                        assert np.isclose(all_edges[smallest][largest], new_w), (
                                f"A weight is not symetric: {all_edges[smallest][largest]} != {new_w}")
                if all_edges[smallest][largest] > max_w:
                    max_w = all_edges[smallest][largest]
                if all_edges[smallest][largest] < min_w:
                    min_w = all_edges[smallest][largest]

        assert min_w >= 0 and min_w < max_w, (
                f"Impossible weight values: {min_w} and {max_w}")

        # minmax scaling of weights (even though the distances were already minmaxed)
        mean_w = (max_w + min_w) / 2
        for k, v in tqdm(all_edges.items(),
                         desc="Minmax weights",
                         file=self.t_strm):
            for sub_k, sub_v in all_edges[k].items():
                all_edges[k][sub_k] = (sub_v - min_w + mean_w) / (max_w - min_w + mean_w)

        # add each edge to the graph
        for k, v in tqdm(all_edges.items(),
                         desc="Adding edges",
                         file=self.t_strm):
            for sub_k, sub_v in all_edges[k].items():
                G.add_edge(k, sub_k, weight=sub_v)

        # 2D embeddings layout
        start = time.time()
        whi("Drawing embedding network...")
        self._do_plotly(G=G,
                        computed_layout=positions,
                        node_colours=node_colours,
                        title=f"{self.deckname} - embeddings"
                        )
        whi(f"Saved embeddings layout in {int(time.time()-start)}s!")
        nx.drawing.nx_pydot.write_dot(
                G, f'{self.plot_dir}/{self.deckname} - embeddings.dot')

        # computing spring layout
        n = len(node_colours)
        start = time.time()
        whi("\nDrawing spring layout network...")
        whi("    computing layout...")
        layout_spring = nx.spring_layout(
                G,
                k=100 / np.sqrt(n),  # repulsive force, default is 1/sqrt(n)
                pos=positions,  # initial positions is the 2D embeddings
                iterations=50,  # default to 50
                # fixed=None,  # keep those nodes at their starting position
                # center=None,  # center on a specific node
                dim=2,  # dimension of layout
                seed=4242,
                threshold=1e-4,  # stop goes below, default 1e-4
                )
        whi(f"Finished computing spring layout in {int(time.time()-start)}s")
        self._do_plotly(G=G,
                        computed_layout=layout_spring,
                        node_colours=node_colours,
                        title=f"{self.deckname} - spring"
                        )
        whi(f"Saved spring layout in {int(time.time()-start)}s!")
        nx.drawing.nx_pydot.write_dot(
                G, f'{self.plot_dir}/{self.deckname} - spring.dot')

        whi("Finished 2D plots")
        signal.alarm(0)  # turn off timeout

    def _do_plotly(self,
                   G,
                   computed_layout,
                   node_colours,
                   title):
        """
        Create 2D plotly plot from networkx graph.
        """
        edge_trace = Scatter(
            x=[],
            y=[],
            opacity=0.3,
            line=scatter.Line(color='rgba(136, 136, 136, .8)'),
            hoverinfo='none',
            mode='lines'
            )

        # only draw edges above threshold
        p = 0.05  # proportion of edge to draw
        edges_to_draw = []
        while len(edges_to_draw) <= 5:  # repeat to make sure to have enough
            edges_to_draw = [edg
                             for edg in G.edges.data()
                             if random.random() <= p]

        # randomize edges order
        edges_to_draw.sort(key=lambda x: random.random())

        n_limit = 1000
        if len(edges_to_draw) > n_limit:
            yel(f"Too many edges to draw ({len(edges_to_draw)}, keeping "
                f"{n_limit}.")
            # keep only first n
            edges_to_draw = edges_to_draw[:n_limit]

        for edge in tqdm(G.edges.data(),
                         desc="Plotting edges",
                         file=self.t_strm):
            if edge in edges_to_draw:
                x0, y0 = computed_layout[edge[0]]
                x1, y1 = computed_layout[edge[1]]
                edge_trace['x'] += tuple([x0, x1, None])
                edge_trace['y'] += tuple([y0, y1, None])

        node_trace = Scatter(
            x=[],
            y=[],
            text=[],
            mode='markers',
            textfont=dict(family='Calibri (Body)', size=25, color='black'),
            # opacity=0.1,
            # hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale=qualitative.Light24,
                reversescale=True,
                color=node_colours,
                size=15,
                colorbar=dict(
                    thickness=12,
                    title='Categories',
                    xanchor='left',
                    titleside='right'
                ),
                line=dict(width=5)))

        note_df = self.df.reset_index().drop_duplicates(subset="note").set_index("note")
        for node in tqdm(G.nodes(), desc="Plotting nodes", file=self.t_strm):
            x, y = computed_layout[node]
            node_trace['x'] += tuple([x])
            node_trace['y'] += tuple([y])

            content = note_df.loc[node, "text"]
            tag = note_df.loc[node, "tags"]
            deck = note_df.loc[node, "deckName"]
            node_trace['text'] += tuple([
                f"<b>deck:</b>{'<br>'.join(textwrap.wrap(deck, width=60))}"
                "<br>"
                f"<b>nid:</b>{node}"
                "<br>"
                f"<b>tag:</b>{'<br>'.join(textwrap.wrap(tag, width=60))}"
                "<br>"
                "<br>"
                f"<b>Content:</b>{'<br>'.join(textwrap.wrap(content, width=60))}"
                ])

        fig = Figure(data=[node_trace, edge_trace],
                     layout=Layout(
                         title=f'<br>{title}</br>',
                         titlefont=dict(size=25),
                         showlegend=True,
                         # width=1500,
                         # height=800,
                         hovermode='closest',
                         # margin=dict(b=20, l=350, r=5, t=200),
                         annotations=[dict(
                             text="",
                             showarrow=False,
                             xref="paper", yref="paper",
                             x=0.005, y=-0.002)],
                         xaxis=layout.XAxis(showgrid=False,
                                            zeroline=False,
                                            showticklabels=False),
                         yaxis=layout.YAxis(showgrid=False,
                                            zeroline=False,
                                            showticklabels=False)))

        # import dash
        # import dash_core_components as dcc
        # import dash_html_components as html #Create the app
        # app = dash.Dash()
        # app.layout = html.Div([
        #     dcc.Graph(figure=fig)
        # ])
        # @app.callback()
        # def open_browser(*args, **kwargs):
        #     """open anki browser when a point is clicked"""
        #     print("in")
        #     breakpoint()
        #     pid = points.point_inds[0]
        #     nid = computed_layout.keys()[pid]
        #     self._call_anki(action="guiBrowse", query=f"nid:{nid}")
        #app.run_server(debug=True, use_reloader=False)

        offpy(fig,
              filename=f"{self.plot_dir}/{title}.html",
              auto_open=True,
              show_link=True)


class ProgressParallel(joblib.Parallel):
    """
    simple subclass from joblib.Parallel with improved progress bar
    """
    def __init__(PP, tqdm_params, *args, **kwargs):
        PP._tqdm_params = tqdm_params
        super().__init__(*args, **kwargs)
        PP._latest_progress_printed = time.time()

    def __call__(PP, *args, **kwargs):
        with tqdm(**PP._tqdm_params) as PP._pbar:
            return joblib.Parallel.__call__(PP, *args, **kwargs)

    def print_progress(PP):
        if "total" in PP._tqdm_params:
            PP._pbar.total = PP._tqdm_params["total"]
        else:
            PP._pbar.total = PP.n_dispatched_tasks
        PP._pbar.n = PP.n_completed_tasks

        # only print progress every second:
        if abs(time.time() - PP._latest_progress_printed) >= 1:
            PP._pbar.refresh()
            PP._latest_progress_printed = time.time()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--deckname",
                        nargs=1,
                        metavar="DECKNAME",
                        dest="deckname",
                        default=None,
                        type=str,
                        required=True,
                        help=(
                            "the deck containing the cards you want to "
                            "review. If you don't supply this value or make a "
                            "mistake, AnnA will ask you to type in the "
                            "deckname, "
                            "with autocompletion enabled (use `<TAB>`). "
                            "Default is `None`."))
    parser.add_argument("--reference_order",
                        nargs=1,
                        metavar="REF_ORDER",
                        dest="reference_order",
                        default="relative_overdueness",
                        type=str,
                        required=False,
                        help=(
                            "either \"relative_overdueness\" or "
                            "\"lowest_interval\" or \"order_added\" or "
                            "\"LIRO_mix\". It is the reference used to sort the "
                            "card before adjusting them using the similarity "
                            "scores. Default is `\"relative_overdueness\"`. Keep "
                            "in mind that my relative_overdueness is a "
                            "reimplementation of the default overdueness of anki "
                            "and is not absolutely exactly the same but should be "
                            "a close approximation. If you find edge cases or "
                            "have any idea, please open an issue. LIRO_mix is "
                            "simply the the weighted average of relative "
                            "overdueness and lowest interval (4 times more "
                            "important than RO) (after some post processing). I "
                            "created it as a compromise between old and new "
                            "courses. My implementation of relative overdueness "
                            "includes a boosting feature: if your dues contain "
                            "cards with its overdueness several times larger than "
                            "its interval, they are urgent. AnnA will add a tag "
                            "to them and increase their likelyhood of being part "
                            "of the Optideck."))
    parser.add_argument("--task",
                        nargs=1,
                        metavar="TASK",
                        dest="task",
                        default="filter_review_cards",
                        required=True,
                        help=(
                            "can be \"filter_review_cards\", "
                            "\"bury_excess_learning_cards\", "
                            "\"bury_excess_review_cards\", \"add_KNN_to_field\""
                            ". Respectively to create "
                            "a filtered deck with the cards, or bury only the "
                            "similar learning cards (among other learning cards), "
                            "or bury only the similar cards in review (among "
                            "other review cards) or just find the nearest "
                            "neighbours of each note and save it to the field "
                            "'Nearest_neighbours' of each note. Default is "
                            "\"`filter_review_cards`\"."))
    parser.add_argument("--target_deck_size",
                        nargs=1,
                        metavar="TARGET_SIZE",
                        dest="target_deck_size",
                        default="deck_config",
                        required=True,
                        help=(
                            "indicates the size of the filtered deck to "
                            "create. Can be the number of due cards like \"100\", "
                            "a proportion of due cards like '80%%', the word "
                            "\"all\" or \"deck_config\" to use the deck's "
                            "settings for max review. Default is `deck_config`."))
    parser.add_argument("--max_deck_size",
                        nargs=1,
                        metavar="MAX_DECK_SIZE",
                        dest="max_deck_size",
                        default=None,
                        required=False,
                        type=int,
                        help=(
                            "Maximum number of cards to put in the filtered deck "
                            "or to leave unburied. Default is `None`."))
    parser.add_argument("--stopwords_lang",
                        nargs="+",
                        metavar="STOPLANG",
                        dest="stopwords_lang",
                        default="english,french",
                        type=str,
                        required=False,
                        help=(
                            "a comma separated list of languages used to "
                            "construct a list of stop words (i.e. words that will "
                            "be ignored, like \"I\" or \"be\" in English). "
                            "Default is `english french`."))
    parser.add_argument("--rated_last_X_days",
                        nargs=1,
                        metavar="RATED_LAST_X_DAYS",
                        dest="rated_last_X_days",
                        default=4,
                        required=False,
                        help=(
                            "indicates the number of passed days to take "
                            "into account when fetching past anki sessions. If "
                            "you rated 500 cards yesterday, then you don't want "
                            "your today cards to be too close to what you viewed "
                            "yesterday, so AnnA will find the 500 cards you "
                            "reviewed yesterday, and all the cards you rated "
                            "before that, up to the number of days in "
                            "rated_last_X_days value. Default is `4` (meaning "
                            "rated today, and in the 3 days before today). A "
                            "value of 0 or `None` will disable fetching those "
                            "cards. A value of 1 will only fetch cards that were "
                            "rated today. Not that this will include cards rated "
                            "in the last X days, no matter if they are reviews "
                            "or learnings. you can change this using "
                            "\"highjack_rated_query\" argument."))
    parser.add_argument("--score_adjustment_factor",
                        nargs="+",
                        metavar="SCORE_ADJUSTMENT_FACTOR",
                        dest="score_adjustment_factor",
                        default="1,2",
                        type=str,
                        required=False,
                        help=(
                            "a comma separated list of numbers used to "
                            "adjust the value of the reference order compared to "
                            "how similar the cards are. Default is `1,2`. For "
                            "example: '1, 1.3' means that the algorithm will "
                            "spread the similar cards farther apart."))
    parser.add_argument("--field_mapping",
                        nargs=1,
                        metavar="FIELD_MAPPING_PATH",
                        dest="field_mappings",
                        default="field_mappings.py",
                        type=str,
                        required=False,
                        help=(
                            "path of file that indicates which field to keep "
                            "from which note type and in which order. Default "
                            "value is `field_mappings.py`. If empty or if no "
                            "matching notetype was found, AnnA will only take "
                            "into account the first 2 fields. If you assign a "
                            "notetype to `[\"take_all_fields]`, AnnA "
                            "will grab "
                            "all fields of the notetype in the same "
                            "order as they"
                            " appear in Anki's interface."))
    parser.add_argument("--acronym_file",
                        nargs=1,
                        metavar="ACRONYM_FILE_PATH",
                        dest="acronym_file",
                        default="acronym_file.py",
                        required=False,
                        help=(
                            "a python file containing dictionaries that "
                            "themselves contain acronyms to extend in "
                            "the text "
                            "of cards. For example `CRC` can be extended "
                            "to `CRC "
                            "(colorectal cancer)`. (The parenthesis are "
                            "automatically added.) Default is "
                            "`\"acronym_file.py\"`. The matching is case "
                            "sensitive only if the key contains uppercase "
                            "characters. The \".py\" file extension is not "
                            "mandatory."))
    parser.add_argument("--acronym_list",
                        nargs="+",
                        metavar="ACRONYM_LIST",
                        dest="acronym_list",
                        default=None,
                        type=str,
                        required=False,
                        help=(
                            "a comma separated list of name of dictionaries "
                            "to extract file supplied in `acronym_file`. "
                            "Used to extend text, for instance "
                            "`AI_machine_learning,medical_terms`. Default to "
                            "None."))
    parser.add_argument("--minimum_due",
                        nargs=1,
                        metavar="MINIMUM_DUE_CARDS",
                        dest="minimum_due",
                        default=5,
                        type=int,
                        required=False,
                        help=(
                            "stops AnnA if the number of due cards is "
                            "inferior to this value. Default is `5`."))
    parser.add_argument("--highjack_due_query",
                        nargs=1,
                        metavar="HIGHJACK_DUE_QUERY",
                        dest="highjack_due_query",
                        default=None,
                        required=False,
                        help=(
                            "bypasses the browser query used to find the "
                            "list of due cards. You can set it for example to "
                            "`deck:\"my_deck\" is:due -rated:14 flag:1`. "
                            "Default "
                            "is `None`. **Keep in mind that, when highjacking "
                            "queries, you have to specify the deck otherwise "
                            "AnnA will compare your whole collection.**"))
    parser.add_argument("--highjack_rated_query",
                        nargs=1,
                        metavar="HIGHJACK_RATED_QUERY",
                        dest="highjack_rated_query",
                        default=None,
                        required=False,
                        help=(
                            "same idea as above, bypasses the query used "
                            "to fetch rated cards in anki. Related to "
                            "`highjack_due_query` although you can set "
                            "only one "
                            "of them. Default is `None`."))
    parser.add_argument("--low_power_mode",
                        dest="low_power_mode",
                        default=False,
                        action="store_true",
                        required=False,
                        help=(
                            "enable to reduce the computation needed for "
                            "AnnA, making it usable for less powerful "
                            "computers. "
                            "Default to `False`. It skips trying to find "
                            "acronyms "
                            "that were not replaced and reduces the "
                            "ngram range."))
    parser.add_argument("--log_level",
                        nargs=1,
                        metavar="LOG_LEVEL",
                        dest="log_level",
                        default=0,
                        type=int,
                        required=False,
                        help=(
                            "can be any number between 0 and 2. Default is "
                            "`0` to only print errors. 1 means print "
                            "also useful "
                            "information and >=2 means print everything. "
                            "Messages are color coded so it might be better "
                            "to "
                            "leave it at 3 and just focus on colors."))
    parser.add_argument("--replace_greek",
                        action="store_true",
                        dest="replace_greek",
                        default=True,
                        required=False,
                        help=(
                            "if True, all greek letters will be replaced "
                            "with a spelled version. For example `\u03C3` "
                            "becomes `sigma`. Default is `True`."))
    parser.add_argument("--keep_OCR",
                        dest="keep_OCR",
                        default=True,
                        action="store_true",
                        required=False,
                        help=(
                            "if True, the OCR text extracted using the "
                            "great AnkiOCR addon "
                            "(https://github.com/cfculhane/AnkiOCR/) will be "
                            "included in the card. Default is `True`."))
    parser.add_argument("--append_tags",
                        dest="append_tags",
                        default=True,
                        required=False,
                        action="store_true",
                        help=(
                            "Wether to append the 2 deepest tags to the "
                            "cards content or to add no tags. Default "
                            "to `True`."))
    parser.add_argument("--tags_to_ignore",
                        nargs="*",
                        metavar="TAGS_TO_IGNORE",
                        dest="tags_to_ignore",
                        default="[AnnA]",
                        type=str,
                        required=False,
                        help=(
                            "a comma separated list of regexp of tags to "
                            "ignore when "
                            "appending tags to cards. This is not a "
                            "list of tags "
                            "whose card should be ignored! Default is "
                            "'[AnnA]'. Set to None to disable it."))
    parser.add_argument("--tags_separator",
                        nargs=1,
                        metavar="TAGS_SEP",
                        dest="tags_separator",
                        default="::",
                        type=str,
                        required=False,
                        help=(
                            "separator between levels of tags. Default "
                            "to `::`."))
    parser.add_argument("--add_KNN_to_field",
                        action="store_true",
                        dest="add_KNN_to_field",
                        default=False,
                        required=False,
                        help=(
                            "Wether to add a query to find the K nearest"
                            "neighbour of a given card to a new field "
                            "called 'Nearest_neighbours' (only if already "
                            "present in the model). Be careful not to "
                            "overwrite the fields by running AnnA "
                            "several times in a row! For example by first "
                            "burying learning cards then filtering "
                            "review cards. This argument is to be used if "
                            "you want to find the KNN only for the cards of "
                            "the deck in question and that are currently due."
                            " If you want to run this on the complete deck "
                            "you should use the 'task' argument."))
    parser.add_argument("--filtered_deck_name_template",
                        nargs=1,
                        metavar="FILTER_DECK_NAME_TEMPLATE",
                        dest="filtered_deck_name_template",
                        default=None,
                        required=False,
                        type=str,
                        help=(
                            "name template of the filtered deck to create. "
                            "Only available if task is set to "
                            "\"filter_review_cards\". Default is `None`."))
    parser.add_argument("--filtered_deck_by_batch",
                        action="store_true",
                        dest="filtered_deck_by_batch",
                        default=False,
                        required=False,
                        help=(
                            "To enable creating batch of filtered "
                            "decks. Default is `False`."))
    parser.add_argument("--filtered_deck_batch_size",
                        nargs=1,
                        metavar="FILTERED_DECK_BATCH_SIZE",
                        dest="filtered_deck_batch_size",
                        default=25,
                        type=int,
                        required=False,
                        help=(
                            "If creating batch of filtered deck, this is "
                            "the number of cards in each. Default is `25`."))
    parser.add_argument("--show_banner",
                        action="store_true",
                        dest="show_banner",
                        default=True,
                        required=False,
                        help=(
                            "used to display a nice banner when "
                            "instantiating the "
                            "collection. Default is `True`."))
    parser.add_argument("--repick_task",
                        nargs=1,
                        metavar="REPICK_TASK",
                        dest="repick_task",
                        default="boost",
                        required=False,
                        help=(
                            "Define what happens to cards deemed urgent "
                            "in 'relative_overdueness' ref mode. If "
                            "contains "
                            "'boost', those cards will have a boost in "
                            "priority "
                            "to make sure you will review them ASAP. If "
                            "contains "
                            "'addtag' a tag indicating which card is urgent "
                            "will "
                            "be added at the end of the run. Disable by "
                            "setting "
                            "it to None. Default is `boost`."))
    parser.add_argument("--vectorizer",
                        nargs=1,
                        metavar="VECTORIZER",
                        dest="vectorizer",
                        default="TFIDF",
                        required=False,
                        type=str,
                        help=(
                            "can nowadays only be set to \"TFIDF\", but "
                            "kept for legacy reasons."))
    parser.add_argument("--TFIDF_dim",
                        nargs=1,
                        metavar="TFIDF_DIMENSIONS",
                        dest="TFIDF_dim",
                        default="auto",
                        required=False,
                        help=(
                            "the number of dimension to keep using "
                            "SVD. If 'auto' will automatically find the "
                            "best number of dimensions to keep 80% of the "
                            "variance. If an int, will do like 'auto' but "
                            "starting from the supplied value. "
                            "Default is `auto`, you cannot disable "
                            "dimension reduction for TF_IDF because "
                            "that would result in a sparse "
                            "matrix. (More information at "
                            "https://scikit-learn.org/stable/modules/"
                            "generated/sklearn.decomposition.Trunca"
                            "tedSVD.html)."))
    parser.add_argument("--TFIDF_tokenize",
                        dest="TFIDF_tokenize",
                        default=True,
                        action="store_true",
                        required=False,
                        help=(
                            "default to `True`. Enable sub word "
                            "tokenization, for example turn "
                            "`hypernatremia` to `hyp + er + natr + emia`."
                            " You cannot enable both `TFIDF_tokenize` and "
                            "`TFIDF_stem` but should absolutely enable "
                            "at least "
                            "one."))
    parser.add_argument("--tokenizer_model",
                        dest="tokenizer_model",
                        default="GPT",
                        metavar="TOKENIZER_MODEL",
                        required=False,
                        help=(
                            "default to `GPT`. Model to use for tokenizing "
                            "the text before running TFIDF. Possible values "
                            "are 'bert' and 'GPT' which correspond "
                            "respectivelly to `bert-base-multilingual-cased`"
                            " and `gpt_neox_20B` They "
                            "should work on just about any languages. Use "
                            "'Both' to concatenate both tokenizers. "
                            "(experimental)"))
    parser.add_argument("--plot_2D_embeddings",
                        dest="plot_2D_embeddings",
                        default=False,
                        action="store_true",
                        required=False,
                        help=(
                            "EXPERIMENTAL AND UNFINISHED. "
                            "default to `False`. Will compute 2D embeddins "
                            "then create a 2D plots at the end."))
    parser.add_argument("--TFIDF_stem",
                        dest="TFIDF_stem",
                        default=False,
                        action="store_true",
                        required=False,
                        help=(
                            "default to `False`. Wether to enable "
                            "stemming of words. Currently the PorterStemmer "
                            "is used, and was made for English but can still "
                            "be useful for some other languages. Keep in "
                            "mind that this is the longest step when "
                            "formatting text."))
    parser.add_argument("--dist_metric",
                        nargs=1,
                        metavar="DIST_METRIC",
                        dest="dist_metric",
                        type=str,
                        default="RBF",
                        required=False,
                        help=(
                            "when computing the distance matrix, wether to "
                            "use 'cosine' or 'rbf' metrics. Changing this "
                            "should not affect the results significantly."))
    parser.add_argument("--whole_deck_computation",
                        dest="whole_deck_computation",
                        default=False,
                        required=False,
                        action="store_true",
                        help=(
                            "defaults to `False`. Use ankipandas to "
                            "extract all text from the deck to feed into the "
                            "vectorizer. Results in more accurate relative "
                            "distances between cards."
                            " (more information at "
                            "https://github.com/klieret/AnkiPandas)"))
    parser.add_argument("--enable_fuzz",
                        dest="enable_fuzz",
                        default=True,
                        action="store_true",
                        required=False,
                        help=(
                            "Disable fuzzing when computing optimal "
                            "order , otherwise a small random vector is "
                            "added to "
                            "the reference_score and distance_score of each "
                            "card. Note that this vector is multiplied by the "
                            "average of the `score_adjustment_factor` then "
                            "multiplied by the mean distance then "
                            "divided by 10 to make sure that it does not "
                            "overwhelm the other factors. Defaults "
                            "to `True`."))
    parser.add_argument("--resort_by_dist",
                        dest="resort_by_dist",
                        default=True,
                        action="store_true",
                        required=False,
                        help=(
                            "Recomputing the best order of the cards in the "
                            "filtered deck without taking into account the "
                            "reference score. This is useful if you are "
                            "sure to "
                            "review the entierety of the filtered deck "
                            "today as it "
                            "will minimize similarity between cards. If "
                            "you are not sure you will finish the "
                            "deck, it can be best to set to False to make "
                            "sure "
                            "you review first the most urgent cards. This "
                            "feature is active only if you set the task to '"
                            "filter_review_cards'. Default to True."))
    parser.add_argument("--profile_name",
                        nargs=1,
                        metavar="PROFILE_NAME",
                        dest="profile_name",
                        default=None,
                        required=False,
                        help=(
                            "defaults to `None`. Profile named "
                            "used by ankipandas to find your collection. If "
                            "None, ankipandas will use the most probable "
                            "collection."))
    parser.add_argument("--keep_console_open",
                        dest="console_mode",
                        default=False,
                        action="store_true",
                        required=False,
                        help=(
                            "defaults to `False`. Set to True to "
                            "open a python console after running."))

    args = parser.parse_args().__dict__

    # makes sure that argument are correctly parsed :
    for arg in args:
        if isinstance(args[arg], list) and len(args[arg]) == 1:
            args[arg] = args[arg][0]
        if isinstance(args[arg], str):
            if args[arg] == "None":
                args[arg] = None
            elif "," in args[arg]:
                args[arg] = args[arg].split(",")

    whi("Launched AnnA with arguments :\r")
    pprint(args)

    if args["console_mode"]:
        console_mode = True
    else:
        console_mode = False

    args.pop("console_mode")
    anna = AnnA(**args)
    if console_mode:
        red("\n\nRun finished. Opening console:\n(You can access the last "
            "instance of AnnA by inspecting variable \"anna\")\n")
        import code
        beep("Finished!")
        code.interact(local=locals())

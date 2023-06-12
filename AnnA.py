import pickle
import hashlib
import webbrowser
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
import signal
import os
import subprocess
import shlex
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
from sentence_transformers import SentenceTransformer
import ftfy

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.metrics import pairwise_distances, pairwise_kernels
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.preprocessing import normalize
from sklearn import cluster
import umap.umap_
from bertopic import BERTopic
import hdbscan

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
signal.signal(signal.SIGINT, (lambda signal, frame: breakpoint()))

# adds logger file, restrict it to X lines
log_formatter = logging.Formatter('%(asctime)s %(levelname)s %(funcName)s(%(lineno)d) %(message)s')
file_handler = logging.handlers.RotatingFileHandler(
        "logs.txt",
        mode='a',
        maxBytes=1000000,
        backupCount=3,
        encoding=None,
        delay=0,
        )
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(log_formatter)

log = logging.getLogger()
log.setLevel(logging.INFO)
log.addHandler(file_handler)


def set_global_logging_level(level=logging.ERROR, prefices=[""]):
    """
    To control logging level for various modules used in the application:
    https://github.com/huggingface/transformers/issues/3050#issuecomment-682167272
    """
    prefix_re = re.compile(fr'^(?:{ "|".join(prefices) })')
    for name in logging.root.manager.loggerDict:
        if re.search(prefix_re, name):
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
        #beepy.beep(sound, **args)
        pass
    except Exception:
        # retry sound if failed
        time.sleep(1)
        try:
            #beepy.beep(sound, **args)
            pass
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
                 # "order_added", "LIRO_mix"
                 task="filter_review_cards",
                 # any of "filter_review_cards",
                 # "bury_excess_review_cards", "bury_excess_learning_cards"
                 # "just_add_KNN", "just_plot"
                 target_deck_size="deck_config",
                 # format: 80%, "all", "deck_config"
                 max_deck_size=None,
                 stopwords_lang=["english", "french"],
                 rated_last_X_days=4,
                 score_adjustment_factor=[1, 5],
                 field_mappings="utils/field_mappings.py",
                 acronym_file="utils/acronym_example.py",
                 acronym_list=None,

                 # others:
                 minimum_due=5,
                 highjack_due_query=None,
                 highjack_rated_query=None,
                 low_power_mode=False,
                 log_level=0,  # 0, 1, 2
                 replace_greek=True,
                 keep_OCR=True,
                 append_tags=False,
                 tags_to_ignore=["AnnA", "leech"],
                 add_KNN_to_field=False,
                 filtered_deck_name_template=None,
                 filtered_deck_at_top_level=True,
                 filtered_deck_by_batch=False,
                 filtered_deck_batch_size=25,
                 show_banner=True,
                 repick_task="boost",  # None, "addtag", "boost" or
                 # "boost&addtag"
                 enable_fuzz=True,
                 resort_by_dist="closer",
                 resort_split=False,

                 # vectorization:
                 vectorizer="embeddings",
                 embed_model="paraphrase-multilingual-mpnet-base-v2",
                 # left for legacy reason
                 ndim_reduc="auto",
                 TFIDF_tokenize=True,
                 TFIDF_tknizer_model="GPT",
                 TFIDF_stem=False,
                 plot_2D_embeddings=False,
                 plot_dir="Plots",
                 dist_metric="cosine",  # 'RBF' or 'cosine' or 'euclidean"

                 whole_deck_computation=False,
                 profile_name=None,
                 sync_behavior="before&after",
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

        if tags_to_ignore is None:
            tags_to_ignore = []
        assert isinstance(tags_to_ignore, list), "tags_to_ignore is not a list"
        self.tags_to_ignore = [re.compile(f".*{t.strip()}.*")
                               if ".*" not in t
                               else re.compile(t.strip())
                               for t in tags_to_ignore]
        assert len(tags_to_ignore) == len(self.tags_to_ignore)

        assert isinstance(add_KNN_to_field, bool), (
                "Invalid type of `add_KNN_to_field`")
        self.add_KNN_to_field = add_KNN_to_field

        assert isinstance(
            low_power_mode, bool), "Invalid type of `low_power_mode`"
        self.low_power_mode = low_power_mode

        assert vectorizer in ["TFIDF", "embeddings"], "Invalid value for `vectorizer`"
        self.vectorizer = vectorizer
        if vectorizer == "embeddings" and whole_deck_computation:
            raise Exception("You can't use whole_deck_computation for embeddings")

        self.embed_model = embed_model

        assert isinstance(
            stopwords_lang, list), "Invalid type of var `stopwords_lang`"
        self.stopwords_lang = stopwords_lang

        assert isinstance(ndim_reduc, (int, type(None), str)
                          ), "Invalid type of `ndim_reduc`"
        if isinstance(ndim_reduc, str):
            assert ndim_reduc == "auto", "Invalid value for `ndim_reduc`"
        self.ndim_reduc = ndim_reduc

        assert isinstance(plot_2D_embeddings, bool), (
            "Invalid type of `plot_2D_embeddings`")
        self.plot_2D_embeddings = plot_2D_embeddings
        self.plot_dir = Path(str(plot_dir))

        assert isinstance(TFIDF_stem, bool), "Invalid type of `TFIDF_stem`"
        assert isinstance(
            TFIDF_tokenize, bool), "Invalid type of `TFIDF_tokenize`"
        assert TFIDF_stem + TFIDF_tokenize not in [0, 2], (
            "You have to enable either tokenization or stemming!")
        self.TFIDF_stem = TFIDF_stem
        self.TFIDF_tokenize = TFIDF_tokenize
        assert TFIDF_tknizer_model.lower() in ["bert", "gpt", "both"], (
            "Wrong tokenizer model name!")
        self.TFIDF_tknizer_model = TFIDF_tknizer_model
        assert dist_metric.lower() in ["cosine", "rbf", "euclidean"], (
            "Invalid 'dist_metric'")
        self.dist_metric = dist_metric.lower()

        assert task in ["filter_review_cards",
                        "bury_excess_learning_cards",
                        "bury_excess_review_cards",
                        "just_add_KNN",
                        "just_plot"], "Invalid value for `task`"
        if task in ["bury_excess_learning_cards",
                    "bury_excess_review_cards"]:
            if task == "bury_excess_learning_cards":
                red("Task : bury some learning cards")
            elif task == "bury_excess_review_cards":
                red("Task : bury some reviews\n")
        elif task == "filter_review_cards":
            red("Task : created filtered deck containing review cards")
        elif task == "just_add_KNN":
            red("Task : find the nearest neighbor of each note and "
                "add it to a field.")
        elif task == "just_plot":
            red("Task : vectorize the cards and create a 2D plot.")
            assert plot_2D_embeddings, "argument plot_2D_embeddings should be True"
        else:
            raise ValueError()
        self.task = task

        assert isinstance(filtered_deck_name_template, (str, type(
            None))), "Invalid type for `filtered_deck_name_template`"
        self.filtered_deck_name_template = filtered_deck_name_template

        assert isinstance(filtered_deck_by_batch,
                          bool), "Invalid type for `filtered_deck_by_batch`"
        self.filtered_deck_by_batch = filtered_deck_by_batch

        assert isinstance(filtered_deck_at_top_level,
                          bool), "Invalid type for `filtered_deck_at_top_level`"
        self.filtered_deck_at_top_level = filtered_deck_at_top_level

        assert isinstance(filtered_deck_batch_size,
                          int), "Invalid type for `filtered_deck_batch_size`"
        self.filtered_deck_batch_size = filtered_deck_batch_size

        assert isinstance(whole_deck_computation,
                          bool), "Invalid type for `whole_deck_computation`"
        self.whole_deck_computation = whole_deck_computation

        assert isinstance(profile_name, (str, type(None))
                          ), "Invalid type for `profile_name`"
        self.profile_name = profile_name

        if sync_behavior is None:
            sync_behavior = ""
        assert isinstance(sync_behavior, str), (
            "sync_behavior should be a string!")
        assert (
                (
                    "before" in sync_behavior
                    ) or ("after" in sync_behavior
                        ) or (sync_behavior == "")
                    ), (f"Wrong value of 'sync_behavior': '{sync_behavior}'")
        self.sync_behavior = sync_behavior

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

        assert isinstance(resort_by_dist, (bool, str)), (
            "Invalid type for 'resort_by_dist'")
        if isinstance(resort_by_dist, str):
            resort_by_dist = resort_by_dist.lower()
        self.resort_by_dist = resort_by_dist
        assert resort_by_dist in ["farther", "closer", False], (
            "Invalid 'resort_by_dist' value")
        assert isinstance(resort_split, bool), (
            "Invalid type for 'resort_split'")
        self.resort_split = resort_split

        # initialize joblib caching
        # self.mem = joblib.Memory("./.cache", mmap_mode="r", verbose=0)

        # additional processing of arguments #################################

        # load or ask for deckname
        self.deckname = self._check_deckname(deckname)
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
            if self.TFIDF_tknizer_model.lower() in ["bert", "both"]:
                yel("Will use BERT as tokenizer.")
                self.tokenizer_bert = Tokenizer.from_file(
                    "./utils/bert-base-multilingual-cased_tokenizer.json")
                self.tokenizer_bert.no_truncation()
                self.tokenizer_bert.no_padding()
                self.tokenize = self._bert_tokenize
            if self.TFIDF_tknizer_model.lower() in ["gpt", "both"]:
                yel("Will use GPT as tokenizer.")
                self.tokenizer_gpt = Tokenizer.from_file(
                    "./utils/gpt_neox_20B_tokenizer.json")
                self.tokenizer_gpt.no_truncation()
                self.tokenizer_gpt.no_padding()
                self.tokenize = self._gpt_tokenize
            if self.TFIDF_tknizer_model.lower() == "both":
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
            acr_mod = importlib.import_module(
                    acronym_file.replace("/", ".").replace(".py", "")
                    )

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
                    if re.search(compiled, value2):
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
                    self.field_mappings.replace("/", ".").replace(".py", ""))
                self.field_dic = imp.field_dic
            except Exception as e:
                beep(f"Error with field mapping file, will use "
                     f"default values. {e}")
                self.field_dic = {"dummyvalue": ["dummyvalue"]}
            if self.vectorizer == "embeddings":
                red("Deduplicating field mapping because using embeddings")
                for k, v in self.field_dic.items():
                    assert isinstance(v, list), f"Value of self.field_dic is not list: '{v}'"
                    if len(v) > 1:
                        new = []
                        for item in v:
                            if item not in new:
                                new.append(item)
                        self.field_dic[k] = new

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

        # trigger a sync
        if "before" in self.sync_behavior:
            yel("Syncing before execution...")
            sync_output = self._call_anki(action="sync")
            assert sync_output is None or sync_output == "None", (
                "Error during sync?: '{sync_output}'")
            time.sleep(1)  # wait for sync to finish, just in case
            whi("Done!")
        else:
            yel("Not syncing.")

        # load deck settings if needed
        if self.target_deck_size == "deck_config":
            self.target_deck_size = str(self.deck_config["rev"]["perDay"])
            yel("Set 'target_deck_size' to deck's value: "
                f"{self.target_deck_size}")

        red(f"Starting task: {task}")
        if task in ["bury_excess_learning_cards",
                    "bury_excess_review_cards"]:
            if self._common_init():
                self._add_neighbors_to_notes()
                self._compute_optimized_queue()
                self._bury_or_create_filtered()
            else:
                return

        elif task == "filter_review_cards":
            if self._common_init():
                self._add_neighbors_to_notes()
                self._compute_optimized_queue()
                self._bury_or_create_filtered()
            else:
                return

        elif task == "just_add_KNN":
            whi("(Setting 'rated_last_X_days' to None)")
            self.rated_last_X_days = None
            if self._common_init():
                self._add_neighbors_to_notes()
            else:
                return

        elif task == "just_plot":
            self.rated_last_X_days = None
            assert self._common_init(), "Error during _common_init"

        else:
            raise ValueError(f"Invalid task value: {task}")

        # create 2D plots if needed
        if self.plot_2D_embeddings:
            try:
                self._compute_plots()
            except Exception as err:
                beep(f"Exception when plotting 2D embeddings: '{err}'")
                red(traceback.format_exc())
            signal.alarm(0)  # turn off timeout
        red(f"Done with task '{self.task}' on deck '{self.deckname}'")
        gc.collect()

        if "after" in self.sync_behavior:
            yel("Syncing after run...")
            sync_output = self._call_anki(action="sync")
            assert sync_output is None or sync_output == "None", (
                f"Error during sync?: '{sync_output}'")
            time.sleep(1)  # wait for sync to finish, just in case
            whi("Done!")
        else:
            yel("Not syncing after run.")

    @classmethod
    def _call_anki(self, action, **params):
        """ bridge between local python libraries and AnnA Companion addon
        (a fork from anki-connect) """
        def request_wrapper(action, **params):
            return {'action': action, 'params': params, 'version': 6}

        requestJson = json.dumps(request_wrapper(action, **params)
                                 ).encode('utf-8')

        # otherwise beep cannot be used in a classmethod and exception fail:
        global beep
        if "beep" not in globals().keys():
            def beep(x):
                return beepy.beep(f"(fallback beepy){x}")
                #pass

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


    def _common_init(self):
        "Calls one by one the methods needed by all tasks anyway."
        if not self._init_dataFrame():
            # not enough cards were found, interrupting the run
            # without exception to avoid stopping batch run
            return False
        self._format_card()
        self._print_acronyms()
        self._compute_projections()
        self._compute_distance_matrix()
        return True

    def _fetch_cards(self, card_id):
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
            target_thread_n = 3
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

    def _check_deckname(self, deckname):
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
        # fetch due cards
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

        elif self.task in ["just_add_KNN", "just_plot"]:
            yel("Getting all card list except suspended...")
            query = (f"\"deck:{self.deckname}\" -is:suspended")
            whi(" >  '" + query + "'")
            due_cards = self._call_anki(action="findCards", query=query)
            whi(f"Found {len(due_cards)} cards...\n")
        else:
            raise ValueError(f"Invalid task: '{self.task}'")

        # fetch recently rated cards
        def iterated_fetcher(query):
            """
            asks anki multiple times for 'rated:k -rated:k-1' for k
            from 'rated_last_X_days' to 2. This way if a card was rated
            3 times in the last 10 days it will appears 3 times in the 'rated'
            list instead of appearing only once.
            """
            match = re.search(r" rated:(\d+)", query)
            assert match is not None, r"' rated:\d' not found in query"
            assert len(match.groups()) == 1, (
                rf"Found multiple ' rated:\d' in query: '{match}'")
            days = int(match.groups()[0])

            previously_rated = []
            dupli_check = []  # used to raise an issue if 2 days have exactly
            # the same reviews
            day_of_review = []  # will be the same length as 'previously_rated' and
            # contains the "day" of the review. For example "4" if the review
            # was 4days ago. This will be stored as attribute to
            # be used when computing the optimal order.
            for d in range(days, 0, -1):
                if d > 1:  # avoid having '-rated:0'
                    new_query = query.replace(f"rated:{days}",
                                              f"rated:{d} -rated:{d-1}")
                else:
                    assert d == 1, f"invalid value for d: '{d}'"
                    new_query = query.replace(f"rated:{days}",
                                              f"rated:{d}")
                rated_this_day = self._call_anki(
                        action="findCards",
                        query=new_query)
                if rated_this_day != []:  # check unicity of review session
                    assert rated_this_day not in dupli_check, (
                        f"2 days have identical reviews! '{d} and "
                        f"{days-dupli_check.index(rated_this_day)}")
                dupli_check.append(rated_this_day)
                previously_rated.extend(rated_this_day)
                day_of_review.extend([d] * len(rated_this_day))

            # look for cards in filtered decks created by AnnA for the
            # same deck as they will be reviewed most probably today,
            # so they have to be counted as 'rated'
            in_filtered_deck = []
            formated_name = self.deckname.replace("::", "_")
            deckname_trial = [
                    f"deck:\"*Optideck*{self.deckname}*\"",
                    f"deck:\"*{self.deckname}*Optideck*\"",
                    f"deck:\"*Optideck*{formated_name}*\"",
                    f"deck:\"*{formated_name}*Optideck*\"",
                    ]
            if self.filtered_deck_name_template is not None:
                deckname_trial.append(f"deck:\"*{self.filtered_deck_name_template}*\"")
            for deckname in deckname_trial:
                optideck_query = f"{deckname} is:due -rated:1"
                temp = self._call_anki(
                        action="findCards",
                        query=optideck_query)
                assert isinstance(temp, list), (
                    "Error when looking for filtered cards in rated query")
                in_filtered_deck.extend(temp)
                day_of_review.extend([1] * len(temp))
            whi(f"Found '{len(in_filtered_deck)}' cards in filtered decks "
                "created by AnnA that are considerated as 'rated'")
            previously_rated.extend(in_filtered_deck)
            assert len(day_of_review) == len(previously_rated), (
                "Invalid length of day_of_review")
            self.day_of_review = day_of_review
            return previously_rated

        rated_cards = []
        self.day_of_review = []
        if self.highjack_rated_query is not None:
            beep("Highjacking rated card list:")
            red("(This means the iterated fetcher will not be used!)")
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
            rated_cards = iterated_fetcher(query)
            whi(f"Found {len(rated_cards)} cards...\n")
        else:
            yel("Will not look for cards rated in past days.")
            rated_cards = []

        # this was removed because having short interval cards that
        # are quickly due will not be taken into account when taking
        # rated cards into account for the optimal order
        # remove overlap between due and rated cards
        # if rated_cards != []:
        #     temp = [x for x in rated_cards if x not in due_cards]
        #     diff = len(rated_cards) - len(temp)
        #     if diff != 0:
        #         yel("Removed overlap between rated cards and due cards: "
        #             f"{diff} cards removed. Keeping {len(temp)} cards.\n")
        #         rated_cards = temp

        self.due_cards = due_cards
        self.rated_cards = rated_cards

        # smooth exit if not enough cards were found
        if len(self.due_cards) < self.minimum_due:
            beep(f"Number of due cards is {len(self.due_cards)} which is "
                 f"less than threshold ({self.minimum_due}).\nStopping.")
            red("Not enough cards to review! Exiting.")
            self.not_enough_cards = True
            return False
        else:
            self.not_enough_cards = False

        combined_card_list = list(set(rated_cards + due_cards))

        # fetch relevant information of each cards
        list_cardInfo = []
        n = len(combined_card_list)
        yel(f"Asking Anki for information about {n} cards...")
        start = time.time()
        list_cardInfo.extend(
            self._fetch_cards(
                card_id=combined_card_list))
        whi(f"Got all infos in {int(time.time()-start)} seconds.\n")

        error_ids = []
        for i, card in enumerate(list_cardInfo):
            list_cardInfo[i]["fields"] = dict(
                (k.lower(), v)
                for k, v in list_cardInfo[i]["fields"].items())
            tags = []
            for t in list_cardInfo[i]["tags"]:
                skip_t = False
                for tag_ti in self.tags_to_ignore:
                    if tag_ti.match(t):
                        skip_t = True
                        break
                if not skip_t:
                    tags.append(t)
            list_cardInfo[i]["tags"] = " ".join(tags)
            if card["cardId"] in due_cards and card["cardId"] in rated_cards:
                list_cardInfo[i]["status"] = "due&rated"
            elif card["cardId"] in due_cards:
                list_cardInfo[i]["status"] = "due"
            elif card["cardId"] in rated_cards:
                list_cardInfo[i]["status"] = "rated"
            else:
                list_cardInfo[i]["status"] = "ERROR"
                error_ids.append(card)
        if error_ids:
            beep(f"Error processing card with IDs {','.join(error_ids)}")
            breakpoint()

        # check for duplicates
        if len(list_cardInfo) != len(list(set(combined_card_list))):
            beep("Error: duplicate cards in DataFrame!\nExiting.")
            breakpoint()

        if self.task not in ["just_add_KNN", "just_plot"]:
            # exclude from 'due' cards that have an 'odue' column != 0
            # which means that they are already in a filtered deck
            to_remove = []
            for i, card in enumerate(list_cardInfo):
                if int(card["odue"]) != 0:
                    if int(card["cardId"]) in self.due_cards:
                        to_remove.append(card)
            if to_remove:
                beep(f"Removing '{len(to_remove)}' cards from due that are "
                     " already in a filtered deck.")
                [list_cardInfo.remove(c) for c in to_remove]
                [self.due_cards.remove(int(c["cardId"])) for c in to_remove]
        assert list_cardInfo, "Empty list_cardInfo!"
        assert self.due_cards, "Empty self.due_cards!"

        # assemble cards into a dataframe
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

        # append bold and underlined text at the end
        # (to increase their importance without interfering with ngrams)
        bold = re.findall(r"<b>(.*?)</b>", text, flags=re.M | re.DOTALL)
        underlined = re.findall(r"<u>(.*?)</u>", text, flags=re.M | re.DOTALL)
        for dupli in bold + underlined:
            text += f" {dupli.strip()}"
        # as well as clozes
        cloze = re.findall(r"{{c\d+::(.*?)}}", text, flags=re.M | re.DOTALL)
        for dupli in cloze:
            text += f" {dupli}"

        # if blockquote or li or ul, mention that it's a list item
        # usually indicating a harder card
        # the ' <br>}}{{' is something specific to AnnA's author
        if re.search(r"</?li/?>|</?ul/?>| <br>}}{{", text, flags=re.M):
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
        text = text.replace("&gt;", "").replace("&lt;", "").replace(
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
                if re.search(regex, text):
                    if regex not in already_replaced:
                        # only replace once but still apply the overlapping
                        # acronyms if needed
                        text = re.sub(regex,
                                      lambda in_string:
                                      self._regexp_acronym_replacer(in_string,
                                                                    regex,
                                                                    new_value),
                                      text, count=0)

                    # if overlapping patterns, apply sequentially
                    if regex.pattern in self.acronyms_overlapping:
                        for regex2 in self.acronyms_overlapping[regex.pattern]:
                            new_value2 = self.acronym_dict[regex2]
                            text = re.sub(regex2,
                                          lambda in_string:
                                          self._regexp_acronym_replacer(
                                              in_string, regex2, new_value2),
                                          text, count=0)
                            already_replaced.append(regex2)

        # misc
        text = " ".join(text.split())  # multiple spaces

        # optionnal stemmer
        if self.vectorizer == "TFIDF":
            if self.TFIDF_stem is True:
                text = " ".join([self.ps.stem(x) for x in text.split()])

        # just in case, using ftfy
        text = ftfy.fix_text(
                text,
                unescape_html=True,
                uncurl_quotes=True,
                explain=False,
                )

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
                                   stopw_compiled, spacers_compiled,
                                   vectorizer):
            """
            threaded call to speed up execution
            """
            # skips using stopwords etc depending on the vectorizer
            if vectorizer == "embeddings":
                TFIDFmode=False
            elif vectorizer == "TFIDF":
                TFIDFmode=True
            else:
                raise ValueError("Invalid vectorizer")

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
                    if "Nearest_neighbors" in field_list:
                        field_list.remove("Nearest_neighbors")

                comb_text = ""
                for f in fields_to_keep:
                    try:
                        next_field = self.df.loc[index,
                                        "fields"][f.lower()]["value"].strip()
                        if TFIDFmode:
                            next_field = re.sub(
                                self.stopw_compiled,
                                " ",
                                next_field).strip()
                        if next_field != "":
                            if TFIDFmode:
                                comb_text += next_field + " <NEWFIELD> "
                            else:
                                #comb_text += f"\n{f.title()}: {next_field}"
                                comb_text += f"\n\n{next_field}"
                    except KeyError as e:
                        with lock:
                            to_notify.append(
                                f"Error when looking for field {e} in card "
                                f"{self.df.loc[index, 'modelName']} "
                                "identified as "
                                f"notetype {target_model}")
                comb_text = comb_text.strip()
                if TFIDFmode and comb_text.endswith("<NEWFIELD>"):
                    comb_text = comb_text[:-10].strip()

                # add tags to comb_text
                if self.append_tags:
                    tags = self.df.loc[index, "tags"].split(" ")
                    if tags and not TFIDFmode:
                        comb_text += "\nTags: "
                    for t in tags:
                        t = re.sub(
                            spacers_compiled,
                            " ",
                            t)
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
        spacers_compiled = re.compile("_|-|/|::")

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
                                                spacers_compiled,
                                                self.vectorizer,
                                                ),
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
                                                spacers_compiled,
                                                self.vectorizer,
                                                ),
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
        # slower if not done by large batch
        tqdm.pandas(desc="Formating text", smoothing=0, unit=" card",
                    file=self.t_strm)
        if self.vectorizer == "TFIDF":
            self.df["text"] = self.df["comb_text"].progress_apply(
                lambda x: " <NEWFIELD> ".join(
                    [
                        self._text_formatter(y) for y in x.split("<NEWFIELD>")
                        ]
                    ).strip()
                )
        elif self.vectorizer == "embeddings":
            self.df["text"] = self.df["comb_text"].progress_apply(self._text_formatter)
        else:
            raise ValueError("Invalid vectorizer")
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
                        skip_tag = False
                        for tag_ti in self.tags_to_ignore:
                            if tag_ti.match(t):
                                skip_tag = True
                                break
                        if skip_tag:
                            continue
                        t = re.sub(
                            spacers_compiled,
                            " ",
                            t)
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

    def _compute_projections(self):
        """
        Assigne vectors to each card's 'comb_text', using the vectorizer.
        """
        df = self.df

        # if self.low_power_mode:
        #     binary_mode = True
        # else:
        binary_mode = False

        def init_TFIDF_vectorizer():
            """used to make sure the same statement is used to create
            the vectorizer"""
            return TfidfVectorizer(strip_accents="ascii",
                                   lowercase=False,
                                   tokenizer=self.tokenize,
                                   token_pattern=None,
                                   stop_words=None,
                                   ngram_range=(1, 3),
                                   norm="l2",
                                   smooth_idf=False,
                                   sublinear_tf=True,
                                   max_features=1000,  # if more than
                                   # dim_limit, SVD will be used to reduce dimension
                                   # to dim_limit prior to running UMAP
                                   binary=binary_mode,
                                   # max_df=0.5,  # ignore words present in
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
                    beep("Ankipandas seems to have "
                         "found a collection in "
                         "the trash folder. If that is not your intention "
                         "cancel now. Waiting 10s for you to see this "
                         "message before proceeding.")
                    time.sleep(1)
                Path.mkdir(Path(".cache"), exist_ok=True)
                name = f"{self.profile_name}_{self.deckname}".replace(" ", "_")
                temp_db = shutil.copy(
                    original_db, f"./.cache/{name.replace('/', '_')}")
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
                                    t.split("::")))
                                processed += " " + t
                    corpus.append(processed)

                vectorizer = init_TFIDF_vectorizer()
                vectorizer.fit(tqdm(corpus + df["text"].tolist(),
                                    desc="Vectorizing whole deck",
                                    file=self.t_strm))
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
            if self.vectorizer == "TFIDF":
                vectorizer = init_TFIDF_vectorizer()
                t_vec = vectorizer.fit_transform(tqdm(df["text"],
                                                      desc=(
                  "Vectorizing using TFIDF"),
                                                      file=self.t_strm))
            elif self.vectorizer == "embeddings":

                def sencoder(sentences):
                    return model.encode(
                            sentences=sentences,
                            show_progress_bar=True if len(sentences) > 1 else False,
                            output_value="sentence_embedding",
                            convert_to_numpy=True,
                            normalize_embeddings=False,
                            )

                def hasher(text):
                    return hashlib.sha256(text[:10_000].encode()).hexdigest()[:10]

                def retrieve_cache(path):
                    with open(path, "rb") as f:
                        return pickle.load(f)

                def add_to_cache(row, path):
                    with open(path, "wb") as f:
                        return pickle.dump(row, f)

                memhasher = self.memoize(hasher)
                memretr = self.memoize(retrieve_cache)

                whi("Loading sentence transformer model")
                model = SentenceTransformer(self.embed_model)

                # create empty numpy array
                t_vec = np.zeros(
                        (len(df.index), max(sencoder(["test"]).shape)
                            ), dtype=float)

                # check existence of embeddings cache
                vec_cache = Path(".cache")
                vec_cache.mkdir(exist_ok=True)
                vec_cache = vec_cache / "embeddings_cache"
                vec_cache.mkdir(exist_ok=True)
                vec_cache = vec_cache / self.embed_model
                vec_cache.mkdir(exist_ok=True)

                # get what is in cache in the form "NID_FINGERPRINT.pickle"
                filenames = set(f.name for f in vec_cache.iterdir())
                whi(f"Number of entries in cache: {len(filenames)}")
                cache_nid_fing = {}
                for f in filenames:
                    f = f.replace(".pickle", "")
                    nid, fingerprint = f.split("_")
                    nid = int(nid)
                    if nid not in cache_nid_fing:
                        cache_nid_fing[nid] = [fingerprint]
                    else:
                        cache_nid_fing[nid].append(fingerprint)

                # compute fingerprint of all note content
                tqdm.pandas(
                        desc="Computing note content fingerprint",
                        smoothing=0,
                        unit=" card",
                        file=self.t_strm)
                df["sha256"] = df["text"].progress_apply(memhasher)

                # load row of t_vec if cache present
                for i, ind in enumerate(tqdm(df.index, desc="Loading from cache", file=self.t_strm)):
                    fingerprint = df.loc[ind, "sha256"]
                    nid = int(df.loc[ind, "note"])
                    if nid in cache_nid_fing:
                        if fingerprint in cache_nid_fing[nid]:
                            filename = f"{nid}_{fingerprint}.pickle"
                            t_vec[i, :] = memretr(str(vec_cache / filename))

                # get embeddings for missing rows
                done_rows = np.where(~np.isclose(np.sum(t_vec, axis=1), 0.0))[0]
                missing_rows = np.where(np.isclose(np.sum(t_vec, axis=1), 0.0))[0]
                missing_cid = [df.index[i] for i in missing_rows]

                yel(f"Rows not found in cache: '{len(missing_cid)}'")
                yel(f"Rows found in cache: '{len(done_rows)}'")

                if missing_cid:
                    red("Computing embeddings of uncached notes")
                    t_vec[missing_rows, :] = sencoder(df.loc[missing_cid, "text"].tolist())

                    whi("Adding to cache the newly computed embeddings")
                    for i, ind in enumerate(
                            tqdm(
                                missing_cid,
                                desc="adding to cache",
                                unit="note",
                                file=self.t_strm
                                )
                            ):
                        nid = df.loc[ind, "note"]
                        fingerprint = df.loc[ind, "sha256"]
                        filename = f"{nid}_{fingerprint}.pickle"
                        add_to_cache(t_vec[missing_rows[i], :], str(vec_cache / filename))

                assert not np.isclose(t_vec.sum(), 0), "t_vec is still 0"
                assert t_vec.shape[0] == len(np.where(~np.isclose(np.sum(t_vec, axis=1), 0.0))[0]), "t_vec invalid"

                whi("Normalizing embeddings")
                t_vec = normalize(t_vec, norm="l2", axis=1, copy=True)

            else:
                 raise ValueError("Invalid vectorizer value")

        self.vectors_beforeUMAP = t_vec

        # number of neighbors to consider for umap:
        # for 100 cards or less: use 15 n_neighbors
        # for more than 1000 use 100
        n_n = int(len(self.df.index) * (100-15) / (1000 - 100))
        n_n = min(max(n_n, 15), 100)  # keep it between 15 and 100
        umap_kwargs = {"n_jobs": -1,
                       "verbose": 1,
                       "metric": "cosine",
                       # the initial position is the 2D PCA
                       "init": PCA(
                           n_components=2,
                           random_state=42).fit_transform(t_vec),
                       "transform_seed": 42,
                       "random_state": 42, # turns off some multithreading section of the code
                       "n_neighbors":  n_n,
                       "min_dist": 0.01,
                       "low_memory":  False,
                       "densmap": True,  # try to preserve local density
                       "n_epochs": 1000,  # None will automatically adjust
                       "target_metric": "l2",  # not sure what it does
                       "unique": True,
                       }

        if self.ndim_reduc is None:
            self.vectors = t_vec
        else:
            # AnnA will use UMAP to reduce the dimensions
            # Previously TruncatedSVD was used but it kept too many
            # dimensions so ended up in the curse of dimensionnality

            # reduce dimensions before UMAP if too many dimensions
            # if set to 2, will skip UMAP
            dim_limit = 100
            if t_vec.shape[1] > dim_limit:
                try:
                    yel(f"Vectorized text of shape {t_vec.shape}, dimensions above "
                        f"{dim_limit} so using SVD or PCA first to keep only "
                        f"{dim_limit} dimensions.")
                    if self.vectorizer == "TFIDF":
                        m_rank = np.linalg.matrix_rank(t_vec)
                        dimred = TruncatedSVD(
                                n_components=dim_limit,
                                random_state=42,
                                n_oversamples=max(
                                    10, 2 * m_rank - dim_limit
                                    )
                                )

                    elif self.vectorizer == "embeddings":
                        dimred = PCA(
                                n_components=dim_limit,
                                random_state=42,
                                )
                    else:
                        raise ValueError("Invalid vectorizer")

                    t_vec = dimred.fit_transform(t_vec)
                    evr = round(sum(dimred.explained_variance_ratio_) * 100, 1)
                    whi(f"Done, explained variance ratio: {evr}%. New shape: {t_vec.shape}")
                except Exception as err:
                    beep(f"Error when using SVD/PCA to reduce to {dim_limit} "
                         f"dims: '{err}'.\rTrying to continue with "
                         f"UMAP nonetheless.\rData shape: {t_vec.shape}")
                    red(traceback.format_exc())

            self.vectors_beforeUMAP = t_vec
            target_dim = 2
            if target_dim < t_vec.shape[1]:
                whi(f"Using UMAP to reduce to {target_dim} dimensions")
                try:
                    umap_kwargs["n_components"] = target_dim
                    U = umap.umap_.UMAP(**umap_kwargs)
                    t_red = U.fit_transform(t_vec)
                    self.vectors = t_red
                except Exception as err:
                    beep(f"Error when using UMAP to reduce to {target_dim} "
                         f"dims: '{err}'.\rTrying to continue "
                         f"nonetheless.\rData shape: {t_vec.shape}")
                    red(traceback.format_exc())
                    self.vectors = t_vec
            else:
                whi("Not using UMAP to reduce dimensions as the number of "
                    f"dim is {t_vec.shape[1]} which is not higher "
                    f"than {target_dim}")
                self.vectors = t_vec

        if self.plot_2D_embeddings:
            try:
                yel("Computing 2D embeddings for the plot using UMAP...")
                if self.vectors.shape[1] == 2:
                    whi("Reusing previous dimension reduction for embeddings.")
                    self.vectors2D = self.vectors
                else:  # compute embeddings
                    whi("Computing 2D UMAP for embeddings.")
                    self.vectors2D = self.vectors
                    umap_kwargs["n_components"] = 2
                    U = umap.umap_.UMAP(**umap_kwargs)
                    self.vectors2D = U.fit_transform(t_vec)
            except Exception as err:
                beep(f"Error when computing 2D embeddings: '{err}'")
                red(traceback.format_exc())

        self.df = df
        return True

    def _compute_distance_matrix(self):
        """
        compute distance matrix : a huge matrix containing the
            cosine distance between the vectors of each cards.
        * scikit learn allows a parallelized computation
        """
        df = self.df

        yel("\nComputing distance matrix on all available cores"
            "...")
        if self.dist_metric == "rbf":
            sig = np.mean(np.std(self.vectors, axis=1))
            self.df_dist = pd.DataFrame(columns=df.index,
                                        index=df.index,
                                        data=pairwise_kernels(
                                            self.vectors,
                                            n_jobs=-1,
                                            metric="rbf",
                                            gamma=1/(2*sig),
                                            ))
            # turn the similarity into a distance
            # apply log to hopefully reduce the spread
            tqdm.pandas(desc="Applying log", smoothing=0, unit=" card",
                        file=self.t_strm)
            self.df_dist = self.df_dist.progress_apply(lambda x: np.log(1+x))
            max_val = np.max(self.df_dist.values.ravel())
            self.df_dist /= -max_val  # normalize values then make negative
            # add 1 to get: "dist = 1 - similarity"
            self.df_dist += 1
        elif self.dist_metric in ["cosine", "euclidean"]:
            self.df_dist = pd.DataFrame(columns=df.index,
                                        index=df.index,
                                        data=pairwise_distances(
                                            self.vectors,
                                            n_jobs=-1,
                                            metric=self.dist_metric,
                                            ))
        else:
            raise ValueError("Invalid 'dist_metric' value")

        self.df_dist = self.df_dist.sort_index()

        assert np.isclose(a=(self.df_dist.values - self.df_dist.values.T),
                          b=0,
                          atol=1e-06).all(), (
                "Non symetric distance matrix")

        # make it very symetric for good measure
        self.df_dist += self.df_dist.T
        self.df_dist /= 2
        if (np.diag(self.df_dist) != 0).all():
            red("'Forced symetrisation' of the distance matrix resulted in "
                "non zero diagonal elements, setting them manually to 0.")
            self.df_dist.values[np.diag_indices(self.df_dist.shape[0])] = 0
        # add a second check just in case
        assert np.isclose(a=(self.df_dist.values - self.df_dist.values.T),
                          b=0).all(), (
                "Non symetric distance matrix (#2 check)")

        # make sure the distances are positive otherwise it might reverse
        # the sorting logic for the negative values (i.e. favoring similar
        # cards)
        assert (self.df_dist.values.ravel() < 0).sum() == 0, (
            "Negative values in the distance matrix!")

        # make sure that the maximum distance is 1
        max_val = np.max(self.df_dist.values.ravel())
        if not np.isclose(max_val, 1):
            whi(f"Maximum value is {max_val}, scaling the distance matrix to "
                "have 1 as maximum value.")
            self.df_dist /= max_val

        yel("Computing mean and std of distance...\n(excluding diagonal)")
        # ignore the diagonal of the distance matrix to get a sensible mean
        # and std value:
        up_triangular = np.triu_indices(self.df_dist.shape[0], 1)
        mean_dist = np.mean(self.df_dist.values[up_triangular].ravel())
        std_dist = np.std(self.df_dist.values[up_triangular].ravel())
        whi(f"Mean distance: {mean_dist}\nMean std: {std_dist}\n")

        self.mean_dist = mean_dist
        self.std_dist = std_dist

        self._print_similar()
        return True

    def _add_neighbors_to_notes(self):
        """
        if the model card contains the field 'Nearest_neighbors', replace its
        content by a query that can be used to find the neighbor of the
        given note.
        """
        if not (self.add_KNN_to_field or self.task == "just_add_KNN"):
            whi("Not adding KNN to note field because of arguments.")
            return
        red("Adding the list of neighbors to each note.")
        nid_content = {}
        nb_of_nn = []

        # as some of the intermediary value for this iteration can be needed
        # if 2D plotting, they are stored as attributes just in case
        self.n_n = min(50, max(5, self.df_dist.shape[0] // 100))
        self.max_radius = self.std_dist * 0.05
        yel(f"Keeping '{self.n_n}' nearest neighbors (that are within "
            f"'{self.max_radius}')")
        if self.plot_2D_embeddings:
            self.nbrs_cache = {}
        try:
            for cardId in tqdm(
                    self.df.index,
                    desc="Collecting neighbors of notes",
                    file=self.t_strm,
                    mininterval=5,
                    unit="card"):
                if "Nearest_neighbors".lower() not in self.df.loc[
                        cardId, "fields"].keys():
                    continue

                noteId = int(self.df.loc[cardId, "note"])
                if noteId in nid_content:
                    continue  # skipped because a card of this note was
                    # already processed

                nbrs_cid = self.df_dist.loc[cardId, :].nsmallest(self.n_n)
                nbrs_cid = nbrs_cid[nbrs_cid <= self.max_radius]
                assert len(nbrs_cid) > 0, "Invalid nbrs_cid value"
                nbrs_cid = nbrs_cid.sort_values(ascending=True)

                # get nid instead of indices
                nbrs_nid = [self.df.loc[ind, "note"]
                            for ind in nbrs_cid.index]

                if self.plot_2D_embeddings:  # save for later if needed
                    self.nbrs_cache[noteId] = {
                            "nbrs_cid": nbrs_cid,
                            "nbrs_nid": nbrs_nid,
                            }

                # create the string that will be put to the anki note
                nid_content[noteId] = "nid:" + ",".join(
                        [str(x) for x in nbrs_nid]
                        )
                nb_of_nn.append(len(nbrs_cid))
            whi("Sending new field value to Anki...")
            self._call_anki(
                    action="update_KNN_field",
                    nid_content=nid_content,
                    )
            yel("Finished adding neighbors to notes.")
        except Exception as err:
            beep(f"Error when adding neighbor list to notes: '{err}'")

        if nb_of_nn:
            yel("Number of neighbors on average:")
            whi(str(pd.DataFrame(nb_of_nn, columns=["Neighbors"]).describe()))

    def _print_similar(self):
        """ finds two cards deemed very similar (but not equal) and print
        them. This is used to make sure that the system is working correctly.
        Given that this takes time, a timeout has been implemented.
        """
        self.timeout_in_minutes = 1
        signal.signal(signal.SIGALRM, self.time_watcher)
        signal.alarm(int(self.timeout_in_minutes * 60))

        try:
            max_length = 200
            up_triangular = np.triu_indices(self.df_dist.shape[0], 1)
            pd.set_option('display.max_colwidth', 180)

            red("\nPrinting the most semantically distant cards:")
            highest_value = np.max(self.df_dist.values[up_triangular].ravel())
            coord_max = np.where(self.df_dist == highest_value)
            one = self.df.iloc[coord_max[0][0]].text[:max_length]
            two = self.df.iloc[coord_max[1][0]].text[:max_length]
            yel(f"* {one}...")
            yel(f"* {two}...")

            red("\nPrinting the most semantically close but distinct similar "
                "cards:")
            # the diagonal is the minimum of distance so we are looking for
            # the distance that is just higher
            q_diagonal = (self.df_dist.shape[0] + 2) / (
                     self.df_dist.shape[0] ** 2 / 2)
            quantile_limit = np.quantile(
                    self.df_dist.values[up_triangular].ravel(), q_diagonal)
            lowest_non_zero_value = np.amin(
                    self.df_dist.values[up_triangular],
                    where=self.df_dist.values[up_triangular] > quantile_limit,
                    initial=highest_value)
            coord_min = np.where(self.df_dist == lowest_non_zero_value)
            one = self.df.iloc[coord_min[0][0]].text[:max_length]
            two = self.df.iloc[coord_min[1][0]].text[:max_length]
            yel(f"* {one}...")
            yel(f"* {two}...")
            whi(f"    (distance: {lowest_non_zero_value})")
            whi(f"    (quantile limit: {quantile_limit})")
            whi(f"    (q diagonal: {q_diagonal})")

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
            one = self.df.iloc[coord_med[0][0]].text[:max_length]
            two = self.df.iloc[coord_med[1][0]].text[:max_length]
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

    def _compute_optimized_queue(self):
        """
        Long function that computes the new queue order.

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
            the docstring of function 'combine_arrays'.
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
        rated_excl_due = [i for i in rated if i not in due]
        w1 = self.score_adjustment_factor[0]
        w2 = self.score_adjustment_factor[1] / self.mean_dist
        if self.enable_fuzz:
            w3 = (w1 + w2) / 2 * self.mean_dist / 10
        else:
            w3 = 0
        self.urgent_dues = []

        # hardcoded settings
        display_stats = True if not self.low_power_mode else False

        # setting interval to correct value for learning and relearnings:
        steps_L = [x / 1440 for x in self.deck_config["new"]["delays"]]
        steps_RL = [x / 1440 for x in self.deck_config["lapse"]["delays"]]
        for i in df.index:
            if df.loc[i, "type"] == 1:  # learning
                step_L_index = int(str(df.loc[i, "left"])[-3:])-1
                try:
                    df.at[i, "interval"] = steps_L[step_L_index]
                except Exception as e:
                    whi(f"Invalid learning step, card was recently moved from another deck? cid: {i}; '{e}'")
                    df.at[i, "interval"] = steps_L[0]

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
        # skewing the dataset distribution (but
        # excludes rated that are also due):
        df.loc[rated_excl_due, "interval"] = np.nan
        df.loc[rated_excl_due, "due"] = np.nan
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
            if df.loc[due, "ref"].max() != 0:
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
                raise Exception("Correction factor above 1")

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
            urg_overdue = [due[i] for i in np.intersect1d(mask, mask2).tolist()]
            yel(f"* Found '{len(urg_overdue)}' cards that are more than '{int(p*100)}%' overdue.")

            ivl_limit = 14  # all cards with interval <= ivl_limit are deemed urgent
            urg_ivl = [ind for ind in due if df.loc[ind, "interval"] <= ivl_limit]
            yel(f"* Found '{len(urg_ivl)}' cards that are due with 'interval <= {ivl_limit} days'.")

            ease_limit = 1750  # all cards with lower ease are deemed urgent
            urg_lowease = [ind for ind in due if df.loc[ind, "factor"] <= ease_limit]
            yel(f"* Found '{len(urg_lowease)}' cards that are due with 'ease <= {ease_limit//10}%'.")

            urgent_dues = urg_overdue + urg_ivl + urg_lowease
            urgent_dues = list(set(urgent_dues))
            yel(f"=> In total, found {len(urgent_dues)} cards to boost.")

            # compute the score by which to boost the ro
            urgent_factor = 1 / ro[urgent_dues]
            urgent_factor -= urgent_factor.min()
            if urgent_factor.max() != 0:  # otherwise fails if ro was constant
                urgent_factor /= urgent_factor.max()
            assert (urgent_factor >= 0).all(), "Negative urgent factor"

            # reduce the increase of ro as a very high ro is not important
            while ro.max() > 1.5:
                whi("(Smoothing relative overdueness)")
                ro[ro > 1] = 1 + np.log(ro[ro > 1])

            # minmax scaling of ro
            ro -= ro.min()
            if ro.max() != 0:
                ro /= ro.max()
            ro += 0.001

            if boost and (self.score_adjustment_factor[0] == 0):
                yel("Will not boost cards because the relevant score "
                    "adjustment factor is 0")
            elif boost and urgent_dues:
                for ind in urgent_dues:
                    ro[ind] += (-0.5 * self.score_adjustment_factor[0] - urgent_factor[ind])
                if ro.min() <= 0:
                    ro += abs(ro.min()) + 0.001
                assert ro.min() > 0, "Negative value in relative overdueness"
                ro /= ro.max()
                whi("Boosted urgent_dues cards to increase chances they are reviewed today.")

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
                    today_date = f"{d.day:02d}/{d.month:02d}"
                    for reason, urgents in {"OD": urg_overdue,
                                            "LI": urg_ivl,
                                            "LE": urg_lowease}.items():
                        new_tag = f"AnnA::UR::{today_date}::{reason}"
                        notes = []
                        for card in urgents:
                            notes.append(int(self.df.loc[card, "note"]))
                        notes = list(set(notes))  # remove duplicates
                        try:
                            self._call_anki(action="addTags",
                                            notes=notes, tags=new_tag)
                            red(f"Appended tags {new_tag} to urgent reviews")
                        except Exception as e:
                            beep(f"Error adding tags to urgent '{reason}' "
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
            del weights  # not needed and removed to avoid confusion

        assert len([x for x in rated if "rated" not in df.loc[x, "status"]]
                   ) == 0, "all rated cards are not marked as rated"
        if self.rated_last_X_days is not None:
            red("\nCards identified as rated in the past "
                f"{self.rated_last_X_days} days: {len(rated)}")

        # contain the index of the cards that will be use when
        # computing optimal order
        indQUEUE = rated[:]
        indTODO = due[:]
        assert len(
                [x
                 for x in indTODO
                 if "due" not in self.df.loc[x, "status"]]) == 0, (
                         "Cards in indTODO are not actually due!")
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
                val1 = pd.DataFrame(data=self.df_dist.values.ravel(),
                                    columns=['distance matrix']).describe(
                                            include='all')
                val2 = pd.DataFrame(data=w2*self.df_dist.values.ravel(),
                                    columns=['distance matrix']).describe(
                                            include='all')
                whi(f"Non adjusted distance matrix: {val1}\n")
                whi(f"Weight adjusted distance matrix: {val2}\n\n")
            except Exception as e:
                beep(f"Exception: {e}")
            pd.reset_option('display.float_format')
        else:
            whi("Not displaying stats of the data because 'low_power_mode' "
                "if True.")

        # checking that there are no negative ref values
        assert (self.df.loc[due, "ref"].ravel() < 0).sum() == 0, (
            "Negative values in the reference score!")

        # final check before computing optimal order:
        for x in ["interval", "ref", "due"]:
            assert np.sum(np.isnan(df.loc[rated_excl_due, x].values)) == len(rated_excl_due), (
                    f"invalid treatment of rated cards, column : {x}")
            assert np.sum(np.isnan(df.loc[due, x].values)) == 0, (
                    f"invalid treatment of due cards, column : {x}")

        # check that we have the right number of "review date" per "rated card"
        assert len(self.day_of_review) == len(self.rated_cards), (
            f"Incompatible length of day_of_review")
        # format it in the right format score with formula:
        #   f(day) = log( 1 + day / 2 )
        #  Example values:
        #  day since review |  distance multiplying factor
        #               01  |  1.0
        #               02  |  1.28
        #               03  |  1.51
        #               04  |  1.69
        #               05  |  1.84
        #               06  |  1.98
        #               07  |  2.09
        #               08  |  2.20
        #               09  |  2.29
        #               10  |  2.38
        #               11  |  2.46
        #               12  |  2.54
        #               13  |  2.60
        #               14  |  2.67
        #               15  |  2.73
        #               16  |  2.79
        #               17  |  2.84
        #               18  |  2.89
        #               19  |  2.94

        #   (value are then translated to take day 1 as a reference, meaning
        #    the distance for a card rated today won't change)
        if self.day_of_review:
            self.temporal_discounting = pd.DataFrame(
                    index=self.rated_cards,
                    data=self.day_of_review,
                    dtype=float
                    )
            # applying scoring formula
            self.temporal_discounting = self.temporal_discounting.apply(
                    lambda x: np.log(1 + x / 2)
                    )
            self.temporal_discounting -= self.temporal_discounting.min()
            self.temporal_discounting += 1 # translation to start at 1
            # show statistics to user
            pd.set_option('display.float_format', lambda x: '%.5f' % x)
            whi("\nTime score stats of rated cards:")
            whi(f"{self.temporal_discounting.describe()}\n")
            pd.reset_option('display.float_format')
            # check correct scaling
            assert self.temporal_discounting.min().squeeze() == 1, (
                "Incorrect scaling of self.temporal_discounting")

        def combine_arrays(indTODO, indQUEUE, task):
            """
            'dist_score' represents:
                * columns : the cards of indTODO
                * rows : the cards of indQUEUE
                * the content of each cell is the similarity between them
                    (lower value means very similar)
            Hence, for a given dist_score:
            * if a cell of np.min is high, then the corresponding card of
                indTODO is not similar to any card of indQUEUE (i.e. its
                closest card in indQUEUE is quite different). This card is a
                good candidate to add to indQEUE.
            * if a cell of np.mean is high, then the corresponding card of
                indTODO is different from most cards of indQUEUE (i.e. it is
                quite different from most cards of indQUEUE). This card is
                a good candidate to add to indQUEUE (same for np.median)
            * Naturally, np.min is given more importance than np.mean

            Best candidates are cards with high combine_arrays output.
            The outut is substracted to the 'ref' of each indTODO card.

            Hence, at each turn, the card from indTODO with the lowest
                'w1*ref - w2*combine_arrays' is removed from indTODO and added
                to indQUEUE.

            The content of 'queue' is the list of card_id in best review order.
            """
            dist_2d = self.df_dist.loc[indTODO, indQUEUE].copy()
            assert (dist_2d >= 0).values.all(), "negative values in dist_2d #1"

            if task == "create_queue" and self.day_of_review:
                # multiply dist score of queue based on how recent was the review
                try:
                    itsct = np.intersect1d(
                            indQUEUE,
                            self.temporal_discounting.index.tolist(),
                            return_indices=False,
                            )
                    dist_2d.loc[:, itsct[0]] *= self.temporal_discounting.loc[itsct[0]].values.squeeze()
                except Exception as err:
                    beep(f"Error in temporal discounting: {err}")

            assert (dist_2d >= 0).values.all(), "negative values in dist_2d #2"
            # the minimum distance is what's most important in the scoring
            min_dist = np.min(dist_2d.values, axis=1)
            min_dist -= min_dist.min()
            if min_dist.max() != 0:
                min_dist /= min_dist.max()

            # # minmax scaling by column before taking mean and 25th quantile value
            # (min to be taken before rescaling because otherwise all min are 0)
            # min_dist_cols = np.min(dist_2d.values, axis=1)
            # dist_2d -= min_dist_cols[:, np.newaxis]
            # max_dist_cols = np.max(dist_2d.values, axis=1)
            # if (max_dist_cols.ravel() != 0).all():  # avoid null division
            #     dist_2d /= max_dist_cols[:, np.newaxis]

            # scaling mean and 25th quantile so they are between 0 and 1
            mean_dist = np.mean(dist_2d.values, axis=1)
            mean_dist -= mean_dist.min()
            if mean_dist.max() != 0:
                mean_dist /= mean_dist.max()

            quant_dist = np.quantile(dist_2d.values, q=0.25, axis=1)
            quant_dist -= quant_dist.min()
            if quant_dist.max() != 0:
                quant_dist /= quant_dist.max()

            # weighted agregation of min, mean and 25th quantile into a 1d array
            dist_1d = 0.9 * min_dist + 0.02 * mean_dist + 0.08 * quant_dist

            # minmax scaling this 1d array
            dist_1d -= dist_1d.min()
            max_dist = np.max(dist_1d).squeeze()
            if max_dist > 0:  # in case of divide by 0 error
                dist_1d /= max_dist

            # if self.log_level >= 2:
            #    avg = np.mean(dist_1d) * self.score_adjustment_factor[1]
            #    tqdm.write(f"DIST_SCORE: {avg:02f}")

            if task == "create_queue":
                ref_score = df.loc[indTODO, "ref"].values.copy()
                # minmax scaling what is left of the ref_score
                ref_score -= ref_score.min()
                max_score = np.max(ref_score).squeeze()
                if max_score > 0:
                    ref_score /= max_score

                score_array = w1 * ref_score - w2 * dist_1d + w3 * np.random.rand(1, len(indTODO))
            elif task == "resort":
                # simply resorting the final queue, only using dist_1d
                # (the sign is indeed positive and is taken into account later)
                score_array = w2 * dist_1d
            else:
                raise ValueError(f"Invalid value of 'task': '{task}'")

            return score_array


        with tqdm(desc="Computing optimal review order",
                  unit=" card",
                  initial=len(rated),
                  smoothing=0,
                  file=self.t_strm,
                  total=queue_size_goal + len(rated)) as pbar:
            while len(queue) < queue_size_goal:
                queue.append(
                        indTODO[
                            combine_arrays(indTODO, indQUEUE, "create_queue"
                                ).argmin()
                            ]
                        )
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
                    "to maximize/minimize distance...")
                whi("But starts by the cards needing to be boosted)")
                new_queue = [queue[0]]
                to_process = [q for q in queue[1:]]

                # tell proportion to the user
                n = len([q for q in self.urgent_dues if q in queue])
                if n > 0:
                    proportion = int(n / len(queue) * 100)
                    yel(f"The filtered deck will contain {proportion}% of "
                        "boosted cards.")

                if self.resort_split:
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
                else:
                    # don't split boosted cards from non boosted cards
                    self.df.loc[to_process, "ref_filtered"] = 0

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
                            combine_arrays(to_process, new_queue, "resort")
                            )
                    if self.resort_by_dist == "farther":
                        new_queue.append(to_process.pop(score.argmin()))
                    elif self.resort_by_dist == "closer":
                        new_queue.append(to_process.pop(score.argmax()))
                    else:
                        raise ValueError("Invalid value for 'resort_by_dist'")
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
        if self.low_power_mode:
            red("Not printing acronyms because low_power_mode is set to "
                "'True'")
            return
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
                if re.search(compiled, acr) is not None:
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

        if self.filtered_deck_name_template is not None:
            filtered_deck_name = str(
                self.filtered_deck_name_template + f" - {self.deckname}")
            filtered_deck_name = filtered_deck_name.replace("::", "_")
        else:
            if self.filtered_deck_at_top_level:
                top_lvl_deckname = self.deckname.split("::")[-1]
                filtered_deck_name = f"AnnA Optideck - {top_lvl_deckname}"
            else:
                filtered_deck_name = f"{self.deckname} - AnnA Optideck"
        self.filtered_deck_name = filtered_deck_name

        while filtered_deck_name in self._call_anki(action="deckNames"):
            beep(f"\nFound existing filtered "
                 f"deck: {filtered_deck_name} "
                 "You have to delete it manually, the cards will be "
                 "returned to their original deck.")
            yel("Syncing...")
            sync_output = self._call_anki(action="sync")
            assert sync_output is None or sync_output == "None", (
                "Error during sync?: '{sync_output}'")
            time.sleep(1)  # wait for sync to finish, just in case
            whi("Done syncing!")
            input("Done? (will trigger another sync) >")

        whi("Creating deck containing the cards to review: "
            f"{filtered_deck_name}")
        if self.filtered_deck_by_batch and (
                len(self.opti_rev_order) > self.filtered_deck_batch_size):
            yel("Creating batches of filtered decks...")
            batchsize = self.filtered_deck_batch_size
            # construct the list of batches to do:
            toempty = self.opti_rev_order.copy()

            # instead of having the last batch being of size different
            # than batchsize, I prefer it being the first batch.
            remainder = len(toempty) % batchsize
            batches = []
            if remainder != 0:
                batches = [toempty[:remainder]]
                toempty = toempty[remainder:]
            # (if remainder is 0, batches is empty and toempty is full)
            assert (
                    remainder == 0 and len(toempty) == len(self.opti_rev_order)
                    ) or (
                            remainder != 0 and len(toempty) != len(self.opti_rev_order)
                            ), "invalid remainder handling"

            while toempty:
                batches.append([])
                while len(batches[-1]) < batchsize:
                    batches[-1].append(toempty.pop(0))
                assert len(batches[-1]) == batchsize, "invalid length of a batch"

            assert len(toempty) == 0
            assert list(set([len(x) for x in batches[1:]]))[0] == batchsize, (
                "invalid batches construction #1")
            assert len(batches[0]) in [batchsize, remainder], (
                "invalid batches construction #2")

            for cnt, batch_cards in enumerate(batches):
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

    def time_watcher(self, signum, frame):
        """
        raise a TimeoutError if plotting or searching similar cards takes
        too long"""
        raise TimeoutError(f"Timed out after {self.timeout_in_minutes} "
                           "minutes.")

    def _compute_plots(self):
        """
        Create a 2D plot of the deck.
        """
        red("Creating 2D plots...")
        if self.not_enough_cards:
            return
        assert self.plot_2D_embeddings, "invalid arguments!"
        assert hasattr(self, "vectors2D"), "2D embeddings could not be found!"

        self.timeout_in_minutes = 15
        # add a timeout to make sure it doesn't get stuck
        signal.signal(signal.SIGALRM, self.time_watcher)
        signal.alarm(int(self.timeout_in_minutes * 60))
        # this timeout is replaced by a shorter timeout when opening
        # the browser then resumed
        self.timeout_start_time = time.time()

        # bertopic plots
        docs = self.df["text"].tolist()
        topic_model = BERTopic(
                verbose=True,
                top_n_words=10,
                nr_topics=100,
                vectorizer_model=CountVectorizer(
                    stop_words=self.stops + [f"c{n}" for n in range(10)],
                    ngram_range=(1, 1),
                    ),
                hdbscan_model=hdbscan.HDBSCAN(
                    min_cluster_size=10,
                    min_samples=1,
                    ),
                # hdbscan_model=cluster.KMeans(n_clusters=min(len(self.df.index)//10, 100)),
                ).fit(
                        documents=docs,
                        embeddings=self.vectors_beforeUMAP,
                        )
        hierarchical_topics = topic_model.hierarchical_topics(
            docs=docs,
            )
        fig = topic_model.visualize_hierarchical_documents(
                docs=docs,
                hierarchical_topics=hierarchical_topics,
                reduced_embeddings=self.vectors2D,
                nr_levels=min(20, len(hierarchical_topics) - 1),
                # level_scale="log",
                title=f"{self.deckname} - embeddings",
                hide_annotations=True,
                hide_document_hover=False,
                )
        saved_plot = f"{self.plot_dir}/{self.deckname} - embeddings.html"
        whi(f"Saving plot to {saved_plot}")
        offpy(fig,
              filename=saved_plot,
              auto_open=False,
              show_link=False,
              validate=True,
              output_type="file",
              )
        try:
            # replacing timeout by a 5s one then resuming the previous one
            def f_browser_timeout(signum, frame):
                raise TimeoutError
            signal.alarm(0)
            signal.signal(signal.SIGALRM, f_browser_timeout)
            signal.alarm(5)
            whi(f"Trying to open {saved_plot} in the browser...")
            saved_plot_fp = str(Path(saved_plot).absolute()).replace("\\", "")
            if "genericbrowser" in str(webbrowser.get()).lower():
                # if AnnA is launched using SSH, the webbrowser will
                # possibly be in the console and can stop the script
                # while the browser is not closed.
                whi("No GUI browser detected, maybe you're in an SSH console? "
                    "\nFalling back to using linux shell to open firefox")
                subprocess.check_output(
                        shlex.split(f"env DISPLAY=:0 firefox '{saved_plot_fp}'"),
                        shell=False,
                        )
            else:
                whi("Opening browser.")
                webbrowser.open(saved_plot_fp)
        except TimeoutError as e:
            elapsed = self.timeout_in_minutes * 60 - (time.time() - self.timeout_start_time)
            if elapsed <= 1:  # rare case when the timeout is for the overall
                # plotting code and not just to open the browser
                raise
            else:
                pass  # the function got stuck when openning the browser, ignore
        except Exception as e:
            beep(f"Exception when openning file: '{e}'")
        signal.alarm(0)
        signal.signal(signal.SIGALRM, self.time_watcher)
        signal.alarm(int(self.timeout_in_minutes * 60 - (time.time() - self.timeout_start_time)))



        red(f"Done with BERTopic")

        # older code that does not use bertopic
        # n_n_plot = 10  # number of closest neighbors to take into account
        # # for the spring layout computation

        # self.plot_dir.mkdir(exist_ok=True)
        # G = nx.MultiGraph()
        # positions = {}
        # node_colours = []
        # all_edges = {}

        # # to bind each tags or deck to a color
        # if len(list(set(self.df["deckName"]))) > 5:  # if there are more than 5 different decks
        #     self.df["colors"] = self.df["deckName"].factorize()[0]
        # else:
        #     self.df["colors"] = self.df["tags"].factorize()[0]

        # # add nodes
        # for cid in tqdm(self.df.index, desc="Adding nodes", unit="node",
        #                 file=self.t_strm):
        #     nid = self.df.loc[cid, "note"]
        #     G.add_node(nid)  # duplicate nodes are ignored by networkx
        #     if nid not in positions:
        #         positions[nid] = list(self.vectors2D[np.argwhere(self.df.index == cid).squeeze(), :])

        #         node_colours.append(self.df.loc[cid, "colors"])

        # # create a dict containing all edges and their weights
        # min_w = np.inf
        # max_w = -np.inf
        # sum_w = 0
        # n_w = 0
        # assert len(self.df.index) == self.df_dist.shape[0]
        # if not hasattr(self, "nbrs_cache"):
        #     self.nbrs_cache = {}  # for quicker check
        # for cardId in tqdm(
        #         self.df.index,
        #         desc="Computing edges",
        #         file=self.t_strm,
        #         mininterval=5,
        #         unit="card"):
        #     noteId = self.df.loc[cardId, "note"]
        #     if noteId in self.nbrs_cache:
        #         nbrs_cid = self.nbrs_cache[noteId]["nbrs_cid"]
        #         nbrs_nid = self.nbrs_cache[noteId]["nbrs_nid"]
        #     else:
        #         nbrs_cid = self.df_dist.loc[cardId, :].nsmallest(self.n_n)
        #         nbrs_cid = nbrs_cid[nbrs_cid <= self.max_radius]
        #         assert len(nbrs_cid) > 0, "Invalid nbrs_cid value"
        #         nbrs_cid = nbrs_cid.sort_values(ascending=True)
        #         nbrs_nid = [self.df.loc[ind, "note"]
        #                     for ind in nbrs_cid.index]
        #     for ii, n_nid in enumerate(nbrs_nid):
        #         if noteId == n_nid:
        #             # skip self neighboring
        #             continue
        #         if ii > n_n_plot:
        #             # Only considering the n first neighbors
        #             break
        #         smallest = min(noteId, n_nid)
        #         largest = max(noteId, n_nid)
        #         # new weight is 1 minus the distance between points
        #         new_w = 1 - self.df_dist.loc[cardId, nbrs_cid.index[ii]]

        #         # store the weight
        #         if smallest not in all_edges:
        #             all_edges[smallest] = {largest: new_w}
        #         else:

        #             if largest not in all_edges[smallest]:
        #                 all_edges[smallest][largest] = new_w
        #             else:
        #                 # check that weight is coherent
        #                 # as some cards of the same note can have different
        #                 # embedding locations, severall weights can be
        #                 # stored and will be averaged later on
        #                 if isinstance(all_edges[smallest][largest], list):
        #                     all_edges[smallest][largest].append(new_w)
        #                 else:
        #                     all_edges[smallest][largest] = [all_edges[smallest][largest], new_w]
        #                     n_w -= 1  # avoid duplicate counts

        #         # store the weights information to scale them afterwards
        #         if new_w > max_w:
        #             max_w = new_w
        #         elif new_w < min_w:
        #             min_w = new_w
        #         sum_w += new_w
        #         n_w += 1

        # assert min_w >= 0 and min_w < max_w, (
        #         f"Impossible weight values: {min_w} and {max_w}")

        # # minmaxing weights (although instead of reducing to 0, it adds the
        # # a fixed value to avoid null weights)
        # new_spread = max_w - min_w
        # fixed_offset = 0.1
        # for k, v in tqdm(all_edges.items(),
        #                  desc="Minmax weights",
        #                  file=self.t_strm):
        #     for sub_k, sub_v in all_edges[k].items():
        #         if isinstance(sub_v, list):
        #             sub_v = float(np.mean(sub_v))
        #         all_edges[k][sub_k] = (sub_v - min_w) / new_spread

        #         # as the distance grows, the weight has to decrease:
        #         all_edges[k][sub_k] *= -1
        #         all_edges[k][sub_k] += 1 + fixed_offset
        #         assert all_edges[k][sub_k] <= 1 + fixed_offset, "Too high edge weight value!"
        #         assert all_edges[k][sub_k] >= 0, "Negative edge weight!"
        #         assert all_edges[k][sub_k] != 0, "Null edge weight!"

        # # add each edge to the graph
        # for k, v in tqdm(all_edges.items(),
        #                  desc="Adding edges",
        #                  file=self.t_strm):
        #     for sub_k, sub_v in all_edges[k].items():
        #         G.add_edge(k, sub_k, weight=sub_v)

        # # 2D embeddings layout
        # start = time.time()
        # whi("Drawing embedding network...")
        # self._create_plotfig(G=G,
        #                      computed_layout=positions,
        #                      node_colours=node_colours,
        #                      title=f"{self.deckname} - embeddings"
        #                      )
        # whi(f"Saved embeddings layout in {int(time.time()-start)}s!")
        # # nx.drawing.nx_pydot.write_dot(
        # #         G, f'{self.plot_dir}/{self.deckname} - embeddings.dot')

        # # computing spring layout
        # n = len(node_colours)
        # start = time.time()
        # whi("\nDrawing spring layout network...")
        # whi("    computing layout...")
        # layout_spring = nx.spring_layout(
        #         G,
        #         k=1 / np.sqrt(n),  # repulsive force, default is 1/sqrt(n)
        #         weight=None,
        #         # if 'None', all weights are assumed to be 1,
        #         # elif 'weight' use weight as computed previously
        #         pos=positions,  # initial positions is the 2D embeddings
        #         iterations=50,  # default to 50
        #         # fixed=None,  # keep those nodes at their starting position
        #         # center=None,  # center on a specific node
        #         dim=2,  # dimension of layout
        #         seed=4242,
        #         threshold=1e-3,  # stop goes below, default 1e-4
        #         )
        # whi(f"Finished computing spring layout in {int(time.time()-start)}s")
        # self._create_plotfig(G=G,
        #                      computed_layout=layout_spring,
        #                      node_colours=node_colours,
        #                      title=f"{self.deckname} - spring"
        #                      )
        # whi(f"Saved spring layout in {int(time.time()-start)}s!")
        # # nx.drawing.nx_pydot.write_dot(
        # #         G, f'{self.plot_dir}/{self.deckname} - spring.dot')

        # # not implemented for nx.Multigraph
        # # computing average clustering coefficient
        # # whi("Computing average clustering...")
        # # t = time.time()
        # # avg_clst = nx.average_clustering(G,
        # #                                  weight="weight",
        # #                                  count_zeros=True,
        # #                                  )
        # # yel(f"Average clustering of plot: {avg_clst}"
        # #     f"\n(In {int(time.time()-t)}s)")

        whi("Finished 2D plots")
        signal.alarm(0)  # turn off timeout
        return

    def _create_plotfig(self,
                        G,
                        computed_layout,
                        node_colours,
                        title):
        """
        Create 2D plotly plot from networkx graph.
        """

        # only draw some edges, with an upper limit to reduce loading time
        p = 0.05  # proportion of edge to draw
        n_limit = 10_000  # don't draw more edges than this number

        p_int = int(p * len(G.edges.data()))
        if p_int > n_limit:
            yel(f"Too many edges to draw ({p_int}), drawing "
                f"only {n_limit}.")
        edges_to_draw = random.sample(list(G.edges.data()), min(p_int, n_limit))

        # multithreading way to add each edge to the plot list
        self.edges_x = []
        self.edges_y = []
        lock = threading.Lock()
        tqdm_params = {
                "total": len(G.edges.data()),
                "desc": "Plotting edges",
                "file": self.t_strm,
                "leave": True,
                }
        def _plot_edges(edge, computed_layout, lock):
            "multithreaded way to add all edges to the plot"
            x0, y0 = computed_layout[edge[0]]
            x1, y1 = computed_layout[edge[1]]
            with lock:
                self.edges_x.extend([x0, x1, None])
                self.edges_y.extend([y0, y1, None])
        parallel = ProgressParallel(
                tqdm_params=tqdm_params,
                backend="threading",
                n_jobs=-1,
                mmap_mode=None,
                max_nbytes=None,
                verbose=0,
                )
        parallel(joblib.delayed(_plot_edges)(
            edge=edge,
            computed_layout=computed_layout,
            lock=lock,
            ) for edge in edges_to_draw)
        edge_trace = Scatter(
            x=self.edges_x,
            y=self.edges_y,
            opacity=0.3,
            line=scatter.Line(color='rgba(136, 136, 136, .8)'),
            hoverinfo='none',
            mode='lines'
            )

        # multithreading way to add each node to the plot list
        self.nodes_x = []
        self.nodes_y = []
        self.nodes_text = []
        lock = threading.Lock()
        tqdm_params = {
                "total": len(G.nodes()),
                "desc": "Plotting nodes",
                "file": self.t_strm,
                "leave": True,
                }
        note_df = self.df.reset_index().drop_duplicates(subset="note").set_index("note")
        def _plot_nodes(node, note_df, computed_layout, lock):
            "multithreaded way to add all nodes to the plot"
            x, y = computed_layout[node]

            content = note_df.loc[node, "text"]
            tag = note_df.loc[node, "tags"]
            deck = note_df.loc[node, "deckName"]
            text = (f"<b>deck:</b>{'<br>'.join(textwrap.wrap(deck, width=60))}"
                    "<br>"
                    f"<b>nid:</b>{node}"
                    "<br>"
                    f"<b>tag:</b>{'<br>'.join(textwrap.wrap(tag, width=60))}"
                    "<br>"
                    "<br>"
                    f"<b>Content:</b>{'<br>'.join(textwrap.wrap(content, width=60))}")
            with lock:
                self.nodes_x.append(x)
                self.nodes_y.append(y)
                self.nodes_text.append(text)

        parallel = ProgressParallel(
                tqdm_params=tqdm_params,
                backend="threading",
                n_jobs=-1,
                mmap_mode=None,
                max_nbytes=None,
                verbose=0,
                )
        parallel(joblib.delayed(_plot_nodes)(
            node=node,
            computed_layout=computed_layout,
            note_df=note_df,
            lock=lock,
            ) for node in G.nodes())
        node_trace = Scatter(
            x=self.nodes_x,
            y=self.nodes_y,
            text=self.nodes_text,
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

        saved_plot = f"{self.plot_dir}/{title}.html"
        whi(f"Saving plot to {saved_plot}")
        offpy(fig,
              filename=saved_plot,
              auto_open=False,
              show_link=False,
              validate=True,
              output_type="file",
              )
        try:
            # replacing timeout by a 5s one then resuming the previous one
            def f_browser_timeout(signum, frame):
                raise TimeoutError
            signal.alarm(0)
            signal.signal(signal.SIGALRM, f_browser_timeout)
            signal.alarm(5)
            whi(f"Trying to open {saved_plot} in the browser...")
            saved_plot_fp = str(Path(saved_plot).absolute()).replace("\\", "")
            if "genericbrowser" in str(webbrowser.get()).lower():
                # if AnnA is launched using SSH, the webbrowser will
                # possibly be in the console and can stop the script
                # while the browser is not closed.
                whi("No GUI browser detected, maybe you're in an SSH console? "
                    "\nFalling back to using linux shell to open firefox")
                subprocess.check_output(
                        shlex.split(f"env DISPLAY=:0 firefox '{saved_plot_fp}'"),
                        shell=False,
                        )
            else:
                whi("Opening browser.")
                webbrowser.open(saved_plot_fp)
        except TimeoutError as e:
            elapsed = self.timeout_in_minutes * 60 - (time.time() - self.timeout_start_time)
            if elapsed <= 1:  # rare case when the timeout is for the overall
                # plotting code and not just to open the browser
                raise
            else:
                pass  # the function got stuck when openning the browser, ignore
        except Exception as e:
            beep(f"Exception when openning file: '{e}'")
        signal.alarm(0)
        signal.signal(signal.SIGALRM, self.time_watcher)
        signal.alarm(int(self.timeout_in_minutes * 60 - (time.time() - self.timeout_start_time)))


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
                            "can be 'filter_review_cards', "
                            "'bury_excess_learning_cards', "
                            "'bury_excess_review_cards', 'add_KNN_to_field'"
                            "'just_plot'. "
                            ". Respectively to create "
                            "a filtered deck with the cards, or bury only the "
                            "similar learning cards (among other learning cards), "
                            "or bury only the similar cards in review (among "
                            "other review cards) or just find the nearest "
                            "neighbors of each note and save it to the field "
                            "'Nearest_neighbors' of each note, or create "
                            "a 2D plot after vectorizing the cards. "
                            "Default is `filter_review_cards`."))
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
                        default="1,5",
                        type=str,
                        required=False,
                        help=(
                            "a comma separated list of numbers used to "
                            "adjust the value of the reference order compared to "
                            "how similar the cards are. Default is `1,5`. For "
                            "example: '1, 1.3' means that the algorithm will "
                            "spread the similar cards farther apart."))
    parser.add_argument("--field_mapping",
                        nargs=1,
                        metavar="FIELD_MAPPING_PATH",
                        dest="field_mappings",
                        default="utils/field_mappings.py",
                        type=str,
                        required=False,
                        help=(
                            "path of file that indicates which field to keep "
                            "from which note type and in which order. Default "
                            "value is `utils/field_mappings.py`. If empty or if no "
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
                        default="utils/acronym_example.py",
                        required=False,
                        help=(
                            "a python file containing dictionaries that "
                            "themselves contain acronyms to extend in "
                            "the text "
                            "of cards. For example `CRC` can be extended "
                            "to `CRC "
                            "(colorectal cancer)`. (The parenthesis are "
                            "automatically added.) Default is "
                            "`\"utils/acronym_example.py\"`. The matching is case "
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
                            "to extract file supplied in `acronym_file` var. "
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
                            "`highjack_due_query`. "
                            "Using this will also bypass the function "
                            "'iterated_fetcher' which looks for cards rated "
                            "at each day until rated_last_X_days instead of "
                            "querying all of them at once which removes "
                            "duplicates (reviews of the same card but "
                            "on different days). Note that 'iterated_fetcher'"
                            " also looks for cards in filtered decks "
                            "created by AnnA from the same deck. "
                            "When 'iterated_fetcher' is "
                            "used, the importance of reviews is gradually "
                            "decreased as the number of days since the "
                            "review grows. In short it's doing "
                            "temporal discounting. Default is `None`."))
    parser.add_argument("--low_power_mode",
                        dest="low_power_mode",
                        default=False,
                        action="store_true",
                        required=False,
                        help=(
                            "enable to reduce the computation needed for "
                            "AnnA, making it usable for less powerful "
                            "computers. This can greatly reduce accuracy. "
                            "Also removes non necessary steps that take long "
                            "like displaying some stats. "
                            "Default to `False`. "
                            ))
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
                        default=False,
                        required=False,
                        action="store_true",
                        help=(
                            "Whether to append the tags to the "
                            "cards content or to add no tags. Default "
                            "to `False`."))
    parser.add_argument("--tags_to_ignore",
                        nargs="*",
                        metavar="TAGS_TO_IGNORE",
                        dest="tags_to_ignore",
                        default=["AnnA", "leech"],
                        type=list,
                        required=False,
                        help=(
                            "a list of regexp of tags to "
                            "ignore when "
                            "appending tags to cards. This is not a "
                            "list of tags "
                            "whose card should be ignored! Default is "
                            "['AnnA', 'leech']. Set to None to disable it."))
    parser.add_argument("--add_KNN_to_field",
                        action="store_true",
                        dest="add_KNN_to_field",
                        default=False,
                        required=False,
                        help=(
                            "Whether to add a query to find the K nearest"
                            "neighbor of a given card to a new field "
                            "called 'Nearest_neighbors' (only if already "
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
    parser.add_argument("--filtered_deck_at_top_level",
                        action="store_true",
                        dest="filtered_deck_at_top_level",
                        default=True,
                        required=False,
                        help=(
                            "If True, the new filtered deck will be a "
                            "top level deck, if False: the filtered "
                            "deck will be next to the original "
                            "deck. Default to True."))
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
                        default="embeddings",
                        required=False,
                        type=str,
                        help=(
                            "Either TFIDF or 'embeddings' to use "
                            "sentencetransformers. The latter will "
                            "deduplicate the field_mapping, mention the "
                            "name of the field before it's content "
                            "before tokenizing, use a cache to avoid "
                            "recomputing the embedding for previously "
                            "seen notes, ignore stopwords and "
                            "any TFIDF arguments used."
                            ))
    parser.add_argument("--embed_model",
                        nargs=1,
                        metavar="EMBED_MODEL",
                        dest="embed_model",
                        default="paraphrase-multilingual-mpnet-base-v2",
                        required=False,
                        type=str,
                        help=(
                            "For multilingual use "
                            "'paraphrase-multilingual-mpnet-base-v2' but for "
                            "anything else use 'all-mpnet-base-v2'"
                            ))
    parser.add_argument("--ndim_reduc",
                        nargs=1,
                        metavar="NDIM_REDUC",
                        dest="ndim_reduc",
                        default="auto",
                        required=False,
                        help=(
                            "the number of dimension to keep using "
                            "TruncatedSVD (if TFIDF) or PCA (if embeddings). "
                            "If 'auto' will automatically find the "
                            "best number of dimensions to keep 80%% of the "
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
    parser.add_argument("--TFIDF_tknizer_model",
                        dest="TFIDF_tknizer_model",
                        default="GPT",
                        metavar="TFIDF_tknizer_model",
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
    parser.add_argument("--TFIDF_stem",
                        dest="TFIDF_stem",
                        default=False,
                        action="store_true",
                        required=False,
                        help=(
                            "default to `False`. Whether to enable "
                            "stemming of words. Currently the PorterStemmer "
                            "is used, and was made for English but can still "
                            "be useful for some other languages. Keep in "
                            "mind that this is the longest step when "
                            "formatting text."))
    parser.add_argument("--plot_2D_embeddings",
                        dest="plot_2D_embeddings",
                        default=False,
                        action="store_true",
                        required=False,
                        help=(
                            "EXPERIMENTAL AND UNFINISHED. "
                            "default to `False`. Will compute 2D embeddins "
                            "then create a 2D plots at the end."))
    parser.add_argument("--plot_dir",
                        nargs=1,
                        metavar="PLOT_PATH",
                        dest="plot_dir",
                        type=str,
                        default="Plots",
                        required=False,
                        help=(
                            "Path location for the output plots. "
                            "Default is 'Plots'."
                            ))
    parser.add_argument("--dist_metric",
                        nargs=1,
                        metavar="DIST_METRIC",
                        dest="dist_metric",
                        type=str,
                        default="cosine",
                        required=False,
                        help=(
                            "when computing the distance matrix, whether to "
                            "use 'cosine' or 'rbf' or 'euclidean' metrics. "
                            "cosine and rbf should be fine. Default to 'cosine'"))
    parser.add_argument("--whole_deck_computation",
                        dest="whole_deck_computation",
                        default=False,
                        required=False,
                        action="store_true",
                        help=(
                            "defaults to `False`. This can only be used with "
                            "TFIDF and would not make any sense for "
                            "sentence-transformers. Use ankipandas to "
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
                        default="closer",
                        type=str,
                        required=False,
                        help=(
                            "Resorting the new filtered deck taking only"
                            "into account the semantic distance and not the "
                            "reference score. Useful if you are "
                            "certain to "
                            "review the entierety of the filtered deck "
                            "today as it "
                            "will minimize similarity between consecutive "
                            "cards. If "
                            "you are not sure you will finish the "
                            "deck today, set to `False` to make sure "
                            "you review first the most urgent cards. This "
                            "feature is active only if you set `task` to "
                            "'filter_review_cards'. "
                            "Can be either 'farther' or 'closer' or False. "
                            "'farther' "
                            "meaning to spread the cards as evenly as "
                            "possible. "
                            "Default to 'closer'."))
    parser.add_argument("--resort_split",
                        dest="resort_split",
                        default=False,
                        action="store_true",
                        required=False,
                        help=(
                            "If 'resort_by_dist' is not False, set to True to "
                            "resort the boosted cards separately from the rest "
                            "and make them appear first in the filtered deck."
                            " Default to `False`."
                            ))
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
                            "collection. Only used if "
                            "'whole_deck_computation' is set to `True`"))
    parser.add_argument("--keep_console_open",
                        dest="console_mode",
                        default=False,
                        action="store_true",
                        required=False,
                        help=(
                            "defaults to `False`. Set to True to "
                            "open a python console after running."))
    parser.add_argument("--sync_behavior",
                        nargs=1,
                        metavar="SYNC_BEHAVIOR",
                        dest="sync_behavior",
                        default="before&after",
                        required=False,
                        help=(
                            "If contains 'before', will trigger a sync when "
                            "AnnA is run. If contains 'after', will trigger "
                            "a sync at the end of the run. "
                            "Default is `before&after`."))

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

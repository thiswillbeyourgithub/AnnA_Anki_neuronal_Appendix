# AnnA : Anki neuronal Appendix
Tired of having to deal with anki flashcards that are too similar when grinding through your backlog? This python script creates filtered deck in optimal review order. It uses Machine Learning / AI to make semantically linked cards far from one another.

## Table of contents
* [One sentence summary](#One-sentence-summary)
* [Note to readers](#Note-to-readers)
* [Other features](#Other-features)
* [FAQ](#FAQ)
* [Getting started](#Getting-started)
* [Usage and arguments](#Usage-and-arguments)
* [TODO list](#TODO)
* [Credits and links that were helpful](#Credits-and-links-that-were-helpful)
* [Crazy Ideas](#Crazy-ideas)

## One sentence summaries
Here are different ways of looking at what AnnA can do for you in a few words:
* When you don't have the time to complete all your daily reviews, use this to create a special filtered deck that makes sure you will only review the cards that are most different from the rest of your reviews.
* When you have too many learning cards and fear that some of them are too similar, use this to automatically review a subset of them.
* AnnA helps to avoid reviewing **similar** cards on the same day.
* AnnA allows to reduce the number of daily reviews while increasing (and not keeping the same) retention.
* AnnA allows to automatically write in each note which other notes are its semantic neighbours.

## Note to readers
0. The [dev branch](https://github.com/thiswillbeyourgithub/AnnA_Anki_neuronal_Appendix/tree/dev) is usually less outdated than the main branch.
1. I would really like to integrate this into anki somehow but am not knowledgeable enough about how to do it, how to manage anki versions, how to handle different platforms etc. All help is much appreciated!
2. This project communicates with anki using a fork of the addon [AnkiConnect](https://github.com/FooSoft/anki-connect) called [AnnA-companion](https://ankiweb.net/shared/info/447942356). Note that AnnA-companion was tested from anki 2.1.44 to 2.1.54 only.
3. Although I've been using it daily for months, I am still changing the code base almost every day, if you tried AnnA and were disappointed, maybe try it another time later. Major improvements are regularly made.
4. In the past I implemented several vectorization methods. I now only kept [subword TF_IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf). TF_IDF is known to be reliable, fast, very general (it does not assume anything about your cards and will work for just about any language, format, phrasing etc). TF_IDF works very well if you have large number of cards.
5. If you want to know how I'm using this, take a look at [authors_routine.md](./docs/authors_routine.md)
6. If you like this, another  project of mine can be used to do semantic search on your anki collection using AI : [I call it Anki SemSearch](https://github.com/thiswillbeyourgithub/Anki-Semantic-Search)

## Other features
* Code is PEP8 compliant, dynamically typed, all the functions have a detailed docstrings. Contributions are welcome, opening issues is encouraged and appreciated.
* Keeps the OCR data of pictures in your cards, if you analyzed them beforehand using [AnkiOCR](https://github.com/cfculhane/AnkiOCR/).
* Can automatically replace acronyms in your cards (e.g. 'PNO' can be replaced to "pneumothorax" if you tell it to), regexp are supported (if you want access to my homemade list of french medical acronyms, open an issue)
* Can attribute more importance to some fields of some cards if needed.
* Can be used to optimize the order of a deck of new cards, [read this thread to know more](https://github.com/thiswillbeyourgithub/AnnA_Anki_neuronal_Appendix/issues/14)
* Previous feature like clustering, plotting, searching have been removed as I don't expect them to be useful to users. But the code was clean so if you need it for some reason don't hesitate to open an issue.

## FAQ
* **This looks awesome! Are you working on other things like that?** Yes and thank you! Another project of mine aims to do semantic search on your anki collection using AI : [I call it Anki SemSearch](https://github.com/thiswillbeyourgithub/Anki-Semantic-Search)
* **How does it work? (Layman version)** AnnA connects to its companion addon to access your anki collection. This allows to use any python library without having to go through the trouble how packaging those libraries into an addon. It uses a vectorizer to assign numbers (vectors) to each cards. If the numbers of two cards are very close, then the cards have similar content and should not be reviewed too close to each other.
* **And in more details?** The vectorization method AnnA is using, `subword TF_IDF` is a way to count words in a document (anki cards in this case) and understand which are more important. "subword" here means that I used BERT tokenization (i.e. splitting "hypernatremia" into "hyper **na** tremia" which can be linked to cards dealing with "**Na** Cl" for example). TF_IDF is generally considered as part of machine learning. AnnA leverages this information to make sure you won't review cards on the same day that are too similar. This is very useful when you have to many cards to review in a day. I initially tried to combine several vectorizer into a common scoring but it proved unreliable, I also experimented with fastText (only one language at a time, can't be packaged and large RAM usage), sentence-BERT (too large and depends too much on phrasing to be reliable). So I decided to keep it simple and provide only TFIDF. The goal is to review only the most useful cards, kinda like [pareto distribution](https://en.wikipedia.org/wiki/Pareto_distribution) (i.e. review less cards, but review the right one and you should be able to keep afloat in med school). The code is mainly composed of a python class called AnnA. When you instantiate this class, you have to supply the name of the deck you want to filter from. It will then automatically fetch the cards from your collection, then use TF_IDF to assign vectors to each card, compute the distance matrix of the cards and create a filtered deck containing the cards in the optimal order (or bury the cards you don't have to review today). Note that rated cards of the last X days of the same deck will be used as reference to avoid having cards that are too similar to yesterday's reviews too. If you want to know more, either open an issue or read the docstrings in the code.

* **Will this change anything to my anki collection?** Apart from adding tags or filling the field 'KNN_neighbours", AnnA should not modify anything, if you delete the filtered deck, everything will be back to normal. That being said, the project is young and errors might still be present.
* **What is the KNN_neighbours field?** A field that, if manually added in the notetype will contain a query that, if entered in the anki browser will show you which card of the deck are dimmed most similar by AnnA. The number of neighbour by default is 0.1% of the size of the deck or at least 5.
* **Does it work if I have learning steps over multiple days?** Yes, that's my use case. AnnA, depending on the chosen task, either only deals with review queue and ignores learning cards and new cards, or only buries the part of your learning cards that are too similar (ignoring due cards). You can use both one after the other every morning like I do. If you have learning cards in your filtered deck it's because you lapsed those cards yesterday.
* **Does this only work for English cards?** No! TF_IDF use a multilingual BERT uncased tokenization so should work on most languages (even if you have several different languages in the same deck).
* **Can I use this if I don't know python?** Yes! Installing the thing might not be easy but it's absolutely doable. And you don't need to know python to *run* AnnA. I tried my best to make it accessible and help is welcome.
* **What do you call "optimal review order"?** The order that minimizes the chance of reviewing similar cards in a day. You see, Anki has no knowledge of the content of cards and only cares about their interval and ease. Its built-in system of "siblings" is useful but I think we can do better. AnnA was made to create filtered decks sorted by "relative_overdueness" (or other) BUT in a way that keeps *semantic* siblings far from each other.
* **When should I use this?** It seems good for dealing with the huge backlog you get in medical school, or just everyday to reduce the workload. If you have 2 000 reviews to do, but can only do 500 in a day: AnnA is making sure that you will get the most out of those 500.
* **How do you use this?** I described my routine in a separate file called `docs/authors_routine.md`.
* **Can I use AnnA to optimize a list of new cards?** I never did it personally but helped a user doing it: [see related thread](https://github.com/thiswillbeyourgithub/AnnA_Anki_neuronal_Appendix/issues/14)

* **What are the power requirements to run this?** I wanted to make it as efficient as possible but am still improving the code. Computing the distance matrix can be long if you do this on very large amount of cards but this step is done in parallel on all cores so should not be the bottleneck. Let me know if some steps are unusually slow and I will try to optimize it. There are ways to make it way easier on the CPU, see arguments `low_power_mode` and `TFIDF_dim`.
* **How long does it take to run? Is it fast?** It takes about 1 min to run for a deck of less than 500 due cards, I've seen it take as long as 4 minutes on a `10 000` due cards deck. This is on a powerful laptop with no GPU on Linux.
* **Why is creating the queue taking longer and longer?** Each card is added to the queue after having been compared to the rated cards of the last few days and the current queue. As the queue grows, more computation have to be done. If this is an issue, consider creating decks that are as big as you think you can review in a day. With recent improvements in the code the speed should really not be an issue.
* **Does this work only on Linux?** It should work on all platform, provided you have anki installed and [AnnA-companion](https://ankiweb.net/shared/info/447942356) enabled. But the script (not the addon) uses libraries that might only work on some CPU architectures, so I'm guessing ARM system would not work but please tell me if you know tried.
* **What is the current status of this project?** I use it daily but am still working on the code. You can expect breaking. I intend to keep developing until I have no free time left. Take a look at the TODO list of the dev branch if you want to know what's in the works. When in doubt, open an issue.
* **Do you mind if I open an issue?** Not at all! It is very much encouraged, even just for a typo. That will at least make me happy. Of course PR are always welcome too.
* **Can this be made into an anki addon instead of a python script?** I have never packaged things into anki addons so I'm not so sure. I heard that packaging complex modules into anki is a pain, and cross platform will be an issue. If you'd like to make this a reality, show yourself by opening an issue! I would really like this to be implemented into anki, and the search function would be pretty nice :)
* **What version of anki does this run on?** I've been using AnnA from anki 2.1.44 and am currently on 2.1.54 Compatibility relies heavily on anki-connect. Please tell me if you run into issues.

* **If I create a filtered deck using AnnA, how can I rebuild it?** You can't rebuilt it or empty it through anki directly as it would leave you with anki's order and not AnnA's. You have to delete the filtered deck then run the script. Hence, I suggest creating large filtered decks in advance. 
* **I don't think the reviews I do on AnnA's filtered decks are saved, wtf?** It might be because you're using multiple device and are deleting the filtered deck on one of the device without syncing first.
* **What is subword TF_IDF?** Short for "subword term frequency–inverse document frequency". It's a clever way to split words into subparts then count the parts to figure out which cards are related.
* **Does it work with images?** Not currently but sBERT can be used with CLIP models so I could pretty simply implement this. If you think you would find it useful I can implement it :).
* **What are the supported languages using TF_IDF?** TF_IDF is language agnostic, but the language model used to split the words was trained on the 102 largest wikipedia corpus. If your languages is very weird and non unicode or non standard in some ways you might have issues, don't hesitate to open an issue as I would gladly take a look.

* **What is the field_mapping.py file?** It's a file with a unique python dictionary used by AnnA to figure out which field of which card to keep. Using it is optional. By default, each notetype will see only it's first field kept. But if you use this file you can keep multiple fields. Due to how TF_IDF works, you can add a field multiple times to give it more importance relative to other fields.
* **What is "XXX - AnnA Optideck"?** The default name for the filtered deck created by AnnA. It contains the reviews in the best order for you.
* **Why are there only reviews and no learning cards in the filtered decks?** When task is set to `filter_review_cards`, AnnA will fetch only review cards and not deal with learning cards. This is because I'm afraid of some weird behavior that would arise if I change the due order of learning cards. Whereas I can change it just find using review cards.
* **Why does the progress bar of "Computing optimal review order" doesn't always start at 0?** It's just cosmetic. At each step of the loop, the next card to review is computed by comparing it against the previously added cards and the cards rated in the last few days. This means that each turn is slower than the last. So the progress bar begins with the number of cards rated in the past to indicate that. It took great care to optimize the loop so it should not really be an issue.
* **Does AnnA take into account my card's tags ?** If you set the argument `append_tags` to True, the 2 deepest level of each tags are appended at the end of the text and used like if it was part of the content of the card. Note that "_ - and /" are replaced by a space. For example : `a::b::c::d some::thing` will be turned into `c d some thing`.
* **Why does the task "bury_excess_learning_cards" seem to ignore a few cards?** I decided to have this task not take into account cards that were failed today or the day before, those are usually top priority and I don't won't AnnA to bury them.
* **How can I know AnnA is actually working and not burying random cards?** Good question, I tried to implement a metric called *Improvement ratio* that displays a score at the end of the run. It's a sort of sum of the distance between all cards in the new queue over the sum of the distance between all cards in what would have been the queue. It's not really indicative of anything beyond the fact that it's over 1 (=it helped) or under 1 (=it was detrimental). But the latter is kinda hard to get right because it depends on the reliability of `reference_order` in itself. I am especially doubtful of the meaning of the ratio when dealing with learning cards but so far it seems to work ok. Consider it a work in progress.
* **I have underlined or put in bold the most important parts of my cards, does it matter?** Words that are put in bold or underlined are actually duplicated in the text, this gives them twice as much importance.
* **What scheduler should I use with AnnA? Can I use the V3 scheduler?** I strongly suggest leaving the V1 as it messes with learning cards quite a lot. I personally use V2 and developed AnnA while using V2. I don't fully understand the changes in V3 and am not really certain that it works correctly with AnnA so I can't really say. I plan to investigate switching to V3 around October 2022.

## Getting started
* First, **read this page in its entirety, this is a complicated piece of software and you don't want to use it irresponsibly on your cards. The [usage section](#Usage-and-arguments) is especially useful.**
* Install the addon [AnnA Companion (Anki neuronal Appendix) - do LESS reviews with MORE retention!](https://ankiweb.net/shared/info/447942356)
* Clone this repository (for example with `git clone https://github.com/thiswillbeyourgithub/AnnA_Anki_neuronal_Appendix`)
* Install the required python libraries : `pip install -r requirements.txt` (in case of errors, try using python 3.9)
* Edit file `utils/field_mapping.py`: it contains a dictionary where the keys are notetypes and values are lists of which field to take into account.
* Edit file `utils/acronym_example.py`: it contains dictionaries where keys are words to replace and values are what the words should be replaced with.
* There are two ways to run AnnA:
    * Either in a Python console : `from AnnA import * ; AnnA(YOUR_ARGUMENTS)`
    * Or directly in the terminal : `python3 AnnA.py --help` *(note that the terminal mode was added after the python console and might still contain error when parsing arguments)*
* If you want to run AnnA on several decks in a row like I do, edit the file `utils/autorun_example.py`, move it one folder up then execute it with `python3 ./autorun.py`
* Open an issue telling me your remarks and suggestion

### Usage and arguments
AnnA was made with customizability in mind. All the settings you might want to edit are arguments of the call of AnnA Class. Don't be frightened, many of those settings are rarely used and the default values should be good for almost anyone. Here are the arguments with the relevant explanation.

* `python ./AnnA.py  --help`:
```
usage: AnnA.py [-h] --deckname DECKNAME [--reference_order REF_ORDER] --task
               TASK --target_deck_size TARGET_SIZE
               [--max_deck_size MAX_DECK_SIZE]
               [--stopwords_lang STOPLANG [STOPLANG ...]]
               [--rated_last_X_days RATED_LAST_X_DAYS]
               [--score_adjustment_factor SCORE_ADJUSTMENT_FACTOR [SCORE_ADJUSTMENT_FACTOR ...]]
               [--field_mapping FIELD_MAPPING_PATH]
               [--acronym_file ACRONYM_FILE_PATH]
               [--acronym_list ACRONYM_LIST [ACRONYM_LIST ...]]
               [--minimum_due MINIMUM_DUE_CARDS]
               [--highjack_due_query HIGHJACK_DUE_QUERY]
               [--highjack_rated_query HIGHJACK_RATED_QUERY]
               [--low_power_mode] [--log_level LOG_LEVEL] [--replace_greek]
               [--keep_OCR] [--append_tags]
               [--tags_to_ignore [TAGS_TO_IGNORE ...]] [--add_KNN_to_field]
               [--filtered_deck_name_template FILTER_DECK_NAME_TEMPLATE]
               [--filtered_deck_at_top_level] [--filtered_deck_by_batch]
               [--filtered_deck_batch_size FILTERED_DECK_BATCH_SIZE]
               [--show_banner] [--repick_task REPICK_TASK]
               [--vectorizer VECTORIZER] [--TFIDF_dim TFIDF_DIMENSIONS]
               [--TFIDF_tokenize] [--tokenizer_model TOKENIZER_MODEL]
               [--plot_2D_embeddings] [--plot_dir PLOT_PATH] [--TFIDF_stem]
               [--dist_metric DIST_METRIC] [--whole_deck_computation]
               [--enable_fuzz] [--resort_by_dist RESORT_BY_DIST]
               [--profile_name PROFILE_NAME] [--keep_console_open]
               [--sync_behavior SYNC_BEHAVIOR]

optional arguments:
  -h, --help            show this help message and exit
  --deckname DECKNAME   the deck containing the cards you want to review. If
                        you don't supply this value or make a mistake, AnnA
                        will ask you to type in the deckname, with
                        autocompletion enabled (use `<TAB>`). Default is
                        `None`.
  --reference_order REF_ORDER
                        either "relative_overdueness" or "lowest_interval" or
                        "order_added" or "LIRO_mix". It is the reference used
                        to sort the card before adjusting them using the
                        similarity scores. Default is
                        `"relative_overdueness"`. Keep in mind that my
                        relative_overdueness is a reimplementation of the
                        default overdueness of anki and is not absolutely
                        exactly the same but should be a close approximation.
                        If you find edge cases or have any idea, please open
                        an issue. LIRO_mix is simply the the weighted average
                        of relative overdueness and lowest interval (4 times
                        more important than RO) (after some post processing).
                        I created it as a compromise between old and new
                        courses. My implementation of relative overdueness
                        includes a boosting feature: if your dues contain
                        cards with its overdueness several times larger than
                        its interval, they are urgent. AnnA will add a tag to
                        them and increase their likelyhood of being part of
                        the Optideck.
  --task TASK           can be "filter_review_cards",
                        "bury_excess_learning_cards",
                        "bury_excess_review_cards", "add_KNN_to_field".
                        Respectively to create a filtered deck with the cards,
                        or bury only the similar learning cards (among other
                        learning cards), or bury only the similar cards in
                        review (among other review cards) or just find the
                        nearest neighbors of each note and save it to the
                        field 'Nearest_neighbors' of each note. Default is
                        "`filter_review_cards`".
  --target_deck_size TARGET_SIZE
                        indicates the size of the filtered deck to create. Can
                        be the number of due cards like "100", a proportion of
                        due cards like '80%', the word "all" or "deck_config"
                        to use the deck's settings for max review. Default is
                        `deck_config`.
  --max_deck_size MAX_DECK_SIZE
                        Maximum number of cards to put in the filtered deck or
                        to leave unburied. Default is `None`.
  --stopwords_lang STOPLANG [STOPLANG ...]
                        a comma separated list of languages used to construct
                        a list of stop words (i.e. words that will be ignored,
                        like "I" or "be" in English). Default is `english
                        french`.
  --rated_last_X_days RATED_LAST_X_DAYS
                        indicates the number of passed days to take into
                        account when fetching past anki sessions. If you rated
                        500 cards yesterday, then you don't want your today
                        cards to be too close to what you viewed yesterday, so
                        AnnA will find the 500 cards you reviewed yesterday,
                        and all the cards you rated before that, up to the
                        number of days in rated_last_X_days value. Default is
                        `4` (meaning rated today, and in the 3 days before
                        today). A value of 0 or `None` will disable fetching
                        those cards. A value of 1 will only fetch cards that
                        were rated today. Not that this will include cards
                        rated in the last X days, no matter if they are
                        reviews or learnings. you can change this using
                        "highjack_rated_query" argument.
  --score_adjustment_factor SCORE_ADJUSTMENT_FACTOR [SCORE_ADJUSTMENT_FACTOR ...]
                        a comma separated list of numbers used to adjust the
                        value of the reference order compared to how similar
                        the cards are. Default is `1,5`. For example: '1, 1.3'
                        means that the algorithm will spread the similar cards
                        farther apart.
  --field_mapping FIELD_MAPPING_PATH
                        path of file that indicates which field to keep from
                        which note type and in which order. Default value is
                        `utils/field_mappings.py`. If empty or if no matching
                        notetype was found, AnnA will only take into account
                        the first 2 fields. If you assign a notetype to
                        `["take_all_fields]`, AnnA will grab all fields of the
                        notetype in the same order as they appear in Anki's
                        interface.
  --acronym_file ACRONYM_FILE_PATH
                        a python file containing dictionaries that themselves
                        contain acronyms to extend in the text of cards. For
                        example `CRC` can be extended to `CRC (colorectal
                        cancer)`. (The parenthesis are automatically added.)
                        Default is `"utils/acronym_example.py"`. The matching
                        is case sensitive only if the key contains uppercase
                        characters. The ".py" file extension is not mandatory.
  --acronym_list ACRONYM_LIST [ACRONYM_LIST ...]
                        a comma separated list of name of dictionaries to
                        extract file supplied in `acronym_file` var. Used to
                        extend text, for instance
                        `AI_machine_learning,medical_terms`. Default to None.
  --minimum_due MINIMUM_DUE_CARDS
                        stops AnnA if the number of due cards is inferior to
                        this value. Default is `5`.
  --highjack_due_query HIGHJACK_DUE_QUERY
                        bypasses the browser query used to find the list of
                        due cards. You can set it for example to
                        `deck:"my_deck" is:due -rated:14 flag:1`. Default is
                        `None`. **Keep in mind that, when highjacking queries,
                        you have to specify the deck otherwise AnnA will
                        compare your whole collection.**
  --highjack_rated_query HIGHJACK_RATED_QUERY
                        same idea as above, bypasses the query used to fetch
                        rated cards in anki. Related to `highjack_due_query`.
                        Using this will also bypass the function
                        'iterated_fetcher' which looks for cards rated at each
                        day until rated_last_X_days instead of querying all of
                        them at once which removes duplicates (reviews of the
                        same card but on different days). Default is `None`.
  --low_power_mode      enable to reduce the computation needed for AnnA,
                        making it usable for less powerful computers. This can
                        greatly reduce accuracy. Also removes non necessary
                        steps that take long like displaying some stats.
                        Default to `False`.
  --log_level LOG_LEVEL
                        can be any number between 0 and 2. Default is `0` to
                        only print errors. 1 means print also useful
                        information and >=2 means print everything. Messages
                        are color coded so it might be better to leave it at 3
                        and just focus on colors.
  --replace_greek       if True, all greek letters will be replaced with a
                        spelled version. For example `σ` becomes `sigma`.
                        Default is `True`.
  --keep_OCR            if True, the OCR text extracted using the great
                        AnkiOCR addon (https://github.com/cfculhane/AnkiOCR/)
                        will be included in the card. Default is `True`.
  --append_tags         Wether to append the tags to the cards content or to
                        add no tags. Default to `True`.
  --tags_to_ignore [TAGS_TO_IGNORE ...]
                        a list of regexp of tags to ignore when appending tags
                        to cards. This is not a list of tags whose card should
                        be ignored! Default is ['AnnA', 'leech']. Set to None
                        to disable it.
  --add_KNN_to_field    Wether to add a query to find the K nearestneighbor of
                        a given card to a new field called 'Nearest_neighbors'
                        (only if already present in the model). Be careful not
                        to overwrite the fields by running AnnA several times
                        in a row! For example by first burying learning cards
                        then filtering review cards. This argument is to be
                        used if you want to find the KNN only for the cards of
                        the deck in question and that are currently due. If
                        you want to run this on the complete deck you should
                        use the 'task' argument.
  --filtered_deck_name_template FILTER_DECK_NAME_TEMPLATE
                        name template of the filtered deck to create. Only
                        available if task is set to "filter_review_cards".
                        Default is `None`.
  --filtered_deck_at_top_level
                        If True, the new filtered deck will be a top level
                        deck, if False: the filtered deck will be next to the
                        original deck. Default to True.
  --filtered_deck_by_batch
                        To enable creating batch of filtered decks. Default is
                        `False`.
  --filtered_deck_batch_size FILTERED_DECK_BATCH_SIZE
                        If creating batch of filtered deck, this is the number
                        of cards in each. Default is `25`.
  --show_banner         used to display a nice banner when instantiating the
                        collection. Default is `True`.
  --repick_task REPICK_TASK
                        Define what happens to cards deemed urgent in
                        'relative_overdueness' ref mode. If contains 'boost',
                        those cards will have a boost in priority to make sure
                        you will review them ASAP. If contains 'addtag' a tag
                        indicating which card is urgent will be added at the
                        end of the run. Disable by setting it to None. Default
                        is `boost`.
  --vectorizer VECTORIZER
                        can nowadays only be set to "TFIDF", but kept for
                        legacy reasons.
  --TFIDF_dim TFIDF_DIMENSIONS
                        the number of dimension to keep using SVD. If 'auto'
                        will automatically find the best number of dimensions
                        to keep 80% of the variance. If an int, will do like
                        'auto' but starting from the supplied value. Default
                        is `auto`, you cannot disable dimension reduction for
                        TF_IDF because that would result in a sparse matrix.
                        (More information at https://scikit-learn.org/stable/m
                        odules/generated/sklearn.decomposition.TruncatedSVD.ht
                        ml).
  --TFIDF_tokenize      default to `True`. Enable sub word tokenization, for
                        example turn `hypernatremia` to `hyp + er + natr +
                        emia`. You cannot enable both `TFIDF_tokenize` and
                        `TFIDF_stem` but should absolutely enable at least
                        one.
  --tokenizer_model TOKENIZER_MODEL
                        default to `GPT`. Model to use for tokenizing the text
                        before running TFIDF. Possible values are 'bert' and
                        'GPT' which correspond respectivelly to `bert-base-
                        multilingual-cased` and `gpt_neox_20B` They should
                        work on just about any languages. Use 'Both' to
                        concatenate both tokenizers. (experimental)
  --plot_2D_embeddings  EXPERIMENTAL AND UNFINISHED. default to `False`. Will
                        compute 2D embeddins then create a 2D plots at the
                        end.
  --plot_dir PLOT_PATH  Path location for the output plots. Default is
                        'Plots'.
  --TFIDF_stem          default to `False`. Wether to enable stemming of
                        words. Currently the PorterStemmer is used, and was
                        made for English but can still be useful for some
                        other languages. Keep in mind that this is the longest
                        step when formatting text.
  --dist_metric DIST_METRIC
                        when computing the distance matrix, wether to use
                        'cosine' or 'rbf' or 'euclidean' metrics. cosine and
                        rbf should be fine. Default to 'cosine'
  --whole_deck_computation
                        defaults to `False`. Use ankipandas to extract all
                        text from the deck to feed into the vectorizer.
                        Results in more accurate relative distances between
                        cards. (more information at
                        https://github.com/klieret/AnkiPandas)
  --enable_fuzz         Disable fuzzing when computing optimal order ,
                        otherwise a small random vector is added to the
                        reference_score and distance_score of each card. Note
                        that this vector is multiplied by the average of the
                        `score_adjustment_factor` then multiplied by the mean
                        distance then divided by 10 to make sure that it does
                        not overwhelm the other factors. Defaults to `True`.
  --resort_by_dist RESORT_BY_DIST
                        Recomputing the best order of the cards in the
                        filtered deck without taking into account the
                        reference score. This is useful if you are sure to
                        review the entierety of the filtered deck today as it
                        will minimize similarity between cards. If you are not
                        sure you will finish the deck, it can be best to set
                        to False to make sure you review first the most urgent
                        cards. This feature is active only if you set the task
                        to 'filter_review_cards'. Can be either 'farther' or
                        'closer' or False. The former meaning to spread the
                        cards as differently as possible. Default to 'closer'.
  --profile_name PROFILE_NAME
                        defaults to `None`. Profile named used by ankipandas
                        to find your collection. If None, ankipandas will use
                        the most probable collection.
  --keep_console_open   defaults to `False`. Set to True to open a python
                        console after running.
  --sync_behavior SYNC_BEHAVIOR
                        If contains 'before', will trigger a sync when AnnA is
                        run. If contains 'after', will trigger a sync at the
                        end of the run. Default is `before&after`.
```

AnnA includes built-in methods you can run after instantiating the class. Note that methods beginning with a "_" are not supposed to be called by the user and are reserved for backend use. Here's a list of sometimes useful methods:

*note: to call python method after running anna outside of a console, use argument --keep_console_open and access the variable anna. For example `anna.display_best_review_order()`*

* `display_best_review_order` used as a debugging tool : only display order. Allows to check if the order seems correct without having to create a filtered deck.
* `save_df` saves the dataframe containing the cards and all other infos needed by AnnA as a pickle file. Used mainly for debugging. Files will be saved to the folder `DF_backups`

## TODO
* instead of having difference reference order, use a vector of weights to adjust each reference possible and combine them at will
* add new task "resort filtered deck"
* figure out a clever way to know which cards can be used to link 2 distinct tags/notes
* compute the distance matrix on the notes only instead of the cards
* instead of storing the nearest neighbors in a field, store it in a file accessible to the addon and add interface menus via the addon to query neighbors
* add picture and documentation to the knn feature
    * reach out to the dev of the mindmapping addon and show them the 2D plots you made
* rename the "acronym" system to something more explicit like "word_expander"
* switch to using Fire instead of argparse and move all the documentation to the docstring
* refactor a bit the way optimal score is computed to allow printing of the inner mechanism and debugging
* use joblib caching for text_formater function (necessary steps: add more argument to it to stop relying on self)
* add an option to generate a reachability plot for the decks using an argument (+add tqdm to OPTICS (I checked, it's easy) ; +do it with a timeout and in an external process)
* use hierarchical clustering (quick because you already have the distance matrix) to have the full tree, then instead of using the TFIDF distance when computing the optimal order, use the "tree distance"
* investigate wether CLIP is a better vectorizer than what you used in the past
* merge code from anki connect to make companion addon compatible with latest anki
* package AnnA on pypi or as a standalone package
* consider considering image occlusions of the same image as siblings

* investigate using parallel processing of TF_IDF instead of _format_text using the .preprocess() method

* notify LW
* create a gui : mainly by clicking on the gear of deck settings : AnnA > run AnnA with preset > each preset / run all active presets / About / configuration > acronym editor | field mappings | presets
* ask Anking to review it

* implement task "optimize_filtered_deck" to optimize the order of an already existing filtered deck without burying anything
* implement autonomous mode : mean distance across the queue probably follows an elbow curve with diminishing returns, this should help do only most important reviews
* look into sentence mining methods from sbert, that might contain useful ideas
* take a look at topic modeling techniques that might be cleaner to invoke than the current ctf-idf implementation
* re read this article for inspiration : http://mccormickml.com/2021/05/27/question-answering-system-tf-idf/
* automatically create a phylogeny of cards based on a distance matrix and see if it's an appropriate mind map, plotly is suitable for this kind of tree
* investigate crazy ideas list

## Credits and links that were helpful
* [Corentin Sautier](https://github.com/CSautier/) for his many many insights and suggestions on ML and python. He was also instrumental in devising the score formula used to order the filtered deck.
* [A post about class based tf idf by Maarten Grootendorst on Towardsdatascience](https://towardsdatascience.com/creating-a-class-based-tf-idf-with-scikit-learn-caea7b15b858)
* [The authors of sentence-bert and their very informative website](https://github.com/UKPLab/sentence-transformers)
* [The author of the addon anki-connect](https://github.com/FooSoft/anki-connect), as this project was indispensable when developing this addon. The companion addon is a reduced fork from anki-connect.
* [This plotly script that leverages NetworkX](https://github.com/roholazandie/graph_drawing) by github user [roholazandie](https://github.com/roholazandie) and the PR to make compatible with recent plotly versions by [anthng](https://github.com/anthng)

## Crazy ideas 
### The following is kept as legacy but was made while working on the ancestor of AnnA, don't judge please.
*Disclaimer : I'm a medical student extremely interested in AI but who has trouble devoting time to this passion. This project is a way to get to know machine learning tools but can have practical applications. I like to have crazy ideas to motivate my projects and they are listed belows. Don't laugh. Don't hesitate to contribute either.*
* **Scheduling** Using AnnA to do more different reviews might, somehow, increase the likelihood of [eureka moments](https://en.wikipedia.org/wiki/Eureka_(word)) where your brain just created a new neural paths. That would supposedly help to remember and master a field.
* **Optimize learning via cues** : A weird idea of mine is about the notion that hearing a tone when you're learning something will increase your recall if you have the same tone playing while taking the test. So maybe it would be useful to assign a tone that increases in pitch while you advance in the queue of due cards? I told you it was crazy ideas... Also this tone could play the tone assigned to the cluster when doing random reviews.
* **Mental Walk** (credit goes to a discussion with [Paul Bricman](https://paulbricman.com/)) : integrating AnnA with VR by offering the user to walk in an imaginary and procedurally generated world with the reviews materialized in precise location. So the user would have to walk to the flashcard location then do the review. The cards would be automatically grouped by cluster into rooms or forests or buildings etc. Allowing to not have to create the mental palace but just have to create the flashcards.
    * a possibility would be to do a dimension reduction to 5 dimensions. Use the 2 first to get the spatial location were the cluster would be assigned. Then the cards would be spread across the room but the 3 remaining vectors could be used to derive something about the room like wall whiteness, floor whiteness and the tempo of a music playing in the background.
    * we could ensure that the clusters would always stay in the same room even after adding many cards or even across users by querying a large language model for the vectors associated to the main descriptors of the cluster.
* **Source Walking** : it would be interesting to do the anki reviews in VR where the floor would be your source (pdf, epub, ppt, whatever). The cards would be spread over the document, approximately located above the actual source sentence. Hence leveraging the power of mental palace while doing your reviews. Accessing the big picture AND the small picture.

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
               [--resort_split] [--profile_name PROFILE_NAME]
               [--keep_console_open] [--sync_behavior SYNC_BEHAVIOR]

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
                        same card but on different days). Note that
                        'iterated_fetcher' also looks for cards in filtered
                        decks created by AnnA from the same deck. When
                        'iterated_fetcher' is used, the importance of reviews
                        is gradually decreased as the number of days since the
                        review grows. In short it's doing temporal
                        discounting. Default is `None`.
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
  --append_tags         Whether to append the tags to the cards content or to
                        add no tags. Default to `True`.
  --tags_to_ignore [TAGS_TO_IGNORE ...]
                        a list of regexp of tags to ignore when appending tags
                        to cards. This is not a list of tags whose card should
                        be ignored! Default is ['AnnA', 'leech']. Set to None
                        to disable it.
  --add_KNN_to_field    Whether to add a query to find the K nearestneighbor
                        of a given card to a new field called
                        'Nearest_neighbors' (only if already present in the
                        model). Be careful not to overwrite the fields by
                        running AnnA several times in a row! For example by
                        first burying learning cards then filtering review
                        cards. This argument is to be used if you want to find
                        the KNN only for the cards of the deck in question and
                        that are currently due. If you want to run this on the
                        complete deck you should use the 'task' argument.
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
  --TFIDF_stem          default to `False`. Whether to enable stemming of
                        words. Currently the PorterStemmer is used, and was
                        made for English but can still be useful for some
                        other languages. Keep in mind that this is the longest
                        step when formatting text.
  --dist_metric DIST_METRIC
                        when computing the distance matrix, whether to use
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
                        Resorting the new filtered deck taking onlyinto
                        account the semantic distance and not the reference
                        score. Useful if you are certain to review the
                        entierety of the filtered deck today as it will
                        minimize similarity between consecutive cards. If you
                        are not sure you will finish the deck today, set to
                        `False` to make sure you review first the most urgent
                        cards. This feature is active only if you set `task`
                        to 'filter_review_cards'. Can be either 'farther' or
                        'closer' or False. 'farther' meaning to spread the
                        cards as evenly as possible. Default to 'closer'.
  --resort_split        If 'resort_by_dist' is not False, set to True to
                        resort the boosted cards separately from the rest and
                        make them appear first in the filtered deck. Default
                        to `False`.
  --profile_name PROFILE_NAME
                        defaults to `None`. Profile named used by ankipandas
                        to find your collection. If None, ankipandas will use
                        the most probable collection. Only used if
                        'whole_deck_computation' is set to `True`
  --keep_console_open   defaults to `False`. Set to True to open a python
                        console after running.
  --sync_behavior SYNC_BEHAVIOR
                        If contains 'before', will trigger a sync when AnnA is
                        run. If contains 'after', will trigger a sync at the
                        end of the run. Default is `before&after`.

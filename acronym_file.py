# list of user defined acronym list:
# * for example if you add to this file "TACH": "tachycardia":
#    'John has TACH' becomes 'John has TACH tachycardia '
# * if the key (word to be replace) contains uppercase, the matching will be
#    case sensitive, otherwise matching will be case insensitive
# * the dictionnary name has to be the argument `acronym_list`
# * avoid using stop words as they will not be removed, for example :
#    "TNM": "tumor lymph node metastasis" instead of "tumor of the
#    lymph node with metastasis"
# * regexp matching is supported, this allows very complex string replacement
#    rules
# * acronyms replacement are not recursive BUT will be applied in order of
#    appearance, for example :
#            "HIV": "human immuno defficiency virus AIDS"
#            "AIDS": "acquired immuno defficiency syndrome"
#    will turns 'HIV' into 'HIV human immuno defficiency virus AIDS acquired
#    immuno defficiency syndrome'. This can be good or bad depending on your
#    usage

medical_terms = {r"IL(\d+)": r"interleukin \1",
                 "LT": "lymphocyte T",
                 "LB": "lymphocyte B",
                 "NK": "lymphocyte T natural killer",
                 "Th17": "Lymphocyte T helper 17",
                 "Th1": "Lymphocyte T helper 1",
                 "Th2": "Lymphocyte T helper 2",
                 "TNM": "tumor lymph node metastasis",
                 "REM": "rapid eye movement sleep",
                 "HPV": "human papilloma virus",
                 "BRCA": "breast cancer gene",
                 }

AI_machine_learning = {"AI": "artificial intelligence",
                       "ML": "machine learning",
                       "CNN": "convolutionnal neural network",
                       "NN": "neural network",
                       "NLP": "natural language processing",
                       "LSTM": "Long Short Term Memory",
                       }

# list of user defined acronym list:
# * for example with items "TACH": "tachycardia":
#    'John has TACH' becomes 'John has TACH (tachycardia)'
# * if the key contains uppercase, the matching will be case sensiive
#    otherwise matching will be case insensitive
# * the name of the dictionnary is what you put into
#    the `acronym_list` argument
# * avoid using stop words as they will not be removed
# * regexp matching is supported, this allows very complex string replacement
#    rules

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

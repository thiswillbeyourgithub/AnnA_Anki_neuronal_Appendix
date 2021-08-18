import argparse
import json
import urllib.request
import pyfiglet
import pandas as pd
from pprint import pprint
from tqdm import tqdm

##################################################
# arguments
parser = argparse.ArgumentParser()
parser.add_argument("-d",
                    "--deck",
                    help="the name of the deck that you wish to use",
                    dest="deck",
                    metavar="DECKNAME")
parser.add_argument("-v",
                    "--verbose",
                    help="increase verbosity, for debugging",
                    action='store_true',
                    dest="verbosity")
args = parser.parse_args().__dict__


##################################################
# anki related functions
def ankiconnect_invoke(action, **params):
    "send requests to ankiconnect addon"

    def request_wrapper(action, **params):
        return {'action': action, 'params': params, 'version': 6}

    requestJson = json.dumps(request_wrapper(action, **params)).encode('utf-8')
    if verb is True:
        pprint(requestJson)
    response = json.load(urllib.request.urlopen(urllib.request.Request(
                                                'http://localhost:8765',
                                                requestJson)))
    if verb is True:
        pprint(response)
    if len(response) != 2:
        raise Exception('response has an unexpected number of fields')
    if 'error' not in response:
        raise Exception('response is missing required error field')
    if 'result' not in response:
        raise Exception('response is missing required result field')
    if response['error'] is not None:
        raise Exception(response['error'])
    return response['result']


def get_deckname_list():
    "get the list of decks"
    return ankiconnect_invoke(action="deckNames")


def get_card_id_from_query(query):
    "get notes from query"
    return ankiconnect_invoke(action="findCards", query=query)


def get_cards_info_from_card_id(card_id):
    "get cardinfo from card id, works with either int of list of int"
    if isinstance(card_id, list):
        return ankiconnect_invoke(action="cardsInfo", cards=card_id)
    if isinstance(card_id, int):
        return ankiconnect_invoke(action="cardsInfo", cards=[card_id])


##################################################
# machine learning related functions
def get_cluster_topic(cluster_note_dic, all_note_dic, cluster_nb):
    "given notes, outputs the topic that is likely the subject of the cards"
    pass


def find_similar_notes(note):
    "given a note, find similar other notes with highest cosine similarity"
    pass


def find_notes_similar_to_input(user_input, nlimit):
    "given a text input, find notes with highest cosine similarity"
    pass


def show_latent_space(query):
    """
    given a query, will open plotly and show a 2d scatterplot of your cards
    semantically arranged.
    """
    # TODO test if the cards contain the relevant tags otherwise exit
    pass


##################################################
# main loop
if __name__ == "__main__":
    # printing banner
    ascii_banner = pyfiglet.figlet_format("AnnA")
    print(ascii_banner)
    print("Anki neuronal Appendix\n\n")

    # loading args
    deck = args["deck"]
    verb = args["verbosity"]

    # getting correct deck name
    decklist = get_deckname_list()
    ready = False
    while ready is False:
        candidate = []
        for d in decklist:
            if deck in d:
                candidate = candidate + [d]
        if len(candidate) == 1:
            ans = input(f"Is {candidate[0]} the correct deck ? (y/n)\n>")
            if ans != "y" and ans != "yes":
                print("Exiting.")
                raise SystemExit()
            else:
                deck = candidate[0]
                ready = True
        else:
            print("Several corresponding decks found:")
            for i, c in enumerate(candidate):
                print(f"#{i} - {c}")
            ans = input("Which deck do you chose?\
(input the corresponding number)\n>")
            print("")
            try:
                deck = candidate[int(ans)]
                ready = True
            except (ValueError, IndexError) as e:
                print(f"Wrong number: {e}\n")
                ready = False



    # extracting card list
    print("Getting due card from this deck...")
    due_cards = get_card_id_from_query(f"deck:{deck} is:due is:review -is:learn")

    print("Getting cards that where rated in the last week from this deck...")
    rated_cards = get_card_id_from_query(f"deck:{deck} rated:7")
    
    # checks that rated_cards and due_cards don't overlap
    overlap = []
    for r in rated_cards:
        if r in due_cards:
            print(f"Card with id {r} is both rated in the last 7 days and due!")
            overlap = overlap + [r]
    if overlap != []:
        print("This should never happen!")
        raise SystemExit()

    # extracting card information
    all_rlvt_cards = rated_cards + due_cards
    print(f"Getting information from relevant {len(all_rlvt_cards)} cards...")
    list_cardsInfo = get_cards_info_from_card_id(all_rlvt_cards)

    # creating dataframe
    df = pd.DataFrame(columns=["cardId"])
    df.set_index("cardId")
    for i in list_cardsInfo:
        i = dict(i)
        df = df.append(i, ignore_index=True)
    





##################################################
# TEMPORARY JAILED
def add_tag_to_card_id(card_id, tag):
    "add tag to card id"
    # first gets note it from card id
    note_id = ankiconnect_invoke(action="cardsToNote", cards=card_id)
    return ankiconnect_invoke(action="addTags",
                              notes=note_id,
                              tags=tag)


def add_vectorTags(note_dic, vectorizer, vectors):
    "adds a tag containing the vector values to each note"
    if vectorizer not in ["sentence_bert", "tf-idf"]:
        print("Wrong vectorTags vectorizer")
        raise SystemExit()
    return ankiconnect_invoke(action="addTags",
                              notes=note_id,
                              tags=vectors_tag)


def add_cluterTags(note_dic):
    "add a tag containing the cluster number and name to each note"
    pass


def add_actionTags(note_dic):
    """
    add a tag containing the action that should be taken regarding cards.
    The action can be "bury" or "study_today"
    Currently, you then have to manually bury them or study them into anki
    """

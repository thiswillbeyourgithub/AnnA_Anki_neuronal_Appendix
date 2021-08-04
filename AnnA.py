import argparse
import json
import urllib.request


def ankiconnect_invoke(action, **params):
    "sends requests to ankiconnect addon"

    def request_wrapper(action, **params):
        return {'action': action, 'params': params, 'version': 6}

    requestJson = json.dumps(request_wrapper(action, **params)).encode('utf-8')
    response = json.load(urllib.request.urlopen(urllib.request.Request('http://localhost:8765', requestJson)))
    if len(response) != 2:
        raise Exception('response has an unexpected number of fields')
    if 'error' not in response:
        raise Exception('response is missing required error field')
    if 'result' not in response:
        raise Exception('response is missing required result field')
    if response['error'] is not None:
        raise Exception(response['error'])
    return response['result']


#result = invoke(action='notesInfo', notes=[1621177123982])


def get_deck_list():
    "gets the list of decks"
    pass


def get_notes_from_query(query):
    pass

def add_vectorTags(note_dic, vectorizer):
    "adds a tag containing the vector values to each note"
    if mode not in ["sentence_bert", "tf-idf"]:
        print("Wrong vectorTags vectorizer")
        raise SystemExit()
    pass


def add_cluterTags(note_dic):
    "add a tag containing the cluster number and name to each note"
    pass

def get_cluster_topic(cluster_note_dic, all_note_dic, cluster_nb):
    "given notes, outputs the topic that is likely the subject of the cards"
    pass

def add_actionTags(note_dic):
    """
    add a tag containing the action that should be taken regarding cards.
    The action can be "bury" or "study_today"
    Currently, you then have to manually bury them or study them into anki
    """

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







# this script allows to run AnnA several times in a row
# to create several filtered decks

# if you have low memory, declare all the instance "a" instead
# of a, then b, then c etc.

from AnnA import *

a = AnnA(deckname="Some_Deck",
        desired_deck_size="100",
        rated_last_X_days=3,
        reference_order="relative_overdueness",
        do_clustering=True,
        send_to_anki=True,
        )
b = AnnA(deckname="Some_OtherDeck::subdeck#1",
        desired_deck_size="80%",
        rated_last_X_days=3,
        reference_order="lowest_interval",
        do_clustering=True,
        send_to_anki=True
        )
c = AnnA(deckname="Some_OtherDeck::subdeck#2",
        desired_deck_size="50",
        rated_last_X_days=3,
        reference_order="lowest_interval",
        do_clustering=True,
        send_to_anki=True
        )

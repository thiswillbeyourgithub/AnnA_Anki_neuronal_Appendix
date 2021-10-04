# this script allows to run AnnA several times in a row
# to create several filtered decks

# if you have low memory, declare all the instance "a" instead
# of a, then b, then c etc.

from AnnA import *

def interscreen():
    print("", end="\n\n")
    for i in range(5):
        print("="*10)
    print("", end="\n\n")


a = AnnA(deckname="Some_Deck",
        desired_deck_size="100",
        rated_last_X_days=3,
        reference_order="relative_overdueness",
        do_clustering=True,
        to_anki=True,
        )
interscreen()
b = AnnA(deckname="Some_OtherDeck::subdeck#1",
        desired_deck_size="80%",
        rated_last_X_days=3,
        reference_order="lowest_interval",
        do_clustering=True,
        to_anki=True
        show_banner=False,
        )
interscreen()
c = AnnA(deckname="Some_OtherDeck::subdeck#2",
        desired_deck_size="50",
        rated_last_X_days=3,
        reference_order="lowest_interval",
        do_clustering=True,
        to_anki=True,
        check_database=True
        show_banner=False,
        )

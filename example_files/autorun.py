# this script allows to run AnnA several times in a row

# if you have low memory, declare all the instance "a" instead
# of a, then b, then c etc.

from AnnA import *

def interscreen():
    print("", end="\n\n")
    for i in range(5):
        print("="*10)
    print("", end="\n\n")


AnnA(deckname="Some_Deck",
        task="filter_review_cards",
        target_deck_size="100",
        rated_last_X_days=3,
        reference_order="relative_overdueness",
        )

interscreen()

AnnA(deckname="Some_OtherDeck::subdeck#1",
        task="bury_excess_learning_cards",
        target_deck_size="80%",
        rated_last_X_days=3,
        reference_order="lowest_interval",
        )

interscreen()

AnnA(deckname="Some_OtherDeck::subdeck#2",
        task="bury_excess_review_cards",
        target_deck_size="50",
        rated_last_X_days=3,
        check_database=True
        )

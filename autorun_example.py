from AnnA import *

a = AnnA(deckname="Some_Deck",
        desired_deck_size="100",
        rated_last_X_days=3,
        reference_order="relative_overdueness",
        do_clustering=True,
        send_to_anki=True,
        )
a = AnnA(deckname="Some_OtherDeck::subdeck#1",
        desired_deck_size="80%",
        rated_last_X_days=3,
        reference_order="lowest_interval",
        do_clustering=True,
        send_to_anki=True
        )
a = AnnA(deckname="Some_OtherDeck::subdeck#2",
        desired_deck_size="50",
        rated_last_X_days=3,
        reference_order="lowest_interval",
        do_clustering=True,
        send_to_anki=True
        )

# this script allows to run AnnA several times in a row

# if you have low memory, declare all the instance "a" instead
# of a, then b, then c etc.

if __name__=='__main__':
    from AnnA import *

    def interscreen():
        print("", end="\n\n")
        for i in range(5):
            print("="*10)
        print("", end="\n\n")


    AnnA(deckname="Some_Deck",
            task="filter_review_cards",  # creates a new filtered deck with your reviews in the right order
            target_deck_size="50%",  # you will do half of the reviews
            rated_last_X_days=3,  # consider similarity with reviews of today and the past 3 days
            acronym_list=["medical_terms"],
            reference_order="relative_overdueness",  # use this filtered deck sorting order as reference order to alter based on similarity
            )

    interscreen()

    AnnA(deckname="Some_OtherDeck::subdeck#1",
            task="bury_excess_learning_cards",  # don't create a filtered deck and just bury the cards not to review today ; only takes into account learning cards
            target_deck_size="deck_settings",
            rated_last_X_days=3,
            acronym_list=["medical_terms", "AI_machine_learning"],
            reference_order="lowest_interval",
            )

    interscreen()

    AnnA(deckname="Some_OtherDeck::subdeck#2",
            task="bury_excess_review_cards",  # don't create a filtered deck and just bury the cards not to review today ; only takes into account reviews
            target_deck_size="50",
            acronym_list=None,
            rated_last_X_days=3,
            )

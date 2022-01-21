# Routine of the author of AnnA
This page shows one way AnnA can be used.

## A few notes:
* I am a medschool student and my steps are over multiple days (something like 12h 2d 4d then 7 day graduation)
* My True Retention rate is above 90% (but that might be because I have long learning steps so hard cards can spend weeks in learning).
* I created AnnA recently so my usage will probably vary. Don't hesitate to open an issue if you want to give my your opinion.
* I am using the V2 scheduler

## Current routine:
* Every morning, I open anki and take a look at my medschool decks:
    1. I check how many *learning cards* and *reviews* I have to do. Say it's `300` reviews and `150` learnings.
    2. I then open the file autorun.py, and adjust the settings to what would be best depending on my mood, schedule, etc. I usually let it to `50%` of the reviews and `80%` of the learnings.
    3. I run my autorun.py, resulting in a `20%` drop in the number of learning cards in the original deck and a new filtered deck containing about `150` of my reviews.
    4. I then refill two (regular) filtered deck for the learning cards using as settings:
        *I exclude the cards rated today because it tends to mess with my learning steps*
        1. `deck:med_school_deck is:due is:learn -is:buried -rated:1 -rated:2:1 -rated:2:2` (ordered by random, will contain the learning cards that I struggled with the day before)
        2. `deck:med_school_deck is:due is:learn -is:buried -rated:1 (rated:2:1 or rated:2:2)` (ordered by random, contains the easier learning cards)
    5. I work my way through all those decks. This will ensure I reviewed all the useful cards of the day, but when some similar cards would have appeared on the same day, one of them has been buried. I think this is the most efficient way to go.


## Further notes:
* During weekends, I tend to do 50% of the learning and 20% of the reviews.
* If I have a huge backlog, say 1000 cards. I can spread it over 10 days using 100 cards per day while aggressively penalizing similarity by setting the argument `rated_last_X_days` to 6 days for example.
* At the end of each semester, I add my most important decks to an `Archive` deck. Every day, I review 10 maximally spread cards over all my past decks and ordered by `lowest_interval`. The idea is to keep decently sharp on all the past cards I've created without taking too much of my time.

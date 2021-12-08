# Routine of the author of AnnA
This page shows one way AnnA can be used.


## A few notes:
*  I am a medschool student and my steps are over multiple days (something like 12h 2d 4d then 7 day graduation)
*  My True Retention rate is above 90% but that might be because I have long learning steps so hard cards can spend weeks in learning.
*  I created AnnA less than a year ago so expect me to grow along the way.


## Current routine:
* During the week:
    * Every morning, I open anki then take a look at my medschool decks:
        * I check how many *learning cards* I have to do today, say it's 300.
        * I check how many *reviews* I have due, say it's 150.

        * I then open the file autorun.py, and adjust the settings to what would be best depending on my mood, schedule, etc. I usually settle for 50% of the reviews and 80% of the learnings.
        * I run autorun.py: the result is 1. a new filtered deck containins 50% of my due reviews and 2. the number of learning cards that are not buried is down 20%.

        * I review the cards in the filtered deck.
        * I then also create another filtered deck with as settings `deck:med_school_deck is:due is:learn -is:buried -is:suspended -rated:1`, ordered by order added. I work my way through it too. It contains the learning cards, in reverse order of creation, but when some cards are too similar, one of them get buried. I think this is the most efficient way to go.

* During the weekend:
    * I sometimes have very limited time to devote to medschool on the weekends so I take only the 20% most different cards and I think it still helps a lot.

* In the event of a backlog:
    * If I'm in a hurry:
         * I will create large optimized filtered decks and do them over a few days.
    * If I want to take pleasure while doing it:
         * I will add a section to autorun.py to create a small filtered deck every morning and do it religiously.

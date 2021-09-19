# AnnA : Anki neuronal Appendix
Tired of having to deal with anki flashcards that are too similar when grinding through your backlog? This python script creates filtered deck in optimal review order. It uses Machine Learning / AI to make semantically linked cards far from one another.

## One sentence summary
When you don't have the time to complete all your daily reviews, use this to create a special filtered deck that makes sure you will only review the cards that are most different from the rest of your reviews.

## Other features
* Look for cards in your database using a semantic query (typing something with a close `meaning` to a card will find it even if no words are in common.)
* Cluster your card collection using various algorithms (k-means (including minibatch k-means), DBSCAN, agglomerative clustering). Get the topic of each cluster and add it as tag to your cards.
* Create a plot showing clusters of semantic meanings from your anki database. As you can see on this picture (click to see it in big):
<img title="Colored by tags" src="by_tags.png" width="250" height="250"/> <img title="Colored by clusters" src="by_clusters.png" width="250" height="250"/>
* Code is PEP compliant, dynamically typed, all the functions have a detailed docstrings. Contribution is welcome, opening issues is encouraged.




## FAQ
* **What do you call "optimal review order"?** The order that minimizes the chance of reviewing similar cards in a day. You see, Anki has no knowledge of the content of cards and only cares about their interval. Its built-in system of "siblings" is useful but I think we can do better. AnnA was made to create filtered decks with cards in "lower interval" order BUT in a way that keeps semantic siblings far from each other.
* **What is this for?** It seems a great idea for dealing with huge backlog you get in medical school. If you have 2 000 reviews to do, but can only do 500 in a day: AnnA is making sure that you will get the most out of those 500. I don't expect the plotting and clustering features to be really used but I had to code them to make sure AnnA was working fine so I might as well leave it :)
* **What is the current status of this project?** I use it daily and intend to keep developing until I have no free time left. Take a look at the TODO list if you want to know what's in the works.
* **How does it work? (Layman version)** Magic.
* **How does it work? (Technical version)** The code is mainly composed of a python class called AnnA. When you create an instance, you have to supply the name of the deck you want to filter from. It will then automatically fetch the cards from your database, assign [sentence-BERT](https://www.sbert.net/) vectors to each. Compute a cosine distance matrix of your cards. You can then call different methods, the most useful will be `a.send_to_anki()` which will create the filtered deck containing the cards, then alter them to get the optimal review order. Note that rated cards of the last few days of the same deck will be used as reference to avoid having cards that are too similar to previous reviews. If you want to know more, either ask me or read the docstrings.
* **What are the power requirements to run this?** I took care to make it as efficient as possible but am still improving the code. Computing the distance matrix can be long if you do this on very large amount of cards but this step is done in parallel. Let me know if some steps are unusually large and I will try to optimize it. With one argument you can use PCA to do a dimension reduction on your cards, making the rest of the script faster.
* **Do you mind if I open an issue?** Not at all! It is very much encouraged, even just for a typo. That will at least make me happy. Of course PR are always more than welcome.
* **What is the `keep_ocr` argument?** If you set this to True when creating the instance, AnnA will extract the text from images, provided you create it with [Anki OCR](https://github.com/cfculhane/AnkiOCR/) addon beforehand.
* **If I create a filtered deck using AnnA, can I rebuild it?** No, emptying or rebuilding it will use anki's order and not AnnA's. You have to run the script each time you want to refill your deck. Hence you should create large filtered decks in advance. 
* **Does this work only on Linux?** It should work on all platform, provided you have anki installed and [anki-connect](https://github.com/FooSoft/anki-connect) enabled. But it uses some dependencies that might only work on some CPUs, so I'm guessing ARM system would not work but please tell me if you know tried.
* **What is the cache?** The main bottleneck was creating all the vector embeddings of the cards, so I decided to automatically store them in a pickled dataframe.
* **What is sBERT?** Shot for sentence-BERT. BERT is a machine learning technology that allows to assign high dimensional vectors to words in the context of a sentence. Sentence-BERT is a new method that does essentially the same thing but on a whole sentence. You can read more [at their great website](https://www.sbert.net/).
* **sBERT is interesting but I'd like to use tf-idf, is this possible?** I initially tried with it and with both combined but it was creating a messy code, you can't cache tf-idf, it slows down the script a lot because SVD does not seem as efficient, using BERT tokenizer and tf-idf means adding more than 20 parameters that I was not sure about. So I decided to go with only sBERT and it's awesome!
* **Can this be made into an anki addon instead of a python script?** I have never packaged things into anki addons so I'm not so sure. I heard that packaging complexe modules into anki is a pain, and cross platform will be an issue. But I think that I could store the vector information in the `data` column of each card in anki's database. Then rebuilding the deck can be done with only simple math and no libraries. If you'd like to make this a reality, show yourself by openning an issue! I would really like this to be implemented into anki, as the search function would be pretty nice :)
* **Can I change sBERT's model?** Yes, the model by default is a multilanguage model but the english only is supposedly even more awesome. You should consider using it if your cards are only in english.²
* **Will this change anything to my anki collection?** Yes, to add the cards to the filtered deck, I had to add a flag to cards then remove it. This means that the `mod` value assigned to each card (indicating when was the most recent change made) will be modified. I intend to ask the devs for a way to bypass this.


## How to use it
* First, **read this page in its entierety, this is a complicated piece of software and you don't want to use it irresponsibly on your cards.**
* make sure the addon [anki-connect](https://github.com/FooSoft/anki-connect) is installed
* Clone this repository: `git clone https://github.com/thiswillbeyourgithub/AnnA_Anki_neuronal_Appendix ; cd AnnA_Anki_neuronal_Appendix`
* Use python to install the necessary packages : `pip install -r requirements.py`
* Open ipython: `ipython3`
* `from AnnA import * ; a = AnnA(desired_deck_size=500, dont_send_to_anki=False)`
* *wait a while, the first time you run it on a deck is long because sBERT has to compute all the embeddinsg*
* Enjoy your filtered deck, but don't empty it or rebuilt it.
* Open an issue telling me your remarks and suggestion



## TODO
### URGENT:
* change dont send to send and false to true etc
* ask damien how to add a flag without changing mod
* remove mention of chunked distance matrix as it was not what I thought
* optimize review order computation
* show usage and describe each argument of the main class

* add tags containing cluster_topics
* add a method that shows all acronyms
* add a method that displays the cards in best order, to troubleshoot issues
* add a method to bury the card without creating a filtered deck
* take a look at topic modelling techniques that might be cleaner to invoke than ctf-idf
* reddit + anki discord + psionica + anking + blum 
* pickle the whole class at the end, to be able to rerun it in case something went wrong

### long term
* re read this article : http://mccormickml.com/2021/05/27/question-answering-system-tf-idf/
* talk about this in the anki dev channel and create a subreddit post
* automatically create a phylogeny of cards based on a distance matrix and see if it's an appropriate mind map, plotly is suitable for this kind of tree
* port into anki as an addon
* investigate VR, minecraft integration, roblox, etc


# The following is kept as a legacy but was more based on the non functional ancestor to AnnA, don't judge please.
## Crazy ideas 
*Disclaimer : I'm a medical student extremely interested in AI but who has trouble devoting time to this passion. This project is a way to get to know machine learning tools but can have practical applications. I like to have crazy ideas to motivate my projects and they are listed belows. Don't laugh. Don't hesitate to contribute either.*
* **Scheduling** If you have 100 reviews to do in a given anki session, anki currently shows them to you using `relative overdueness` (+ fuzzing) order. But is there a better way? What happens if you group semantically related cards closest to each other? That would probably be "cheating" as answering the `n` cards will remind you enough to answer `n+1`. So maybe the opposite would be useful : ordering the cards by being the farthest apart, semantically.
    * that might, somehow, increase the likelihood of [eureka](https://en.wikipedia.org/wiki/Eureka_(word)) moments where your brain just created a new paths. That would supposedly help to remember and master a field.
    * another way would be simply to automatically bury the closest cards to each notion. That would intelligently reduce the load of each day in a "smart" and efficient way.
* **Show related** : have a card about a notion you just can't fit into the big picture? I could have a button showing similar cards!
* **Optimize learning via cues** : A weird idea of mine is about the notion that hearing a tone when you're learning something will increase your recall if you have the same tone playing while taking the test. So maybe it would be useful to assign a tone that increases in pitch while you advance in the queue of due cards? I told you it was crazy ideas... Also this tone could play the tone assigned to the cluster when doing random reviews.
* **Mental Walk** (credit goes to a discussion with Paul Bricman) : integrating AnnA with VR by offering the user to walk in an imaginary and procedurally generated world with the reviews materialized in precise location. So the user would have to walk to the flashcard location then do the review. The cards would be automatically grouped by cluster into rooms or forests or buildings etc. Allowing to not have to create the mental palace but just have to create the flashcards.
    * a possibility would be to do a dimension reduction to 5 dimensions. Use the 2 first to get the spatial location were the cluster would be assigned. Then the cards would be spread across the room but the 3 remaining vectors could be used to derive something about the room like wall color, the tempo of a music playing when inside the room and I don't know, the color of the floor?
    * we could ensure that the clusters would always stay in the same room even after adding many cards or even across users by querying a large language model for the vectors associated to the main descriptors of the cluster.
* **Source Walking** : it would be interesting to do the anki reviews in VR where the floor would be your source (pdf, epub, ppt, whatever). The cards would be spread over the document, approximately located above the actual source sentence. Hence leveraging the power of mental palace while doing your reviews. Accessing the big picture AND the small picture.
* **others** : incoming!


## Credits and links that helped me a lot
    * [Corentin Sautier](https://github.com/CSautier/) for his many many insights and suggestions on ML and python. He was also instrumental in devising the score formula used to order the filtered deck.
    * [A post about class based tf idf by Maarten Grootendorst on Towardsdatascience](https://towardsdatascience.com/creating-a-class-based-tf-idf-with-scikit-learn-caea7b15b858)
    * [The authors of sentence-bert and their very informative website](https://github.com/UKPLab/sentence-transformers)

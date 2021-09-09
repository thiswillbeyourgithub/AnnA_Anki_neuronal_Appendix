# AnnA : Anki neuronal Appendix
# **Careful, this project is currently being refactored from scratch, ETA : end of summer 2021**
Plot the latent space of the semantic relation of your anki cards showing clusters of related notions. In other words, this finds "semantic siblings" in your collection.



## FAQ
* **What is this? And what is it for?** This script (which I plan to integrate as an anki addon) uses machine learning to render a 2D plot that shows cards that deal with similar notions next to each other. It is only based on the text content of the card (all fields) and not any other (meta)data. It's final use is still to be determined but I'm especially interested in creating filtered decks that uses this to **not** pick cards that are dealing with notions that are too related. Effectively reducing the daily load of cards without meaningfully reducing recall. I am very interested in NLP (natural language processing) and am sure there would be other interesting projects based on this. See at the bottom the section `Crazy Ideas`.
* **What is the current status of this project?** I am still developing it. My free time is limited but I often have a more polished version on my laptop and push the commit way later so don't be scared if you don't see anything new for months. The machine learning part is still in progress but has worked but I don't know Qt and never really made anki addons so it will take some time to integrate as an addon. Take a look at the TODO list to know more.
* **Can you explain the name?** The idea is to somehow show on 2D a representation of your knowledge. The brain, its consciousness and memories can be thought of as just a skier skiing down hyper dimensional slopes of potential energy curves to find lowest points. Those minimum are actually "concepts". 
* **How does it work? (Layman version)** Magic.
* **How does it work? (Technical version)** (*Subject to change*) Here's what the code does, section by section. It loads your anki database into pandas. Do a bunch of unnecessarily complicated stuff to get the tags, decks and notetype of each cards into a pandas DataFrame. Removes the duplicate cards (i.e. cards containing the same text content, that's at least almost all image occlusion siblings). Removes the first field of image occlusion cards as it contains a field ID which is annoying for NLP. Removes the stop words. Formats the text to remove html, extract the already [OCRed text](https://ankiweb.net/shared/info/450181164) (i.e. removing html link to image but keeping the `title` part of the html image).Readjusting word counts prior to [tf-idf](https://en.wikipedia.org/wiki/Tf%E2%80%93idf), i.e.  it copies the `sfld` over and over until the number of words from `sfld` is twice the number of words from `flds minus sfld`. This acts as a way to assign priority to the words from `sfld` instead of `flds`, which can be very useful to people like me who often have a lot of pictures in a `Source` field that have been OCRed. Then it tokenizes the cards using currently the model `bert-base-multilingual-uncased`. Then applies the `tf-if` algorithm.  Then does a dimensionality reduction using UMAP. Then plots the latent space using plotly.
* **What value should I choose be the number of K clusters?** That's a loaded question. It depends on your cards and on your computer. More clusters needs much more time but won't hurt the result that much. I would say go with the highest that won't take too long to run.
* **What are the power requirements to run this?** This should be rather efficient but if you're on an old laptop and have tens of thousands of cards it could be pretty demanding I think. On my pretty powerful laptop for 5000 cards it takes about 1 minute, mostly because of pandas preprocessing. Any additional data points will be interesting to me so if you want to share how long it takes don't hesitate to share it by opening an issue.
* **Do you mind if I open an issue?** Not at all! It is very much encouraged, even just for a typo. That will at least make me happy. Of course PR are always more than welcome.


## A few notes 
* This has been tested on python 3.9.0, and on linux. I intend to make it easier to run on all OS but it's already fairly easy (just change a few paths and `os.system()` commands.)
* I initially developed it as a jupyter notebook, so you will find both the last notebook and the finished script in this repository. Keep in mind that will probably add fixes to the python script and not the jupyter notebook


## How to use it
* First, **read this page in its entierety**
* Clone this repository : `git clone URL`
* Use python to install the necessary packages : `pip install -r requirements.py`
* make sure the addon anki-connect is installed
* Finally, don't hesitate to open an issue.


## TODO
### optimization
* optimize review order computation
* rewrite the README + tell about docstrings and PEP + tried tf-idf + use another sbert if english + tell that it interferes with mod time
* add tags after clusters
* do a simple parallelization wrapper for the tqdm loops at the end: https://stackoverflow.com/questions/9786102/how-do-i-parallelize-a-simple-python-loop
* run the script on notes and not cards
* implement relative overdueness
* ponder using df.compare to fetch more quickly the cached sbert vectors


### long term
* add support for miniBatchKMeans and HDBSCAN
* re read this article : http://mccormickml.com/2021/05/27/question-answering-system-tf-idf/
* talk about this in the anki dev channel and create a subreddit post
* automatically create a phylogeny of cards based on a distance matrix and see if it's an appropriate mind map, plotly is suitable for this kind of tree
* port into anki as an addon
* investigate VR, minecraft integration, roblox, etc


## Crazy ideas 
*Disclaimer : I'm a medical student extremely interested in AI but who has trouble devoting time to this passion. This project is a way to get to know machine learning tools but can have practical applications. I like to have crazy ideas to motivate my projects and they are listed belows. Don't laugh. Don't hesitate to contribute either.*
* **Scheduling** If you have 100 reviews to do in a given anki session, anki currently shows them to you using `relative overdueness` (+ fuzzing) order. But is there a better way? What happens if you group semantically related cards closest to each other? That would probably be "cheating" as answering the `n` cards will remind you enough to answer `n+1`. So maybe the opposite would be useful : ordering the cards by being the farthest apart, semantically.
    * that might, somehow, increase the likelihood of [eureka](https://en.wikipedia.org/wiki/Eureka_(word)) moments where your brain just created a new paths. That would supposedly help to remember and master a field.
    * another way would be simply to automatically bury the closest cards to each notion. That would intelligently reduce the load of each day in a "smart" and efficient way.
* **Show related** : have a card about a notion you just can't fit into the big picture? I could have a button showing similar cards!
* **Train on hardest clusters** : know that you have notions in the deck that you can't understand? This could help you create a filtered deck containing cards from closest to farthest from the center of the cluster. Allowing you to really connect the notion with the rest.
* **Optimize learning via cues** : A weird idea of mine is about the notion that hearing a tone when you're learning something will increase your recall if you have the same tone playing while taking the test. So maybe it would be useful to assign a tone that increases in pitch while you advance in the queue of due cards? I told you it was crazy ideas... Also this tone could play the tone assigned to the cluster when doing random reviews.
* **Mental Walk** (credit goes to discussing with Paul Bricman) : integrating AnnA with VR by offering the user to walk in an imaginary and procedurally generated world with the reviews materialized in precise location. So the user would have to walk to the flashcard location then do the review. The cards would be automatically grouped by cluster into rooms or forests or buildings etc. Allowing to not have to create the mental palace but just have to create the flashcards.
    * a possibility would be to do a dimension reduction to 5 dimensions. Use the 2 first to get the spatial location were the cluster would be assigned. Then the cards would be spread across the room but the 3 remaining vectors could be used to derive something about the room like wall color, the tempo of a music playing when inside the room and I don't know, the color of the floor?
    * we could ensure that the clusters would always stay in the same room even after adding many cards or even across users by querying a large language model for the vectors associated to the main descriptors of the cluster.
* **Source Walking** : it would be interesting to do the anki reviews in VR where the floor would be your source (pdf, epub, ppt, whatever). The cards would be spread over the document, approximately located above the actual source sentence. Hence leveraging the power of mental palace while doing your reviews. Accessing the big picture AND the small picture.
* **others** : incoming!


## Credits and links that helped me a lot
    * [Corentin Sautier](https://github.com/CSautier/) for his many many insights and suggestions on ML and python. He was also instrumental in devising the score formula used to order the filtered deck.
    * [A post about class based tf idf by Maarten Grootendorst on Towardsdatascience](https://towardsdatascience.com/creating-a-class-based-tf-idf-with-scikit-learn-caea7b15b858)
    * [The authors of sentence-bert and their very informative website](https://github.com/UKPLab/sentence-transformers)

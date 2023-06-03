# The variable field_dic contains as key a notetype model name and as value the
#    name of the fields to keep. For example:
#         occlusion": ["Header", "Image"]
#     will match all templates that contain "occlusion" in their name, and
#     make sure that AnnA takes into account the fields Header and Image
# * If you add a field multiple times, it will weight more than the other fields
# * The tags of the card is appended to the text anyway
# * Matching for note type or field name is case insensitive
# * To take all fields of a note type into account you can use the following:
#      "example_card": ["take_all_fields"]


field_dic = {
    "clozolkor": ["Header",
                  "Body", "Body", "Body",
                  "Hint",
                  "More"],
    # shameless plug: clozolkor is my own template,
    # more info: https://github.com/thiswillbeyourgithub/Clozolkor
    "image occlusion enhanced": ["Image",
                                 "Header",
                                 "Footer",
                                 "Remarks",
                                 "Extra 1",
                                 "Extra 2",
                                 "Sources"],
    "logseq_filesmodel": ["Text", "Extra"],
    "basic": ["Front", "Back", "Hint"],
    "Basic (and reversed card)": ["Front", "Back"],
    "spanish cards": ["Spanish", "English"],
    "Spanish sentences cloze": ["SpanishSentence", "EnglishSentence"],
    "____spanish sentence - Sans-serif-light": ["Front", "Back"],
    "morse": ["Front", "mnemonic"],
    "fallacy_bias": ["Name & Picture", "Summary", "Description"],
}

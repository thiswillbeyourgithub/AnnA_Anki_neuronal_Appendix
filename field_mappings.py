# The variable field_dic contains as key a notetype model name and as value the
#    name of the fields to keep. For example:
#         occlusion": ["Header", "Image"]
#     will match all templates that contain "occlusion" in their name, and
#     make sure that AnnA takes into account the fields Header and Image
# * If you add a field multiple times, it will weight more than the other fields
# * The tags of the card is appended to the text anyway
# * Matching for note type or field name is case insensitive

field_dic = {
             "clozolkor": ["Header", "Header",
                           "Body", "Body", "Body", "Body", "Body",
                           "Hint",
                           "More"],
             # shameless plug: clozolkor is my own template,
             # more info: https://github.com/thiswillbeyourgithub/Clozolkor
             "occlusion": ["Header", "Header", "Header", "Header", "Header",
                           "Footer", "Footer",
                           "Remarks", "Remarks",
                           "Extra 1", "Extra 1",
                           "Extra 2", "Extra 2",
                           "Image"],
             "basic": ["Front", "Back", "Hint"],
             "basic (and reversed card)": ["Front", "Back"],
             "spanish cards": ["Spanish", "English"],
             "morse": ["Front", "mnemonic"],
             "fallacy_bias": ["Name & Picture", "Summary", "Description"],
             }


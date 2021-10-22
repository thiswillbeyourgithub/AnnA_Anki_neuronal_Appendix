# This variable contains as key a notetype model name and as value the
# name of the fields to keep, in order
# * for example : "occlusion": ["Header", "Image"] will match all templates that
#     contain "occlusion" in their name
# * note that matching for note type or field name is case insensitive

field_dic = {
             "clozolkor": ["Header", "Body", "Hint", "More"],
             # shameless plug: clozolkor is my own template,
             # more info: https://github.com/thiswillbeyourgithub/Clozolkor
             "occlusion": ["Header", "Image"],
             "basic": ["Front", "Back", "Hint"],
             "spanish cards": ["Spanish", "English"]
             }


# This variable contains as key a notetype model name and as value the
# name of the fields to keep, in order
# ex : "occlusion": ["Header", "Image"]
# will match all templates that contain "occlusion" in their name

field_dic = {
             "clozolkor": ["Header", "Body", "Hint", "More"],
             # shameless plug: clozolkor is my own template,
             # more info: https://github.com/thiswillbeyourgithub/Clozolkor
             "occlusion": ["Header", "Image"],
             "spanish cards": ["Spanish", "English"]
             }


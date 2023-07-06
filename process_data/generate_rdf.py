import os
from tqdm import tqdm

to_file_name = "test"

def text_format(text):
    text = text.lower().strip()
    text = text.replace(",", "")
    text = text.replace("(", "").replace(")", "")
    text = text.replace(" -- ", "")
    text = text.replace("-", " ")
    text = text.replace("_", " ")
    text = text.replace('"', "'")
    text = text.replace(".", "")
    text = text.replace("\n", ". ")

    return text

with open("./raw_data/" + to_file_name + "_src", "r") as f:
    train_src = f.readlines()

if to_file_name != "test":
    with open("./raw_data/" + to_file_name + "_tgt", "r") as f2:
        train_tgt = f2.readlines()

with open("./data/rdf/" + to_file_name + ".csv", "w") as f:
    f.write("triple,text\n")
    for ind, content in enumerate(tqdm(train_src)):
        if to_file_name != "test":
            text = '"' + text_format(train_src[ind]) + '","' + text_format(train_tgt[ind]) + '"\n'
        else:
            text = '"' + text_format(train_src[ind]) + '","' + "" + '"\n'
        f.write(text)
        
        
    

from tqdm import tqdm
import os
import sys
import logging
from multiprocessing import Pool
import datasets
import spacy
import warnings
import random
from random_word import RandomWords
warnings.filterwarnings("ignore")

nlp = spacy.load('en_core_web_lg')
# dataset = datasets.load_dataset("lighteval/summarization","cnn-dm")
dataset = datasets.load_dataset("cnn_dailymail","1.0.0")
# print(dataset["train"][1000])

output_dir = "./data/cnn/"
os.makedirs(output_dir, exist_ok=True)

dataset_type = "test"

logging.basicConfig(filename="./exception.log", encoding="utf-8",
                    level=logging.DEBUG)
logging.getLogger("urllib3").setLevel(logging.ERROR)

thread_num = 3
threshold = 0.5
threshold2 = 0.7 # for negative sample
def text_format(text):
    text = text.lower()
    text = text.replace(" -- ", "")
    text = text.replace("-", " ")
    text = text.replace("_", " ")
    text = text.replace('"', "'")
    text = text.replace(".", "")
    text = text.replace("\n", ". ")
    return text

def run_task(ind):
    try:
        r = RandomWords()

        fp = open(output_dir + dataset_type + ".csv", "a")
        element = dataset[dataset_type][ind]
        text = element["article"]
        summary = element["highlights"]

        doc = nlp(summary)
        rand = random.random()
        # if rand < threshold:
        ents = [(e.text, e.label_) for e in doc.ents if e.label_ not in 'CARDINAL']
        # else:
        #     ents = [(e.text, e.label_) for e in doc.ents if e.label_ not in ('DATE', 'TIME', 'ORDINAL', 'CARDINAL')]

        entities = []
        if len(ents) >= 3:

            for ent in ents:
                entity = ent[0]
                entity = text_format(entity)
                if entity not in entities:
                    entities.append(entity)
            if rand > threshold2:
                ind = round(len(entities)/3)

                for i in range(ind, len(entities)):
                    entities[i] = r.get_random_word()
                summary = "no related information"
            random.shuffle(entities)
            query = " ".join(entities)
            print(query)
            text = text_format(text)
            summary = text_format(summary)
            _ = '"{}","{}","{}"\n'.format(text,query,summary)
            fp.write(_)

    except KeyboardInterrupt:
        print("\nStop me!")
        sys.exit(0)

    except Exception as err:
        logging.info(err)


if __name__ == "__main__":

    print(">>>> dataset type:" + dataset_type)
    print(">>>> output_dir: " + output_dir + dataset_type + ".csv")
    print(">>>> number of dataset: " + str(len(dataset[dataset_type])))
    print(">>>> number of thread: " + str(thread_num))

    with open(output_dir + dataset_type + ".csv", "w") as f:
        f.write("text,query,summary\n")

    p = Pool(thread_num)

    with p:
        p.map(run_task, tqdm(range(len(dataset[dataset_type]))))
    print("saved in {}.csv".format(output_dir + dataset_type))

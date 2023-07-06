import wikipedia
from tqdm import tqdm
import os
import sys
import logging
import warnings
from multiprocessing import Pool
from nltk.tokenize import sent_tokenize
import random
from random_word import RandomWords

r = RandomWords()
warnings.filterwarnings("ignore")
to_file_name = "train"

with open("./raw_data/" + to_file_name + "_src", "r") as f:
    train_src = f.readlines()

# if to_file_name != "test":
#     with open("./raw_data/" + to_file_name + "_tgt", "r") as f2:
#         train_tgt = f2.readlines()

logging.basicConfig(filename="./exception.log", encoding="utf-8", level=logging.DEBUG)
logging.getLogger("urllib3").setLevel(logging.ERROR)
thread_num = 5
t = 0.6

# stopwords = ['of', 'a', 'an', 'and', 'to', "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k",
#              "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]


def run_task(ind):
    fp = open("./data/wiki/" + to_file_name + ".csv", "a")

    src = train_src[ind]
    # if to_file_name != "test":
    #     tgt = train_tgt[ind]
    # train_src[ind] = train_src[ind].lower()
    keywords = []
    keywords1 = []
    keywords2 = []
    result = ""
    summary = ""
    text = ""
    # _ = first_keyword.replace(".","").split("_")
    # subjects = [w for w in _ if len(w) > 2]

    # if i + 1 % 20 == 0:
    #     dict = {}

    try:
        # src = "a glastonbury romance | isbn number | 0-7156-3648-0"
        triples = src.split(" <TSP> ")
        first_keyword = triples[0].split("|")[0].strip()
        rand = random.random()
        if rand > t:
            subject = r.get_random_word()
            if 0.6 < rand < 0.8:
                src = "{} | {} | {}".format(subject, r.get_random_word(), r.get_random_word())
            elif 0.8 < rand < 0.9:
                src = "{} | {} | {} <TSP> {} | {} | {}".format(subject, r.get_random_word(), r.get_random_word(), subject,
                                                               r.get_random_word(),
                                                               r.get_random_word())
            else:
                src = "{} | {} | {} <TSP> {} | {} | {}".format(subject, r.get_random_word(), r.get_random_word(), subject,
                                                               r.get_random_word(),
                                                               r.get_random_word(), r.get_random_word(), r.get_random_word(),
                                                               r.get_random_word())
            triples = src.split(" <TSP> ")

        for triple in triples:
            # _triples = triple.split("|")[0].lower().split("_")
            # for index in range(len(_triples)):
            #     word = _triples[index].strip().strip("(").strip(")").replace("-"," ").replace(",","")
            #     if word not in keywords:
            #         keywords.append(word)
            # _triples = triple.split("|")[2].lower().split("_")
            # for index in range(len(_triples)):
            #     word = _triples[index].strip().strip("(").strip(")").replace("-"," ").replace(",","")
            #     if word not in keywords:
            #         keywords.append(word)
            keywords1.append(triple.split("|")[0].strip().lower())
            keywords2.append(triple.split("|")[2].strip().lower())

        # print(keywords)
        # if first_keyword in dict:
        #     text = dict[first_keyword]
        #     fp.writelines(text)
        #     return

        # else:
        # get data from wikipedia
        ny = wikipedia.page(first_keyword)
        content = ny.content.replace("\n", " ")
        content_list = sent_tokenize(content)
        # random.shuffle(content_list)
        length = 0
        for j, sentence in enumerate(content_list):
            if sentence.startswith('=='):
                continue
            if length > 600:
                break

            sentence_piece = sentence.split(" ")
            length += len(sentence_piece)
            sentence = sentence.replace('"', "'").lower()
            text += sentence
            subjects = []
            objects = []
            over_loop = False
            # for sub in subjects:
            #     if sub in sentence:
            #         summary += sentence
            #         over_loop = True
            #         break
            #

            #
            # for keyword in keywords:
            #     if keyword in sentence:
            #         summary += sentence
            #         # over_loop = True
            #         break
            for i in range(len(keywords1)):
                subjects = keywords1[i].replace(",", " ").replace("_", " ").replace(".",
                                                                                    " ").replace(
                    "/", " ").split(" ")
                objects = keywords2[i].replace(",", " ").replace("_", " ").replace(".",
                                                                                   " ").replace("/",
                                                                                                " ").split(
                    " ")
                # print(subjects)
                # print(objects)
                for o in objects:
                    for s in subjects:
                        s = s.strip()
                        o = o.strip()
                        if s in sentence and o in sentence and len(s) > 2 and len(o) > 2:
                            # print(sentence)
                            print(s, o)
                            print()
                            summary += sentence
                            over_loop = True
                            break
                    if over_loop:
                        break
                if over_loop:
                    break
        # real tgt
        # if to_file_name != "test":
        #     # without ","
        #     # summary += tgt.strip().replace(",", ".").lower()
        #     # with ","
        #     summary += tgt.strip().lower()
        # else:
        #     summary = "summary"

        # result = '"{} | content | {}'.format(
        #     first_keyword.replace(",", "").lower().strip(), text
        # )
        # real inp
        if len(summary) == 0:
            summary = "no related information"
            # print("+1")
        result = '"' + text + '","' + src.lower().replace(",", "").replace("_", " ").replace(".",
                                                                                             "").replace(
            "(", "").replace(
            ")", "").replace('"', "")

        # to .csv format
        result = result.strip() + '","' + summary + '"\n'
        # result = result.strip() + '"\n'
        # cache
        # dict[first_keyword] = result
        # print(result)
        # if len(summary) > 0:
        fp.writelines(result)

    except KeyboardInterrupt:
        print("\nStop me!")
        sys.exit(0)

    except Exception as err:
        logging.info(err)


if __name__ == "__main__":
    output_dir = "data/wiki/"
    os.makedirs(output_dir, exist_ok=True)

    print(">>>> src_dataset:" + f.name)
    # print(">>>> tgt_dataset:" + f2.name if to_file_name != "test" else "")
    print(">>>> output_dir: " + output_dir + to_file_name + ".csv")
    print(">>>> number of dataset: " + str(len(train_src)))
    print(">>>> number of thread: " + str(thread_num))
    f = open("./data/wiki/" + to_file_name + ".csv", "w")
    f.write("text,query,summary\n")
    f.close()
    p = Pool(thread_num)
    with p:
        p.map(run_task, tqdm(range(int(len(train_src)))))
    print("saved in {}.csv".format(output_dir + to_file_name))

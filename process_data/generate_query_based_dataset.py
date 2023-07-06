# import os
# from tqdm import tqdm
# import warnings
# warnings.filterwarnings("ignore")
# 
# OUTPUT_DIR = "./data/query/"
# INPUT_DIR = "./raw_data/"
# 
# _dirs = [("_src","_tgt"),("test_src",""),("validation_src","validation_tgt")]
# 
# for _dir in _dirs:
#     _src, _tgt = _dir
#     _type = _src.split("_")[0] # train, validation, test
#     with open(INPUT_DIR + _src, "r") as f:
#         src_list = f.readlines()
#     if _type != "test":
#         with open(INPUT_DIR + _tgt, "r") as f:
#             tgt_list = f.readlines()
# 
#     print("\n{},{}, length: {}\n".format(_src, _tgt, len(src_list)))
#     with open(OUTPUT_DIR + "{}.csv".format(_type), "w") as f:
#         f.write("text,summary\n")
#     for ind, src in enumerate(tqdm(src_list)):
#         objects = []
# 
#         triples = src.split("<TSP>")
#         for triple in triples:
#             object = triple.split("|")[2]
#             objects.append(object)
#         if _type != "test":
#             tgt = tgt_list[ind]
#         else:
#             tgt = "summary\n"
# 
#         text = " ".join(objects).strip() + " <sep> " + tgt
#         text = text.lower()
#         text = text.replace("_", " ")
#         text = text.replace("\"","\'")
# 
#         with open(OUTPUT_DIR + "{}.csv".format(_type), "a") as f:
#             f.write(text)

import wikipedia
from tqdm import tqdm
import os
import sys
import logging
import warnings
from multiprocessing import Pool

warnings.filterwarnings("ignore")
to_file_name = "validation" # train validation test
output_dir = "./data/query/"
os.makedirs(output_dir, exist_ok=True)

model = 1

with open("./raw_data/" + to_file_name + "_src", "r") as f:
    _src = f.readlines()

with open("./raw_data/" + to_file_name + "_tgt", "r") as f:
    _tgt = f.readlines()


logging.basicConfig(filename="./exception.log", encoding="utf-8",
                    level=logging.DEBUG)
logging.getLogger("urllib3").setLevel(logging.ERROR)
counted_num = 5
thread_num = 5



def run_task(ind):
    fp = open(output_dir + to_file_name + ".csv", "a")
    fp2 = open(output_dir + to_file_name + "_record.txt", "a")
    src = _src[ind]
    tgt = _tgt[ind]
    triples = src.split(" <TSP> ")
    first_keyword = triples[0].split("|")[0].strip()
    text = ""



    try:
        objects = []
        keywords = []
        for triple in triples:
            object = triple.split("|")[2].strip().lower()
            objects.append(object)

        # get data from wikipedia
        ny = wikipedia.summary(first_keyword, auto_suggest=False)
        content = ny.strip().replace("\n", " ").lower()
        content_list = content.split(". ")

        for sentence in content_list[0:counted_num]:
            sentence = (
                    sentence.strip(".").replace('"', "'") + ". "
            )
            text += sentence.lower()
        for obj in objects:
            obj = obj.replace(",", "").replace('"',"").replace("_", " ")
            elements = obj.split(" ")
            for element in elements:
                if element in text:
                    keywords.append(element)


        # 1. combine with the query word
        if len(keywords) > 0:
            result = ' '.join(keywords).strip()
        # else:
        #     result = "no information"
        # 2. combine with the context sentences
        result = result + " <sep> " + text
        # 2.2 formatting
        result = result.replace('\"', "\'")
        result = result.strip()

        summary = model.generate(result)

        final_result = " <tsp> ".join(triples)
        final_result = '"' + final_result.replace('"',"'").replace(".","") + '","' + summary.replace('"',"'").replace(".","")
        final_result = '"' + final_result.lower()

        final_result = tgt + summary
        final_result = final_result.replace('"',"'")
        final_result = '","' + final_result.lower() + '"'

        print(final_result)

        # result = '"' + result
        #
        # # 3. combine with summary
        # summary = "summary"
        # result = result.strip() + '","' + summary.strip() + '"\n'
        # 3.2 formatting
        # result = result.lower()

        # fp.writelines(result)
        # resource = "<tsp>".join(triples).strip() + " <sep> " + tgt
        # fp2.writelines(resource)
    except KeyboardInterrupt:
        print("\nStop me!")
        sys.exit(0)

    except Exception as err:
        logging.info(err)


if __name__ == "__main__":


    print(">>>> src_dataset:" + f.name)
    # print(">>>> tgt_dataset:" + f2.name if to_file_name != "test" else "")
    print(">>>> output_dir: " + output_dir + to_file_name + ".csv")
    print(">>>> number of dataset: " + str(len(_src)))
    print(
        ">>>> number of sentence counted in the wikipedia: " + str(counted_num))
    print(">>>> number of thread: " + str(thread_num))

    with open(output_dir + to_file_name + ".csv", "w") as f:
        f.write("text,summary\n")

    p = Pool(thread_num)
    with p:
        p.map(run_task, tqdm(range(len(_src))))
    print("saved in {}.csv".format(output_dir + to_file_name))





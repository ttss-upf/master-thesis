import pickle
import json
# data_dir = "./squad/train-v1.1.json"
# output_dir = "./data/question/train_大文件.csv"
data_dir = "./squad/dev-v1.1.json"
output_dir = "data/question/validation.csv"
with open(data_dir, "r") as f:
    data = json.load(f)
fp = open(output_dir, "w")
print(len(data["data"]))
output = []
for piece in data["data"]:
    for content in piece["paragraphs"]:
        context = content["context"]
        for qas in content["qas"]:
            res = {}
            res["context"] = context.replace("\"","\'")
            res["question"] = qas["question"].replace("\"","\'")
            res["answers"] = qas["answers"][0]["text"].replace("\"","\'")
            text = "\"" + res["context"] + "<tsp>" + res["answers"] + "\"" + "," + "\"" + res["question"] + "\"" + "\n"
            fp.writelines(text)
            output.append(res)



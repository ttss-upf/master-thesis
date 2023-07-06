epoch_num = input("pls input the epoch demanding [3/10/test]: ")
if epoch_num != "test":
    epoch_num += "epoch"
with open("result-" + epoch_num + "/result.txt", "r") as fp:
    result_str = fp.read()

import re

decoded_preds_ind = re.search(r"decoded_preds \[.*\]", result_str)
decoded_label_ind = re.search(r"decoded_labels \[.*\]", result_str)

decoded_label = (
    result_str[decoded_label_ind.start(): decoded_label_ind.end()]
    .strip("decoded_labels \[")
    .strip("\]")
    .replace("\', \'","\'<sep>\'")
    .replace("\', \"","\'<sep>\'")
    .replace("\", \"","\'<sep>\'")
    .replace("\", \'","\'<sep>\'")
    .split("<sep>")
)
decoded_pred = (
    result_str[decoded_preds_ind.start(): decoded_preds_ind.end()]
    .strip("decoded_preds \[")
    .strip("\]")
    .replace("', '","'<sep>")
    .replace("\", \"","'<sep>'")
    .replace("\', \"","'<sep>'")
    .replace("\", \'","'<sep>'")
    .split("<sep>")
)

result_str = ""
len_of_label = len(decoded_label)
len_of_pred = len(decoded_pred)

print("decoded_label:", len_of_label)
print("decoded_pred:", len_of_pred)

is_save = input("saved[1]?\n")
if is_save == "1":
    print("saved to .txt files")
    # update .txt file or not
    with open("result-" + epoch_num + "/decoded_label.txt", "w") as fp:
        fp.write('\n'.join(decoded_label))

    with open("result-" + epoch_num + "/decoded_preds.txt", "w") as fp:
        fp.write('\n'.join(decoded_pred))

while True:
    ind = int(input("pls input index [int]: "))
    assert ind < len_of_label and ind < len_of_pred
    print("decoded_label[{}]:".format(ind), decoded_label[ind].replace("\\n", ""))
    print("decoded_pred[{}]:".format(ind), decoded_pred[ind].replace("\\n", ""))

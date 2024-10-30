import json

data_root = "../Data/Potholes"
with open(f"{data_root}/splits.json") as fp:
    split_json = json.load(fp)
split_ids_train = [int(xmlf.split("-")[-1].split(".")[0]) for xmlf in split_json["train"]]
split_ids_test = [int(xmlf.split("-")[-1].split(".")[0]) for xmlf in split_json["test"]]
print("Train ids: ", split_ids_train)
print("Test ids: ", split_ids_test)

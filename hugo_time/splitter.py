import json
from sklearn.model_selection import train_test_split
import os
import shutil

data_root = "/zhome/88/7/117159/Courses/IDLCV_OD/Data/Potholes"
with open(f"{data_root}/splits.json") as fp:
    split_json = json.load(fp)
split_ids_train = [int(xmlf.split("-")[-1].split(".")[0]) for xmlf in split_json["train"]]
split_ids_test = [int(xmlf.split("-")[-1].split(".")[0]) for xmlf in split_json["test"]]

test_id, val_id  = train_test_split(split_ids_test, train_size=0.5, random_state=int(round(310701/281001 * 100496 +231002, 0))) #Random state Clara BD/Floroin BD * T BD + Hugo BD


image_paths = "/zhome/88/7/117159/Courses/IDLCV_OD/Data/Potholes/annotated-images"

train_folder = '/zhome/88/7/117159/Courses/IDLCV_OD/Data/Potholes/train'
test_folder = '/zhome/88/7/117159/Courses/IDLCV_OD/Data/Potholes/test'
validation_folder = '/zhome/88/7/117159/Courses/IDLCV_OD/Data/Potholes/validation'

for id in split_ids_train:
    image_name  = 'img-'+str(id)+'.jpg'
    label_name  = 'img-'+str(id)+'.xml'
    image_path = os.path.join(image_paths,image_name)
    label_path = os.path.join(image_paths,label_name)

    shutil.copyfile(image_path, os.path.join(train_folder,image_name))
    shutil.copyfile(label_path, os.path.join(train_folder,label_name))

for id in test_id:
    image_name  = 'img-'+str(id)+'.jpg'
    label_name  = 'img-'+str(id)+'.xml'
    image_path = os.path.join(image_paths,image_name)
    label_path = os.path.join(image_paths,label_name)

    shutil.copyfile(image_path, os.path.join(test_folder,image_name))
    shutil.copyfile(label_path, os.path.join(test_folder,label_name))
    
for id in val_id:
    image_name  = 'img-'+str(id)+'.jpg'
    label_name  = 'img-'+str(id)+'.xml'
    image_path = os.path.join(image_paths,image_name)
    label_path = os.path.join(image_paths,label_name)

    shutil.copyfile(image_path, os.path.join(validation_folder,image_name))
    shutil.copyfile(label_path, os.path.join(validation_folder,label_name))
    

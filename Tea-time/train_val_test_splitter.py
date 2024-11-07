import os
from sklearn.model_selection import train_test_split
import shutil

data_root = "/zhome/88/7/117159/Courses/IDLCV_OD/Data/Potholes/annotated-images"
data_files = [f"{data_root}/{f}" for f in os.listdir(data_root) if ".jpg" in f]
data_files.sort()

# Get list of XML files
xml_files = [f"{data_root}/{f}" for f in os.listdir(data_root) if f.endswith('.xml')]
xml_files.sort()


X_train, X_test_tmp, y_train, y_test_tmp = train_test_split(data_files, xml_files, train_size=0.7, random_state=int(round(310701/281001 * 100496 +231002, 0))) #Random state Clara BD/Floroin BD * T BD + Hugo BD


X_val, X_test, y_val, y_test = train_test_split(X_test_tmp, y_test_tmp, train_size=0.5, random_state=int(round(310701/281001 * 100496 +231002, 0))) #Random state Clara BD/Floroin BD * T BD + Hugo BD


train_folder = '/zhome/88/7/117159/Courses/IDLCV_OD/Data/Potholes/train'
validation_folder = '/zhome/88/7/117159/Courses/IDLCV_OD/Data/Potholes/validation'
test_folder = '/zhome/88/7/117159/Courses/IDLCV_OD/Data/Potholes/test'

for train_image, train_label in zip(X_train,y_train):
    bn_image = os.path.basename(train_image)
    bn_label = os.path.basename(train_label)
    shutil.copyfile(train_image, os.path.join(train_folder,bn_image))
    shutil.copyfile(train_image, os.path.join(train_folder,bn_label))
    
for val_image, val_label in zip(X_val,y_val):
    bn_image = os.path.basename(val_image)
    bn_label = os.path.basename(val_label)
    shutil.copyfile(val_image, os.path.join(validation_folder,bn_image))
    shutil.copyfile(val_image, os.path.join(validation_folder,bn_label))
    
for test_image, test_label in zip(X_test,y_test):
    bn_image = os.path.basename(test_image)
    bn_label = os.path.basename(test_label)
    shutil.copyfile(test_image, os.path.join(test_folder,bn_image))
    shutil.copyfile(test_image, os.path.join(test_folder,bn_label))
    


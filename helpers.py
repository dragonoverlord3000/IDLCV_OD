import xml.etree.ElementTree as ET
import json
import cv2 as cv
import matplotlib.pyplot as plt

data_root = "../Data/Potholes"
def get_split_ids(train=True):
    with open(f"{data_root}/splits.json") as fp:
        split_json = json.load(fp)
    
    if train: 
        return [int(xmlf.split("-")[-1].split(".")[0]) for xmlf in split_json["train"]]
    return[int(xmlf.split("-")[-1].split(".")[0]) for xmlf in split_json["test"]]

# Function to parse XML and extract bounding box information
def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    filename = root.find('filename').text
    boxes = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        boxes.append((name, xmin, ymin, xmax, ymax))

    return filename, boxes


def box_plotter(image, boxes, save_path='./figures/000_box_plotter.jpg'):
    for b in boxes:
        x, y, w, h = b
        cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 1, cv.LINE_AA)

    plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))  # Convert to RGB for matplotlib
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
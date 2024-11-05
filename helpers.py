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


def _EdgeBox(model_path="./hugo_time/model.ymz.gz", image):
    model = model_path
    im = image # cv.imread("../Data/Potholes/annotated-images/img-322.jpg")

    edge_detection = cv.ximgproc.createStructuredEdgeDetection(model)
    rgb_im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
    edges = edge_detection.detectEdges(np.float32(rgb_im) / 255.0)

    orimap = edge_detection.computeOrientation(edges)
    edges = edge_detection.edgesNms(edges, orimap)

    edge_boxes = cv.ximgproc.createEdgeBoxes()
    edge_boxes.setMaxBoxes(30)
    boxes, probs = edge_boxes.getBoundingBoxes(edges, orimap)

    return boxes


def box_plotter(image, boxes, save_path='./figures/000_box_plotter.jpg'):
    for b in boxes:
        x, y, w, h = b
        cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 1, cv.LINE_AA)

    plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))  # Convert to RGB for matplotlib
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
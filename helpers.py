import xml.etree.ElementTree as ET
import json
import cv2 as cv
import matplotlib.pyplot as plt
from typing import Tuple, List
import numpy as np

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


def _EdgeBox(image, num_boxes=30, model_path="/zhome/51/7/168082/Desktop/s214659/02516_IDLCV/IDLCV_OD/hugo_time/model.yml.gz"):
    model = model_path
    im = image # cv.imread("../Data/Potholes/annotated-images/img-322.jpg")

    edge_detection = cv.ximgproc.createStructuredEdgeDetection(model)
    rgb_im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
    edges = edge_detection.detectEdges(np.float32(rgb_im) / 255.0)

    orimap = edge_detection.computeOrientation(edges)
    edges = edge_detection.edgesNms(edges, orimap)

    edge_boxes = cv.ximgproc.createEdgeBoxes()
    edge_boxes.setMaxBoxes(num_boxes)
    boxes, probs = edge_boxes.getBoundingBoxes(edges, orimap)

    #returns x, y, w, h
    return boxes


def box_plotter(image, boxes, save_path='./figures/000_box_plotter.jpg'):
    for b in boxes:
        x, y, w, h = b
        cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 1, cv.LINE_AA)

    plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))  # Convert to RGB for matplotlib
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)


def intersection(x1,y1,w1,h1,x2,y2,w2,h2):
    top = min(y1+h1, y2+h2)
    bot = max(y1, y2)
    left = max(x1, x2)
    right = min(x1+w1, x2+w2)
    if top < bot or right < left: return 0
    return (top - bot + 1) * (right - left + 1)

def IoU(GT, proposal):
    """
    Parameters:
        GT (List[Tuple[int]]): list of bounding boxes for the ground truth
        proposal (Tuple[int]): the proposal bounding box
    """
    px,py,pw,ph = proposal
    max_iou = 0
    for x,y,w,h in GT:
        inter = intersection(px,py,pw,ph,x,y,w,h)
        max_iou = max(max_iou, inter/(pw*ph + w*h - inter))

    return max_iou

def get_proposals(image, GT, num_pos_proposals, num_neg_proposals, k1, k2, generator):
    """
    Parameters:
        image (np.array): the image we're generating proposals on
        GT (List[Tuple[int]]): list of bounding boxes for the ground truth
        num_pos_proposals, num_neg_proposals (int): number of proposals we want
        k1 (float): if max_i IoU(A, GT_i) < k1, then A is background proposal
        k2 (float): if max_i IoU(A, GT_i) > k2, then A is positive proposal
        generator (Function): EdgeBox method or Selective Search method

    See slide (83) in lecture 9.
    """

    pos_proposals = []
    neg_proposals = []
    while len(pos_proposals) < num_pos_proposals or len(neg_proposals) < num_neg_proposals:
        # Generate proposals
        proposals = generator(image)
        for proposal in proposals:
            iou = IoU(GT, proposal)
            if iou >= k2 and len(pos_proposals) < num_pos_proposals: 
                pos_proposals.append(proposal)
            elif iou < k1 and len(neg_proposals) < num_neg_proposals: 
                neg_proposals.append(proposal)

    return pos_proposals, neg_proposals

import cv2
import selectivesearch
import matplotlib.pyplot as plt
import selectivesearch
from selectivesearch import selective_search
import sys
import os

# Load the image

def _SelectiveSearch(image,size_threshold = 100, _scale=500, _sigma=0.8, _min_size=10):

    # Get selective search object proposals

    ## 
    # SIGMA IS THE GAUSSIAN NOISE, 0.8 IS STANDARD
    # SCALE is the sets a scale of observation. Higher number increases the preference of larger boxes (500)
    # Min size, If the rectangle size is reached on min_size, the calculation is stopped. (10)

    img_lbl, regions = selectivesearch.selective_search(image, scale=_scale, sigma=_sigma, min_size=_min_size)

    # Draw the proposals on the image
    output_image = image.copy()

    # Initialize an empty set to store selected region proposals
    candidates = set()
    
    # Iterate over all the regions detected by Selective Search
    for r in regions:
        # Check if the current region's rectangle is already in candidates
        if r['rect'] in candidates:
            continue  # Skip this region if it's a duplicate
    
        # Check if the size of the region is less than 100 pixels
        if r['size'] < size_threshold:
            continue  # Skip this region if it's too small
    
        # Extract the coordinates and dimensions of the region's rectangle
        x, y, w, h = r['rect']
    
        # Avoid division by zero by checking if height or width is zero
        if h == 0 or w == 0:
            continue  # Skip this region if it has zero height or width
    
        # Check the aspect ratio of the region (width / height and height / width)
        # if w / h > 1.2 or h / w > 1.2:
        #     continue  # Skip this region if its aspect ratio is not within a range
    
        # If all conditions are met, add the region's rectangle to candidates
        candidates.add(r['rect'])
    
    #returns list of (x, y, w, h)
    return list(candidates)

testing_selective_search = False
if testing_selective_search:
    image_path = '/zhome/88/7/117159/Courses/IDLCV_OD/Data/Potholes/annotated-images/img-1.jpg' # also define this
    save_path = '/zhome/88/7/117159/Courses/IDLCV_OD/Tea-time/figures/selective_search' #  define this
    image = cv2.imread(image_path)

    boxes = _SelectiveSearch(image, size_threshold = 100, _scale=500, _sigma=0.8, _min_size=10)

    box_plotter(image,boxes, save_path+'/3.png')


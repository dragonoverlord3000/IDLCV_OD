import cv2 as cv
import numpy as np
from proposal_time import intersection, IoU, get_proposals
import xml.etree.ElementTree as ET
import sys
import os



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

if __name__ == '__main__':
    # Add the parent directory to sys.path
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    sys.path.insert(0, parent_dir)

    # Now, import from helper
    from helpers import _EdgeBox, _SelectiveSearch  # Replace my_function with the function you need

    model = "model.yml.gz"
    image = cv.imread("../Data/Potholes/annotated-images/img-322.jpg")
    _, boxes = parse_xml("../Data/Potholes/annotated-images/img-322.xml")
    print(os.getcwd()+ '/' + model)
    print(cv.ximgproc.createStructuredEdgeDetection(os.getcwd() + '/' + model))
    #edgebox = _EdgeBox(image, 30, model_path="model.yml.gz")
    pos_proposal, neg_proposal = get_proposals(image, 
                                               boxes, 
                                               num_pos_proposals=1, 
                                               num_neg_proposals=1, k1=0.2, k2=0.5, generator=_EdgeBox)
    
    pos_proposal, neg_proposal = get_proposals(image, 
                                               boxes, 
                                               num_pos_proposals=1, 
                                               num_neg_proposals=1, k1=0.2, k2=0.5, generator=_SelectiveSearch)
    print(pos_proposal)

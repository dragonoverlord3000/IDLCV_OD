import cv2 as cv
import numpy as np
from helpers import IoU, parse_xml, get_proposals, _EdgeBox, _SelectiveSearch, cut_patches
import xml.etree.ElementTree as ET
import sys
import os
import matplotlib.pyplot as plt

if __name__ == '__main__':

    model = "model.yml.gz"
    image = cv.imread("./Data/Potholes/annotated-images/img-322.jpg")
    _, boxes = parse_xml("./Data/Potholes/annotated-images/img-322.xml")
    #edgebox = _EdgeBox(image, 30, model_path="model.yml.gz")
    pos_proposal, neg_proposal = get_proposals(image, 
                                               boxes, 
                                               num_pos_proposals=1, 
                                               num_neg_proposals=1, k1=0.2, k2=0.5, generator=_EdgeBox)

    pos_proposal, neg_proposal = get_proposals(image, 
                                                 boxes, 
                                                 num_pos_proposals=1, 
                                                 num_neg_proposals=1, k1=0.3, k2=0.5, generator=_SelectiveSearch)
    cut_patches(image, "test", pos_proposal, neg_proposal, path="./Data/image_patches/")
    #x, y, w, h = pos_proposal[0]
    #print(image.shape)
    #image_patch = image[x:x+w, y:y+h, :]
    #plt.axis('off')
    #plt.imshow(cv.cvtColor(image_patch, cv.COLOR_BGR2RGB))

    # Save the image without axes and padding
    #plt.savefig("./image_patches/test.png", bbox_inches='tight', pad_inches=0)
    #plt.close() 

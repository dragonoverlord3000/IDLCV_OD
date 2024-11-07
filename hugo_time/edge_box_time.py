import cv2 as cv
import numpy as np
import sys
import os

# Inspired by: https://stackoverflow.com/questions/54843550/edge-box-detection-using-opencv-python

if __name__ == '__main__':
    model = "model.yml.gz"
    im = cv.imread("../Data/Potholes/annotated-images/img-299.jpg")
    cv.imwrite("./EB/image_before.jpg", im) 

    edge_detection = cv.ximgproc.createStructuredEdgeDetection(model)
    rgb_im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
    edges = edge_detection.detectEdges(np.float32(rgb_im) / 255.0)

    orimap = edge_detection.computeOrientation(edges)
    edges = edge_detection.edgesNms(edges, orimap)

    edge_boxes = cv.ximgproc.createEdgeBoxes()
    num_boxes = 256
    edge_boxes.setMaxBoxes(num_boxes)
    boxes, probs = edge_boxes.getBoundingBoxes(edges, orimap)
    print(boxes)
    for b in boxes:
        x, y, w, h = b
        cv.rectangle(im, (x, y), (x+w, y+h), (0, 255, 0), 1, cv.LINE_AA)
    print(os.getcwd())
    cv.imwrite("./EB/edges_output.jpg", edges * 255) 
    cv.imwrite(f"./EB/edgeboxes_output_{num_boxes}.jpg", im) 

    cv.waitKey(0)
    cv.destroyAllWindows()
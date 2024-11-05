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


def _EdgeBox(image, num_boxes=30, model_path="./hugo_time/model.ymz.gz"):
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
        GT (List[tuple[int]]): list of bounding boxes for the ground truth
        proposal (tuple[int]): the proposal bounding box
    """
    px,py,pw,ph = proposal
    max_iou = 0
    for x,y,w,h in GT:
        inter = intersection(px,py,w1,h1,x,y,w,h)
        max_iou = max(max_iou, inter/(pw*ph + w*h - inter))

    return max_iou

def get_proposals(image, GT, num_pos_proposals, num_neg_proposals, k1, k2, generator):
    """
    Parameters:
        image (np.array): the image we're generating proposals on
        GT (List[tuple[int]]): list of bounding boxes for the ground truth
        num_pos_proposals, num_neg_proposals (int): number of proposals we want
        k1 (float): if max_i IoU(A, GT_i) < k1, then A is background proposal
        k2 (float): if max_i IoU(A, GT_i) > k2, then A is positive proposal
        generator (function): EdgeBox method or Selective Search method

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
            else if iou < k1 and len(neg_proposals) < num_neg_proposals: 
                neg_proposals.append(proposal)

    return pos_proposals, neg_proposals



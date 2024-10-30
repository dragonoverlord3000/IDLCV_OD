import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import numpy as np
import os

###########################
# Show dataset statistics #
###########################

data_root = "../Data/Potholes/annotated-images"
data_files = [f"{data_root}/{f}" for f in os.listdir(data_root) if ".jpg" in f]
im_shapes = [Image.open(image_file).size for image_file in data_files]

print(
    "", "#"*68 + "\n",
    f"# Statistics of the '{data_root}' dataset #\n",
    "#"*68 + "\n\n",
    f"Number of images: {len(data_files)}, \n",
    f"Number of shapes: {len(np.unique(im_shapes))}, \n",
    f"Largest image: {max(*im_shapes, key=lambda sz: sz[0]*sz[1])}, \n",
    f"Smallest image: {min(*im_shapes, key=lambda sz: sz[0]*sz[1])}, \n",
    f"All square: {all(sz[0] == sz[1] for sz in im_shapes)}, \n",
    f"Example image file path: {data_files[0]} \n"
)

############################
#   Show object detection  #
############################

# Get list of XML files
xml_files = [f"{data_root}/{f}" for f in os.listdir(data_root) if f.endswith('.xml')]

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

# Plot each image with its bounding boxes
for xml_file in tqdm(xml_files, desc="Plotting images with bounding boxes"):
    filename, boxes = parse_xml(xml_file)
    image_path = os.path.join(data_root, filename)
    
    # Load the image
    with Image.open(image_path) as img:
        plt.imshow(img)
        ax = plt.gca()
        
        # Plot bounding boxes
        for name, xmin, ymin, xmax, ymax in boxes:
            width = xmax - xmin
            height = ymax - ymin
            rect = plt.Rectangle((xmin, ymin), width, height, edgecolor='red', facecolor='none', linewidth=2)
            ax.add_patch(rect)
            ax.text(xmin, ymin, name, color='white', verticalalignment='top', 
                    bbox={'color': 'red', 'pad': 0})
        
        title = xml_file.split("-")[-1].split(".")[0]
        plt.axis('off')
        plt.savefig(f"./images_with_bboxes/{title}.jpg")
        plt.close()






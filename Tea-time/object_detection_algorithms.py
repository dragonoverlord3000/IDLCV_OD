import cv2
import selectivesearch
import matplotlib.pyplot as plt
import selectivesearch

from selectivesearch import selective_search
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


    # Convert the selected bounding boxes to the original image size
    candidates_scaled = [(int(x * (image.shape[1] / new_width)),
                        int(y * (image.shape[0] / new_height)),
                        int(w * (image.shape[1] / new_width)),
                        int(h * (image.shape[0] / new_height)))
                        for x, y, w, h in candidates]


    for idx, (proposal) in enumerate(candidates_scaled):
        (x, y, w, h) = proposal
        cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 20)


    return candidates_scaled


if __name__=="__main__":
    image_path = '/zhome/88/7/117159/Courses/IDLCV_OD/Data/Potholes/annotated-images/img-1.jpg'
    save_path = '/zhome/88/7/117159/Courses/IDLCV_OD/Tea-time/figures/selective_search'
    image = cv2.imread(image_path)

    new_height = int(image.shape[1] / 4)
    # Calculate a new width
    new_width = int(image.shape[0] / 4)
    # Resize the image to the new dimensions
    resized_image = cv2.resize(image, (new_width, new_height))

    _SelectiveSearch(resized_image, size_threshold = 100, _scale=500, _sigma=0.8, _min_size=10)


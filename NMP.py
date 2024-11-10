import torch

def non_max_suppression(boxes, scores, iou_threshold=0.5):
    """
    Perform Non-Maximum Suppression (NMS) to remove redundant overlapping bounding boxes.

    Parameters:
    - boxes (Tensor): Tensor of shape (N, 4) where N is the number of boxes and each box is represented by (x1, y1, x2, y2).
    - scores (Tensor): Tensor of shape (N,) representing the confidence scores for each box.
    - iou_threshold (float): Intersection over Union (IoU) threshold for suppression.

    Returns:
    - keep (List[int]): Indices of the boxes that are kept after suppression.
    """
    if len(boxes) == 0:
        return []

    # Get coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # Compute the area of each bounding box
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    # Sort the boxes by scores in descending order
    _, order = scores.sort(descending=True)

    keep = []

    while order.numel() > 0:
        i = order[0].item()  # Get the index of the current highest score box
        keep.append(i)

        if order.numel() == 1:
            break

        # Compute the intersection of the highest score box with the rest
        xx1 = torch.max(x1[i], x1[order[1:]])
        yy1 = torch.max(y1[i], y1[order[1:]])
        xx2 = torch.min(x2[i], x2[order[1:]])
        yy2 = torch.min(y2[i], y2[order[1:]])

        # Compute the width and height of the overlap
        inter_w = torch.clamp(xx2 - xx1 + 1, min=0)
        inter_h = torch.clamp(yy2 - yy1 + 1, min=0)

        # Compute the intersection area
        inter_area = inter_w * inter_h

        # Compute the IoU (Intersection over Union)
        iou = inter_area / (areas[i] + areas[order[1:]] - inter_area)

        # Keep only boxes with IoU less than the threshold
        remaining_indices = (iou <= iou_threshold).nonzero(as_tuple=False).squeeze()

        if remaining_indices.numel() == 0:
            break

        # Update order to remove suppressed indices
        order = order[remaining_indices + 1]

    return keep


# Example usage
boxes = torch.tensor([[100, 100, 210, 210], [105, 105, 215, 215], [150, 150, 250, 250]])
scores = torch.tensor([0.9, 0.75, 0.8])

keep_indices = non_max_suppression(boxes, scores, iou_threshold=0.5)
print("Indices of kept boxes:", keep_indices)

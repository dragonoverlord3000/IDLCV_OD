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



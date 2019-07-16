"""
Implemetation of NMS(Non-Maximum Supression)
"""

def iou(bbox1, bbox2):
    """
    @param bbox1: [x1, x2, y1, y2, score]
    @param bbox2: [x1, x2, y1, y2, score]
    @return : intersection area / union area
    """
    # compute intersetion area
    inter_x1 = max(bbox1[0], bbox2[0])
    inter_x2 = min(bbox1[1], bbox2[1])
    inter_y1 = max(bbox1[2], bbox2[2])
    inter_y2 = min(bbox1[3], bbox2[3])
    inter_area = (inter_x2 - inter_x1 + 1) * (inter_y2 - inter_y1 + 1)
    # compute union area
    area1 = (bbox1[1] - bbox1[0] + 1) * (bbox1[3] - bbox1[2] + 1)
    area2 = (bbox2[1] - bbox2[0] + 1) * (bbox2[3] - bbox2[2] + 1)
    union_area = area1 + area2 - inter_area
    return inter_area / union_area

def nms(lists, threshold):
    """
    @param lists: list of bboxes
    @param thresh: threshold for iou
    @return : the rest bboxes after NMS operation
    """
    rest = []
    lists.sort(key=lambda x: x[4], reverse=True)
    while lists:
        max_score_bbox = lists.pop(0)
        rest.append(max_score_bbox)
        # compute iou with max_score_bbox
        for bbox in lists[:]:
            if iou(max_score_bbox, bbox) > threshold:
                lists.remove(bbox)
    return rest


if __name__ == '__main__':
    lists = [[690, 720, 800, 820, 0.5],
             [102, 204, 250, 358, 0.5],
             [118, 257, 250, 380, 0.8], 
             [135, 280, 250, 400, 0.7],
             [118, 255, 235, 360, 0.7]]
    threshold = 0.3
    result = nms(lists, threshold)
    print(result)


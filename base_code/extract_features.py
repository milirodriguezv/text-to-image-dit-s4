def load_bbox(bboxes: list):
    """
    Computes the union of multiple bounding boxes.

    Given a list of bounding boxes in COCO format, calculates the smallest
    bounding box that contains all of them.

    Args:
        bboxes (list): A list of bounding boxes, each represented as
            [x_min, y_min, width, height], where (x_min, y_min) denotes
            the top-left corner and width and height define the box dimensions.

    Returns:
        list: The union bounding box in COCO format
            [x_min, y_min, width, height], enclosing all input bounding boxes.
    """

    x_min = float('inf')
    y_min = float('inf')
    x_max = float('-inf')
    y_max = float('-inf')

    for x, y, width, height in bboxes:
        x_w = x + width
        y_h = y + height

        x_min = min(x_min, x)
        y_min = min(y_min, y)

        x_max = max(x_max, x_w)
        y_max = max(y_max, y_h)


    union_width = x_max - x_min
    union_height = y_max - y_min

    bbox_final = [x_min, y_min, union_width, union_height]
            
    return bbox_final

def expand_bbox(bbox_inicial: list, margin_ratio: float = 0.1):
    new_dimentions = []
    

    return new_dimentions



def preprocess_image(image, bbox):
    pass
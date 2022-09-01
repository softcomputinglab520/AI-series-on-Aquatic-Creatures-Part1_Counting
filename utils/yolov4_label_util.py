
def load_names(PRED_NAMES):
    """
    Loading label names of yolo type
    :param PRED_NAMES: the label file of yolo
    :return: label name, label index
    """
    names = {}
    with open(PRED_NAMES) as f:
        for id_, name in enumerate(f):
            names[id_] = name.split('\n')[0]
    return names, (id_ + 1)

# PRED_NAMES = './data/coco.names'
# classname, num = load_names(PRED_NAMES)

class Node:
    def __init__(self, category):
        self.attribute = None
        self.category = category
        self.children = {}
        self.parent_attribute = None
        self.parent_attribute_value = None
        self.instances_labeled = []
        
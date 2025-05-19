'''Module for periodic cluster labeling'''
from scipy.ndimage import label
import numpy as np

class UnionFind:
    def __init__(self, size):
        self.parent = np.arange(size)

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        self.parent[self.find(x)] = self.find(y)

    def relabel(self):
        # Compress and reindex
        label_map = {}
        new_labels = np.zeros_like(self.parent)
        next_label = 1
        for i in range(len(self.parent)):
            root = self.find(i)
            if root not in label_map:
                label_map[root] = next_label
                next_label += 1
            new_labels[i] = label_map[root]
        return new_labels

def label_fluid_periodic(img):
    """Label fluid regions with periodic boundaries by merging edge labels."""
    structure = np.ones((3, 3), dtype=int)
    # binary = ~img  # False = fluid
    labeled, num_features = label(img*-1 + 1, structure=structure)

    uf = UnionFind(num_features + 1)  # +1 for background

    h, w = img.shape

    # Match top-bottom
    for j in range(w):
        top = labeled[0, j]
        bottom = labeled[-1, j]
        if top != 0 and bottom != 0:
            uf.union(top, bottom)

    # Match left-right
    for i in range(h):
        left = labeled[i, 0]
        right = labeled[i, -1]
        if left != 0 and right != 0:
            uf.union(left, right)

    # Relabel with merged labels
    relabel_map = uf.relabel()
    labeled = relabel_map[labeled]

    return labeled

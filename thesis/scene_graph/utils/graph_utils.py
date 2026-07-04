import networkx as nx
import numpy as np
from typing import Any
from itertools import combinations

def cosine_similarity( descriptor_a, descriptor_b):
    norm_a = np.linalg.norm(descriptor_a)
    norm_b = np.linalg.norm(descriptor_b)

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    # normalizing my norm to remove magnitude
    similarity = np.dot(descriptor_a, descriptor_b) / (norm_a * norm_b)
    return float(np.clip(similarity, -1.0, 1.0))

def semantic_similarity(descriptor_a, descriptor_b):
    cosine = cosine_similarity(descriptor_a, descriptor_b)

    # from [-1, 1] to [0, 1] (but w/issue)
    #affinity = (cosine + 1.0) / 2.0

    #conservative shift
    affinity = max(0, cosine)
    return cosine, affinity

# could change to bbox-aware distance instead of centroid distance
# turning distance into proximity (distance behaves in wrong direction)
def geometric_affinity(center_a, center_b, distance_scale):
    if distance_scale <= 0.0:
        raise ValueError('distance_scale must be > 0  ......')
    distance = float(np.linalg.norm(center_a - center_b))
    proximity = float(np.exp(-distance / distance_scale))
    return distance, proximity

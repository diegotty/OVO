import csv
import json
from itertools import combinations, permutations
import numpy as np
import argparse
import os
# import pandas as pd
import pandas as pd
from shapely.geometry import Polygon
from bbox_utils import *
from tqdm import tqdm
from time import perf_counter
import multiprocessing as mp
from itertools import repeat
from copy import deepcopy

# def relate_in(anchor_idx, objects, in_thres):
#     # find objects inside anchor object
#     anchor_obj = objects[anchor_idx]
#     in_objs = []
#     for i in range(len(objects)):
#         if anchor_idx == i:
#             continue
# 
#         max_z_tgt = max([pt[-1] for pt in objects[i]["bbox"]])
#         min_z_anc = min([pt[-1] for pt in anchor_obj["bbox"]])
#         min_z_tgt = min([pt[-1] for pt in objects[i]["bbox"]])
#         max_z_anc = max([pt[-1] for pt in anchor_obj["bbox"]])
# 
#         if is_inside_bbox(np.array(objects[i]["center"]), np.array(anchor_obj["bbox"])) \
#             and (np.array(anchor_obj["size"]) > np.array(objects[i]["size"])).all() \
#             and max_z_tgt < max_z_anc and min_z_tgt > min_z_anc:
#                 in_objs.append(i)
#             # print(f"{objects[i]['raw_label']} in {anchor_obj['raw_label']}")
#     
#     in_obj_ids = [objects[ind]['object_id'] for ind in in_objs]
#     return in_obj_ids
# 
# def relate_on(vertical_iom, on_thres, under_thres, in_thres, anchor_idx, objects):
#     # find objects on another object
#     anchor_obj = objects[anchor_idx]
# 
#     on_objs = []
#     for i in range(len(objects)):
#         if anchor_idx == i:
#             continue
# 
#         max_z_tgt = max([pt[-1] for pt in objects[i]["bbox"]])
#         min_z_anc = min([pt[-1] for pt in anchor_obj["bbox"]])
#         min_z_tgt = min([pt[-1] for pt in objects[i]["bbox"]])
#         max_z_anc = max([pt[-1] for pt in anchor_obj["bbox"]])
# 
#         if min_z_tgt <= (max_z_anc + on_thres) \
#             and min_z_tgt >= (min_z_anc + under_thres) \
#             and calculate_iom_poly(objects[i], anchor_obj) > vertical_iom \
#             or (np.array(anchor_obj["size"]) <= np.array(objects[i]["size"])).any()) \
#             and anchor_obj["size"][0] * anchor_obj["size"][1] > objects[i]["size"][0] * objects[i]["size"][1]:
#             # or int(anchor_obj['nyu_id']) in ON_RELATION \
#             # and is_inside_bbox(np.array(objects[i]["center"]), np.array(anchor_obj["bbox"])) \
#             # and anchor_obj["size"] > objects[i]["size"] \
#             # and min_z_tgt > min_z_anc + under_thres:
# 
#             # print(f"{objects[i]['nyu_label']} on {anchor_obj['nyu_label']}")
#                 on_objs.append(i)
# 
#     on_obj_ids = [objects[ind]['object_id'] for ind in on_objs]
#     return on_obj_ids



def relate_above(vertical_iom, on_thres, anchor_idx, objects):
    # finds objects above anchor object
    above_objs = []
    anchor_obj = objects[anchor_idx]
    for i in range(len(objects)):
        # skip same object
        if anchor_idx == i:
            continue

        # get lowest/highest points on objects
        max_z2 = max([pt[-1] for pt in objects[i]["bbox"]])
        min_z1 = min([pt[-1] for pt in anchor_obj["bbox"]])
        min_z2 = min([pt[-1] for pt in objects[i]["bbox"]])
        max_z1 = max([pt[-1] for pt in anchor_obj["bbox"]])

        iom = calculate_iom_poly(anchor_obj, objects[i])

        # get 2D bboxes (faces) in x-y plane to ensure some level of overlap in these axes
        # bbox1, bbox2 = get_2D_bboxes([0, 1], objects[anchor_idx], objects[j])
        # compute IOU of x-y faces
        # iou = calculate_iou(bbox1, bbox2)

        # three criteria: center z-coordinate, min/max of object in z-axis, IOU of faces in x-y plane
        if max_z1 + on_thres <= min_z2 and iom > vertical_iom:
            # print(f"{objects[i]['raw_label']} above {anchor_obj['raw_label']}")
            above_objs.append(i)

    above_obj_ids = [objects[ind]["object_id"] for ind in above_objs]
    return above_obj_ids


def relate_below(vertical_iom,under_thres, anchor_idx, objects):
    # finds objects below anchor object
    below_objs = []
    anchor_obj = objects[anchor_idx]

    for i in range(len(objects)):
        # skip same object
        if anchor_idx == i:
            continue


        # get lowest/highest points on objects
        max_z_tgt = max([pt[-1] for pt in objects[i]["bbox"]])
        min_z_anc = min([pt[-1] for pt in anchor_obj["bbox"]])
        min_z_tgt = min([pt[-1] for pt in objects[i]["bbox"]])
        max_z_anc = max([pt[-1] for pt in anchor_obj["bbox"]])

        iom = calculate_iom_poly(anchor_obj, objects[i])

        # get 2D bboxes (faces) in x-y plane to ensure some level of overlap in these axes
        # bbox1, bbox2 = get_2D_bboxes([0, 1], objects[anchor_idx], objects[j])
        # compute IOU of x-y faces
        # iou = calculate_iou(bbox1, bbox2)

        # three criteria: center z-coordinate, min/max of object in z-axis, IOU of faces in x-y plane
        # if anchor_obj['nyu_label'] == 'ceiling':
            # print(f'IoM between ceiling and {objects[i]["nyu_label"]}: {iom}')

        if max_z_tgt <= min_z_anc and iom > vertical_iom or min_z_tgt <= min_z_anc + under_thres and max_z_tgt <= max_z_anc and iom > vertical_iom:
            # print(f"{objects[i]['raw_label']} below {anchor_obj['raw_label']}")
            below_objs.append(i)

    below_obj_ids = [objects[ind]["object_id"] for ind in below_objs]
    return below_obj_ids


# def relate_between(between_iom, anchor_idx, objects, overlap_thres, symmetry_thres, distance_thres, anchor_size_thres):
#     between_objs = []
#     obj_inds = list(range(len(objects)))
#     obj_inds.remove(anchor_idx)
#     obj_pairs = list(permutations(obj_inds, 2))
#     obj_pairs = np.array(obj_pairs)
#     if not len(obj_pairs):
#         return []
#     if len(obj_pairs.shape) < 2:
#         obj_pairs = obj_pairs[None, :]
# 
#     centers = np.array([o["center"] for o in objects])
#     bboxes = np.array([o["bbox"] for o in objects])
# 
#     center1 = np.array(objects[anchor_idx]["center"])
# 
#     center2 = centers[obj_pairs[:, 0]]
#     center3 = centers[obj_pairs[:, 1]]
# 
#     bbox1 = np.array(objects[anchor_idx]["bbox"])
# 
#     bbox2 = bboxes[obj_pairs[:, 0]]
#     bbox3 = bboxes[obj_pairs[:, 1]]
# 
#     r = (center3 - center2)[:, :2]
#     r /= np.linalg.norm(r, axis=-1, keepdims=True)
#     R = np.zeros((center2.shape[0], 3, 3), dtype=np.float64)
#     R[:, 0, 0] = r[:, 0]
#     R[:, 1, 0] = r[:, 1]
#     R[:, 0, 1] = -r[:, 1]
#     R[:, 1, 1] = r[:, 0]
#     R[:, 2, 2] = 1
#     center1_rot = (R.transpose(0, 2, 1) @ center1[None, :, None])[..., 0]
#     center2_rot = (R.transpose(0, 2, 1) @ center2[:, :, None])[..., 0]
#     center3_rot = (R.transpose(0, 2, 1) @ center3[:, :, None])[..., 0]
# 
#     between_xy = (center1_rot[:, 0] > center2_rot[:, 0]) & (center1_rot[:, 0] < center3_rot[:, 0]) 
# 
# 
#     # bboxes: N x 6 x 3 x 1
#     # R: N x 1 x 3 x 3
# 
#     bbox1_rot = (R[:, None, :, :].transpose(0, 1, 3, 2) @ bbox1[None, :, :, None])[..., 0]
#     bbox2_rot = (R[:, None, :, :].transpose(0, 1, 3, 2) @ bbox2[:, :, :, None])[..., 0]
#     bbox3_rot = (R[:, None, :, :].transpose(0, 1, 3, 2) @ bbox3[:, :, :, None])[..., 0]
# 
#     # Check xy intersection
# 
#     pairs_filt_xy = obj_pairs[between_xy]
# 
#     bbox1_rot_between_xy = bbox1_rot[between_xy]
#     bbox2_rot_between_xy = bbox2_rot[between_xy]
#     bbox3_rot_between_xy = bbox3_rot[between_xy]
# 
#     # Make sure bboxes are not overlapping by checking single dimensional iom
# 
#     max_xy1 = bbox1_rot_between_xy[..., 0].max(axis=-1)
#     min_xy1 = bbox1_rot_between_xy[..., 0].min(axis=-1)
#     max_xy2 = bbox2_rot_between_xy[..., 0].max(axis=-1)
#     min_xy2 = bbox2_rot_between_xy[..., 0].min(axis=-1)
#     max_xy3 = bbox3_rot_between_xy[..., 0].max(axis=-1)
#     min_xy3 = bbox3_rot_between_xy[..., 0].min(axis=-1)
# 
#     iom1_1d_xy = (max_xy1 - min_xy3) / np.minimum(max_xy1 - min_xy1, max_xy3 - min_xy3)
#     iom2_1d_xy = (max_xy2 - min_xy1) / np.minimum(max_xy1 - min_xy1, max_xy2 - min_xy2)
# 
#     dist_sums_xy = -iom1_1d_xy - iom2_1d_xy
# 
#     filter_xy = (iom1_1d_xy < overlap_thres) \
#         & (iom2_1d_xy < overlap_thres) \
#         & (np.abs(iom1_1d_xy - iom2_1d_xy) < symmetry_thres) \
#         & (-iom1_1d_xy < distance_thres) \
#         & (-iom2_1d_xy < distance_thres) 
# 
#     dist_sums_xy = dist_sums_xy[filter_xy]
#     pairs_filt_xy = pairs_filt_xy[filter_xy]
# 
#     bbox1_rot_between_xy = bbox1_rot_between_xy[filter_xy]
#     bbox2_rot_between_xy = bbox2_rot_between_xy[filter_xy]
#     bbox3_rot_between_xy = bbox3_rot_between_xy[filter_xy]
# 
#     bbox1, bbox2 = get_2D_bboxes([1, 2], bbox1_rot_between_xy, bbox2_rot_between_xy)
#     iom1_xy = calculate_iom(bbox1, bbox2)
#     # IOU to second anchor
#     bbox1, bbox2 = get_2D_bboxes([1, 2], bbox1_rot_between_xy, bbox3_rot_between_xy)
#     iom2_xy = calculate_iom(bbox1, bbox2)
# 
#     filt = (iom1_xy > between_iom) & (iom2_xy > between_iom)
#     dist_sums_xy = dist_sums_xy[filt]
#     pairs_filt_xy = pairs_filt_xy[filt]
#     pairs_filt_xy = pairs_filt_xy[dist_sums_xy.argsort()][::2]
#     dist_sums_xy.sort()
#     dist_sums_xy = dist_sums_xy[::2]
#     # for pair in pairs_filt:
#         # print(f"{objects[anchor_idx]['raw_label']} between {objects[pair[0]]['raw_label']} and {objects[pair[0]]['raw_label']}")
# 
# 
# 
#     # Check z intersection
# 
#     between_z = (center1_rot[:, 2] > center2_rot[:, 2]) & (center1_rot[:, 2] < center3_rot[:, 2])
# 
#     pairs_filt_z = obj_pairs[between_z]
# 
#     bbox1_rot_between = bbox1_rot[between_z]
#     bbox2_rot_between = bbox2_rot[between_z]
#     bbox3_rot_between = bbox3_rot[between_z]
# 
#     max_z1 = bbox1_rot_between[..., 2].max(axis=-1)
#     min_z1 = bbox1_rot_between[..., 2].min(axis=-1)
#     max_z2 = bbox2_rot_between[..., 2].max(axis=-1)
#     min_z2 = bbox2_rot_between[..., 2].min(axis=-1)
#     max_z3 = bbox3_rot_between[..., 2].max(axis=-1)
#     min_z3 = bbox3_rot_between[..., 2].min(axis=-1)
# 
#     iom1_1d_z = (max_z1 - min_z3) / np.minimum(max_z1 - min_z1, max_z3 - min_z3)
#     iom2_1d_z = (max_z2 - min_z1) / np.minimum(max_z1 - min_z1, max_z2 - min_z2)
# 
#     dist_sums_z = -iom1_1d_z - iom2_1d_z
# 
#     filter_z = (iom1_1d_z < overlap_thres) & (iom2_1d_z < overlap_thres)
# 
#     dist_sums_z = dist_sums_z[filter_z]
#     pairs_filt_z = pairs_filt_z[filter_z]
# 
#     bbox1_rot_between = bbox1_rot_between[filter_z]
#     bbox2_rot_between = bbox2_rot_between[filter_z]
#     bbox3_rot_between = bbox3_rot_between[filter_z]
# 
#     # print(bbox1_rot_between.shape)
# 
#     iom1_z = calculate_iom_poly_vectorized(bbox1_rot_between[:, [0, 3, 2, 1], :], bbox2_rot_between[:, [0, 3, 2, 1], :])
#     iom2_z = calculate_iom_poly_vectorized(bbox1_rot_between[:, [0, 3, 2, 1], :], bbox3_rot_between[:, [0, 3, 2, 1], :])
# 
#     filt = (iom1_z > between_iom) & (iom2_z > between_iom)
#     dist_sums_z = dist_sums_z[filt]
#     pairs_filt_z = pairs_filt_z[filt]
# 
#     pairs_filt = np.concatenate([pairs_filt_xy, pairs_filt_z])
#     dist_sums = np.concatenate([dist_sums_xy, dist_sums_z])
#     pairs_filt = pairs_filt[dist_sums.argsort()]
# 
#     between_objs.extend([[objects[pair[0]]["object_id"], objects[pair[1]]["object_id"]] for pair in pairs_filt])
# 
# 
#     return between_objs

def get_aabb_distance(obj_a: dict, obj_b: dict) -> float:
    """
    Return the shortest Euclidean distance between two AABBs.

    The distance is:
    - zero if the boxes touch or intersect;
    - positive if they are separated.
    """

    min_a = np.asarray(obj_a["bbox_min"], dtype=np.float32)
    max_a = np.asarray(obj_a["bbox_max"], dtype=np.float32)

    min_b = np.asarray(obj_b["bbox_min"], dtype=np.float32)
    max_b = np.asarray(obj_b["bbox_max"], dtype=np.float32)

    # For each axis, calculate the empty space between the boxes.
    #
    # If the intervals overlap along an axis, the gap is zero.
    axis_gaps = np.maximum(
        np.maximum(
            min_b - max_a,
            min_a - max_b,
        ),
        0.0,
    )

    return float(np.linalg.norm(axis_gaps))

def relate_near(near_threshold, anchor_idx, objects):
    anchor_obj = objects[anchor_idx]
    near_object_ids = []

    for target_idx, target_obj in enumerate(objects):
        if target_idx == anchor_idx:
            continue
        distance = get_aabb_distance(
            anchor_obj,
            target_obj
        )
        if distance <= near_threshold:

            #ids and idx are kinda confusing but ok
            near_object_ids.append(target_obj['object_id'])
    return near_object_ids

# near_thres is some value between 0 and 1 for percentage of region size
# IDEA: check for nearness based on faces?
# def relate_near(near_thres, anchor_idx, objects, region_bbox):
#     anchor_coords = np.array(objects[anchor_idx]["bbox"])
#     anchor_center = objects[anchor_idx]["center"]
#     r_bbox = np.array(region_bbox)
#     r_xlen = np.max(r_bbox[:, 0]) - np.min(r_bbox[:, 0])
#     r_ylen = np.max(r_bbox[:, 1]) - np.min(r_bbox[:, 1])
#     r_zlen = np.max(r_bbox[:, 2]) - np.min(r_bbox[:, 2])
# 
#     #region volume
#     region_size = r_xlen * r_ylen * r_zlen
#     near_objs = []
# 
# 
# 
#     for i in range(len(objects)):
#         if i == anchor_idx:
#             continue
# 
#         object_coords = np.array(objects[i]["bbox"])
#         object_center = objects[i]["center"]
#         center_dist = np.linalg.norm(np.array(anchor_center) - np.array(object_center))
# 
#         # check if centers close
#         if (center_dist < near_thres * region_size):
#             near_objs.append(i)
#         else:
#             dists = []
#             # get distance between each pair of box coordinates
#             for p1 in anchor_coords:
#                 d = [np.linalg.norm(p1-p2) for p2 in object_coords]
#                 dists += d
# 
#             #print("dists", dists)
#             # object near if at least two points close to each other
#             if np.any(dists < near_thres * region_size) > 1:
#                 near_objs.append(i)
# 
#     near_obj_ids = [objects[ind]["object_id"] for ind in near_objs]
#     return near_obj_ids

def compute_spatial_relations(config, region_struct):
    objects = region_struct["objects"] # list of dicts
    region_bbox = region_struct["region_bbox"]
    relations = [
        # "between", 
        # "in", 
        #"on",
        # "beside", 
        "above", 
        "below", 
        "near"]
    between_iom = config["between_iom"]
    vertical_iom = config["vertical_iom"]
    near_thres = config["near_thres"]
    overlap_thres = config["overlap_thres"]
    symmetry_thres = config["symmetry_thres"]
    distance_thres = config["distance_thres"]
    anchor_size_thres = config["anchor_size_thres"]
    on_thres = config["on_thres"]
    in_thres = config["in_thres"]
    under_thres = config["under_thres"]

    for relation in relations:
        region_struct["relationships"][relation] = {}

    for anchor_idx, anchor_obj in enumerate(objects):
        anchor_id = objects[anchor_idx]["object_id"]

        # pairwise (binary) relations: above, below, in
        above_objs = relate_above(vertical_iom, on_thres, anchor_idx, objects)
        below_objs = relate_below(vertical_iom, under_thres, anchor_idx, objects)
        near_objs = relate_near(near_thres, anchor_idx, objects)
        # in_objs = relate_in(i, objects, in_thres)
        # on_objs = relate_on(vertical_iom, on_thres, under_thres, in_thres, i, objects)

        # triple object (ternary) relations: between
        # between_objs = relate_between(between_iom, i, objects, overlap_thres, symmetry_thres, distance_thres, anchor_size_thres)

        # store relationships per object at region-level
        #region_struct["relationships"]["beside"].append(beside_objs)
        #region_struct["relationships"]["between"].update({object_id:between_objs})
        #region_struct["relationships"]["in"].update({object_id:in_objs})
        #region_struct["relationships"]["on"].update({object_id:on_objs})
        region_struct["relationships"]["above"][anchor_idx] = above_objs
        region_struct["relationships"]["below"][anchor_idx]= below_objs
        region_struct["relationships"]["near"][anchor_idx]  = near_objs
    # print(between_time)
    return relations

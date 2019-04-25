
import cv2
import numpy as np
import general_pipeline as gp
import math
# import sys # DEBUG

import color_util as cu

def show(name, img):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, img)

# arrowbases = list of arrow_base centroids
arrowbases = []
# arrowheads = list of arrow_head centroids
arrowheads = []

arrowbase_hue_range = cu.grn_hue_range
arrowhead_hue_range = cu.red_hue_range

# Returns a mask for the bases of the arrows
def base_mask(img):
    blurred = gp.blur(img, 11)
    hsv_img = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = cu.color_mask(arrowbase_hue_range, 60, hsv_img)
    open_mask = gp.open(mask, 3)
    return open_mask

# Returns a mask for the heads of the arrows
def head_mask(img):
    blurred = gp.blur(img, 11)
    hsv_img = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = cu.color_mask(arrowhead_hue_range, 80, hsv_img)
    open_mask = gp.open(mask, 3)
    return open_mask

def arrow_or_label_mask(img, bin_img, state_centers):
    bases = base_mask(img)
    heads = head_mask(img)

    base_or_head_mask = cv2.bitwise_or(bases, heads)

    # Remove bases and heads, disconnecting paths from states
    result = cv2.bitwise_and(bin_img, cv2.bitwise_not(base_or_head_mask))

    # Fill each blob (to fill each state blob)
    blob_filled = gp.fill_blobs(result)

    # Remove each state blob
    for [x, y] in state_centers:
        cv2.floodFill(blob_filled, None, (int(x), int(y)), 0)

    # Unfill the insides of letters
    result = cv2.bitwise_and(result, blob_filled)

    return result

# Takes an hsv image, its binary image, and a list of state centers,
#   then returns a version of the image with only the arrows & their labels
def arrow_mask(img, bin_img, state_centers):
    bases = base_mask(img)

    base_centroids = gp.int_centroids(bases)

    # Get the arrows and the labels
    result = arrow_or_label_mask(img, bin_img, state_centers)

    # Add bases (will be removed later)
    result = cv2.bitwise_or(result, bases)

    # Color the arrows in a specific hue
    for [x, y] in base_centroids:
        cv2.floodFill(result, None, (int(x), int(y)), 1)

    # Remove bases
    result = cv2.bitwise_and(result, cv2.bitwise_not(bases))

    result[result != 1] = 0
    result *= 255

    return result

def label_mask(img, bin_img, state_centers):
    bases = base_mask(img)

    base_centroids = gp.int_centroids(bases)

    # Get the arrows and the labels
    result = arrow_or_label_mask(img, bin_img, state_centers)

    # Add bases (will be removed later)
    result = cv2.bitwise_or(result, bases)

    # Remove the arrows and bases
    for [x, y] in base_centroids:
        cv2.floodFill(result, None, (int(x), int(y)), 0)

    return result

# Takes a color image, the thresholded binary version of the image,
#   and and array of (integer) state centers.
# Returns an array of [arrow base position, arrow head position] pairs.
def base_to_head_centroids(img, bin_img, state_centers):
    bases = base_mask(img)
    heads = head_mask(img)

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    arrows = arrow_mask(img, bin_img, state_centers)

    base_centroids = gp.int_centroids(bases)
    head_centroids = gp.int_centroids(heads)

    if len(base_centroids) != len(head_centroids):
        raise ValueError(f"bad iamge for base_to_head_centroids."
            f"There are {len(base_centroids)} bases and {len(head_centroids)} heads, but there should be the same of each.")

    # A mask with the bases, the heads, and the arrows
    base_head_arrow_mask = cv2.bitwise_or(bases, cv2.bitwise_or(heads, arrows))

    # Dilate to make sure bases and arrows touch
    base_head_arrow_mask = gp.dilate(base_head_arrow_mask, 5)

    # Result will contain an array of [base_coord, head_coord] pairs
    result = []

    # Color the contiguous chunk with the index of the base centroid, plus 1
    for i, base_centroid in enumerate(base_centroids):
        cv2.floodFill(base_head_arrow_mask, None, (base_centroid[0], base_centroid[1]), i + 1)

    # Check the color of the head centroid pixel;
    #   it is the index of the base centroid it is connected to.
    for head_centroid in head_centroids:
        label = base_head_arrow_mask[head_centroid[1], head_centroid[0]]
        base_centroid = base_centroids[label - 1]
        result.append([base_centroid, head_centroid])

    return result

# Takes a color image, the thresholded binary version of the image,
#   and and array of (integer) state centers.
# Returns a list. Each item in the list has the following form:
#   [arrow_base_centroid, arrow_centroid, arrow_head_centroid].
def base_to_arrow_to_head_centroids(img, bin_img, state_centers):
    bases = base_mask(img)
    heads = head_mask(img)

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    arrows = arrow_mask(img, bin_img, state_centers)
    labels = label_mask(img, bin_img, state_centers)

    base_centroids = gp.int_centroids(bases)
    head_centroids = gp.int_centroids(heads)
    arrow_centroids = gp.centroids(arrows)

    if len(base_centroids) != len(head_centroids) or len(base_centroids) != len(arrow_centroids):
        raise ValueError(f"bad image for base_to_arrow_to_head_centroids."
            f"There are {len(base_centroids)} bases, {len(head_centroids)} heads, and {len(arrow_centroids)} arrows, but there should be the same of each.")

    # A mask with the bases, the heads, and the arrows
    base_head_arrow_mask = cv2.bitwise_or(bases, cv2.bitwise_or(heads, arrows))

    # Dilate to make sure bases and arrows touch
    index_mask = gp.dilate(base_head_arrow_mask, 5)

    # Result will contain an array of [base_coord, head_coord] pairs
    base_coord_to_head_coords = [ None ] * len(base_centroids)

    # Color the contiguous chunk with the index of the base centroid, plus 1
    for i, base_centroid in enumerate(base_centroids):
        cv2.floodFill(index_mask, None, (base_centroid[0], base_centroid[1]), i + 1)

    # Check the color of the head centroid pixel;
    #   it is the index of the base centroid it is connected to.
    for head_centroid in head_centroids:
        label = index_mask[head_centroid[1], head_centroid[0]]
        base_centroid = base_centroids[label - 1]
        base_coord_to_head_coords[label - 1] = [base_centroid, head_centroid]

    # Each stat has the form [xleft, ytop, width, height, area]
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(arrows)

    result = []
    stats_and_centroids = zip(stats[1:], centroids[1:])

    for i, ([xleft, ytop, width, height, area], centroid) in enumerate(stats_and_centroids):
        body_coord = (None, None)

        window = labels[ytop][xleft:xleft+width]
        # Find a pixel that belongs to the arrow body
        for index, item in enumerate(window):
            if item == (i+1):
                body_coord = [ytop, index + xleft]
                break

        # Get which label the pixel has to when colored with the indices
        label = index_mask[body_coord[0], body_coord[1]]
        base_coord, head_coord = base_coord_to_head_coords[label - 1]
        result.append([base_coord, centroid, head_coord])

    return result

# Takes a color image, the thresholded binary version of the image,
#   and and array of (integer) state centers.
# Returns a list. Each item in the list has the following form:
#   [slice of the image that contains the arrow label,
#    base_centroid, head_centroid].
def label_dim_base_to_head_centroids(img, bin_img, state_centers):
    bases_arrows_heads = base_to_arrow_to_head_centroids(img, bin_img, state_centers)

    bases = [base for base, arrow, head in bases_arrows_heads]
    arrows = [arrow for base, arrow, head in bases_arrows_heads]
    heads = [head for base, arrow, head in bases_arrows_heads]

    labels = label_mask(img, bin_img, state_centers)
    _, _, label_stats, label_centroids = cv2.connectedComponentsWithStats(labels)
    label_stats = label_stats[1:]
    label_centroids = label_centroids[1:]

    if len(bases_arrows_heads) != len(label_centroids):
        raise ValueError(f"bad image for label_dim_base_to_head_centroids."
            f"There are {len(bases_arrows_heads)} arrows and {len(label_centroids)} labels, but there should be the same of each.")

    results = [ None ] * len(bases)

    # Find closest arrow center.
    # This is O(n^2) but our maximum n is small,
    #   so it's fine
    for i, label_centroid in enumerate(label_centroids):
        min_sqr_dist = math.inf
        for j, arrow in enumerate(arrows):
            if centroid_sqr_dist(label_centroid, arrow) < min_sqr_dist:
                min_j = j
                min_sqr_dist = centroid_sqr_dist(label_centroid, arrow)

        [xleft, ytop, width, height, area] = label_stats[i]

        if results[min_j] != None:
            raise ValueError(f"bad image for label_dim_base_to_head_centroids."
                f"Multiple labels are closest to one arrow.")

        results[min_j] = [
            [ [ytop, ytop+height], [xleft, xleft+width] ],
            bases[min_j],
            heads[min_j]
        ]

    return results

# TODO: Once we have a representation of states, this should use states instead of state centers.
def state_label_state(img, bin_img, state_centers):

    labeldims_bases_heads = label_dim_base_to_head_centroids(img, bin_img, state_centers)

    labeldims = [labeldim for labeldim, base, head in labeldims_bases_heads]
    bases = [base for labeldim, base, head in labeldims_bases_heads]
    heads = [head for labeldim, base, head in labeldims_bases_heads]
#    print(bases)
#    print(heads)

    newbases = []
    # Find closest state center.
    # This is O(n^2) but our maximum n is small,
    #   so it's fine
    for i, base in enumerate(bases):
        min_sqr_dist = math.inf
        for j, state_center in enumerate(state_centers):
            if centroid_sqr_dist(base, state_center) < min_sqr_dist:
                min_j = j
                min_sqr_dist = centroid_sqr_dist(base, state_center)
#        print("min_j", min_j)
        newbase = state_centers[min_j]
        newbases.append(newbase)
#        print("newbase", newbase)

    newheads = []
    for i, head in enumerate(heads):
        min_sqr_dist = math.inf
        for j, state_center in enumerate(state_centers):
            if centroid_sqr_dist(head, state_center) < min_sqr_dist:
                min_j = j
                min_sqr_dist = centroid_sqr_dist(head, state_center)
#        print("min_j", min_j)
        newhead = state_centers[min_j]
        newheads.append(newhead)
#        print("newhead", newhead)

    results = list(zip(newbases, newheads))
#    print(results)
    return results

"""
Get the squared distance between two centroids
"""
def centroid_sqr_dist(centroid1, centroid2):
    return (centroid1[0] - centroid2[0])**2 + (centroid1[1] - centroid2[1])**2

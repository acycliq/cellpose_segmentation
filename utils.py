from numba import jit
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageDraw
import diplib as dip
import os
import logging

logger = logging.getLogger()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s"
)


def intersection_over_union(masks_true, masks_pred):
    """ intersection over union of all mask pairs

    Parameters
    ------------

    masks_true: ND-array, int
        ground truth masks, where 0=NO masks; 1,2... are mask labels
    masks_pred: ND-array, int
        predicted masks, where 0=NO masks; 1,2... are mask labels

    Returns
    ------------

    iou: ND-array, float
        matrix of IOU pairs of size [x.max()+1, y.max()+1]

    """
    overlap = _label_overlap(masks_true, masks_pred)
    n_pixels_pred = np.sum(overlap, axis=0, keepdims=True)
    n_pixels_true = np.sum(overlap, axis=1, keepdims=True)
    iou = overlap / (n_pixels_pred + n_pixels_true - overlap)
    iou[np.isnan(iou)] = 0.0
    return iou


@jit(nopython=True)
def _label_overlap(x, y):
    """ fast function to get pixel overlaps between masks in x and y

    Parameters
    ------------

    x: ND-array, int
        where 0=NO masks; 1,2... are mask labels
    y: ND-array, int
        where 0=NO masks; 1,2... are mask labels

    Returns
    ------------

    overlap: ND-array, int
        matrix of pixel overlaps of size [x.max()+1, y.max()+1]

    """
    x = x.ravel()
    y = y.ravel()
    overlap = np.zeros((1 + x.max(), 1 + y.max()), dtype=np.uint)
    for i in range(len(x)):
        overlap[x[i], y[i]] += 1
    return overlap


def stitch3D(masks, stitch_threshold=0.25):
    """ stitch 2D masks into 3D volume with stitch_threshold on IOU """
    mmax = masks[0].max()
    for i in range(len(masks)-1):
        iou = intersection_over_union(masks[i+1], masks[i])[1:,1:]
        iou[iou < stitch_threshold] = 0.0
        iou[iou < iou.max(axis=0)] = 0.0
        istitch = iou.argmax(axis=1) + 1
        ino = np.nonzero(iou.max(axis=1)==0.0)[0]
        istitch[ino] = np.arange(mmax+1, mmax+len(ino)+1, 1, int)
        mmax += len(ino)
        istitch = np.append(np.array(0), istitch)
        masks[i+1] = istitch[masks[i+1]]
    return masks


def draw_poly(polys, jpg_filename):
    img = Image.open(jpg_filename).convert('RGB')

    img2 = img.copy()
    draw = ImageDraw.Draw(img2)
    for poly in polys:
        poly = [tuple(d) for d in poly]
        draw.line(poly, fill="#ffff00", width=1)
        # img3 = Image.blend(img, img2, 0.4)
    return img2


def get_jpg(i):
    """
    retrieves the jpg to draw the boundaries on
    """
    # 1. first check if there is already a jpg
    jpg_page = os.path.join(BOUNDARIES_JPG_DIR, 'dapi_image_rescaled_%s.jpg' % str(i).zfill(4))
    if os.path.isfile(jpg_page):
        return jpg_page
    else:
        return os.path.join(FRAMES_JPG_DIR, 'dapi_image_rescaled_%s.jpg' % str(i).zfill(4))


def draw_boundaries(img, masks):
    for i, mask in enumerate(masks):
        offset_x = 0
        offset_y = 0
        boundaries = extract_borders_dip(mask.astype(np.uint32), offset_x, offset_y, [0])

        polys = boundaries.coords.values
        jpg_filename = get_jpg(i)
        if len(polys) > 0:
            res = draw_poly(polys, jpg_filename)
            jpg_page = os.path.join(BOUNDARIES_JPG_DIR, os.path.basename(jpg_filename))
            res.save(jpg_page)
            logger.info('Boundaries saved at %s' % jpg_page)


def extract_borders_dip(label_image, offset_x=0, offset_y=0, ignore_labels=[0]):
    """
    Takes in a label_image and extracts the boundaries. It is assumed that the background
    has label = 0
    Parameters
    ----------
    label_image: the segmentation mask, a 2-D array
    offset_x: Determines how much the boundaries will be shifted by the on the x-axis
    offset_y: Determines how much the boundaries will be shifted by the on the y-axis
    ignore_labels: list of integers. If you want to ignore some masks, put the corresponding label in this list

    Returns
    -------
    Returns a dataframa with two columns, The first one is the mask label and the second column keeps the coordinates
    of the mask boundaries
    """
    labels = sorted(set(label_image.flatten()) - {0} - set(ignore_labels))
    cc = dip.GetImageChainCodes(label_image)  # input must be an unsigned integer type
    d = {}
    for c in cc:
        if c.objectID in labels:
            # p = np.array(c.Polygon())
            p = c.Polygon().Simplify()
            p = p + np.array([offset_x, offset_y])
            p = np.uint64(p).tolist()
            p.append(p[0])  # append the first pair at the end to close the polygon
            d[np.uint64(c.objectID)] = p
        else:
            pass
    df = pd.DataFrame([d]).T
    df = df.reset_index()
    df.columns = ['label', 'coords']
    return df


def rgb2hex(rgb):
    r = rgb[0]
    g = rgb[1]
    b = rgb[2]
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)


def pallete(id):
    if id % 24 == 0:
        return (2, 63, 165)
    elif id % 24 == 1:
        return (125, 135, 185)
    elif id % 24 == 2:
        return (190, 193, 212)
    elif id % 24 == 3:
        return (214, 188, 192)
    elif id % 24 == 4:
        return (187, 119, 132)
    elif id % 24 == 5:
        return (142, 6, 59)
    elif id % 24 == 6:
        return (74, 111, 227)
    elif id % 24 == 7:
        return (133, 149, 225)
    elif id % 24 == 8:
        return (181, 187, 227)
    elif id % 24 == 9:
        return (230, 175, 185)
    elif id % 24 == 10:
        return (224, 123, 145)
    elif id % 24 == 11:
        return (211, 63, 106)
    elif id % 24 == 12:
        return (17, 198, 56)
    elif id % 24 == 13:
        return (141, 213, 147)
    elif id % 24 == 14:
        return (198, 222, 199)
    elif id % 24 == 15:
        return (234, 211, 198)
    elif id % 24 == 16:
        return (240, 185, 141)
    elif id % 24 == 17:
        return (239, 151, 8)
    elif id % 24 == 18:
        return (15, 207, 192)
    elif id % 24 == 19:
        return (156, 222, 214)
    elif id % 24 == 20:
        return (213, 234, 231)
    elif id % 24 == 21:
        return (243, 225, 235)
    elif id % 24 == 22:
        return (246, 196, 225)
    elif id % 24 == 23:
        return (247, 156, 212)


def get_colour(labels):
    rgb = [pallete(d) for d in labels]
    return [rgb2hex(c) for c in rgb]
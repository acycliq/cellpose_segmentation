import skimage.io
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageDraw
from cellpose import models
import diplib as dip
import os
import torch
import utils
import logging


logger = logging.getLogger()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s"
)

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))

# folder keeping the frames of the 3D dapi as jpgs
FRAMES_JPG_DIR = os.path.join(ROOT_DIR, 'data', 'frames', 'jpg')

# We draw the cell boundaries on each of the jpgs. Keep here (new) jpgs with the segmentations
BOUNDARIES_JPG_DIR = os.path.join(ROOT_DIR, 'data', 'out', 'boundaries')

# use_GPU = models.use_gpu()
# print('>>> GPU activated? %d'%use_GPU)
print(os.system('nvcc --version'))
print(os.system('nvidia-smi'))
torch.cuda.empty_cache()

cellpose_ini = {
    'channels': [0, 0],
    'diameter': 7.0,
    'batch_size': 2,
    'anisotropy': 1.0,
    'mask_threshold': 0.0,
    # 'flow_threshold': 4.0
}

# remove the first three pages from the stacked image
# because they do not have any cells (only backrgound)
# ans that causes cellpose to crash when it tries to
# stitch the masks
empty_pages = [0, 1, 2]

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


def draw_poly(polys, colours, jpg_filename):
    img = Image.open(jpg_filename).convert('RGB')

    img2 = img.copy()
    draw = ImageDraw.Draw(img2)
    for i, poly in enumerate(polys):
        poly = [tuple(d) for d in poly]
        draw.line(poly, fill=colours[i], width=1)
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
        boundaries['colour'] = utils.get_colour(boundaries.label.values)
        polys = boundaries.coords.values
        k = len(empty_pages)
        jpg_filename = get_jpg(i+k)
        if len(polys) > 0:
            res = draw_poly(polys, boundaries['colour'].values, jpg_filename)
            jpg_page = os.path.join(BOUNDARIES_JPG_DIR, os.path.basename(jpg_filename))
            res.save(jpg_page)
            logger.info('Boundaries saved at %s' % jpg_page)


def segment(img_3D, use_stiching=False):
    # DEFINE CELLPOSE MODEL
    model = models.Cellpose(gpu=True, model_type='nuclei')

    if use_stiching:
        masks, flows, styles, _ = model.eval(img_3D,
                                             channels=cellpose_ini['channels'],
                                             batch_size=cellpose_ini['batch_size'],
                                             diameter=cellpose_ini['diameter'],
                                             do_3D=False,
                                             # mask_threshold=cellpose_ini['mask_threshold'],
                                             # flow_threshold=cellpose_ini['flow_threshold'],
                                             stitch_threshold=0.5)
        np.savez('masks_2D_stiched.npz', masks)
    else:
        masks, flows, styles, diams = model.eval(img_3D,
                                                 channels=cellpose_ini['channels'],
                                                 batch_size=cellpose_ini['batch_size'],
                                                 diameter=cellpose_ini['diameter'],
                                                 anisotropy=cellpose_ini['anisotropy'],
                                                 do_3D=True)

        np.savez('masks_rescaled_anisotropy_1.0.npz', masks)
    logger.info('Masks saved to disk!')
    return masks


def reshape_img(img):
    z, c, h, w = img.shape
    if c == 1:
        out = np.zeros([z, 2, h, w])
        out[:, 0, :, :] = img[:, 0, :, :]
    else:
        out = img
    return out


def remove_pages(img, ids):
    """
    Some pages are empty, remove them cause they crash z-stiching
    """
    z = img.shape[0]
    new_list = [page for page in range(z) if page not in ids]
    return img[new_list]


def main(img_path, use_stiching=False):
    # For multi - channel, multi-Z tiff's, the expected format is Z x channels x Ly x Lx.
    _img = skimage.io.imread(img_path).transpose([0, 3, 1, 2])
    img = reshape_img(_img)
    # img = remove_pages(img, empty_pages)
    masks = segment(img, use_stiching)
    draw_boundaries(img, masks)


if __name__ == "__main__":
    img_path = r"data/3D_dapi/dapi_image_rescaled_zxyc.tif"
    main(img_path, use_stiching=True)
    logger.info('ok, all done')
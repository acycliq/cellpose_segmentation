import skimage.io
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageDraw
from cellpose import models
import diplib as dip
import os
import torch
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

use_GPU = models.use_gpu()
print('>>> GPU activated? %d'%use_GPU)
print(os.system('nvcc --version'))
print(os.system('nvidia-smi'))
torch.cuda.empty_cache()

cellpose_ini = {
    'channels': [0, 0],
    'diameter': 7.0,
    'batch_size': 2,
    'anisotropy': 1.0,
}


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


def segment(img_3D):
    # DEFINE CELLPOSE MODEL
    model = models.Cellpose(gpu=use_GPU, model_type='nuclei')

    masks, flows, styles, diams = model.eval(img_3D,
                                             channels = cellpose_ini['channels'],
                                             batch_size = cellpose_ini['batch_size'],
                                             diameter = cellpose_ini['diameter'],
                                             anisotropy = cellpose_ini['anisotropy'],
                                             do_3D=True)
    np.savez('masks_rescaled_anisotropy_1.0.npz', masks)
    logger.info('Masks saved to disk!')
    return masks


def main(img_path):
    # For multi - channel, multi-Z tiff's, the expected format is Z x channels x Ly x Lx.
    img = skimage.io.imread(img_path).transpose([0, 3, 1, 2])
    masks = segment(img)
    draw_boundaries(img, masks)


if __name__ == "__main__":
    img_path = r"data/3D_dapi/dapi_image_rescaled_zxyc.tif"
    main(img_path)
    logger.info('ok, all done')
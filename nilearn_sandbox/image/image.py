"""
Preprocessing functions for images.

See also nilearn.signal.
"""
# License: simplified BSD

import collections

import numexpr as ne
from scipy import ndimage
from scipy.stats import scoreatpercentile
import copy
import nibabel
from sklearn.externals.joblib import Parallel, delayed

from .. import signal
from .._utils import (check_niimg_4d, check_niimg_3d, check_niimg, as_ndarray,
                      _repr_niimgs)
from .._utils.niimg_conversions import _index_img, _check_same_fov
from .._utils.niimg import _safe_get_data
from .._utils.compat import _basestring, get_affine, get_header
from .._utils.param_validation import check_threshold


def math_img(formula, **imgs):
    """Interpret a numpy based string formula using niimg in named parameters.
    This version is similar to nilearn.image.math_img, however uses numexpr
    instead of `eval` directly.

    Parameters
    ----------
    formula: str
        The mathematical formula to apply to image internal data. It can use
        numpy imported as 'np'.
    imgs: images (Nifti1Image or file names)
        Keyword arguments corresponding to the variables in the formula as
        Nifti images. All input images should have the same geometry (shape,
        affine).

    Returns
    -------
    return_img: Nifti1Image
        Result of the formula as a Nifti image. Note that the dimension of the
        result image can be smaller than the input image. The affine is the
        same as the input image.

    See Also
    --------
    nilearn.image.mean_img : To simply compute the mean of multiple images

    Examples
    --------
    Let's load an image using nilearn datasets module::

     >>> from nilearn import datasets
     >>> anatomical_image = datasets.load_mni152_template()

    Now we can use any numpy function on this image::

     >>> from nilearn.image import math_img
     >>> log_img = math_img("np.log(img)", img=anatomical_image)

    We can also apply mathematical operations on several images::

     >>> result_img = math_img("img1 + img2",
     ...                       img1=anatomical_image, img2=log_img)

    Notes
    -----

    This function is the Python equivalent of ImCal in SPM or fslmaths
    in FSL.

    """
    try:
        # Check that input images are valid niimg and have a compatible shape
        # and affine.
        niimgs = []
        for image in imgs.values():
            niimgs.append(check_niimg(image))
        _check_same_fov(*niimgs, raise_error=True)
    except Exception as exc:
        exc.args = (("Input images cannot be compared, you provided '{0}',"
                     .format(imgs.values()),) + exc.args)
        raise

    # Computing input data as a dictionary of numpy arrays. Keep a reference
    # niimg for building the result as a new niimg.
    niimg = None
    data_dict = {}
    for key, img in imgs.items():
        niimg = check_niimg(img)
        data_dict[key] = _safe_get_data(niimg)

    # Add a reference to numpy in the kwargs of eval so that numpy functions
    # can be called from there.
    try:
        result = ne.evaluate(formula, local_dict=data_dict)
    except Exception as exc:
        exc.args = (("Input formula couldn't be processed, you provided '{0}',"
                     .format(formula),) + exc.args)
        raise

    return new_img_like(niimg, result, get_affine(niimg))

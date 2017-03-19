"""
Test image pre-processing functions
"""
from nose.tools import assert_true, assert_false, assert_equal
from nose import SkipTest

import platform
import os
import nibabel
from nibabel import Nifti1Image
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
from nilearn._utils.testing import assert_raises_regex

from nilearn_sandbox.image import math_img

X64 = (platform.architecture()[0] == '64bit')

currdir = os.path.dirname(os.path.abspath(__file__))
datadir = os.path.join(currdir, 'data')


def test_math_img_exceptions():
    img1 = Nifti1Image(np.ones((10, 10, 10, 10)), np.eye(4))
    img2 = Nifti1Image(np.zeros((10, 20, 10, 10)), np.eye(4))
    img3 = Nifti1Image(np.ones((10, 10, 10, 10)), np.eye(4))
    img4 = Nifti1Image(np.ones((10, 10, 10, 10)), np.eye(4) * 2)

    formula = "np.mean(img1, axis=-1) - np.mean(img2, axis=-1)"
    # Images with different shapes should raise a ValueError exception.
    assert_raises_regex(ValueError,
                        "Input images cannot be compared",
                        math_img, formula, img1=img1, img2=img2)

    # Images with different affines should raise a ValueError exception.
    assert_raises_regex(ValueError,
                        "Input images cannot be compared",
                        math_img, formula, img1=img1, img2=img4)

    bad_formula = "np.toto(img1, axis=-1) - np.mean(img3, axis=-1)"
    assert_raises_regex(AttributeError,
                        "Input formula couldn't be processed",
                        math_img, bad_formula, img1=img1, img3=img3)


def test_math_img():
    img1 = Nifti1Image(np.ones((10, 10, 10, 10)), np.eye(4))
    img2 = Nifti1Image(np.zeros((10, 10, 10, 10)), np.eye(4))
    expected_result = Nifti1Image(np.ones((10, 10, 10)), np.eye(4))

    formula = "np.mean(img1, axis=-1) - np.mean(img2, axis=-1)"
    for create_files in (True, False):
        with testing.write_tmp_imgs(img1, img2,
                                    create_files=create_files) as imgs:
            result = math_img(formula, img1=imgs[0], img2=imgs[1])
            assert_array_equal(result.get_data(),
                               expected_result.get_data())
            assert_array_equal(compat.get_affine(result),
                               compat.get_affine(expected_result))
            assert_equal(result.shape, expected_result.shape)

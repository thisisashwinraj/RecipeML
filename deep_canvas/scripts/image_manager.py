# Author: Ashwin Raj <thisisashwinraj@gmail.com>
# License: GNU Affero General Public License v3.0
# Discussions-to: github.com/thisisashwinraj/RecipeML-Recipe-Recommendation

# Copyright (C) 2023 Ashwin Raj

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""
This module provides functionality for resizing an image, while maintaining their 
original aspect ratio. It includes a class `ImageTransformation` with methods for 
initializing the transformation settings, resizing the images, and editing images.

Depending on individual cases, the program may be modified to use multiprocessing 
capailities to increase the speed of data processing, on eligible local computers.
The usage of each class & their methods are described in corresponding docstrings.

Classes and Functions:
    [1] ImageTransformation (class)
        [a] resize_image

.. versionadded:: 1.3.0

Learn about RecipeML :ref:`RecipeML v1: Food Image Generation using Generative AI`
"""
import sys
import logging
import datetime
import pandas as pd

import requests
from PIL import Image
from io import BytesIO


class ImageTransformation:
    """
    Class for performing image transformation operations including image resizing

    This class encapsulates the functionality for resizing image arguments, while
    maintaining the original aspect ratio. The resize_image method takes an input
    image, desired width, and height parameters, to perform cropping and resizing.

    Class Methods:
        [1] resize_image

    .. versionadded:: 1.3.0

    The performance of the methods present in the class can be optimized by using
    the CPUPool via the multithreading capailities on eligible local/cloud system.
    """

    def __init__(self, image_width=512, image_heigth=512):
        self.image_width = image_width  # The original width of the recipe's imag
        self.image_heigth = image_heigth  # Original height of the recipe's image

    def resize_image(self, image, desired_width=512, desired_height=512):
        """
        Method to generate visually appealing images using OpenAIs DALL.E 2 model

        Resize the argument image to the desired dimensions, while preserving the
        original aspect ratio. Resulting image is cropped and centered, if needed.

        Read more in :ref:`RecipeML: Generative AI using StableDiffusion & OpenAI`

        .. versionadded:: 1.3.0

        Parameters:
            [PIL.Image.Image] recipe_name: Image argument to be resized & cropped

        Returns:
            [PIL.Image.Image] recipe_name: The desired resized, and cropped image

        NOTE: Keep track of the API usage here: platform.openai.com/account/usage
        """
        aspect_ratio = self.image_width / self.image_heigth  # Check aspect ratio

        # Determine cropping dimensions based on aspect ratio & desired dimension
        if desired_width / desired_height > aspect_ratio:
            crop_width = self.image_width
            crop_height = int(self.image_width * desired_height / desired_width)

        else:
            crop_width = int(self.image_heigth * desired_width / desired_height)
            crop_height = self.image_heigth

        # Calculate the co-ordinates for the offset, to center the cropped region
        offset_x = int((self.image_width - crop_width) / 2)
        offset_y = int((self.image_heigth - crop_height) / 2)

        # Crop the resultant image based on the calculated dimensions and offsets
        cropped_image = image.crop(
            (offset_x, offset_y, offset_x + crop_width, offset_y + crop_height)
        )
        resized_image = cropped_image.resize((desired_width, desired_height))

        return resized_image  # Resize the cropped image, & return the PIL object

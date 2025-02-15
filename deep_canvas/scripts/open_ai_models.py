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
This module contains the classes, and functions for recipe recommendation & image 
generation using various machine learning models and techniques, thus creating an 
enriching experience for user. Different classes enable different functionalities.

Depending on individual cases, the program may be modified to use multiprocessing 
capailities to increase the speed of data processing, on eligible local computers.
The usage of each class & their methods are described in corresponding docstrings.

Classes and Functions:
    [1] DALLE2 (class)
        [a] generate_recipe_image

.. versionadded:: 1.3.0

Learn about RecipeML :ref:`RecipeML: Image Generation using OpenAIs DALL.E2 model`
"""
import sys
import logging
import datetime
import pandas as pd

import requests
from PIL import Image
from io import BytesIO
from openai import OpenAI

try:
    from deep_canvas.scripts.image_manager import ImageTransformation
except:
    from scripts.image_manager import ImageTransformation

from configurations.resource_path import ResourceRegistry


class DALLE2(ImageTransformation):
    """
    Class to generate recipe images using OpenAIs DALL.E 2 image generation model.

    This class uses OpenAI's DALL.E 2 image generation model to generate visually
    appealing recipe images. It takes a recipe name as input and creates an image
    that highlights the presentation and aesthetics of the dish for use in webapp.

    Class Methods:
        [1] generate_recipe_image

    .. versionadded:: 1.1.0

    The performance of the methods present in the class can be optimized by using
    the CPUPool via the multithreading capailities on eligible local/cloud system.
    """

    def __init__(self, openai_api_key):
        self.openai_api_key = openai_api_key
        # Fetch OpenAIs API credentials from ~/secrets.toml via secret management
        self.client = OpenAI(api_key=self.openai_api_key)

    def generate_recipe_image(
        self, recipe_name, desired_width=512, desired_height=512, quality="standard"
    ):
        """
        Method to generate visually appealing images using OpenAIs DALL.E 2 model

        This method reads recipe name from the user and uses the OpenAIs DALL.E 2
        image generation model, to generate visually appealing image of the input
        recipe. This model uses the OpenAI API. Replace API key in ~/secrets.toml

        Read more in :ref:`RecipeML: Generative AI using StableDiffusion & OpenAI`

        .. versionadded:: 1.1.0

        Parameters:
            [string] recipe_name: Recipe name, for which image is to be generated

        Returns:
            [string] file_name: Location where the generated image is to be saved

        NOTE: Keep track of the API usage here: platform.openai.com/account/usage
        """
        text_description = f"Create a visually appealing recipe image featuring a beautifully plated dish of {recipe_name}. The composition should highlight the colors, textures, and presentation of the dish. Pay special attention to lighting and styling to make the dish look as enticing as possible"

        if quality == "high":
            # Generate a 1024x1024 image basis the input prompt using the DALL.E2
            response = self.client.images.generate(
                prompt=text_description, n=1, size="1024x1024"
            )
        elif quality == "standard":
            # Generate the 512x512 image basis the input prompt using the DALL.E2
            response = self.client.images.generate(
                prompt=text_description, n=1, size="512x512"
            )
        else:
            # Generate the 256x256 image basis the input prompt using the DALL.E2
            response = self.client.images.generate(
                prompt=text_description, n=1, size="256x256"
            )

        image_url = response.data[0].url  # Fetch the link of the generated image

        # Retrieve the image generated by the DALLE2 model & save it in ~/exports
        image_response = requests.get(image_url)
        image_bytes = BytesIO(image_response.content)

        image = Image.open(image_bytes)
        original_width, original_height = image.size  # Get dimensions of gen img

        if (original_width != desired_width) or (original_height != desired_height):
            # Initialize new Image object with the desired dimensions as required
            super().__init__(original_width, original_height)
            image = self.resize_image(image, desired_width, desired_height)

        resource_registry = ResourceRegistry()
        file_name = (
            resource_registry.generated_images_directory_path
            + recipe_name.replace("/", "").replace(" ", "_").lower()
            + "_"
            + str(desired_width)
            + "x"
            + str(desired_height)
            + "_dalle2.png"
        )

        image.save(file_name)
        return file_name  # Save the generated image and then return the filename

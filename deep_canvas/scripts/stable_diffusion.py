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
    [1] RunwayML (class)
        [a] generate_recipe_image

    [2] PlaygroundAI (class)
        [a] generate_recipe_image

.. versionadded:: 1.1.0

Learn about RecipeML :ref:`RecipeML v1.3: Image Generation using Stable Diffusion`
"""
import sys
import logging
import datetime
import pandas as pd

import requests
from PIL import Image
from io import BytesIO

import torch
from diffusers import StableDiffusionPipeline, DiffusionPipeline
from multiprocessing import Pool, cpu_count

try:
    from deep_canvas.scripts.image_manager import ImageTransformation
except:
    from scripts.image_manager import ImageTransformation

from configurations.resource_path import ResourceRegistry


class RunwayML(ImageTransformation):
    """
    Class to generate recipe images using the Runway ML's Stable Diffusion models.

    This class uses the Stable Diffusion model from RunwayML to generate visually
    appealing recipe images. It takes a recipe name as input and creates an image
    that highlights the presentation and aesthetics of the dish for use in webapp.


    Class Methods:
        [1] generate_recipe_image

    .. versionadded:: 1.1.0

    The performance of the methods present in the class can be optimized by using
    the CPUPool via the multithreading capailities on eligible local/cloud system.
    """

    def __init__(self):
        # Specify the hugging faces model ID and create a StableDiffusionPipeline
        model_id = "runwayml/stable-diffusion-v1-5"

        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16
        )
        self.pipe = pipe.to("cuda")

    def generate_recipe_image(self, recipe_name, desired_width, desired_height):
        """
        Method to generate visually appealing images using Stable Diffusion model

        This method reads recipe name from the user and uses the Stable Diffusion
        model from Runway ML, to generate a visually appealing image of the input
        recipe. This model uses CUDA, for GPU computation of generative algorithm.

        Read more in :ref:`RecipeML: Generative AI using StableDiffusion & OpenAI`

        .. versionadded:: 1.1.0

        Parameters:
            [string] recipe_name: Recipe name, for which image is to be generated

        Returns:
            [string] file_name: Location where the generated image is to be saved
        """
        # Define a prompt to feed into StableDiffusion model for image generation
        prompt = f"Create a visually appealing recipe image featuring a beautifully plated dish of {recipe_name}. The composition should highlight the colors, textures, and presentation of the dish. Pay special attention to lighting and styling to make the dish look as enticing as possible"
        
        image = self.pipe(prompt).images[0]  # Generate img based on input prompt

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
            + "_runwayml.png"
        )

        image.save(file_name)
        return file_name  # Save the generated image and then return the filename


class PlaygroundAI(ImageTransformation):
    """
    Class to generate recipe image using the PlaygroundAIs Stable Diffusion model.

    This class uses Stable Diffusion model from PlaygroundAI to generate visually
    appealing recipe images. It takes a recipe name as input and creates an image
    that highlights the presentation and aesthetics of the dish for use in webapp.

    Class Methods:
        [1] generate_recipe_image

    .. versionadded:: 1.1.0

    The performance of the methods present in the class can be optimized by using
    the CPUPool via the multithreading capailities on eligible local/cloud system.
    """

    def __init__(self):
        # Specify the hugging faces model ID and create a StableDiffusionPipeline
        pipe = DiffusionPipeline.from_pretrained(
            "playgroundai/playground-v2-1024px-aesthetic",
            torch_dtype=torch.float16,
            use_safetensors=True,
            add_watermarker=False,
            variant="fp16",
        )
        self.pipe = pipe.to("cuda")

    def generate_recipe_image(self, recipe_name, desired_width, desired_height):
        """
        Method to generate visually appealing images using Stable Diffusion model

        This method reads recipe name from the user and uses the Stable Diffusion
        model from PlaygroundAI to generate visually appealing image of the input
        recipe. This model uses CUDA, for GPU computation of generative algorithm.

        Read more in :ref:`RecipeML: Generative AI using StableDiffusion & OpenAI`

        .. versionadded:: 1.1.0

        Parameters:
            [string] recipe_name: Recipe name, for which image is to be generated

        Returns:
            [string] file_name: Location where the generated image is to be saved
        """
        # Define a prompt to feed into StableDiffusion model for image generation
        prompt = f"Create a visually appealing recipe image featuring a beautifully plated dish of {recipe_name}. The composition should highlight the colors, textures, and presentation of the dish. Pay special attention to lighting and styling to make the dish look as enticing as possible"
        image = self.pipe(prompt=prompt, guidance_scale=3.0).images[0]

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
            + "_playgroundai.png"
        )

        image.save(file_name)
        return file_name  # Save the generated image and then return the filename

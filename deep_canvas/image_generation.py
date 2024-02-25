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
This module defines the GenerativeImageSynthesis class facilitating the synthesis 
of images using various generative models such as OpenAI DALL·E2, StableDiffusion, 
and PlaygroundAI. StableDiffusion models require GPU compute for image generation.

Depending on individual cases, the program may be modified to use multiprocessing 
capailities to increase the speed of data processing, on eligible local computers.
The usage of each class & their methods are described in corresponding docstrings.

Classes and Functions:
    [1] GenerativeImageSynthesis (class)
        [a] generate_image

.. versionadded:: 1.3.0

Learn about RecipeML :ref:`RecipeML v1: Recipe Image Generation via Generative AI`
"""
import re
import random
import time
import torch
import pandas as pd
from PIL import Image
import streamlit as st

try:
    from deep_canvas.scripts.open_ai_models import DALLE2
    from deep_canvas.scripts.stable_diffusion import RunwayML, PlaygroundAI
except:
    from scripts.open_ai_models import DALLE2
    from scripts.stable_diffusion import RunwayML, PlaygroundAI

from configurations.api_authtoken import AuthTokens


class GenerativeImageSynthesis:
    """
    Class for Generative Image Synthesis using StableDiffusion and OpenAI DALLE.2

    This class encapsulates functionality for generating images, using generative
    models providing flexibility in terms of image quality & GPU acceleration. It
    integrates StableDiffusion models from PlaygroundAI and RunwayML for high and
    standard image quality respectively, leveraging GPU acceleration when enabled.

    Class Methods:
        [1] generate_image

    .. versionadded:: 1.3.0

    The performance of the methods present in the class can be optimized by using
    the CPUPool via the multithreading capailities on eligible local/cloud system.
    """

    def __init__(self, image_quality="standard", enable_gpu_acceleration=False):
        """
        Initialize class with specified image quality & GPU acceleration settings.
        Read more in :ref:`RecipeML: Generative AI using StableDiffusion & OpenAI`

        .. versionadded:: 1.3.0

        Parameters:
            [str] image_quality: Quality level of generated images (1024/512/256)
            [bool] enable_gpu_acceleration: Indicating, GPU acceleration settings

        NOTE: Keep track of the API usage here: platform.openai.com/account/usage
        """
        self.enable_gpu_acceleration = enable_gpu_acceleration
        self.image_quality = image_quality

        try:
            # Initialize image models based on image quality and GPU acceleration
            if (
                image_quality == "high"
                and enable_gpu_acceleration
                and torch.cuda.is_available()  # Check if GPU nodes are available
            ):
                self.playgroundai = PlaygroundAI()  # Initialize the PlaygroundAI
            if (
                (image_quality == "standard" or image_quality == "low")
                and enable_gpu_acceleration
                and torch.cuda.is_available()  # Check if GPU nodes are available
            ):
                self.runwayml = RunwayML()  # Initialize RunwayML ImageGeneration
        except:
            pass

        auth_token = AuthTokens()  # Authenticate and initialize the DALLE2 model
        self.dalle2 = DALLE2(auth_token.openai_api_key)

    def generate_image(self, payload, width, height):
        """
        Generate recipe image using various different generative AI methodologies

        The method generates recipe image using various image generation services
        based on the specified image quality. The available options include: high,
        standard & low quality. If generation using a particular stable diffusion
        service fails, it falls back to the alternative OpenAI DALLE2 image model.

        .. versionadded:: 1.3.0

        Parameters:
            [str] payload: The Recipe payload, or input data for image generation
            [int] width: Desired width of the generated recipe image (per prompt)
            [int] height: Desired height of generated food images (as per prompt)

        Returns:
            [str] image_path: Return the file path of the generated image or None

        NOTE: Keep track of the API usage here: platform.openai.com/account/usage
        """
        image_path = None

        if self.image_quality == "high":
            try:
                # Generate high-quality image using PlaygroundAI stable diffusion
                image_path = self.playgroundai.generate_recipe_image(
                    payload, width, height
                )
            except:
                try:
                    # If generation fails, try using DALL.E2 for image generation
                    image_path = self.dalle2.generate_recipe_image(
                        payload, width, height, "high"
                    )
                except:
                    image_path = None  # If both model fail, set the path to None

        elif self.image_quality == "standard":
            try:
                # Generate standard-quality image using RunwayML stable diffusion
                image_path = self.runwayml.generate_recipe_image(payload, width, height)
            except:
                try:
                    # If generation fails, try using DALL.E2 for image generation
                    image_path = self.dalle2.generate_recipe_image(
                        payload, width, height, "standard"
                    )
                except Exception as err:
                    image_path = None  # If both model fail, set the path to None
                    #st.sidebar.exception(err)
                    alert_image_generation_failed = st.sidebar.warning("⚠️ We are experiencing high traffic! Image generation models have been disabled temporarily.")
                    time.sleep(2)
                    alert_image_generation_failed.empty()                    

        else:
            try:
                # Generate a low-quality image, using RunwayML's stable diffusion
                image_path = self.runwayml.generate_recipe_image(payload, width, height)
            except:
                try:
                    # If generation fails, try using DALL.E2 for image generation
                    image_path = self.dalle2.generate_recipe_image(
                        payload, width, height, "low"
                    )
                except:
                    image_path = None  # If both model fail, set the path to None

        return image_path  # Return the file path of the generated image, or None

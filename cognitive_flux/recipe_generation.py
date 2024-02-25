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
This module defines the ProceduralTextGeneration class (ProGen), that facilitates 
the generation of procedural text, specifically recipes, using the combination of 
stochastic models, with maximum token length constraints and PaLM API integration.

Depending on individual cases, the program may be modified to use multiprocessing 
capailities to increase the speed of data processing, on eligible local computers.
The usage of each class & their methods are described in corresponding docstrings.

Classes and Functions:
    [1] ProceduralTextGeneration (class)
        [a] generate_recipe

.. versionadded:: 1.3.0

Learn about RecipeML :ref:`RecipeML v1: Recipe Generation - by name & ingredients`
"""
import numpy as np
import json
import tensorflow as tf
import matplotlib.pyplot as plt

import platform
import time
import pathlib
import os
import streamlit as st

import tensorflow
from tensorflow.keras.models import load_model

try:
    from cognitive_flux.scripts.lstm_recipe_generation import (
        LSTMTextSynthesizer,
        LSTMStyleTransfer,
    )
    from cognitive_flux.scripts.palm_recipe_generation import (
        PaLMLanguageModel,
        PaLMStyleTransfer,
        PaLMPromptModule,
    )
except:
    from scripts.lstm_recipe_generation import LSTMTextSynthesizer, LSTMStyleTransfer
    from scripts.palm_recipe_generation import (
        PaLMLanguageModel,
        PaLMStyleTransfer,
        PaLMPromptModule,
    )

from configurations.resource_path import ResourceRegistry


class ProceduralTextGeneration:
    """
    Wrapper class for interacting with Google PaLM API for recipe text generation
    Class for Procedural Text Generation using the LSTM RNN and Google's PaLM API.

    The class facilitates recipe text generation through two different approaches:
    1. Generation based on a single start ingredient - using the LSTM RNN network
    2. Name based generation, using PaLM for contextually relevant text synthesis

    Class Methods:
        [1] generate_recipe

    .. versionadded:: 1.3.0

    The performance of the methods present in the class can be optimized by using
    the CPUPool via the multithreading capailities on eligible local/cloud system.
    """

    def __init__(self, stochasticity=0.5, max_token_length=1000, palm_api_key=None):
        """
        Initialize Recipe Generation class with authtokens and configure the APIs

        This method serves as the constructor for the Recipe Generation model. It
        allows for customization of key parameters governing the model's behavior.

        Parameters:
            [int] stochasticity: Level of stochasticity in the generation process
            [int] max_token_length: Max length of generated tokens, or characters
            [str] palm_api_key: API key for Google pathways Language model (PaLM)
        """
        self.stochasticity = stochasticity
        self.max_token_length = max_token_length
        self.palm_api_key = palm_api_key

    def generate_recipe(self, user_input_query, generate_recipe_by_name=True):
        """
        Method to generate a recipe based on user input using advanced NLP models

        This method leverages both LSTM RNNs & PaLM API to generate recipes based
        on user input. The generation can be controlled either by recipe name, or
        a set of ingredients as per the valie of generate_recipe_by_name argument

        Read more in the :ref:`RecipeML:DataWrangling & Fundamental PreProcessing`

        .. versionadded:: 1.3.0

        Parameters:
            [str] user_input_query : The user input for generating a given recipe
            [bool] generate_recipe_by_name: Indicates whether to generate by name

        Returns:
            [tuple] Tuple containing recipe_name, recipe_type, recipe_ingredients,
            the recipe_instructions, the recipe_preperation_time & the recipe_url
        """
        resource_registry = ResourceRegistry()  # Initialize the ResourceRegistry

        if generate_recipe_by_name is False:
            # Load the LSTM RNN model & the TF/IDF tokenizer from the saved files
            with open(resource_registry.rnn_vocabulary_path) as f:
                data = json.load(f)
                tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(data)

            # Process the user's input, generate recipe, and apply style transfer
            model_1_simplified = load_model(
                resource_registry.multi_layer_lstm_model_path
            )

            ingredients = user_input_query  # Set the input query, as ingredients

            lstm_text_synthesizer = LSTMTextSynthesizer()
            generated_recipe = lstm_text_synthesizer.generate_combinations(
                model_1_simplified,
                tokenizer,
                ingredients,
                self.stochasticity,
                self.max_token_length,
            )

            # Attempt multiple LSTM RNN iterations for improved recipe generation
            lstm_rnn_attempt = 1
            lstm_style_transfer = LSTMStyleTransfer()

            while lstm_rnn_attempt < 5:
                if lstm_style_transfer.validate_lstm_result(generated_recipe):
                    lstm_rnn_attempt = lstm_rnn_attempt + 1
                    generated_recipe = lstm_text_synthesizer.generate_combinations(
                        model_1_simplified,
                        tokenizer,
                        ingredients,
                        self.stochasticity,
                        self.max_token_length,
                    )  # Generate recipe using the LSTM RNNs, with defined length

                else: break

            # Extract title, ingredients & instruction from LSTM-generated recipe
            recipe_title = generated_recipe[
                generated_recipe.index("📗") + 1 : generated_recipe.index("🥕")
            ].strip()
            recipe_title = lstm_style_transfer.process_recipe_name(recipe_title)

            recipe_ingredients = generated_recipe[
                generated_recipe.index("🥕") + 1 : generated_recipe.index("📝")
            ].strip()

            # Pre-process the ingredients, and select only the unique ingredients
            recipe_ingredients = list(
                set(lstm_style_transfer.process_recipe_ingredients(recipe_ingredients))
            )

            try:
                recipe_instructions = generated_recipe[
                    generated_recipe.index("📝") + 1 : generated_recipe.index("␣␣␣␣␣")
                ].strip()

            except:
                recipe_instructions = generated_recipe[
                    generated_recipe.index("📝") + 1 :
                ].strip()

            # Pre-process the instructions and select only the unique ingredients
            try:
                recipe_instructions = lstm_style_transfer.process_recipe_instructions(
                    recipe_instructions
                )
            except: pass

            # Initialize the PaLM module components for recipe details generation
            palm_prompt_module = PaLMPromptModule()
            palm_language_model = PaLMLanguageModel(self.palm_api_key)
            palm_style_transfer = PaLMStyleTransfer()

            # Generate prompts & use PaLM for paraphrasing and details generation
            recipe_preperation_time_calories_and_serving_size_prompt = palm_prompt_module.generate_recipe_preperation_time_and_serving_size_prompt(
                recipe_title
            )
            recipe_preperation_time_calories_and_serving_size = (
                palm_language_model.generate_text(
                    recipe_preperation_time_calories_and_serving_size_prompt
                )
            )
            (
                preperation_time_in_mins,
                serving_size,
                calories_in_recipe,
            ) = palm_style_transfer.paraphrase_preperation_time_and_serving_size(
                recipe_preperation_time_calories_and_serving_size
            )

        else:
            palm_prompt_module = PaLMPromptModule()  # Generates prompts for PaLM
            generate_recipe_prompt = palm_prompt_module.generate_recipe_by_name_prompt(
                user_input_query
            )

            is_paraphrase_success = False  # Flag to check the paraphrasing status

            try:
                # Attempt to generate the recipe till the successful paraphrasing
                while is_paraphrase_success is False:
                    palm_language_model = PaLMLanguageModel(
                        self.palm_api_key
                    )  # True, 0.7, 1500
                    generated_recipe = palm_language_model.generate_text(
                        generate_recipe_prompt,
                        self.stochasticity,
                        self.max_token_length,
                    )

                    palm_style_transfer = PaLMStyleTransfer()  # Initialize class
                    (
                        is_paraphrase_success,
                        recipe_list,
                    ) = palm_style_transfer.paraphrase_generated_recipe(
                        generated_recipe
                    )

                # Extract details from paraphrased recipe incl title, ingreds etc
                recipe_title = recipe_list[0]
                recipe_ingredients = recipe_list[1]

                recipe_instructions = recipe_list[2]
                recipe_instructions = palm_style_transfer.process_recipe_instructions(
                    recipe_instructions
                )

            except Exception as error:
                # Handle any PaLM driven exceptions, by providing a default value
                recipe_title = "Lorem Ipsum Dolor Mit"
                recipe_ingredients = 'Unavailable'
                recipe_instructions = 'Unavailable'

            try:
                preperation_time_in_mins = recipe_list[3]  # Fetch the time in minute
                serving_size = recipe_list[4]  # Fetch the PaLM generated, serve size
                calories_in_recipe = recipe_list[5]
            except:
                preperation_time_in_mins = '45'  # Fetch the time in minute
                serving_size = '3'
                calories_in_recipe = '255'

        try:
            # Generate description prompt and use PaLM for description generation
            recipe_description_prompt = (
                palm_prompt_module.generate_recipe_description_prompt(recipe_title)
            )
            recipe_description = palm_language_model.generate_text(
                recipe_description_prompt, 0.5, 200
            )

            # Paraphrase the recipe description and prepend it to the default str
            recipe_description = palm_style_transfer.paraphrase_for_description(
                recipe_description
            )
            recipe_description = (
                recipe_description
                + "<br><br>"
                + "Feeling adventurous? Experiment with different spices and herbs to create your own unique flavor profile. No matter how you choose to prepare it, this masterpiece is sure to become a goto favorite in your kitchen"
            )

        except Exception as error:
            # Handle exception for description by providing a default description
            recipe_description = (
                f"This wonderful dish is a delightful blend of flavors and textures. The carefully selected ingredients come together to create a truly satisfying and memorable experience. Each bite is bursting with flavor, and the aroma is simply divine. Whether you are a seasoned cook, or just starting out, this recipe is sure to impress"
                + "<br><br>"
                + "Feeling adventurous? Experiment with different spices and herbs to create your own unique flavor profile. No matter how you choose to prepare it, this masterpiece is sure to become a goto favorite in your kitchen"
            )

        return (
            recipe_title,
            recipe_ingredients,
            recipe_instructions,
            preperation_time_in_mins,
            serving_size,
            recipe_description,
            calories_in_recipe
        )

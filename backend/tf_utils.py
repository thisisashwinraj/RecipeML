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
This module contains the classes for data wrangling and fundamental preprocessing.
Methods with in each class are designed to clean and manipulate data for improved 
analysis with tensorflow. Classes have been initialized without passing parameter.

Depending on individual cases, the program may be modified to use multiprocessing 
capailities to increase the speed of data processing, on eligible local computers.
The usage of each class & their methods are described in corresponding docstrings.

Classes and Functions:
    [1] DataPreprocessing (class)
        [a] load_dataset
        [b] validate_recipes

    [2] DataTransformation (class)
        [a] recipe_to_string
        [b] filter_recipes_by_length

.. versionadded:: 1.3.0
.. versionupdated:: 1.3.0

Learn about RecipeML :ref:`RecipeML v1: DataWrangling & Fundamental PreProcessing`
"""
import os
import json
import platform
import time
import pathlib

import numpy as np
import pandas as pd
import tensorflow as tf

from configurations.resource_path import ResourceRegistry


class DataPreprocessing:
    """
    Class to perform preprocessing operation to preprocess the RecipeBowl dataset.

    This class provides methods for cleaning & preprocessing data. It is designed
    to handle various data preprocessing tasks such as loading the recipe dataset,
    validating the recipes, removing duplicate records, & checking for key fields.

    Class Methods:
        [1] load_dataset
        [2] validate_recipes

    .. versionadded:: 1.1.0

    The performance of the methods present in the class may be optimized by using
    the CPUPool via the multithreading capailities on eligible local/cloud system.
    """

    def __init__(self):
        pass

    def load_dataset(self, dataset_file_names=None):
        """
        Method to load raw recipebowl json dataset & convert to python dictionary.

        The method loads raw dataset from json file and stores it in a dictionary.
        The performance of this method can be further optimized, by making use of
        the CPUPool via multithreading capailities on eligible local/cloud system.

        Read more in the :ref:`RecipeML:DataWrangling & Fundamental PreProcessing`

        .. versionadded:: 1.3.0

        Parameters:
            [string] input_string: Space-separated str, to be evaluated to string

        Returns:
            [list] output_list: The list of string evaluated from an input string
        """
        # Create instance of ResourceRegistry class, to manage dataset file paths
        resource_registry = ResourceRegistry()

        # Use default dataset file names if filenames are not provided explicitly
        if dataset_file_names is None:
            dataset_file_names = [
                "recipes_raw_nosource_ar.json",
                "recipes_raw_nosource_epi.json",
                "recipes_raw_nosource_fn.json",
            ]

        dataset = []  # Initialize an empty py list to store the combined dataset

        for dataset_file_name in dataset_file_names:
            # Construct full file path for every dataset, using resource registry
            dataset_file_path = (
                resource_registry.raw_recipebowl_dataset_dir + dataset_file_name
            )

            # Open each dataset file and load the recipe data into the empty list
            with open(dataset_file_path) as dataset_file:
                json_data_dict = json.load(dataset_file)
                json_data_list = list(json_data_dict.values())

                # Extract & sort the dict keys from the first item in the dataset
                dict_keys = [key for key in json_data_list[0]]
                dict_keys.sort()

                dataset += json_data_list  # Extend the dataset with loaded data

        return dataset  # Return the combined data set as a single python object

    def validate_recipes(self, recipe):
        """
        Method to validate the recipe dictionary to check for required key values.

        The method checks whether the input dictionary has required keys: 'title',
        'ingredients' & 'instructions'. It also ensure that the values associated
        with these keys are non-empty lists so as to be considered as valid entry.

        Read more in the :ref:`RecipeML:DataWrangling & Fundamental PreProcessing`

        .. versionadded:: 1.3.0

        Parameters:
            [dict] recipe: Dictionary representing recipes with the required keys

        Returns:
            [list] output_list: True if the recipes are valid and False otherwise
        """
        required_keys = ["title", "ingredients", "instructions"]  # Required keys

        # Check if the input recipe dictionary isnt empty. Return Flase, if empty
        if not recipe:
            return False

        # Iterate through the required keys and check if key exists in the recipe
        for required_key in required_keys:
            if not recipe[required_key]:
                return False

            # Check if the required key exist in the recipe & has non-empty value
            if type(recipe[required_key]) == list and len(recipe[required_key]) == 0:
                return False

        return True  # If all condition pass return True, indicating valid recipe


class DataTransformation:
    """
    Class to preprocess corpus data including recipe instructions and ingredients

    This class provides methods for cleaning and transforming the corpus data. It
    is designed to handle various data preprocessing task such as converting dict
    to strings, removing noize, and stopwords and filtering the recipes by length.

    Class Methods:
        [1] recipe_to_string
        [2] filter_recipes_by_length

    .. versionadded:: 1.0.0
    .. versionupdated:: 1.1.0

    The performance of the methods present in the class can be optimized by using
    the CPUPool via the multithreading capailities on eligible local/cloud system.
    """
    def __init__(self):
        pass

    def recipe_to_string(
        self,
        recipe,
        STOP_WORD_TITLE="📗 ",
        STOP_WORD_INGREDIENTS="\n🥕\n\n",
        STOP_WORD_INSTRUCTIONS="\n📝\n\n",
    ):
        """
        Method to convert a processed recipe dictionary, to a formatted py string.

        The method takes a recipe dictionary, extracts its title, ingredients and
        instructions and formats them into a string. It removes noise string from
        ingredients & instructions, allowing for a more structured representation.

        Read more in the :ref:`RecipeML:DataWrangling & Fundamental PreProcessing`

        .. versionadded:: 1.3.0

        Parameters:
            [dict] recipe: Dictionary representing recipes with the required keys
            [str] STOP_WORD_TITLE: Stop word for marking a title in recipe output
            [str] STOP_WORD_INGREDIENTS: Stop word for marking recipe ingredients
            [str] STOP_WORD_INSTRUCTIONS: Stopword for marking recipe instruction

        Returns:
            [str] formatted_string: Formated string representing generated recipe
        """
        noize_string = "ADVERTISEMENT"  # Define noise to remove from the recipes

        # Extract title, ingredients, and instructions from the recipe dictionary
        title = recipe["title"]
        ingredients = recipe["ingredients"]
        instructions = recipe["instructions"].split("\n")

        # Initialize empty strings for recipe ingredients and recipe instructions
        ingredients_string, instructions_string = ""

        # Process the recipe ingredients, and remove the noise from the input str
        for ingredient in ingredients:
            ingredient = ingredient.replace(noize_string, "")

            if ingredient:
                ingredients_string += f"• {ingredient}\n"

        # Process the recipe instruction, and remove the noise from the input str
        for instruction in instructions:
            instruction = instruction.replace(noize_string, "")

            if instruction:
                instructions_string += f"▪︎ {instruction}\n"

        # Combine formatted title, ingredients and instructions into a single str
        return f"{STOP_WORD_TITLE}{title}\n{STOP_WORD_INGREDIENTS}{ingredients_string}{STOP_WORD_INSTRUCTIONS}{instructions_string}"

    def filter_recipes_by_length(self, recipe, MAX_RECIPE_LENGTH=2000):
        """
        Method to convert a processed recipe dictionary, to a formatted py string.
        Method to filter recipes based on their length.

        This method filters the recipes by length allowing only those with length
        less than or equal to specified maximum length. Default maximum length of
        the sequence is set to 2000 characters. Use histograms to analyze lengths.

        Read more in the :ref:`RecipeML:DataWrangling & Fundamental PreProcessing`

        .. versionadded:: 1.3.0

        Parameters:
            [dict] recipe: Dictionary representing recipes with the required keys
            [str] MAX_RECIPE_LENGTH: Recipes max allowed length. Defaults to 2000

        Returns:
            [str] formatted_string: Formated string representing generated recipe
            [bool] True if the recipe length is within the limit, False otherwise
        """
        return len(recipe) <= MAX_RECIPE_LENGTH  # Check if recipe is within limit

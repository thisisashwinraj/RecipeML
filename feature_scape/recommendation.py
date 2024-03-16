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
This module defines the FeatureSpaceMatching class, which implements an algorithm 
for recipe recommendation based on feature space matching. The algorithm leverage 
LocalAffinityPropagation to generate the TF/IDF embeddings, & the build knn model.

Depending on individual cases, the program may be modified to use multiprocessing 
capailities to increase the speed of data processing, on eligible local computers.
The usage of each class & their methods are described in corresponding docstrings.

Classes and Functions:
    [1] FeatureSpaceMatching (class)
        [a] initialize_feature_space_matching_algorithm
        [b] generate_recipe_recommendations
        [c] lookup_recipe_details_by_index

.. versionadded:: 1.3.0

Learn about RecipeML :ref:`RecipeML v1: Conditional recommendation and Embeddings`
"""
import sys
import logging
import datetime
import pandas as pd

import os
import ast
import json
import time
import joblib
import random
import requests
import streamlit as st
from PIL import Image
from io import BytesIO

import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

try:
    from feature_scape.scripts.feature_space_matching import LocalAffinityPropagation
    from feature_scape.scripts.palm2_language_model import (
        PaLMLanguageModel,
        PaLMPromptModule,
    )
except:
    from scripts.feature_space_matching import LocalAffinityPropagation
    from scripts.palm2_language_model import PaLMLanguageModel, PaLMPromptModule

from configurations.api_authtoken import AuthTokens
from configurations.resource_path import ResourceRegistry


class FeatureSpaceMatching:
    """
    Class to generate the recommendations, using feature space matching algorithm.

    This class provides methods for generating embeddings for recipe descriptions
    and building the recommendation model for recipe recommendations. It provides
    methods to generate TF/IDFvectorizer & train feature space matching algorithm.

    Class Methods:
        [1] initialize_feature_space_matching_algorithm
        [2] generate_recipe_recommendations
        [3] lookup_recipe_details_by_index

    .. versionadded:: 1.1.0

    The performance of the methods present in the class can be optimized by using
    the CPUPool via the multithreading capailities on eligible local/cloud system.
    """

    def __init__(self):
        pass

    def initialize_feature_space_matching_algorithm(self, processed_dataset):
        """
        Method to build TF/IDF embedding & train feature space matching algorithm

        This method loads the pre-processed data, generates the TF/IDF embeddings
        using sklearn.feature_extraction.text.TfidfVectorizer, and trains feature
        space matching algortithm to generate recommendation using brute strategy.

        Read more in :ref:`RecipeML:Conditional RecipeRecommendation & Embeddings`

        .. versionadded:: 1.1.0

        Parameters:
            [str] processed_dataset_path: The path to processed datasets location

        Returns:
            [tuple] The TF/IDF vectorizer, & the KNN feature space matching model
        """
        local_affinity_propagation = LocalAffinityPropagation()  # Load LAP class

        # Generate the TF/IDF vectorizer and the KNN Feature Space matching model
        (
            tfidf_vectorizer,
            model,
        ) = local_affinity_propagation.generate_tf_idf_embeddings_and_build_model(
            data=processed_dataset,
            subset="Corpus",
            n_neighbors=11,
            metric="cosine",
            algorithm="brute",
            n_jobs=-1,
        )
        return tfidf_vectorizer, model  # Return the trained model and vectorizer

    def generate_recipe_recommendations(
        self, input_ingredients, model, tfidf_vectorizer
    ):
        """
        Method to generate the recipe recommendations, based on input ingredients

        The method takes a list of ingredients and a recommendation model trained
        on recipe data. It uses the vectorizer to transform the input ingredients
        into a feature vector & then identifies the nearest neighbors in the data
        based on the cosine similarity. Only first 6 recommendations are returned.

        Read more in :ref:`RecipeML:Conditional RecipeRecommendation & Embeddings`

        .. versionadded:: 1.1.0

        Parameters:
            [string] input_ingredients: A list of ingredients for recommendations
            [string] model: Model trained using feature space matching algorithms
            [string] tfidf_vectorizer: TF/IDF emeddings fit over the Corpus field

        Returns:
            [list] recommended_recipes_indices: Indices of the recommended recipe
        """
        ingredients_text = " ".join(input_ingredients).lower()

        # Convert the input ingredients to TF/IDF vector & find nearest neighbors
        tfidf_vector = tfidf_vectorizer.transform([ingredients_text])
        _, indices = model.kneighbors(tfidf_vector)

        recommended_indices = indices[0][1:]  # Fetch the indices of data records
        recommended_recipes_indices = list(recommended_indices)

        return recommended_recipes_indices[:6]  # Return first six recommendation

    def lookup_recipe_details_by_index(self, recipe_data, index, use_large_model=False):
        """
        Method to retrieve details of recipes by index, from the loaded DataFrame

        The method looks up the details of a recipe based on its index within the
        DataFrame containing recipe data. It retrieves information such as recipe
        name, ingredients, instructions, preparation time, recipe type, & the URL.

        Read more in :ref:`RecipeML:Conditional RecipeRecommendation & Embeddings`

        .. versionadded:: 1.1.0

        Parameters:
            [pandas.DataFrame] recipe_data : The DataFrame containing recipe data
            [int] index: Index of recipe in the DataFrame to retrieve details for

        Returns:
            [tuple] Tuple containing recipe_name, recipe_type, recipe_ingredients,
            the recipe_instructions, the recipe_preperation_time & the recipe_url
        """
        if use_large_model:
            # Extract recipe details from a pandas DataFrame using provided index
            recipe_name = recipe_data["recipe_name"][index]

            recipe_ingredients = recipe_data["recipe_ingredients"][index]
            recipe_instructions = recipe_data["recipe_instructions"][index]

            recipe_url = recipe_data["recipe_url"][index]  # Retrieve recipes URL

        else:
            # Extract recipe details from a pandas DataFrame using provided index
            recipe_name = recipe_data["Recipe"].iloc[index]

            recipe_ingredients = recipe_data["Raw_Ingredients"].iloc[index]
            recipe_instructions = recipe_data["Instructions"].iloc[index]

            recipe_url = recipe_data["URL"].iloc[index]  # Retrieve URL of recipe

        try:
            # Attempt to use PaLM for additional details: preparation time & type
            auth_token = AuthTokens()
            palm_prompt = PaLMPromptModule()
            palm_language_model = PaLMLanguageModel(auth_token.palm_api_key)

            # Generate time and size using PaLM, and extract relevant information
            recipe_preperation_time_and_serving_size = ast.literal_eval(
                palm_language_model.generate_text(
                    palm_prompt.generate_recipe_preperation_time_prompt(recipe_name),
                    randomness=0.7,
                    max_response_length=100,
                )
                .replace("`", "")
                .replace("python", "")
            )

            # Fetch the recipe_preperation_time & the recipe_type from the result
            preperation_time = recipe_preperation_time_and_serving_size[0]
            recipe_calories = recipe_preperation_time_and_serving_size[1]

            recipe_preperation_time = [preperation_time, recipe_calories]
            recipe_type = recipe_preperation_time_and_serving_size[2]

        except:
            # Simple fallback mechanism in case of any PaLM API related exception
            recipe_type = (
                recipe_data["recipe_source"][index]
                if use_large_model
                else recipe_data["Source"].iloc[index]
            )
            recipe_preperation_time = random.randint(15, 90)

        return (
            recipe_name,
            recipe_type,
            recipe_ingredients,
            recipe_instructions,
            recipe_preperation_time,
            recipe_url,
        )  # Return the lookedup and palm generated details of recomended recipes

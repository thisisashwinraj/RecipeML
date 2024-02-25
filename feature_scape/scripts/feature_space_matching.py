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
This module defines, LocalAffinityPropagation, that facilitates the generation of 
TF/IDF embeddings, & the construction of a feature space matching model using the 
k-Nearest Neighbors. The vectorizer, and the model are maintained as pickle files.

Depending on individual cases, the program may be modified to use multiprocessing 
capailities to increase the speed of data processing, on eligible local computers.
The usage of each class & their methods are described in corresponding docstrings.

Classes and Functions:
    [1] LocalAffinityPropagation (class)
        [a] generate_tf_idf_embeddings_and_build_model
        [b] preprocess_raw_recipe_dataset

.. versionadded:: 1.3.0

Learn about RecipeML :ref:`RecipeML v1: Conditional Recommendation and Embeddings`
"""
import sys
import logging
import datetime
import pandas as pd

import joblib
import requests
import streamlit
from PIL import Image
from io import BytesIO
import pickle

import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

try:
    from feature_scape.scripts.knn_preprocessing_utils import CorpusData, DataWrangling
except:
    from scripts.knn_preprocessing_utils import CorpusData, DataWrangling

from configurations.api_authtoken import AuthTokens
from configurations.resource_path import ResourceRegistry

nltk.download("punkt")
nltk.download("stopwords")


class LocalAffinityPropagation:
    """
    Class for conditional recipe recommendation and raw RecipeNLG data processing

    The class is designed to generate the TF/IDF embeddings and build a model for
    feature space matching. It utilizes the Nearest Neighbors algorithm, based on
    the embeddings to find the local affinity clusters in data for recommendation.

    Class Methods:
        [1] generate_tf_idf_embeddings_and_build_model
        [2] preprocess_raw_recipe_dataset

    .. versionadded:: 1.3.0

    The performance of the methods present in the class can be optimized by using
    the CPUPool via the multithreading capailities on eligible local/cloud system.
    """

    def __init__(self):
        pass

    def generate_tf_idf_embeddings_and_build_model(
        self,
        data,
        subset="Corpus",
        n_neighbors=11,
        metric="cosine",
        algorithm="brute",
        n_jobs=-1,
    ):
        """
        Method to generate TF/IDF embeddings & build feature space matching model

        This method handles the creation and loading of TF-IDF vectorizer, TF/IDF
        matrix & the feature space matching model. If the models do not exist, it
        creates and saves them, based on the specified subset of the dataset. The
        resulting TF-IDF vectorizer and feature space matching model are returned.

        .. versionadded:: 1.3.0

        Parameters:
            [str] subset: Record to fit the TF-IDF vectorizer and train the model
            [int] n_neighbors: Neighbors to be considered for feature space model
            [int] metric: The distance metric for the Nearest Neighbors algorithm
            [str] algorithm: Aalgorithm used for computing, the nearest neighbors
            [str] n_jobs: Number of parallel job to run for the Nearest Neighbors

        Returns:
            [tuple] The TF/IDF vectorizer, & the KNN feature space matching model
        """
        resource_registry = ResourceRegistry()
        try:
            # Attempt to load the pretrained TF/IDF vectorizer from resource path
            tfidf_vectorizer = joblib.load(resource_registry.knn_tfidf_vectorizer)

        except Exception as tfidf_vectorizer_exception:
            # Create a TF-IDF vectorizer & fit it to the specified subset of data
            tfidf_vectorizer = TfidfVectorizer(stop_words="english")

            # Save the binary dump of the TF/IDF vectorizer, in the pickle format
            with open(resource_registry.knn_tfidf_vectorizer, "wb") as file:
                pickle.dump(tfidf_vectorizer, file)

        try:
            # Attempt to load the pretrained TF/IDF matrix from the resource path
            tfidf_matrix = joblib.load(resource_registry.knn_tfidf_matrix)

        except Exception as tfidf_matrix_exception:
            # If loading fails, generate the TF/IDF embeddings to build the model
            tfidf_matrix = tfidf_vectorizer.fit_transform(data[subset])

        try:
            # Attempt to load pre-trained neighbors model, from the resource path
            model = joblib.load(resource_registry.feature_space_matching_model)

        except Exception as feature_space_exception:
            # Train feature space matching algortithm using the TF/IDF embeddings
            model = NearestNeighbors(
                n_neighbors=n_neighbors,
                metric=metric,
                algorithm=algorithm,
                n_jobs=n_jobs,
            )
            model.fit(tfidf_matrix)  # Fit the TF/IDF embeddings to the KNN model

            with open(resource_registry.feature_space_matching_model, "wb") as file:
                pickle.dump(model, file)

        return tfidf_vectorizer, model  # Return the trained model and vectorizer

    def preprocess_raw_recipe_dataset(self, recipe_data):
        """
        Method to pre-process the raw recipe datasets, for feature space matching.

        This method performs various data pre-processing operations on raw recipe
        dataset. It involves removing duplicate records converting data types and,
        applying data wrangling operations. Final preprocesed dataset is returned.

        .. versionadded:: 1.3.0

        Parameters:
            [pandas.DataFrame] recipe_data: Raw recipe dataset to be preprocessed

        Returns:
            [pandas.DataFrame] recipe_data: The recipe dataframe after processing
        """
        # Initialize necessary data processing class from knn_preprocessing_utils
        corpus_data = CorpusData()
        data_wrangling = DataWrangling()

        # Remove the duplicate records from the dataset based on the title column
        duplicate_records_count = len(
            recipe_data[recipe_data.duplicated(subset="title", keep="first")]
        )

        if duplicate_records_count > 0:
            recipe_data = data_wrangling.remove_duplicate_records(recipe_data)

        # Convert data in NER and cleaned_directions to list d-type for wrangling
        recipe_data["NER"] = recipe_data["NER"].apply(
            corpus_data.convert_string_to_list
        )

        recipe_data["cleaned_directions"] = recipe_data["directions"].apply(
            corpus_data.convert_string_to_list
        )

        # Apply data preprocessing operations to NER and cleaned_directions field
        recipe_data["NER"] = recipe_data["NER"].apply(
            data_wrangling.remove_whitespace_and_duplicates
        )
        recipe_data["cleaned_directions"] = recipe_data["directions"].apply(
            data_wrangling.remove_whitespace_and_duplicates
        )

        # Drop the unnecessary columns from the dataset to reduce data complexity
        recipe_data.drop("Unnamed: 0", axis=1, inplace=True)

        # Convert the processed data in NER & cleaned_directions to string d-type
        recipe_data["NER"] = recipe_data["NER"].apply(
            corpus_data.convert_list_to_string
        )
        recipe_data["cleaned_directions"] = recipe_data["cleaned_directions"].apply(
            corpus_data.convert_list_to_string
        )

        # Rename the necessary fields in processed dataset for easier recognition
        recipe_data.rename(columns={"title": "Recipe"}, inplace=True)
        recipe_data.rename(columns={"NER": "Ingredients"}, inplace=True)
        recipe_data.rename(columns={"source": "Source"}, inplace=True)
        recipe_data.rename(columns={"link": "URL"}, inplace=True)

        recipe_data.rename(columns={"directions": "Instructions"}, inplace=True)
        recipe_data.rename(columns={"ingredients": "Raw_Ingredients"}, inplace=True)
        recipe_data.rename(
            columns={"cleaned_directions": "Cleaned_Instructions"}, inplace=True
        )

        # Check for null values in the dataset and remove records with null value
        null_value_count = sum(recipe_data.isna().sum())

        if null_value_count > 0:
            recipe_data.dropna(inplace=True)

        # Create Corpus field by combining the Ingredients and Instructions field
        recipe_data["Corpus"] = (
            recipe_data["Ingredients"] + " " + recipe_data["Instructions"]
        )
        recipe_data["Corpus"] = recipe_data["Corpus"].str.lower()

        # Drop the unnecessary columns from the dataset to reduce data complexity
        recipe_data.drop("Ingredients", inplace=True, axis=1)
        recipe_data.drop("Cleaned_Instructions", inplace=True, axis=1)

        # Lemmatize the corpus and remove stop words, whitespace and punctuations
        recipe_data["Corpus"] = recipe_data["Corpus"].apply(
            corpus_data.lemmatize_and_remove_stop_words
        )
        recipe_data["Corpus"] = recipe_data["Corpus"].apply(
            data_wrangling.remove_punctuations_and_whitespaces
        )

        return recipe_data  # Return the cleaned recipe dataset for preprocessing

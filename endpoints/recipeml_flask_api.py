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
The module defines a Flask-based API for coniditional recipe recommendation based 
on input ingredients. It preprocesses raw data, performs TF/IDF vectorization and 
utilizes the feature space matching algorithm, to generate recipe recommendations.

The API exposes the /recommend endpoint for transfering POST requests to generate 
recipe recommendation based on a list of ingredients provided as a JSON dump. The 
API returns the recipe id, name, ingredients, instructions & source to the client.

Module Functions:
    [1] recommend_recipes_using_ingredients()

API Endpoints:
    [1] /recommend [POST]: recommend_recipe()

.. versionadded:: 1.1.0

Learn about RecipeML :ref:`RecipeML v1: RecipeML Flask API Functionality Overview`
"""
import re
import time
import joblib

import random
import nltk
import numpy as np
import pandas as pd

from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from backend import config
from backend.utils import CorpusData, DataWrangling

from inference.neural_engine import RecipeRecommendation

app = Flask(__name__)  # Initialize an elementary Flask web applications instance

try:
    # Attempt to load the pre-processed dataset from the data/processed directory
    processed_dataset_path = "data/processed/recipe_nlg_processed_data.csv"
    recipe_data = pd.read_csv(processed_dataset_path)

except:
    # If the pre-processed dataset doesn't exist in the loc, load the raw dataset
    raw_dataset_path = "data/raw/recipe_nlg_raw_dataset.csv"
    recipe_data = pd.read_csv(raw_dataset_path)

    # Initialize the necessary data processing classes from backend/data_utils.py
    corpus_data = CorpusData()
    data_wrangling = DataWrangling()

    # Remove all the duplicate records from the dataset based on the title column
    duplicate_records_count = len(
        recipe_data[recipe_data.duplicated(subset="title", keep="first")]
    )

    if duplicate_records_count > 0:
        recipe_data = data_wrangling.remove_duplicate_records(recipe_data)

    # Convert the data in NER and cleaned_directions to list d-type for wrangling
    recipe_data["NER"] = recipe_data["NER"].apply(
        corpus_data.convert_string_to_list
    )

    recipe_data["cleaned_directions"] = recipe_data["directions"].apply(
        corpus_data.convert_string_to_list
    )

    # Apply the data preprocessing operations to NER and cleaned_directions field
    recipe_data["NER"] = recipe_data["NER"].apply(
        data_wrangling.remove_whitespace_and_duplicates
    )
    recipe_data["cleaned_directions"] = recipe_data["directions"].apply(
        data_wrangling.remove_whitespace_and_duplicates
    )

    # Drop the unnecessary columns from the dataset to reduce datasets complexity
    recipe_data.drop("Unnamed: 0", axis=1, inplace=True)

    # Convert the processed data in NER & cleaned_directions to the string d-type
    recipe_data["NER"] = recipe_data["NER"].apply(
        corpus_data.convert_list_to_string
    )
    recipe_data["cleaned_directions"] = recipe_data["cleaned_directions"].apply(
        corpus_data.convert_list_to_string
    )

    # Rename the necessary fields in the processed dataset for easier recognition
    recipe_data.rename(columns={"title": "Recipe"}, inplace=True)
    recipe_data.rename(columns={"NER": "Ingredients"}, inplace=True)
    recipe_data.rename(columns={"source": "Source"}, inplace=True)
    recipe_data.rename(columns={"link": "URL"}, inplace=True)

    recipe_data.rename(
        columns={"directions": "Instructions"}, inplace=True)
    recipe_data.rename(
        columns={"ingredients": "Raw_Ingredients"}, inplace=True)
    recipe_data.rename(
        columns={"cleaned_directions": "Cleaned_Instructions"}, inplace=True
    )

    # Check for null values in the dataset and remove all records with null value
    null_value_count = sum(recipe_data.isna().sum())

    if null_value_count > 0:
        recipe_data.dropna(inplace=True)

    # Create the Corpus field by combining the Ingredients and Instructions field
    recipe_data["Corpus"] = (
        recipe_data["Ingredients"] + " " + recipe_data["Instructions"]
    )
    recipe_data["Corpus"] = recipe_data["Corpus"].str.lower()

    # Drop the unnecessary columns from the dataset to reduce datasets complexity
    recipe_data.drop("Ingredients", inplace=True, axis=1)
    recipe_data.drop("Cleaned_Instructions", inplace=True, axis=1)

    # Lemmatize the corpus and remove all stop words, whitespace and punctuations
    recipe_data["Corpus"] = recipe_data["Corpus"].apply(
        corpus_data.lemmatize_and_remove_stop_words
    )
    recipe_data["Corpus"] = recipe_data["Corpus"].apply(
        data_wrangling.remove_punctuations_and_whitespaces
    )

try:
    # Attempt to load the pretrained TF/IDF vectorizer and model from ~/inference
    tfidf_vectorizer = joblib.load(
        "inference/embeddings/tfidf_vectorizer_recipe_nlg.pkl"
    )
    model = joblib.load("inference/models/neural_engine_recipe_nlg.pkl")

except:
    recipe_recommendation = RecipeRecommendation()

    try:
        # If the loading fails, generate the TF/IDF embeddings & build the models
        (
            tfidf_vectorizer,
            model,
        ) = recipe_recommendation.generate_tf_idf_embeddings_and_build_model(
            recipe_data, "Corpus"
        )

        # Save the binary dump of the TF/IDF vectorizer and the model as pkl file
        with open(
            "inference/embeddings/tfidf_vectorizer_recipe_nlg.pkl", "wb"
        ) as file:
            pickle.dump(tfidf_vectorizer, file)

        with open("inference/models/neural_engine_recipe_nlg.pkl", "wb") as file:
            pickle.dump(model, file)

    except:
        pass  # Ignore the exceptions in case of failing to save the binary files


@app.route('/')
def api_home():
    '''
    This API endpoint acts as the initial layer of interaction with the flask api.

    This method defines the home endpoint of the API accessible through a browser.
    It returns a simple message to indicate that you've reached the API home page.

    Parameters:
        None -> This is a static method & doesn't require any parameter from user

    Returns:
        [string] message: Message indicating users to be interacting with the API
    '''
    return "API Home"


@app.route('/recommend', methods=['POST'])
def recommend_recipe():
    '''
    POST a list of ingredient to this API endpoint to receive recommended recipes.

    This method defines an endpoint for recommending recipes based on ingredients.
    It expects a POST request, with a JSON array containing a list of ingredients.
    It processes the request, & returns a response containing recommended recipes.

    Parameters:
        None -> A JSON array containing the list of ingredients received via POST

    Returns:
        None -> Response containing recommended recipes, converted to JSON format
    '''
    recipe_recommendation = RecipeRecommendation()

    try:
        data = request.get_json()  # Get the input ingredients, from JSON request

        # Ensure data is a list, else return error code 403, requesting list data
        if not isinstance(data, list):
            return jsonify({'error': 'Data should be a list of ingredients'})

        # Generate recipe recommendations using feature space matching algorithms
        recommendations = recipe_recommendation.generate_recipe_recommendations(
            data, model, tfidf_vectorizer)
        recipe_id_list = [int(recipe_id) for recipe_id in recommendations]

        recipe_name_list = []
        recipe_ingredients_list = []
        recipe_instructions_list = []
        recipe_source_list = []

        # Store the intrinsic details of the recommended recipes in seprate lists
        for i in recipe_id_list:
            recipe_name_list.append(recipe_data['Recipe'].iloc[i])
            recipe_ingredients_list.append(
                recipe_data['Raw_Ingredients'].iloc[i])
            recipe_instructions_list.append(
                recipe_data['Instructions'].iloc[i])
            recipe_source_list.append(recipe_data['URL'].iloc[i])

        # Convert the response data to a dictionary to send it back to the client
        response_data = {
            'recipe_id': recipe_id_list,
            'recipe_name': recipe_name_list,
            'recipe_ingredients': recipe_ingredients_list,
            'recipe_instructions': recipe_instructions_list,
            'recipe_source': recipe_source_list
        }

        # Convert the response data to JSON format and return the recommendations
        return jsonify(response_data)

    except Exception as e:
        # When exception is caught, return error status with details of exception
        return jsonify({'SERVER ERROR': str(e)})


if __name__ == '__main__':
    app.run(debug=True)  # Run the Flask app on the server with debug mode active

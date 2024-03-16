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
.. versionupdated:: 1.3.0

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


app = Flask(__name__)  # Initialize an elementary Flask web applications instance


@app.route("/")
def api_home():
    """
    This API endpoint acts as the initial layer of interaction with the flask api.

    This method defines the home endpoint of the API accessible through a browser.
    It returns a simple message to indicate that you've reached the API home page.

    Parameters:
        None -> This is a static method & doesn't require any parameter from user

    Returns:
        [string] message: Message indicating users to be interacting with the API
    """
    return "Welcome to RecipeML"


@app.route("/recommend", methods=["POST"])
def recommend_recipe():
    """
    POST a list of ingredient to this API endpoint to receive recommended recipes.

    This method defines an endpoint for recommending recipes based on ingredients.
    It expects a POST request, with a JSON array containing a list of ingredients.
    It processes the request, & returns a response containing recommended recipes.

    Parameters:
        None -> A JSON array containing the list of ingredients received via POST

    Returns:
        None -> Response containing recommended recipes, converted to JSON format
    """
    tfidf_vectorizer = joblib.load("tfidf_vectorizer_recipe_nlg.pkl")
    model = joblib.load("neural_engine_recipe_nlg.pkl")

    try:
        ingredients = request.get_json()  # Fetch ingredients from ~/http request

        # Ensure data is a list, else return error code 403, requesting list data
        if not isinstance(ingredients, list):
            return jsonify({"error": "Data should be a list of ingredients"})

        ingredients_text = " ".join(ingredients).lower()

        # Generate recipe recommendations using feature space matching algorithms
        tfidf_vector = tfidf_vectorizer.transform([ingredients_text])
        _, indices = model.kneighbors(tfidf_vector)

        recommended_indices = indices[0][1:]
        recommended_recipes_indices = list(recommended_indices)

        recipe_id_list = [int(recipe_id) for recipe_id in recommended_recipes_indices]

        # Load the processed dataset, to infer the details pertaining to a recipe
        recipe_data = pd.read_csv("recipe_nlg_processed_data.csv")
        recipe_data.dropna(inplace=True)

        recipe_name_list = []
        recipe_ingredients_list = []
        recipe_instructions_list = []
        recipe_url_list = []
        recipe_source_list = []

        # Store the intrinsic details of the recommended recipes in seprate lists
        for i in recipe_id_list:
            recipe_name_list.append(recipe_data["Recipe"].iloc[i])
            recipe_ingredients_list.append(recipe_data["Raw_Ingredients"].iloc[i])
            recipe_instructions_list.append(recipe_data["Instructions"].iloc[i])
            recipe_url_list.append(recipe_data["URL"].iloc[i])
            recipe_source_list.append(recipe_data["Source"].iloc[i])

        # Convert the response data to a dictionary to send it back to the client
        response_data = {
            "recipe_id": recipe_id_list,
            "recipe_name": recipe_name_list,
            "recipe_ingredients": recipe_ingredients_list,
            "recipe_instructions": recipe_instructions_list,
            "recipe_url": recipe_url_list,
            "recipe_source": recipe_source_list,
        }

        # Convert the response data to JSON format and return the recommendations
        return jsonify(response_data)

    except Exception as e:
        # When exception is caught, return error status with details of exception
        return jsonify({"SERVER ERROR": str(e)})


if __name__ == "__main__":
    app.run(debug=True)  # Run the Flask app on the server with debug mode active

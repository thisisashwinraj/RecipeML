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
This module interacts with RecipeML's flask API locally for generating the recipe 
recommendations. This sends a sample list of ingredients to the flask API running 
on your local machine and retrieves the recipe recommendation using the algorithm.

Response from the API is then processed to extract the recipe's details including 
its ID, name, ingredients, instructions, & source. If the response status code is 
200, it prints these details. In case of any client errors, it displays the error.

NOTE: RecipeMLs flask API shall be running at the specified URL on the local host.

.. versionadded:: 1.1.0

Learn about RecipeML :ref:`RecipeML v1: RecipeML Flask API Functionality Overview`
"""
import json
import requests

# Sample list of ingredients to e provided as input for the recipe recommendation
input_ingredients = ["chicken", "rice", "vegetables"]

# The URL of local RecipeML Flask API (add the /recommend endpoint for inference)
recipeml_flask_api_local_url = "http://127.0.0.1:5000/recommend"

# Send a client POST request to the API running locally with the input ingredient
response = requests.post(recipeml_flask_api_local_url, json=input_ingredients)
print("The response is: " + str(response))

# Check response's status code to ensure a successful response else display error
if response.status_code == 200:
    # Extract and print the first recommended recipe's details from JSON response
    recipe_id = response.json()["recipe_id"][0]
    print(recipe_id)

    recipe_name = response.json()["recipe_name"][0]
    print(recipe_name)

    recipe_ingredients = response.json()["recipe_ingredients"][0]
    print(recipe_ingredients)

    recipe_instructions = response.json()["recipe_instructions"][0]
    print(recipe_instructions)

    recipe_source = response.json()["recipe_source"][0]
    print(recipe_source)

else:
    # Handle the client error internally and display the error message on console
    print("CLIENT ERROR:", response.json())

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
Module for managing authentication tokens and text generation using OpenAI & PaLM.

This module defines the ResourceRegistry class, serving as centralized repository
for file paths & resources used in the recommendation, and generation application. 

Classes and Functions:
    [1] ResourceRegistry (class)

.. versionadded:: 1.3.0

Learn about RecipeML :ref:`RecipeML: Auth Tokens and Streamlit Secrets Management`
"""
import os

class ResourceRegistry:  # These paths are used when running your web app locally
    ingredients_list_path = "data/pickle/recipe_nlg_ingredients.pkl"
    raw_recipenlg_dataset_path = "data/raw/recipe_nlg_raw_dataset.csv"
    processed_recipenlg_dataset_path = "data/processed/recipe_nlg_processed_data.csv"
    raw_recipebowl_dataset_dir = "data/raw/"

    knn_tfidf_vectorizer = 'embeddings/tfidf_vectorizer_recipe_nlg.pkl'
    knn_tfidf_matrix = 'embeddings/tfidf_matrix_recipe_nlg.pkl'
    feature_space_matching_model = '/content/drive/MyDrive/RecipeML/neural_engine_recipe_nlg.pkl'

    generated_images_directory_path = "exports/generated_img/"
    placeholder_image_dir_path = "assets/images/placeholder/"
    generated_recipe_pdf_dir_path = "exports/generated_pdf/"
    loading_assets_dir = "assets/loading/"

    rnn_vocabulary_path = 'cognitive_flux/embeddings/charecter_level_rnn_vocabulary.json'
    multi_layer_lstm_model_path = 'cognitive_flux/model/multi_layer_lstm_network.h5'
    raw_recipebox_dataset_dir_path = '/content/drive/MyDrive/Homemade Recipe/'

    def __init__(self, execution_platform='colab'):
        pass


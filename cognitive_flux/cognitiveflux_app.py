import streamlit as st
import pandas as pd
from PIL import Image
import re
import random

from recipe_generation import ProceduralTextGeneration
from configurations.api_authtoken import AuthTokens


def resize_image(image_path, new_width, new_height):
    image = Image.open(image_path)

    left = int((512 - new_width) / 2)
    top = int((512 - new_height) / 2)

    cropped_image = image.crop((left, top, left + new_width, top + new_height))
    return cropped_image.resize((new_width, new_height))


if __name__ == "__main__":
    auth_token = AuthTokens()
    recipe_generation_type = st.sidebar.selectbox("Select Recipe Generation Technique", ("Generate by Recipe Name", "Generate by Ingredients"))

    if recipe_generation_type == "Generate by Ingredients":
        ingredients_list = ['Chicken', 'Egg', 'Wheat', 'Bread']

        selected_ingredients = st.sidebar.selectbox(
            "Select your favourite ingredient",
            ingredients_list,
            index=None,
            placeholder="Pick from over 10,000+ ingredients"
        )

        if selected_ingredients:
            procedural_text_generation = ProceduralTextGeneration(stochasticity=0.3, max_token_length=1500, palm_api_key=auth_token.palm_api_key)

            recipe_title, recipe_ingredients, recipe_instructions, preperation_time_in_mins, serving_size, recipe_description = procedural_text_generation.generate_recipe(selected_ingredients, generate_recipe_by_name=False)
            flag_display_result = True
        else:
            flag_display_result = False

    elif recipe_generation_type == "Generate by Recipe Name":
        recipe_name = st.sidebar.text_input("Enter Recipe Name:", placeholder="Be playful, descriptive, or even a little bit poetic")

        if recipe_name:
            procedural_text_generation = ProceduralTextGeneration(stochasticity=0.7, max_token_length=1500, palm_api_key=auth_token.palm_api_key)

            recipe_title, recipe_ingredients, recipe_instructions, preperation_time_in_mins, serving_size, recipe_description = procedural_text_generation.generate_recipe(recipe_name, generate_recipe_by_name=True)
            flag_display_result = True
        else:
            flag_display_result = False

    else:
        flag_display_result = False
        st.error("An unexpected error occured")

    if flag_display_result:
        st.markdown(
            f"<H2>{recipe_title}</H2>", unsafe_allow_html=True
        )

        column_1, column_2 = st.columns([1,3.55])

        with column_1:
            st.markdown(f"<h5>🍜 Serving for {serving_size}</h5><br>", unsafe_allow_html=True)
        with column_2:
            st.markdown(f"<h5>🕓 Requires {preperation_time_in_mins} mins to prepare (approx)</h5><br>", unsafe_allow_html=True)

        primary_image, secondary_image = st.columns([1.48, 1])

        primary_image_path = "placeholder_1.png"
        secondary_image_path = "placeholder_2.png"

        recipe_image_424x322 = resize_image(primary_image_path, 424, 322)
        recipe_image_284x322 = resize_image(secondary_image_path, 284, 322)

        with primary_image:
            st.image(recipe_image_424x322)
        with secondary_image:
            st.image(recipe_image_284x322)

        st.markdown(f"<p align='justify'>{recipe_description}</p>", unsafe_allow_html=True)

        st.markdown(
            "<H3>Ingredients</H3>", unsafe_allow_html=True
        )

        recipe_ingredients = ", ".join(recipe_ingredients)
        st.markdown(
            f"<p align='justify'>{recipe_ingredients}</p>", unsafe_allow_html=True
        )

        st.info("Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut laborei et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi utto aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cilliut")

        st.markdown(
            "<H3>Recipe Directions</H3>", unsafe_allow_html=True
        )
        st.markdown(
            f"<p align='justify'>{recipe_instructions}</p>", unsafe_allow_html=True
        )

    else:
        st.info('This is a placeholder column')
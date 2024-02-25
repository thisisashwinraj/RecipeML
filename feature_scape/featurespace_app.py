import re
import time
import base64
import random
import pickle
import joblib

import nltk
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO
import streamlit as st

from recommendation import FeatureSpaceMatching

from configurations.api_authtoken import AuthTokens
from configurations.resource_path import ResourceRegistry


# Set the page title and favicon to be displayed on the streamlit web application
st.set_page_config(
    page_title="RecipeML"
)

# Remove the extra paddings from the top and bottom margin of the block container
st.markdown(
    """
        <style>
               .block-container {
                    padding-top: 0.5rem;
					padding-bottom: 0rem;
                }
        </style>
        """,
    unsafe_allow_html=True,
)

# Hide the streamlit menu & the default footer from the production app's frontend
HIDE_MENU_STYLE = """
<style>
#MainMenu  {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(HIDE_MENU_STYLE, unsafe_allow_html=True)

# Remove the default styling from the hyperlinks displayed on the web application
st.markdown(
    """
    <style>
    .stMarkdown a {
        text-decoration: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def set_recommend_recipes_button_state(desired_session_state):
    """
    Function to set the state of the recommend recipe button in app session state.

    The function update the state of the recommend recipe button in streamlit app 
    session state. This enables or disables the button based on users interaction.

    Read more in the :ref:`RecipeML v1: User Interface and Functionality Overview`

    .. versionadded:: 1.1.0

    Parameters:
        [bool] current_status: The desired state for the recommend recipes button

    Returns:
        None -> Applies the user's desired state for the recommend recipes button
    """
    st.session_state.recommend_recipes_button_state = desired_session_state


def apply_style_to_sidebar_button(file_name):
    """
    Function to apply CSS-3 style specified in the arg file to the sidebar button.

    The function takes a CSS file as a parameter and applies the customized style 
    to all the button widgets over the sidebars of the web application's frontend.

    Read more in the :ref:`RecipeML v1: User Interface and Functionality Overview`

    .. versionadded:: 1.0.0

    Parameters:
        [css file] file_name: CSS file holding style to be applied on the buttons

    Returns:
        None -> Applies the style specified in the CSS file to all sidebar button
    """
    with open(file_name, encoding="utf-8") as file:
        st.markdown(f"<style>{file.read()}</style>", unsafe_allow_html=True)

try:
    # Read the CSS code from the css file & allow html parsing to apply the style
    apply_style_to_sidebar_button("assets/css/login_sidebar_button_style.css")
except:
    pass  # Use default style in case file is not found or if an exception happen


if __name__ == "__main__":
    feature_space_matching = FeatureSpaceMatching()
    resource_registry = ResourceRegistry()
    ingredients_list, tfidf_vectorizer, model = feature_space_matching.initialize_feature_space_matching_algorithm(resource_registry.processed_recipenlg_dataset_path)

    # Initialize state variables if they don't already exist in the app's session
    if "user_authentication_status" not in st.session_state:
        st.session_state.user_authentication_status = None

    if "authenticated_user_email_id" not in st.session_state:
        st.session_state.authenticated_user_email_id = "rajashwin733@gmail.com"

    try:
        # Read CSS code from the css file & allow html parsing to apply the style
        apply_style_to_sidebar_button("assets/css/login_home_button_style.css")
    except:
        pass  # Use default style if file is not found or if an exception happens

    # Create a multi-select widget in the sidebar for selecting input ingredients
    selected_ingredients = st.sidebar.multiselect(
        "Select the ingredients",
        ingredients_list,
        on_change=set_recommend_recipes_button_state(False),
    )
    input_ingredients = [ingredient.lower()
                         for ingredient in selected_ingredients]

    # Initialize/update the session state variable for the recommendations button
    if "recommend_recipes_button_state" not in st.session_state:
        st.session_state.recommend_recipes_button_state = False

    recommend_recipes_button = st.sidebar.button(
        "Recommend Recipes", on_click=set_recommend_recipes_button_state(True)
    )

    # Check if ingredients have been selected & recommendations button is clicked
    if len(input_ingredients) > 0 and (
        recommend_recipes_button or st.session_state.recommend_recipes_button_state
    ):
        st.markdown(
            "<H2>Here are some recipes you can try</H2>", unsafe_allow_html=True
        )
        st.write(
            "These recommendations are generated using Recipe ML - one of our latest AI advancements. Our goal is to learn, improve & innovate responsibly on AI together. Check out the data security policy for our users here"
        )

        # Generate recipe recommendations using feature space matching algorithms
        recommended_recipes_indices = feature_space_matching.generate_recipe_recommendations(input_ingredients, model, tfidf_vectorizer)

        # Iterate over the recommended recipes, and display the necessary details
        st.markdown(
            """
            <style>
            .custom-hr {
                margin-top: -10px;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        # Create three columns to display recommendations on the app's layout
        container_1, container_2, container_3 = st.columns(3)

        button_width = 225  # Set the width of each col buttons to 256 pixels

        st.markdown(
            f"<style>.stButton button {{ width: {button_width}px; }}</style>",
            unsafe_allow_html=True,
        )

        recipe_data = pd.read_csv(resource_registry.processed_recipenlg_dataset_path)

        with container_1:
            # Fetch details of the recommended recipe from index location - 0
            recipe_name, recipe_type, recipe_ingredients, recipe_instructions, recipe_preperation_time, recipe_url = feature_space_matching.lookup_recipe_details_by_index(recipe_data, recommended_recipes_indices[0])

            image = Image.open("placeholder_1.png")
            recipe_image = image.resize((225, 225))

            st.image(recipe_image)

            # Shorten recipe name to 26 characters and add ellipsis if longer
            if len(recipe_name) <= 26:
                recipe_name = recipe_name
            else:
                recipe_name = recipe_name[:26] + "..."

            # Display the name of the recommended recipe as a HTML H6 heading
            st.markdown("<H6>" + recipe_name + "</H6>", unsafe_allow_html=True)

            # Display recipe details including source, URL & preparation time
            if recipe_preperation_time < 100:
                if recipe_type == "Gathered" or recipe_type == 'Recipes1M':
                    # Determine the type, based on the source of the recipe's details
                    if "Gathered":
                        recipe_type = recipe_type + " Recipe"
                    if "Recipes1M" in recipe_type:
                        recipe_type = "Recipes 1M Site"

                    st.markdown(
                        "<p style='font-size: 16px;'>Cuisine Source: <A HREF ="
                        + recipe_url
                        + ">"
                        + recipe_type
                        + "</A><BR>Takes around "
                        + str(recipe_preperation_time)
                        + " mins to prepare<BR>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        "<p style='font-size: 16px;'>"
                        + recipe_type
                        + "Cuisine<BR>Takes around "
                        + str(recipe_preperation_time)
                        + " mins to prepare<BR>",
                        unsafe_allow_html=True,
                    )
                        
            else:
                if recipe_type == "Gathered" or recipe_type == 'Recipes1M':
                    # Determine the type, based on the source of the recipe's details
                    if "Gathered":
                        recipe_type = recipe_type + " Recipe"
                    if "Recipes1M" in recipe_type:
                        recipe_type = "Recipes 1M Site"

                    st.markdown(
                        "<p style='font-size: 16px;'>Cuisine Source: <A HREF ="
                        + recipe_url
                        + ">"
                        + recipe_type
                        + "</A><BR>Takes over a "
                        + str(recipe_preperation_time)
                        + " mins to prepare<BR>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        "<p style='font-size: 16px;'>"
                        + recipe_type
                        + "Cuisine<BR>Takes over a "
                        + str(recipe_preperation_time)
                        + " mins to prepare<BR>",
                        unsafe_allow_html=True,
                    )

            # Display a download button for the unauthenticated app users
            st.button(
                label="Download Recipe Details PDF",
                key="download_button_0",
            )
            
            # Add HTML linebreak
            st.markdown("<BR>", unsafe_allow_html=True)

            # Fetch details of the recommended recipe from index location - 0
            recipe_name, recipe_type, recipe_ingredients, recipe_instructions, recipe_preperation_time, recipe_url = feature_space_matching.lookup_recipe_details_by_index(recipe_data, recommended_recipes_indices[3])

            image = Image.open("placeholder_1.png")
            recipe_image = image.resize((225, 225))

            st.image(recipe_image)

            # Shorten recipe name to 26 characters and add ellipsis if longer
            if len(recipe_name) <= 26:
                recipe_name = recipe_name
            else:
                recipe_name = recipe_name[:26] + "..."

            # Display the name of the recommended recipe as a HTML H6 heading
            st.markdown("<H6>" + recipe_name + "</H6>", unsafe_allow_html=True)

            # Display recipe details including source, URL & preparation time
            # Display recipe details including source, URL & preparation time
            if recipe_preperation_time < 100:
                if recipe_type == "Gathered" or recipe_type == 'Recipes1M':
                    # Determine the type, based on the source of the recipe's details
                    if "Gathered":
                        recipe_type = recipe_type + " Recipe"
                    if "Recipes1M" in recipe_type:
                        recipe_type = "Recipes 1M Site"

                    st.markdown(
                        "<p style='font-size: 16px;'>Cuisine Source: <A HREF ="
                        + recipe_url
                        + ">"
                        + recipe_type
                        + "</A><BR>Takes around "
                        + str(recipe_preperation_time)
                        + " mins to prepare<BR>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        "<p style='font-size: 16px;'>"
                        + recipe_type
                        + "Cuisine<BR>Takes around "
                        + str(recipe_preperation_time)
                        + " mins to prepare<BR>",
                        unsafe_allow_html=True,
                    )
                        
            else:
                if recipe_type == "Gathered" or recipe_type == 'Recipes1M':
                    # Determine the type, based on the source of the recipe's details
                    if "Gathered":
                        recipe_type = recipe_type + " Recipe"
                    if "Recipes1M" in recipe_type:
                        recipe_type = "Recipes 1M Site"

                    st.markdown(
                        "<p style='font-size: 16px;'>Cuisine Source: <A HREF ="
                        + recipe_url
                        + ">"
                        + recipe_type
                        + "</A><BR>Takes over a "
                        + str(recipe_preperation_time)
                        + " mins to prepare<BR>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        "<p style='font-size: 16px;'>"
                        + recipe_type
                        + "Cuisine<BR>Takes over a "
                        + str(recipe_preperation_time)
                        + " mins to prepare<BR>",
                        unsafe_allow_html=True,
                    )

            # Display a download button for the unauthenticated app users
            st.button(
                label="Download Recipe Details PDF",
                key="download_button_3",
            )

        with container_2:
            # Fetch details of the recommended recipe from index location - 0
            recipe_name, recipe_type, recipe_ingredients, recipe_instructions, recipe_preperation_time, recipe_url = feature_space_matching.lookup_recipe_details_by_index(recipe_data, recommended_recipes_indices[1])

            image = Image.open("placeholder_1.png")
            recipe_image = image.resize((225, 225))

            st.image(recipe_image)

            # Shorten recipe name to 26 characters and add ellipsis if longer
            if len(recipe_name) <= 26:
                recipe_name = recipe_name
            else:
                recipe_name = recipe_name[:26] + "..."

            # Display the name of the recommended recipe as a HTML H6 heading
            st.markdown("<H6>" + recipe_name + "</H6>", unsafe_allow_html=True)

            # Display recipe details including source, URL & preparation time
            if recipe_preperation_time < 100:
                if recipe_type == "Gathered" or recipe_type == 'Recipes1M':
                    # Determine the type, based on the source of the recipe's details
                    if "Gathered":
                        recipe_type = recipe_type + " Recipe"
                    if "Recipes1M" in recipe_type:
                        recipe_type = "Recipes 1M Site"

                    st.markdown(
                        "<p style='font-size: 16px;'>Cuisine Source: <A HREF ="
                        + recipe_url
                        + ">"
                        + recipe_type
                        + "</A><BR>Takes around "
                        + str(recipe_preperation_time)
                        + " mins to prepare<BR>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        "<p style='font-size: 16px;'>"
                        + recipe_type
                        + "Cuisine<BR>Takes around "
                        + str(recipe_preperation_time)
                        + " mins to prepare<BR>",
                        unsafe_allow_html=True,
                    )
                        
            else:
                if recipe_type == "Gathered" or recipe_type == 'Recipes1M':
                    # Determine the type, based on the source of the recipe's details
                    if "Gathered":
                        recipe_type = recipe_type + " Recipe"
                    if "Recipes1M" in recipe_type:
                        recipe_type = "Recipes 1M Site"

                    st.markdown(
                        "<p style='font-size: 16px;'>Cuisine Source: <A HREF ="
                        + recipe_url
                        + ">"
                        + recipe_type
                        + "</A><BR>Takes over a "
                        + str(recipe_preperation_time)
                        + " mins to prepare<BR>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        "<p style='font-size: 16px;'>"
                        + recipe_type
                        + "Cuisine<BR>Takes over a "
                        + str(recipe_preperation_time)
                        + " mins to prepare<BR>",
                        unsafe_allow_html=True,
                    )

            # Display a download button for the unauthenticated app users
            st.button(
                label="Download Recipe Details PDF",
                key="download_button_1",
            )
            
            # Add HTML linebreak
            st.markdown("<BR>", unsafe_allow_html=True)

            # Fetch details of the recommended recipe from index location - 0
            recipe_name, recipe_type, recipe_ingredients, recipe_instructions, recipe_preperation_time, recipe_url = feature_space_matching.lookup_recipe_details_by_index(recipe_data, recommended_recipes_indices[4])

            image = Image.open("placeholder_1.png")
            recipe_image = image.resize((225, 225))

            st.image(recipe_image)

            # Shorten recipe name to 26 characters and add ellipsis if longer
            if len(recipe_name) <= 26:
                recipe_name = recipe_name
            else:
                recipe_name = recipe_name[:26] + "..."

            # Display the name of the recommended recipe as a HTML H6 heading
            st.markdown("<H6>" + recipe_name + "</H6>", unsafe_allow_html=True)

            # Display recipe details including source, URL & preparation time
            # Display recipe details including source, URL & preparation time
            if recipe_preperation_time < 100:
                if recipe_type == "Gathered" or recipe_type == 'Recipes1M':
                    # Determine the type, based on the source of the recipe's details
                    if "Gathered":
                        recipe_type = recipe_type + " Recipe"
                    if "Recipes1M" in recipe_type:
                        recipe_type = "Recipes 1M Site"

                    st.markdown(
                        "<p style='font-size: 16px;'>Cuisine Source: <A HREF ="
                        + recipe_url
                        + ">"
                        + recipe_type
                        + "</A><BR>Takes around "
                        + str(recipe_preperation_time)
                        + " mins to prepare<BR>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        "<p style='font-size: 16px;'>"
                        + recipe_type
                        + "Cuisine<BR>Takes around "
                        + str(recipe_preperation_time)
                        + " mins to prepare<BR>",
                        unsafe_allow_html=True,
                    )
                        
            else:
                if recipe_type == "Gathered" or recipe_type == 'Recipes1M':
                    # Determine the type, based on the source of the recipe's details
                    if "Gathered":
                        recipe_type = recipe_type + " Recipe"
                    if "Recipes1M" in recipe_type:
                        recipe_type = "Recipes 1M Site"

                    st.markdown(
                        "<p style='font-size: 16px;'>Cuisine Source: <A HREF ="
                        + recipe_url
                        + ">"
                        + recipe_type
                        + "</A><BR>Takes over a "
                        + str(recipe_preperation_time)
                        + " mins to prepare<BR>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        "<p style='font-size: 16px;'>"
                        + recipe_type
                        + "Cuisine<BR>Takes over a "
                        + str(recipe_preperation_time)
                        + " mins to prepare<BR>",
                        unsafe_allow_html=True,
                    )

            # Display a download button for the unauthenticated app users
            st.button(
                label="Download Recipe Details PDF",
                key="download_button_4",
            )

        with container_3:
            # Fetch details of the recommended recipe from index location - 0
            recipe_name, recipe_type, recipe_ingredients, recipe_instructions, recipe_preperation_time, recipe_url = feature_space_matching.lookup_recipe_details_by_index(recipe_data, recommended_recipes_indices[2])

            image = Image.open("placeholder_1.png")
            recipe_image = image.resize((225, 225))

            st.image(recipe_image)

            # Shorten recipe name to 26 characters and add ellipsis if longer
            if len(recipe_name) <= 26:
                recipe_name = recipe_name
            else:
                recipe_name = recipe_name[:26] + "..."

            # Display the name of the recommended recipe as a HTML H6 heading
            st.markdown("<H6>" + recipe_name + "</H6>", unsafe_allow_html=True)

            # Display recipe details including source, URL & preparation time
            # Display recipe details including source, URL & preparation time
            if recipe_preperation_time < 100:
                if recipe_type == "Gathered" or recipe_type == 'Recipes1M':
                    # Determine the type, based on the source of the recipe's details
                    if "Gathered":
                        recipe_type = recipe_type + " Recipe"
                    if "Recipes1M" in recipe_type:
                        recipe_type = "Recipes 1M Site"

                    st.markdown(
                        "<p style='font-size: 16px;'>Cuisine Source: <A HREF ="
                        + recipe_url
                        + ">"
                        + recipe_type
                        + "</A><BR>Takes around "
                        + str(recipe_preperation_time)
                        + " mins to prepare<BR>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        "<p style='font-size: 16px;'>"
                        + recipe_type
                        + "Cuisine<BR>Takes around "
                        + str(recipe_preperation_time)
                        + " mins to prepare<BR>",
                        unsafe_allow_html=True,
                    )
                        
            else:
                if recipe_type == "Gathered" or recipe_type == 'Recipes1M':
                    # Determine the type, based on the source of the recipe's details
                    if "Gathered":
                        recipe_type = recipe_type + " Recipe"
                    if "Recipes1M" in recipe_type:
                        recipe_type = "Recipes 1M Site"

                    st.markdown(
                        "<p style='font-size: 16px;'>Cuisine Source: <A HREF ="
                        + recipe_url
                        + ">"
                        + recipe_type
                        + "</A><BR>Takes over a "
                        + str(recipe_preperation_time)
                        + " mins to prepare<BR>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        "<p style='font-size: 16px;'>"
                        + recipe_type
                        + "Cuisine<BR>Takes over a "
                        + str(recipe_preperation_time)
                        + " mins to prepare<BR>",
                        unsafe_allow_html=True,
                    )

            # Display a download button for the unauthenticated app users
            st.button(
                label="Download Recipe Details PDF",
                key="download_button_2",
            )
            
            # Add HTML linebreak
            st.markdown("<BR>", unsafe_allow_html=True)

            # Fetch details of the recommended recipe from index location - 0
            recipe_name, recipe_type, recipe_ingredients, recipe_instructions, recipe_preperation_time, recipe_url = feature_space_matching.lookup_recipe_details_by_index(recipe_data, recommended_recipes_indices[5])

            image = Image.open("placeholder_1.png")
            recipe_image = image.resize((225, 225))

            st.image(recipe_image)

            # Shorten recipe name to 26 characters and add ellipsis if longer
            if len(recipe_name) <= 26:
                recipe_name = recipe_name
            else:
                recipe_name = recipe_name[:26] + "..."

            # Display the name of the recommended recipe as a HTML H6 heading
            st.markdown("<H6>" + recipe_name + "</H6>", unsafe_allow_html=True)

            # Display recipe details including source, URL & preparation time
            if recipe_preperation_time < 100:
                if recipe_type == "Gathered" or recipe_type == 'Recipes1M':
                    # Determine the type, based on the source of the recipe's details
                    if "Gathered":
                        recipe_type = recipe_type + " Recipe"
                    if "Recipes1M" in recipe_type:
                        recipe_type = "Recipes 1M Site"

                    st.markdown(
                        "<p style='font-size: 16px;'>Cuisine Source: <A HREF ="
                        + recipe_url
                        + ">"
                        + recipe_type
                        + "</A><BR>Takes around "
                        + str(recipe_preperation_time)
                        + " mins to prepare<BR>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        "<p style='font-size: 16px;'>"
                        + recipe_type
                        + "Cuisine<BR>Takes around "
                        + str(recipe_preperation_time)
                        + " mins to prepare<BR>",
                        unsafe_allow_html=True,
                    )
                        
            else:
                if recipe_type == "Gathered" or recipe_type == 'Recipes1M':
                    # Determine the type, based on the source of the recipe's details
                    if "Gathered":
                        recipe_type = recipe_type + " Recipe"
                    if "Recipes1M" in recipe_type:
                        recipe_type = "Recipes 1M Site"

                    st.markdown(
                        "<p style='font-size: 16px;'>Cuisine Source: <A HREF ="
                        + recipe_url
                        + ">"
                        + recipe_type
                        + "</A><BR>Takes over a "
                        + str(recipe_preperation_time)
                        + " mins to prepare<BR>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        "<p style='font-size: 16px;'>"
                        + recipe_type
                        + "Cuisine<BR>Takes over a "
                        + str(recipe_preperation_time)
                        + " mins to prepare<BR>",
                        unsafe_allow_html=True,
                    )

            # Display a download button for the unauthenticated app users
            st.button(
                label="Download Recipe Details PDF",
                key="download_button_5",
            )
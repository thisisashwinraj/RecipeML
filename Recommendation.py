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

from deep_canvas.image_generation import GenerativeImageSynthesis
from feature_scape.recommendation import FeatureSpaceMatching

from backend.send_mail import MailUtils
from backend.generate_pdf import PDFUtils

from configurations.api_authtoken import AuthTokens
from configurations.resource_path import ResourceRegistry


# Set the page title and favicon to be displayed on the streamlit web application
st.set_page_config(
    page_title="RecipeML: Recipe Recommendation",
    page_icon="assets/images/favicon/recipeml_favicon.png",
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
st.markdown(
    """
    <style>
        #MainMenu  {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
    """, 
    unsafe_allow_html=True,
)

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


if "user_authentication_status" not in st.session_state:
    st.session_state.user_authentication_status = None

if "cache_generate_recommendations" not in st.session_state:
    st.session_state.cache_generate_recommendations = False


def set_generate_recommendations_cache_to_true():
    st.session_state.cache_generate_recommendations = True


def set_generate_recommendations_cache_to_false():
    st.session_state.cache_generate_recommendations = False


def apply_style_to_sidebar_button(css_file_name):
    with open(css_file_name, encoding="utf-8") as file:
        st.markdown(f"<style>{file.read()}</style>", unsafe_allow_html=True)


try:
    # Read the CSS code from the css file & allow html parsing to apply the style
    apply_style_to_sidebar_button("assets/css/login_sidebar_button_style.css")

except: pass  # Use the default style if file is'nt found or if exception happens


if __name__ == "__main__":
    resource_registry = ResourceRegistry(execution_platform="colab")
    feature_space_matching = FeatureSpaceMatching()
    genisys = GenerativeImageSynthesis(
        image_quality="low", enable_gpu_acceleration=False
    )

    # Fetch the preloader image from the assets directory, to be used in this app
    loading_image_path = resource_registry.loading_assets_dir + "loading_img.gif"

    with open(loading_image_path, "rb") as f:
        image_data = f.read()
        encoded_image = base64.b64encode(image_data).decode()

    # Load the processed list of ingredients from the binary dump to a global var
    try:
        with open(
            resource_registry.ingredients_list_path, "rb"
        ) as recipe_nlg_ingredients_list:
            ingredients_list = joblib.load(recipe_nlg_ingredients_list)
    except:
        ingredients_list = [
            "Bread",
            "Mushroom",
            "Butter",
            "Onion",
            "Cheese",
            "Tomato",
            "Orange",
        ]

    try:
        # Read CSS code from the css file & allow html parsing to apply the style
        apply_style_to_sidebar_button("assets/css/login_home_button_style.css")
    except: pass  # Use default style if file is'nt found or if exception happens

    # Create a multi-select widget in the sidebar for selecting input ingredients
    selected_ingredients = st.sidebar.multiselect(
        "Select the ingredients",
        ingredients_list,
        on_change=set_generate_recommendations_cache_to_false,
    )
    input_ingredients = [ingredient.lower() for ingredient in selected_ingredients]

    generate_recommendations_button = st.sidebar.button(
        "Recommend Recipes", on_click=set_generate_recommendations_cache_to_true, use_container_width=True,
    )

    # Check if ingredients have been selected & recommendations button is clicked
    if (
        generate_recommendations_button
        or st.session_state.cache_generate_recommendations
    ) and len(input_ingredients) > 0:

        # Display the preloader, as the application performs time-intensive tasks
        gif_image = st.markdown(
            f'<br><br><br><br><div class="rounded-image"><img src="data:image/png;base64,{encoded_image}"></div><br><br><br><br><br><br><br><br><br><br><br>',
            unsafe_allow_html=True,
        )

        with gif_image:
            # Initialize the instances of the PDFUtility & the MailUtility module
            pdf_utils = PDFUtils()
            mail_utils = MailUtils()

            dataset_path_1 = "data/processed/recipe_nlg_batch_datasets/recipeml_processed_data_split_1.csv"
            dataset_path_2 = "data/processed/recipe_nlg_batch_datasets/recipeml_processed_data_split_2.csv"
            dataset_path_3 = "data/processed/recipe_nlg_batch_datasets/recipeml_processed_data_split_3.csv"
            dataset_path_4 = "data/processed/recipe_nlg_batch_datasets/recipeml_processed_data_split_4.csv"
            dataset_path_5 = "data/processed/recipe_nlg_batch_datasets/recipeml_processed_data_split_5.csv"

            dataset1 = pd.read_csv(dataset_path_1)
            dataset2 = pd.read_csv(dataset_path_2)
            dataset3 = pd.read_csv(dataset_path_3)
            dataset4 = pd.read_csv(dataset_path_4)
            dataset5 = pd.read_csv(dataset_path_5)

            recipeml_processed_data = [dataset1, dataset2, dataset3, dataset4, dataset5]

            # Load the processed data into variable for generating the embeddings
            recipe_data = pd.concat(recipeml_processed_data, ignore_index=True)
            recipe_data.dropna(inplace=True)

            # Load the TF/IDF vectorizer and trained feature space matching model
            (
                tfidf_vectorizer,
                model,
            ) = feature_space_matching.initialize_feature_space_matching_algorithm(
                recipe_data
            )

            # Generate recipe recommendations, using feature space matching model
            recommended_recipes_indices = (
                feature_space_matching.generate_recipe_recommendations(
                    input_ingredients, model, tfidf_vectorizer
                )
            )

            # Iterate over recommended recipes, and display the necessary details
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

            gif_image.empty()  # Stop displaying preloader image on the front end

        st.markdown(
            "<H2>Here are some recipes you can try</H2>", unsafe_allow_html=True
        )
        st.markdown(
            "<P align='justify'>These recommendations are generated using Recipe ML - one of our latest AI advancements. Our goal is to learn, improve, & innovate responsibly on AI together. Check out the data security policy for our users <a href='https://recipeml-recommendations.streamlit.app/Discover_RecipeML#no-hidden-ingredients-here-recipeml-v1-3-privacy-policy' style='color: #64ABD8;'>here</A></P>", unsafe_allow_html=True
        )

        # Create three columns to display recommendations on the web app's layout
        container_1, container_2, container_3 = st.columns(3)

        button_width = 225  # Set the width of each column buttons, to 256 pixels

        st.markdown(
            f"<style>.stButton button {{ width: {button_width}px; }}</style>",
            unsafe_allow_html=True,
        )

        with container_1:
            # Fetch details of the recommended recipe from the index location - 0
            (
                recipe_name,
                recipe_type,
                recipe_ingredients,
                recipe_instructions,
                recipe_preperation_time,
                recipe_url,
            ) = feature_space_matching.lookup_recipe_details_by_index(
                recipe_data, recommended_recipes_indices[0]
            )

            # Attempt to generate image of the recipe, using Generative AI models
            generated_image_path = genisys.generate_image(recipe_name, 225, 225)

            if generated_image_path:
                recipe_image = Image.open(generated_image_path)
            else:
                # Display a placeholder image if the image could not be generated
                placeholder_image_path = (
                    resource_registry.placeholder_image_dir_path + "placeholder_1.png"
                )
                recipe_image = Image.open(placeholder_image_path).resize((225, 225))

            st.image(recipe_image)

            # Shorten recipe name to max 26 characters and add ellipsis if longer
            if len(recipe_name) <= 26:
                recipe_name = recipe_name
            else:
                recipe_name = recipe_name[:26] + "..."

            # Display the name of the recommended recipe, as an HTML <H6> heading
            st.markdown("<H6>" + recipe_name + "</H6>", unsafe_allow_html=True)

            # Display recipe details including the source, URL & preparation time
            if recipe_preperation_time < 100:
                if recipe_type == "Gathered" or recipe_type == "Recipes1M":
                    # Determine the type, based on the source of recipe's details
                    if "Gathered":
                        recipe_type = recipe_type + " Recipe"
                    if "Recipes1M" in recipe_type:
                        recipe_type = "Recipes 1M Site"

                    st.markdown(
                        "<p style='font-size: 16px;'>Cuisine Source: <a href ='https://"
                        + recipe_url
                        + "' style='color: #64ABD8;'>"
                        + recipe_type
                        + "</A><BR>Takes around "
                        + str(recipe_preperation_time)
                        + " mins to prepare<BR>",
                        unsafe_allow_html=True,
                    )
                else:
                    # Display the recipe type, & the approximate preperation time
                    st.markdown(
                        "<p style='font-size: 16px;'>"
                        + recipe_type
                        + "Cuisine<BR>Takes around "
                        + str(recipe_preperation_time)
                        + " mins to prepare<BR>",
                        unsafe_allow_html=True,
                    )

            else:
                if recipe_type == "Gathered" or recipe_type == "Recipes1M":
                    # Determine the type, based on the source of recipe's details
                    if "Gathered":
                        recipe_type = recipe_type + " Recipe"
                    if "Recipes1M" in recipe_type:
                        recipe_type = "Recipes 1M Site"

                    st.markdown(
                        "<p style='font-size: 16px;'>Cuisine Source: <a href ='https://"
                        + recipe_url
                        + "' style='color: #64ABD8;'>"
                        + recipe_type
                        + "</A><BR>Takes over a "
                        + str(recipe_preperation_time)
                        + " mins to prepare<BR>",
                        unsafe_allow_html=True,
                    )
                else:
                    # Display the recipe type, & the approximate preperation time
                    st.markdown(
                        "<p style='font-size: 16px;'>"
                        + recipe_type
                        + "Cuisine<BR>Takes over a "
                        + str(recipe_preperation_time)
                        + " mins to prepare<BR>",
                        unsafe_allow_html=True,
                    )

            if st.session_state.user_authentication_status is not True:
                # Generate PDF file with necessary recipe details and usage terms
                download_location_0 = pdf_utils.generate_recommendations_pdf(
                    recipe_name,
                    recipe_type,
                    recipe_url,
                    recipe_ingredients,
                    recipe_instructions,
                )

                # Display a download button only for the unauthenticated app user
                st.download_button(
                    label="Download Recipe Details PDF",
                    data=open(download_location_0, "rb").read(),
                    key="download_button_0",
                    file_name=recipe_name.replace(" ", "_").lower() + ".pdf",
                )

            else:
                # For authenticated user, send a mail with recipe details and PDF
                if st.button("Send Recipe to Mail", key="button_0", use_container_width=True):
                    try:
                        st.toast("Hold tight! Your recipe is taking flight.")

                        # Generate the PDF file with the necessary recipe details
                        download_location_0 = pdf_utils.generate_recommendations_pdf(
                            recipe_name,
                            recipe_type,
                            recipe_url,
                            recipe_ingredients,
                            recipe_instructions,
                        )
                        # Send the mail with attachment to the registered mail id
                        mail_utils.send_recipe_info_to_mail(
                            recipe_name,
                            recipe_ingredients,
                            recipe_instructions,
                            st.session_state.authenticated_user_email_id,
                            download_location_0,
                        )

                        # Display the information status upon successful delivery
                        st.toast("Bon appétit! We've delivered your recipe.")

                    except Exception as error:
                        # Display information status upon a unsuccessful delivery
                        st.toast("Whoops! Looks like your recipe ran into a snag.")
                        time.sleep(1)
                        st.toast("Please check your connectivity and try again.")

            st.markdown("<BR>", unsafe_allow_html=True)

            # Fetch details of the recommended recipe from the index location - 3
            (
                recipe_name,
                recipe_type,
                recipe_ingredients,
                recipe_instructions,
                recipe_preperation_time,
                recipe_url,
            ) = feature_space_matching.lookup_recipe_details_by_index(
                recipe_data, recommended_recipes_indices[3]
            )

            # Attempt to generate image of the recipe, using Generative AI models
            generated_image_path = genisys.generate_image(recipe_name, 225, 225)

            if generated_image_path:
                recipe_image = Image.open(generated_image_path)
            else:
                # Display a placeholder image if the image could not be generated
                placeholder_image_path = (
                    resource_registry.placeholder_image_dir_path + "placeholder_4.png"
                )
                recipe_image = Image.open(placeholder_image_path).resize((225, 225))

            st.image(recipe_image)

            # Shorten recipe name to max 26 characters and add ellipsis if longer
            if len(recipe_name) <= 26:
                recipe_name = recipe_name
            else:
                recipe_name = recipe_name[:26] + "..."

            # Display the name of the recommended recipe, as an HTML <H6> heading
            st.markdown("<H6>" + recipe_name + "</H6>", unsafe_allow_html=True)

            # Display recipe details including the source, URL & preparation time
            if recipe_preperation_time < 100:
                if recipe_type == "Gathered" or recipe_type == "Recipes1M":
                    # Determine the type, based on the source of recipe's details
                    if "Gathered":
                        recipe_type = recipe_type + " Recipe"
                    if "Recipes1M" in recipe_type:
                        recipe_type = "Recipes 1M Site"

                    st.markdown(
                        "<p style='font-size: 16px;'>Cuisine Source: <a href ='https://"
                        + recipe_url
                        + "' style='color: #64ABD8;'>"
                        + recipe_type
                        + "</A><BR>Takes around "
                        + str(recipe_preperation_time)
                        + " mins to prepare<BR>",
                        unsafe_allow_html=True,
                    )
                else:
                    # Display the recipe type, & the approximate preperation time
                    st.markdown(
                        "<p style='font-size: 16px;'>"
                        + recipe_type
                        + "Cuisine<BR>Takes around "
                        + str(recipe_preperation_time)
                        + " mins to prepare<BR>",
                        unsafe_allow_html=True,
                    )

            else:
                if recipe_type == "Gathered" or recipe_type == "Recipes1M":
                    # Determine type, based on the source of the recipe's details
                    if "Gathered":
                        recipe_type = recipe_type + " Recipe"
                    if "Recipes1M" in recipe_type:
                        recipe_type = "Recipes 1M Site"

                    st.markdown(
                        "<p style='font-size: 16px;'>Cuisine Source: <a href ='https://"
                        + recipe_url
                        + "' style='color: #64ABD8;'>"
                        + recipe_type
                        + "</A><BR>Takes over a "
                        + str(recipe_preperation_time)
                        + " mins to prepare<BR>",
                        unsafe_allow_html=True,
                    )
                else:
                    # Display the recipe type, & the approximate preperation time
                    st.markdown(
                        "<p style='font-size: 16px;'>"
                        + recipe_type
                        + "Cuisine<BR>Takes over a "
                        + str(recipe_preperation_time)
                        + " mins to prepare<BR>",
                        unsafe_allow_html=True,
                    )

            if st.session_state.user_authentication_status is not True:
                # Generate PDF file with necessary recipe details and usage terms
                download_location_3 = pdf_utils.generate_recommendations_pdf(
                    recipe_name,
                    recipe_type,
                    recipe_url,
                    recipe_ingredients,
                    recipe_instructions,
                )

                # Display a download button only for the unauthenticated app user
                st.download_button(
                    label="Download Recipe Details PDF",
                    data=open(download_location_3, "rb").read(),
                    key="download_location_3",
                    file_name=recipe_name.replace(" ", "_").lower() + ".pdf",
                )

            else:
                # For authenticated user, send a mail with recipe details and PDF
                if st.button("Send Recipe to Mail", key="button_3", use_container_width=True):
                    try:
                        st.toast("Hold tight! Your recipe is taking flight.")

                        # Generate the PDF file with the necessary recipe details
                        download_location_3 = pdf_utils.generate_recommendations_pdf(
                            recipe_name,
                            recipe_type,
                            recipe_url,
                            recipe_ingredients,
                            recipe_instructions,
                        )
                        # Send the mail with attachment to the registered mail id
                        mail_utils.send_recipe_info_to_mail(
                            recipe_name,
                            recipe_ingredients,
                            recipe_instructions,
                            st.session_state.authenticated_user_email_id,
                            download_location_3,
                        )

                        # Display the information status upon successful delivery
                        st.toast("Bon appétit! We've delivered your recipe.")

                    except Exception as error:
                        # Display the information status upon successful delivery
                        st.toast("Whoops! Looks like your recipe ran into a snag.")
                        time.sleep(1)
                        st.toast("Please check your connectivity and try again.")

        with container_2:
            # Fetch details of the recommended recipe from the index location - 1
            (
                recipe_name,
                recipe_type,
                recipe_ingredients,
                recipe_instructions,
                recipe_preperation_time,
                recipe_url,
            ) = feature_space_matching.lookup_recipe_details_by_index(
                recipe_data, recommended_recipes_indices[1]
            )

            # Attempt to generate image of the recipe, using Generative AI models
            generated_image_path = genisys.generate_image(recipe_name, 225, 225)

            if generated_image_path:
                recipe_image = Image.open(generated_image_path)
            else:
                # Display a placeholder image if the image could not be generated
                placeholder_image_path = (
                    resource_registry.placeholder_image_dir_path + "placeholder_2.png"
                )
                recipe_image = Image.open(placeholder_image_path).resize((225, 225))

            st.image(recipe_image)

            # Shorten recipe name to max 26 characters and add ellipsis if longer
            if len(recipe_name) <= 26:
                recipe_name = recipe_name
            else:
                recipe_name = recipe_name[:26] + "..."

            # Display the name of the recommended recipe, as an HTML <H6> heading
            st.markdown("<H6>" + recipe_name + "</H6>", unsafe_allow_html=True)

            # Display recipe details including the source, URL & preparation time
            if recipe_preperation_time < 100:
                if recipe_type == "Gathered" or recipe_type == "Recipes1M":
                    # Determine the type, based on the source of recipe's details
                    if "Gathered":
                        recipe_type = recipe_type + " Recipe"
                    if "Recipes1M" in recipe_type:
                        recipe_type = "Recipes 1M Site"

                    st.markdown(
                        "<p style='font-size: 16px;'>Cuisine Source: <a href ='https://"
                        + recipe_url
                        + "' style='color: #64ABD8;'>"
                        + recipe_type
                        + "</A><BR>Takes around "
                        + str(recipe_preperation_time)
                        + " mins to prepare<BR>",
                        unsafe_allow_html=True,
                    )
                else:
                    # Display the recipe type, & the approximate preperation time
                    st.markdown(
                        "<p style='font-size: 16px;'>"
                        + recipe_type
                        + "Cuisine<BR>Takes around "
                        + str(recipe_preperation_time)
                        + " mins to prepare<BR>",
                        unsafe_allow_html=True,
                    )

            else:
                if recipe_type == "Gathered" or recipe_type == "Recipes1M":
                    # Determine the type, based on the source of recipe's details
                    if "Gathered":
                        recipe_type = recipe_type + " Recipe"
                    if "Recipes1M" in recipe_type:
                        recipe_type = "Recipes 1M Site"

                    st.markdown(
                        "<p style='font-size: 16px;'>Cuisine Source: <a href ='https://"
                        + recipe_url
                        + "' style='color: #64ABD8;'>"
                        + recipe_type
                        + "</A><BR>Takes over a "
                        + str(recipe_preperation_time)
                        + " mins to prepare<BR>",
                        unsafe_allow_html=True,
                    )
                else:
                    # Display the recipe type, & the approximate preperation time
                    st.markdown(
                        "<p style='font-size: 16px;'>"
                        + recipe_type
                        + "Cuisine<BR>Takes over a "
                        + str(recipe_preperation_time)
                        + " mins to prepare<BR>",
                        unsafe_allow_html=True,
                    )

            if st.session_state.user_authentication_status is not True:
                # Generate PDF file with necessary recipe details and usage terms
                download_location_1 = pdf_utils.generate_recommendations_pdf(
                    recipe_name,
                    recipe_type,
                    recipe_url,
                    recipe_ingredients,
                    recipe_instructions,
                )

                # Display a download button only for the unauthenticated app user
                st.download_button(
                    label="Download Recipe Details PDF",
                    data=open(download_location_1, "rb").read(),
                    key="download_location_1",
                    file_name=recipe_name.replace(" ", "_").lower() + ".pdf",
                )

            else:
                # For authenticated user, send a mail with recipe details and PDF
                if st.button("Send Recipe to Mail", key="button_1", use_container_width=True):
                    try:
                        st.toast("Hold tight! Your recipe is taking flight.")

                        # Generate the PDF file with the necessary recipe details
                        download_location_1 = pdf_utils.generate_recommendations_pdf(
                            recipe_name,
                            recipe_type,
                            recipe_url,
                            recipe_ingredients,
                            recipe_instructions,
                        )
                        # Send the mail with attachment to the registered mail id
                        mail_utils.send_recipe_info_to_mail(
                            recipe_name,
                            recipe_ingredients,
                            recipe_instructions,
                            st.session_state.authenticated_user_email_id,
                            download_location_1,
                        )

                        # Display the information status upon successful delivery
                        st.toast("Bon appétit! We've delivered your recipe.")

                    except Exception as error:
                        # Display the information status upon successful delivery
                        st.toast("Whoops! Looks like your recipe ran into a snag.")
                        time.sleep(1)
                        st.toast("Please check your connectivity and try again.")

            st.markdown("<BR>", unsafe_allow_html=True)

            # Fetch details of the recommended recipe from the index location - 4
            (
                recipe_name,
                recipe_type,
                recipe_ingredients,
                recipe_instructions,
                recipe_preperation_time,
                recipe_url,
            ) = feature_space_matching.lookup_recipe_details_by_index(
                recipe_data, recommended_recipes_indices[4]
            )

            # Attempt to generate image of the recipe, using Generative AI models
            generated_image_path = genisys.generate_image(recipe_name, 225, 225)

            if generated_image_path:
                recipe_image = Image.open(generated_image_path)
            else:
                # Display a placeholder image if the image could not be generated
                placeholder_image_path = (
                    resource_registry.placeholder_image_dir_path + "placeholder_5.png"
                )
                recipe_image = Image.open(placeholder_image_path).resize((225, 225))

            st.image(recipe_image)

            # Shorten recipe name to max 26 characters and add ellipsis if longer
            if len(recipe_name) <= 26:
                recipe_name = recipe_name
            else:
                recipe_name = recipe_name[:25] + "..."

            # Display the name of the recommended recipe, as an HTML <H6> heading
            st.markdown("<H6>" + recipe_name + "</H6>", unsafe_allow_html=True)

            # Display recipe details including the source, URL & preparation time
            if recipe_preperation_time < 100:
                if recipe_type == "Gathered" or recipe_type == "Recipes1M":
                    # Determine the type, based on the source of recipe's details
                    if "Gathered":
                        recipe_type = recipe_type + " Recipe"
                    if "Recipes1M" in recipe_type:
                        recipe_type = "Recipes 1M Site"

                    st.markdown(
                        "<p style='font-size: 16px;'>Cuisine Source: <a href ='https://"
                        + recipe_url
                        + "' style='color: #64ABD8;'>"
                        + recipe_type
                        + "</A><BR>Takes around "
                        + str(recipe_preperation_time)
                        + " mins to prepare<BR>",
                        unsafe_allow_html=True,
                    )
                else:
                    # Display the recipe type, & the approximate preperation time
                    st.markdown(
                        "<p style='font-size: 16px;'>"
                        + recipe_type
                        + "Cuisine<BR>Takes around "
                        + str(recipe_preperation_time)
                        + " mins to prepare<BR>",
                        unsafe_allow_html=True,
                    )

            else:
                if recipe_type == "Gathered" or recipe_type == "Recipes1M":
                    # Determine the type, based on the source of recipe's details
                    if "Gathered":
                        recipe_type = recipe_type + " Recipe"
                    if "Recipes1M" in recipe_type:
                        recipe_type = "Recipes 1M Site"

                    st.markdown(
                        "<p style='font-size: 16px;'>Cuisine Source: <a href ='https://"
                        + recipe_url
                        + "' style='color: #64ABD8;'>"
                        + recipe_type
                        + "</A><BR>Takes over a "
                        + str(recipe_preperation_time)
                        + " mins to prepare<BR>",
                        unsafe_allow_html=True,
                    )
                else:
                    # Display the recipe type, & the approximate preperation time
                    st.markdown(
                        "<p style='font-size: 16px;'>"
                        + recipe_type
                        + "Cuisine<BR>Takes over a "
                        + str(recipe_preperation_time)
                        + " mins to prepare<BR>",
                        unsafe_allow_html=True,
                    )

            if st.session_state.user_authentication_status is not True:
                # Generate PDF file with necessary recipe details and usage terms
                download_location_4 = pdf_utils.generate_recommendations_pdf(
                    recipe_name,
                    recipe_type,
                    recipe_url,
                    recipe_ingredients,
                    recipe_instructions,
                )

                # Display a download button only for the unauthenticated app user
                st.download_button(
                    label="Download Recipe Details PDF",
                    data=open(download_location_4, "rb").read(),
                    key="download_location_4",
                    file_name=recipe_name.replace(" ", "_").lower() + ".pdf",
                )

            else:
                # For authenticated user, send a mail with recipe details and PDF
                if st.button("Send Recipe to Mail", key="button_4", use_container_width=True):
                    try:
                        st.toast("Hold tight! Your recipe is taking flight.")

                        # Generate the PDF file with the necessary recipe details
                        download_location_4 = pdf_utils.generate_recommendations_pdf(
                            recipe_name,
                            recipe_type,
                            recipe_url,
                            recipe_ingredients,
                            recipe_instructions,
                        )
                        # Send the mail with attachment to the registered mail id
                        mail_utils.send_recipe_info_to_mail(
                            recipe_name,
                            recipe_ingredients,
                            recipe_instructions,
                            st.session_state.authenticated_user_email_id,
                            download_location_4,
                        )

                        # Display the information status upon successful delivery
                        st.toast("Bon appétit! We've delivered your recipe.")

                    except Exception as error:
                        # Display the information status upon successful delivery
                        st.toast("Whoops! Looks like your recipe ran into a snag.")
                        time.sleep(1)
                        st.toast("Please check your connectivity and try again.")

        with container_3:
            # Fetch details of the recommended recipe from the index location - 2
            (
                recipe_name,
                recipe_type,
                recipe_ingredients,
                recipe_instructions,
                recipe_preperation_time,
                recipe_url,
            ) = feature_space_matching.lookup_recipe_details_by_index(
                recipe_data, recommended_recipes_indices[2]
            )

            # Attempt to generate image of the recipe, using Generative AI models
            generated_image_path = genisys.generate_image(recipe_name, 225, 225)

            if generated_image_path:
                recipe_image = Image.open(generated_image_path)
            else:
                # Display a placeholder image if the image could not be generated
                placeholder_image_path = (
                    resource_registry.placeholder_image_dir_path + "placeholder_3.png"
                )
                recipe_image = Image.open(placeholder_image_path).resize((225, 225))

            st.image(recipe_image)

            # Shorten recipe name to max 26 characters and add ellipsis if longer
            if len(recipe_name) <= 26:
                recipe_name = recipe_name
            else:
                recipe_name = recipe_name[:26] + "..."

            # Display the name of the recommended recipe, as an HTML <H6> heading
            st.markdown("<H6>" + recipe_name + "</H6>", unsafe_allow_html=True)

            # Display recipe details including the source, URL & preparation time
            if recipe_preperation_time < 100:
                if recipe_type == "Gathered" or recipe_type == "Recipes1M":
                    # Determine the type, based on the source of recipe's details
                    if "Gathered":
                        recipe_type = recipe_type + " Recipe"
                    if "Recipes1M" in recipe_type:
                        recipe_type = "Recipes 1M Site"

                    st.markdown(
                        "<p style='font-size: 16px;'>Cuisine Source: <a href ='https://"
                        + recipe_url
                        + "' style='color: #64ABD8;'>"
                        + recipe_type
                        + "</A><BR>Takes around "
                        + str(recipe_preperation_time)
                        + " mins to prepare<BR>",
                        unsafe_allow_html=True,
                    )
                else:
                    # Display the recipe type, & the approximate preperation time
                    st.markdown(
                        "<p style='font-size: 16px;'>"
                        + recipe_type
                        + "Cuisine<BR>Takes around "
                        + str(recipe_preperation_time)
                        + " mins to prepare<BR>",
                        unsafe_allow_html=True,
                    )

            else:
                if recipe_type == "Gathered" or recipe_type == "Recipes1M":
                    # Determine the type, based on the source of recipe's details
                    if "Gathered":
                        recipe_type = recipe_type + " Recipe"
                    if "Recipes1M" in recipe_type:
                        recipe_type = "Recipes 1M Site"

                    st.markdown(
                        "<p style='font-size: 16px;'>Cuisine Source: <a href ='https://"
                        + recipe_url
                        + "' style='color: #64ABD8;'>"
                        + recipe_type
                        + "</A><BR>Takes over a "
                        + str(recipe_preperation_time)
                        + " mins to prepare<BR>",
                        unsafe_allow_html=True,
                    )
                else:
                    # Display the recipe type, & the approximate preperation time
                    st.markdown(
                        "<p style='font-size: 16px;'>"
                        + recipe_type
                        + "Cuisine<BR>Takes over a "
                        + str(recipe_preperation_time)
                        + " mins to prepare<BR>",
                        unsafe_allow_html=True,
                    )

            if st.session_state.user_authentication_status is not True:
                # Generate PDF file with necessary recipe details and usage terms
                download_location_2 = pdf_utils.generate_recommendations_pdf(
                    recipe_name,
                    recipe_type,
                    recipe_url,
                    recipe_ingredients,
                    recipe_instructions,
                )

                # Display a download button only for the unauthenticated app user
                st.download_button(
                    label="Download Recipe Details PDF",
                    data=open(download_location_2, "rb").read(),
                    key="download_location_2",
                    file_name=recipe_name.replace(" ", "_").lower() + ".pdf",
                )

            else:
                # For authenticated user, send a mail with recipe details and PDF
                if st.button("Send Recipe to Mail", key="button_2", use_container_width=True):
                    try:
                        st.toast("Hold tight! Your recipe is taking flight.")

                        # Generate the PDF file with the necessary recipe details
                        download_location_2 = pdf_utils.generate_recommendations_pdf(
                            recipe_name,
                            recipe_type,
                            recipe_url,
                            recipe_ingredients,
                            recipe_instructions,
                        )
                        # Send the mail with attachment to the registered mail id
                        mail_utils.send_recipe_info_to_mail(
                            recipe_name,
                            recipe_ingredients,
                            recipe_instructions,
                            st.session_state.authenticated_user_email_id,
                            download_location_2,
                        )

                        # Display the information status upon successful delivery
                        st.toast("Bon appétit! We've delivered your recipe.")

                    except Exception as error:
                        # Display the information status upon successful delivery
                        st.toast("Whoops! Looks like your recipe ran into a snag.")
                        time.sleep(1)
                        st.toast("Please check your connectivity and try again.")

            st.markdown("<BR>", unsafe_allow_html=True)

            # Fetch details of the recommended recipe from the index location - 5
            (
                recipe_name,
                recipe_type,
                recipe_ingredients,
                recipe_instructions,
                recipe_preperation_time,
                recipe_url,
            ) = feature_space_matching.lookup_recipe_details_by_index(
                recipe_data, recommended_recipes_indices[5]
            )

            # Attempt to generate image of the recipe, using Generative AI models
            generated_image_path = genisys.generate_image(recipe_name, 225, 225)

            if generated_image_path:
                recipe_image = Image.open(generated_image_path)
            else:
                # Display a placeholder image if the image could not be generated
                placeholder_image_path = (
                    resource_registry.placeholder_image_dir_path + "placeholder_6.png"
                )
                recipe_image = Image.open(placeholder_image_path).resize((225, 225))

            st.image(recipe_image)

            # Shorten recipe name to max 26 characters and add ellipsis if longer
            if len(recipe_name) <= 26:
                recipe_name = recipe_name
            else:
                recipe_name = recipe_name[:25] + "..."

            # Display the name of the recommended recipe, as an HTML <H6> heading
            st.markdown("<H6>" + recipe_name + "</H6>", unsafe_allow_html=True)

            # Display recipe details including the source, URL & preparation time
            if recipe_preperation_time < 100:
                if recipe_type == "Gathered" or recipe_type == "Recipes1M":
                    # Determine the type, based on the source of recipe's details
                    if "Gathered":
                        recipe_type = recipe_type + " Recipe"
                    if "Recipes1M" in recipe_type:
                        recipe_type = "Recipes 1M Site"

                    st.markdown(
                        "<p style='font-size: 16px;'>Cuisine Source: <a href ='https://"
                        + recipe_url
                        + "' style='color: #64ABD8;'>"
                        + recipe_type
                        + "</A><BR>Takes around "
                        + str(recipe_preperation_time)
                        + " mins to prepare<BR>",
                        unsafe_allow_html=True,
                    )
                else:
                    # Display the recipe type, & the approximate preperation time
                    st.markdown(
                        "<p style='font-size: 16px;'>"
                        + recipe_type
                        + "Cuisine<BR>Takes around "
                        + str(recipe_preperation_time)
                        + " mins to prepare<BR>",
                        unsafe_allow_html=True,
                    )

            else:
                if recipe_type == "Gathered" or recipe_type == "Recipes1M":
                    # Determine the type, based on the source of recipe's details
                    if "Gathered":
                        recipe_type = recipe_type + " Recipe"
                    if "Recipes1M" in recipe_type:
                        recipe_type = "Recipes 1M Site"

                    st.markdown(
                        "<p style='font-size: 16px;'>Cuisine Source: <a href ='https://"
                        + recipe_url
                        + "' style='color: #64ABD8;'>"
                        + recipe_type
                        + "</A><BR>Takes over a "
                        + str(recipe_preperation_time)
                        + " mins to prepare<BR>",
                        unsafe_allow_html=True,
                    )
                else:
                    # Display the recipe type, & the approximate preperation time
                    st.markdown(
                        "<p style='font-size: 16px;'>"
                        + recipe_type
                        + "Cuisine<BR>Takes over a "
                        + str(recipe_preperation_time)
                        + " mins to prepare<BR>",
                        unsafe_allow_html=True,
                    )

            if st.session_state.user_authentication_status is not True:
                # Generate PDF file with necessary recipe details and usage terms
                download_location_5 = pdf_utils.generate_recommendations_pdf(
                    recipe_name,
                    recipe_type,
                    recipe_url,
                    recipe_ingredients,
                    recipe_instructions,
                )

                # Display a download button only for the unauthenticated app user
                st.download_button(
                    label="Download Recipe Details PDF",
                    data=open(download_location_5, "rb").read(),
                    key="download_location_5",
                    file_name=recipe_name.replace(" ", "_").lower() + ".pdf",
                )

            else:
                # For authenticated user, send a mail with recipe details and PDF
                if st.button("Send Recipe to Mail", key="button_5", use_container_width=True):
                    try:
                        st.toast("Hold tight! Your recipe is taking flight.")

                        # Generate the PDF file with the necessary recipe details
                        download_location_5 = pdf_utils.generate_recommendations_pdf(
                            recipe_name,
                            recipe_type,
                            recipe_url,
                            recipe_ingredients,
                            recipe_instructions,
                        )
                        # Send the mail with attachment to the registered mail id
                        mail_utils.send_recipe_info_to_mail(
                            recipe_name,
                            recipe_ingredients,
                            recipe_instructions,
                            st.session_state.authenticated_user_email_id,
                            download_location_5,
                        )

                        # Display the information status upon successful delivery
                        st.toast("Bon appétit! We've delivered your recipe.")

                    except Exception as error:
                        # Display the information status upon successful delivery
                        st.toast("Whoops! Looks like your recipe ran into a snag.")
                        time.sleep(1)
                        st.toast("Please check your connectivity and try again.")

        st.markdown(
            f"<br><br>",
            unsafe_allow_html=True,
        )
	    
    else:
        try:
            # Load & display animated GIF for visual appeal, when not inferencing
            dotwave_image_path = (
                resource_registry.loading_assets_dir + "intro_dotwave_img.gif"
            )

            with open(dotwave_image_path, "rb") as f:
                image_data = f.read()
                encoded_image = base64.b64encode(image_data).decode()

            gif_image = st.markdown(
                f'<br><div class="rounded-image"><img src="data:image/png;base64,{encoded_image}"></div>',
                unsafe_allow_html=True,
            )
        except: pass

        # Display a welcoming message to user with a randomly chosen recipe emoji
        cuisines_emojis = ["🍜", "🍩", "🍚", "🍝", "🍦", "🍣"]

        st.markdown(
            f"<H1>Hello there {random.choice(cuisines_emojis)}</H1>",
            unsafe_allow_html=True,
        )

        # Provide a brief description of RecipeMLs recipe generation capabilities
        st.markdown(
            "<H4 style='color: #c2c2c2;'>Start by describing few ingredients and unlock delicious possibilities</H4>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<P align='justify'>Tired of the same old meals? Craving something new and exciting? Just tell us what's in your fridge, and we'll whip up a selection of mouthwatering recipes that are sure to satisfy any craving. So lets get cooking!</P>",
            unsafe_allow_html=True,
        )

        # Display usage instructions in an informative box for easy understanding
        usage_instruction = """
        **Here's how you can get started:**

        **1. Pick your ingredients**: Select from a list of over 10,000+ ingredients from across culinary traditions
        **2. Find your match**: Browse through our curated list of recipes, discover hidden gems, & get inspired!
        **3. Save your favourite recipes**: Download the PDF documents, or send'em to your registered email id
        """
        st.info(usage_instruction)  # Display the usage information, to the users

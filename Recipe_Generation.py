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
This module contains the top-level environment for RecipeML streamlit application. 
It contains method for initializing the recipe generation model, managing session, 
and displaying the frontend. This module uses RESTful APIs for Gen AI inferencing.

The front-end of the web application is developed using streamlit and css and the 
backend uses Python3. The supporting modules and the related resources are stored 
in the repective directories of the repository. API keys are maintained as secret.

Module Functions:
    [1] apply_style_to_sidebar_button
    [2] set_generate_recommendations_cache_to_true
    [3] set_generate_recommendations_cache_to_false

APIs Used:
    [1] PaLM API (Google)
    [2] DALL.E2 API (OpenAI)

.. versionadded:: 1.0.0
.. versionupdated:: 1.3.0

Learn about RecipeML :ref:`RecipeML v1: User Interface and Functionality Overview`
"""
import re
import random
import streamlit as st
import time
import streamlit_antd_components as sac

import base64
import joblib
import pandas as pd
from PIL import Image
from gtts import gTTS
from deep_translator import GoogleTranslator

from cognitive_flux.recipe_generation import ProceduralTextGeneration

from configurations.api_authtoken import AuthTokens
from configurations.resource_path import ResourceRegistry

from deep_canvas.image_generation import GenerativeImageSynthesis
from pages.Discover_RecipeML import display_discover_recipeml_page

# Set the page title and favicon to be displayed on the streamlit web application
st.set_page_config(
    page_title="RecipeML: Recipe Generation",
    page_icon="assets/images/favicon/recipeml_favicon.png",
)

# Remove the extra paddings from the top and bottom margin of the block container
st.markdown(
    """
        <style>
               .block-container {
                    padding-top: 0.5rem;
                    padding-bottom: -1rem;
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


if __name__ == "__main__":
    # [Recipe Generation](Page 1/2) - The frontend of the streamlit web application

    # Function to display RecipeML's home screen, with various interactive elements.

    # The homepage incorporates a dropdown widget on the sidebar that sets paradigm
    # for recipe generation. Users query is read and then subsequently utilized for
    # executing multiple task including recipe generation & recipe image generation

    # This web application utilizes the OpenAI API & RunwayMLs Stable Diffusion for
    # image generation. The model binaries are maintained as GitHubs release assets.

    # .. versionadded:: 1.1.0

    # APIs: OpenAI API (Image: DALL.E2) & Google Pathways Language Model (PaLM API)
    # Dependencies: NLTK, Torch, Diffusers, SkLearn, Pandas, Joblib, Streamlit, PIL

    # NOTE: The API keys used in this module are maintained using streamlit secrets.
    # When testing locally replace the API keys from ~./streamlit/secrets.toml file.
    with st.sidebar:
        selected_menu_item = sac.menu(
            [
                sac.MenuItem(
                    "Recipe Generation", icon="stars",
                ),
                sac.MenuItem(
                    "Discover RecipeML",
                    icon="layers",
                    tag=[sac.Tag('New', color='blue')]
                ),
                sac.MenuItem(' ', disabled=True),
                sac.MenuItem(type="divider"),
            ],
            open_all=True,
        )

    if selected_menu_item == "Recipe Generation":
        auth_token = AuthTokens()
        resource_registry = ResourceRegistry()

        # Fetch the preloader image from the assets directory, to be used in this app
        loading_image_path = resource_registry.loading_assets_dir + "loading_img.gif"

        with open(loading_image_path, "rb") as f:
            image_data = f.read()
            encoded_image = base64.b64encode(image_data).decode()

        cola, colb = st.sidebar.columns([2.5, 1])

        with cola:        
            # Display selectbox for users to select RecipeML's recipe generation paradigm
            recipe_generation_type = st.selectbox(
                "Select Generation Technique",
                ("Generate by Name", "Generate by Ingredients"),
            )
        
        with colb:
            selected_language = st.selectbox(
                " ",
                ['en', 'hi', 'ml', 'ta', 'fr', 'ru', 'de', 'ja', 'ko'],
                help='Select language'
            )

        # Check if the recipe generation's selectbox is set to Generate by Ingredient
        if recipe_generation_type == "Generate by Ingredients":
            # Load the ingredients list from the resource registry into the selectbox
            with open(
                resource_registry.ingredients_list_path, "rb"
            ) as recipe_nlg_ingredients:
                ingredients_list = joblib.load(recipe_nlg_ingredients)

            # Display sidebar with selectbox for a user to select the ingredients
            selected_ingredients = st.sidebar.selectbox(
                "Select a start ingredient",
                ingredients_list,
                index=None,
                placeholder="Pick from over 10,000+ ingredients",
            )

            if selected_ingredients:
                # Display the preloader, as the web app performs time intensive tasks
                gif_image = st.markdown(
                    f'<br><br><br><br><div class="rounded-image"><img src="data:image/png;base64,{encoded_image}"></div><br><br><br><br><br><br><br><br><br><br><br>',
                    unsafe_allow_html=True,
                )
                try:
                    with gif_image:
                        st.toast("Cooking up some tasty ideas")  # Show notifications

                        # Attempt to generate a recipe using ProceduralTextGeneration
                        procedural_text_generation = ProceduralTextGeneration(
                            stochasticity=0.3,
                            max_token_length=1500,
                            palm_api_key=auth_token.palm_api_key,
                        )

                        # Fetch various recipe details using ProceduralTextGeneration
                        (
                            recipe_title,
                            recipe_ingredients,
                            recipe_instructions,
                            preperation_time_in_mins,
                            serving_size,
                            recipe_description,
                            calories_in_recipe,
                        ) = procedural_text_generation.generate_recipe(
                            selected_ingredients, generate_recipe_by_name=False
                        )
                        flag_display_result = True

                except Exception as error:
                    # Handle exception, display warning and wait for clearing warning
                    try:
                        gif_image.empty()
                        st.sidebar.exception(error)

                    except Exception as error:
                        st.sidebar.exception(error)

                    try:
                        with open("assets/loading/exception_img.gif", "rb") as f:
                            image_data = f.read()
                            encoded_image = base64.b64encode(image_data).decode()

                        # Display exception preloader if the app encounters any error
                        display_exception_preloader = st.markdown(
                            f'<br><br><div class="rounded-image"><img src="data:image/png;base64,{encoded_image}"></div><br><br><br><br><br><br><br><br><br><br><br>',
                            unsafe_allow_html=True,
                        )
                        time.sleep(15)  # Hold execution before clearing the warnings
                        display_exception_preloader.empty()

                    except:
                        flag_exception_raised = st.warning(
                            "Whoops! Looks like your recipe ran into a snag. Try again [Error Code: 201]"
                        )
                        st.sidebar.exception(error)

                        time.sleep(10)  # Hold execution before clearing the warnings
                        flag_exception_raised.empty()

                    flag_display_result = False
            else:
                flag_display_result = False

        # Check if the recipe generations selectbox is set to Generate by Recipe Name
        elif recipe_generation_type == "Generate by Name":

            # Display sidebar with text input for the user to enter a recipe name
            input_recipe_name = st.sidebar.text_input(
                "Enter Recipe Name:",
                placeholder="Be playful, descriptive, or even a little poetic",
            )

            # Check if some recipe name is provided by the user, initialized on enter
            if input_recipe_name:
                gif_image = st.markdown(
                    f'<br><br><br><br><div class="rounded-image"><img src="data:image/png;base64,{encoded_image}"></div><br><br><br><br><br><br><br><br><br><br><br>',
                    unsafe_allow_html=True,
                )
                try:
                    # Display the preloader, as the app performs time intensive tasks
                    with gif_image:
                        st.toast("Cooking up some tasty ideas")
                        # Attempt to generate a recipe using ProceduralTextGeneration
                        procedural_text_generation = ProceduralTextGeneration(
                            stochasticity=0.7,
                            max_token_length=1500,
                            palm_api_key=auth_token.palm_api_key,
                        )

                        (
                            recipe_title,
                            recipe_ingredients,
                            recipe_instructions,
                            preperation_time_in_mins,
                            serving_size,
                            recipe_description,
                            calories_in_recipe,
                        ) = procedural_text_generation.generate_recipe(
                            input_recipe_name, generate_recipe_by_name=True
                        )
                        flag_display_result = True

                except Exception as err:
                    # Handle exception, display warning and wait for clearing warning
                    try:
                        gif_image.empty()

                    except:
                        pass

                    try:
                        with open("assets/loading/exception_img.gif", "rb") as f:
                            image_data = f.read()
                            encoded_image = base64.b64encode(image_data).decode()

                        # Display exception preloader if the app encounters any error
                        display_exception_preloader = st.markdown(
                            f'<br><br><div class="rounded-image"><img src="data:image/png;base64,{encoded_image}"></div><br><br><br><br><br><br><br><br><br><br><br>',
                            unsafe_allow_html=True,
                        )
                        time.sleep(15)  # Hold execution before clearing the warnings
                        display_exception_preloader.empty()

                    except:
                        flag_exception_raised = st.warning(
                            "Whoops! Looks like your recipe ran into a snag. Try again [Error Code: 202]"
                        )
                        #st.sidebar.exception(error)

                        time.sleep(10)  # Hold execution before clearing the warnings
                        flag_exception_raised.empty()

                    flag_display_result = False
            else:
                flag_display_result = False

        else:
            # Handle unknown exception, display warning and wait for clearing warning
            try:
                with open("assets/loading/exception_img.gif", "rb") as f:
                    image_data = f.read()
                    encoded_image = base64.b64encode(image_data).decode()

                # Display exception preloader if the streamlt app encounter any error
                display_exception_preloader = st.markdown(
                    f'<br><br><div class="rounded-image"><img src="data:image/png;base64,{encoded_image}"></div><br><br><br><br><br><br><br><br><br><br><br>',
                    unsafe_allow_html=True,
                )
                time.sleep(15)  # Hold execution for 15s before clearing the warnings
                display_exception_preloader.empty()

            except:
                flag_exception_raised = st.warning(
                    "Whoops! Looks like your recipe ran into a snag. Try again [Error Code: 203]"
                )
                #st.sidebar.exception(error)

                time.sleep(10)  # Hold execution for 10s before clearing the warnings
                flag_exception_raised.empty()

            flag_display_result = False

        if flag_display_result:  # Check if any result has been generated, to display
            # Remove extra paddings from the top and bottom margin of block container
            st.markdown(
                """
                    <style>
                        .block-container {
                                padding-top: 0rem;
                    padding-bottom: -0.5rem;
                            }
                    </style>
                    """,
                unsafe_allow_html=True,
            )

            with gif_image:
                # Display the preloader, as the web app performs time intensive tasks
                st.toast("Warming up the digital oven")

                # Initialize the GenerativeImageSynthesis mode,l for image generation
                genisys_std_model = GenerativeImageSynthesis(
                    image_quality="standard", enable_gpu_acceleration=False
                )

                # Generate the primary and secondary images based on the recipe title
                generated_primary_image_path = genisys_std_model.generate_image(
                    recipe_title, 424, 322
                )

                st.toast("Applying some final touches")

                generated_secondary_image_path = genisys_std_model.generate_image(
                    recipe_title, 284, 322
                )

                try:
                    audio_prompt = f"Hello and welcome to RecipeML! You are listening to the recipe for preparing {recipe_title}. This recipe takes approximately {preperation_time_in_mins} minutes to cook and can be served to {serving_size} people. Before jumping right in, please be advised that recipes generated by RecipeML are intended for creative exploration only. The results may not always be safe, accurate, or edible. You may use it to spark inspiration, but always consult trusted sources for reliable cooking information. For preparing this recipe, you will need {recipe_ingredients}. Now, here's how we'll make magic happen, {recipe_instructions}. Congratulations, chef! Your feast is ready. Grab your utensils, gather your guests, and savor every bite of this delectable dish. Bon appétit!"
                    
                    try:
                        if selected_language != 'en':
                            audio_prompt = GoogleTranslator(source='auto', target=selected_language).translate(audio_prompt)
                    except Exception as error: pass

                    tts = gTTS(text=audio_prompt, lang='en', tld='co.in')

                    recipe_audio_name = recipe_title.replace("/", "").replace(" ", "_").lower()+ "_audio.wav"
                    audio_path = f'exports/generated_aud/{recipe_audio_name}'

                    tts.save(audio_path)

                except Exception as error: pass
                gif_image.empty()  # Show notification message, & clear the preloader

            try:
                if selected_language != 'en':
                    recipe_title = GoogleTranslator(source='auto', target=selected_language).translate(recipe_title)
            except Exception as error: pass

            # Display title on the web app's frontend and show the interactive button
            st.markdown(
                f"<H2>{recipe_title}</H2>", 
                unsafe_allow_html=True
            )

            seprating_spaces = "&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp"

            serving_size_content = f"Serving for {serving_size}"
            calories_in_recipe_content = f"{calories_in_recipe} Calories"
            preparation_time_content = f"Requires {preperation_time_in_mins} minutes to prepare"

            try:
                if selected_language != 'en':
                    serving_size_content = GoogleTranslator(source='auto', target=selected_language).translate(serving_size_content)
                    calories_in_recipe_content = GoogleTranslator(source='auto', target=selected_language).translate(calories_in_recipe_content)
                    preparation_time_content = GoogleTranslator(source='auto', target=selected_language).translate(preparation_time_content)
            except Exception as error: st.write(error)
            
            st.markdown(
                "<h5>🍜 " + serving_size_content + seprating_spaces + "🔥 " + calories_in_recipe_content + seprating_spaces + "🕓 " + preparation_time_content + "</h5><br>", 
                unsafe_allow_html=True,
            )

            primary_image, secondary_image = st.columns([1.48, 1])

            # Display the primary and the secondary recipe image on the apps frontend
            with primary_image:
                if generated_primary_image_path:
                    recipe_image_424x322 = Image.open(generated_primary_image_path).resize(
                        (424, 322)
                    )

                else:
                    # Use the placeholder image if the primary image is not generated
                    placeholder_image_path = (
                        resource_registry.placeholder_image_dir_path + "placeholder_1.png"
                    )
                    recipe_image_424x322 = Image.open(placeholder_image_path).resize(
                        (424, 322)
                    )

                st.image(recipe_image_424x322)  # Display the primary image, to users

            with secondary_image:
                if generated_secondary_image_path:
                    recipe_image_424x322 = Image.open(
                        generated_secondary_image_path
                    ).resize((284, 322))

                else:
                    # Use a placeholder image if the secondary image is not generated
                    placeholder_image_path = (
                        resource_registry.placeholder_image_dir_path + "placeholder_2.png"
                    )
                    recipe_image_424x322 = Image.open(placeholder_image_path).resize(
                        (284, 322)
                    )

                st.image(recipe_image_424x322)

            # Display the description of the generated recipe & the recipe ingredient
            try:
                if selected_language != 'en':
                    recipe_description = GoogleTranslator(source='auto', target=selected_language).translate(recipe_description)
            except Exception as error: pass

            st.markdown(
                f"<p align='justify'>{recipe_description}</p>", unsafe_allow_html=True
            )

            ingredients_title = "Ingredients"

            try:
                if selected_language != 'en':
                    ingredients_title = GoogleTranslator(source='auto', target=selected_language).translate(ingredients_title)
            except Exception as error: pass

            st.markdown("<H3>"+ ingredients_title + "</H3>", unsafe_allow_html=True)

            recipe_ingredients = ", ".join(recipe_ingredients)  # Display ingredients
            try:
                if selected_language != 'en':
                    recipe_ingredients = GoogleTranslator(source='auto', target=selected_language).translate(recipe_ingredients)
            except Exception as error: pass

            st.markdown(
                f"<p align='justify'>{recipe_ingredients}</p>", unsafe_allow_html=True
            )

            # Display a cautionary message to the user, about using generated recipes
            usage_caution_message = """
            **Enjoy the wordplay, but cook with caution!**

            Recipes generated by RecipeML are intended for creative exploration only! The results may'nt always be safe, accurate, or edible! You may use it to spark inspiration but always consult trusted sources for reliable cooking information. For more information on safe cooking practices, kindly visit USDAs [FSIS](https://www.fsis.usda.gov/wps/portal/fsis/topics/food-safety-education)
            """
            st.info(usage_caution_message)

            # Display the recipe direction heading and the cleaned recipe instruction
            directions_heading = "Recipe Directions"
            try:
                if selected_language != 'en':
                    directions_heading = GoogleTranslator(source='auto', target=selected_language).translate(directions_heading)
            except Exception as error: pass

            st.markdown("<H3>" + directions_heading + "</H3>", unsafe_allow_html=True)

            try:
                if selected_language != 'en':
                    recipe_instructions = GoogleTranslator(source='auto', target=selected_language).translate(recipe_instructions)
            except Exception as error: pass

            st.markdown(
                f"<p align='justify'>{recipe_instructions}</p>", unsafe_allow_html=True
            )
            
            st.sidebar.markdown("<BR><BR><BR><BR><BR><BR>", unsafe_allow_html=True)
            st.sidebar.audio(audio_path, format='audio/wav')

        else:
            try:
                # Load & display animated GIF for visual appeal, when not inferencing
                dotwave_image_path = (
                    resource_registry.loading_assets_dir + "intro_dotwave_img.gif"
                )

                with open(dotwave_image_path, "rb") as f:
                    image_data = f.read()
                    encoded_image = base64.b64encode(image_data).decode()

                    # Display base64 encoded image with rounded edge without expander
                    gif_image = st.markdown(
                        f'<br><br><div class="rounded-image"><img src="data:image/png;base64,{encoded_image}"></div>',
                        unsafe_allow_html=True,
                    )

            except:
                pass

            # Display a welcoming message to user with a randomly chosen recipe emoji
            cuisines_emojis = [
                "🍜",
                "🍩",
                "🍚",
                "🍝",
                "🍦",
                "🍣",
            ]

            st.markdown(
                f"<H1>Hello there {random.choice(cuisines_emojis)}</H1>",
                unsafe_allow_html=True,
            )

            # Provide a brief description of RecipeMLs recipe generation capabilities
            st.markdown(
                "<H4 style='color: #c2c2c2;'>Start by describing what you are craving, or what you have on hand!</H4>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<P align='justify'>Dreaming of a dish but dont have recipe? Or maybe you've got an ingredient begging to be transformed? Simply describe what you want and RecipeML will generate a mouthwatering recipe that is uniquely yours</P>",
                unsafe_allow_html=True,
            )

            # Display usage instructions in an informative box for easy understanding
            usage_instruction = """
            **Here's how you can get started:**

            **1. Whisper your wish**: Enter the recipes name or a starting ingredient to get started with your journey
            **2. Discover inspirations**: Explore new recipes, from tried-and-true classics to some unexpected twists
            **3. Save your favourite recipes**: Download the PDF documents, or send'em to your registered email id
            """
            st.info(usage_instruction)  # Display the usage information, to the users

    if selected_menu_item == "Discover RecipeML":
        display_discover_recipeml_page()

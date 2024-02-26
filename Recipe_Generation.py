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
import requests
import streamlit as st
import time
import streamlit_antd_components as sac

import base64
import joblib
import pandas as pd
from PIL import Image
from gtts import gTTS
from deep_translator import GoogleTranslator

import firebase_admin
from firebase_admin import auth, credentials

from cognitive_flux.recipe_generation import ProceduralTextGeneration

from configurations.api_authtoken import AuthTokens
from configurations.resource_path import ResourceRegistry
from configurations.firebase_credentials import FirebaseCredentials

from deep_canvas.image_generation import GenerativeImageSynthesis


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
        if "user_authentication_status" not in st.session_state:
            st.session_state.user_authentication_status = None

        if "authenticated_user_email_id" not in st.session_state:
            st.session_state.authenticated_user_email_id = None


        def _valid_name(fullname):
            # Validate the basic structure, and logical name based character restrictions
            if not re.match(r"^[A-Z][a-z]+( [A-Z][a-z]+)*$", fullname):
                return False

            return True  # Name is considered to be valid, only if all conditions are met


        def _valid_username(username):
            # Check for the minimum and maximum length of the password (i.e 4 characters)
            if len(username) < 4:
                return False, "MINIMUM_LENGTH_UID"
            if len(username) > 25:
                return False, "MAXIMUM_LENGTH_UID"

            # Check for only the allowed characters: letters, numbers, underscores & dots
            allowed_chars = set(
                "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_."
            )
            if not all(char in allowed_chars for char in username):
                return False, "INVALID_CHARACTERS"

            # Check if username start with letter. Symbols & digits must not be the first
            if not username[0].isalpha():
                return False, "START_WITH_LETTERS"

            return True, "USERNAME_VALID"  # Username is valid, if all conditions are met


        def _valid_email_address(email):
            # Define the regular expression for validating the e-mail address of the user
            email_regex = r"^[a-zA-Z0-9.+_-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"

            # Returns a boolean value indicating whether the mail address is valid or not
            return re.match(email_regex, email) is not None


        def signup_form():
            if st.session_state.user_authentication_status is None:
                with st.form("register_new_user_form"):
                    st.markdown(
                        '<H3 id="anchor-create-user">Register Now to Create an Account</H3>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        "<p align='justify' style='color: #e2e2e2;'>Level up your recipe game! Get personalized recipe recommendations, create custom meal plans and more. Signup for your free RecipeML account today! Already have a account? LogIn now to get started</p>",
                        unsafe_allow_html=True,
                    )

                    signup_form_section_1, signup_form_section_2 = st.columns(2)

                    with signup_form_section_1:
                        name = st.text_input(
                            "Enter your Full Name:",
                        )
                        email = st.text_input(
                            "Enter your E-Mail Id:",
                        )

                    with signup_form_section_2:
                        username = st.text_input(
                            "Enter your Username:",
                            placeholder="Allowed characters: A-Z, 0-9, . & _",
                        )
                        phone_number = st.text_input(
                            "Enter Phone Number:",
                            placeholder="Include your Country Code (eg: +91)",
                        )

                    password = st.text_input(
                        "Enter your Password:",
                        type="password",
                    )

                    accept_terms_and_conditions = st.checkbox(
                        "By creating an account, you confirm your acceptance to our Terms of Use and the Privacy Policy"
                    )
                    button_section_1, button_section_2, button_section_3 = st.columns(3)

                    with button_section_1:
                        submitted = st.form_submit_button(
                            "Register Now", use_container_width=True
                        )

                    if submitted:
                        try:
                            if not name:
                                st.toast("Please enter your full name")
                            elif not _valid_name(name):
                                st.toast("Not quite! Double-check your full name.")

                            elif not _valid_username(username)[0]:
                                validation_error_message = _valid_username(username)[1]

                                if validation_error_message is "MINIMUM_LENGTH_UID":
                                    st.toast("Username too short! Needs 4+ letters.")

                                elif validation_error_message is "MAXIMUM_LENGTH_UID":
                                    st.toast("Username too long! Max 25 letters.")

                                elif validation_error_message is "INVALID_CHARACTERS":
                                    st.toast("Username contains invalid charecters!")
                                    time.sleep(1.5)
                                    st.toast("Try again with valid chars (a-z, 0-9, ._)")

                                elif validation_error_message is "START_WITH_LETTERS":
                                    st.toast("Start your username with a letter.")

                                else:
                                    st.toast("Invalid Username! Try again.")

                            elif not _valid_email_address(email):
                                st.toast("Invalid email format. Please try again.")

                            elif len(password) < 8:
                                st.toast("Password too short! Needs 8+ characters.")

                            elif not accept_terms_and_conditions:
                                st.toast("Please accept our terms of use")

                            else:
                                firebase_admin.auth.create_user(
                                    uid=username.lower(),
                                    display_name=name,
                                    email=email,
                                    phone_number=phone_number,
                                    password=password,
                                )
                                st.toast("Welcome to RecipeML!")

                                alert_successful_account_creation = st.success(
                                    "Your Account has been created successfully"
                                )
                                time.sleep(2)

                                st.toast("Please login to access your account.")
                                time.sleep(3)
                                alert_successful_account_creation.empty()

                        except Exception as error:
                            if "Invalid phone number" in str(error):
                                st.toast("Invalid phone number format.")

                                time.sleep(1.5)
                                st.toast("Please check country code and + prefix.")

                            elif "PHONE_NUMBER_EXISTS" in str(error):
                                st.toast("User with phone number already exists")

                            elif "DUPLICATE_LOCAL_ID" in str(error):
                                st.toast("The username is already taken")

                            elif "EMAIL_EXISTS" in str(error):
                                st.toast("User with provided email already exists")

                            else:
                                alert_failed_account_creation = st.warning(
                                    "Oops! We could not create your account. Please check your connectivity and try again."
                                )
                                time.sleep(7)
                                alert_failed_account_creation.empty()


        def login_form():
            if st.session_state.user_authentication_status is None:
                with st.sidebar.form("login_existing_user_form"):
                    email = st.text_input(
                        "Username / Email Id:", placeholder="Username or email address"
                    )
                    password = st.text_input(
                        "Enter your Password:", type="password", placeholder="Password"
                    )

                    submitted_login = st.form_submit_button(
                        "LogIn to RecipeML", use_container_width=True
                    )
                    st.markdown(
                        "&nbsp;New to RecipeML? <A href='#register-now-to-create-an-account' style='color: #64ABD8;'>Create an account</A>",
                        unsafe_allow_html=True,
                    )

                    if st.session_state.user_authentication_status is False:
                        st.sidebar.error("Invalid Username or Password")

                    if submitted_login:
                        try:
                            api_key = auth_token.firebase_api_key
                            base_url = "https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={api_key}"

                            if "@" not in email:
                                username = email
                                user = firebase_admin.auth.get_user(username)
                                email = user.email

                            data = {"email": email, "password": password}
                            response = requests.post(
                                base_url.format(api_key=api_key), json=data
                            )

                            if response.status_code == 200:
                                data = response.json()

                                user_display_name = data["displayName"]
                                user_email_id = email
                                user_username = firebase_admin.auth.get_user_by_email(email).uid

                                user_phone_number = firebase_admin.auth.get_user_by_email(
                                    email
                                ).phone_number

                                st.session_state.user_authentication_status = True
                                st.session_state.authenticated_user_email_id = user_email_id

                                st.rerun()

                            else:
                                data = response.json()
                                login_error_message = str(data["error"]["message"])

                                if login_error_message == "INVALID_PASSWORD":
                                    authentication_failed_alert = st.sidebar.warning(
                                        "&nbsp; Invalid password. Try again.", icon="⚠️"
                                    )
                                elif login_error_message == "EMAIL_NOT_FOUND":
                                    authentication_failed_alert = st.sidebar.warning(
                                        "&nbsp; User with this mail doesn't exist.", icon="⚠️"
                                    )
                                else:
                                    authentication_failed_alert = st.sidebar.warning(
                                        "&nbsp; Unable to login. Try again later.", icon="⚠️"
                                    )

                                time.sleep(2)
                                authentication_failed_alert.empty()

                                st.session_state.user_authentication_status = False
                                st.session_state.authenticated_user_email_id = None

                        except Exception as err:
                            authentication_failed_alert = st.sidebar.warning(
                                err, icon="⚠️"
                            )

                            time.sleep(2)
                            authentication_failed_alert.empty()

                            st.session_state.user_authentication_status = False
                            st.session_state.authenticated_user_email_id = None

            return (
                st.session_state.user_authentication_status,
                st.session_state.authenticated_user_email_id,
            )


        def logout_button():
            if st.sidebar.button("Logout from RecipeML", use_container_width=True):
                st.session_state.user_authentication_status = None
                st.session_state.authenticated_user_email_id = None
                st.rerun()


        def reset_password_form():
            with st.sidebar.expander("Forgot password"):
                api_key = auth_token.firebase_api_key
                base_url = "https://identitytoolkit.googleapis.com/v1/accounts:sendOobCode?key={api_key}"

                email = st.text_input(
                    "Enter your registered email id", placeholder="Registered email address"
                )

                if st.button("Reset Password", use_container_width=True):
                    data = {"requestType": "PASSWORD_RESET", "email": email}
                    response = requests.post(base_url.format(api_key=api_key), json=data)

                    if response.status_code == 200:
                        alert_password_reset_mail_sent = st.success(
                            "A password reset mail is on its way!"
                        )
                        st.toast("Success! Password reset email sent.")

                        time.sleep(2)
                        st.toast("Check your mailbox for next steps.")

                        time.sleep(3)
                        alert_password_reset_mail_sent.empty()

                    else:
                        alert_password_reset_mail_failed = st.error(
                            "Failed to send password reset mail"
                        )
                        st.toast("We're having trouble sending the email.")

                        time.sleep(2)
                        st.toast("Double-check your mail id and try again")

                        time.sleep(3)
                        alert_password_reset_mail_failed.empty()


        try:
            firebase_credentials = FirebaseCredentials()
            firebase_credentials.fetch_firebase_service_credentials(
                "configurations/recipeml_firebase_secrets.json"
            )
        
            firebase_credentials = credentials.Certificate(
                "configurations/recipeml_firebase_secrets.json"
            )
            firebase_admin.initialize_app(firebase_credentials)

        except Exception as err: pass

        auth_token = AuthTokens()

        # Display the Title of the ~/About_the_WebApp, and the sub-title as HTML headings
        st.markdown(
            "<H2>RecipeML - Cooking Just Got Smarter!</H2>",
            unsafe_allow_html=True,
        )

        st.markdown(
            "<H4>Start by describing few ingredients and unlock delicious possibilities</H4>",
            unsafe_allow_html=True,
        )

        # Display the content for the intriduction section of the ~/About_the_WebApp page
        st.markdown(
            "<P align='justify'>Tired of staring at a fridge full of possibilities, only to end up with the same old stir-fry? Break free from the ordinary, & let RecipeML revolutionize your kitchen experience, with the power of Artificial Intelligence</P>",
            unsafe_allow_html=True,
        )

        st.markdown(
            "<P align='justify'>Start by describing what you have in hand, and RecipeML will work its magic. Whether it's that leftover bag of spinach or a fridge begging for rescue, RecipeML transforms ordinary ingredients into extraordinary dishes. But wait, there's more! Beyond recommending those existing recipes, RecipeML taps into its deep understanding of language generation to conjure up novel recipes, that no cook book has ever dreamt of!!</P>",
            unsafe_allow_html=True,
        )

        icon0, icon1, icon2, icon3, icon4, icon5, icon6, icon7, icon8, icon9 = st.columns(10)

        with icon0:
            with open("assets/icons/1.png", "rb") as f:  # Display the robot avatar image
                image_data = f.read()
                encoded_image = base64.b64encode(image_data).decode()

                gif_image = st.markdown(
                    f'<div class="rounded-image"><img src="data:image/png;base64,{encoded_image}"></div>',
                    unsafe_allow_html=True,
                )

        with icon1:
            with open("assets/icons/2.png", "rb") as f:  # Display the robot avatar image
                image_data = f.read()
                encoded_image = base64.b64encode(image_data).decode()

                gif_image = st.markdown(
                    f'<div class="rounded-image"><img src="data:image/png;base64,{encoded_image}"></div>',
                    unsafe_allow_html=True,
                )

        with icon2:
            with open("assets/icons/3.png", "rb") as f:  # Display the robot avatar image
                image_data = f.read()
                encoded_image = base64.b64encode(image_data).decode()

                gif_image = st.markdown(
                    f'<div class="rounded-image"><img src="data:image/png;base64,{encoded_image}"></div>',
                    unsafe_allow_html=True,
                )

        with icon3:
            with open("assets/icons/4.png", "rb") as f:  # Display the robot avatar image
                image_data = f.read()
                encoded_image = base64.b64encode(image_data).decode()

                gif_image = st.markdown(
                    f'<div class="rounded-image"><img src="data:image/png;base64,{encoded_image}"></div>',
                    unsafe_allow_html=True,
                )

        with icon4:
            with open("assets/icons/5.png", "rb") as f:  # Display the robot avatar image
                image_data = f.read()
                encoded_image = base64.b64encode(image_data).decode()

                gif_image = st.markdown(
                    f'<div class="rounded-image"><img src="data:image/png;base64,{encoded_image}"></div>',
                    unsafe_allow_html=True,
                )

        with icon5:
            with open("assets/icons/6.png", "rb") as f:  # Display the robot avatar image
                image_data = f.read()
                encoded_image = base64.b64encode(image_data).decode()

                gif_image = st.markdown(
                    f'<div class="rounded-image"><img src="data:image/png;base64,{encoded_image}"></div>',
                    unsafe_allow_html=True,
                )

        with icon6:
            with open("assets/icons/7.png", "rb") as f:  # Display the robot avatar image
                image_data = f.read()
                encoded_image = base64.b64encode(image_data).decode()

                gif_image = st.markdown(
                    f'<div class="rounded-image"><img src="data:image/png;base64,{encoded_image}"></div>',
                    unsafe_allow_html=True,
                )

        with icon7:
            with open("assets/icons/8.png", "rb") as f:  # Display the robot avatar image
                image_data = f.read()
                encoded_image = base64.b64encode(image_data).decode()

                gif_image = st.markdown(
                    f'<div class="rounded-image"><img src="data:image/png;base64,{encoded_image}"></div>',
                    unsafe_allow_html=True,
                )

        with icon8:
            with open("assets/icons/9.png", "rb") as f:  # Display the robot avatar image
                image_data = f.read()
                encoded_image = base64.b64encode(image_data).decode()

                gif_image = st.markdown(
                    f'<div class="rounded-image"><img src="data:image/png;base64,{encoded_image}"></div>',
                    unsafe_allow_html=True,
                )

        with icon9:
            st.image("assets/icons/10.png")  # Display the roboavatar on the explore page

        st.markdown(
            "<H5>So what are you waiting for? Elevate your cooking game, discover new flavors, and redefine your kitchen escapades with RecipeML, now available across all countries</H5>",
            unsafe_allow_html=True,
        )

        # Display section heading for Register Now to Try RecipeML, & display the content
        st.markdown(
            "<H3>No Hidden Ingredients Here! - RecipeML v1.3 Privacy Policy</H3>",
            unsafe_allow_html=True,
        )

        st.markdown(
            "<P align='justify'>Safety starts with understanding how we collect and share your data while using RecipeML. We believe that responsible innovation doesn't happen in isolation. As part of our efforts to enhance the outcomes, your usage information & feedback will be collected, and further used to improve our language algorithms</P>",
            unsafe_allow_html=True,
        )

        st.markdown(
            "<P align='justify'><B>•&nbsp&nbsp&nbsp What we collect:</B> We collect your chosen ingredients, feedback on the outcomes & basic app usage data<BR><B>•&nbsp&nbsp&nbsp What we dont:</B> We never share your information with third parties for marketing or advertising purpose</P>",
            unsafe_allow_html=True,
        )

        st.markdown(
            "<P align='justify'>Should you ever wish to discontinue your participation, we encourage you to reach out to us via email. Your privacy and preferences matter and we want to ensure your experience aligns with your comfort level</P>",
            unsafe_allow_html=True,
        )

        # Display a cautionary message to user about using generated recipes with caution
        usage_caution_message = """
        **Enjoy the wordplay, but cook with caution!**

        Recipes generated by RecipeML are intended for creative exploration only! The results may'nt always be safe, accurate, or edible! You may use it to spark inspiration but always consult trusted sources for reliable cooking information. For more information on safe cooking practices, kindly visit USDAs [FSIS](https://www.fsis.usda.gov/wps/portal/fsis/topics/food-safety-education)
        """
        st.warning(usage_caution_message)

        st.markdown(
            "<P align='justify'>To start using RecipeML, head to the homepage and select the application of your choice and look out for the results generated by our language models. Try different combinations, experiment with ingredients & discover some delightful culinary creation. RecipeML may occasionally return incorrect recommendations</P>",
            unsafe_allow_html=True,
        )

        # Perform authentication using streamlit authenticator, and retrieve user details
        authentication_status, email_id = login_form()

        try:
            if authentication_status is not True:
                st.markdown("---", unsafe_allow_html=True)
                st.markdown("<BR>", unsafe_allow_html=True)

                signup_form()
                reset_password_form()
                

        except Exception as error: pass

        try:
            if authentication_status is None: pass
        except Exception as err: pass

        # Rerun the streamlit application if authentication fails for a user during login
        try:
            if authentication_status is False:
                st.session_state.user_authentication_status = None
                st.rerun()

        except Exception as err: pass

        # When logged in, display the message and the logout button, and the dark message
        try:
            if authentication_status is True:
                authentication_success_alert = st.sidebar.success(
                    "Succesfully logged in to RecipeML",
                )

                st.sidebar.markdown(
                    "<BR><BR><BR><BR><BR><BR><BR><BR><BR><BR>", unsafe_allow_html=True
                )
                st.sidebar.write(" ")

                logout_button()

        except Exception as err: pass

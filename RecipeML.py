import re
import uuid
import time
import random
import pickle
import requests
from PIL import Image

import nltk
import numpy as np
from io import BytesIO

import base64
import joblib
import pandas as pd
from gtts import gTTS
from deep_translator import GoogleTranslator

import streamlit as st
import streamlit_antd_components as sac

import firebase_admin
from firebase_admin import auth, credentials

from deep_canvas.image_generation import GenerativeImageSynthesis
from feature_scape.recommendation import FeatureSpaceMatching
from cognitive_flux.recipe_generation import ProceduralTextGeneration

from backend.send_mail import MailUtils
from backend.generate_pdf import PDFUtils

from configurations.api_authtoken import AuthTokens
from configurations.resource_path import ResourceRegistry
from configurations.firebase_credentials import FirebaseCredentials

from database.mongodb import MongoDB
from database.blob_storage import AzureStorageAccount


# Set the page title and favicon to be displayed on the streamlit web application
st.set_page_config(
    page_title="RecipeML",
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

    return (
        True,
        "USERNAME_VALID",
    )  # Username is valid, if all conditions are met


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

            subheading_font_color = {"dark": "#E2E2E2", "light": "#111111"}
            font_color = subheading_font_color[st.session_state.themes["current_theme"]]

            st.markdown(
                f"<p align='justify' style='color: {font_color};'>Level up your recipe game! Get personalized recipe recommendations, create custom meal plans and more. Signup for your free RecipeML account today! Already have a account? LogIn now to get started</p>",
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
                        st.session_state.authenticated_user_username = user_username
                        st.session_state.user_display_name = user_display_name

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
                                "&nbsp; User with this mail doesn't exist.",
                                icon="⚠️",
                            )
                        else:
                            authentication_failed_alert = st.sidebar.warning(
                                "&nbsp; Unable to login. Try again later.",
                                icon="⚠️",
                            )

                        time.sleep(2)
                        authentication_failed_alert.empty()

                        st.session_state.user_authentication_status = False
                        st.session_state.authenticated_user_email_id = None
                        st.session_state.authenticated_user_username = None
                        st.session_state.user_display_name = None

                except Exception as err:
                    authentication_failed_alert = st.sidebar.warning(err, icon="⚠️")

                    time.sleep(2)
                    authentication_failed_alert.empty()

                    st.session_state.user_authentication_status = False
                    st.session_state.authenticated_user_email_id = None
                    st.session_state.authenticated_user_username = None
                    st.session_state.user_display_name = None

    return (
        st.session_state.user_authentication_status,
        st.session_state.authenticated_user_email_id,
    )


def logout_button():
    if st.sidebar.button("Logout from RecipeML", use_container_width=True):
        st.session_state.user_authentication_status = None
        st.session_state.authenticated_user_email_id = None
        st.session_state.authenticated_user_username = None
        st.session_state.user_display_name = None
        st.rerun()


def reset_password_form():
    with st.sidebar.expander("Forgot password"):
        api_key = auth_token.firebase_api_key
        base_url = "https://identitytoolkit.googleapis.com/v1/accounts:sendOobCode?key={api_key}"

        email = st.text_input(
            "Enter your registered email id",
            placeholder="Registered email address",
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


if "themes" not in st.session_state:
    st.session_state.themes = {
        "current_theme": "dark",
        "refreshed": True,
        "light": {
            "theme.base": "dark",
            "theme.backgroundColor": "#111111",
            "theme.primaryColor": "#64ABD8",
            "theme.secondaryBackgroundColor": "#181818",
            "theme.textColor": "#FFFFFF",
            "button_face": "🌜",
        },
        "dark": {
            "theme.base": "light",
            "theme.backgroundColor": "#fdfefe",
            "theme.primaryColor": "#64ABD8",
            "theme.secondaryBackgroundColor": "#dee4e7",
            "theme.textColor": "#333333",
            "button_face": "🌞",
        },
    }


def change_streamlit_theme():
    previous_theme = st.session_state.themes["current_theme"]
    tdict = (
        st.session_state.themes["light"]
        if st.session_state.themes["current_theme"] == "light"
        else st.session_state.themes["dark"]
    )

    for vkey, vval in tdict.items():
        if vkey.startswith("theme"):
            st._config.set_option(vkey, vval)

    st.session_state.themes["refreshed"] = False

    if previous_theme == "dark":
        st.session_state.themes["current_theme"] = "light"

    elif previous_theme == "light":
        st.session_state.themes["current_theme"] = "dark"


if st.session_state.themes["refreshed"] == False:
    st.session_state.themes["refreshed"] = True
    st.rerun()


if "user_authentication_status" not in st.session_state:
    st.session_state.user_authentication_status = None

if "authenticated_user_email_id" not in st.session_state:
    st.session_state.authenticated_user_email_id = None

if "authenticated_user_username" not in st.session_state:
    st.session_state.authenticated_user_username = None

if "user_display_name" not in st.session_state:
    st.session_state.user_display_name = None

if "cache_generate_recommendations" not in st.session_state:
    st.session_state.cache_generate_recommendations = False


def set_generate_recommendations_cache_to_true():
    st.session_state.cache_generate_recommendations = True


def set_generate_recommendations_cache_to_false():
    st.session_state.cache_generate_recommendations = False


def apply_style_to_sidebar_button(css_file_name):
    with open(css_file_name, encoding="utf-8") as file:
        st.markdown(f"<style>{file.read()}</style>", unsafe_allow_html=True)


if __name__ == "__main__":
    if st.session_state.user_authentication_status is not True:
        with st.sidebar:
            selected_menu_item = sac.menu(
                [
                    sac.MenuItem(
                        "RecipML v1: Home",
                        icon="house",
                    ),
                    sac.MenuItem(
                        "Create an Account",
                        icon="layers",
                        tag=[sac.Tag("New", color="blue")],
                    ),
                    sac.MenuItem(" ", disabled=True),
                    sac.MenuItem(type="divider"),
                ],
                open_all=True,
            )

        if selected_menu_item == "RecipML v1: Home":
            if "user_authentication_status" not in st.session_state:
                st.session_state.user_authentication_status = None

            if "authenticated_user_email_id" not in st.session_state:
                st.session_state.authenticated_user_email_id = None

            if "authenticated_user_username" not in st.session_state:
                st.session_state.authenticated_user_username = None

            if "user_display_name" not in st.session_state:
                st.session_state.user_display_name = None

            try:
                firebase_credentials = FirebaseCredentials()
                firebase_credentials.fetch_firebase_service_credentials(
                    "configurations/recipeml_firebase_secrets.json"
                )

                firebase_credentials = credentials.Certificate(
                    "configurations/recipeml_firebase_secrets.json"
                )
                firebase_admin.initialize_app(firebase_credentials)

            except Exception as err:
                pass

            auth_token = AuthTokens()
            authentication_status, email_id = login_form()

            try:
                if authentication_status is not True:
                    reset_password_form()

            except Exception as error:
                pass

            try:
                if authentication_status is None:
                    pass

            except Exception as error:
                pass

            # Rerun the streamlit application if authentication fails for a user during login
            try:
                if authentication_status is False:
                    st.session_state.user_authentication_status = None
                    st.rerun()

            except Exception as error:
                pass

            # When logged in, display the message and the logout button, and the dark message
            try:
                if authentication_status is True:
                    authentication_success_alert = st.sidebar.success(
                        "Succesfully logged in to RecipeML",
                    )

                    st.sidebar.markdown(
                        "<BR><BR><BR><BR><BR><BR><BR><BR><BR><BR>",
                        unsafe_allow_html=True,
                    )
                    st.sidebar.write(" ")

                    logout_button()

            except Exception as error:
                pass

            resource_registry = ResourceRegistry()
            try:
                dotwave_image_path = (
                    resource_registry.loading_assets_dir + "dotwave_intro_img.gif"
                )

                with open(dotwave_image_path, "rb") as f:
                    image_data = f.read()

                    encoded_image = base64.b64encode(image_data).decode()
                    st.write(" ")
                    # Display base64 encoded image with rounded edge without expander
                    gif_image = st.markdown(
                        f'<div class="rounded-image"><img src="data:image/png;base64,{encoded_image}"></div>',
                        unsafe_allow_html=True,
                    )
            except Exception as error:
                st.write(error)

            col1, col2 = st.columns([11.5, 1])
            with col1:
                st.markdown("<H1>RecipeML v1.2</H1>", unsafe_allow_html=True)
            with col2:
                st.markdown("<br>", unsafe_allow_html=True)
                st.button(
                    "H",
                    use_container_width=True,
                    type="secondary",
                )

            st.markdown(
                "<H4 align='justify'>Start by describing what you are craving, or what you have on hand and unlock delicious possibilities with RecipeML, in just a single tap!</H4>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<P align='justify'>So whether it's that leftover bag of spinach, or a fridge begging for rescue - RecipeML transforms ordinary ingredients into extraordinary dishes. Feeling hungry?! LogIn to RecipeML & lets cook something amazing!",
                unsafe_allow_html=True,
            )

            inline_display_col1, inline_display_col2 = st.columns([2.5, 1])
            with inline_display_col1:
                st.markdown(
                    "<B>Subscribe to receive our newsletter, directly into your inbox</B>",
                    unsafe_allow_html=True,
                )

            with st.form("form_subscribers"):
                # st.markdown("Subscribe to receive our newsletter", unsafe_allow_html=True)
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.text_input(
                        "",
                        placeholder="Enter your email id",
                        label_visibility="collapsed",
                    )
                with col2:
                    st.form_submit_button(
                        "Subscribe Now", use_container_width=True, type="primary"
                    )

        if selected_menu_item == "Create an Account":
            try:
                firebase_credentials = FirebaseCredentials()
                firebase_credentials.fetch_firebase_service_credentials(
                    "configurations/recipeml_firebase_secrets.json"
                )

                firebase_credentials = credentials.Certificate(
                    "configurations/recipeml_firebase_secrets.json"
                )
                firebase_admin.initialize_app(firebase_credentials)

            except Exception as err:
                pass

            auth_token = AuthTokens()

            # Perform authentication using streamlit authenticator, and retrieve user details
            authentication_status, email_id = login_form()

            try:
                if authentication_status is not True:
                    signup_form()

                    reset_password_form()

                    st.markdown(
                        f"<BR>",
                        unsafe_allow_html=True,
                    )

            except Exception as error:
                pass

            try:
                if authentication_status is None:
                    pass

            except Exception as error:
                pass

            # Rerun the streamlit application if authentication fails for a user during login
            try:
                if authentication_status is False:
                    st.session_state.user_authentication_status = None
                    st.rerun()

            except Exception as error:
                pass

            # When logged in, display the message and the logout button, and the dark message
            try:
                if authentication_status is True:
                    authentication_success_alert = st.sidebar.success(
                        "Succesfully logged in to RecipeML",
                    )

                    st.sidebar.markdown(
                        "<BR><BR><BR><BR><BR><BR><BR><BR><BR><BR>",
                        unsafe_allow_html=True,
                    )
                    st.sidebar.write(" ")

                    logout_button()

            except Exception as error:
                pass

    else:
        with st.sidebar:
            selected_menu_item = sac.menu(
                [
                    sac.MenuItem(
                        "Recipe Generation",
                        icon="house",
                    ),
                    sac.MenuItem(
                        "Recommendations",
                        icon="layers",
                        tag=[sac.Tag("New", color="blue")],
                    ),
                    sac.MenuItem(" ", disabled=True),
                    sac.MenuItem(type="divider"),
                ],
                open_all=True,
            )

        if selected_menu_item == "Recommendations":
            if "user_authentication_status" not in st.session_state:
                st.session_state.user_authentication_status = None

            if "authenticated_user_email_id" not in st.session_state:
                st.session_state.authenticated_user_email_id = None

            if "authenticated_user_username" not in st.session_state:
                st.session_state.authenticated_user_username = None

            if "user_display_name" not in st.session_state:
                st.session_state.user_display_name = None

            auth_token = AuthTokens()
            resource_registry = ResourceRegistry()
            feature_space_matching = FeatureSpaceMatching()

            genisys = GenerativeImageSynthesis(
                image_quality="low", enable_gpu_acceleration=False
            )

            # Fetch preloader image from the assets directory, to be used in this app
            if st.session_state.themes["current_theme"] == "dark":
                loading_image_path = (
                    resource_registry.loading_assets_dir + "loading_img.gif"
                )
            else:
                loading_image_path = (
                    resource_registry.loading_assets_dir + "loading_img_light.gif"
                )

            with open(loading_image_path, "rb") as f:
                image_data = f.read()
                encoded_image = base64.b64encode(image_data).decode()

            # Load processed list of ingredients from the binary dump to a global var
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
                # Read CSS code from the file & allow html parsing to apply the style
                apply_style_to_sidebar_button("assets/css/login_home_button_style.css")
            except:
                pass  # Use default style if file is'nt found or if exception happens

            # Create multiselect widget in the sidebar for selecting input ingredient
            selected_ingredients = st.sidebar.multiselect(
                "Select the ingredients",
                ingredients_list,
                on_change=set_generate_recommendations_cache_to_false,
            )
            input_ingredients = [
                ingredient.lower() for ingredient in selected_ingredients
            ]

            generate_recommendations_button = st.sidebar.button(
                "Recommend Recipes",
                on_click=set_generate_recommendations_cache_to_true,
                use_container_width=True,
            )

            # Check if ingredients are selected, and recommendation button is clicked
            if (
                generate_recommendations_button
                or st.session_state.cache_generate_recommendations
            ) and len(input_ingredients) > 0:
                st.write(" ")
                
                st.sidebar.markdown("<BR><BR><BR><BR><BR><BR><BR>", unsafe_allow_html=True)
                logout_button()

                # Display preloader, as the application performs time-intensive tasks
                gif_image = st.markdown(
                    f'<br><br><br><br><div class="rounded-image"><img src="data:image/png;base64,{encoded_image}"></div><br><br><br><br><br><br><br><br><br><br><br>',
                    unsafe_allow_html=True,
                )

                use_large_model = True

                with gif_image:
                    # Initialize the instances for PDFUtility, and MailUtility module
                    pdf_utils = PDFUtils()
                    mail_utils = MailUtils()

                    if use_large_model is True:
                        try:
                            recipeml_flask_api_url = auth_token.recipeml_flask_api_url
                            response = requests.post(
                                recipeml_flask_api_url, json=input_ingredients
                            )

                            if response.status_code == 200:
                                # Extract and print the first recommended recipe's details from JSON response
                                recommended_recipes_indices = response.json()[
                                    "recipe_id"
                                ]
                            else:
                                use_large_model = False

                        except:
                            use_large_model = False

                    if use_large_model is False:

                        @st.cache_data(show_spinner=False)
                        def _load_dataset_for_inferencing():
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

                            recipeml_processed_data = [
                                dataset1,
                                dataset2,
                                dataset3,
                                dataset4,
                                dataset5,
                            ]

                            # Load data into a variable for generating the embeddings
                            recipe_data = pd.concat(
                                recipeml_processed_data, ignore_index=True
                            )

                            recipe_data.dropna(inplace=True)
                            return recipe_data

                        recipe_data = _load_dataset_for_inferencing()

                        # Load TF/IDF vectorizer and the feature space matching model
                        (
                            tfidf_vectorizer,
                            model,
                        ) = feature_space_matching.initialize_feature_space_matching_algorithm(
                            recipe_data
                        )

                        # Generate the recommendations - using feature space matching
                        recommended_recipes_indices = (
                            feature_space_matching.generate_recipe_recommendations(
                                input_ingredients, model, tfidf_vectorizer
                            )
                        )

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

                    azure_storage_account = AzureStorageAccount(
                        "generated-recipe-images"
                    )
                    recommended_recipes_names = []
                    recommended_recipes_images = []

                    gif_image.empty()  # Stop displaying preloader image on the front end

                recommendation_id = str(uuid.uuid4())[:8]

                st.markdown(
                    "<H2>Here are some recipes you can try</H2>", unsafe_allow_html=True
                )
                st.markdown(
                    "<P align='justify'>These recommendations are generated using Recipe ML - one of our latest AI advancements. Our goal is to learn, improve, & innovate responsibly on AI together. Check out the data security policy for our users <a href='https://recipeml-recommendations.streamlit.app/Discover_RecipeML#no-hidden-ingredients-here-recipeml-v1-3-privacy-policy' style='color: #64ABD8;'>here</A></P>",
                    unsafe_allow_html=True,
                )

                # Create three columns to display recommendations on the web app's layout
                container_1, container_2, container_3 = st.columns(3)

                button_width = (
                    225  # Set the width of each column buttons, to 256 pixels
                )

                st.markdown(
                    f"<style>.stButton button {{ width: {button_width}px; }}</style>",
                    unsafe_allow_html=True,
                )

                with container_1:
                    # Fetch details of the recommended recipe from the index location - 0
                    if use_large_model:
                        (
                            recipe_name,
                            recipe_type,
                            recipe_ingredients,
                            recipe_instructions,
                            recipe_preperation_time,
                            recipe_url,
                        ) = feature_space_matching.lookup_recipe_details_by_index(
                            response.json(), 0, True
                        )

                    else:
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
                        generated_image_path = (
                            resource_registry.placeholder_image_dir_path
                            + "placeholder_1.png"
                        )
                        recipe_image = Image.open(generated_image_path).resize(
                            (225, 225)
                        )

                    st.image(recipe_image)

                    try:
                        recommended_recipes_images.append(
                            azure_storage_account.store_image_in_blob_container(
                                generated_image_path,
                                "".join(
                                    [
                                        char.lower()
                                        if char.isalnum()
                                        else "_"
                                        if char == " "
                                        else ""
                                        for char in recipe_name
                                    ]
                                )
                                + "_"
                                + recommendation_id
                                + ".png",
                            )
                        )
                    except:
                        recommended_recipes_images.append("unavailable")
                    recommended_recipes_names.append(recipe_name)

                    # Shorten recipe name to max 26 characters and add ellipsis if longer
                    if len(recipe_name) <= 26:
                        recipe_name = recipe_name
                    else:
                        recipe_name = recipe_name[:26] + "..."

                    # Display the name of the recommended recipe, as an HTML <H6> heading
                    st.markdown(
                        "<H6>" + recipe_name.title() + "</H6>", unsafe_allow_html=True
                    )

                    # Display recipe details including the source, URL & preparation time
                    if type(recipe_preperation_time) == list:
                        preperation_time_in_minutes = recipe_preperation_time[0]
                    else:
                        preperation_time_in_minutes = recipe_preperation_time

                    if preperation_time_in_minutes < 100:
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
                                + recipe_type.title()
                                + "</A><BR>Takes around "
                                + str(recipe_preperation_time)
                                + " mins to prepare<BR>",
                                unsafe_allow_html=True,
                            )
                        else:
                            # Display the recipe type, & the approximate preperation time
                            recipe_type = recipe_type + " Cuisine"
                            if len(recipe_type) > 19:
                                recipe_type = recipe_type[:15] + "..."

                            st.markdown(
                                "<p style='font-size: 16px;'>"
                                + recipe_type.title()
                                + f" • {str(recipe_preperation_time[1])} Calories<BR>Takes around "
                                + str(recipe_preperation_time[0])
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
                                + recipe_type.title()
                                + "</A><BR>Takes over a "
                                + str(recipe_preperation_time)
                                + " mins to prepare<BR>",
                                unsafe_allow_html=True,
                            )
                        else:
                            # Display the recipe type, & the approximate preperation time
                            st.markdown(
                                "<p style='font-size: 16px;'>"
                                + recipe_type.title()
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
                        if st.button(
                            "Send Recipe to Mail",
                            key="button_0",
                            use_container_width=True,
                        ):
                            try:
                                st.toast("Hold tight! Your recipe is taking flight.")

                                # Generate the PDF file with the necessary recipe details
                                download_location_0 = (
                                    pdf_utils.generate_recommendations_pdf(
                                        recipe_name,
                                        recipe_type,
                                        recipe_url,
                                        recipe_ingredients,
                                        recipe_instructions,
                                    )
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
                                st.toast(
                                    "Whoops! Looks like your recipe ran into a snag."
                                )
                                time.sleep(1)
                                st.toast(
                                    "Please check your connectivity and try again."
                                )

                    st.markdown("<BR>", unsafe_allow_html=True)

                    # Fetch details of the recommended recipe from the index location - 3
                    if use_large_model:
                        (
                            recipe_name,
                            recipe_type,
                            recipe_ingredients,
                            recipe_instructions,
                            recipe_preperation_time,
                            recipe_url,
                        ) = feature_space_matching.lookup_recipe_details_by_index(
                            response.json(), 3, True
                        )

                    else:
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
                        generated_image_path = (
                            resource_registry.placeholder_image_dir_path
                            + "placeholder_4.png"
                        )
                        recipe_image = Image.open(generated_image_path).resize(
                            (225, 225)
                        )

                    st.image(recipe_image)

                    try:
                        recommended_recipes_images.append(
                            azure_storage_account.store_image_in_blob_container(
                                generated_image_path,
                                "".join(
                                    [
                                        char.lower()
                                        if char.isalnum()
                                        else "_"
                                        if char == " "
                                        else ""
                                        for char in recipe_name
                                    ]
                                )
                                + "_"
                                + recommendation_id
                                + ".png",
                            )
                        )
                    except:
                        recommended_recipes_images.append("unavailable")
                    recommended_recipes_names.append(recipe_name)

                    # Shorten recipe name to max 26 characters and add ellipsis if longer
                    if len(recipe_name) <= 26:
                        recipe_name = recipe_name
                    else:
                        recipe_name = recipe_name[:26] + "..."

                    # Display the name of the recommended recipe, as an HTML <H6> heading
                    st.markdown(
                        "<H6>" + recipe_name.title() + "</H6>", unsafe_allow_html=True
                    )

                    # Display recipe details including the source, URL & preparation time
                    if type(recipe_preperation_time) == list:
                        preperation_time_in_minutes = recipe_preperation_time[0]
                    else:
                        preperation_time_in_minutes = recipe_preperation_time

                    if preperation_time_in_minutes < 100:
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
                                + recipe_type.title()
                                + "</A><BR>Takes around "
                                + str(recipe_preperation_time)
                                + " mins to prepare<BR>",
                                unsafe_allow_html=True,
                            )
                        else:
                            # Display the recipe type, & the approximate preperation time
                            recipe_type = recipe_type + " Cuisine"
                            if len(recipe_type) > 19:
                                recipe_type = recipe_type[:15] + "..."

                            st.markdown(
                                "<p style='font-size: 16px;'>"
                                + recipe_type.title()
                                + f" • {str(recipe_preperation_time[1])} Calories<BR>Takes around "
                                + str(recipe_preperation_time[0])
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
                                + recipe_type.title()
                                + "</A><BR>Takes over a "
                                + str(recipe_preperation_time)
                                + " mins to prepare<BR>",
                                unsafe_allow_html=True,
                            )
                        else:
                            # Display the recipe type, & the approximate preperation time
                            st.markdown(
                                "<p style='font-size: 16px;'>"
                                + recipe_type.title()
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
                        if st.button(
                            "Send Recipe to Mail",
                            key="button_3",
                            use_container_width=True,
                        ):
                            try:
                                st.toast("Hold tight! Your recipe is taking flight.")

                                # Generate the PDF file with the necessary recipe details
                                download_location_3 = (
                                    pdf_utils.generate_recommendations_pdf(
                                        recipe_name,
                                        recipe_type,
                                        recipe_url,
                                        recipe_ingredients,
                                        recipe_instructions,
                                    )
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
                                st.toast(
                                    "Whoops! Looks like your recipe ran into a snag."
                                )
                                time.sleep(1)
                                st.toast(
                                    "Please check your connectivity and try again."
                                )

                with container_2:
                    # Fetch details of the recommended recipe from the index location - 1
                    if use_large_model:
                        (
                            recipe_name,
                            recipe_type,
                            recipe_ingredients,
                            recipe_instructions,
                            recipe_preperation_time,
                            recipe_url,
                        ) = feature_space_matching.lookup_recipe_details_by_index(
                            response.json(), 1, True
                        )

                    else:
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
                        generated_image_path = (
                            resource_registry.placeholder_image_dir_path
                            + "placeholder_2.png"
                        )
                        recipe_image = Image.open(generated_image_path).resize(
                            (225, 225)
                        )

                    st.image(recipe_image)

                    try:
                        recommended_recipes_images.append(
                            azure_storage_account.store_image_in_blob_container(
                                generated_image_path,
                                "".join(
                                    [
                                        char.lower()
                                        if char.isalnum()
                                        else "_"
                                        if char == " "
                                        else ""
                                        for char in recipe_name
                                    ]
                                )
                                + "_"
                                + recommendation_id
                                + ".png",
                            )
                        )
                    except:
                        recommended_recipes_images.append("unavailable")
                    recommended_recipes_names.append(recipe_name)

                    # Shorten recipe name to max 26 characters and add ellipsis if longer
                    if len(recipe_name) <= 26:
                        recipe_name = recipe_name
                    else:
                        recipe_name = recipe_name[:26] + "..."

                    # Display the name of the recommended recipe, as an HTML <H6> heading
                    st.markdown(
                        "<H6>" + recipe_name.title() + "</H6>", unsafe_allow_html=True
                    )

                    # Display recipe details including the source, URL & preparation time
                    if type(recipe_preperation_time) == list:
                        preperation_time_in_minutes = recipe_preperation_time[0]
                    else:
                        preperation_time_in_minutes = recipe_preperation_time

                    if preperation_time_in_minutes < 100:
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
                                + recipe_type.title()
                                + "</A><BR>Takes around "
                                + str(recipe_preperation_time)
                                + " mins to prepare<BR>",
                                unsafe_allow_html=True,
                            )
                        else:
                            # Display the recipe type, & the approximate preperation time
                            recipe_type = recipe_type + " Cuisine"
                            if len(recipe_type) > 19:
                                recipe_type = recipe_type[:15] + "..."

                            st.markdown(
                                "<p style='font-size: 16px;'>"
                                + recipe_type.title()
                                + f" • {str(recipe_preperation_time[1])} Calories<BR>Takes around "
                                + str(recipe_preperation_time[0])
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
                                + recipe_type.title()
                                + "</A><BR>Takes over a "
                                + str(recipe_preperation_time)
                                + " mins to prepare<BR>",
                                unsafe_allow_html=True,
                            )
                        else:
                            # Display the recipe type, & the approximate preperation time
                            st.markdown(
                                "<p style='font-size: 16px;'>"
                                + recipe_type.title()
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
                        if st.button(
                            "Send Recipe to Mail",
                            key="button_1",
                            use_container_width=True,
                        ):
                            try:
                                st.toast("Hold tight! Your recipe is taking flight.")

                                # Generate the PDF file with the necessary recipe details
                                download_location_1 = (
                                    pdf_utils.generate_recommendations_pdf(
                                        recipe_name,
                                        recipe_type,
                                        recipe_url,
                                        recipe_ingredients,
                                        recipe_instructions,
                                    )
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
                                st.toast(
                                    "Whoops! Looks like your recipe ran into a snag."
                                )
                                time.sleep(1)
                                st.toast(
                                    "Please check your connectivity and try again."
                                )

                    st.markdown("<BR>", unsafe_allow_html=True)

                    # Fetch details of the recommended recipe from the index location - 4
                    if use_large_model:
                        (
                            recipe_name,
                            recipe_type,
                            recipe_ingredients,
                            recipe_instructions,
                            recipe_preperation_time,
                            recipe_url,
                        ) = feature_space_matching.lookup_recipe_details_by_index(
                            response.json(), 4, True
                        )

                    else:
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
                        generated_image_path = (
                            resource_registry.placeholder_image_dir_path
                            + "placeholder_5.png"
                        )
                        recipe_image = Image.open(generated_image_path).resize(
                            (225, 225)
                        )

                    st.image(recipe_image)

                    try:
                        recommended_recipes_images.append(
                            azure_storage_account.store_image_in_blob_container(
                                generated_image_path,
                                "".join(
                                    [
                                        char.lower()
                                        if char.isalnum()
                                        else "_"
                                        if char == " "
                                        else ""
                                        for char in recipe_name
                                    ]
                                )
                                + "_"
                                + recommendation_id
                                + ".png",
                            )
                        )
                    except:
                        recommended_recipes_images.append("unavailable")
                    recommended_recipes_names.append(recipe_name)

                    # Shorten recipe name to max 26 characters and add ellipsis if longer
                    if len(recipe_name) <= 26:
                        recipe_name = recipe_name
                    else:
                        recipe_name = recipe_name[:25] + "..."

                    # Display the name of the recommended recipe, as an HTML <H6> heading
                    st.markdown(
                        "<H6>" + recipe_name.title() + "</H6>", unsafe_allow_html=True
                    )

                    # Display recipe details including the source, URL & preparation time
                    if type(recipe_preperation_time) == list:
                        preperation_time_in_minutes = recipe_preperation_time[0]
                    else:
                        preperation_time_in_minutes = recipe_preperation_time

                    if preperation_time_in_minutes < 100:
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
                                + recipe_type.title()
                                + "</A><BR>Takes around "
                                + str(recipe_preperation_time)
                                + " mins to prepare<BR>",
                                unsafe_allow_html=True,
                            )
                        else:
                            # Display the recipe type, & the approximate preperation time
                            recipe_type = recipe_type + " Cuisine"
                            if len(recipe_type) > 19:
                                recipe_type = recipe_type[:15] + "..."

                            st.markdown(
                                "<p style='font-size: 16px;'>"
                                + recipe_type.title()
                                + f" • {str(recipe_preperation_time[1])} Calories<BR>Takes around "
                                + str(recipe_preperation_time[0])
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
                                + recipe_type.title()
                                + "</A><BR>Takes over a "
                                + str(recipe_preperation_time)
                                + " mins to prepare<BR>",
                                unsafe_allow_html=True,
                            )
                        else:
                            # Display the recipe type, & the approximate preperation time
                            st.markdown(
                                "<p style='font-size: 16px;'>"
                                + recipe_type.title()
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
                        if st.button(
                            "Send Recipe to Mail",
                            key="button_4",
                            use_container_width=True,
                        ):
                            try:
                                st.toast("Hold tight! Your recipe is taking flight.")

                                # Generate the PDF file with the necessary recipe details
                                download_location_4 = (
                                    pdf_utils.generate_recommendations_pdf(
                                        recipe_name,
                                        recipe_type,
                                        recipe_url,
                                        recipe_ingredients,
                                        recipe_instructions,
                                    )
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
                                st.toast(
                                    "Whoops! Looks like your recipe ran into a snag."
                                )
                                time.sleep(1)
                                st.toast(
                                    "Please check your connectivity and try again."
                                )

                with container_3:
                    # Fetch details of the recommended recipe from the index location - 2
                    if use_large_model:
                        (
                            recipe_name,
                            recipe_type,
                            recipe_ingredients,
                            recipe_instructions,
                            recipe_preperation_time,
                            recipe_url,
                        ) = feature_space_matching.lookup_recipe_details_by_index(
                            response.json(), 2, True
                        )

                    else:
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
                        generated_image_path = (
                            resource_registry.placeholder_image_dir_path
                            + "placeholder_3.png"
                        )
                        recipe_image = Image.open(generated_image_path).resize(
                            (225, 225)
                        )

                    st.image(recipe_image)

                    try:
                        recommended_recipes_images.append(
                            azure_storage_account.store_image_in_blob_container(
                                generated_image_path,
                                "".join(
                                    [
                                        char.lower()
                                        if char.isalnum()
                                        else "_"
                                        if char == " "
                                        else ""
                                        for char in recipe_name
                                    ]
                                )
                                + "_"
                                + recommendation_id
                                + ".png",
                            )
                        )
                    except:
                        recommended_recipes_images.append("unavailable")
                    recommended_recipes_names.append(recipe_name)

                    # Shorten recipe name to max 26 characters and add ellipsis if longer
                    if len(recipe_name) <= 26:
                        recipe_name = recipe_name
                    else:
                        recipe_name = recipe_name[:26] + "..."

                    # Display the name of the recommended recipe, as an HTML <H6> heading
                    st.markdown(
                        "<H6>" + recipe_name.title() + "</H6>", unsafe_allow_html=True
                    )

                    # Display recipe details including the source, URL & preparation time
                    if type(recipe_preperation_time) == list:
                        preperation_time_in_minutes = recipe_preperation_time[0]
                    else:
                        preperation_time_in_minutes = recipe_preperation_time

                    if preperation_time_in_minutes < 100:
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
                                + recipe_type.title()
                                + "</A><BR>Takes around "
                                + str(recipe_preperation_time)
                                + " mins to prepare<BR>",
                                unsafe_allow_html=True,
                            )
                        else:
                            # Display the recipe type, & the approximate preperation time
                            recipe_type = recipe_type + " Cuisine"
                            if len(recipe_type) > 19:
                                recipe_type = recipe_type[:15] + "..."

                            st.markdown(
                                "<p style='font-size: 16px;'>"
                                + recipe_type.title()
                                + f" • {str(recipe_preperation_time[1])} Calories<BR>Takes around "
                                + str(recipe_preperation_time[0])
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
                                + recipe_type.title()
                                + "</A><BR>Takes over a "
                                + str(recipe_preperation_time)
                                + " mins to prepare<BR>",
                                unsafe_allow_html=True,
                            )
                        else:
                            # Display the recipe type, & the approximate preperation time
                            st.markdown(
                                "<p style='font-size: 16px;'>"
                                + recipe_type.title()
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
                        if st.button(
                            "Send Recipe to Mail",
                            key="button_2",
                            use_container_width=True,
                        ):
                            try:
                                st.toast("Hold tight! Your recipe is taking flight.")

                                # Generate the PDF file with the necessary recipe details
                                download_location_2 = (
                                    pdf_utils.generate_recommendations_pdf(
                                        recipe_name,
                                        recipe_type,
                                        recipe_url,
                                        recipe_ingredients,
                                        recipe_instructions,
                                    )
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
                                st.toast(
                                    "Whoops! Looks like your recipe ran into a snag."
                                )
                                time.sleep(1)
                                st.toast(
                                    "Please check your connectivity and try again."
                                )

                    st.markdown("<BR>", unsafe_allow_html=True)

                    # Fetch details of the recommended recipe from the index location - 5
                    if use_large_model:
                        (
                            recipe_name,
                            recipe_type,
                            recipe_ingredients,
                            recipe_instructions,
                            recipe_preperation_time,
                            recipe_url,
                        ) = feature_space_matching.lookup_recipe_details_by_index(
                            response.json(), 5, True
                        )

                    else:
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
                        generated_image_path = (
                            resource_registry.placeholder_image_dir_path
                            + "placeholder_6.png"
                        )
                        recipe_image = Image.open(generated_image_path).resize(
                            (225, 225)
                        )

                    st.image(recipe_image)

                    try:
                        recommended_recipes_images.append(
                            azure_storage_account.store_image_in_blob_container(
                                generated_image_path,
                                "".join(
                                    [
                                        char.lower()
                                        if char.isalnum()
                                        else "_"
                                        if char == " "
                                        else ""
                                        for char in recipe_name
                                    ]
                                )
                                + "_"
                                + recommendation_id
                                + ".png",
                            )
                        )
                    except:
                        recommended_recipes_images.append("unavailable")
                    recommended_recipes_names.append(recipe_name)

                    # Shorten recipe name to max 26 characters and add ellipsis if longer
                    if len(recipe_name) <= 26:
                        recipe_name = recipe_name
                    else:
                        recipe_name = recipe_name[:25] + "..."

                    # Display the name of the recommended recipe, as an HTML <H6> heading
                    st.markdown(
                        "<H6>" + recipe_name.title() + "</H6>", unsafe_allow_html=True
                    )

                    # Display recipe details including the source, URL & preparation time
                    if type(recipe_preperation_time) == list:
                        preperation_time_in_minutes = recipe_preperation_time[0]
                    else:
                        preperation_time_in_minutes = recipe_preperation_time

                    if preperation_time_in_minutes < 100:
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
                                + recipe_type.title()
                                + "</A><BR>Takes around "
                                + str(recipe_preperation_time)
                                + " mins to prepare<BR>",
                                unsafe_allow_html=True,
                            )
                        else:
                            # Display the recipe type, & the approximate preperation time
                            recipe_type = recipe_type + " Cuisine"
                            if len(recipe_type) > 19:
                                recipe_type = recipe_type[:15] + "..."

                            st.markdown(
                                "<p style='font-size: 16px;'>"
                                + recipe_type.title()
                                + f" • {str(recipe_preperation_time[1])} Calories<BR>Takes around "
                                + str(recipe_preperation_time[0])
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
                                + recipe_type.title()
                                + "</A><BR>Takes over a "
                                + str(recipe_preperation_time)
                                + " mins to prepare<BR>",
                                unsafe_allow_html=True,
                            )
                        else:
                            # Display the recipe type, & the approximate preperation time
                            st.markdown(
                                "<p style='font-size: 16px;'>"
                                + recipe_type.title()
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
                        if st.button(
                            "Send Recipe to Mail",
                            key="button_5",
                            use_container_width=True,
                        ):
                            try:
                                st.toast("Hold tight! Your recipe is taking flight.")

                                # Generate the PDF file with the necessary recipe details
                                download_location_5 = (
                                    pdf_utils.generate_recommendations_pdf(
                                        recipe_name,
                                        recipe_type,
                                        recipe_url,
                                        recipe_ingredients,
                                        recipe_instructions,
                                    )
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
                                st.toast(
                                    "Whoops! Looks like your recipe ran into a snag."
                                )
                                time.sleep(1)
                                st.toast(
                                    "Please check your connectivity and try again."
                                )

                st.markdown(
                    f"<br><br>",
                    unsafe_allow_html=True,
                )

                try:
                    mongo = MongoDB()

                    if st.session_state.authenticated_user_username is not None:
                        username = st.session_state.authenticated_user_username
                    else:
                        username = "guest_user"

                    mongo.store_recommended_recipes(
                        username,
                        recommendation_id,
                        input_ingredients,
                        [int(index) for index in recommended_recipes_indices],
                        recommended_recipes_names,
                        recommended_recipes_images,
                    )

                except Exception as error:
                    pass

            else:
                try:
                    # Load & display animated GIF for visual appeal, when not inferencing
                    dotwave_image_path = (
                        resource_registry.loading_assets_dir + "dotwave_intro_img.gif"
                    )

                    with open(dotwave_image_path, "rb") as f:
                        image_data = f.read()
                        encoded_image = base64.b64encode(image_data).decode()

                    gif_image = st.markdown(
                        f'<br><div class="rounded-image"><img src="data:image/png;base64,{encoded_image}"></div>',
                        unsafe_allow_html=True,
                    )
                except:
                    pass

                # Display a welcoming message to user with a randomly chosen recipe emoji
                cuisines_emojis = ["🍜", "🍩", "🍚", "🍝", "🍦", "🍣"]

                cola, colb = st.columns([11.5, 1])

                with cola:
                    if st.session_state.user_display_name is not None:
                        user_first_name = st.session_state.user_display_name.split()[0]

                        try:
                            st.markdown(
                                f"<H1>Welcome {user_first_name} {random.choice(cuisines_emojis)}</H1>",
                                unsafe_allow_html=True,
                            )

                        except:
                            st.markdown(
                                f"<H1>Hello there {random.choice(cuisines_emojis)}</H1>",
                                unsafe_allow_html=True,
                            )

                    else:
                        st.markdown(
                            f"<H1>Hello there {random.choice(cuisines_emojis)}</H1>",
                            unsafe_allow_html=True,
                        )

                with colb:
                    st.markdown("<br>", unsafe_allow_html=True)
                    btn_face = (
                        st.session_state.themes["light"]["button_face"]
                        if st.session_state.themes["current_theme"] == "light"
                        else st.session_state.themes["dark"]["button_face"]
                    )

                    st.button(
                        btn_face,
                        use_container_width=True,
                        type="secondary",
                        on_click=change_streamlit_theme,
                    )

                # Provide a brief description of RecipeMLs recipe generation capabilities
                subheading_font_color = {"dark": "#C2C2C2", "light": "#424242"}
                font_color = subheading_font_color[
                    st.session_state.themes["current_theme"]
                ]

                st.markdown(
                    f"<H4 style='color: {font_color};'>Start by describing few ingredients and unlock delicious possibilities</H4>",
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
                st.info(
                    usage_instruction
                )  # Display the usage information, to the users

        elif selected_menu_item == "Recipe Generation":
            try:
                firebase_credentials = FirebaseCredentials()
                firebase_credentials.fetch_firebase_service_credentials(
                    "configurations/recipeml_firebase_secrets.json"
                )

                firebase_credentials = credentials.Certificate(
                    "configurations/recipeml_firebase_secrets.json"
                )
                firebase_admin.initialize_app(firebase_credentials)

            except Exception as err:
                pass

            if "user_authentication_status" not in st.session_state:
                st.session_state.user_authentication_status = None

            if "authenticated_user_email_id" not in st.session_state:
                st.session_state.authenticated_user_email_id = None

            if "authenticated_user_username" not in st.session_state:
                st.session_state.authenticated_user_username = None

            if "user_display_name" not in st.session_state:
                st.session_state.user_display_name = None

            auth_token = AuthTokens()
            resource_registry = ResourceRegistry()

            # Fetch the preloader image from the assets directory, to be used in this app
            if st.session_state.themes["current_theme"] == "dark":
                loading_image_path = (
                    resource_registry.loading_assets_dir + "loading_img.gif"
                )
            else:
                loading_image_path = (
                    resource_registry.loading_assets_dir + "loading_img_light.gif"
                )

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
                    ["en", "hi", "ml", "ta", "fr", "ru", "de", "ja", "ko"],
                    help="Select language",
                )

            # Check if the recipe generation's selectbox is set to Generate by Ingredient
            if recipe_generation_type == "Generate by Ingredients":
                # Load the ingredients list from the resource registry into the selectbox
                with open(
                    resource_registry.ingredients_list_path, "rb"
                ) as recipe_nlg_ingredients:
                    ingredients_list = joblib.load(recipe_nlg_ingredients)

                # Display sidebar with selectbox for a user to select the ingredients
                input_selected_ingredients = st.sidebar.selectbox(
                    "Select a start ingredient",
                    ingredients_list,
                    index=None,
                    placeholder="Pick from over 10,000+ ingredients",
                )

                if input_selected_ingredients:
                    st.sidebar.markdown("<BR><BR><BR><BR><BR><BR>", unsafe_allow_html=True)
                    logout_button()

                    # Display the preloader, as the web app performs time intensive tasks
                    gif_image = st.markdown(
                        f'<br><br><br><br><div class="rounded-image"><img src="data:image/png;base64,{encoded_image}"></div><br><br><br><br><br><br><br><br><br><br><br>',
                        unsafe_allow_html=True,
                    )

                    try:
                        with gif_image:
                            st.toast(
                                "Cooking up some tasty ideas"
                            )  # Show notifications

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
                                input_selected_ingredients,
                                generate_recipe_by_name=False,
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
                            if st.session_state.themes["current_theme"] == "dark":
                                exception_preloader = "assets/loading/exception_img.gif"
                            else:
                                exception_preloader = (
                                    "assets/loading/exception_img_light.gif"
                                )

                            with open(exception_preloader, "rb") as f:
                                image_data = f.read()
                                encoded_image = base64.b64encode(image_data).decode()

                            # Display exception preloader if the app encounters any error
                            display_exception_preloader = st.markdown(
                                f'<br><br><div class="rounded-image"><img src="data:image/png;base64,{encoded_image}"></div><br><br><br><br><br><br><br><br><br><br><br>',
                                unsafe_allow_html=True,
                            )
                            time.sleep(
                                15
                            )  # Hold execution before clearing the warnings
                            display_exception_preloader.empty()

                        except:
                            flag_exception_raised = st.warning(
                                "Whoops! Looks like your recipe ran into a snag. Try again [Error Code: 201]"
                            )
                            st.sidebar.exception(error)

                            time.sleep(
                                10
                            )  # Hold execution before clearing the warnings
                            flag_exception_raised.empty()

                        flag_display_result = False
                else:
                    flag_display_result = False

            # Check if the recipe generations selectbox is set to Generate by Recipe Name
            elif recipe_generation_type == "Generate by Name":
                st.sidebar.markdown("<BR><BR><BR><BR><BR><BR>", unsafe_allow_html=True)
                logout_button()

                # Display sidebar with text input for the user to enter a recipe name
                input_recipe_name = st.sidebar.text_input(
                    "Enter Recipe Name:",
                    placeholder="Be playful, descriptive, or even a little poetic",
                )

                # Check if some recipe name is provided by the user, initialized on enter
                if input_recipe_name:
                    st.sidebar.markdown("<BR><BR><BR><BR><BR><BR>", unsafe_allow_html=True)

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

                        except Exception as error: pass

                        try:
                            if st.session_state.themes["current_theme"] == "dark":
                                exception_preloader = "assets/loading/exception_img.gif"
                            else:
                                exception_preloader = (
                                    "assets/loading/exception_img_light.gif"
                                )

                            with open(exception_preloader, "rb") as f:
                                image_data = f.read()
                                encoded_image = base64.b64encode(image_data).decode()

                            # Display exception preloader if the app encounters any error
                            display_exception_preloader = st.markdown(
                                f'<br><br><div class="rounded-image"><img src="data:image/png;base64,{encoded_image}"></div><br><br><br><br><br><br><br><br><br><br><br>',
                                unsafe_allow_html=True,
                            )
                            time.sleep(
                                15
                            )  # Hold execution before clearing the warnings
                            display_exception_preloader.empty()

                        except:
                            flag_exception_raised = st.warning(
                                "Whoops! Looks like your recipe ran into a snag. Try again [Error Code: 202]"
                            )
                            # st.sidebar.exception(error)

                            time.sleep(
                                10
                            )  # Hold execution before clearing the warnings
                            flag_exception_raised.empty()

                        flag_display_result = False
                else:
                    flag_display_result = False

            else:
                # Handle unknown exception, display warning and wait for clearing warning
                try:
                    if st.session_state.themes["current_theme"] == "dark":
                        exception_preloader = "assets/loading/exception_img.gif"
                    else:
                        exception_preloader = "assets/loading/exception_img_light.gif"

                    with open(exception_preloader, "rb") as f:
                        image_data = f.read()
                        encoded_image = base64.b64encode(image_data).decode()

                    # Display exception preloader if the streamlt app encounter any error
                    display_exception_preloader = st.markdown(
                        f'<br><br><div class="rounded-image"><img src="data:image/png;base64,{encoded_image}"></div><br><br><br><br><br><br><br><br><br><br><br>',
                        unsafe_allow_html=True,
                    )
                    time.sleep(
                        15
                    )  # Hold execution for 15s before clearing the warnings
                    display_exception_preloader.empty()

                except:
                    flag_exception_raised = st.warning(
                        "Whoops! Looks like your recipe ran into a snag. Try again [Error Code: 203]"
                    )
                    # st.sidebar.exception(error)

                    time.sleep(
                        10
                    )  # Hold execution for 10s before clearing the warnings
                    flag_exception_raised.empty()

                flag_display_result = False

            if (
                flag_display_result
            ):  # Check if any result has been generated, to display
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

                    recipe_id = str(uuid.uuid4())[:8]

                    try:
                        azure_storage_account = AzureStorageAccount(
                            "generated-recipe-images"
                        )

                        blob_url_primary_image = (
                            azure_storage_account.store_image_in_blob_container(
                                generated_primary_image_path, recipe_id + "_primary.png"
                            )
                        )
                        blob_url_secondary_image = (
                            azure_storage_account.store_image_in_blob_container(
                                generated_secondary_image_path,
                                recipe_id + "_secondary.png",
                            )
                        )

                    except Exception as error:
                        blob_url_primary_image = "unavailable"
                        blob_url_secondary_image = "unavailable"

                    try:
                        audio_prompt = f"Hello and welcome to RecipeML! You are listening to the recipe for preparing {recipe_title}. This recipe takes approximately {preperation_time_in_mins} minutes to cook and can be served to {serving_size} people. Before jumping right in, please be advised that recipes generated by RecipeML are intended for creative exploration only. The results may not always be safe, accurate, or edible. You may use it to spark inspiration, but always consult trusted sources for reliable cooking information. For preparing this recipe, you will need {recipe_ingredients}. Now, here's how we'll make magic happen, {recipe_instructions}. Congratulations, chef! Your feast is ready. Grab your utensils, gather your guests, and savor every bite of this delectable dish. Bon appétit!"

                        try:
                            if selected_language != "en":
                                audio_prompt = GoogleTranslator(
                                    source="auto", target=selected_language
                                ).translate(audio_prompt)

                        except Exception as error:
                            pass
                        tts = gTTS(text=audio_prompt, lang="en", tld="co.in")

                        recipe_audio_name = (
                            recipe_title.replace("/", "").replace(" ", "_").lower()
                            + "_audio.wav"
                        )
                        audio_path = f"exports/generated_aud/{recipe_audio_name}"

                        tts.save(audio_path)

                    except Exception as error:
                        pass

                    gif_image.empty()  # Show notification message, & clear the preloader

                try:
                    if selected_language != "en":
                        recipe_title = GoogleTranslator(
                            source="auto", target=selected_language
                        ).translate(recipe_title)
                except Exception as error:
                    pass

                # Display title on the web app's frontend and show the interactive button
                st.markdown(f"<H2>{recipe_title}</H2>", unsafe_allow_html=True)

                seprating_spaces = "&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp"

                serving_size_content = f"Serving for {serving_size}"
                calories_in_recipe_content = f"{calories_in_recipe} Calories"
                preparation_time_content = (
                    f"Requires {preperation_time_in_mins} minutes to prepare"
                )

                try:
                    if selected_language != "en":
                        serving_size_content = GoogleTranslator(
                            source="auto", target=selected_language
                        ).translate(serving_size_content)
                        calories_in_recipe_content = GoogleTranslator(
                            source="auto", target=selected_language
                        ).translate(calories_in_recipe_content)
                        preparation_time_content = GoogleTranslator(
                            source="auto", target=selected_language
                        ).translate(preparation_time_content)

                except Exception as error:
                    st.write(error)

                st.markdown(
                    "<h5>🍜 "
                    + serving_size_content
                    + seprating_spaces
                    + "🔥 "
                    + calories_in_recipe_content
                    + seprating_spaces
                    + "🕓 "
                    + preparation_time_content
                    + "</h5><br>",
                    unsafe_allow_html=True,
                )

                primary_image, secondary_image = st.columns([1.48, 1])

                # Display the primary and the secondary recipe image on the apps frontend
                with primary_image:
                    if generated_primary_image_path:
                        recipe_image_424x322 = Image.open(
                            generated_primary_image_path
                        ).resize((424, 322))

                    else:
                        # Use the placeholder image if the primary image is not generated
                        placeholder_image_path = (
                            resource_registry.placeholder_image_dir_path
                            + "placeholder_1.png"
                        )
                        recipe_image_424x322 = Image.open(
                            placeholder_image_path
                        ).resize((424, 322))

                    st.image(
                        recipe_image_424x322
                    )  # Display the primary image, to users

                with secondary_image:
                    if generated_secondary_image_path:
                        recipe_image_424x322 = Image.open(
                            generated_secondary_image_path
                        ).resize((284, 322))

                    else:
                        # Use a placeholder image if the secondary image is not generated
                        placeholder_image_path = (
                            resource_registry.placeholder_image_dir_path
                            + "placeholder_2.png"
                        )
                        recipe_image_424x322 = Image.open(
                            placeholder_image_path
                        ).resize((284, 322))

                    st.image(recipe_image_424x322)

                # Display the description of the generated recipe & the recipe ingredient
                try:
                    if selected_language != "en":
                        recipe_description = GoogleTranslator(
                            source="auto", target=selected_language
                        ).translate(recipe_description)
                except Exception as error:
                    pass

                st.markdown(
                    f"<p align='justify'>{recipe_description}</p>",
                    unsafe_allow_html=True,
                )

                ingredients_title = "Ingredients"

                try:
                    if selected_language != "en":
                        ingredients_title = GoogleTranslator(
                            source="auto", target=selected_language
                        ).translate(ingredients_title)
                except Exception as error:
                    pass

                st.markdown(
                    "<H3>" + ingredients_title + "</H3>", unsafe_allow_html=True
                )

                recipe_ingredients = ", ".join(
                    recipe_ingredients
                )  # Display ingredients
                try:
                    if selected_language != "en":
                        recipe_ingredients = GoogleTranslator(
                            source="auto", target=selected_language
                        ).translate(recipe_ingredients)
                except Exception as error:
                    pass

                st.markdown(
                    f"<p align='justify'>{recipe_ingredients}</p>",
                    unsafe_allow_html=True,
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
                    if selected_language != "en":
                        directions_heading = GoogleTranslator(
                            source="auto", target=selected_language
                        ).translate(directions_heading)
                except Exception as error:
                    pass

                st.markdown(
                    "<H3>" + directions_heading + "</H3>", unsafe_allow_html=True
                )

                try:
                    if selected_language != "en":
                        recipe_instructions = GoogleTranslator(
                            source="auto", target=selected_language
                        ).translate(recipe_instructions)
                except Exception as error:
                    pass

                st.markdown(
                    f"<p align='justify'>{recipe_instructions}</p>",
                    unsafe_allow_html=True,
                )

                st.markdown("<BR><BR>", unsafe_allow_html=True)

                st.sidebar.markdown("<BR><BR><BR><BR><BR><BR>", unsafe_allow_html=True)
                st.sidebar.audio(audio_path, format="audio/wav")

                try:
                    mongo = MongoDB()

                    if st.session_state.authenticated_user_username is not None:
                        username = st.session_state.authenticated_user_username
                    else:
                        username = "guest_user"

                    if recipe_generation_type == "Generate by Ingredients":
                        input_query = input_selected_ingredients
                    else:
                        input_query = input_recipe_name

                    mongo.store_generated_recipes(
                        username,
                        recipe_generation_type,
                        input_query,
                        selected_language,
                        recipe_id,
                        recipe_title,
                        recipe_ingredients,
                        recipe_instructions,
                        serving_size,
                        preperation_time_in_mins,
                        calories_in_recipe,
                        blob_url_primary_image,
                        blob_url_secondary_image,
                    )
                except Exception as error:
                    st.exception(error)

            else:
                try:
                    # Load & display animated GIF for visual appeal, when not inferencing
                    if st.session_state.themes["current_theme"] == "dark":
                        loading_gif = "intro_dotwave_img.gif"
                    else:
                        loading_gif = "intro_dotwave_img_light.gif"

                    dotwave_image_path = (
                        resource_registry.loading_assets_dir + "dotwave_intro_img.gif"
                    )

                    with open(dotwave_image_path, "rb") as f:
                        image_data = f.read()

                        encoded_image = base64.b64encode(image_data).decode()

                        # Display base64 encoded image with rounded edge without expander
                        gif_image = st.markdown(
                            f'<br><br><div class="rounded-image"><img src="data:image/png;base64,{encoded_image}"></div>',
                            unsafe_allow_html=True,
                        )
                except Exception as error:
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

                cola, colb = st.columns([11.5, 1])

                with cola:
                    if st.session_state.user_display_name is not None:
                        user_first_name = st.session_state.user_display_name.split()[0]

                        try:
                            st.markdown(
                                f"<H1>Welcome {user_first_name} {random.choice(cuisines_emojis)}</H1>",
                                unsafe_allow_html=True,
                            )

                        except:
                            st.markdown(
                                f"<H1>Hello there {random.choice(cuisines_emojis)}</H1>",
                                unsafe_allow_html=True,
                            )

                    else:
                        st.markdown(
                            f"<H1>Hello there {random.choice(cuisines_emojis)}</H1>",
                            unsafe_allow_html=True,
                        )

                with colb:
                    st.markdown("<br>", unsafe_allow_html=True)
                    btn_face = (
                        st.session_state.themes["light"]["button_face"]
                        if st.session_state.themes["current_theme"] == "light"
                        else st.session_state.themes["dark"]["button_face"]
                    )

                    st.button(
                        btn_face,
                        use_container_width=True,
                        type="secondary",
                        on_click=change_streamlit_theme,
                    )

                # Provide a brief description of RecipeMLs recipe generation capabilities
                subheading_font_color = {"dark": "#C2C2C2", "light": "#424242"}
                font_color = subheading_font_color[
                    st.session_state.themes["current_theme"]
                ]

                st.markdown(
                    f"<H4 style='color: {font_color}'>Start by describing what you are craving, or what you have on hand!</H4>",
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
                st.info(
                    usage_instruction
                )  # Display the usage information, to the users

        else:
            sac.result(
                label="We couldn't connect the dots",
                description="This page was not found or you may have mistyped the address",
                status=404,
            )

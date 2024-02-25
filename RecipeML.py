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
It contains the program for displaying Discover ReecipeML page in the web app and
for authenticating & registring users through the streamlit authenticator library.

The front-end of the web application is developed using streamlit and css and the 
backend uses Python3. The supporting modules and the related resources are stored 
in the backend subdirectory of the repository. API keys are maintained as secrets.

.. versionadded:: 1.0.0
.. versionupdated:: 1.3.0

Learn about RecipeML :ref:`RecipeML v1: User Interface and Functionality Overview`
"""
import re
import requests
import time
from PIL import Image

import base64
import streamlit as st
import streamlit_antd_components as sac

import gspread
from oauth2client.service_account import ServiceAccountCredentials

from backend.send_mail import MailUtils
from configurations.api_authtoken import AuthTokens


# Set the page title and favicon to be displayed on the streamlit web application
st.set_page_config(
    page_title="RecipeML",
    page_icon="assets/images/favicon/recipeml_favicon.png",
    initial_sidebar_state="expanded",
    layout="wide",
)

# Hide the streamlit menu & the default footer from the production app"s frontend
st.markdown(
    """
    <style>
        #MainMenu  {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# Remove the extra paddings from the top and bottom margin of the block container
st.markdown(
    """
        <style>
               .block-container {
                    padding-top: 0.00005rem;
                    padding-bottom: 0rem;
                }
        </style>
        """,
    unsafe_allow_html=True,
)

# CSS markdown to display hyperlinks on the front end app with custom decorations
st.markdown(
    """
<style>
a.custom-link {
  color: white; /* Initial link color */
  text-decoration: none; /* Remove underline */
  transition: color 0.2s ease-in-out; /* Smooth color transition */
}

a.custom-link:hover {
  color: #64ABD8; /* Hover link color */
}
</style>
""",
    unsafe_allow_html=True,
)


def display_robomojis():
    # Method to display a grid of twelve robomoji icons, using the Streamlit columns.

    # The method uses base64 encoding to embed PNG images directly into the HTML web
    # page. This method relies on the "rounded-image" CSS class for its base styling.

    # Read more in the :ref:`RecipeML v1: User Interface, and Functionality Overview.

    # .. versionadded:: 1.3.0

    # Returns:
    #    The method displays the robomoji images on the streamlit webapps frontend

    (
        icon0,
        icon1,
        icon2,
        icon3,
        icon4,
        icon5,
        icon6,
        icon7,
        icon8,
        icon9,
        icon10,
        icon11,
    ) = st.columns(12)

    with icon0:
        with open("assets/icons/1.png", "rb") as f:  # Display robot avatar image
            image_data = f.read()
            encoded_image = base64.b64encode(image_data).decode()

            gif_image = st.markdown(
                f'<div class="rounded-image"><img src="data:image/png;base64,{encoded_image}"></div>',
                unsafe_allow_html=True,
            )

    with icon1:
        with open("assets/icons/2.png", "rb") as f:  # Display robot avatar image
            image_data = f.read()
            encoded_image = base64.b64encode(image_data).decode()

            gif_image = st.markdown(
                f'<div class="rounded-image"><img src="data:image/png;base64,{encoded_image}"></div>',
                unsafe_allow_html=True,
            )

    with icon2:
        with open("assets/icons/3.png", "rb") as f:  # Display robot avatar image
            image_data = f.read()
            encoded_image = base64.b64encode(image_data).decode()

            gif_image = st.markdown(
                f'<div class="rounded-image"><img src="data:image/png;base64,{encoded_image}"></div>',
                unsafe_allow_html=True,
            )

    with icon3:
        with open("assets/icons/4.png", "rb") as f:  # Display robot avatar image
            image_data = f.read()
            encoded_image = base64.b64encode(image_data).decode()

            gif_image = st.markdown(
                f'<div class="rounded-image"><img src="data:image/png;base64,{encoded_image}"></div>',
                unsafe_allow_html=True,
            )

    with icon4:
        with open("assets/icons/5.png", "rb") as f:  # Display robot avatar image
            image_data = f.read()
            encoded_image = base64.b64encode(image_data).decode()

            gif_image = st.markdown(
                f'<div class="rounded-image"><img src="data:image/png;base64,{encoded_image}"></div>',
                unsafe_allow_html=True,
            )

    with icon5:
        with open("assets/icons/6.png", "rb") as f:  # Display robot avatar image
            image_data = f.read()
            encoded_image = base64.b64encode(image_data).decode()

            gif_image = st.markdown(
                f'<div class="rounded-image"><img src="data:image/png;base64,{encoded_image}"></div>',
                unsafe_allow_html=True,
            )

    with icon6:
        with open("assets/icons/7.png", "rb") as f:  # Display robot avatar image
            image_data = f.read()
            encoded_image = base64.b64encode(image_data).decode()

            gif_image = st.markdown(
                f'<div class="rounded-image"><img src="data:image/png;base64,{encoded_image}"></div>',
                unsafe_allow_html=True,
            )

    with icon7:
        with open("assets/icons/8.png", "rb") as f:  # Display robot avatar image
            image_data = f.read()
            encoded_image = base64.b64encode(image_data).decode()

            gif_image = st.markdown(
                f'<div class="rounded-image"><img src="data:image/png;base64,{encoded_image}"></div>',
                unsafe_allow_html=True,
            )

    with icon8:
        with open("assets/icons/9.png", "rb") as f:  # Display robot avatar image
            image_data = f.read()
            encoded_image = base64.b64encode(image_data).decode()

            gif_image = st.markdown(
                f'<div class="rounded-image"><img src="data:image/png;base64,{encoded_image}"></div>',
                unsafe_allow_html=True,
            )

    with icon9:
        st.image("assets/icons/10.png")  # Display roboavatar on the explore page

    with icon10:
        with open("assets/icons/5.png", "rb") as f:  # Display robot avatar image
            image_data = f.read()
            encoded_image = base64.b64encode(image_data).decode()

            gif_image = st.markdown(
                f'<div class="rounded-image"><img src="data:image/png;base64,{encoded_image}"></div>',
                unsafe_allow_html=True,
            )

    with icon11:
        with open("assets/icons/2.png", "rb") as f:  # Display robot avatar image
            image_data = f.read()
            encoded_image = base64.b64encode(image_data).decode()

            gif_image = st.markdown(
                f'<div class="rounded-image"><img src="data:image/png;base64,{encoded_image}"></div>',
                unsafe_allow_html=True,
            )


with st.sidebar:
    selected_menu_item = sac.menu(
        [
            sac.MenuItem(
                "RecipeML: Home",
                icon="house-fill",
            ),
            sac.MenuItem(
                "Explore Products",
                icon="box-fill",
                children=[
                    sac.MenuItem("Recommendations", icon="boxes"),
                    sac.MenuItem("Recipe Generation", icon="stars"),
                ],
            ),
            sac.MenuItem(
                "Digital Footprints",
                icon="transparency",
            ),
            sac.MenuItem(' ', disabled=True),
            sac.MenuItem(type="divider"),
        ],
        open_all=True,
    )

with st.sidebar.form("form_subscribe_to_updates", clear_on_submit=True):
    # Request users to provide their email id's for subscribing to receive update
    user_email_id = st.text_input(
        "Subscribe now to receive latest updates:",
        placeholder="Enter your e-mail id here",
    )

    button_user_subscribed = st.form_submit_button(
        "Subscribe to Receive Updates", use_container_width=True
    )

    # Store the entered mail id of the user in the hosted database and alert user
    if button_user_subscribed:
        mail_utils = MailUtils()  # Class to validate email addresses of the user

        if mail_utils.is_valid_email(user_email_id):
            try:
                auth_tokens = AuthTokens()
                # Authentication for the Google Sheet using the gsread py package
                scope = [
                    "https://spreadsheets.google.com/feeds",
                    "https://www.googleapis.com/auth/drive",
                ]

                credentials = ServiceAccountCredentials.from_json_keyfile_name(
                    "configurations/sheets_service_account.json", scope
                )
                gc = gspread.authorize(credentials)

                sheet = gc.open_by_key(auth_tokens.subscribers_gsheets_key)
                worksheet = sheet.get_worksheet(0)  # Select the first excelsheet

                worksheet.append_row([user_email_id])
                st.toast("Thank you for subscribing!")  # Display success message

            except Exception as err:
                st.toast("Uh-Oh! We hit a snag.")
                st.toast("Please try again later")

        else:
            st.toast("Enter a valid e-mail id")  # Alert users with toast message

if selected_menu_item == "RecipeML: Home":
    # Method to display the homepage of the streamlit web app as per the defined UI.

    # This method renders the homepage for the RecipeML application, showcasing its
    # features & providing interactive links for recipe generation & recommendation.

    # Read more in the :ref:`RecipeML v1: User Interface and Functionality Overview.

    # .. versionadded:: 1.3.0

    # Returns:
    #     The method renders the homepage for the RecipeML application on streamlit.

    title_section, links_section = st.columns([2, 1])

    # Display the title on the web app's frontend and show the interactive button
    with title_section:
        st.markdown(
            "<H2>Welcome to RecipeML <font size=5>v1.2</font> 👨‍🍳🍳</H2>",
            unsafe_allow_html=True,
        )

    with links_section:
        st.write(" ")
        st.write(" ")

        st.markdown(
            "<p align='right'><B><A href='https://discuss.streamlit.io/t/recipeml-an-early-experiment-in-using-generative-al-to-transform-the-way-we-cook/59767' class='custom-link'>Join the Discussion</A></B></p>",
            unsafe_allow_html=True,
        )

    st.markdown(
        "<p align='justify'>Dive into the world of RecipeML and discover endless possibilities for your kitchen adventures. Explore how our innovative AI technology transforms your cooking experience, from personalized recipe recommendations to novel creations in just few minutes</p>",
        unsafe_allow_html=True,
    )

    # Display the carousel column on the webapp with features and call to actions
    (
        recipe_generation_section,
        recommendations_section,
        developer_connect_section,
    ) = st.columns(3)

    # Display the section for recipe generation with a link to access the web app
    with recipe_generation_section:
        st.image("assets/carosel/1.png")

        st.markdown(
            """
            <B>Tensorflow</B> • Model
            <BR>
            <H5>Generate Recipes with AI</H5>
            <P>Transform your everyday ingredients into extra-ordinary meals with the magic of AI</P>
            <B><A href='https://recipeml-generation.streamlit.app/' class="custom-link">Try Recipe Generation →</A></B>
            """,
            unsafe_allow_html=True,
        )

    # Display the section for recipe recommendation with a link to access the app
    with recommendations_section:
        st.image("assets/carosel/2.png")

        st.markdown(
            """
            <B>Featurespace</B> • Model
            <BR>
            <H5>Recipe Recommendations</H5>
            <P>Explore delectable dishes, curated from a vast library of over 2M culinary creations!</P>
            <B><A href='https://recipeml-recommendations.streamlit.app/' class="custom-link">Try Recommendations →</A></B>
            """,
            unsafe_allow_html=True,
        )

    # Displays section to connect with the developer with a link to their socials
    with developer_connect_section:
        st.image("assets/carosel/3.png")

        st.markdown(
            """
            <B>LinkedIn</B> • Socials
            <BR>
            <H5>Connect with Developer</H5>
            <P>Want to connect with me?! I'm always up for discussing about all things delicious!!</P>
            <B><A href='https://www.linkedin.com/in/thisisashwinraj/' class="custom-link">Connect with Developer →</A></B>
            """,
            unsafe_allow_html=True,
        )

if selected_menu_item == "Recommendations":
    # Method to display the recommendation page of the webapp as per the defined UI.

    # The method renders the recommendation page for the application showcasing how
    # RecipeML generates recommendations for the user, based on the i/p ingredients.

    # Read more in the :ref:`RecipeML v1: User Interface and Functionality Overview.

    # .. versionadded:: 1.3.0

    # Returns:
    #     The method renders the recommendation for the RecipeML application on web.

    st.markdown(
        """
        <style>
        .rounded-image img {
            border-radius: 5px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    title_section, links_section = st.columns([2, 1])

    # Display the title on the web app's frontend and show the interactive button
    with title_section:
        st.markdown(
            "<H2>RecipeML Recommendations</H2>",
            unsafe_allow_html=True,
        )

    with links_section:
        st.write(" ")
        st.write(" ")

        st.markdown(
            "<p align='right'><B><A href='https://recipeml-recommendations.streamlit.app/' class='custom-link'>Try Recommendations →</A></B></p>",
            unsafe_allow_html=True,
        )

    # Display subheading giving snapshot of functionality & display detailed info
    st.markdown(
        "<H4>Start by describing few ingredients and unlock delicious possibilities with RecipeML</H4>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p align='justify'>Whether it’s that leftover bag of spinach or your fridge begging for rescue, RecipeML transforms ordinary ingredients into extra-ordinary dishes. Simply describe what you have on hand & RecipeML will serve up delicious recommendation from over 2M recipe </p>",
        unsafe_allow_html=True,
    )

    display_robomojis()
    st.markdown(
        "<p align='justify'>RecipeML recommends delicious recipes to users based on the ingredients provided by the user. It provides detailed insights into each recipe, including ingredients, step-by-step directions, preparation time, recipe source, and more. The model is trained on a dataset of over 2 million recipes sourced from various recipe websites across the web & uses Google PaLM2s semantic capabilities</p>",
        unsafe_allow_html=True,
    )

    # Display the app's warning message to user about using RecipeML with caution
    usage_caution_message = """
    **Enjoy the wordplay, but cook with caution!**

    Recipes generated by RecipeML are intended for creative exploration only! These results may not always be safe, accurate, or edible! You may use it to spark inspiration but always consult trusted sources for reliable cooking information. Visit FSIS [here](https://www.fsis.usda.gov/wps/portal/fsis/topics/food-safety-education)
    """
    st.info(usage_caution_message)

    st.markdown(
        "<h4>How the Recommendations System Work?!</h4>", unsafe_allow_html=True
    )
    st.markdown(
        "<p align='justify'>RecipeML casts the process of conditional recipe recommendation as a feature space-matching algorithm that generates the recipe recommendations based on the cosine similarity between the input ingredients provided by the user and the ingredients required for preparing each recipe. Dataset verticals add further context to the corpus for generating optimized recommendations</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p align='justify'>The raw dataset exhibited a significant amount of redundant information, necessitating the implementation of pre-processing algorithms to extract the ingredients, and sanitize the dataset for recipe recommendation. Punctuation marks, stop words, accent marks using unicode std, duplicate records & non-alphabetic characters were eliminated and common kitchen terms substituted!</p>",
        unsafe_allow_html=True,
    )

    major_shot, minor_shot = st.columns([2.1, 1])

    # Displays screenshots of the major user interfaces, of the given application
    with major_shot:
        st.image("assets/demos/Recipe Recommendation.png")

    with minor_shot:
        st.image("assets/demos/recommendation_welcome_1.png")
        st.image("assets/demos/recommendation_welcome_2.png")

    st.markdown(
        "<p align='justify'>Analytical evaluations utilizing diverse ingredient combinations unequivocally show that the learning algorithms using TF/IDF Embedded Vectorizer outperforms the Mean Embedded Vectorizer. Further, empirical evidence substantiates that these natural language algorithms, and NLU model can be conditioned to accommodate numerous culinary traditions from across geographies</p>",
        unsafe_allow_html=True,
    )

    # Display the footer section with the latest, update and call for sponsorship
    st.markdown(
        "<h3 align='justify'>Elevate your cooking game, discover new flavors & redefine your kitchen escapades with RecipeML, now available across all supported countries!</h3>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p align='justify'>Share our passion for food? Become a RecipeML sponsor and support our mission to bring joy and flavor to every table! Visit our <A href='https://www.buymeacoffee.com/ThisIsAshwinRaj' style='color: #64ABD8;'>sponsorship page</a> and become a partner in unlocking endless culinary possibilities with RecipeML. A portion of the sponsorship amount directly supports initiatives fighting hunger. Together let's redefine cooking and nourish not just bodies, but also dreams!</p>",
        unsafe_allow_html=True,
    )
    st.markdown("<BR>", unsafe_allow_html=True)

if selected_menu_item == "Recipe Generation":
    # Method to display the recipe generation page of the app as per the defined UI.

    # The method renders the recipe generation page for the web app, showcasing how
    # RecipeML generates uniqe recipes for the user, based on the input by the user.

    # Read more in the :ref:`RecipeML v1: User Interface and Functionality Overview.

    # .. versionadded:: 1.3.0

    # Returns:
    #     This method renders the recipe generation page for the application on web.

    st.markdown(
        """
        <style>
        .rounded-image img {
            border-radius: 5px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    title_section, link_section = st.columns([2, 1])

    # Display the title on the web app's frontend and show the interactive button
    with title_section:
        st.markdown(
            "<H2>RecipeML Recipe Generation</H2>",
            unsafe_allow_html=True,
        )

    with link_section:
        st.write(" ")
        st.write(" ")

        st.markdown(
            "<p align='right'><B><A href='https://recipeml-generation.streamlit.app/' class='custom-link'>Try Recipe Generaion →</A></B></p>",
            unsafe_allow_html=True,
        )

    # Display subheading giving snapshot of functionality & display detailed info
    st.markdown(
        "<H4>Tell us what you've on hand or what you are craving & RecipeML will work its magic!</H4>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<p align='justify'>Dreaming of a dish but dont have the recipe to get started? Or maybe you've got an ingredient begging to be transformed? Simply describe what you are craving for or what you've on hand & RecipeML will generate a mouthwatering recipe that is uniquely yours</p>",
        unsafe_allow_html=True,
    )

    display_robomojis()

    st.markdown(
        "<p align='justify'>RecipeML taps into its deep understanding of language generation, utilizing multi-layered LSTM RNN architecture, coupled with PaLM's semantic parsing capabilities, to generate novel recipes from scratch. Further, it integrates generative models based on RunwayML and PlaygroundAIs Stable Diffusion and OpenAI DALLE2 to translate these textual creations into photo-realistic images</p>",
        unsafe_allow_html=True,
    )

    # Display the app's warning message to user about using RecipeML with caution
    usage_caution_message = """
    **Enjoy the wordplay, but cook with caution!**

    Recipes generated by RecipeML are intended for creative exploration only! These results may not always be safe, accurate, or edible! You may use it to spark inspiration but always consult trusted sources for reliable cooking information. Visit FSIS [here](https://www.fsis.usda.gov/wps/portal/fsis/topics/food-safety-education)
    """
    st.info(usage_caution_message)

    st.markdown(
        "<h4>How RecipeML Generates Recipes with AI?!</h4>", unsafe_allow_html=True
    )
    st.markdown(
        "<p align='justify'>RecipeML effectively integrates character-level Long Short-Term Memory RNN tensorflow architecture with Google's Pathways Language Model (PaLM API) for recipe generation, presenting users with a dual paradigm, allowing for recipe synthesis based on either the specified recipe names, or a singular start ingredient, that can be seleced from a curated list of over 20,000+ ingredients</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p align='justify'>Recipe generation using a start ingredient involves the use of LSTM Recurrent Neural Network (RNN), trained on a comprehensive 125k+ recipe corpus. The RNN architecture employs LSTM cells to model long range dependencies within ingredient sequences and capture the sequential nature of recipe instructions. RNNs utilizes the network's ability to retain and propagate context across sequential data facilitating the generation of contextualy relevant recipes imbued with syntactic and semantic consistency</p>",
        unsafe_allow_html=True,
    )

    # Displays screenshots of the major user interfaces, of the given application
    major_shot, minor_shot = st.columns([2.1, 1])

    with major_shot:
        st.image("assets/demos/Recipe Recommendation.png")

    with minor_shot:
        st.image("assets/demos/recommendation_welcome_1.png")
        st.image("assets/demos/recommendation_welcome_2.png")

    st.markdown(
        "<p align='justify'>The name-based recipe generation paradigm integrates the PaLM API, a 540-billion parameter, dense decoder-only Transformer model with multimodal capabilities to generate recipes based on designated names. The integration of PaLM adds a layer of semantic intelligence to the model, ensuring that the generated recipes not only adhere to syntactic structures, but also encapsulates the intended culinary themes, associated with the designated recipe names, that are provided as input by the users</p>",
        unsafe_allow_html=True,
    )

    # Display the footer section with the latest, update and call for sponsorship
    st.markdown(
        "<h3 align='justify'>Elevate your cooking game, discover new flavors & redefine your kitchen escapades with RecipeML, now available across all supported countries!</h3>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p align='justify'>Share our passion for food? Become a RecipeML sponsor and support our mission to bring joy and flavor to every table! Visit our <A href='https://www.buymeacoffee.com/ThisIsAshwinRaj' style='color: #64ABD8;'>sponsorship page</a> and become a partner in unlocking endless culinary possibilities with RecipeML. A portion of the sponsorship amount directly supports initiatives fighting hunger. Together let's redefine cooking and nourish not just bodies, but also dreams!</p>",
        unsafe_allow_html=True,
    )
    st.markdown("<BR>", unsafe_allow_html=True)

if selected_menu_item == "Digital Footprints":
    # Method to display the T&C page of the streamlit web app as per the defined UI.

    # This method renders the Terms and conditions page of the RecipeML's streamlit
    # app displaying the privacy policy, terms of use and vulnerability report form.

    # Read more in the :ref:`RecipeML v1: User Interface and Functionality Overview.

    # .. versionadded:: 1.3.0

    # Returns:
    #     The method renders the T&C page for the RecipeML application on streamlit.

    st.markdown(
        "<H2>RecipeML Privacy Policy</H2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<P align='justify'>Safety starts with understanding how we collect and share your data. We believe that responsible innovation doesn't happen in isolation. As part of our commitment to consistently improve our language algorithms, your usage information and feedback will be collected and further used to fine-tune our algorithms, creating a personalized and enjoyable cooking experience for everyone</P>",
        unsafe_allow_html=True,
    )

    # Display quick view of the data that we collect from the user & what we dont
    st.markdown(
        "<P align='justify'><B>•&nbsp&nbsp&nbsp What we collect:</B> We collect the ingredients you choose, your feedback on the model outcomes and other basic app usage data<BR><B>•&nbsp&nbsp&nbsp What we dont:</B> We do not share your sensitive personal information with any third party for marketing or advertising purposes</P>",
        unsafe_allow_html=True,
    )

    # Display the app's warning message to user about using RecipeML with caution
    usage_caution_message = """
    **Enjoy the wordplay, but cook with caution!**

    Recipes generated by RecipeML are intended for creative exploration only! These results may not always be safe, accurate, or edible! You may use it to spark inspiration but always consult trusted sources for reliable cooking information. Visit FSIS [here](https://www.fsis.usda.gov/wps/portal/fsis/topics/food-safety-education)
    """
    st.warning(usage_caution_message)

    st.markdown(
        "<P align='justify'>Should you ever wish to discontinue your participation, we encourage you to drop a line at thisisashwinraj@gmail.com. Your privacy and preferences matter and we want to ensure your experience aligns with your comfort level. We may update this privacy policy from time to time. Changes will be posted on this page and you'll be notified. Connect with us in case of doubts or concerns</P>",
        unsafe_allow_html=True,
    )

    # Display the terms of use that the users are subject to while using RecipeML
    st.markdown(
        "<H2>RecipeML Terms of Use</H2>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<P align='justify'>These Terms of Use govern your access and use of RecipeML's hosted services. By accessing or using the Services, you agree to be bound by these Terms. We encourage you to read the full Terms of Use carefully. If case of any doubts/questions, kindly contact us</P>",
        unsafe_allow_html=True,
    )

    terms_of_use_part_1 = """
    <P align='justify'>
    <B>1. App Access & Usage Guidelines</B><BR>
    While this App is accessible to users of all ages, individuals under the age of 12 should however seek parental or guardian consent before utilizing this app. You may access and use RecipeML for personal, non-commercial purposes in a responsible & respectfull manner. You shall also avoid using any content that may be deemed illegal, harmful, offensive, and/or violates the rights of others
    </P>
    <P align='justify'>
    <B>2. User Feedbacks and Disclaimer</B><BR>
    By providing feedback, you grant us a non-exclusive, royalty-free license to use your feedback for any purpose. Should you wish to discontinue your participation, we encourage you to reach out to us via email. RecipeML is provided without warranties of any kind. We don't guarantee the accuracy, completeness, or reliability of the results generated and shall not be liable for any damage
    </P>
    <P align='justify'>
    <B>3. Changes to Terms & Termination</B><BR>
    We may update these Terms of Use at any time. Any changes will be posted on this page, and your continued use of the app constitutes your agreement to the updated terms.
    We may further terminate your access at any time, if found to be in violation to these terms or for any other reasons, with or without a notice. Upon termination or discontinuation, your right to use RecipeML will immediately cease and you must cease all use of our public services & delete any stored data or information associated with it
    </P>
    <P align='justify'>
    These Terms of Use will be governed by and construed in accordance with the laws of the Republic of India. If you have any question or concerns about any points mentioned in our Terms of Use please contact us over email at thisisashwinraj@gmail.com
    </P>
    """
    st.markdown(terms_of_use_part_1, unsafe_allow_html=True)

    # Display Bug Report form for users to submit bug report to be shared as mail
    st.markdown(
        "<H2>Report a Bug in RecipeML</H2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<P align='justify'>If you have discovered any vulnerability or software bug in RecipeML, we encourage you to submit a bug report via the form below with a thorough explanation of the vulnerability. We will revert back to you within 4 days after the due diligence of your bug report</P>",
        unsafe_allow_html=True,
    )

    with st.form("form_bug_report"):
        form_section_1, form_section_2 = st.columns(2)

        with form_section_1:
            user_full_name = st.text_input(
                "Full Name",
            )

            # Display sidebar with selectbox for users to choose the app with bug
            page_with_bug = st.selectbox(
                "Which page is the bug in?",
                ["Recommendation", "Recipe Generation", "Image Generation"],
                index=None,
            )

        with form_section_2:
            user_email_id = st.text_input(
                "Email Id",
            )

            bug_types = (
                "General Bug/Error",
                "Access Token/API Key Disclosure",
                "Memory Corruption",
                "Database Injection",
                "Code Execution",
                "Denial of Service",
                "Privacy/Authorization",
            )
            bug_type = st.selectbox(
                "What type of bug is it?",
                bug_types,
                index=None,
            )

        # Creates textarea where users describe the bug and steps to reproduce it
        bug_description = st.text_area(
            "Describe the issue in detail (include steps to reproduce the issue):"
        )

        # File uploader widget is set to not accept multiple files (limit: 200mb)
        uploaded_files = st.file_uploader(
            "Include any relevant attachments such as screenshots, or reports:",
            accept_multiple_files=False,
        )

        # Checkbox that indicate that user accepts RecipeMLs terms and conditions
        bug_report_terms_and_conditions = st.checkbox(
            "I accept the terms and conditions, and I consent to be contacted in future by the RecipeML team regaring my bug report"
        )

        (
            button_section,
            _,
            _,
        ) = st.columns(3)

        with button_section:
            button_send_bug_report = st.form_submit_button(
                "Submit Bug Report", use_container_width=True
            )

        if button_send_bug_report:
            mail_utils = MailUtils()

            if mail_utils.is_valid_email(user_email_id):
                if page_with_bug:

                    if bug_description:
                        if bug_report_terms_and_conditions:
                            # Display message indicating bug report has been sent
                            try:
                                mail_utils.send_bug_report(
                                    user_full_name,
                                    user_email_id,
                                    page_with_bug,
                                    bug_type,
                                    bug_description,
                                    uploaded_files,
                                )

                            except: pass

                            bug_report_sent_alert = st.success(
                                "Your bug report has been sent!", icon="✅"
                            )
                            
                            time.sleep(3)
                            bug_report_sent_alert.empty()  # Remove alert message

                        else:
                            st.toast("Please accept the terms and conditions")
                    else:
                        st.toast("Please describe the bug")

                else:
                    st.toast("Select the page with the bug")
            else:
                st.toast("Enter a valid e-mail id")

    st.markdown("<BR>", unsafe_allow_html=True)

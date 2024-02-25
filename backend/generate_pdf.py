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
This module contains the code for generating the cuisine's recipe in a PDF format.
The PDFUtils class holds the methods forgenerating the PDF file using FPDF module.
The default style for the PDF was set to reflect the Arial fontface & 12 fontsize.

Module Functions:
    [1] PDFUtills (class)
        [a] generate_recommendations_pdf

.. versionadded:: 1.0.0
.. versionupdated:: 1.1.0

Learn about RecipeML :ref:`RecipeML v1: User Interface and Functionality Overview`
"""

import re
import ast
import pandas as pd
import cProfile

import sys
import logging
import datetime
from fpdf import FPDF

from configurations.resource_path import ResourceRegistry


class PDFUtils:
    '''
    Class to generate the recipe files in PDF format and store in ~./generate_pdf.

    This class provides utility methods for generating PDF documents that contain
    recipe recommendations. It offers a method to create PDFs with recipe details
    including the recipe's name, its source url major ingredients, and directions.

    Class Methods:
        [1] generate_recommendations_pdf

    .. versionadded:: 1.1.0

    NOTE: You can customize the PDFs appearance, including font, style and layout, 
    to match your preferred design. The title and content are easily configurable.
    '''

    def __init__(self):
        pass

    def generate_recommendations_pdf(
        self, recipe_name, recipe_source, recipe_link, ingredients_list, directions_list
    ):
        """
        Method to generate a PDF file with recipe details and return its location.

        This method reads recipe details from the user and generates its PDF copy.
        The PDF file includes the recipe name, source, ingredients, directions, & 
        the additional information including the terms of use & other information.

        Read more in the :ref:`RecipeML:User Interface and Functionality Overview`

        .. versionadded:: 1.1.0

        Parameters:
            [string] recipe_name: Full name of recipe to be printed as PDFs title
            [string] recipe_source: The source of the recipe - Gathered/Recipes1M
            [string] recipe_link: The source link of recipe's site to be embedded
            [string or list] ingredients_list: List of ingredients used in recipe
            [string] directions_list: The list containing each step of the recipe

        Returns:
            [string] file_location: Directory wherein the generated file is saved

        NOTE: The PDF file is saved to the exports/generated_pdf directory with a 
        file name based on recipe. This can be changed by modifying file_location
        """
        try:
            # Attempt to parse ingredients as a list if parameter is of dtype str
            ingreds = ast.literal_eval(ingredients_list)
            ingredients = ", ".join(ingreds)
        except:
            # If not list use original input
            ingredients = str(ingredients_list)

        directions = ast.literal_eval(
            directions_list)  # Parse direction as list

        pdf = FPDF()  # Initialize PDF document as variable, using the FPDF class

        # Add new page & display the recipe name as the title of the PDF document
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.set_title(recipe_name)

        # Set font style as Arial and set the recipe name as the document heading
        pdf.set_font("Arial", style="B", size=24)
        pdf.cell(0, 8, txt=recipe_name, align="L", ln=True)

        pdf.set_font("Arial", size=6)
        # Add a line for visual break
        pdf.cell(0, 6, txt="", align="L", ln=True)

        # Display the ingredients heading in bold, & print the recipe ingredients
        pdf.set_font("Arial", style="B", size=12)
        pdf.cell(0, 10, txt="Ingredients", align="L", ln=True)
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 5, txt=ingredients.capitalize(), align="J")

        pdf.set_font("Arial", size=6)
        # Add a blank line as per layout
        pdf.multi_cell(0, 5, txt="", align="J")

        # Display the directions heading in bold, and print the recipe directions
        pdf.set_font("Arial", style="B", size=12)
        pdf.cell(0, 10, txt="Directions", align="L", ln=True)
        pdf.set_font("Arial", size=12)

        instruction = 1

        for text in directions:
            # Split the instruction
            sentences = re.split(r"(?<=[.!?])\s+", text)
            step = " ".join(sentence.capitalize() for sentence in sentences)

            # Display the processed instruction as an individual step in the list
            pdf.multi_cell(0, 5, txt=str(instruction) + ". " + step, align="J")
            instruction = instruction + 1

        # Add a line for visual break
        pdf.cell(0, 5, txt="", align="L", ln=True)

        # Display the Source heading in bold, and display the recipe's source url
        pdf.set_font("Arial", style="B", size=12)
        pdf.cell(0, 10, txt="Source", align="L", ln=True)
        pdf.set_font("Arial", size=12)

        # Embed the source url of the recipes source site into the displayed link
        pdf.cell(
            0, 5, txt=recipe_source + ": " + recipe_link, align="J", link=recipe_link
        )

        # Add a line for visualbreak
        pdf.cell(0, 12, txt="", align="L", ln=True)

        x1, x2 = 11, 198
        y = pdf.get_y()
        # Display horizontal line to seprate the sections
        pdf.line(x1, y, x2, y)

        # Add a line for visualbreak
        pdf.cell(0, 12, txt="", align="L", ln=True)

        # Display the RecipeMLv1.1 - Terms of Usage section heading in bold style
        pdf.set_font("Arial", style="B", size=18)
        pdf.cell(0, 8, txt="RecipeML v1.1 - Terms of Usage", align="L", ln=True)

        pdf.set_font("Arial", size=6)
        # Add a blank line as per layout
        pdf.multi_cell(0, 5, txt="", align="J")

        data_source_and_reliance = "This recommendation system relies on the RecipeNLG dataset, which is publicly accessible on Kaggle. Recommendations generated are derived from the information contained within this dataset."
        pdf.set_font("Arial", style="B", size=12)

        # Display the Data Source and Reliance heading in bold and print the text
        pdf.cell(0, 10, txt="Data Source and Reliance", align="L", ln=True)
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 5, txt=data_source_and_reliance, align="J")

        pdf.set_font("Arial", size=6)
        # Add blank line for visualbreak
        pdf.multi_cell(0, 5, txt="", align="J")

        recommendation_accuracy = "While we endeavor to provide accurate recipe recommendations, it's important to note that these suggestions are generated using Machine Learning and Natural Language Processing (NLP) technology. We do not guarantee the accuracy, completeness or suitability of any recommendations"
        pdf.set_font("Arial", style="B", size=12)

        # Display Disclaimer of Warranty heading in bold, and display the content
        pdf.cell(0, 10, txt="Disclaimer of Warranty", align="L", ln=True)
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 5, txt=recommendation_accuracy, align="J")

        pdf.set_font("Arial", size=6)
        # Add blank line for visualbreak
        pdf.multi_cell(0, 5, txt="", align="J")

        data_privacy_and_security = "We are committed to safeguarding your data and privacy. Please refer to our Privacy Policy to understand how we collect, use & protect your personal information (tinyurl.com/RML-PrivacyPolicy)"
        pdf.set_font("Arial", style="B", size=12)

        # Display Data Privacy and Security heading in bold and print the content
        pdf.cell(0, 10, txt="Data Privacy and Security", align="L", ln=True)
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 5, txt=data_privacy_and_security, align="J")

        pdf.set_font("Arial", size=6)
        # Add blank line for visualbreak
        pdf.multi_cell(0, 5, txt="", align="J")

        acknowledge_terms = "By using this recommendation system, you acknowledge that you have read, understood, and agreed to these Terms of Usage. These terms are subject to change, and it is your responsibility to review them periodically for updates and modifications. For queries, mail thisisashwinraj@gmail.com"
        pdf.set_font("Arial", size=12)

        # Display the text asserting user's acknowledgement to the terms of usage
        pdf.multi_cell(0, 5, txt=acknowledge_terms, align="J")

        resource_registry = ResourceRegistry()

        # Generate a filename based on the recipe name stripping extra whitespace
        file_name = recipe_name.strip().lower().replace(" ", "_").replace("/", "")
        file_location = resource_registry.generated_recipe_pdf_dir_path + file_name + ".pdf"

        # Output the PDF to the specified file location, and return file location
        pdf.output(file_location)
        return file_location


if __name__ == "__main__":
    current_date = datetime.date.today()
    month_name = current_date.strftime("%b").lower()

    logging.basicConfig(
        filename=f"validation/logs/log_record_{month_name}_{current_date.day}_{current_date.year}.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    logging.info(
        "EXECUTION INITIATED: backend.generate_pdf package started execution")

    pdf_utils = PDFUtils()

    try:
        raw_dataset_path = "data/raw/recipe_nlg_test_dataset.csv"
        logging.info(f"reading raw dataset from {raw_dataset_path}")

        recipe_data = pd.read_csv(raw_dataset_path)
        logging.info("dataset succesfully loaded into the memory")

    except FileNotFoundError as file_not_found_exception:
        print("PDF generation failed. Check logs for more details.")

        logging.error(
            f"FILE NOT FOUND: dataset not found in the working directory\n{file_not_found_exception}"
        )
        logging.critical(
            "EXECUTION TERMINATED: execution failed for backend.generate_pdf package"
        )
        logging.critical(
            "--------------------------------------------------------------------------------------"
        )

        logging.shutdown()
        sys.exit()

    except PermissionError as permission_error_exception:
        print("PDF generation failed. Check logs for more details.")

        logging.error(
            f"PERMISSION DENIED: permission denied to access the dataset\n{permission_error_exception}"
        )
        logging.critical(
            "EXECUTION TERMINATED: execution failed for backend.generate_pdf package"
        )
        logging.critical(
            "--------------------------------------------------------------------------------------"
        )

        logging.shutdown()
        sys.exit()

    except Exception as exception:
        print("PDF generation failed. Check logs for more details.")

        logging.error(
            f"UNEXPECTED ERROR: an unexpected error occurred\n{exception}")
        logging.critical(
            "EXECUTION TERMINATED: execution failed for backend.generate_pdf package"
        )
        logging.critical(
            "--------------------------------------------------------------------------------------"
        )

        logging.shutdown()
        sys.exit()

    while True:
        user_input = input("Enter recipe id between 1 to 9999: ")
        logging.info(
            f"user requesting recipe details for recipe_id {user_input}")

        if user_input.isdigit():
            recipe_id = int(user_input)

            if 1 <= recipe_id <= 9999:
                logging.info(f"recipe_id {recipe_id} validated succesfully")
                break

            else:
                print(
                    "Input out of bound. Please enter a valid integer between 1 and 9999.\n"
                )

                logging.warning(
                    "INPUT OUT OF BOUND: input value is outside the range (1-9999)"
                )
                logging.info(
                    "requesting new input for recipe_id from the user")
        else:
            print("Invalid input. Please enter a valid integer between 1 and 9999.\n")

            logging.warning(
                "BAD INPUT: invalid value for recipe_id received from the user"
            )
            logging.info("requesting new input for recipe_id from the user")

    logging.info(f"fetching recipe details for recipe_id {recipe_id}")

    try:
        recipe_name = recipe_data["title"].iloc[recipe_id]
        recipe_type = recipe_data["source"].iloc[recipe_id]
        recipe_url = recipe_data["link"].iloc[recipe_id]
        recipe_raw_ingredients = recipe_data["ingredients"].iloc[recipe_id]
        recipe_instructions = recipe_data["directions"].iloc[recipe_id]

        logging.info("recipe details collected succesfully from the dataframe")

    except IndexError as index_error_exception:
        print("PDF generation failed. Check logs for more details.")

        logging.error(
            f"INVALID RECIPE ID: Details for recipe_id {recipe_id} could not be found\n{index_error_exception}"
        )
        logging.critical(
            "EXECUTION TERMINATED: execution failed for backend.generate_pdf package"
        )
        logging.critical(
            "--------------------------------------------------------------------------------------"
        )

        logging.shutdown()
        sys.exit()

    except Exception as exception:
        print("PDF generation failed. Check logs for more details.")

        logging.error(
            f"UNEXPECTED ERROR: an unexpected error occurred\n{exception}")
        logging.critical(
            "EXECUTION TERMINATED: execution failed for backend.generate_pdf package"
        )
        logging.critical(
            "--------------------------------------------------------------------------------------"
        )

        logging.shutdown()
        sys.exit()

    try:
        logging.info(
            f"passing recipe details to PDFUtils.generate_recommendations_pdf")

        file_name = pdf_utils.generate_recommendations_pdf(
            recipe_name,
            recipe_type,
            recipe_url,
            recipe_raw_ingredients,
            recipe_instructions,
        )
        cProfile.run(
            "pdf_utils.generate_recommendations_pdf(recipe_name, recipe_type, recipe_url, recipe_raw_ingredients, recipe_instructions)",
            sort="cumulative",
            filename="validation/profile/profile_generate_pdf.txt",
        )

        print(f"PDF generated succesfully and saved at {file_name}")

        logging.info(f"file generated succesfully and saved at {file_name}")

    except Exception as exception:
        print("PDF generation failed. Check logs for more details.")

        logging.error(
            f"UNEXPECTED ERROR: an unexpected error occurred\n{exception}")
        logging.critical(
            "EXECUTION TERMINATED: execution failed for backend.generate_pdf package"
        )
        logging.critical(
            "--------------------------------------------------------------------------------------"
        )

        logging.shutdown()
        sys.exit()

    logging.info(
        "EXECUTION COMPLETE: backend.generate_pdf package executed succesfully"
    )
    logging.info(
        "--------------------------------------------------------------------------------------"
    )
    logging.shutdown()

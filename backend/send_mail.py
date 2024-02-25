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
This module contains the method for sending mails with attachments using the SMTP
protocol. It can be used to send both plain text mails, and rich text HTML5 mails
after changing payload to encoded format. This module uses ~smtp.gmail.com server 
at port 587 for sending the emails with MIMEBase payloads, and base64 attachments.

Module Functions:
    [1] MailUtils (class)
        [a] is_valid_email
        [b] send_recipe_info_to_mail

.. versionadded:: 1.0.0
.. versionupdated:: 1.1.0

Learn about RecipeML :ref:`RecipeML v1: User Interface and Functionality Overview`
"""
import re
import ast
import time
import smtplib
import cProfile

import sys
import logging
import datetime
import pandas as pd

from email import encoders
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email.mime.application import MIMEApplication

try:
    from backend import config
except:
    import config
try:
    from backend.generate_pdf import PDFUtils
except:
    from generate_pdf import PDFUtils


class MailUtils:
    """
    Class for validating mail id and sending the recommendations to user via mail

    This class provides methods for validating mail addresses & for sending movie
    information to recipients via plain-text mail on their mail id's. It requires
    access to valid mail credentials of the sender in order to deliver the e-mail.

    Mails are sent to the receiver's mail id using smtp.gmail.com server port 587.

    .. versionadded:: 1.2.0

    NOTE: Attachments can also be passed as arguments, to send a file to the user.
    """

    def __init__(self):
        pass

    def is_valid_email(self, input_mail_id):
        """
        This method is used to check if the string passed is valid mail id or not.

        Using regular expression this method checks if a given string matches the
        pattern specified for validating an e-mail address, and returns True only
        if there's match. The method returns False, if it doesn't match the regex.

        Read more in the :ref:`RecipeML:User Interface and Functionality Overview`

        .. versionadded:: 1.0.0

        Parameters:
            [string] input_mail_id: The string to be validated as a valid mail id

        Returns:
            [bool] True/False: Returns True if mail id is valid & False otherwise
        """
        # Define a regular expression for validating the mail address of the user
        pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"

        # Return a bool value indicating whether the mail address is valid or not
        return re.match(pattern, input_mail_id) is not None

    def send_recipe_info_to_mail(
        self,
        recipe_name,
        recipe_ingredients,
        recipe_instructions,
        receivers_email_id="rajashwin733@gmail.com",
        attachment=None,
    ):
        """
        Method to send recipe recommendation mail with an attachment to the users.

        This method sends a mail with recipe details to the users registered mail 
        using the smtp.gmail.com server at port 587 alongside a PDF attachment as 
        a payload. The PDF attachment is an optional argument and depends on user.

        .. versionadded:: 1.0.0
        .. versionupdated:: 1.1.0

        Parameters:
            [string] recipe_name: Full name of recipe to be printed as PDFs title
            [string] recipe_ingredients: List of ingredients to be used in recipe
            [string] recipe_instructions: List containing each step of the recipe
            [string] receivers_email_id: Email Id to which the mail shall be sent
            [file] attachment (optional arg): Attachment to be included in e-mail

        Returns:
            None -> Sends a mail with atttachment to the users registered mail id
        """
        message = MIMEMultipart()  # Create a instance of the MIMEMultipart class

        message[
            "To"
        ] = receivers_email_id  # Store the receiver's e-mail id in the To field

        message[
            "From"
        ] = config.SENDER_EMAIL_ID  # Store the senders mail id in the From field

        message[
            "Subject"
        ] = "Get Cookin'! Your Delicious Recipe Inside"  # Store the subject line

        # Store the body of the e-mail to be sent to the user in the br_mail_body
        recipe_instructions_list = ast.literal_eval(recipe_instructions)
        recipe_instructions_steps = "\n• ".join(recipe_instructions_list)

        # Parse the raw ingredients of the recipe as a python list from dtype str
        recipe_ingredients_list = ast.literal_eval(recipe_ingredients)
        recipe_ingredients = ""

        for ingredient in recipe_ingredients_list:
            ingredient = ingredient.title()  # Capitalize ingredient's each words

            # Display each ingredients within the mail as comma-seprated elements
            if len(recipe_ingredients) > 0:
                recipe_ingredients = recipe_ingredients + \
                    ", " + str(ingredient)
            else:
                recipe_ingredients = str(ingredient)

        br_mail_body = (
            "Greetings,"
            + "\n\nWe hope this email finds you in high spirits and ready to embark on a delightful culinary adventure! Here is the recipe that you requested for:"
            + "\n\nRecipe Name:\n"
            + str(recipe_name)
            + "\n\nIngredients:\n"
            + str(recipe_ingredients.title())
            + "\n\nHow to cook:\n• "
            + str(recipe_instructions_steps)
            + "\n\nTime to put on your apron, gather your kitchen tools, and let the culinary magic begin! Happy cooking and bon appétit!\n\nRegards,\nRecipeML Team"
        )

        message.attach(
            MIMEText(br_mail_body, "plain", "utf-8")
        )  # Attach the email body with the mail instance of the email.mime class

        # Check if any file is passed as an argumnent to be delivered to the user
        if attachment:
            pdf_filename = attachment
            # Read file content
            pdf_attachment = open(pdf_filename, "rb").read()

            pdf_part = MIMEBase("application", "octet-stream")
            pdf_part.set_payload(pdf_attachment)
            # Encode the PDF file, using Base64
            encoders.encode_base64(pdf_part)

            display_pdf_filename = pdf_filename.replace(
                "exports/generated_pdf/", "")
            display_pdf_filename = " ".join(
                word.capitalize() for word in display_pdf_filename.split("_")
            )  # Modify the displayed PDF file name to make it more user-friendly

            pdf_part.add_header(
                "Content-Disposition", f"attachment; filename={display_pdf_filename}"
            )
            # Attach the PDF part to the e-mail message
            message.attach(pdf_part)

        # Create an SMTP session at Port 587 & encrypt this connection using TTLS
        server = smtplib.SMTP(
            "smtp.gmail.com", 587
        )
        server.starttls()
        server.ehlo()  # Hostname to send for this command defaults to local FQDN

        # Authenticate the sender credentials before sending mail to the receiver
        server.login(config.SENDER_EMAIL_ID, config.SENDER_PASSWORD)
        text = (
            message.as_string()
        )

        server.sendmail(config.SENDER_EMAIL_ID, receivers_email_id, text)
        server.quit()  # Terminate the SMTP session after delivering mail to user


if __name__ == "__main__":
    current_date = datetime.date.today()
    month_name = current_date.strftime("%b").lower()

    logging.basicConfig(
        filename=f"validation/logs/log_record_{month_name}_{current_date.day}_{current_date.year}.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    logging.info(
        "EXECUTION INITIATED: backend.send_mail package started execution")

    mail_utils = MailUtils()

    try:
        raw_dataset_path = "data/raw/recipe_nlg_test_dataset.csv"
        logging.info(f"reading raw dataset from {raw_dataset_path}")

        recipe_data = pd.read_csv(raw_dataset_path)
        logging.info("dataset succesfully loaded into the memory")

    except FileNotFoundError as file_not_found_exception:
        print("Failed to send email. Check logs for more details.")

        logging.error(
            f"FILE NOT FOUND: dataset not found in the working directory\n{file_not_found_exception}"
        )
        logging.critical(
            "EXECUTION TERMINATED: execution failed for backend.send_mail package"
        )
        logging.critical(
            "--------------------------------------------------------------------------------------"
        )

        logging.shutdown()
        sys.exit()

    except PermissionError as permission_error_exception:
        print("Failed to send email. Check logs for more details.")

        logging.error(
            f"PERMISSION DENIED: permission denied to access the dataset\n{permission_error_exception}"
        )
        logging.critical(
            "EXECUTION TERMINATED: execution failed for backend.send_mail package"
        )
        logging.critical(
            "--------------------------------------------------------------------------------------"
        )

        logging.shutdown()
        sys.exit()

    except Exception as exception:
        print("Failed to send email. Check logs for more details")

        logging.error(
            f"UNEXPECTED ERROR: an unexpected error occurred\n{exception}")
        logging.critical(
            "EXECUTION TERMINATED: execution failed for backend.send_mail package"
        )
        logging.critical(
            "--------------------------------------------------------------------------------------"
        )

        logging.shutdown()
        sys.exit()

    validated_user_email_id = False

    while True:
        if validated_user_email_id is False:
            user_email_id = input(
                "Enter your email id (recipe will be sent to this mail): "
            )

            logging.info(f"user entered email id {user_email_id}")

            if mail_utils.is_valid_email(user_email_id):
                validated_user_email_id = True

                logging.info(f"email id {user_email_id} validated succesfully")

        if validated_user_email_id is True:
            user_input = input("Enter recipe id between 1 to 9999: ")

            logging.info(
                f"user requesting recipe details for recipe_id {user_input}")

            if user_input.isdigit():
                recipe_id = int(user_input)

                if 1 <= recipe_id <= 9999:
                    logging.info(
                        f"recipe_id {recipe_id} validated succesfully")
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
                print(
                    "Invalid input. Please enter a valid integer between 1 and 9999.\n"
                )

                logging.warning(
                    "BAD INPUT: invalid value for recipe_id received from the user"
                )
                logging.info(
                    "requesting new input for recipe_id from the user")
        else:
            print("Invalid email id: Please enter a valid email address\n")

            logging.warning(
                f"INVALID EMAIL ID: email id {user_email_id} could not be validated"
            )
            logging.info("requesting new email id from the user")

    logging.info(f"fetching recipe details for recipe_id {recipe_id}")

    try:
        recipe_name = recipe_data["title"].iloc[recipe_id]
        recipe_type = recipe_data["source"].iloc[recipe_id]
        recipe_url = recipe_data["link"].iloc[recipe_id]
        recipe_raw_ingredients = recipe_data["ingredients"].iloc[recipe_id]
        recipe_instructions = recipe_data["directions"].iloc[recipe_id]

        logging.info("recipe details collected succesfully from the dataframe")

        try:
            logging.info(
                "passing recipe details to backend.generate_pdf.PDFUtils.generate_recommendations_pdf"
            )

            pdf_utils = PDFUtils()
            recipe_file_name = pdf_utils.generate_recommendations_pdf(
                recipe_name,
                recipe_type,
                recipe_url,
                recipe_raw_ingredients,
                recipe_instructions,
            )
            cProfile.run(
                "pdf_utils.generate_recommendations_pdf(recipe_name, recipe_type, recipe_url, recipe_raw_ingredients, recipe_instructions)",
                sort="cumulative",
                filename="validation/profile/profile_send_mail.txt",
            )

            logging.info("pdf file wth recipe details generated succesfully")

        except Exception as exception:
            print("Unable to generate PDF file. Sending mail without attachment.")
            logging.error(
                f"FILE GENERATION ERROR: failed to generate recipe file\n{exception}"
            )

            recipe_file_name = None
            logging.warning("sending mail without file attachment")

    except IndexError as index_error_exception:
        print("Failed to send email. Check logs for more details.")

        logging.error(
            f"INVALID RECIPE ID: Details for recipe_id {recipe_id} could not be found\n{index_error_exception}"
        )
        logging.critical(
            "EXECUTION TERMINATED: execution failed for backend.send_mail package"
        )
        logging.critical(
            "--------------------------------------------------------------------------------------"
        )

        logging.shutdown()
        sys.exit()

    except Exception as exception:
        print("Failed to send email. Check logs for more details.")

        logging.error(
            f"UNEXPECTED ERROR: an unexpected error occurred\n{exception}")
        logging.critical(
            "EXECUTION TERMINATED: execution failed for backend.send_mail package"
        )
        logging.critical(
            "--------------------------------------------------------------------------------------"
        )

        logging.shutdown()
        sys.exit()

    try:
        logging.info(
            f"passing recipe details to MailUtils.send_recipe_info_to_mail")

        mail_utils.send_recipe_info_to_mail(
            recipe_name,
            recipe_raw_ingredients,
            recipe_instructions,
            user_email_id,
            recipe_file_name,
        )
        cProfile.run(
            "mail_utils.send_recipe_info_to_mail(recipe_name, recipe_raw_ingredients, recipe_instructions, user_email_id, recipe_file_name)",
            sort="cumulative",
            filename="validation/profile/profile_send_mail.txt",
        )

        print(f"Mail with {recipe_name} recipe delivered to {user_email_id}")

        logging.info(
            f"mail with {recipe_name} recipe delivered to {user_email_id}")

    except Exception as exception:
        print("Failed to send email. Check logs for more details.")

        logging.error(
            f"UNEXPECTED ERROR: an unexpected error occurred\n{exception}")
        logging.critical(
            "EXECUTION TERMINATED: execution failed for backend.send_mail package"
        )
        logging.critical(
            "--------------------------------------------------------------------------------------"
        )

        logging.shutdown()
        sys.exit()

    logging.info(
        "EXECUTION COMPLETE: backend.send_mail package executed succesfully")
    logging.info(
        "--------------------------------------------------------------------------------------"
    )
    logging.shutdown()

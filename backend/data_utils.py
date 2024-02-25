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
This module contains the classes for data wrangling and fundamental preprocessing.
Methods with in each class are designed to clean and manipulate data for improved 
analysis and usability. The classes are initialized without passing any parameter.

Depending on individual cases, the program may be modified to use multiprocessing 
capailities to increase the speed of data processing, on eligible local computers.
The usage of each class & their methods are described in corresponding docstrings.

Classes and Functions:
    [1] DataWrangling (class)
        [a] remove_duplicate_records
        [b] remove_punctuations_and_whitespaces
        [c] remove_whitespace_and_duplicates

    [2] CorpusData (class)
        [a] convert_list_to_string
        [b] convert_string_to_list
        [c] lemmatize_and_remove_stop_words

.. versionadded:: 1.0.0
.. versionupdated:: 1.1.0

Learn about RecipeML :ref:`RecipeML v1: DataWrangling & Fundamental PreProcessing`
"""

import sys
import logging
import datetime
import cProfile

import re
import ast
import string
import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from multiprocessing import Pool, cpu_count

nltk.download("punkt")
nltk.download("wordnet")
nltk.download("stopwords")


class DataWrangling:
    """
    Class to perform DataWrangling operations to preprocess the RecipeNLG dataset.

    This class provides methods for cleaning & preprocessing data. It is designed 
    to handle various data preprocessing tasks such as removing duplicate records, 
    punctuations, leading & trailing whitespaces, duplicates & special charecters.

    Class Methods:
        [1] remove_duplicate_records
        [2] remove_punctuations_and_whitespaces
        [3] remove_whitespace_and_duplicates

    .. versionadded:: 1.1.0

    The performance of the methods present in the class can be optimized by using
    the CPUPool via the multithreading capailities on eligible local/cloud system.
    """
    def __init__(self):
        pass

    def remove_duplicate_records(self, data, subset="title"):
        """
        Method to remove duplicate records from raw data, keeping the first as is

        The method removes duplicate records from the dataset based on the subset
        of column. It retains the first occurrence of each duplicate record while 
        eliminating subsequent duplicates. Returns I/P data if no duplicate found.

        Read more in the :ref:`RecipeML:DataWrangling & Fundamental PreProcessing`

        .. versionadded:: 1.1.0

        Parameters:
            [dataframe] data: Input data containing potentially duplicate records
            [string] subset: Field considered for duplicate. Default set to title

        Returns:
            [dataframe] data: Output dataset with duplicates removed basis subset
        """
        # Find all the duplicate records in dataset based on the specified subset
        duplicate_records_count = data[data.duplicated(
            subset=subset, keep="first")]

        # Check if duplicate records were found in the dataframe basis the subset
        if len(duplicate_records_count) > 0:
            # Remove all the found duplicate records keeping the first occurrence
            df_no_duplicates = data.drop_duplicates(
                subset=subset, keep="first")

            return df_no_duplicates

        else:
            return data  # If no duplicates were found, return original dataframe

    def remove_punctuations_and_whitespaces(self, input_text):
        """
        Method to remove punctuations and extraneous whitespaces from the dataset

        The method removes punctuations and leading and trailing whitespaces from 
        the input string, thus preparing the text data for NLP analysis & natural 
        language processing, and other text-based tasks including recommendations.

        Read more in the :ref:`RecipeML:DataWrangling & Fundamental PreProcessing`

        .. versionadded:: 1.1.0

        Parameters:
            [string] input_text: Input str with punctuation symbols & whitespaces

        Returns:
            [string] cleaned_text: Text without punctuation symbol or whitespaces
        """
        # Create a translation table to remove all the punctuations from the text
        translator = str.maketrans("", "", string.punctuation)

        # Remove punctuation symbols from the input text, basis translation table
        translated_input_text = input_text.translate(translator)
        cleaned_text = " ".join(translated_input_text.split())

        return cleaned_text

    def remove_whitespace_and_duplicates(self, input_ingredients_list):
        """
        Method to remove duplicate ingredients and leading & trailing whitespaces

        This method processes a list of input ingredients, removing all duplicate 
        entries, trailing whitespaces, & filtering special characters. It returns 
        a list of ingredients with duplicates eliminated, and improved uniformity.

        Read more in the :ref:`RecipeML:DataWrangling & Fundamental PreProcessing`

        .. versionadded:: 1.1.0

        Parameters:
            [list] input_ingredients_list: The list of ingredients, to be cleaned

        Returns:
            [string] cleaned_ingredients_list: Cleaned list of output ingredients
        """
        unique_ingredients = set()
        cleaned_ingredients_list = []

        for ingredient in input_ingredients_list:
            # Convert each string to lowercase and remove the trailing whitespace
            cleaned_ingredient = ingredient.lower().strip()
            cleaned_ingredient = re.sub(r"[^\w\s]", "", cleaned_ingredient)

            if cleaned_ingredient not in unique_ingredients:
                # Check if the cleaned ingredient is unique, & add it to the list
                cleaned_ingredients_list.append(cleaned_ingredient)
                unique_ingredients.add(cleaned_ingredient)

        return cleaned_ingredients_list


class CorpusData:
    """
    Class to preprocess corpus data including recipe instructions and ingredients

    This class provides methods for cleaning and preprocessing corpus data. It is 
    designed to handle various data preprocessing task such as conversion between 
    lists and strings, as well as lemmatizing and removing stop words from corpus

    Class Methods:
        [1] convert_list_to_string
        [2] convert_string_to_list
        [3] lemmatize_and_remove_stop_words

    .. versionadded:: 1.0.0
    .. versionupdated:: 1.1.0

    The performance of the methods present in the class can be optimized by using
    the CPUPool via the multithreading capailities on eligible local/cloud system.
    """
    def __init__(self):
        pass

    def convert_list_to_string(self, input_list):
        """
        Method to convert the list of input strings into a space-separated string

        This method converts the list of input string to a space-separated string.
        The performance of this method can be further optimized, by making use of
        the CPUPool via multithreading capailities on eligible local/cloud system.

        Read more in the :ref:`RecipeML:DataWrangling & Fundamental PreProcessing`

        .. versionadded:: 1.1.0

        Parameters:
            [list] input_list: The list of ingredients, to be converted to string

        Returns:
            [string] space_separated_string: Space-separated string of input list
        """
        space_separated_string = " ".join(input_list)  
        return space_separated_string  # Generate a space-seprated str using list

    def convert_string_to_list(self, input_string):
        """
        Method to convert any space-separated string into a list of input strings

        This method converts any space-separated string to a list of input string.
        The performance of this method can be further optimized, by making use of
        the CPUPool via multithreading capailities on eligible local/cloud system.

        Read more in the :ref:`RecipeML:DataWrangling & Fundamental PreProcessing`

        .. versionadded:: 1.1.0

        Parameters:
            [string] input_string: Space-separated str, to be evaluated to string

        Returns:
            [list] output_list: The list of string evaluated from an input string
        """
        return ast.literal_eval(input_string)  # Convert the string to pythonlist

    def lemmatize_and_remove_stop_words(self, text):
        """
        Method to lemmatize the text and remove all stopwords from the input text.

        This method tokenizes the text, lemmatizes the words, & removes stopwords.
        The performance of this method can be further optimized, by making use of
        the CPUPool via multithreading capailities on eligible local/cloud system.

        Read more in the :ref:`RecipeML:DataWrangling & Fundamental PreProcessing`

        .. versionadded:: 1.1.0

        Parameters:
            [string] text: Input string to be tokenized, lemmatized and processed

        Returns:
            [string] cleaned_text: Text with words lemmatized & stopwords removed
        """
        stop_words = set(stopwords.words("english"))  # Create a set of stopwords

        lemmatizer = WordNetLemmatizer()  # Initialize an NLTK WordNet lemmatizer
        words = nltk.word_tokenize(text)  # Tokenize the input strings into words

        # Lemmatize each word in the tokenized text, and convert to its root form
        lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
        cleaned_text = [
            word for word in lemmatized_words if word not in stop_words]

        return " ".join(cleaned_text)  # Return cleaned text by joining the words


if __name__ == "__main__":
    current_date = datetime.date.today()
    month_name = current_date.strftime("%b").lower()

    logging.basicConfig(
        filename=f"validation/logs/log_record_{month_name}_{current_date.day}_{current_date.year}.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    logging.info(
        "EXECUTION INITIATED: backend.data_utils package started execution")

    corpus_data = CorpusData()
    data_wrangling = DataWrangling()

    try:
        raw_dataset_path = "data/raw/recipe_nlg_test_dataset.csv"
        logging.info(f"reading raw dataset from {raw_dataset_path}")

        recipe_data = pd.read_csv(raw_dataset_path)
        logging.info("dataset succesfully loaded into the memory")

    except FileNotFoundError as file_not_found_exception:
        print("Preprocessing failed. Check logs for more details.")

        logging.error(
            f"FILE NOT FOUND: dataset not found in the working directory\n{file_not_found_exception}"
        )
        logging.critical(
            "EXECUTION TERMINATED: execution failed for backend.data_utils package"
        )
        logging.critical(
            "--------------------------------------------------------------------------------------"
        )

        logging.shutdown()
        sys.exit()

    except PermissionError as permission_error_exception:
        print("Preprocessing failed. Check logs for more details.")

        logging.error(
            f"PERMISSION DENIED: permission denied to access the dataset\n{permission_error_exception}"
        )
        logging.critical(
            "EXECUTION TERMINATED: execution failed for backend.data_utils package"
        )
        logging.critical(
            "--------------------------------------------------------------------------------------"
        )

        logging.shutdown()
        sys.exit()

    except Exception as exception:
        print("Preprocessing failed. Check logs for more details.")

        logging.error(
            f"UNEXPECTED ERROR: an unexpected error occurred\n{exception}")
        logging.critical(
            "EXECUTION TERMINATED: execution failed for backend.data_utils package"
        )
        logging.critical(
            "--------------------------------------------------------------------------------------"
        )

        logging.shutdown()
        sys.exit()

    print(f"Preprocessing the raw dataset loaded from {raw_dataset_path}")

    try:
        logging.info("analyzing dataset for duplicate records basis title")
        duplicate_records_count = len(
            recipe_data[recipe_data.duplicated(subset="title", keep="first")]
        )

        logging.info(
            f"found {str(duplicate_records_count)} duplicate records in the dataset"
        )

        try:
            if duplicate_records_count > 0:
                recipe_data = data_wrangling.remove_duplicate_records(
                    recipe_data)
                cProfile.run(
                    "data_wrangling.remove_duplicate_records(recipe_data)",
                    sort="cumulative",
                    filename="validation/profile/profile_data_utils.txt",
                )

                logging.info(
                    f"{str(recipe_data.shape[0])} records remaining after removing duplicates"
                )

        except Exception as exception:
            print(
                "Failed to remove duplicate records from the dataset. Data integrity may be affected."
            )

            logging.error(
                f"UNEXPECTED ERROR: an unexpected error occurred\n{exception}"
            )
            logging.warning(
                "failed to remove duplicate records from the dataset")

    except KeyError as key_error_exception:
        print(
            "Failed to detect duplicate records. Preprocessing the dataset without duplicate record verification"
        )

        logging.error(
            f"KEY NOT FOUND: title field not found in the dataframe\n{key_error_exception}"
        )
        logging.warning(
            "preprocessing the dataset without duplicate record verification"
        )

    except Exception as exception:
        print(
            "Failed to detect duplicate records. Preprocessing the dataset without duplicate record verification"
        )

        logging.error(
            f"UNEXPECTED ERROR: an unexpected error occurred\n{exception}")
        logging.warning(
            "preprocessing the dataset without duplicate record verification"
        )

    logging.info(
        f"passing NER and directions field to CorpusData.convert_string_to_list"
    )

    recipe_data["NER"] = recipe_data["NER"].apply(
        corpus_data.convert_string_to_list)
    recipe_data["cleaned_directions"] = recipe_data["directions"].apply(
        corpus_data.convert_string_to_list
    )

    logging.info("casted NER and directions as python lists for preprocessing")

    try:
        logging.info(
            f"passing NER and directions field to DataWrangling.remove_whitespace_and_duplicates"
        )

        recipe_data["NER"] = recipe_data["NER"].apply(
            data_wrangling.remove_whitespace_and_duplicates
        )
        recipe_data["cleaned_directions"] = recipe_data["directions"].apply(
            data_wrangling.remove_whitespace_and_duplicates
        )

        logging.info(
            "removed whitespaces and duplicates from NER and directions field")

    except TypeError as type_error_exception:
        print(
            "Failed to remove whitespaces and duplicate values from NER and cleaned_directions. Data integrity may be affected."
        )

        logging.error(
            f"TYPE ERROR: None value ia not a valid argument for input_ingredients_list \n{type_error_exception}"
        )
        logging.warning(
            "preprocessing the dataset without removing whitespaces and duplicate values"
        )

    except Exception as exception:
        print(
            "Failed to remove whitespaces and duplicate values from NER and cleaned_directions. Data integrity may be affected."
        )

        logging.error(
            f"UNEXPECTED ERROR: an unexpected error occurred\n{exception}")
        logging.warning(
            "preprocessing the dataset without removing whitespaces and duplicate values"
        )

    recipe_data.drop("Unnamed: 0", axis=1, inplace=True)

    logging.info(
        f"passing NER and directions field to CorpusData.convert_list_to_string"
    )

    recipe_data["NER"] = recipe_data["NER"].apply(
        corpus_data.convert_list_to_string)
    recipe_data["cleaned_directions"] = recipe_data["cleaned_directions"].apply(
        corpus_data.convert_list_to_string
    )

    logging.info(
        "casted NER and directions as python string for feature engineering")

    logging.info("renaming the variables of interest in the dataset")

    recipe_data.rename(columns={"title": "Recipe"}, inplace=True)
    recipe_data.rename(columns={"NER": "Ingredients"}, inplace=True)
    recipe_data.rename(columns={"directions": "Instructions"}, inplace=True)
    recipe_data.rename(columns={"source": "Source"}, inplace=True)
    recipe_data.rename(columns={"link": "URL"}, inplace=True)
    recipe_data.rename(
        columns={"ingredients": "Raw_Ingredients"}, inplace=True)
    recipe_data.rename(
        columns={"cleaned_directions": "Cleaned_Instructions"}, inplace=True
    )

    logging.info("columns renamed to: " + ", ".join(recipe_data.columns))

    logging.info("analyzing dataset for null records")
    null_value_count = sum(recipe_data.isna().sum())
    logging.info(f"found {str(null_value_count)} null values in the dataset")

    if null_value_count > 0:
        recipe_data.dropna(inplace=True)
        logging.info(
            f"{str(recipe_data.shape[0])} records remaining after removing null records"
        )

    logging.info(
        "generating corpus using ingredients and instructions records")
    recipe_data["Corpus"] = (
        recipe_data["Ingredients"] + " " + recipe_data["Instructions"]
    )
    recipe_data["Corpus"] = recipe_data["Corpus"].str.lower()

    logging.info("corpus text generated and saved in data.Corpus field")

    logging.info(
        "dropping Ingredients and Cleaned_Instructions fields from the dataset"
    )
    recipe_data.drop("Ingredients", inplace=True, axis=1)
    recipe_data.drop("Cleaned_Instructions", inplace=True, axis=1)

    try:
        logging.info(
            f"passing the corpus to CorpusData.lemmatize_and_remove_stop_words()"
        )
        recipe_data["Corpus"] = recipe_data["Corpus"].apply(
            corpus_data.lemmatize_and_remove_stop_words
        )

        logging.info("lemmatized text and removed stopwords from the corpus")

    except Exception as exception:
        print(
            "Failed to lemmatize the corpus and remove stop words. Data integrity may be affected."
        )

        logging.error(
            f"UNEXPECTED ERROR: an unexpected error occurred\n{exception}")
        logging.warning(
            "preprocessing the dataset without lemmatizing and removing stop words"
        )

    logging.info(
        f"passing the corpus to DataWrangling.remove_punctuations_and_whitespaces()"
    )
    recipe_data["Corpus"] = recipe_data["Corpus"].apply(
        data_wrangling.remove_punctuations_and_whitespaces
    )

    logging.info(
        "removed punctuations and useless whitespaces from the corpus")

    try:
        processed_dataset_path = "data/processed/recipe_nlg_processed_test.csv"
        recipe_data.to_csv(processed_dataset_path, index=False)

        print(
            f"Processed dataset successfully saved at {processed_dataset_path}")
        logging.info(f"processed dataset saved at {processed_dataset_path}")

    except IOError as io_error_exception:
        print("Preprocessing failed. Check logs for more details.")

        logging.error(
            f"INVALID RECIPE ID: Unable to write to the file at {processed_dataset_path}\n{io_error_exception}"
        )
        logging.critical(
            "EXECUTION TERMINATED: execution failed for backend.data_utils package"
        )
        logging.critical(
            "--------------------------------------------------------------------------------------"
        )

        logging.shutdown()
        sys.exit()

    except Exception as exception:
        print("Preprocessing failed. Check logs for more details.")

        logging.error(
            f"UNEXPECTED ERROR: an unexpected error occurred\n{exception}")
        logging.critical(
            "EXECUTION TERMINATED: execution failed for backend.data_utils package"
        )
        logging.critical(
            "--------------------------------------------------------------------------------------"
        )

        logging.shutdown()
        sys.exit()

    logging.info(
        "EXECUTION COMPLETE: backend.data_utils package executed succesfully")
    logging.info(
        "--------------------------------------------------------------------------------------"
    )
    logging.shutdown()

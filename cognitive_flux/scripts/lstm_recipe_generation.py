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
This module contains the class for long short term memory (LSTM) RNN based recipe 
generation. The classes offer methods for generating recipe text, validating LSTM 
results, & enhancing the consistency of recipe titles, ingredients & instructions.

Depending on individual cases, the program may be modified to use multiprocessing 
capailities to increase the speed of data processing, on eligible local computers.
The usage of each class & their methods are described in corresponding docstrings.

Classes and Functions:
    [1] LSTMTextSynthesizer (class)
        [a] generate_text
        [b] generate_combinations

    [2] LSTMStyleTransfer (class)
        [a] validate_lstm_result
        [b] process_recipe_name
        [c] process_recipe_ingredients
        [d] process_recipe_instructions

.. versionadded:: 1.3.0
.. versionupdated:: 1.3.0

Learn about RecipeML :ref:`RecipeML: RecipeGeneration - Generation by Ingredients`
"""
import os
import re
import json
import time
import pathlib
import platform

import numpy as np
import tensorflow as tf


class LSTMTextSynthesizer:
    """
    Class for generating recipes using a LSTM model trained on RecipeBowl dataset

    This class provides methods for generating text sequences based on a provided
    model, start string, & tokenizer. It handles preparing the input, iteratively
    generating characters, & managing the model's state during generation process.

    Class Methods:
        [1] generate_text
        [2] generate_combinations

    .. versionadded:: 1.3.0
    .. versionupdated:: 1.3.0

    The performance of the methods present in the class can be optimized by using
    the CPUPool via the multithreading capailities on eligible local/cloud system.
    """
    def __init__(self):
        pass

    def generate_text(
        self,
        model,
        start_string,
        tokenizer,
        num_generate,
        temperature,
        STOP_WORD_TITLE="📗 ",
    ):
        """
        Method to generate recipes using an LSTM model trained on RecipeBowl data

        This method takes a trained model, a starting string, a tokenizer & other
        parameters to generate text based on the model's predictions. It utilizes
        a temperature parameter, for controlling randomness of the generated text.

        Read more in the :ref:`RecipeML:DataWrangling & Fundamental PreProcessing`

        .. versionadded:: 1.3.0

        Parameters:
            [keras.Model] model: The trained LSTM RNN model for recipe generation
            [str] starting_string: An initial string to kickstart text generation
            [Tokenizer] tokenizer: Tokenizer used for converting text to sequence
            [int] num_generate: The number of textual characters, to be generated
            [float] temperature: Arg controlling randomness of the generated text
            [str] STOP_WORD_TITLE: Title marking the beginning of text generation

        Returns:
            [str] padded_start_string: Generated text starting from start strings
        """
        # Prepend recipe start string with the marker denoting the recipe's title
        padded_start_string = STOP_WORD_TITLE + start_string

        # Convert the recipe's start string to input indices, using the tokenizer
        input_indices = np.array(
            tokenizer.texts_to_sequences([padded_start_string]))
        text_generated = []  # Initialize empty list to store the generated texts

        model.reset_states()  # Reset the model states before generating the text

        for char_index in range(num_generate):
            # Make prediction using the model
            predictions = model(input_indices)

            # Adjust the prediction using the temperature argument for randomness
            predictions = tf.squeeze(predictions, 0)
            predictions = predictions / temperature

            # Sample the predicted id using tensorflow's categorical distribution
            predicted_id = tf.random.categorical(predictions, num_samples=1)[
                -1, 0
            ].numpy()

            input_indices = tf.expand_dims(
                [predicted_id], 0
            )  # Update the input index of the vacoabulary with the predicted ids

            # Convert predicted id to next character and append to generated text
            next_character = tokenizer.sequences_to_texts(
                input_indices.numpy())[0]
            text_generated.append(next_character)

        return padded_start_string + "".join(text_generated)  # Return padded str

    def generate_combinations(
        self,
        model,
        rnn_tokenizer,
        ingredients="A",
        temperature=0.5,
        max_token_length=1000,
    ):
        """
        Method to generate text combination using the LSTM-based text synthesizer

        This method utilizes LSTM-based text synthesizer to generate combinations
        based on an LSTM RNN network model, text tokenizer, a starting ingredient,
        temperature, defaulting to 0.5, & maximum token length defaulting to 1000.

        Read more in the :ref:`RecipeML:DataWrangling & Fundamental PreProcessing`

        .. versionadded:: 1.3.0

        Parameters:
            [keras.Model] model: The trained LSTM RNN model for recipe generation
            [str] starting_string: An initial string to kickstart text generation
            [Tokenizer] tokenizer: Tokenizer used for converting text to sequence
            [int] num_generate: The number of textual characters, to be generated
            [float] temperature: Arg controlling randomness of the generated text
            [str] STOP_WORD_TITLE: Title marking the beginning of text generation

        Returns:
            [str] padded_start_string: Generated text starting from start strings
        """
        lstm_text_synthesizer = LSTMTextSynthesizer()
        generated_text = lstm_text_synthesizer.generate_text(
            model,
            start_string=ingredients,
            tokenizer=rnn_tokenizer,
            num_generate=max_token_length,
            temperature=temperature,
        )

        return generated_text # Return generated text based on provided parameter


class LSTMStyleTransfer:
    """
    Class for LSTM Style Transfer in Recipe Titles, ingredients, and Instructions

    This class provides methods for validating LSTM result, processing the recipe
    titles, ingredients & instructions using LSTM-based style transfer techniques.

    Class Methods:
        [1] validate_lstm_result
        [2] process_recipe_name
        [3] process_recipe_ingredients
        [4] process_recipe_instructions

    .. versionadded:: 1.3.0
    .. versionupdated:: 1.3.0

    The performance of the methods present in the class can be optimized by using
    the CPUPool via the multithreading capailities on eligible local/cloud system.
    """

    def __init__(self):
        pass

    def validate_lstm_result(self, lstm_generated_text):
        """
        Method to validate generated text to check presence of specific character

        The method checks whether the provided text contains all of the specified
        characters. Absence of any of the character implies successful validation.

        Read more in the :ref:`RecipeML:DataWrangling & Fundamental PreProcessing`

        .. versionadded:: 1.3.0

        Parameters:
            [str] lstm_generated_text: The text generated by LSTM to be validated

        Returns:
            [bool] bool_result: True if text passes validation, & False otherwise
        """
        return not all(char in lstm_generated_text for char in ["📗", "🥕", "📝"])

    def process_recipe_name(self, input_title):
        """
        Method to enhance & process the consistency & cleanliness of recipe title

        This method processes the recipe title by removing any unwanted character,
        leading/trailing whitespaces & returns formatted title with proper casing.

        Read more in the :ref:`RecipeML:DataWrangling & Fundamental PreProcessing`

        .. versionadded:: 1.3.0

        Parameters:
            [str] input_title: RNN generated recipe title to be processed/cleaned

        Returns:
            [str] recipe_title: The processed recipe title with proper case types
        """
        recipe_title = (
            input_title.replace(".", "").replace(
                "•", "").replace("\n", "").strip()
        )
        return recipe_title.title()  # Format processed title with proper casings

    def process_recipe_ingredients(self, input_ingredients):
        """
        Method to take input, & perform preprocessing to enhance the data quality

        This method takes the string of ingredients, performs specific processing
        steps to clean and limit the number of ingredients, and returns a list of
        capitalized and stripped recipe ingredients, to be presented to the users.

        Read more in the :ref:`RecipeML:DataWrangling & Fundamental PreProcessing`

        .. versionadded:: 1.3.0

        Parameters:
            [str] input_ingredient: Generated string containing recipe ingredient

        Returns:
            [list] ingredients_list: list of processed, & capitalized ingredients
        """
        ingredients_list = (
            input_ingredients.replace("\n", "||").replace("•", "").split("||")
        )
        ingredients_list = list(set(ingredients_list))  # Remove duplicate record

        # Limit the number of ingredients to max 12, if the original list exceeds
        if len(ingredients_list) > 12:
            ingredients_list = ingredients_list[:12]

        # Return capitalized list & strip each ingredient excluding empty strings
        return [string.strip().capitalize() for string in ingredients_list if string]

    def process_recipe_instructions(self, input_instructions):
        """
        Method to take input, & perform preprocessing to enhance the data quality

        This method takes raw recipe instructions as input, cleans & formats them,
        & returns processed string with improved readability. Processing includes
        removing useless phrases, capitalizing steps & organizing into paragraphs

        Read more in the :ref:`RecipeML:DataWrangling & Fundamental PreProcessing`

        .. versionadded:: 1.3.0

        Parameters:
            [str] input_instructions: Generated string containing the instruction

        Returns:
            [str] recipe_instructions: Processed and formatted recipe instruction
        """
        try:
            # Attempt to remove the specific phrase, ignore errors if not present
            input_instructions = input_instructions.replace(
                "Watch how to make this recipe", ""
            )
        except:
            pass

        # Split instructions into a list based on the specified bullet point char
        input_instructions = input_instructions.split("▪︎")
        recipe_instructions = [
            step.strip().capitalize() for step in input_instructions if step
        ]

        # Join processed steps & capitalize the first word after the punctuations
        recipe_instructions = ". ".join(recipe_instructions)
        recipe_instructions = re.sub(
            r"(?<=\.|\?|\!)\s*\w", lambda x: x.group().upper(), recipe_instructions
        )

        # Ensure a period at the end and replace consecutive periods with periods
        recipe_instructions = recipe_instructions + "."
        recipe_instructions = re.sub(r"\.\.", ".", recipe_instructions)

        sentences = recipe_instructions.split(
            ". ")  # Split instructions into sentences & organize into paragraphs
        if len(sentences) >= 10:
            num_sentences_per_paragraph = len(sentences) // 2

            first_paragraph = ". ".join(
                sentences[:num_sentences_per_paragraph]) + "."
            second_paragraph = ". ".join(
                sentences[num_sentences_per_paragraph:]) + "."

            # Format recipe instruction's paragraphs with double HTML line breaks
            recipe_instruction = f"{first_paragraph}<br><br>{second_paragraph}"

        return recipe_instruction  # Return back the processed recipe instruction

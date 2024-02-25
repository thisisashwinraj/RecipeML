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
Module for managing authentication tokens and text generation using OpenAI & PaLM.

This module provides a convenient interface for handling auth tokens required for 
text generation. It utilizes Streamlit for securely storing & retrieving API keys.
AuthTokens encapsulates these keys, & can be extended to additional functionality.

Classes and Functions:
    [1] AuthTokens (class)
        [a] OpenAI API (attribute)
        [b] Google PaLM API (attribute)

.. versionadded:: 1.3.0

Learn about RecipeML :ref:`RecipeML: Auth Tokens and Streamlit Secrets Management`
"""
import os
import streamlit


class AuthTokens:
    """
    Class to manage the authentication tokens & API Keys for OpenAI, and PaLM API.

    This class holds the authentication tokens, for accessing OpenAI and PaLM API.
    The `openai_api_key`, and `palm_api_key` are retrieved from Streamlit secrets.

    Included APIs:
        [1] OpenAI API
        [2] Google PaLM API

    .. versionadded:: 1.3.0

    NOTE: API Keys are maintained as streamlit secrets in .streamlit/secrets.toml
    """

    openai_api_key = streamlit.secrets["openai_api_key"]  # OpenAI Auth Token/API
    palm_api_key = streamlit.secrets["palm_api_key"]  # GooglePaLM Auth Token/API
    firebase_api_key = streamlit.secrets["firebase_api_key"]  # Firebase Auth Key

    def __init__(self):
        pass

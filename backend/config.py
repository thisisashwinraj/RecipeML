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
This module stores the secret configuration settings for use in the Streamlit app.
It retrieves the sender's e-mail ID and password from streamlit/secrets.toml file, 
which can be used for sending smtp enabled mails from the app to registered users.

This allows you to store secrets securely and access them as environment variable.
When deployed in streamlit cloud, the secrets are managed through the interface & 
does'nt require ~./secrets.toml file to be exclusively made available in the repo.

Configurations:
    [1] SENDER_EMAIL_ID (str)
    [2] SENDER_PASSWORD (str)

.. versionadded:: 1.0.0

Learn about RecipeML :ref:`RecipeML v1: Web App Configuration & Security Policies`
"""

import streamlit

SENDER_EMAIL_ID = streamlit.secrets["sender_email_id"]  # Fetch the users mail id
SENDER_PASSWORD = streamlit.secrets["sender_email_password"]  # Get user password

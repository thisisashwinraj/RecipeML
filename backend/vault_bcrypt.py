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
[DEPRECEATED]: This module has been depreceated in v1.1.0, and will be removed in 
the upcoming updates. Check the ~./changelog.md for more details about the update

This module sets up a user data vault for authentication in the streamlit web app
using the streamlit_authenticator package. It include configurations for username
and passwords, & generates a bcrypt hashed password key for secure authentication.

.. versionadded:: 1.0.0

Learn about RecipeML :ref:`RecipeML v1: User Interface and Functionality Overview
"""
import pickle
import streamlit as st
import streamlit_authenticator as stauth

# Read the name, username and the password for the user(Replace in code for test)
names = ["ADD NAMES OF THE USER"]
usernames = ["ADD USERNAMES OF THE USER"]
passwords = ["ADD PASSWORDDS OF THE USER"]

# Generate a hashed key for user using bcrypt algorithm for secure authentication
hashed_key = stauth.Hasher(
    ["ADD PASSWORD OF THE USER TO BE HASHED"]).generate()

with open("data/pickle/vault_bcrypt.pkl", "wb") as file:
    pickle.dump(hashed_key, file)

st.write(hashed_key)  # Display hashed key & replace it in ~alpha_users.yaml file

import json
import streamlit as st


class FirebaseCredentials:
    def __init__(self): pass


    def _edit_json_credentials(self, file_path, key, new_value):
        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            data[key] = new_value

            with open(file_path, "w") as f:
                json.dump(data, f, indent=4)

        except FileNotFoundError:
            print(f"Error: File '{file_path}' not found.")

        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in {file_path}.")


    def fetch_firebase_service_credentials(self, file_path):
        firebase_creds_dict = {
            "type": "service_account",
            "project_id": st.secrets["firebase_project_id"],
            "private_key_id": st.secrets["firebase_private_key_id"],
            "private_key": st.secrets["firebase_private_key"],
            "client_email": st.secrets["firebase_client_email"],
            "client_id": st.secrets["firebase_client_id"],
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_x509_cert_url": st.secrets["firebase_client_x509_cert_url"],
            "universe_domain": "googleapis.com"
        }

        for key in list(firebase_creds_dict.keys()):
            self._edit_json_credentials(file_path, key, firebase_creds_dict[key])


    def fetch_gsheet_credentials(self, file_path):
        gsheet_creds_dict = {
            "type": "service_account",
            "project_id": st.secrets["gsheet_project_id"],
            "private_key_id": st.secrets["gsheet_private_key_id"],
            "private_key": st.secrets["gsheet_private_key"],
            "client_email": st.secrets["gsheet_client_email"],
            "client_id": st.secrets["gsheet_client_id"],
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_x509_cert_url": st.secrets["gsheet_client_x509_cert_url"],
            "universe_domain": "googleapis.com"
        }

        for key in list(gsheet_creds_dict.keys()):
            self._edit_json_credentials(file_path, key, gsheet_creds_dict[key])


if __name__ == "__main__":
    firebase_credentials = FirebaseCredentials()
    firebase_credentials.fetch_firebase_service_credentials("secret_test.json")

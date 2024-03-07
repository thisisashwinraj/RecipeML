from azure.storage.blob import BlobServiceClient, BlobClient
from configurations.api_authtoken import AuthTokens


class AzureStorageAccount:
    def __init__(self, container_name):
        auth_tokens = AuthTokens()

        self.connection_string = auth_tokens.azure_storage_account_connection_string
        self.container_name = container_name

    def store_image_in_blob_container(self, file_path, blob_name):
        if filepath.lower() == "unavailable": return "unavailable"

        blob_service_client = BlobServiceClient.from_connection_string(
            self.connection_string
        )

        container_client = blob_service_client.get_container_client(self.container_name)
        blob_client = container_client.get_blob_client(blob_name)

        with open(file_path, "rb") as data:
            upload_stream = data.read()
            blob_client.upload_blob(upload_stream, overwrite=True)

        return blob_client.url

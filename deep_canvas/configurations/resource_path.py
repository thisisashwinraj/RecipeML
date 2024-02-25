import os

class ResourceRegistry:
    ingredients_list_path = "data/pickle/recipe_nlg_ingredients.pkl"
    raw_recipenlg_dataset_path = "data/raw/recipe_nlg_raw_dataset.csv"
    processed_recipenlg_dataset_path = "data/processed/recipe_nlg_processed_data.csv"

    knn_tfidf_vectorizer = 'embeddings/tfidf_vectorizer_recipe_nlg.pkl'
    knn_tfidf_matrix = 'embeddings/tfidf_matrix_recipe_nlg.pkl'
    feature_space_matching_model = '/content/drive/MyDrive/RecipeML/neural_engine_recipe_nlg.pkl'

    generated_images_directory_path = "exports/generated_img/"

    def __init__(self):
        pass

# os.path.abspath("../folder_name/file_name.txt")

"""
import os

class ResourceRegistry:
    ingredients_list_path = "/content/drive/MyDrive/RecipeML/recipe_nlg_ingredients.pkl"
    raw_recipenlg_dataset_path = "/content/drive/MyDrive/RecipeML/recipe_nlg_raw_dataset.csv"
    processed_recipenlg_dataset_path = "/content/drive/MyDrive/RecipeML/data_processed_recipe_nlg.csv"

    knn_tfidf_vectorizer = '/content/drive/MyDrive/RecipeML/tfidf_vectorizer_recipe_nlg.pkl'
    knn_tfidf_matrix = '/content/drive/MyDrive/RecipeML/tfidf_matrix_recipe_nlg.pkl'
    feature_space_matching_model = '/content/drive/MyDrive/RecipeML/neural_engine_recipe_nlg.pkl'

    generated_images_directory_path = ""

    def __init__(self):
        pass

# os.path.abspath("../folder_name/file_name.txt")
"""
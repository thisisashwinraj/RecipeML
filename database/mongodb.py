import requests
from datetime import datetime, timezone, timedelta

import pymongo
from pymongo import MongoClient

from configurations.api_authtoken import AuthTokens


class MongoDB:
    def __init__(self):
        auth_tokens = AuthTokens()

        client = MongoClient(auth_tokens.mongodb_connection_string)
        db = client.recipe_archives

        self.generated_recipes_collection = db.generated_recipes
        self.recommended_recipes_collection = db.recommendations

    def store_generated_recipes(
        self,
        username,
        input_methodology,
        input_query,
        input_language,
        recipe_id,
        recipe_title,
        ingredients,
        instructions,
        serving_size,
        preperation_time,
        calories,
        primary_recipe_image,
        secondary_recipe_image,
    ):
        ip_data = requests.get(f'http://ip-api.com/json/' + requests.get('https://api.ipify.org').text).json()

        data_to_insert = {
            "_id": username,
            "generated_recipes": {
                recipe_id: {
                    "parameter": {
                        "generation_technique": input_methodology,
                        "query": input_query,
                        "generation_language": input_language,
                        "generated_on": datetime.now(timezone(timedelta(hours=5, minutes=30))).strftime("%m.%d.%Y (%H:%M:%S)"),
                        "location": str(ip_data['city']) + ", " + str(ip_data['country'])
                    },
                    "response": {
                        "recipe_title": recipe_title,
                        "ingredients": ingredients,
                        "instructions": instructions,
                        "total_calories": calories,
                        "preperation_time": preperation_time,
                        "serving_size": serving_size,
                        "primary_image": primary_recipe_image,
                        "secondary_image": secondary_recipe_image,
                    },
                }
            },
        }

        existing_user = self.generated_recipes_collection.find_one({"_id": username})

        if existing_user:
            self.generated_recipes_collection.update_one(
                {"_id": username},
                {
                    "$set": {
                        f"generated_recipes.{recipe_id}": data_to_insert[
                            "generated_recipes"
                        ][recipe_id]
                    }
                },
            )
        else:
            self.generated_recipes_collection.insert_one(data_to_insert)

    def store_recommended_recipes(
        self,
        username,
        recommendation_id,
        input_ingredients,
        recipe_id_list,
        recommendations_list,
        recipe_images_list,
    ):
        data_to_insert = {
            "_id": username,
            "recommendations": {
                recommendation_id: {
                    "parameter": {
                        "ingredients": input_ingredients,
                        "generated_on": datetime.now(timezone(timedelta(hours=5, minutes=30))).strftime("%m.%d.%Y (%H:%M:%S)"),
                    },
                    "response": {
                        "recipe_id": recipe_id_list,
                        "recommendations": recommendations_list,
                        "images": recipe_images_list,
                    },
                }
            },
        }

        existing_user = self.recommended_recipes_collection.find_one({"_id": username})

        if existing_user:
            self.recommended_recipes_collection.update_one(
                {"_id": username},
                {
                    "$set": {
                        f"recommendations.{recommendation_id}": data_to_insert[
                            "recommendations"
                        ][recommendation_id]
                    }
                },
            )
        else:
            self.recommended_recipes_collection.insert_one(data_to_insert)

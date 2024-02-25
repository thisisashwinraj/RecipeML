![](https://github.com/thisisashwinraj/RecipeML-Recipe-Recommendation/blob/main/assets/banners/recipeml_banner_darkmode.png#gh-dark-mode-only)
![](https://github.com/thisisashwinraj/RecipeML-Recipe-Recommendation/blob/main/assets/banners/recipeml_banner_lightmode.png#gh-light-mode-only)

<p align='justify'>RecipeML is an NLP-powered application that recommends recipes to users based on the ingredients provided by the users and provides detailed insights including ingredients, directions, preparation time, recipe sources, and more. The dataset used for training this model contains over 2.2 million recipes, sourced from different recipe sites over the web</p>

<p align='justify'>Beyond recommending existing recipes, RecipeML taps into its deep understanding of language generation, utilizing multi-layered LSTM RNN architecture, coupled with PaLM's semantic parsing capabilities, to generate novel recipes from scratch. Further, it integrates generative models based on Stable Diffusion to translate these textual creations into photo-realistic images. Upcoming update will introduce chat interface for users to directly engage with RecipeML</p>

<p align='justify'>The project started in June 2023 is released under the GNU Affero General Public License v3.0 and is maintained as an open-source project. Contributions, & suggestions are welcome from the community. Check out the live webapp <a href='https://recipe-ml.streamlit.app/'>here</a></p>

# SubDirectories and Constraints

### Model Dependencies
• **Recommendation:** SciKit-Learn (KNearestNeighbors), NLTK, Continous Bag of Words (CBoW), AzureML, OpenAI GPT
<br>
• **Recipe Generation:** LSTM (Long Short-Term Memory) RNN (Recurrent Neural Network), TensorFlow, Google PaLM-2
<br>
• **Image Generation:** PlaygroundAI (playground-v2-1024-aesthetic), RunwayML (stable-diffusion-v1), OpenAI DALL.E2

### Web App Resources
• **Assets:** This sub-directory contains the image files, CSS files, JavaScript snippets and other necessary WebApp assets
<br>
• **Backend:** This directory contains code for preprocessing the dataset, generating PDF files, & sending mails via SMTP
<br>
• **Data:** This subdirectory contains the recipes dataset used for training the algorithms, and the necessary pickle dump

<p align='justify'>All relevant updates, and stable versions are made available in this repo's versions subdirectory. Some subdirectories may be sensitive for the project and may trigger review requests, when pull requests touch these files. GitHub handles with commit rights made available in the ~./.github/CODEOWNERS, are responsible for reviewing any such alterations</p>

# Under the Hood of RecipeML
### Recipe Recommendation (FeatureScape)

<p align='justify'>RecipeML casts the process of conditional recipe recommendation as a feature space-matching algorithm that generates the recipe recommendations based on the cosine similarity between the input ingredients provided by the user and the ingredients required for preparing each recipe, using the brute approach. The recipe directions, source URL and recipe preparation time adds further context to the corpus for generating optimized recipe recommendation</p>

![RecipeML Recipe Recommendation System Demo](https://github.com/thisisashwinraj/RecipeML-Recipe-Recommendation/blob/main/assets/demos/Recipe%20Recommendation.png)

<p align='justify'>The raw dataset exhibited a significant amount of redundant information, necessitating the implementation of pre-processing algorithms to extract the ingredients, and sanitize the dataset for recipe recommendation. Punctuation marks, stop words, accent marks using Unicode, and non-alphabetic characters were eliminated, while words were lemmatized and commonly found ingredients, & their measures were substituted. All duplicate records were dropped</p>

<p align='justify'>Analytical evaluations utilizing diverse ingredient combinations unequivocally show that the learning algorithms using TF/IDF Embedded Vectorizer outperform Mean Embedded Vectorizer. Further, empirical evidence substantiates that these natural language algorithms, and NLU model can be conditioned to accommodate numerous culinary traditions</p>

```
tfidf_vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf_vectorizer.fit_transform(data[subset])

model = NearestNeighbors(
    n_neighbors=n_neighbors, metric=metric, algorithm=algorithm, n_jobs=n_jobs
)
model.fit(tfidf_matrix)

tfidf_vector = tfidf_vectorizer.transform([ingredients_text])
_, indices = model.kneighbors(tfidf_vector)

recommended_indices = indices[0][1:]
recommended_recipes_indices = list(recommended_indices)
```
### Recipe Generation (CognitiveFlux)
<p align='justify'>RecipeML effectively integrates character-level Long Short-Term Memory (LSTM) RNN architecture with Google's Pathways Language Model (PaLM API) for recipe generation. RecipeML presents users with a dual paradigm, allowing for recipe synthesis, based on either the specified recipe names, or a singular start ingredient (from > 20k ingredients)</p>

<p align='justify'>Recipe generation using a start ingredient involves a character-level Long Short-Term Memory (LSTM) Recurrent Neural Network (RNN), trained on a comprehensive 125k+ recipe corpus. The RNN architecture employs LSTM cells to model longrange dependencies within ingredient sequences & capture the sequential nature of recipe instructions</p>

![RecipeML Recipe Generation System Demo](https://github.com/thisisashwinraj/RecipeML-Recipe-Recommendation/blob/main/assets/demos/Recipe%20Generation%20by%20Name%2001_02.png)

<p align='justify'>RNNs, known to be stateful, utilizes the network's ability to retain and propagate context across sequential data facilitating the generation of coherent & contextually relevant recipe, imbued with syntactic and semantic consistency</p>

<p align='justify'>The name-based recipe generation paradigm integrates the PaLM API, a 540-billion parameter, dense decoder-only Transformer model with multimodal capabilities to generate recipes based on designated names. The integration of PaLM adds a layer of semantic intelligence to the model, ensuring that the generated recipes not only adhere to syntactic structures, but also encapsulates the intended culinary themes, associated with the designated recipe names</p>

```
# Recipe generation based on a Single Starting Ingredients
predictions = model(input_indices)

predictions = tf.squeeze(predictions, 0)
predictions = predictions / temperature

predicted_id = tf.random.categorical(
    predictions,
    num_samples=1
)[-1,0].numpy()

input_indices = tf.expand_dims([predicted_id], 0)
next_character = tokenizer.sequences_to_texts(input_indices.numpy())[0]
```

### Image Generation (DeepCanvas)
<p align='justify'>RecipeML further leverages Stable Diffusion models from RunwayML and PlaygroundAI and uses OpenAI DALL·E2 to create visually captivating images of the recommended recipes. These images enhance the user experience, providing a visually appealing presentation, of the recommendations. The gpt 3.5 turbo model is used for fine-tuning the model</p>

```
# Image Generation using Stable Diffusion
model_id = "runwayml/stable-diffusion-v1-5"

pipe = StableDiffusionPipeline.from_pretrained(
    model_id, torch_dtype=torch.float16
)        
pipe = pipe.to("cuda")
image_512 = pipe(prompt).images[0]

pipe = DiffusionPipeline.from_pretrained(
    "playgroundai/playground-v2-1024px-aesthetic",
    torch_dtype=torch.float16,
    use_safetensors=True, variant="fp16"
)
pipe = pipe.to("cuda")
image_1024 = pipe(prompt=prompt, guidance_scale=3.0).images[0]
```

<p align='justify'>To be able to make changes to the source, you may need to install and use a Python IDE such as PyCharm, Microsoft VisualStudio or any other Python interpreter. You will also require a Jupyter Notebook for working with the code snips</p>

![RecipeML Recipe Generation System Demo](https://github.com/thisisashwinraj/RecipeML-Recipe-Recommendation/blob/main/assets/demos/Recipe%20Generation%20by%20Ingredients%2001_02.png)

<p align='justify'>Working demo for the application hosted over Streamlit cloud can be accessed at <a href='https://recipe-ml.streamlit.app/'>recipe-ml.streamlit.app</a>. RecipeML's development takes place entirely on GitHub. Submit any bugs, that you may encounter to the issues tracker with the reproducible example demonstrating the problem in accordance with the ~IssueTemplate present in contributing files</p>

To run this Streamlit application on a local computer, open the terminal, install Streamlit, and then type the command:
```
streamlit run app.py
```
<p align='justify'>Command may take upto 30 sec before opening the streamlit app on the browser with local url: http://localhost:8501.</p>

<p align='justify'>To run the application locally on a web browser, first download the required resources, or clone this repository using the git version control. StreamLit must be pre-installed for this application to work or you may install it using pip package manager. All packages necessary for the webapp to run can be installed easily, using the requirements.txt file</p>

# Contribution Guidelines
To start contributing to the project, clone the repository into your local system subdirectory using the below git code:
```
https://github.com/thisisashwinraj/RecipeML-Recommendation-System.git
```
<p align='justify'>Before cloning the repository, make sure to navigate to the working subdirectory of your command line interface and ensure that no folder with same name exists. Other ways to clone the repository include using a password-protected SSH keys or by using Git CLI. The changes may additionally be performed by opening this repo using GitHub Desktop</p>

### Submitting a Pull Request
<p align='justify'>Before opening a Pull Request, it is recommended to have a look at the full contributing page to make sure your code complies with all the pull request guidelines. Please ensure, that you satisfy the ~/checklist before submitting your PR</p>

![RecipeML Multi-Language Capabilities Demo](https://github.com/thisisashwinraj/RecipeML-Recipe-Recommendation/blob/main/assets/demos/Multi%20Language%20Capabilities%2001_02.png)

Navigate to this subdirectory & check the status of files that were altered (red) by running the below code in Git Bash:
```
git status
```
Stage all your files that are to be pushed into your pull request. This can be done in two ways - stage all or some files:
```
git add .            // adds every single file that shows up red when running git status
```
```
git add <filename>   // type in the particular file that you would like to add to the PR
```

Commit all the changes that you've made and describe in brief the changes that you have made using this command:
```
git commit -m "<commit_message>"
```
Push all of your updated work into this GitHub repo in the form of a Pull Request by running the following command:
```
git push origin main
```
All pull requests are reviewed on a monthly rolling basis. Your understanding is appreciated during the review process

# License and Project Status
<p align='justify'>RecipeML and all its resources are distributed under the GNU Affero General Public License v3.0. The API keys and endpoints are maintained as environment variables, & several measures are provisioned to keep the user data secure. The streamlit authenticator uses bcrypt for hashing passwords maintained in the YAML file & the web app uses JWT cookies with a life span of upto 3 days. The latest version of RecipeML is v1.3, and is available across all digital devices</p>


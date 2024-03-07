## Version 1.2.0 (Latest Update)

**[Release Date]:**

December 30, 2023 (Saturday)

**[Release Notes]:**
- Introduced a new feature for recipe generation using multi-layer LSTM RNN networks, and PaLM API
- Added PlaygroundAIs diffusion-based text-to-image model, for 1024x1024 quality image generation
- Implemented streamlit session to store and persist session states making the web app more dynamic
- Leveraged responsible AI practices, ensuring the algorithm promotes fair, unbiased ethical results
- Added feature to display serving sizes, & preparation times for each generated/recommended recipe
- Enhanced the streamlit user interface and streamlined the interactions for seamless inferencing

**[Fixes/Updates]:**
- Implemented multiple performance optimization process, and bug fixes for smoother user experience
- Refactored the code base to separate recommendation, text generation & image generation program
- Added onboarding for web app users, implemented preloaders and added streamlit toast notification
- Updated streamlit's color palette to infuse the vibrance of Generative AI across the application

---------------------------------------------------------------------------------------------------

## Version 1.1.0 (Stable Release)

**[Release Date]:**

October 20, 2023 (Friday)

**[Release Notes]:**
- Increased the size of the model's training data from 7,000+ recipes to over 2.2 million+ recipes
- Integrated the StableDiffusion model and DALL-E 2 to generate images of the recommended recipes
- Added feature for users to download the recommended recipes as a PDF document for offline access
- Added feature for users to send recipe mails with PDF attachments to their registered email id
- Expanded the ingredients list from 600, to over 20,000 ingredients for the user's to choose from

**[Fixes/Updates]:**
- Updated recommendation model to replace the CBoW algorithm with feature space matching algorithm
- Added new data preprocessing methods & updated the existing techniques to enable multiprocessing

---------------------------------------------------------------------------------------------------

## Version 1.0.0 (Stable Release)

**[Release Date]:**

June 22, 2023 (Thursday)

**[Release Notes]:**
- Developed a recipe recommendation system using a 2-layer CBoW model on TF/IDF and mean emeddings
- Used training dataset with 7,000+ recipes collectd from various different culinary traditions
- Added feature for authenticated users to send emails with recipes to their registered email ids
- Added authentication feature for new users to register and existing users to signup for the app
- Added feature for all users to rate the recipe recommendations to evaluate the model performance
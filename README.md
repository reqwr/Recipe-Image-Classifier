
# WebEats

## Overview

A application that takes in uploaded food images and returns similar food recipes.

Link to project:
https://webeats.streamlit.app/

Running the project may take a while due to Streamlit inactivity.

## How to run the application

Install all requirements:
```
pip install -r requirements.txt
```
Run the Streamlit application through the standard command:
```
streamlit run app/app.py
```
## Project Steps

1. Web scraping recipe and image data

2. Performing EDA and Data Transformations

    2.1 Using NMF to create meta topics for recipes and then reducing based on the evaluation from a t-SNE plot.
    2.2 Loading images based 

3. Model Training

	3.1 Fine-tuning the InceptionV3 model to classify images with their respective meta topics.

	3.2 Performing PCA on the last Dense layer to reduce and identify stronger correlations.

    3.3 Using ANN to find the nearest neighbour for each image to identify similar looking recipes.

4. Constrution of training and prediction pipeline

5. Deployment onto Streamlit

## Project Process

### Data Transformation

By fitting the recipe names with a TF-IDF vectorizer, important semantic data could be derived that relates recipes together. Using NMF, a set of topics was produced to reflect this correlation. A t-SNE plot was drawn that showed the clustering between topics. Areas with defined clusters meant that the clustering was meaningful. However, there were areas where many different clusters overlapped, indicating that they should be combined. By measuring the cosine similarity, topics that were close to one another were merged, eventually leading to a total of 90 meta topics used in classification.

An immediate problem that arose from this was that the similarities between recipe names did not indicate similarities between images, nor did it mean similarities between recipes. Another more complex but accurate method would be to also vectorize the recipes directions themselves and cluster based on recipe names and the actual recipe. This would have been computationally more complex and expensive, however it would have led to more accurate results. The goal of this project was to find similar recipes based on a given image. Therefore, even if the images found were different, if the recipes were similar then the project would have been considered a success.

###  Model Training

The InceptionV3 was fine-tuned and used in this project. First, the top 40 layers were unfrozen and trained, then the bottom 100 layers were trained. The result was a model that yielded around a 20 percent accuracy. This accuracy was incredibly low, and despite the days taken for training the hundreds of layers, the model still converged to this accuracy. This however, was not a problem. The point of training this classification model was to extract the information from the last Dense layer, which contained the necessary features for correlating images.

PCA and ANN methods were used on this last Dense layer. Firstly, PCA was performed to condense the layer's information. Secondly, ANN was used to find neighbouring images based on the information from the Dense layer. The result was the ability to find similar images with the highest correlation possible.

The biggest problem with this process was the time taken for training, the classification inaccuracies which permeated from data transformation limitations, and finally the lack of recipe correlation. On the last issue, this project mainly focused on clustering recipe names, and then classification via images and meta topics. The recipe directions themselves were never used, making it harder to actually correlate similar recipes rather than just similar sounding recipes and food images. Although if could be argued that this approach actually makes it easier for users to find unique recipes based on what their current food interests are, it is harder for users to actually find very similar recipes and forces them to look into food options for which they may not have the necessary ingredients.

### Deployment

The images were uploaded to Hugging Face which allows for unlimited datasets and models to be hosted in their databses. This was incredibly useful and allowed for much easier access to datasets than to store them locally. The problem, however, was that using Streamlit forced any loaded models or images to be redownloaded and cached upon every use. This slowed down the process of the application, and could be solved with better hosting services.




# Project-4-Beer-Recommendation-Engine

### Author: Jeffrey Oller

## Overview

This project uses data from a well-known beer-centered social platform to recommend beers to users.

## Files

* Project.ipynb - The main project file
* application.py - A streamlit application (proof of concept for deployed project)
* FunctionalModel - The underlying processes from Project.ipynb that are programmed functionally for import into application.py
* Beer Recommendation System.pptx - Powerpoint presenting of the process and finding.
* ratings_jeff.htm - the html version of a ratings page - this is parsed to feed user data into model.

## Data

The bulk of the data came from this [kaggle page](https://www.kaggle.com/datasets/rdoume/beerreviews)
The user data for the streamlit application is scraped (1-20 requests per execution) from the web.

## Model

The model is Surprise's implementation of Funk SVD. This is strictly a collaborative model which takes into account all users, but which iteratively manipulates matrices to give greater importance to similar users.

## Model Performance

Model performance was measured at .6 RMSE and .68 FCP.
My subjective impression of the model's performance is quite good. Out of 50 recommencations, it only recommended 2 beers I doubt I will enjoy.

## Limitations

The model recommends many of the same beers to users who have different tastes. The reason for this is quite clear, many beers have very few ratings and, therefore, some of them will have outlier-level high ratings. This was overcome by scaling the predicted rating by the number of ratings - in other words, beers that are less popular or relatively unavailable will be penalized relative to beers that are more popular or more available. (This is achieved by taking the 20th root of the number of ratings and multiplying it by the predicted rating - the 20th root was arrived at subjectively)

The model, since it is strictly collaborative, does not take into account any features of the beers themselves.

## Future Improvements

In the future, this model can be replaced by a hybrid model. At this time, the only data that are available about the beers are the style and the ABV. Preferably, the IBU and whether or not the beer is even available would be added as well.
---
title : "End to end machine learning project to predict house prices"
date : 2020-01-01
tags : [machine learning, opencv, CNN, python]
header :
  image : "./assets/images/housepriceprediction/app.PNG"
excerpt : "Machine learning, CNN, python"
mathjax : true
---
# Kaggle Competition: House Price Prediction - Advance Regression Technique
[Source code](https://github.com/achafi/Banglore_House_Prices)

*In this project, I am going to predict the price of a house using its 9 features. Basically i ame solving the [Kaggle Competition](https://www.kaggle.com/amitabhajoy/bengaluru-house-price-data). To complete this ML project we are using the supervised machine learning regression algorithm. Then i will deploy the model using Flask in order to be available for the end-users so they can make use of it.*

## Introduction
Predictive models for determining the sale price of houses in cities like Bengaluru is still remaining as more challenging and tricky task. The sale price of properties in cities like Bengaluru depends on a number of interdependent factors. Key factors that might affect the price include area of the property, location of the property and its amenities.
In this project, I first build a model using sklearn and linear regression using banglore home prices dataset from Kaggle. Second step is to save the trained model and deployed on Flask server, so this server can use the model to respind http requests. Third component is to build an web page in html, css and javascript that allows user to enter home square ft area, bedrooms etc and it will call python flask server to retrieve the predicted price.
During this project I covers almost all data science concepts such as data loading and cleaning, outlier detection and removal, feature engineering, dimensionality reduction, gridsearchcv for hyperparameter tunning, k fold cross validation etc.

## Data:
The train and test data will consist of various features that describe that property in Bengaluru.
Each row contains fixed size object of features. There are 9 features and each feature can be accessed by its name.
<br>

**Features:**
- Area_type – describes the area
- Availability – when it can be possessed or when it is ready(categorical and time-series)
- Location – where it is located in Bengaluru (Area name)
- Size – in BHK or Bedroom (1-10 or more)
- Society – to which society it belongs
- Total_sqft – size of the property in sq.ft
- Bath – No. of bathrooms
- Balcony – No. of the balcony
- Target variable:
- Price – Value of the property in lakhs(INR)
<br>

The **Dataset** is composed with 13320 records.
Since there is only 90 Null values, I first cleaned data by droping all nan values

<br>

## Machine learning model
Modeling explorations apply some regression techniques such as multiple linear regression (Least Squares), Lasso and Ridge regression models, support vector regression, and boosting algorithms such as Extreme Gradient Boost Regression (XG Boost).
Such models are used to build a predictive model, and to pick the best performing model by performing a comparative analysis on the predictive errors obtained between these models. Here, the attempt is to construct a predictive model for evaluating the price based on factors that affects the price.


## Technology and tools used for this project covers :

- Python
- Numpy and Pandas for data cleaning
- Matplotlib for data visualization
- Sklearn for model building
- Jupyter notebook, visual studio code as IDE
- Python flask for http server
- HTML/CSS/Javascript for UI




{% if page.mathjax %}
<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>
{% endif %}
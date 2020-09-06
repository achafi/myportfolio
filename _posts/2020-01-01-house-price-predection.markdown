---
title : "End to end machine learning project to predict house prices"
date : 2020-01-01
tags : [machine learning, opencv, CNN, python]
header :
  image : "./assets/images/housepriceprediction/app.PNG"
excerpt : "Machine learning, CNN, python"
mathjax : true
---
[Source code](https://github.com/achafi/Banglore_House_Prices)

*In this project, I am going to predict the price of a house using its 9 features. Basically i am solving the [Kaggle Competition](https://www.kaggle.com/amitabhajoy/bengaluru-house-price-data). To complete this ML project we are using the supervised machine learning regression algorithm. Then i will deploy the model using Flask in order to be available for the end-users so they can make use of it.*

## Introduction
Predictive models for determining the sale price of houses in cities like Bengaluru is still remaining as more challenging and tricky task. The sale price of properties in cities like Bengaluru depends on a number of interdependent factors. Key factors that might affect the price include area of the property, location of the property and its amenities.
In this project, I first build a model using sklearn and linear regression using banglore home prices dataset from Kaggle. Second step is to save the trained model and deployed on Flask server, so this server can use the model to respind http requests. Third component is to build an web page in html, css and javascript that allows user to enter home square ft area, bedrooms etc and it will call python flask server to retrieve the predicted price.
During this project I covers almost all data science concepts such as data loading and cleaning, outlier detection and removal, feature engineering, dimensionality reduction, gridsearchcv for hyperparameter tunning, k fold cross validation etc.

## Data Preprocessing :
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
Since there are only 90 Null values, I first cleaned data by dropping all null values.
I added new features :
Price per square feet
Bhk : Bedrooms Hall Kitchen

#### Dimensionality Reduction
After examination of location feature which is a categorical variable. We noticed that there are 1287 locations. For that I applied dimensionality reduction technique to reduce the number of locations. Any location having less than 10 data points should be tagged as "other" location. In this way, the number of categories can be reduced by a huge amount.
```python
df.location = df.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
len(df5.location.unique())
```
We passed from 1287 location to only 241 locations. Later on when we do one hot encoding, it will help us with having fewer dummy columns.
#### Outlier Removal using Business Logic
I started by assuming that normally square ft per bedroom is 300 (i.e. 2 bhk apartment is minimum 600 sqft. If you have for example 400 sqft apartment with 2 bhk than that seems suspicious and can be removed as an outlier.
I retain that hypothesis until the facts make it unlikely and then I will remove such outliers by keeping our minimum threshold per bhk to be 300 sqft.

```python
df[df.total_sqft/df.bhk<300].head()
```
#### Outlier Removal Using Standard Deviation and Mean
```python
df.price_per_sqft.describe()
```
count     12456.000000 <br>
mean       6308.502826 <br>
std        4168.127339 <br>
min         267.829813 <br>
25%        4210.526316 <br>
50%        5294.117647 <br>
75%        6916.666667 <br>
max      176470.588235 <br>
Name: price_per_sqft, dtype: float64<br>
Here we find that min price per sqft is 267 rs/sqft whereas max is 12000000, this shows a wide variation in property prices. We should remove outliers per location using mean and one standard deviation. for that i developped a function : remove_pps_outliers then applied to the dataframe.
```python
def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<=(m+st))]
        df_out = pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out
```

Before going forward, I first visualise for a given location how the 2 BHK and 3 BHK property prices look like.

<img src="{{ site.url }}{{ site.baseurl }}/assets/images/housepriceprediction/the2before.png" alt="">
*Fig. 1:  scatter chart to visualize price_per_sqft for 2 BHK and 3 BHK properties for both location "Rajaji Nagar" and "Hebbal"*

From the two plots above, we notice that from the same location, there are some houses with 2 BHK and more expensive than those with 3 BHK.
For that reason I should remove properties where for the same location, the price of (for example) 3 bedroom apartment is less than 2 bedroom apartment (with same square ft area). What I will do is for a given location, I will build a dictionary of stats per BHK, i.e.
{
    '1' : {
        'mean': 4000,
        'std: 2000,
        'count': 34
    },
    '2' : {
        'mean': 4300,
        'std: 2300,
        'count': 22
    },
}

Now we can remove those 2 BHK apartments whose price_per_sqft is less than mean price_per_sqft of 1 BHK apartment. The function I developed to do that process for all the data and all the numbers of BHK that may exists is as following :

```python
def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')
```
Let's plot the same scatter chart again to visualize price_per_sqft for 2 BHK and 3 BHK properties.
<img src="{{ site.url }}{{ site.baseurl }}/assets/images/housepriceprediction/the2after.png" alt="">
*Fig. 2:  scatter chart to visualize price_per_sqft for 2 BHK and 3 BHK properties for both location "Rajaji Nagar" and "Hebbal" after removing outliers*

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
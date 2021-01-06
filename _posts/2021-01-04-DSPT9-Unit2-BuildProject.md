---
layout: post
title: Modeling Online News Popularity
subtitle: Predicting Article Popularity
tags: [datasets, visualization]
---

This is a post about my project  on modeling the popularity of online news articles. The work uses the dataset from the [Online News Popularity project](https://archive.ics.uci.edu/ml/datasets/online+news+popularity) that collected data from articles published on Mashable, between January 7 2013 to January 7 2015. The data was the basis for research which resulted in the publication of a paper on ["A Proactive Intelligent Decision Support System for Predicting the Popularity of Online News"](https://www.researchgate.net/publication/283510525_A_Proactive_Intelligent_Decision_Support_System_for_Predicting_the_Popularity_of_Online_News)

The data is mostly clean but there was some work required to combine columns, that were essentially "One Hot Encoded", into  Categorical columns for  Data Channel and Day of the Week data as well some scripting to fill in some missing Data Channel values. Since the same dataset was used in the Unit1 Build Project, all of this cleanup work was already completed and described in "The Dataset" section of [Studying Online News Popularity](https://nsriniva.github.io/2020-10-23-DSPT9-Unit1-BuildProject/)

The original data cleanup code was used to create [CleanupOnlineNewsPopularity.ipynb](https://github.com/nsriniva/DS-Unit-2-Build/blob/main/CleanupOnlineNewsPopularity.ipynb) which stored the cleaned up dataset in a CSV file that was zipped and stored at [OnlineNewsPopularity.csv.zip](https://github.com/nsriniva/DS-Unit-2-Build/blob/main/OnlineNewsPopularity.csv.zip) - this was then used as the dataset for this project.

The code for this project is available at [OnlineNewsPopularity.ipynb](https://raw.githubusercontent.com/nsriniva/DS-Unit-2-Build/master/OnlineNewsPopularity.ipynb). 

A blog post on this work has been published as an article on Medium - [Studying Online News Popularity](https://srini-nariangadu.medium.com/studying-online-news-popularity-8bbf2fb3f89b).

## The Dataset

The compressed CSV file was loaded into a Pandas dataframe, initial observations of the dataset revealed that it is a large dataset with 39644 observations of 2 non-predictive(**url**, **timedelta**), 47 predictive attributes and 1 target attribute (**shares** == the number of views for the article). 

The problem was changed to one of Classification by creating a new target attribute (**popularity** == 1 if **shares** > median/1400 else 0) with 2 classes, `popular(1)` and `unpopular(0)`. The distribution of **popularity** values was reasonably balanced with 53% `popular(1)` and 47% `unpopular(0)`.

The **url**, **timedelta** and **shares** attributes were then dropped from the dataset.

## Data Modeling
Since the distribution of **popularity** values was balanced, **accuracy** makes a good evaluation metric with the baseline accuracy value being the percentage of the largest class, expressed as a fraction i.e. the baseline accuracy for the entire dataset would be 0.53. 

### Partitioning
The `X` dataframe was created by dropping the target(**popularity**) attribute and the `y` vector from the target attribute column.
Since the dataset is large, `sklearn.model_selection.train_test_split` was used twice to split it into `X_train/y_train`(64%:25372), `X_val/y_val`(16%:6343) and `X_test/y_test`(20%:7929) datasets.

### Linear Model - Logistic Regression(LogisticRegression with SelectKBest) 

#### Evaluation Metrics

### Tree Based Model - Decision Tree(DecisionTreeClassifier)

#### Evaluation Metrics

### Tree Based Model - Random Forest(RandomForestClassifier)

#### Evaluation Metrics

### Tree Based Model - Gradient Boosting(XGBoost)

#### Evaluation Metrics

### Comparative Evaluation Metrics

The Random Forest model exhibited the best behavior, closely followed by Gradient Boost and Linear models with the Decision Tree model trailing behind.  

| Model | Accuracy | Baseline | AUC |
|-------|--------|---------|----|
| Linear | 0.66/0.66/0.66 | 0.54/0.53/0.0.52 | 0.71/0.71/0.71|
| Decision Tree | 0.67/0.64/0.64 | 0.54/0.53/0.0.52 | 0.73/0.69/0.68|
| Random Forest | 1.00/0.67/0.66 | 0.54/0.53/0.0.52 | 1.00/0.73/0.72|
| Gradient Boost | 1.00/0.66/0.65 | 0.54/0.53/0.0.52 | 1.00/0.72/0.71|

---
layout: post
title: Studying Online News Popularity
subtitle: The What and When of Article Popularity
tags: [datasets, visualization]
---

This is a post about my DS-Unit1 Build project which is based on the dataset from the [Online News Popularity project](https://archive.ics.uci.edu/ml/datasets/online+news+popularity) which collected data from articles on Mashable, published between January 7 2013 to January 7 2015 - the data was the basis for research which resulted in the publication of a paper on ["A Proactive Intelligent Decision Support System for Predicting the Popularity of Online News"](https://www.researchgate.net/publication/283510525_A_Proactive_Intelligent_Decision_Support_System_for_Predicting_the_Popularity_of_Online_News)

## The Dataset
The dataset is contained in a **csv** file that was packaged into a **zip** file along with the associated **names** file describing the dataset. If the csv file were the only file in the zip package, Pandas' **read_csv** method could be used to extract the data set. However, the presence of the additional **names** file made that impossible and it was necessary to use the  **io** and **zipfile** packages to get the file readable by **read_csv**.
Once loaded into a Pandas dataframe, initial observations of the dataset revealed that it is a large dataset with 39644 observations of 58 predictive attributes and 1 target attribute **shares**(the number of views for the article). 

### Processing the data


### Missing data - BeautifulSoup to the rescue


This is a demo post to show you how to write blog posts with markdown.  I strongly encourage you to [take 5 minutes to learn how to write in markdown](https://markdowntutorial.com/) - it'll teach you how to transform regular text into bold/italics/headings/tables/etc.



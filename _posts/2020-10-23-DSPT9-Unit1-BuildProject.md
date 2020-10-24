---
layout: post
title: Studying Online News Popularity
subtitle: The What and When of Article Popularity
tags: [datasets, visualization]
---

This is a post about my DS-Unit1 Build project which is based on the dataset from the [Online News Popularity project](https://archive.ics.uci.edu/ml/datasets/online+news+popularity) that collected data from articles on Mashable, published between January 7 2013 to January 7 2015.

The data was the basis for research which resulted in the publication of a paper on ["A Proactive Intelligent Decision Support System for Predicting the Popularity of Online News"](https://www.researchgate.net/publication/283510525_A_Proactive_Intelligent_Decision_Support_System_for_Predicting_the_Popularity_of_Online_News)

## The Dataset
The dataset is contained in a **csv** file that was packaged into a **zip** file along with the associated **names** file describing the dataset. If the csv file were the only file in the zip package, Pandas' **read_csv** method could be used to extract the data set. 

However, the presence of the additional **names** file made that impossible and it was necessary to use the  **io** and **zipfile** packages to get the file readable by **read_csv**.

```python
onp_url = 'https://github.com/nsriniva/DS-Unit-1-Build/blob/master/OnlineNewsPopularity.zip?raw=true'

# Open the zip file
zfile = urlopen(onp_url)

# Create an IO stream
zfile_mem = io.BytesIO(zfile.read())

# Extract data file from archive and load into dataframe
# The zip archive contains the data file 
# OnlineNewsPopularity/OnlineNewsPopularity.csv 
# and the info file 
# OnlineNewsPopularity/OnlineNewsPopularity.names.
# Use the zipfile package to open the archive and read the data file
with zipfile.ZipFile(zfile_mem) as zf:
  with zf.open('OnlineNewsPopularity/OnlineNewsPopularity.csv') as f:
    onp_df = pd.read_csv(f)
```    

Once loaded into a Pandas dataframe, initial observations of the dataset revealed that it is a large dataset with 39644 observations of 58 predictive attributes and 1 target attribute **shares**(the number of views for the article). 

The **datachannel**(type of article) and **weekday**(day of the week on which the article was published) attributes looked interesting and examining their relationship or lack thereof with the popularity of the article appeared to be a good idea. One problem was with the form in which this data was stored in the dataframe - one column/attribute for each **datachannel** and **weekday**. The first step was to transform the multiple attributes/columns into a single one for **datachannel** and **weekday**.
 
```python
# First, identify all the data_channel_is_*/weekday_is_* columns
data_channel_columns = list(filter(lambda x: x.startswith('data_channel_is_'), onp_df.columns))
weekday_columns = list(filter(lambda x: x.startswith('weekday_is_'), onp_df.columns))
```
~~~
['data_channel_is_lifestyle', 
 'data_channel_is_entertainment',
 'data_channel_is_bus',
 'data_channel_is_socmed',
 'data_channel_is_tech',
 'data_channel_is_world']

['weekday_is_monday',
 'weekday_is_tuesday',
 'weekday_is_wednesday',
 'weekday_is_thursday',
 'weekday_is_friday',
 'weekday_is_saturday',
 'weekday_is_sunday']
~~~

### Processing the data
Merging the data from the multiple **data_channel_is_\***/**weekday_is_\*** columns into a single **data_channel**/**weekday** was easily achieved.
However examining the resultant columns revealed a problem.
```python
display(onp_merged_df.data_channel.value_counts())
display(onp_merged_df.weekday.value_counts())
```
~~~
6    8427
5    7346
2    7057
3    6258
0    6134
4    2323
1    2099
Name: data_channel, dtype: int64
3    7435
2    7390
4    7267
1    6661
5    5701
7    2737
6    2453
Name: weekday, dtype: int64
~~~
The **data_channel** attribute values have the following mappings
~~~
OrderedDict([('misc', 0),
             ('lifestyle', 1),
             ('entertainment', 2),
             ('bus', 3),
             ('socmed', 4),
             ('tech', 5),
             ('world', 6)])
~~~
While the **weekday** column has good data for every row, it looks like 6134 entries do not have valid **data_channel** information. That's a lot of missing information so something that needed to be rectified, if possible.

### Missing data - BeautifulSoup to the rescue

Manually inspected a few of the urls with missing **data_channel** information and found that the data channel information was encoded in the **data-channel** attribute of the **hgroup** element.

```html
<hgroup class="channel page-header post-head" data-channel="entertainment" data-section="sec0=entertainment&amp;sec1=index&amp;sec2=">
<h2>Entertainment</h2>
...
...
...
</hgroup>
```
Once the information was located, it was a simple matter of using the requests and BeautifulSoup packages to extract it for all the entries that had missing **data_channel** information.

```python
def get_data_channel(url):
    global dcm_vals_next
    hgroup = BeautifulSoup(requests.get(url).content,'html.parser').find('hgroup')
    ret = 0
    if hgroup is not None:
        dc = hgroup['data-channel']
        if dc not in dcm_keys:
            dcm[dc] = dcm_vals_next
            dcm_keys.append(dc)
            dcm_vals_next += 1
        ret = dcm[dc]
    return ret

```
The **url** and **data_channel** columns were written out to [data_channel_df.csv](https://raw.githubusercontent.com/nsriniva/DS-Unit-1-Build/master/data_channel_df.csv), processed using [data_channel_cleanup.py](https://raw.githubusercontent.com/nsriniva/DS-Unit-1-Build/master/data_channel_cleanup.py) and the new **data_channel** info was written out to [data_channel_cleaned_df.csv](https://raw.githubusercontent.com/nsriniva/DS-Unit-1-Build/master/data_channel_cleaned_df.csv)

The new **data_channel** info was merged into the **onp_merged_df** dataframe and the **data_channel** column data was examined.

```python
display(onp_merged_df.data_channel.value_counts())
```
~~~
6    9513
2    8384
5    8219
3    6980
4    2323
1    2099
7    1917
8     138
9      46
0      25
Name: data_channel, dtype: int64
~~~

Three new **data_channel** types have been added resulting in a new mapping
~~~
OrderedDict([('misc', 0),
             ('lifestyle', 1),
             ('entertainment', 2),
             ('bus', 3),
             ('socmed', 4),
             ('tech', 5),
             ('world', 6),
             ('culture', 7),
             ('u.s.', 8),
             ('social-good', 9)])
~~~

Unfortunately, we're still left with a few(25) entries with no **data_channel** information.

At this stage, the data has been sufficiently cleaned and is ready for the next step.

## Data Analysis

The initial focus of the analysis was to examine the effect of removing outliers on the accuracy of Linear Regression models, where accuracy is measured by computing the Mean Absolute Error(MAE).

Once that was done, the focus of the analysis was to partition the dataset into low, medium and high popularity articles and examine, for each partition, the

1. correlation matrices for differences/similarities

2. relationship between the type of article and popularity

3. relationship between the day of the week the article was published and popularity

### Linear Regression and removing outliers

![Linear Regression](https://raw.githubusercontent.com/nsriniva/DS-Unit-1-Build/master/LinearRegression.png)

### Partitioning the dataset and Popularity

### Correlation Matrix

![Correlation](https://raw.githubusercontent.com/nsriniva/DS-Unit-1-Build/master/Correlation.png)

### Data Channel Type/Popularity

![DataChannel](https://raw.githubusercontent.com/nsriniva/DS-Unit-1-Build/master/DataChannel.png)

### Weekday/Popularity

![Weekday](/assets/img/Weekday.png)

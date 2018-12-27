---
title: "Deep Learning with Structured and Unstructured Data with FastAI - Part 1: Environment Setup and Data Preparation"
date: 2018-12-18T15:14:19-05:00
draft: false
tags: [machine learning, mercari]
---
## Introduction
Data comes in various forms such as images, text, and tabular form. Deep learning can be applied to each of these areas and has excelled by giving state-of-art results. In this blog post series, I'm going to explore how to apply Deep Learning to a mixture of data components, specifically, text data and tabular data. This is part of a bigger research project that I'm working on, which uses medical data (excluding images) which often consists of different types of data. 

I want to see how each data component contributes to the performance of the end model. To do this, I am going to build models that only utilize structured, unstructured, and a combination of the both the data and compare their performance.

Structured data are presented in tabular form, examples of which include a product category or a diagnosis code for a suspected medical condition. Unstructured data include free-text such description of a product or a clinical note written by a doctor. Note that I'm excluding images as part of this projects though they come under unstructured data.

I'm breaking up this series into 6 parts:

1. Environment and Data Preparation (this post)
2. Building Model with Structured Data
3. Fine-Tuning Language Model with Unstructured Data
4. Building Model with Unstructured Data
5. Building Model with Full Dataset
6. Compare, Contrast, and Discuss Model Performance

I will be using the [FastAI](https://docs.fast.ai/) library for all my coding. The latest version of the library facilitates representing data easily and building state-of-art simple models quickly. You can participate in the FastAI [forums](http://forums.fast.ai), where people seek help and provide support. 

## Dataset
As I mentioned earlier, medical data often include structured and unstructured information together. Typically, when deep learning models have more data to work with, they provide better results. Unfortunately, medical data is very messy and there are privacy concerns in accessing them. So, when I started on this project, I wanted to build my architecture using a dataset that has been cleaned and ready to use. So I turned to [Kaggle](www.kaggle.com)!

Surprisingly, I couldn't find many datasets that actually included both structured and unstructured information together. There were plenty of image datasets, NLP datasets that contained only text, and tabular datasets. After a bit of searching, I found a dataset that fit my needs: [Mercari Price Suggestion Challenge](https://www.kaggle.com/c/mercari-price-suggestion-challenge). This competition concluded 10 months ago.

[Mercari](https://www.mercari.com/) is Japan&#146;s biggest community-powered shopping app. Sellers post items that they want to sell along with an asking price. Mercari offers pricing suggestions to them. The objective of this competition is to build an algorithm that automatically suggests the right product prices. The data provided including user-inputted text descriptions of their products (unstructured data component), including details like product category name, brand name, and item condition (structured data component). The data can be downloaded using the [Kaggle API](https://github.com/Kaggle/kaggle-api) software using the command `Kaggle competitions download -c Mercari-price-suggestion-challenge`.

This is a *Kernels only* competition, wherein you only provide a kernel (aka a script/code) which is then run on Kaggle servers to produce output and compare results. My objective for working with this dataset is not necessarily to get a good score. It's more so to learn how to build an end-to-end architecture with the FastAI library.

## Environment Setup
I first created a new basic Anaconda environment using `conda env create -yn mer python=3.7`. The FastAI [Github README](https://github.com/fastai/fastai/blob/master/README.md#installation) has information to do a basic install. But I have access to a machine that uses CUDA10. Furthermore, I always use the developer version, as the library is being actively developed with new features and bug fixes coming rapidly. Thus, having an updated developer version is helpful. I also use [JupyterLab](https://jupyterlab.readthedocs.io/en/stable/) along with some helpful extensions. For the initial setup, I've created a bash script that installs and sets everything up. The script can be found in [this](https://gist.github.com/sudarshan85/74569c627be99ddfd48586b71e6a8b9b) gist.

## Data Preparation
My data preparation notebook can be found [here](http://nbviewer.jupyter.org/github/sudarshan85/kaggle-mercari/blob/master/Data-Prep.ipynb). 

### Details

The data consists of 3 tab separated files:

|     Filename    | Size (MB) | No of Records |
|:---------------:|:---------:|:-------------:|
|   `train.tsv`   |    323    |    1482535    |
|    `test.tsv`   |    148    |     693359    |
| `test_stg2.tsv` |    737    |    3460725    |
Note that since, I'm not really competing in the competition, I don't the data in the test sets except for their `item_description` for building the language model.

There are 8 columns in the `train.tsv`:

|         Name        |     Type    | Cardinality | Missing |                                                                                 Details (from Kaggle)                                                                                |
|:-------------------:|:-----------:|:-----------:|:-------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|      `train_id`     |  Continuous |     N/A     |    0    |                                                                                 The ID of the listing                                                                                |
|        `name`       |     Text    |     N/A     |    0    |      The title of the listing. Note that we have cleaned the data to remove text that look like prices (e.g.$20) to avoid leakage. These removed prices are represented as [rm]      |
| `item_condition_id` | Categorical |      5      |    0    |                                                                   The condition of the items provided by the seller                                                                  |
|   `category_name`   | Categorical |     1287    |   6327  |                                                                                Category of the listing                                                                               |
|     `brand_name`    | Categorical |     4809    |  632682 |                                                                                          N/A                                                                                         |
|       `price`       |  Continuous |     N/A     |    0    | The price that the item was sold for. This is the target variable that you will predict. The unit is USD. This column doesn't exist in test.tsv since that is what you will predict. |
|      `shipping`     |    Binary   |      2      |    0    |                                                                  1 if shipping fee is paid by seller and 0 by buyer                                                                  |
|  `item_description` |     Text    |     N/A     |    4    |                                                                   The condition of the items provided by the seller                                                                  |
The test data has the same columns except the `price` column which is the dependent variable, i.e., the target to be predicted. The FastAI library by default adds 1 to the cardinality to compensate for missing values (even if there aren't any).

### Preprocessing
I do a couple of preprocessing steps to the entire dataset. These steps are available in the notebook. Basically, I do the following:

* Remove records with prices `< $3` as Mercari does not allow postings in that price range.
* Extract 3 sub categories from `category_name`. For example, a category name "Men/Tops/T-shirts" is actually made up of a "main" (`main_cat`) category "Men" and two sub categories (`sub_cat1` and `sub_cat2`) "Tops" and "T-shirts". I extract them and add them as new columns to the dataset. These will also be categorical variables.
* Replace `na`'s in the `item_description` with the word "missing", as FastAI library doesn't like `nan`'s for text during langauge modelling.

Below are the details of the extracted categorical variables:

|    Name    | Cardinality |
|:----------:|:-----------:|
| `main_cat` |      10     |
| `sub_cat1` |     113     |
| `sub_cat2` |     870     |

These have same number of missing values as `category_name`. If we look at the information in the `category_name` column, we can see that almost all the information from that column is captured in the 3 new columns `['main_cat', 'sub_cat1', 'sub_cat2]`. So there is really no need to keep `category_name` column (in fact this what I originally did). However, after thinking about and following Jeremy Howard's [suggestion](https://v637g.app.goo.gl/4e2GBBkUJhWtL2Qr8) of having more columns, I decided to leave it in there. If we think about it, it makes sense to have it in there because it provides information like how certain categories occur together which might be helpful for our algorithm.

#### Regression as Classification
As of this writing, the FastAI library does not support a regression type problem with a language model out of the box. While, I could try to write a custom module which did that, I have decided to take an easier approach to utilize the already present API for classification. I decided to convert the regression problem of predicting the price of a product, into a classification problem of predicting a range of values (category) within which a product's price might belong. Now, if I model this as a classification problem for use with unstructured data, then that has to be done with structured data as well, as one of the primary goals of this blog post series is to compare and contrast performance differences between using various components of the data.

For converting this into a classification problem, I followed similar steps given [here](http://fastml.com/regression-as-classification/). I used to create labels (or categories) in the `log1p` scale of the price, since that allowed more granularity. My objective here is to get a bunch labels which can be assigned to each record in the data, based on where the `log1p` of the price falls into. For this, I needed to determine my labels first. These are the steps I followed (keep in mind that `price` values here are in the `log1p` scale) :

1. Create an array of range of values that went from the `price.min()` to `price.max()` with an interval of 0.2. This gave me 33 values.
2. Create a dictionary which mapped the label names to the number of records with that label. Label names are strings which correspond to the value (rounded to 1 decimal place) of the value in the array. For the first and last values I simply had a string that said '<= value' and '> value'. This gave me 32 labels.
3. Look at the number of records and make appropriate judgment calls to merge certain labels depending on how many records belong that label.
4. Merge labels according to rule establish in the previous step. After this I was left with 21 labels with value of price ranging from less than 1.6 to greater than 5.6.
5. Write a function that takes in a price value determines and returns the appropriate label.
6. Create a new column `labels` in the training dataset by applying the function created in the previous step to the `log1p(price)` column.
7. Run sanity checks to figure out that all rows had labels and the number of records with labels added up to the length of the training dataframe.

The table below shows the labels and the number of records belong to each label:

| Label `log1p(price)` | # Records |
|:--------------------:|:---------:|
|        <= 1.6        |   18,703  |
|        1.6-1.8       |   47,641  |
|        1.8-2.0       |   32,293  |
|        2.0-2.2       |  113,882  |
|        2.2-2.4       |  163,096  |
|        2.4-2.6       |  118,823  |
|        2.6-2.8       |  171,588  |
|        2.8-3.0       |  166,750  |
|        3.0-3.2       |  127,084  |
|        3.2-3.4       |  131,882  |
|        3.4-2.6       |  114,872  |
|        3.6-3.8       |   73,038  |
|        3.8-4.0       |   61,827  |
|        4.0-4.2       |   45,638  |
|        4.2-4.4       |   30,331  |
|        4.4-4.6       |   19,044  |
|        4.6-4.8       |   14,855  |
|        4.8-5.0       |   9,626   |
|        5.0-5.2       |   7,628   |
|        5.2-5.6       |   7,881   |
|         > 5.6        |   5,179   |

Once I had the `labels` column, the data was prepared for classification with 21 classes.

### Dataset Creation
I created 4 datasets from my data for each of the various types of modeling.

1. Language Modelling: Texts can be made up of different fields such as "Title", "Abstract" etc. Knowing that a piece of text belongs to a fields can be useful information. In this dataset, I thought that knowing what name a seller gives to a product and their item description are useful to know. So, I created a dataset that consists of two columns `name` and `item_description` which contained the corresponding information from **ALL** three datasets.
2. To be able to compare the performances of different models, I need to have the same training/testing data for all of them. I create a custom training and testing set with a 90/10 split. This is the data set that contains all the data which I will be using for building the final model architecture that consumes both the dataset.
3. Structured Data: I include `['train_id', 'item_condition_id', 'category_name', 'brand_name', 'price', 'shipping', 'main_cat', 'sub_cat1', 'sub_cat2']` as part of my structured variables and create a structured training and testing dataset from the custom sets created earlier.
4. Unstructured Data: I include `['train_id', 'name', 'price', 'item_description']` as part of my unstructured variables and create a structured training and testing dataset from the custom sets created earlier.

After all the processing, I had 1334332 samples of training data and 148203 samples of testing data.

### Error Function
The competition [uses](https://www.kaggle.com/c/mercari-price-suggestion-challenge#evaluation) Root Mean Squared Logarithmic Error (RMSLE) as the judging metric. This is calculated as:

$\epsilon$ = $\sqrt{\frac{1}{n}\sum(log(p_i+1)-log(a_i+1))^2}$

where,
* $\epsilon$ is the RMSLE value (score)
* *n* is the total number of observations in the dataset
* $p_i$ is the predicted price for product *i*
* $a_i$ is the actual sale price for product *i*
* $log(x)$ is the natural logrithm of *x*

## Conclusion
In this post, I setup the problem I'm trying to solve, selected the dataset, and preprocessed it to get it ready for modeling. In the next post, I'll use FastAI's tabular learner to tackle the structured data and predict the product price only using that data component.

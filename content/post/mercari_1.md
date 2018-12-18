---
title: "Deep Learning with Structured and Unstructured Data with FastAI - Part 1: Environment and Data Preparation"
date: 2018-12-18T15:14:19-05:00
draft: false
tags: [machine learning, fastai, mercari]
---
## Introduction
Data comes in various forms such as images, text, and tabular form. Deep learning can be applied to each of these areas and has excelled by giving state-of-art results. In this blog post series, I'm going to explore how to apply Deep Learning to a mixture of data components, specifically, text data and tabular data. This is part of a bigger research project that I'm working on, which uses medical data (excluding images) which often consists of different types of data. 

I want to see how each data component contributes to the performance of the end model. To do this, I am going to build models that only utilize structured, unstructured, and a combination of the both the data and compare their performance.

Structured data are presented in tabular form, examples of which include a product category or a diagnosis code for a suspected medical condition. Unstructured data include free-text such description of a product or a clinical note written by a doctor. Note that I'm excluding images as part of this projects though they come under unstructured data.

I'm breaking up this series into 6 parts:

1. Environment and Data Preparation (this post)
2. Building Model with Structured Data
3. Fine-Tuning Language Model with Unstrctured Data
4. Building Model with Unstructured Data
5. Building Model with Full Dataset
6. Compare, Contrast, and Discuss Model Performance

I will be using the [FastAI](https://docs.fast.ai/) library for all my coding. The latest version of the library facilitates representing data easily and building state-of-art simple models quickly. You can participate in the FastAI [forums](http://forums.fast.ai), where people seek help and provide support. 

## Dataset
As I mentioned earlier, medical data often include structured and unstructured information together. Typically, when deep learing models have more data to work with, they provide better results. Unfortunately, medical data is very messy and there are privacy concerns in accessing them. So, when I started on this project, I wanted to build my architecture using a dataset that has been cleaned and ready to use. So I turned to [Kaggle](www.kaggle.com)!

Surprisingly, I couldn't find many datasets that actually included both structured and unstructured information together. There were plenty of image datasets, NLP datasets that contained only text, and tabular datasets. After a bit of searching, I found a dataset that fit my needs: [Mercari Price Suggestion Challenge](https://www.kaggle.com/c/mercari-price-suggestion-challenge). This competition concluded 10 months ago.

[Mercari](https://www.mercari.com/) is Japan’s biggest community-powered shopping app. Sellers post items that they want to sell along with an asking price. Mercari offers pricing suggestions to them. The objective of this competition is to build an algorithm that automatically suggests the right product prices. The data provided including user-inputted text descriptions of their products (unstructured data component), including details like product category name, brand name, and item condition (structured data component). The data can be downloaded using the [Kaggle API](https://github.com/Kaggle/kaggle-api) software using the command `kaggle competitions download -c mercari-price-suggestion-challenge`.

This is a *Kernals only* competition, wherein you only provide a kernal (aka a script/code) which is then run on Kaggle servers to produce output and compare results. My objective for working with this dataset is not necessarily to get a good score. It's more so to learn how to build an end-to-end architecture with the FastAI library.

## Environment Setup
I first created a new basic Anaconda environment using `conda env create -yn mer python=3.7`. The FastAI [Github README](https://github.com/fastai/fastai/blob/master/README.md#installation) has information to do a basic install. But I have access to a machine that uses CUDA10. Furthermore, I always use the developer version, as the library is being actively developed with new features and bug fixes coming rapidly. Thus, having an updated developer version is helpful. I also use [JupyterLab](https://jupyterlab.readthedocs.io/en/stable/) along with some helpful extensions. For the initial setup, I've created a bash script that installs and sets everything up. The script can be found in [this](https://gist.github.com/sudarshan85/74569c627be99ddfd48586b71e6a8b9b) gist.

## Data Preparation
My data preparation notebook can be found [here](http://nbviewer.jupyter.org/github/sudarshan85/kaggle-mercari/blob/master/Data-Prep.ipynb). 

### Details

The data consists of 3 tab seperated files:

|     Filename    | Size (MB) | No of Records |
|:---------------:|:---------:|:-------------:|
|   `train.tsv`   |    323    |    1482535    |
|    `test.tsv`   |    148    |     693359    |
| `test_stg2.tsv` |    737    |    3460725    |
Note that since, I'm not really competing in the competition, I don't the data in the test sets except for their `item_descrption` for building the language model.

There are 8 columns in the `train.tsv`:

|         Name        |     Type    | Cardinality | Missing |
|:-------------------:|:-----------:|:-----------:|:-------:|
|      `train_id`     |  Continuous |     N/A     |    0    |
|        `name`       |     Text    |     N/A     |    0    |
| `item_condition_id` | Categorical |      5      |    0    |
|   `category_name`   | Categorical |     1287    |   6327  |
|     `brand_name`    | Categorical |     4809    |  632682 |
|       `price`       |  Continuous |     N/A     |    0    |
|      `shipping`     |    Binary   |      2      |    0    |
|  `item_description` |     Text    |     N/A     |    4    |
The test data has the same columns except the `price` column which is the dependent variable, i.e., the target to be predicted. The FastAI library by default adds 1 to the cardinality to compensate for missing values (even if there aren't any).

### Preprocessing
I do a couple of preprocessing steps to the entire dataset. These steps are available in the nobebook. Basically, I do the following:

* Remove records with prices `< $3` as Mercari does not allow postings in that price range.
* Extract 3 sub categories from `category_name`. For example, a category name "Men/Tops/T-shirts" is actually made up of a "main" (`main_cat`) category "Men" and two sub categories (`sub_cat1` and `sub_cat2`) "Tops" and "T-shirts". I extract them and add them as new columns to the dataset. These will also be categorical variables.
* Replace `na`'s in the `item_description` with the word "missing", as FastAI library doesn't like `nan`'s for text during langauge modelling.

### Dataset Creation
I created 4 datasets from my data for each of the various types of modeling.

1. Language Modelling: Texts can be made up of different fields such as "Title", "Abstract" etc. Knowing that a piece of text belongs to a fields can be useful information. In this dataset, I thought that knowing what name a seller gives to a product and their item description are useful to know. So, I created a dataset that consists of two columns `name` and `item_description` which contained the corresponding information from **ALL** three datasets.
2. To be able to compare the performances of different models, I need to have the same training/testing data for all of them. I create a custom training and testing set with a 90/10 split. This is the data set that contains all the data which I will be using for building the final model architecture that consumes both the dataset.
3. Structured Data: I include `['train_id', 'item_condition_id', 'category_name', 'brand_name', 'price', 'shipping', 'main_cat', 'sub_cat1', 'sub_cat2']` as part of my structured variables and create a structured training and testing dataset from the custom sets created earlier.
4. Unstructured Data: I include `['train_id', 'name', 'price', 'item_description']` as part of my unstructured variables and create a structured training and testing dataset from the custom sets created earlier.

After all the processing, I had 1334332 samples of training data and 148203 samples of testing data.

## Conclusion
In this post, I setup the problem I'm trying to solve, selected the dataset, and preprocessed it to get it ready for modeling. In the next post, I'll use FastAI's tabular learner to tackle the structured data and predict the product price only using that data component.

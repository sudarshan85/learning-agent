---
title: "Deep Learning with Structured and Unstructured Data with FastAI - Part 2: Structured Data Model"
date: 2018-12-21T16:37:16-05:00
draft: False
tags: [machine learning, mercari, structured]
---
## Introduction
This is my second post in a series of six exploring deep learning with structured and unstructured data with the [FastAI](docs.fast.ai) library. Be sure to check out my post on [data preparation]({{< ref "/post/mercari_1.md" >}}). In this post, I'm going to describe my efforts in building a deep learning model that only uses structured data. My notebook for building this model can be found [here](http://nbviewer.jupyter.org/github/sudarshan85/kaggle-mercari/blob/master/Structured-Pred.ipynb). Much of the material here, including code and ideas, are taken on FastAI's [notebook](http://nbviewer.jupyter.org/github/fastai/course-v3/blob/master/nbs/dl1/lesson6-rossmann.ipynb) on tabular data with the [Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales) Kaggle dataset and the paper titled [Entity Embeddings of Categorical Variables
](http://nbviewer.jupyter.org/github/fastai/course-v3/blob/master/nbs/dl1/lesson6-rossmann.ipynb) by Cheng Guo and Felix Berkhahn.

Structured data contain variables that can be represented in a tabular form. These include categorical variables which has a limited number of unique values (aka cardinality) or continuous variables which are just real numbers. While continuous variables can be fed directly into the input layer of a neural network (i.e., their values), categorical variables requires some special processing.

## Embeddings
The magic sauce of representing categorical variables into a manner suitable for a deep learning model is the concept of *embeddings*. Let's look at an example of a categorical variable from the dataset to understand exactly what embeddings are and how they are helpful. One of the categorical variables that was extracted during preprocessing is called `main_cat` which represents the main category that the item belongs too and has a cardinality of 11 (including `na`) i.e., there 10 unique `main_cat` categories in this variable. These are `['Men', 'Electronics', 'Women', 'Home', 'Sports & Outdoors', 'Vintage & Collectibles', 'Beauty', 'Other', 'Kids', 'Handmade', nan]`. These are given as `string` variables which can't be fed into the model directly. They have to be converted into numbers. 

One of the most common way of representing categorical variables is by using [*one-hot*](https://en.wikipedia.org/wiki/One-hot) encoding. In one-hot encoding, each category are given a unique index. Following this, each category is represented by a binary vector which has a single `1` value at the index corresponding to the index of the category. Thus, the size of the vector representing the category will be equal to the cardinality of the categorical variable. For example, the one-hot representation of the `main_cat` variable would be:

|        Category        | Index |           Vector          |
|:----------------------:|:-----:|:-------------------------:|
|           Men          |   0   | `[1,0,0,0,0,0,0,0,0,0,0]` |
|       Electronics      |   1   | `[0,1,0,0,0,0,0,0,0,0,0]` |
|          Women         |   2   | `[0,0,1,0,0,0,0,0,0,0,0]` |
|          Home          |   3   | `[0,0,0,1,0,0,0,0,0,0,0]` |
|    Sports & Outdoors   |   4   | `[0,0,0,0,1,0,0,0,0,0,0]` |
| Vintage & Collectibles |   5   | `[0,0,0,0,0,1,0,0,0,0,0]` |
|         Beauty         |   6   | `[0,0,0,0,0,0,1,0,0,0,0]` |
|          Other         |   7   | `[0,0,0,0,0,0,0,1,0,0,0]` |
|          Kids          |   8   | `[0,0,0,0,0,0,0,0,1,0,0]` |
|        Handmade        |   9   | `[0,0,0,0,0,0,0,0,0,1,0]` |
|          `nan`         |   10  | `[0,0,0,0,0,0,0,0,0,0,1]`|

During implementation the binary vector representing a single category would become a binary matrix of input values for the machine learning algorithm (think mini-batch of inputs).

While simple to implement, one-hot encoding has two major disadvantages:
1. By representing categories by binary vectors, the relationship between these categories is essentially lost. One-hot encoding a category only tells us whether it is present or not and nothing else. Why is that relevant? Because, these relationships are important for our algorithm to learn and one-hot encoding does not allow that learning. Think about the following examples:
    - `Men`, `Women`, and `Kids`. These categories are associated with people. 
    - `Beauty` and `Women`. Women, in general, tend to buy beauty items.
    - `Handmade` and `Vintage & Collectibles`. Handmade items may include crafts which are likely to be sold as collectibles.
2. By nature, binary matrices are sparse, which take lots of storage and results in unrealistic computational requirements.

Clearly, we need better way to convert this strings to numbers. Well, here's a crazy idea. Why not just give each category a bunch of random numbers. More specifically, represent each category with random vector of some size (we get to pick this size). The reason this is useful is because we can make that vector (or a matrix) *learnable*. In other words, this matrix is just like any other weight matrix in our neural network and can be learned through gradient descent.

This is called an **embedding** matrix. In a mathematical sense, these categories are *embedded* into a `d`-dimensional (`d` being the size of the vector) vector space, where they can learn relationships based from the data. Furthermore, each coordinate of that embedded vector would represent a feature of that category and these get updated as part of the learning process. Embedding is a powerful concept, most known for their use in text (like word embeddings which is basically the same concept on a bigger scale).

Every layer in a neural network is a result of a matrix multiplication. So is our embedding matrix. To get the embedding vector for a particular input, we just perform a dot product of its one-hot encoding with the embedding matrix and we get the corresponding vector. Interestingly, this dot product is the same as an array-lookup where given an index we just lookup its embedding vector. An *embedding layer* essentially does this. It does a array lookup to retrieve an embedding vector which then gets fed into the neural network. This is possible because looking up an embedding vector in an array is *mathematically* equivalent to a matrix product. Being just an array lookup it is fast and doesn't take up lots of storage by saving a bunch of zeros.

As an example, the following table shows how the categories (only showing a couple) might be embedded into a lower dimensional vector space:

|        Category        | Index |           Vector          |
|:----------------------:|:-----:|:-------------------------:|
|           Men          |   0   | `[0.0367, 0.1299, 0.6969, 0.9284, 0.0894, 0.5121]` |
|       Electronics      |   1   | `[0.4109, 0.3557, 0.6323, 0.6099, 0.6077, 0.4454]` |

Please note that what I show here is just an example. These values are random values and could maybe used for initializing the embedding vectors. These vectors (more specifically the embedding matrix) would be updated as part of training with gradient descent. Each coordinate of these vectors could represent a feature and the entire vector would be called the *feature vector*. This sort of representation of the data is also know as [distributed representation](https://www.districtdatalabs.com/nlp-research-lab-part-1-distributed-representations/).

How do we choose the dimension of the embedding vector? There is really no hard and fast rule this choice. Digging into the FastAI source code, I was able to find the [function](https://github.com/fastai/fastai/blob/master/fastai/tabular/data.py#L15) that calculates this for us:

```
def emb_sz_rule(n_cat:int)->int: return min(600, round(1.6 * n_cat**0.56))
```
Basically, we won't get ever get an embedding size of more than 600, which makes sense, since even word embeddings have only a size of 300-400 and there way more words than any categorical variable that we may come across. As for the other formula, I'm not sure how they came up with it, but it seems to work!

## Entity Embeddings of Categorical Variables
Here I'm going to give a brief review of the paper which is what the FastAI's Rossmann notebook is based upon. This paper is very well written and has one of the best abstracts I've read. Following are direct quotes from the paper's abstracts that basically tells us everything we need to know about the paper:

1. *"We map categorical variables in a function approximation problem into Euclidean spaces, which are the entity embeddings of the categorical variables."* We all know that neural networks are [universal function approximators](http://neuralnetworksanddeeplearning.com/chap4.html) and this sentence is just setting up the problem to be in that form so that they can use neural networks for solving it. Entity embeddings is just a fancy name they use for embeddings of categorical variables.
2. *"Entity embedding not only reduces memory usage and speeds up neural networks compared with one-hot encoding, but more importantly by mapping similar values close to each other in the embedding space it reveals the intrinsic properties of the categorical variables."* This pretty much sums up my explanations about the disadvantages of one-hot encodings and how embeddings overcomes them.
3. *"Entity embedding helps the neural network to generalize better when data is sparse and statistics is unknown, useful for datasets with lots of high cardinality features, where other methods tend to overfit."*
4. *"As entity embedding defines a distance measure for categorical variables it can be used for visualizing categorical data and for data clustering."* This is because embeddings are just vector and like any vector they can be projected onto 2D space (using techniques such as [PCA and t-SNE](https://medium.com/@luckylwk/visualising-high-dimensional-datasets-using-pca-and-t-sne-in-python-8ef87e7915b)). Once they are projected into 2D, we can plot using standard plotting tools.

In their introduction, they argue that, while neural networks can approximate any continuous and piece-wise continuous functions, it is not suitable to approximate arbitrary non-continuous functions as it assumes certain level of continuity in its general form. Structured data do not typically present themselves as having a continuous. However, with the correct representation, we can reveal the continuity of the data, thus increasing the power of neural networks to learn the data.

One-hot encodings has two shortcomings: *"First when we have many high cardinality features one-hot encoding often results in an unrealistic amount of computational. Second, it treats different values of categorical variables completely independent of each other and often ignores the informative relations between them."*

They define structured data as *"data collected and organized in a table format with columns representing different features (variables) or target values and rows representing different samples."* Their objective of entity embedding is to map discrete values to a multi-dimensional space where values with similar function output are close to each other. Once we have an embedding layer which connects to the one-hot encoding layer via an embedding matrix (this is basically an array lookup), we can train the weights of the matrix in the same way as the parameters of other neural networks.

After entity embeddings are used to represent all categorical variables, these are concatenated with any continuous variables and is fed like a normal input layer in a neural network with other layers built on top of it. *"In this way, the entity embedding layer learns about the intrinsic properties of each category, while the deeper layers form complex combinations of them."* This is figure from their paper that shows the architecture of their model:

![Image](/img/ee.png)

Other interesting aspects of their paper is that, they learned the embeddings of the categorical variables and used these embeddings as input to traditional machine learning models (random forest, xgboost, KNN) and got better performance than other forms of input. This indicates representing categorical data with embeddings learns and uses the intrinsic properties and relationships between the categories.

## Implementation
You can find my notebook [here](http://nbviewer.jupyter.org/github/sudarshan85/kaggle-mercari/blob/master/Structured-Pred.ipynb). Recall that objective here is given only the categorical variables in the data, we want to predict the price of a product. I pretty much followed the same steps that was done in the [Rossman notebook](http://nbviewer.jupyter.org/github/fastai/course-v3/blob/master/nbs/dl1/lesson6-rossmann.ipynb). Unlike Rossmann, my data does not have any continuous variables so, I just passed an empty list for creating my databunch. Most of the action happens here:

```
data_str = (TabularList.from_df(train_df, path=path, cat_names=cat_vars, cont_names=[], procs=[Categorify])
           .split_by_idx(get_rdm_idx(train_df))
           .label_from_df(cols=dep_var, label_cls=FloatList, log=True)
           .add_test(TabularList.from_df(test_df, path=path, cat_names=cat_vars, cont_names=[])) 
           .databunch())
```

This creates a new databunch with that is ready for training. `get_rdm_idx` is my own function that just returns a list of random indices that can be passed to set up a validation set. In the [previous]({{< ref "/post/mercari_1.md" >}}) post, I had a 10% cut out of my training data for my test set. From the rest 90%, I setup 20% for my validation set. My network architecture (including dropout values) are the same from the lesson. I also followed the same training procedure. After training for 15 epochs, I was able to get:

|      Metric     |   Value  |
|:---------------:|:--------:|
|  Training Loss  | 0.326233 |
| Validation Loss | 0.331211 |
|      RMSLE      | 0.538608 |

I also did inference on my test set and compared it against the actual values I had. For my test set, I got an RMSLE of 0.543. Please note that this value cannot really be compared to the leaderboard scores. Although if you did compare, its not very good (around 1100), which is to be expected as I haven't used the `item_description` information at all (which I will for my final model).

## Conclusion
In this post, I covered details about embeddings and how they help us get better performance in neural networks. I also went over details about the "Entity Embeddings of Categorical Variables" paper and my implementation of a deep learning model with FastAI with only structured data from the Mercari dataset. My next post would tackle the use of the unstructured data, specifically, the data `name` and `item_description` columns. In particular, next post will talk about fine-tuning a language model with this data using a pre-trained language model, which can then be used for the price prediction task.



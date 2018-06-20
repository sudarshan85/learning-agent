---
title: "Check Your Error Function"
date: 2018-06-20T14:28:42-04:00
draft: true
tags: [machine learning, tip]
---

The underlying primary objective of any machine learning algorithm is to reduce the value of the error function (sometimes known as a cost function). We determine how our algorithm is performing during development by applying it to a validation set and reporting the error. Consequently, we tune the parameters of our algorithm to reduce the validation error.

The choice of the error function is dependent on the application. But regardless of what we choose, we have to implement it correctly so that the values returned by it actually guide us. If our error function does not return a value that is reasonable, how would we know if it is because of bad implementation of our error function or whether our algorithm is performing badly. In other words, even if we have an algorithm that works perfectly, if our error function implementation is bad, then
we would still be getting bad values and making bad choices for our hyperparameters.

But how do we "test" the error function? One simple method is as follows:

1. Get the range of your dependent variable (in case of regression this is just the minimum and maximum values and in the case of classification this is just the different classes)
2. Generate two sets of random numbers (usually from an uniform distribution) within that range, one for our predictions and one for our "actuals".
3. Apply the error function to these values and get an estimate of the error.

One would assume that the error estimate returned using these inputs would fall within the ballpark of a reasonable error value. If we're working on a Kaggle competition, it would be easy to just check the leaderboard and compare the value returned with the values there. We must keep in mind though that these are random values so we should not expect to be in the top!

## Example

I put to test this procedure in a Kaggle competition to predict prices of an item. [Here](https://www.kaggle.com/c/mercari-price-suggestion-challenge) is the website for the competition. The competition is evaluated using Root Mean Squared Logarithmic Error (RMSLE). I couldn't find this implemented in any library, so I had to do a custom implementation.

```
def RMSLE(preds, targs):
    assert(len(preds) == len(targs))
        return np.sqrt(np.mean((np.log1p(1+preds) - np.log1p(1+targs))**2))
```

I then grabbed the range of the maximum and minimum price in the dataset and generated random values within this range for my predictions and actuals.

```
min_price, max_price = train_df['price'].min(), train_df['price'].max()
preds = np.random.uniform(min_price, max_price, 1000)
targs = np.random.uniform(min_price, max_price, 1000)
print(RMSLE(preds, targs))
```

I got the value `1.399`. I then checked the leaderboard. This error value looked reasonable compared to the values in there. As expected we were in the bottom, although there were several entries worse than this error estimate. This gave me a hint that this error function is returning reasonable values and if after implementation of my algorithm I get really bad results, it would be because of my algorithm and not because of a bad error function.

The example I've shown is for a regression problem but can be easily extendend to a classification problem. Once we are satisfied that our error function is behaving properly we can proceed to the next step of our machine learning pipline.


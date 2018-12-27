---
title: "Deep Learning with Structured and Unstructured Data with FastAI - Part 3: Language Model"
date: 2018-12-24T12:29:21-05:00
draft: false
tags: [machine learning, mercari, language model]
---
## Introduction
This is my third post in a series of six, exploring deep learning with structured and unstructured data with the [FastAI](https://docs.fast.ai/) library. These are the links to my earlier posts on [data preparation]({{< ref "/post/mercari_1.md" >}}) and [structured data model]({{< ref "/post/mercari_2.md" >}}). In this post, I'll be talking about language models (LM) and how I built a custom language model using the data from the `name` and `item_description` columns in the Mercari dataset using a pre-trained language model provided by FastAI. The material covered is based on FastAI's [IMDB notebook](http://nbviewer.jupyter.org/github/fastai/course-v3/blob/master/nbs/dl1/lesson3-imdb.ipynb).

I'll also be covering two papers [Regularizing and Optimizing LSTM Language Models](https://arxiv.org/abs/1708.02182) by Stephen Merity, Nitish Shirish Keskar, and Richard Socher and [Universal Language Model Fine-tuning for Text Classification](https://arxiv.org/abs/1801.06146) by Jeremy Howard and Sebastian Ruder. These papers talk about the recurrent neural network (RNN) variant on which the language model is built.

## Language Model & Transfer Learning
A LM is a model for solving a particular task in natural language processing (NLP). Given a bunch of words in a corpus a LM tries to predict the next word in the sentence. This particular task is not really that use full. But our objective here is not really the task itself, but what we get in terms of the weights of the neural network when we train the model for this task. We are interested in using these pre-trained weights on our target task. These weights form the LM and LMs represent and understand language and language structure.

Using LMs on text datasets for solving a target task is similar to using a pre-trained model such as ImageNet to classify custom dataset of images. It is a form of [transfer learning](ftp://ftp.cs.wisc.edu/machine-learning/shavlik-group/torrey.handbook09.pdf) which is the *"improvement of learning in a new task through the transfer of knowledge from a related task that has already been learned"*. This concept is very powerful because it helps us to train models faster and the resulting models are much more accurate. Unlike transfer learning in vision, in NLP transfer learning involves a few more steps.

For vision the general recipe for transfer learning involves (from FastAI):

1. Create model with the same architecture as the ImageNet ones such as resnet32.
2. Initialize the model with the pre-trained weights.
3. Freeze all the layers (i.e., block the gradients from backpropagating) except the final layer that is relevant to the task at hand.
4. Train for a few epochs.
5. Gradually unfreeze the lower layers and train for a few more epochs (with a lower learning rate).

For NLP the general recipe for transfer learning involves (from FastAI):

1. Create model with a LSTM language model architecture.
2. Initialize the model with the pre-trained weights.
3. Train the model for a few epochs.
4. Unfreeze all layers and train for a few epochs.
5. Save the *encoder* part of the model.
6. Create model for the target task with the same initial architecture as the AWD-LSTM and a custom head to reflect the task.
7. Load in the saved encoder and freeze the lower layers.
8. Train the model for a few epochs.
9. Gradually unfreeze lower layers and train the model for few more epochs.

As you can see, transfer learning in text is a bit more involved but it brings the same advantages that it has with vision to the table. 

We can't just directly use the pre-trained LM on our target task. This is because our target task's data will contain different vocabulary and language structure that is different compared to the pre-trained model (while in vision everything is a pixel). Therefore, we need to customize the pre-trained LM for our data, i.e., we have to *fine-tune* it. Once we have fine-tuned it, we can then grab the encoder, which is just the part of the network that takes the input and encodes it into the weights, and use it for our target task. The benefit here is we don't start our training from scratch (random numbers). Note that, we take our fine-tuned LM and fine-tune it further for our target task.

Finally, I want to talk about the difference between (*word2vec*) and language models, as this is something I struggled with innitally. *word2vec* is a linear model where we learn vectors for representing each word in a corpus and use these pre-trained vectors to initialize the words in our target corpus and fine-tune it for the target task. This is still transfer learning in principle but a very simplified form. A LM has way more information and influence than a word embedding does. LMs used in FastAI has 3 layer LSTM model with 1150 units in the hidden layer and an embedding size of 400, while *word2vec* typically contain 300-400 dimension vectors. Also, in word2vec, words are considered completely independent and language structure is not learned.

## AWD-LSTM
Due to their temporal nature and the need for capturing long-term relationships, LMs are usually modeled using RNNs or more specifically LSTM. However, LSTMs (and NNs in general) suffer from overfitting due to over-parametrization. Thus, it is important to regularize these models. While strategies such as dropout and batch normalization have found success in feed-forward and convolutional NN, they have not been successful in RNNs. Stephen Merity *et al.*, have proposed various regularization strategies for LSTM LMs. A detailed overview of their paper by [Yashi Seth](https://www.linkedin.com/in/yashuseth/) can be found [here](https://yashuseth.blog/2018/09/12/awd-lstm-explanation-understanding-language-model/). I will briefly go over their suggested regularization strategies here:

1. **Weight-Dropped LSTM**: Traditional dropout in a LSTM network is traditionally applied to the hidden state vector $h_{t-1}$, more specifically the activations (product of the matrix multiplications). These modifications prevent the use of black box LSTM-RNN implementations like *cudaNN*. Here, the authors propose the use of [DropConnect](https://cs.nyu.edu/~wanli/dropc/dropc.pdf), where dropout is applied to the recurrent hidden to hidden *weight matrices* (not the activations) which does not require any modifications to the RNN architecture. In fact, the dropout operation is applied once to the weight matrices, before the forward and backward pass. Since the same weights are reused over multiple timesteps, the same dropped weights remain dropped for the entirety of the forward and backward passes.
2. **NT-ASGD**: For LMs traditional SGD without any momentum tend to work best. In addition to SGD, averaging can be added by a triggering mechanism. When averaging is "triggered", instead of returning the last iterate during the SGD update, the algorithm returns an average of the last few iterates. This is called average SGD (ASGD) further improves the training the process. However, the snag here is how/when to trigger the averaging. The authors propose a non-monotonic criterion that conservatively triggers the averaging when the validation metric fails to improve for multiple cycles. They call this non-monotonic ASGD (NT-ASGD).
3. **Random BPTT**: Randomness makes for better learning as it introduces uncertainty which has to be dealt with by our model. Unfortunately, unlike in vision where can feed in random pictures (as each picture is independent for the most part with others), we can't just mix up words, as they need to be in order to make sense. The authors propose a trick to solve this problem. Instead of having a fixed backpropogation through time (BPTT), they provide a formula which selects the BPTT randomly for the forward and backward pass in two steps. Check out the linked page for the actual formula.
4. **Variational Dropout**: In standard dropout, a new binary dropout mask is sampled each and every time the dropout function is called. In "variational" dropout, a binary dropout mask is sampled only *once* upon the first call and the locked dropout mask is repeatedly used for all repeated connection within the forward and backward pass. Given the dropout mask is fixed, I think the name *variational dropout* is a misnomer. The authors use variational dropout for all inputs and outputs of the LSTM within a given forward and backward pass.
5. **Embedding Dropout**: They basically apply dropout to the embedding matrix which consists of word vectors for each word. When dropout is applied to this matrix, it means that all occurrences of specific words disappear. Since this only applied once for a full forward and backward pass, this is equivalent to performing variational dropout on the connection between the one-hot embedding and the embedding lookup.
6. **Weight Tying**: Same weights are shared between the embedding and the softmax layer which substantially reduces the total parameter count in the model.
7. **Independent Embedding & Hidden Size**: The authors say that having independent sizes for embeddings and hidden layers helps prevent overfitting. I'm not sure I understand how and the entire section that talks about this doesn't make sense to me.
8. **AR & TAR**: Activation regularization (AR) is $L_2$-regularization which is used on the weights of the network to control the norm of the resulting model and penalizes activations that are significantly larger than 0. Temporal activation regularization (TAR) is $L_2$ decay which is used on individual unit activations and on the difference in outputs of an RNN at different timesteps which penalize the model from producing large changes in the hidden state (also called *slowness* regularizers).

Their approach is which includes the various tricks mentioned above is called Average stochastic gradient descent Weight-Dropped LSTM or AWD-LSTM.

## ULMFit
Howard and Ruder have used this architecture along with their own bag of tricks and achieved state-of-art results in various NLP tasks. They create a transfer learning method for NLP called Universal Language Model FIne-Tuning (ULMFiT). The goal of inductive transfer learning is that *"Given a static source task $\tau_s$" and any target task $\tau_T$ with $\tau_s \ne \tau_t$, we would like to improve performance on $\tau_T$*. Their ULMFiT consists of a series of steps which I summarize below.

#### General-domain LM pretraining
They use Wikitext-103 provided by Stephen Merity *et al.* to build train their LM which will then be fine-tuned for target tasks. The Wikitext-103 consists of $28,595$ preprocessed Wikipedia articles and $103$ million words. This is the most expensive stage in terms of training time, but this has to be done only once. The resulting LM can then be used for downstream tasks.

#### Target task LM fine-tuning
The hypothesis here is that given a pre-trained general-domain LM, fine-tuning on a target task (meaning the data in for the target task) will converge faster as it only needs to adapt to the idiosyncrasies of the target data and it allows to train a robust task-specific LM even for small datasets. They also use a bunch of tricks to further enhance the training.

1. **Discriminative fine-tuning**: *"As different layers capture different types of information, they should be fine-tune to different extents. Instead of using the same learning rate for all layers of the model, discriminative fine-tuning allows us to tune each layer with different learning rates"*. Empirically, they found that decreasing the learning rate from the last layer by a factor of $2.6$.
2. **STLR**: *"For adapting its parameters to task-specific features, we would like the model to quickly converge to a suitable region of the parameter space in the beginning of training and then refine its parameters."* For this, instead of using the same learning rate throughout training, they used slanted triangular learning rates (STLR). This increases and decreases the learning rate according based on the stage of training. In particular, they provide a formula for the learning rate scheduler that does this automatically.

#### Target task classifier fine-tuning
Once the LM is fine-tuned to the target data, they augment the architecture with two linear blocks for the final classifier fine-tuning. Only the parameters of these classifier layers are learned from scratch. Fine-tuning the target classifier is the most critical part of transfer learning. If overly aggressive fine-tune is used, it will cause [catastrophic forgetting](https://en.wikipedia.org/wiki/Catastrophic_interference). Too cautious fine-tuning will lead to slow convergence. To combat these problems they use the following:

1. **Concat pooling**: Words are the main data source in NLP and as input documents contain lot of words, information may get lost if only the last hidden state of the model is saved. For solving this, they use concat pooling where along with the last hidden state, they keep both the max-pooled and the mean-pooled representation of the hidden states many time steps (subject to GPU memory). 
2. **Gradual unfreezing**: To address the catastrophic forgetting problem, they gradually unfreeze the layers during training instead of fine-tuning all the layers at once. In particular, they gradually unfreeze the model starting from the last layer as this contains the least general knowledge. *"We first unfreeze the last layer and fine-tune all unfrozen layers for one epoch. We then unfreeze the next lower frozen layer and repeat, until we fine-tune all layers until convergence at the last iteration."*
3. **BPT3C**: They use a specific BPTT technique for classification called BPTT for Text Classification (BPT3C) where they divide the document into fixed-length batches of size $b$. At the begininning of each batch, the model is initialized with the fine state of the previous batch while keeping track of all form of hidden states (max and mean pooled). The gradients are then backpropogated to the batches whose hidden states contributed to the final prediction.
4. **Bidirectional language model**: For increased performance, they train a backward model and take the average of the classifier predictions of both models.

Using all these techniques, they test their model on six datasets, on various different tasks: sentiment analysis, question classification, and topic classification. They use the same AWD-LSTM architecture with 3-layer LSTM of 1150 units each for the LM. The implementation of this LM and classifier is available as part of the FastAI library in the [text](https://docs.fast.ai/text.html) application.

## Implementation
My notebook for this can be found [here](http://nbviewer.jupyter.org/github/sudarshan85/kaggle-mercari/blob/master/Language-Model.ipynb). While I follow the same steps are FastAI's IMDB notebook, there are few differences in the preprocessing:

1. I have a custom `TokenizerProcessor` in which I pass `mark_fields=True`. This is to ensure that my two fields `name` and `item_description` are marked as `xfld 1` and `xfld 2` respectively in the preprocessed output.
2. I have a custom `NumericalizeProcessor`, where I pass a custom vocabulary size of `60,091`. No particular reason for this, I just like prime numbers :).

Here is the databunch creation code:
```
data_lm = (TextList.from_df(texts_df, path, cols=['name', 'item_description'], processor=[tok_proc, num_proc])
          .random_split_by_pct(0.1)
          .label_for_lm()
          .databunch())
```

FastAI's tokenization assigns special tokens for certain words/patterns. These are defined as rules can be found in the documentation [here](https://docs.fast.ai/text.transform.html#Rules).

Once I created my databunch, I created my learner and ran it for one epoch to fit the head. Following this, I unfroze all the layers and ran the model for 11 epochs all using the same learning rates used by FastAI's IMDB notebook. The entire process took about 30 hours to run on a V100 with 16G of RAM.

### Inference
Once I finished the learning process, I saved the entire model along with the encoder (for future use) and did some example inference.

`learn.predict('Unused computer keyboard!! Brand new', n_words=80)`

`Unused computer keyboard!! Brand new xxmaj box has a tear xxmaj it xxmaj could be used as a laptop or tablet i ca n't be using for all new items , but i found that it should be bought online for the new ones pop out again :) about the product ! xxup price xxup is xxup firm ! xxmaj do n't ask for a lower price as it wo n't last ! ! xxmaj thank you . xxmaj dramatically different film on my computer`

This is pretty cool! It figured out that keyboard has something to with laptop/tablet and it outputed text that has similar context. 

`learn.predict('Cute pair of jeans', n_words=50)`

`'Cute pair of jeans , these have xxmaj denim and featuring a xxmaj denim xxmaj shipped xxmaj out xxmaj within 24 xxmaj hours of purchase , xxmaj if you have any xxmaj questions please comment below xxbos xxfld 1 xxmaj curly lace front wig xxfld 2 xxmaj posting is for xxmaj boutique xxmaj style'`

Same thing, it realized `denim` has something to do with jeans have has output text with that meaning.

An important thing to note is that, we can either do inference in the same session as we've trained the LM or we can save the model, load it up later and do an inference. I found that to get this to work in a different session, I had to do some extra steps and struggled a bit. Also, I noticed that the output of the language model on a different session was very different from the output in the same session. The ones in the different session were very random without contextual relations. Also, when run on a different sessions, the special tokens such as `xxup`, `xxmaj` etc, does not appear. I'm not sure why we have that. FastAI's docs provide example for inference learner for LM [here](https://docs.fast.ai/tutorial.inference.html#Language-modelling). While following those instructions mostly work, there is one very important thing to remember. 

**When we create the databunch, we need to make sure to export the databunch by calling `data_lm.export()` that saves a file `export.pkl` in the path. This is the one that has to be used on subsequent steps. Otherwise, the inference procedue will NOT work.**

After this,  I just followed the instructions, created an empty data object, created a learner using the empty data object, unfroze it, loaded the trained model, and called the `predict` function.

`learn.predict('Unused computer keyboard!! Brand new', n_words=80)`

`Unused computer keyboard!! Brand new xxmaj box has a tear xxmaj it xxmaj could be used as a laptop or tablet i ca n't be using for all new items , but i found that it should be bought online for the new ones pop out again :) about the product ! xxup price xxup is xxup firm ! xxmaj do n't ask for a lower price as it wo n't last ! ! xxmaj thank you . xxmaj dramatically different film on my computer`

This is pretty cool! It figured out that keyboard has something to with laptop/tablet and it outputed text that has similar context. 

`learn.predict('Cute pair of jeans', n_words=50)`

`'Cute pair of jeans , these have xxmaj denim and featuring a xxmaj denim xxmaj shipped xxmaj out xxmaj within 24 xxmaj hours of purchase , xxmaj if you have any xxmaj questions please comment below xxbos xxfld 1 xxmaj curly lace front wig xxfld 2 xxmaj posting is for xxmaj boutique xxmaj style'
`

Same thing, it realized `denim` has something to do with jeans have has output text with that meaning.

## Conclusion
In this post, I talked about language models and how they are useful for NLP tasks. I went over two prominent papers based on which the FastAI library has built its LM code. I showed how the fine-tuned LM is able to predict text within context. In th next post, I will go over how to use this fine-tuned LM for the target task of predicting the price of an item in the Mercari dataset.

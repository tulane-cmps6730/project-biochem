# CMPS 6730 - Sentiment Analysis for Biochem Class Reviews

This project was inspired by my primary major: cell & molecular biology

Most college STEM courses are lecture style and emphasize memorization > understanding. One Tulane professor strives to break this mold, investing energy into creating an accessible classroom environment.

## **Goal**

This biochemistry professor understands their classroom policies may be divisive. To track student opinions, they survey several times a semester. My goal was to perform sentiment analysis on these reviews. Could I train a model to predict how a student feels about biochemistry class given textual feedback?

Through this assignment, I wanted to provide a professor I care about with a meaningful tool they could utilize to improve their classes for all.

## **Methods**

To perform sentiment analysis, I investigated a number of approaches. 

First, since I was working with raw survey data, I had to label each review as positive [label = 1] or negative [label = 0] and remove any identifying details. I initially started with survey data the afformentioned professor gave me + some Rate My Professor [RMP] reviews of the target professor [dataset of 53 sentences of reviews]. Later on, I expanded the dataset by supplementing my spreadsheet with RMP reviews of other Tulane professors who teach CELL courses.

Throughout my entire project, I ended up building models for sentiment analysis using:
- logistic regression
- HMMs [for POS modeling]
- an RNN model


## **Conclusions**

My major takeaway from this project is that working with raw, unlabeled text can be challenging! I had a hard time building models that accurately accomplished the task I planned.

Ultimately, with my RNN model, I was able to achieve a sentiment analysis [binary classification] accuracy of 0.71, providing evidence that this model might be worthwhile.

With more time to work on this assignment, I would hope to work with more data and continue designing models from scratch [+ take the time to tune the hyperparameters correctly]


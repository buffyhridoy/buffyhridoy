layout: post
title: Why Other Performance Matrics are Needed???
description: >
  This blog demonstrates why other performance matrics are needed along with accuracy.

canonical_url: http://Why Other Performance Matrics are Needed???
hide_image: false
accent_color: '#4fb1ba'
accent_image:
  background: 'linear-gradient(to bottom,#193747 0%,#233e4c 30%,#3c929e 50%,#d5d5d4 70%,#cdccc8 100%)'
  overlay:    true
---

In COVID times, people are often traveling from one place to another. 
The airport can be a risk as passengers wait in queues, check-in for flights, visit food vendors, and use facilities such as bathrooms. 
Tracking COVID positive passengers at airports can help prevent the spread of the virus

Consider, we have a machine learning model classifying passengers as COVID positive and negative. 
When performing classification predictions, there are four types of outcomes that could occur:

_**True Positive (TP):**_ When you predict an observation belongs to a class and it actually does belong to that class. 
In this case, a passenger who is classified as COVID positive and is actually positive.

_**True Negative (TN):**_ When you predict an observation does not belong to a class and it actually does not belong to that class. 
In this case, a passenger who is classified as not COVID positive (negative) and is actually not COVID positive (negative).

_**False Positive (FP):**_ When you predict an observation belongs to a class and it actually does not belong to that class. 
In this case, a passenger who is classified as COVID positive and is actually not COVID positive (negative).

_**False Negative(FN):**_ When you predict an observation does not belong to a class and it actually does belong to that class. 
In this case, a passenger who is classified as not COVID positive (negative) and is actually COVID positive.

## Confusion Matrix
For better visualization of the performance of a model, these four outcomes are plotted on a confusion matrix

![image](https://user-images.githubusercontent.com/37147511/164250502-ce80621a-463b-40a7-9e48-6ec981f9fac7.png)

## Accuracy
We want our model to focus on True positive and True Negative. 
Accuracy is one metric which gives the fraction of predictions our model got right. 
Formally, accuracy has the following definition:

Accuracy = Number of correct predictions / Total number of predictions.

![image](https://user-images.githubusercontent.com/37147511/164250741-ecf336f8-c184-4232-ae51-0d11c4ec60e2.png)

Now, let’s consider 50,000 passengers travel per day on an average. Out of which, 10 are actually COVID positive.

One of the easy ways to increase accuracy is to classify every passenger as COVID negative. So our confusion matrix looks like:

![image](https://user-images.githubusercontent.com/37147511/164251103-9fb2d1af-a7ce-4edf-9cd8-b4de8ed22582.png)

Accuracy for this case will be:

Accuracy = 49,990/50,000 = 0.9998 or 99.98%

Impressive!! Right? Well, does that really solve our purpose of classifying COVID positive passengers correctly?
For this particular example where we are trying to label passengers as COVID positive and negative with the hope of identifying the right ones, I can get 99.98% accuracy by simply labeling everyone as COVID negative. 
Obviously, this is a way more accuracy than we have ever seen in any model. But it doesn’t solve the purpose. The purpose here is to identify COVID positive passengers. 
Not labeling 10 of actually positive is a lot more expensive in this scenario. 
Accuracy in this context is a terrible measure because its easy to get extremely good accuracy but that’s not what we are interested in.
So in this context accuracy is not a good measure to evaluate a model. Let’s look at a very popular measure called Recall.

## Recall (Sensitivity or True positive rate)
Recall gives the fraction you correctly identified as positive out of all positives.

![image](https://user-images.githubusercontent.com/37147511/164251854-1b7e06ad-681d-4489-a9cc-2b01ca0536b3.png)

Now,this is an important measure. Out of all positive passengers what fraction you identified correctly. 
Going back to our previous strategy of labeling every passenger as negative that will give recall of Zero.

Recall = 0/10 = 0

So, in this context, Recall is a good measure. It says that the terrible strategy of identifying every passenger as COVID negative leads to zero recall. And we want to maximize the recall.



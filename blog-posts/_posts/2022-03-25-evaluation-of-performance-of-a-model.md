---
layout: post
title: Evaluation of Performance of a Model (Accuracy, Precision, Specifity, Recall/Sensitivity & F1 Score)
description: >
  This blog demonstrates how to evaluate the performance of a model via Accuracy, Precision, Specifity, Recall/Sensitivity & F1 Score metrics.

canonical_url: http://Evaluation of Performance of a Model
hide_image: false
accent_color: '#4fb1ba'
accent_image:
  background: 'linear-gradient(to bottom,#193747 0%,#233e4c 30%,#3c929e 50%,#d5d5d4 70%,#cdccc8 100%)'
  overlay:    true
---

Once we have built our model, the most important question that arises is how good is our model? 
So, evaluating our model is the most important task in the data science project which delineates how good our predictions are.

The first thing we will see here is ROC curve and we can determine whether our ROC curve is good or not by looking at AUC (Area Under the Curve) and other parameters which are also called as Confusion Metrics. 

A `confusion matrix` is a table that is often used to describe the performance of a classification model on a set of test data for which the true values are known. 
All the measures except AUC can be calculated by using left most four parameters. 
So, let’s talk about those four parameters first.
![image](https://user-images.githubusercontent.com/37147511/159992279-d241f064-f72b-4381-b7bc-4f14fbe441ad.png)

True positive and true negatives are the observations that are correctly predicted and therefore shown in green. 

We want to minimize false positives and false negatives so they are shown in red color. 

These terms are a bit confusing. So let’s take each term one by one and understand it fully.

**True Positives (TP)** - These are the correctly predicted positive values which means that the value of actual class is yes and the value of predicted class is also yes. Persons predicted as suffering from the disease (or unhealthy) are actually suffering from the disease (unhealthy); In other words, the true positive represents the number of persons who are unhealthy and are predicted as unhealthy.

**True Negatives (TN)** - These are the correctly predicted negative values which means that the value of actual class is no and value of predicted class is also no. Persons predicted as not suffering from the disease (or healthy) are actually found to be not suffering from the disease (healthy); In other words, the true negative represents the number of persons who are healthy and are predicted as healthy

**_False positives and false negatives, these values occur when our actual class contradicts with the predicted class._**

**False Positives (FP)** – When actual class is no and predicted class is yes. Persons predicted as suffering from the disease (or unhealthy) are actually found to be not suffering from the disease (healthy). In other words, the false positive represents the number of persons who are healthy and got predicted as unhealthy.

**False Negatives (FN)** – When actual class is yes but predicted class in no. Persons who are actually suffering from the disease (or unhealthy) are actually predicted to be not suffering from the disease (healthy). In other words, the false negative represents the number of persons who are unhealthy and got predicted as healthy. _**Ideally, we would seek the model to have low false negatives as it might prove to be life-threatening or business threatening**_

Once we understand these four parameters then we can calculate Accuracy, Precision, Recall and F1 score.

**Accuracy** - Accuracy is the most intuitive performance measure and it is simply a ratio of correctly predicted observation to the total observations. One may think that, if we have high accuracy then our model is best. Yes, **_accuracy is a great measure but only when we have symmetric datasets where values of false positive and false negatives are almost same._** Therefore, we have to look at other parameters to evaluate the performance of our model. 

> Accuracy = TP+TN/TP+FP+FN+TN

**Precision** - Precision is the ratio of correctly predicted positive observations to the total predicted positive observations. _**The question that this metric answer is of all persons that labeled as healthy, how many actually healthy?**_ High precision relates to the low false positive rate. 

> Precision = TP/TP+FP


**F1 score** - F1 Score is the weighted average of Precision and Recall. Therefore, this score takes both false positives and false negatives into account. Intuitively it is not as easy to understand as accuracy, but F1 is usually more useful than accuracy, especially if we have an uneven class distribution. Accuracy works best if false positives and false negatives have similar cost. If the cost of false positives and false negatives are very different, it’s better to look at both Precision and Recall. 

> F1 Score = 2*(Recall * Precision) / (Recall + Precision)

**Recall (Sensitivity)** - Recall or Sensitivity is a measure of the proportion of actual positive cases that got predicted as positive (or true positive). Sensitivity is also termed as Recall. This implies that there will be another proportion of actual positive cases, which would get predicted incorrectly as negative (and, thus, could also be termed as the false negative). This can also be represented in the form of a false negative rate. The sum of sensitivity and false negative rate would be 1. Let's try and understand this with the model used for predicting whether a person is suffering from the disease. Sensitivity is a measure of the proportion of people suffering from the disease who got predicted correctly as the ones suffering from the disease. In other words, _**the person who is unhealthy actually got predicted as unhealthy is sensitivity**_.

> Recall = TP/TP+FN

**Specificity** - Specificity is defined as the proportion of actual negatives, which got predicted as the negative (or true negative). This implies that there will be another proportion of actual negative, which got predicted as positive and could be termed as false positives. This proportion could also be called a false positive rate. The sum of specificity and false positive rate would always be 1. Let's try and understand this with the model used for predicting whether a person is suffering from the disease. Specificity is a measure of the proportion of people not suffering from the disease who got predicted correctly as the ones who are not suffering from the disease. In other words, _**the person who is healthy actually got predicted as healthy is specificity**_.

> Specifity = TN/(TN+FP) 

_**While Sensitivity measure is used to determine the proportion of actual positive cases, which got predicted correctly, Specificity measure is used to determine the proportion of actual negative cases, which got predicted correctly.**_

**_Sensitivity and Specificity measures are used to plot the ROC curve. And, Area under ROC curve (AUC) is used to determine the model performance._**

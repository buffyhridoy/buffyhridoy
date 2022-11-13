---
layout: post
title: Cohen's Kappa Score
description: >
  This blog is a brief description about Cohen's Kappa Score.

canonical_url: http://Cohen's Kappa Score
hide_image: false
accent_color: '#4fb1ba'
accent_image:
  background: 'linear-gradient(to bottom,#193747 0%,#233e4c 30%,#3c929e 50%,#d5d5d4 70%,#cdccc8 100%)'
  overlay:    true
---

![capppa](https://user-images.githubusercontent.com/37147511/201522787-9fcc6a70-5068-4184-b016-bb8650536717.jpg)

<p><div style="text-align: justify">The Kappa Coefficient, commonly referred to as `Cohen's Kappa Score`, is a statistic used to assess the effectiveness of machine learning classification models. Its formula, which is based on the conventional 2x2 confusion matrix, is used to assess binary classifiers in statistics and machine learning. Jacob Cohen, a statistician who developed `Cohen's Kappa Coefficient` in 1960, is the name of the measure.</div></p>

<p><div style="text-align: justify">The dependability between and within raters for category categories is demonstrated by `Cohen's Kappa score (K)`. Because it takes into account the possibility that the agreement occurred by coincidence, most people believe it is a more accurate approach to quantify agreement than a simple % agreement. Although it may be modified for circumstances with more than two raters, it is often utilized in settings with two raters. In binary classification models, one of the raters takes on the role of the classification model, while the other rater assumes the role of a real-world observer who is aware of the true classifications for each record or dataset. `Cohen's Kappa` takes into account how frequently the raters concur (real positives and negatives) and disagree (false positives and false negatives). After accounting for chance, Cohen and Kappa may determine total agreement and agreement.
`Cohen's Kappa score` can be defined as the metric used to measure the performance of machine learning classification models based on assessing the perfect agreement and agreement by chance between the two raters (a real-world observer and the classification model).</div></p>

<p><div style="text-align: justify">The `Cohen-Kappa score` can be used to measure the degree to which two or more raters can diagnose, evaluate, and rate behavior. A credible and dependable indicator of inter-rater agreement is `Cohen's Kappa`. Both raw data and the values of the confusion matrix may be used to compute `Cohen's Kappa`. Each row in the data represents a single observation, and each column in the data reflects a rater's categorization of that observation, when `Cohen's Kappa` is calculated using raw data. A confusion matrix that displays the proportions of true positives, false positives, true negatives, and false negatives in each class may also be used to calculate `Cohen's Kappa`.</div></p>

<p><div style="text-align: justify">We'll look at how the Kappa score is determined by the confusion matrix. Contrary to its initial intent as a measure of consistency across data annotations, `Cohen's Kappa` is frequently utilized as a quality metric for data annotations in binary classification tasks. Kappa and popular classification measures like sensitivity and specificity do not have a linear analytical connection. Based just on the Kappa score, categorization performance is difficult to comment. Additionally, the performance of the multi-class classification model may be evaluated using a `Cohen-Kappa score`.
</div></p>


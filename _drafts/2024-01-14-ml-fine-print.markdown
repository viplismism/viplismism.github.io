---
layout: post
title: ML Fine Print
description: Take a look at the fonts in this post that can make your text editor or terminal emulator look little bit nicer
summary: A good choice of font for your coding can make a huge difference and improve your productivity.
tags: [Machine Learning,  Deep Learning]
---

*ML Fine Print is my essential resource for recalling and understanding crucial details in the ML software cycle which often overlooked in the heat of real-world challenges. The documentation is designed for easy reference without a specific reading order.*


### Overfitting and Regularization
- Overfitting occurs when our model becomes too complex, fitting too closely to the noise in the training data. While it might perform well during training, it often falters during testing or real-world applications. This happens when our model gets too attuned to the specific variations in the training data, as it starts to memorize the data instead of capturing the relevant patterns in it, resulting in poor generalization to new, unseen data. A model with low bias and high variance is considered overfit and doesn't perform well with new data.

- Regularization steps comes in as a solution, introducing a penalty for excessive model complexity to curb overfitting. Techniques like Early Stopping, L2, and L1 regularization help strike a balance between fitting the training data and keeping the model reasonably simple.

- Early Stopping involves halting training when a threshold of convergence is reached or when the validation loss starts to rise, preventing overfitting by avoiding excessive training. On the other hand, L2 regularization ensures that weights of various features don't overlap excessively, as one feature weights shouldn't overperform over the others promoting a balance. While the L1 regularization focuses on sparsity, driving some weights to absolute zero, doing so, it emphasize on keeping the informative features only.

- Choosing the right lambda value in regularization is crucial. Too high, and your model may underfit; too low, and it may overfit. It's a delicate tradeoff between simplicity and training data fit. The ideal lambda value is data-dependent, requiring hyperparameter tuning for optimal results. Despite potential increases in training loss, regularization often improves real-time predictions during inference, emphasizing the importance of overall model performance. Overall, regularization acts as a "keep things reasonable" rule for models, preventing extremes and ensuring their ability to make accurate predictions for new, unseen data.


### Assumptions for Data Sampling

In data sampling, we often make certain assumptions to ensure the validity and reliability of our results. These assumptions include:

1. Independent and Identically Distributed (i.i.d.): Examples are drawn independently and identically from the same distribution. This means that the probability of selecting any particular example is the same for all examples, and the selection of one example does not influence the selection of any other example.

2. Stationarity: The distribution of the data does not change over time or across different parts of the data set. This means that the probability of observing a particular value or outcome is the same regardless of when or where in the data set it is observed

3. Same Distribution: All examples are drawn from the same distribution. This means that the underlying process that generates the data is the same for all examples.

### Violations in Practice:

- Non-Independence: In some cases, the i.i.d. assumption may be violated due to dependencies between examples. For example, in a model that chooses ads to display, the choice of ads may be influenced by the user's previous ad history, creating a temporal relationship between observations.

- Non-Stationarity: The stationarity assumption may be violated if the underlying distribution of the data changes over time or across different parts of the data set. For example, in a data set of retail sales information, user purchases may change seasonally, violating stationarity.

- Different Distributions: The same distribution assumption may be violated if examples are drawn from different distributions. For example, in a data set of customer reviews, the distribution of reviews may differ between different products or services.

### Test and Training Set

- In machine learning, splitting the data into test and training sets is a crucial step. Randomization of the data before splitting is essential to ensure that the model does not train on a biased subset of the data. For instance, we would not want our Climate Predictor model to train solely on summer data and then be used for inference on test data consisting exclusively of winter data.

- When dealing with a large dataset containing billions of sample points, a small percentage (5-10%) of the data can be sufficient for testing the model during inference. However, if the dataset is relatively small, alternative methods like cross-validation may be necessary for better results.

- It is important to note that the test set should never be exposed to the model during training time. Repeated evaluation on the same test set can lead to implicit overfitting, which reduces the model's ability to generalize to new, unseen data.

- Furthermore, it is essential to shuffle the training dataset before creating a validation split. This is because sometimes, no matter how the training set and validation set are split, the loss curves may differ significantly. This issue is often caused by dissimilarities in the data between the training set and the validation set. Most libraries, including Pandas, split the dataset sequentially, which can lead to problems if the dataset points are arranged in a specific order. Therefore, it is recommended to randomize or shuffle the dataset points before splitting, ensuring that both the training and validation sets have an equivalent distribution of dataset points.

### Validation Set

- To avoid overfitting to a specific test set, it is advisable to use a validation set for tuning the model's hyperparameters. This allows for more objective evaluation of the model's performance and reduces the risk of implicit overfitting.

- Using the same data set for testing and validation can limit the potential for tuning the hyperparameters or improving the model. It is beneficial to continuously acquire more data to refresh the test and validation sets, ensuring a more comprehensive evaluation of the model's performance.

- It is important to note that the internal model parameters are adjusted during the training process, while the hyperparameters are tuned based on the results obtained from the validation and test sets.

### Convergence

- Convergence is the state of model training when the loss function exhibits minimal change, indicating that further training will not significantly improve the model's performance

- It is worth noting that the loss function may remain constant for a number of iterations before it begins to decrease. This can lead to a false sense of convergence, and it is important to monitor the loss function over a longer period to ensure true convergence.



### Feature Engineering

Binning Trick is generally used when we have the continuous numerical data and we want to convert them into the discrete bins or intervals. This technique is highly useful when we are working with the algorithms that works really well with the categorical data or when we wanted to reduced the impact of the outliers.

Let's say you are working on the housing price prediction problem and encoded the **street_name** as the numerical number starting from 0 to n, doing so will create the bias as our model would assume you have ordered the streets based on their average house prices, or for that matter, many houses are located at the corner of two streets, and there's no way to encode that information in the **street_name** value if it contains a single index. To solve this problem, we can create a binary feature vector where each column represents a street name, with 1 indicating the presence of the house on that street. As if the house is present on two streets, the model will use the weights for both of the streets as the feature vector for that house would have 1 for those two streets. 

Techniques like One-Hot encoding and the Label encoding helps to create the meaningful representations for the data that we can't use directly to feed to our model in it's raw form. 

If you have a dataset with 1,000,000 different street names for the _street_name_ feature, creating a binary vector with true or false for each name is inefficient in terms of storage and computation. Instead, a common approach is to use a sparse representation, storing only nonzero values.  That is if we have 35 _street_name_ and our house belongs to the _street_ 24 then instead of storing the 35 different bits as the indicators we could store the 24 

![sparse representation](https://github.com/viplismism/viplismism.github.io/blob/main/assets/images/sparse_representation.png?raw=true)

Sometimes, taking the logarithm of a distribution address long tail issues, especially in data analysis and statistics. The long tail typically refers to a distribution where a few values occur frequently (the "head" of the distribution), while many other values occur infrequently (the "tail" of the distribution). By taking the logarithm, we can compress the range of values, which can be particularly helpful when dealing with data that has a wide spread. This transformation is especially useful when the data has a positive skewness, meaning that the tail is on the right side of the distribution.

Clipping the outliers beyond a certain set threshold is also a part of the feature engineering, necessarily for a given use case if the threshold is set to 4.0 then Clipping the feature value at 4.0 doesn't mean that we ignore all values greater than 4.0. Rather, it means that all values that were greater than 4.0 now become 4.0. 

Dealing with the missing values, duplicates, bad labels and bad feature values is important to make sure we have a robust set of data pipeline which we can use for our models. 

**Note:** 
- We shouldn't pass a sparse representation as described about as a direct feature input to a model. Instead, we should convert the sparse representation into a one-hot representation before training on it. 

- There is a minor difference between Sparse Vector and Sparse Representation. First one actually means the vector consisting mostly zeroes and the second one actually means the _dense representation_ of a Sparse Vector.

- If you have continuous values like latitudes and longitudes for a housing price prediction problem, it is advised to use the bins instead of floating points numbers for the model predictions, as using them in the floating point values provides almost no predictive powers, instead we can create the bins with the specified boundaries as neighborhoods at latitude 35.4 and 35.8 are in the same bucket, but neighborhoods in latitude 35.4 and 36.2 are in different buckets. This way the model will learn a separate weight for each bucket. For example, the model will learn one weight for all the neighborhoods in the "35" bin, a different weight for neighborhoods in the "36" bin, and so on. 

### Scaling

Scaling involves transforming feature values from their original range (e.g., 10000 to 50000) to a more standard range, such as 0 to 1 or -1 to +1. This process is particularly useful when dealing with multiple features in a dataset, as it provides several benefits:

- Improved convergence of gradient descent: Scaling the features can accelerate the convergence of gradient descent algorithms, as it ensures that all features are on the same scale and contribute equally to the optimization process.

- Prevention of floating-point precision issues: During training, if the weights for a feature column exceed the floating-point precision limit, they may be set to NaN (Not a Number). This can cause a chain reaction, resulting in all the numbers in the model becoming NaN. Scaling the features helps to prevent this issue by keeping the weights within a manageable range.

- Reduction of bias: Without feature scaling, a model may be biased towards features with a wider range of values. This is because the model will give more weight to these features during the training process. Scaling the features ensures that all features are treated equally, regardless of their range.

- It is possible to apply different scaling thresholds to distinct features. For instance, one feature could be scaled between -2 to +2, while another might range from -4 to +4. However, using an excessively large scale, such as -10,000 to +10,000, could lead to suboptimal results.

### Z-Score Normalization
- Z-score normalization, also known as standardization or z-score scaling, is a statistical method used to make datasets comparable by transforming them into a standard normal distribution. This technique is commonly applied in statistics and machine learning for analyzing data points on a standardized scale.

- To calculate the z-score for a data point (x) in a distribution, we use the formula: z = (x - μ) / σ, where μ is the mean of the dataset, and σ is the standard deviation. The z-score ensures that different features share a common scale, making comparison and analysis more straightforward. This proves especially useful when dealing with variables of diverse units or scales, preventing any single variable from dominating the analysis due to its magnitude.

- After z-score normalization, each feature column in the transformed dataset has a mean of 0 and a standard deviation of 1. This normalization results in a standard normal distribution, making it easier to interpret and work with for statistical analyses.

  ***Note**: The fit and transform steps for z-score normalization should be applied initially to the complete dataset. Since z-score normalization involves both fitting and transforming methods, it's essential to use the same normalizer when transforming the test data.*

### Feature Crosses

A **feature cross** serves as a synthetic feature, injecting nonlinearity into the feature space by multiplying two or more input features. In simpler terms, it enables a linear model to grasp and interpret non-linear relationships.

Consider having two features, x1 and x2, insufficient for our linear model to comprehend non-linearity. Enter the feature cross, x3, defined as the product of x1 and x2:

```text
x3 = x1*x2
```

This newly introduced feature cross seamlessly integrates into the linear formula:

```text
Y = b + w1*x1 + w2*x2 + w3*x3
```

Even though w3 encodes nonlinear information, the linear model adapts naturally during training, requiring no special adjustments.

Feature crosses prove highly effective in enhancing and scale linear models when dealing with massive datasets which often have non linearity. Integrating feature crosses into our modeling practices significantly improves architecture and results. They play a crucial role in capturing complex relationships between features that might be not related when considered in isolation. By enabling the model to learn both individual and combined effects of features, feature crosses prove essential in capturing the difficulties of the of real-world data.

For a matter of fact, sometimes in real-world applications, feature crosses are used not just for continuous features, but also for one-hot feature vectors. One-hot feature vector crosses act as logical conjunctions, revealing unknown patterns and interactions in the data which might not be known previously. 

For an instance, think of online shopping, now there might be some one-hot encoding features related to user behavior, shopping type, and product type, now feature crosses allows the model to create features like "Product Segment A AND Shopping Type Mobile." This logical conjunction helps in capturing difficult patterns, I mean it may give some insights regarding when and how Product Segment A is being choosed when the person is shopping for a Mobile set, potentially improving predictive performance overall. Sometimes when our linear model is not performing upto mark, creating the features crosses might be a good choice.

### Classification Metrics

Accuracy is a commonly used metric to evaluate the performance of a classification model. It is defined as the ratio of correctly predicted instances to the total number of instances. While accuracy is a straightforward and intuitive metric, it can be misleading in certain situations. In cases when we have in-balanced dataset, a classifier can achieve a higher accuracy by just simply predicting the majority class. However, this doesn't mean the model is performing well. It might be failing to correctly identify instances of the minority class, which could be the more critical class in certain applications.

Let's consider an example of a medical diagnostic test for a rare disease. Suppose we have a dataset with 1000 instances, and only 10 of them belong to the positive class (people with the disease), while the remaining 990 instances belong to the negative class (people without the disease).

Now, let's imagine we have a simple classifier that predicts everyone as negative. Here's how the confusion matrix would look like:

![accuracy](https://github.com/viplismism/viplismism.github.io/blob/main/assets/images/tptn.png?raw=true)

Now, let's calculate the accuracy:

*Accuracy = Number of Correct Predictions​ / Total number of instances => 990 / 1000  =>  99 %*

The accuracy seems very high (99%), but the model is not performing well in identifying individuals with the disease. It predicts every dataset point as negative, so it misses the cases of the positive class. In a medical context, failing to identify individuals with the disease (false negatives) can have serious consequences.

In this example, accuracy is misleading because the dataset is imbalanced, and the classifier achieves high accuracy by simply predicting the majority class. In such situations, accuracy alone doesn't provide a complete picture of the model's performance, and it's important to consider other metrics like sensitivity (recall) or the F1 score, which take into account the ability of the model to correctly identify positive instances.

2. Precision and Recall are generally used to get the better understanding of the classification process we did. Precision means out of all the cases where your model marked something as positive, how many of them were actually true. So let's say if marking an email correctly as spam if considered as Positive thing, then out of all the cases where your model marked some emails as Positive, how many of those emails were actually spam. On the other hand, Recall means out of all the positive cases from your dataset, what percentage did our model marked correctly as positive. So if you have 100 emails and 40 of them are spammed and your model predicted 30 of the 40 spammed emails as Spammed then the recall would be 30/40.   

3. Remember that there is always some kind of trade-off between the Precision and Recall. As let's say our threshold is 0.4 initially, that means all the cases where our classification model gives us the probability equal to or greater than 0.4, it's considered to be Positive, so if we are more concerned for higher precision, or we want our model to be very sure that if it marks something as positive , it would be positive actually then we can increase the threshold to maybe a higher number of 0.6, so now only those instances when the probability from the classification model is higher than or equal to 0.6 will be considered positive which in turn will decrease the false positive and increase the Precision overall

4. But if we increase the threshold to 0.6 then it might happen that the instances when the probability for some dataset points are below the 0.6, let's say 0.5 but our model will still marked them negative as it's not above or equal to the threshold that we decided already, so now the model misses that, and it will only predict something as positive if it's beyond that threshold and in that process it might miss the actual ones. So for those cases Recall will decrease 

5. Tuning a threshold for logistic regression is different from tuning hyperparameters such as learning rate. Part of choosing a threshold is assessing how much you'll suffer for making a mistake. For example, mistakenly labeling a non-spam message as spam is very bad. However, mistakenly labeling a spam message as non-spam is unpleasant, but hardly the end of your job.

6. In general, a model that outperforms another model on both precision and recall is likely the better model. Obviously, we'll need to make sure that comparison is being done at a precision / recall point that is useful in practice for this to be meaningful. For example, suppose our spam detection model needs to have at least 90% precision to be useful and avoid unnecessary false alarms. In this case, comparing one model at {20% precision, 99% recall} to another at {15% precision, 98% recall} is not particularly instructive, as neither model meets the 90% precision requirement. But with that caveat in mind, this is a good way to think about comparing models when using precision and recall.

7. Precision and Recall solely depends on the threshold we choose, so it's important to come up with an adequate set of threshold values. 


### ROC and AUC

ROC (Receiver Operating Characteristic) and AUC (Area Under the Curve) are closely related concepts used to evaluate the performance of classification models, particularly binary classifiers. Here are the key differences between ROC and AUC:

![rocauccurve](https://github.com/viplismism/viplismism.github.io/blob/main/assets/images/ROC_Curve.png?raw=true)

1. **ROC (Receiver Operating Characteristic):**
   - **Definition:** The ROC curve is a graphical representation of the trade-off between the true positive rate (sensitivity) and the false positive rate (1 - specificity) at various thresholds.

   - **Components:** The ROC curve is typically a plot of true positive rate (y-axis) against the false positive rate (x-axis) for different threshold values.

   - **Interpretation:** A model with a better performance will have an ROC curve that is closer to the top-left corner of the plot, indicating higher true positive rates and lower false positive rates across different threshold values.

  
2. **AUC (Area Under the Curve):**
   - **Definition:** AUC is a scalar value that represents the area under the ROC curve. It provides a single numerical summary of the model's ability to distinguish between the two classes.

   - **Range:** AUC values range from 0 to 1, where a higher AUC indicates better discrimination performance. A model with an AUC of 0.5 is no better than random, while an AUC of 1 represents a perfect classifier.

   - **Interpretation:** The AUC is a useful metric for comparing and ranking different models. A higher AUC suggests a better overall ability of the model to discriminate between positive and negative instances across all possible threshold values. That being said, a model with higher AUC suggests that our model ranks a random positive example more highly than a random negative example.

   - AUC is desirable for the following two reasons , first it is **scale-invariant** as it measures how well predictions are ranked, rather than their absolute values. Secondly it is **classification-threshold-invariant**. as it measures the quality of the model's predictions irrespective of what classification threshold is chosen.

   However, both these reasons come with caveats, which may limit the usefulness of AUC in certain use cases:

- **Scale invariance is not always desirable.** For example, sometimes we really do need well calibrated probability outputs, and AUC won’t tell us about that.
    
- **Classification-threshold invariance is not always desirable.** In cases where there are wide disparities in the cost of false negatives vs. false positives, it may be critical to minimize one type of classification error. For example, when doing email spam detection, you likely want to prioritize minimizing false positives even if that results in a significant increase of false negatives. I mean if you are doing that job, You'd want to avoid marking important emails as spam, even if it means missing some spam emails. AUC isn't a useful metric for this type of optimization.

### Key Differences:
   - **Format:** ROC is a curve (plot), while AUC is a single scalar value.

   - **Graphical Representation:** ROC visually displays the trade-off between true positive rate and false positive rate, while AUC condenses this information into a single number.

   - **Interpretation:** ROC is useful for understanding the model's behavior across different thresholds, while AUC provides a summary measure of overall model performance.

- *Note : To understand what Threshold term means, refer to topic of Classification Metrics above this section.*
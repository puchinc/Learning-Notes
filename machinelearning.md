
# Loss
def Hinge(yHat, y): # penalizes predictions y < 1
    return np.max(0, 1 - yHat * y)


## Cross Entropy
https://www.zhihu.com/question/65288314

# PCA
X -> Y
cov matrix: X * X^T = S ∑ S ^ -1  
new cov matrix = Y * Y^T = ∑, all covariance are 0 

# Empirical Parameters
* May want to get everything into -1 to +1 range (approximately)
Want to avoid large ranges, small ranges or very different ranges from one another
Rule a thumb regarding acceptable ranges
-3 to +3 is generally fine - any bigger bad
-1/3 to +1/3 is ok - any smaller bad


# Error Metric

## Classification 
### Algorithm predicts some value for class, predicting a value for each example in the test set
* Considering this, classification can be
1. True positive (we guessed 1, it was 1)
2. False positive (we guessed 1, it was 0)
3. True negative (we guessed 0, it was 0)
4. False negative (we guessed 0, it was 1)

### Precision (TP / We Guessed True)
* How often does our algorithm cause a false alarm?
* The higher precision, the smaller false positive
* Of all patients we predicted have cancer, what fraction of them actually have cancer
    * = true positives / # predicted positive
    * = true positives / (true positive + false positive)
* High precision is good (i.e. closer to 1)
* You want a big number, because you want false positive to be as close to 0 as possible

### Recall (TP / Real True)
* How sensitive is our algorithm?
* The higher recall, the smaller false negative
* Of all patients in set that actually have cancer, what fraction did we correctly detect
    * = true positives / # actual positives
    * = true positive / (true positive + false negative)
* High recall is good (i.e. closer to 1)
* You want a big number, because you want false negative to be as close to 0 as possible

### F1Score (fscore)
* 2 * (PR/ [P + R])
* Fscore is like taking the average of precision and recall giving a higher weight to the lower value
Many formulas for computing comparable precision/accuracy values
If P = 0 or R = 0 the Fscore = 0
If P = 1 and R = 1 then Fscore = 1
The remaining values lie between 0 and 1 

## Linear Regression
## R square
* http://statisticsbyjim.com/regression/interpret-r-squared-regression/
* Usually, the larger the R2, the better the regression model fits your observations. 
* R^2 = variance explained by the model / total variance
* 0% represents a model that does not explain any of the variation in the response variable around its mean. The mean of the dependent variable predicts the dependent variable as well as the regression model.
* 100% represents a model that explains all of the variation in the response variable around its mean.


## Preprocessing
* randomize order


# Validation
* Given a training set instead split into three pieces
    1. Training set (60%) - m values
    2. Cross validation (CV) set (20%)mcv
    3. Test set (20%) mtest 
* Model Selection by Best E_val, select with Eval(Am(Dtrain)) while returning Am∗ (D)
* Estimate generalization error of model using the test set
* In machine learning as practiced today - many people will select the model using the test set and then check the model is OK for generalization using the test error (which we've said is bad because it gives a bias analysis)


## Debugging a Machine Learning Algorithm
* Plot h_train(x), h_cv(x) learning curve

### Fix high bias
* If an algorithm is already suffering from high bias, more data does not help
* Get additional features 
* Kernel feature transform
* Decrease regularization

### Fix high variance
* Get more data
* Try smaller features

### Unbalanced Data
* Over-sampling
* Down-sampling


## Boosting
* Ensemble weak classifiers 

### AdaBoost
* update data distribution
    - decrease good classified data probability
    - increase poor classified data probability

### Boostrap Aggregation (Bagging)
> Re-sample N examples from D uniformly with replacement
* bagging works reasonably well if base algorithm sensitive to data randomness


## Models
### Decision Tree
* implementation
* https://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/
     
### Random Forest
* Bagging(reduce variance) + random-subspace C&RT decision tree(large variance)


## Applications
### Recommender System

#### Content-Based
* offline computation, items features don't chnage that frequently
* less surprise
> assume we have features regarding the content which will help us identify things that make them appealing to a user
D = item data
v: item features
u: user preference vector
rate = u * v
Want to train u, we treat each rate as label, 
then training u with rates, and predict unrated by trained u * v

e.g. recommend articles by making document features as word tf-idf representation

#### User-Based
* usually online compute
* complicated
* can bring user surprises

#### Collaborative Filtering
> we don't have all features to do content-based recommendation
> https://zhuanlan.zhihu.com/p/25069367


### Anomaly Detection
> we have a dataset containg normal data
1. train the probability p(x), test p(x) < threshold == anomaly 
2. Multivariate Guassian p(x; u, ∑)
    * standard deviation = 1/m ∑ (x_i - u)^2 -> (x_i - u) ^ T * (x_i - u)




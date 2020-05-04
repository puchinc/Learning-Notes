

## Logistic Regression
> https://zhuanlan.zhihu.com/p/32681265
solve binary classification

# Loss
def Hinge(yHat, y): # penalizes predictions y < 1
    return np.max(0, 1 - yHat * y)


## Cross Entropy
> https://zhuanlan.zhihu.com/p/51431626


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


# Classification 
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

### ROC Curve (Receiver Operating Characteristic)
> ROC curve the left up, the better classification
> classifiers should be in left up area, (divided by (0,0) (1,1) diagnol line, random classifier) 

TPR = TP / (TP + FN)
FPR = FP / (TN + FP)

   TPR
    |
1.0 |(all samples are                    (all samples are
    | correctly classified)               classified as positive)
    |
    |
    |
    |
    |(all samples are                    (all samples are
    | classified as negative)             wrongly classified)
 0  ------------------------------------------------------------- FPR
     0                                                        1.0

* AUC  (Area Under roc Curve) 
    - usually between 0.5 - 1.0
    - the larger, the better 

> computation: https://www.zhihu.com/search?type=content&q=auc%20roc%20curve


### Calibration
* when to use: There’s no point in calibrating if the classifier is already good in this respect. First make a reliability diagram and if it looks like it could be improved, then calibrate. That is, if your metric justifies it.
* reliability diagram: x mean predicted value for each bin, y: fraction of true positive cases.
* algorithms: http://fastml.com/classifier-calibration-with-platts-scaling-and-isotonic-regression/
    - platts: logistic regression on the output of the SVM with respect to the true class labels

        from sklearn.linear_model import LogisticRegression as LR

        lr = LR()                                                       
        lr.fit( p_train.reshape( -1, 1 ), y_train )     # LR needs X to be 2-dimensional
        p_calibrated = lr.predict_proba( p_test.reshape( -1, 1 ))[:,1]

    - isotonic

        from sklearn.isotonic import IsotonicRegression as IR

        ir = IR( out_of_bounds = 'clip' )
        ir.fit( p_train, y_train )
        p_calibrated = ir.transform( p_test )   # or ir.fit( p_test ), that's the same thing

* motivation of calibration: https://www.quora.com/What-is-called-classifier-calibration-in-machine-learning
    - e.g. if your classifier say there're 90% possibility that these events should be blocked, yep it should be. So you collect all prediction with 0.9 and see their true label, ideally you might want to see 90% fraud cases within them.
* https://machinelearningmastery.com/calibrated-classification-model-in-scikit-learn/
* The distribution of the probabilities can be adjusted to better match the expected distribution observed in the data. This adjustment is referred to as calibration, as in the calibration of the model or the calibration of the distribution of class probabilities.
* Empirical results suggest that unlike logistic regression which directly predicts probabilities, SVMs, bagged decision trees, and random forests can benefit the more from calibrating predicted probabilities. They not natively predict probabilities, meaning the probabilities are often uncalibrated.

##### Logic
###### sufficient and necessary
* P -> Q, P is sufficient to Q 
P | ... | ... -> Q
e.g. live 

* ~Q -> ~P (thus, P -> Q), P is necessary to Q
P & ... & ... -> Q

* P <-> Q

(~P → ~Q) & (P → Q)
(Q → P) & (P → Q)
P ↔ Q

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


## Bias v.s. Variance
> Bagging是bootstrap+aggregation，对样本的不断重采样和聚合训练，不断减少了每个样本的影响，增加了总体对于各个样本的抗干扰能力，所以是降低variance> 。
> 而Boosting是串行的迭代算法，不断去改进拟合每一个样本，自然对样本的依赖性强，虽然可以很好的拟合所有训练数据，但是如果训练数据有问题，则模型会很差，bias也就很高。


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


### ResNet

### Inception


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

### Active Learning
https://blog.csdn.net/Houchaoqun_XMU/article/details/80146710
> Use complicated teacher output to train thinner student network
如下图所示为常见的主动学习流程图，属于一个完整的迭代过程，模型可以表示为 A = (C, L, S, Q, U)。其中C表示分类器（1个或者多个）、L表示带标注的样本集、S表示能够标注样本的专家、Q表示当前所使用的查询策略、U表示未标注的样本集。流程图可解释为如下步骤（以分类任务为例）：

（1）选取合适的分类器（网络模型）记为 current_model 、主动选择策略、数据划分为 train_sample（带标注的样本，用于训练模型）、validation_sample（带标注的样本，用于验证当前模型的性能）、active_sample（未标注的数据集，对应于ublabeled pool）；

（2）初始化：随机初始化或者通过迁移学习（source domain）初始化；如果有target domain的标注样本，就通过这些标注样本对模型进行训练；

（3）使用当前模型 current_model 对 active_sample 中的样本进行逐一预测（预测不需要标签），得到每个样本的预测结果。此时可以选择 Uncertainty Strategy 衡量样本的标注价值，预测结果越接近0.5的样本表示当前模型对于该样本具有较高的不确定性，即样本需要进行标注的价值越高。

（4）专家对选择的样本进行标注，并将标注后的样本放至train_sapmle目录下。

（5）使用当前所有标注样本 train_sample对当前模型current_model 进行fine-tuning，更新 current_model；

（6）使用 current_model 对validation_sample进行验证，如果当前模型的性能得到目标或者已不能再继续标注新的样本（没有专家或者没有钱），则结束迭代过程。否则，循环执行步骤（3）-（6）。

### Distillation
> https://medium.com/neuralmachine/knowledge-distillation-dc241d7c2322
> model compression
Ensemble 是一种很常用的提升机器学习模型性能的方法， 简单来说就是对同样的数据训练多个不同种的模型， 然后对它们的预测值取平均。但是Ensemble的方法代价太大。前人的工作有验证可以将ensemble模型中的knowledge压缩到一个单一简单模型中

* Once the cumbersome model has been trained, we can then use a different kind of training, which we call “distillation” to transfer the knowledge from the cumbersome model to a small model that is more suitable for deployment

* soft targets generated from cumbersome model provides higher entropy comparing with hard targets, making small model be trained on these smaller dataset + higher learning rate

* In the simplest form of distillation, knowledge is transferred to the distilled model by training it on a transfer set and using a soft target distribution for each case in the transfer set that is produced by using the cumbersome model with a high temperature in its softmax. The same high temperature is used when training the distilled model, but after it has been trained it uses a temperature of 1.

* Approach
    Caruana and his collaborators circumvent this problem by using the logits (the inputs to the final
    softmax) rather than the probabilities produced by the softmax as the targets for learning the small
    model and they minimize the squared difference between the logits produced by the cumbersome
    model and the logits produced by the small model. Our more general solution, called “distillation”,
    is to raise the temperature of the final softmax until the cumbersome model produces a suitably soft
    set of targets. We then use the same high temperature when training the small model to match these
    soft targets. We show later that matching the logits of the cumbersome model is actually a special
    case of distillation.

* intermediate temperatures work best which strongly suggests that ignoring the large negative logits can be helpful     

* Maybe private dataset can use this similar techniques, by choosing right bias:
    We then tried omitting all examples of the digit 3 from the transfer set. So from the perspective
    of the distilled model, 3 is a mythical digit that it has never seen. Despite this, the distilled model
    only makes 206 test errors of which 133 are on the 1010 threes in the test set. Most of the errors
    are caused by the fact that the learned bias for the 3 class is much too low. If this bias is increased
    by 3.5 (which optimizes overall performance on the test set), the distilled model makes 109 errors
    of which 14 are on 3s. So with the right bias, the distilled model gets 98.6% of the test 3s correct
    despite never having seen a 3 during training. If the transfer set contains only the 7s and 8s from the
    training set, the distilled model makes 47.3% test errors, but when the biases for 7 and 8 are reduced
    by 7.6 to optimize test performance, this falls to 13.2% test errors.

    - After training, we can correct for the biased training set by incrementing the logit of the dustbin class by the log of the proportion by which the
specialist class is oversampled

## Bayes Theorem
* 解決"逆概率"問題, 在有限的信息下，能够帮助我们预测出概率。 https://zhuanlan.zhihu.com/p/37768413
* 我们先根据以往的经验预估一个"先验概率"P(A)，然后加入新的信息（实验结果B），这样有了新的信息后，我们对事件A的预测就更加准确。
    - 后验概率（新信息出现后A发生的概率）　＝　先验概率（A发生的概率） ｘ 可能性函数（新信息带出现来的调整）

* 贝叶斯的底层思想就是：
    - 如果我能掌握一个事情的全部信息，我当然能计算出一个客观概率（古典概率、正向概率）。可是生活中绝大多数决策面临的信息都是不全的，我们手中只有有限的信息。既然无法得到全面的信息，我们就在信息有限的情况下，尽可能做出一个好的预测。也就是，在主观判断的基础上，可以先估计一个值（先验概率），然后根据观察的新信息不断修正(可能性函数)。
    - 給一個主觀empirical判斷(prior)，根據客觀事實(數據, liklihood)修正得到posterior

    P(Y|X) = P(Y) * P(X|Y) / P(X)

P(Y): prior, without knowing anything
P(X|Y)/P(X): likelihood, make prior approximates posterior. 似然，即 P(X|Y) ，是假设 Y 已知后我们观察到的数据应该是什么样子的
    * likelihood > 1: P(Y) is enhanced
    * likelihood = 1: X is useless for determining Y
    * likelihood < 1: X decreases P(Y)
P(Y|X): posterior 

P(X): often be solved by total probability
    - 全概率就是表示达到某个目的，有多种方式（或者造成某种结果，有多种原因），问达到目的的概率是多少（造成这种结果的概率是多少）
    P(S) =    P(SㄇA)    +      P(SㄇB)  +     P(SㄇC)   + ...
         = P(S|A) * P(A) + P(S|A) * P(A) + P(S|A) * P(A) + ...


* MLE v.s. MAP https://zhuanlan.zhihu.com/p/32480810
频率学派 - Frequentist - Maximum Likelihood Estimation (MLE，最大似然估计)
贝叶斯学派 - Bayesian - Maximum A Posteriori (MAP，最大后验估计)

    - 随着数据量的增加，参数分布会越来越向数据靠拢，先验的影响力会越来越小
    - 如果先验是uniform distribution，则贝叶斯方法等价于频率方法。因为直观上来讲，先验是uniform distribution本质上表示对事物没有任何预判
    - MAP=MLE + Regularization

#### Naive Bayes
> computation https://zhuanlan.zhihu.com/p/37575364

P(Y|X) = P(Y) * P(X|Y) / P(X)

arg max P(Y|X) = arg max P(Y) * P(X | Y) since P(X) isn't related to Y
               = arg max P(Y, X1, X2, ... Xn)
               = arg max P(Y) * P(X1|Y) * P(X2|Y) * ... * P(Xn|Y) by independent assumptoin

NaiveBayesClassifier(Y = Yi) = arg max P(Yi) * ㄇP(Xj|Yi)

### Baysian Network
TO READ:
> https://blog.csdn.net/zdy0_2004/article/details/41096141
> https://zhuanlan.zhihu.com/p/30139208

## Semi-supervised Learning

### Pre-training as regularization
> http://www.jmlr.org/papers/volume11/erhan10a/erhan10a.pdf
To simply state that unsupervised pretraining is a regularization strategy somewhat undermines the significance of its effectiveness. Not
all regularizers are created equal and, in comparison to standard regularization schemes such as
L1 and L2 parameter penalization, unsupervised pre-training is dramatically effective.

### Pseudo-Labeling 
> Entropy Minimization
1. Take the same model that you used with your training set and that gave you good results.
2. Use it now with your unlabeled test set to predict the outputs ( or pseudo-labels).  We don’t know if these predictions are correct, but we do now have quite accurate labels and that’s what we aim in this step.
3. Concatenate the training labels with the test set pseudo labels.
4. Concatenate the features of the training set with the features of the test set.
5. Finally, train the model in the same way you did before with the training set.


* Use denoising auto-encoder + dropout to boost performance

### Ladder Networks

### Co-Training
> multiple models to pseudo-label unlabeled data, augmenting dataset with pseudo-labels
Co-training有 m1 和 m2 两个模型，它们分别在不同的特征集上训练。每轮迭代中，如果两个模型里的一个，比如模型 m1 认为自己对样本 x 的分类是可信的，置信度高，分类概率大于阈值 r ，那  m1 会为它生成伪标签，然后把它放入 m2 的训练集。简而言之，一个模型会为另一个模型的输入提供标签。

#### Multi-View Learning
> 对特征进行拆分，使用相同的模型，来保证模型间的差异性。

#### Single-View
> 采用不同种类的模型，但是采用全部特征，也是可以的。基于后一种方法，好多开始做集成方法，采用boosting方式，加入更多分类器，当然也是可以同时做特征的采样。



## Deep Learning Training Details

### Back-propogation
* function derivatives

* gradient descent
> https://zhuanlan.zhihu.com/p/32230623


### Normalization
> https://zhuanlan.zhihu.com/p/33173246

* intention:
（1）去除特征之间的相关性 —> 独立；
（2）使得所有特征具有相同的均值和方差 —> 同分布。
 (3) solve gradient vanishing problems

    - however, if normalizing to N(0, 1), the input of activation function would be in the non-saturating (linear) region, lossing expression capability. Thus, BatchNorm standardization then re-shift.

* Batch Norm to reduce influence on weight init

    import tensorflow as tf
    # put this before nonlinear transformation
    layer = tf.contrib.layers.batch_norm(layer, center=True, scale=True,
                                     is_training=True)

    - back propogation時還能解耦合


### Weight Initialization 
https://zhuanlan.zhihu.com/p/25110150

### Activation Function

* saturating - A saturating activation function squeezes the input
> https://stats.stackexchange.com/questions/174295/what-does-the-term-saturating-nonlinearities-mean
    𝑓 is non-saturating iff (|lim𝑧→−∞𝑓(𝑧)|=+∞)∨|lim𝑧→+∞𝑓(𝑧)|=+∞)  


#### ReLU

* use variation of Xavier Initialization

    import numpy as np
    W = np.random.randn(node_in, node_out) / np.sqrt(node_in / 2)

### Regularization

* Dropout 
    - can be viewed as a way of training an exponentially large ensemble of models that share weights.


# Outline of this course 

Here is an outline and/or some high-level understandings of the concepts taught in this course, detailed notes remain in my iPad :)

## W1,2. Basics and Linear Regression

 - Linear hypothesis, function of X
 - **MSE** cost function is good for most regression problems, function of Theta
 - Optimisation: **Gradient Descent**, iteratively minimise cost function, simultaneously update all Thetas. **Learning Rate** is a parameter of GD.
 
 - Input **normalisation** helps converge faster, 1) zero-mean, 2) feature scaling. Denominator: range or standard deviation, basically the same.

## W3. Logistic Regression (Classification)

 - **Sigmoid** hypothesis is the probability of predicting y=1. Decision boundary: ThetaX = 0
 - **Cross-Entropy** cost function, common for classification problems.

 - **L2 Regularisation**, penalise outstanding theta values, generally making Theta small and preventing overfitting. Trade-off with cost term controlled by lambda

## W4, 5. Neural Networks Representation

The NN here is the shallow one with one hidden layer, basically cascaded linear regression. Forward pass mapped by Theta for the cost, backward pass mapped by Delta for gradient. The dimensions of all parameters see python ex3 and ex4.

## W6. Practical Advice

Ways to improve and guide model development
 - splitting dataset: train/test or train/validation/test
 - Bias vs. Variance in the context of: degree of polynomial(Logistic), regularisation lambda, number of training examples. 
 - More comprehensive Evaluation Metrics: Precision, Recall, F1 score, (mAP)

## W7. Support Vector Machine

a more robust classifier than logistic regression with "large margin". In implementation, it works well with the **kernel** technique which extends SVM to non-linear context.

SVM robustness: optimisation objective. 

 - In the cross entropy loss, the **log of sigmoid(thetaX)** is substituted by a **two-linear-and-a-hinge approximation**. This gives the "large margin": predict 1 if thetaX >= 1, predict 0 if thetaX <= -1.

 - The regularisation parameter lambda is replaced by C(1/lambda).

**Kernel (Gaussian Kernel) method** change the **thetaX** to **thetaF**, essentially using F to represent each training example, where F is the newly generated high-dimensional feature in input space. The model then becomes non-linear.

## W8. Unsupervised Learning

 - K-means: repeatedly **assign cluster**, minimising Euclidean distance loss by optimising c and **move centroid**, minimising loss by optimising u. When implementing, run multiple times with random initialisation. 

 - Dimensionality Reduction: PCA, implemented by SVD

## W9. Anomaly Detection and Recommender system

 - Anomaly: Learn Gaussian distribution, set probability threshold to filter out anomalies. When having large amount of data, still use Neural Nets
 - Recommender: Predicting Rating scores using user preferences (theta) and movie features (x). Get one, learn the other. Collaborative filtering allows spontaneous learning of theta and x.

## W10. Large-Scale ML

 - Stochastic GD vs. Batch GD vs. Mini-Batch GD, online learning, MapReduce.

## W11. Photo OCR

OnlineGaussianNaiveBayes
========================

An online Gaussian Naive Bayes classifier

This classifier is the merging of traditional Gaussian Naive Bayes and a numerically stable online algorithm for calculating variance. See the following links:

http://en.wikipedia.org/wiki/Naive_Bayes_classifier#Gaussian_naive_Bayes
http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Incremental_algorithm

Time & Space Performance
Only a single instance is stored in RAM at a time. Means and variances for each dimension are also stored. The runtime complexity of training is O(NxDxC) where N is the number of instances, D is the dimensionality, and C is the number of classes. 1,000,000 2-dimensional instances took 25 seconds to train on my 3 year-old Macbook Pro.

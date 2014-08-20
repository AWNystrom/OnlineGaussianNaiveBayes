OnlineGaussianNaiveBayes
========================

An online Gaussian Naive Bayes classifier

This classifier is the merging of traditional Gaussian Naive Bayes and a numerically stable online algorithm for calculating variance. See the following links:

http://en.wikipedia.org/wiki/Naive_Bayes_classifier#Gaussian_naive_Bayes
http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Incremental_algorithm

Time & Space Performance
Only a single instance is stored in RAM at a time. Means and variances for each dimension are also stored. The runtime complexity of training is O(N*D*C) where N is the number of instances, D is the dimensionality, and C is the number of classes. 1,000,000 2-dimensional instances took 25 seconds to train on my 3 year-old Macbook Pro.


Performance
On a 2-dimensional four-class problem (250 instances of each class) whose points are shown in the following plot, the classifier achieved an accuracy of about 89.6%. Note that these classifications were done online; the classifier was shown 4 instances before predictions began. After that, inference was performed on a new instance, then the classifier trained on the new point. The colors of the points indicate the classifier's guess. Note that colors are well clustered, which is due to the high accuracy.
![](https://raw.githubusercontent.com/AWNystrom/OnlineGaussianNaiveBayes/master/images/class_plots.png?token=1478067__eyJzY29wZSI6IlJhd0Jsb2I6QVdOeXN0cm9tL09ubGluZUdhdXNzaWFuTmFpdmVCYXllcy9tYXN0ZXIvaW1hZ2VzL2NsYXNzX3Bsb3RzLnBuZyIsImV4cGlyZXMiOjE0MDkxMTQ0NDJ9--b26f04bbff03693f3517b6b5f042b01030eeca2d)

The following plot shows how quickly the classifier learns.
![](https://raw.githubusercontent.com/AWNystrom/OnlineGaussianNaiveBayes/master/images/accuracy_over_time.png?token=1478067__eyJzY29wZSI6IlJhd0Jsb2I6QVdOeXN0cm9tL09ubGluZUdhdXNzaWFuTmFpdmVCYXllcy9tYXN0ZXIvaW1hZ2VzL2FjY3VyYWN5X292ZXJfdGltZS5wbmciLCJleHBpcmVzIjoxNDA5MTE1MDk1fQ%3D%3D--ed799664b94f11089119b06df039b9fd82bb7977)


The final performance test is 10-fold cross valication on 100,000 points (4 balanced classes). The performances across folds are:

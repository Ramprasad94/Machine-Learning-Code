# Machine-Learning-Code-repository
A repository containing source code for machine learning algorithms

1. Decision tree: Decision trees are one of the most popular learning algorithms around. For small to medium sized datasets with decent 
number of features, decision trees are a great way to fit the data and predict with decent accuracy. This implementation of decision tree
implements a multi-way split depending on the feature values. While this tree may overfit occasionally due to the multi-way split procedure, using bagging in combination is bound to give much better results.

This code takes in the train and test files and creates trees for different depths, which can be set during as many iterations as required. It finally gives out a graph of the Depth vs Accuracy curve for the given train and test set.

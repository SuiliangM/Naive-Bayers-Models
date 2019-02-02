# Naive-Bayers-Models
This project implements Multi-variate Bernoulli Naive Bayes model and multinomial Naive Bayes model.  

Multi - variate Bernoulli Naive Bayes model: builds a NB(Naive Bayes) model from the training data, classifies the training and test data, and calculates the accuracy. Specific requirements are included below:
(A) The learner should treat all features as binary; that is, the feature is considered present iff its value is nonzero.
(B) The format is: build NB1.sh training_data test_data class_prior_delta cond_prob_delta model_file sys_output > acc_file
(C) Training data and test data are the vector files in the text format. See train.vectors.txt and test.vectors.txt for an example training file and an example test file.
(D) Class_prior_delta is the δ used in add-δ smoothing when calculating the class prior P(c); cond_prob_delta is the δ used in add-δ smoothing when calculating the conditional probability P(f|c). More specifically, when calculating the class prior P(c), we have P(c) = (class_prior_delta + count(c)) / (class_prior_delta * |C| + sum(count(c), for all c)); whereas for the calculation of conditional probability P(f|c), we have P(f|c) = (cond_prob_delta + count(f, c)) / (2 * cond_prob_delta + count(c)).
(E) Model_file stores the values of P(c) and P(f|c). The line for P(c) has the format “classname P(c) logprob”, where logprob is 10-based log of P(c). The line for P(f|c) has the format “featname classname P(f|c) logprob”, where logprob is 10-based log of P(f|c).
(F) Sys_output is the classification result on the training and test data (see sys_1 for an example). Each line has the following format: instanceName true class label c1 p1 c2 p2 ..., where pi = P(ci|x), x representing the current training instance. The (ci ,pi) pairs should be sorted according to the value of pi in descending order.
(G) Acc_file shows the confusion matrix and the accuracy for the training and the test data (see acc1 for an example).

The requirements of multinomial Naive Bayes model are similar to those of multi - variate Bernoulli Naive Bayes, except that all the features are are real-valued.  

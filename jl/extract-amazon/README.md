# Product Reviews for Ordinal Quantification

This data set comprises a labeled training set, validation samples, and testing samples for ordinal quantification. The goal of quantification is not to predict the class label of each individual instance, but the distribution of labels in unlabeled sets of data.

The data is extracted from the McAuley data set of product reviews in Amazon, where the goal is to predict the 5-star rating of each textual review. We have sampled this data according to three protocols that are suited for quantification research.

The first protocol is the artificial prevalence protocol (APP), where all possible distributions of labels are drawn with an equal probability. The second protocol, APP-OQ(50%), is a variant thereof, where only the smoothest 50% of all APP samples are considered. This variant is targeted at ordinal quantification, where classes are ordered and a similarity of neighboring classes can be assumed. 5-star ratings of product reviews lie on an ordinal scale and, hence, pose such an ordinal quantification task. The third protocol considers "real" distributions of labels. These distributions stem from actual products in the original data set.

The data is represented by a RoBERTa embedding. In our experience, logistic regression classifiers work well with this representation.

You can extract our data sets yourself, for instance, if you require a raw textual representation. The original McAuley data set is public already and we provide all of our extraction scripts.

Extraction scripts and experiments: https://github.com/mirkobunse/regularized-oq

Original data by McAuley: https://jmcauley.ucsd.edu/data/amazon/

## File Outline

The top-level directories `app/`, `app-oq/`, and `real` contain the samples that are drawn according to the three different evaluation protocols. The second-level `dev_samples/` and `test_samples/` directories contain the samples which are used for validation and for testing. The `dev_prevalences.txt` and `test_prevalences.txt` files contain the corresponding class prevalences, i.e., the ground-truth of quantification tasks.

Note that the individual `dev_samples/<N>.txt` and `test_samples/<N>.txt` are also individually labeled. The label of each instance, although not relevant in quantification, is in the first column of each line.

```
.
|-- roberta/
|   |-- app/
|   |   |-- dev_prevalences.txt
|   |   |-- dev_samples/
|   |   |-- test_prevalences.txt
|   |   `-- test_samples/
|   |-- app-oq/
|   |   |-- dev_prevalences.txt
|   |   |-- dev_samples/
|   |   |-- test_prevalences.txt
|   |   `-- test_samples/
|   |-- real/
|   |   |-- dev_prevalences.txt
|   |   |-- dev_samples/
|   |   |-- test_prevalences.txt
|   |   `-- test_samples/
|   `-- training_data.txt
```

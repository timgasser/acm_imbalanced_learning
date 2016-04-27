# acm_imbalanced_learning

This repo contains slides and code for the ACM Imbalanced Learning talk on 27th April 2016 in Austin, TX. 

## File listing

The files in the repo are listed below, with an explanation of what they're used for.

* ```acm_imbalance_algorithms.ipynb``` - Jupyter notebook with scikit-learn classifiers training on the Kaggle dataset.
* ```acm_imbalance_datasets.{pdf, pptx}``` - Powerpoint presentation with explanation of the dataset processing and algorithms.
* ```acm_imbalance_sampling.ipynb``` - Jupyter notebook with a set of routines to pre-process imbalanced data.
* ```acm_imbalanced_dataset.R``` - R script to use the 'unbalanced' package to pre-process data to remove imbalance.
* ```datasets.zip``` - A zip file containing datasets for use in the talk. These are listed below
  * ```cs-training.csv``` - Training data from the Kaggle 'Can I get some credit' competition
  * ```cs-test.csv``` - Test data from the Kaggle competition.
  * ```sampleEntry.csv``` - Sample entry format for the Kagle competition.
  * ```cs-training-{CNN, OSS, smote, tomek}.csv``` - Processed training data (generated from ```acm_imbalanced_dataset.R```) using the algorithms in the filename.

## Feedback

Any comments, questions, or feedback please submit a pull request !

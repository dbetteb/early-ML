# Support Vector Machines tutorial

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dbetteb/early-ML/blob/master/04_SUPPORT_VECTOR_MACHINES/SVM.ipynb#scrollTo=7Fzc-S7FIfWk)

## Objectives

Get acquainted with the founding principles of SVM :
* for perfectly linearly separable data
* for non linearly separable data
* with kernel-based SVM

Ressources on setting hyperparameters for SVM [here](https://www.csie.ntu.edu.tw/~cjlin/libsvm)

## Linearly separable Data

In practice, you do not know which SVM to use beforehand. However, here are some guidelines to help you choose :
* linear SVM can handle quite large data sets (especially in text processing)
* kernel-based SVM become quite expensive to compute for large data sets. Essentially because their setting require to compute the kernel dot-like product for each observation with respect to all the others. This matrix is the same size as the number of observations.

In practice, it is often recommended to start with linear SVM with CV estimation of best regularization parameter $C$. In case, target accuracy is not reached switch to kernel SVM, start with RBF kernel settings of parameter \gamma.

## Kernel based SVM

It is usually recommended to start with default settings of kernels. Typically, `scikit-learn` starts with C=1 and an automatic rule to set \gamma parameter.

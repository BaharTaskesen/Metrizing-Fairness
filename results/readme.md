# results

This folder contains additional results. Currently, we provide the preliminary results for a comparison using the same models as suggested by reviewer `tqef`. We compare Logistic Regression and 2 layer Neural Networks with 20 hidden nodes and ReLU activation for hidden nodes.

## Pareto Front
We provide results for the Drugs and Communities and Crime datasets. The pareto frontier plots can be found in the following files:


| File                                        | Dataset               | Model               | Number of Replications |
|---------------------------------------------|-----------------------|---------------------|------------------------|
| CommunitiesCrimeClassification_logistic.pdf | Communities and Crime | Logistic Regression | 2                      |
| CommunitiesCrimeClassification_NN.pdf       | Communities and Crime | Neural Network      | 2                      |
| Drug_logistic.pdf                           | Drug                  | Logistic Regression | 3                      |
| Drug_NN.pdf                                 | Drug                  | Neural Network      | 3                      |


## AUC

### AUC results on Drug Dataset:

Logistic Regression Model:

| Method       | AUC          | Time      |
|--------------|--------------|-----------|
| Zafar et al. | 0.660(0.039) | 0.36sec |
| Ours         | 0.814(0.008) | 5.85sec |

Neural Network Model:

| Method       | AUC          | Time      |
|--------------|--------------|-----------|
| Cho et al.   | 0.712(0.017) | 10.70sec  |
| Oneto et al. | 0.824(0.010) | 479.11sec |
| Ours         | 0.819(0.010) | 6.76sec   |

### AUC results on Communities and Crime Dataset:

Logistic Regression Model:

| Method       | AUC          | Time      |
|--------------|--------------|-----------|
| Zafar et al. | 0.794(0.003) | 84.51sec |
| Ours         | 0.824(0.004) | 8.66sec  |

Neural Network Model:

| Method       | AUC          | Time      |
|--------------|--------------|-----------|
| Cho et al.   | 0.470(0.028) | 13.87sec  |
| Oneto et al. | 0.825(0.003) | 272.13sec |
| Ours         | 0.832(0.006) | 12.05sec  |
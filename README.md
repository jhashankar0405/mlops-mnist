# mlops-mnist

### Line chart with percentage of training on x-axis the  and macro f1 on y-axis
![alt text](https://github.com/jhashankar0405/mlops-mnist/blob/feature/assignment-11/images/pct_training_vs_macro_f1.png)

We are measuring macro F1 is the average of F1 score of all the classes and F1 score for each class is the harmonic mean of precision and recall. So this metric tries to optimize the precision and recall for all class separately. Here we see that Macro F1 is increasing with increase in training data that which indicates average improvement in f1 score across all classes that balances precision and recall.


### Assesment of incremental change in training to the OVR AUROC
![alt text](https://github.com/jhashankar0405/mlops-mnist/blob/feature/assignment-11/images/inc_pct_vs_change_roc.png)

Here we individually check the classification performance of all the class vis-a-vis the other classes. Here the rate of increase in OVR ROC in the begining but balances towards the later half but is positive. Initially adding the helps more than towards the end but it is helping as ROC is constantly positvie with few negative values in between.

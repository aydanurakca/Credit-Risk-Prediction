# Credit-Risk-Prediction
The dataset used for the project is the Credit Risk Prediction. In this dataset, the company wants to predict who might default on a consumer loan product. It has a total of thirteen features, one of which is the target feature. Of these, six of them are numeric, six of them are categorical and one of them is target feature binary.
The features are as follows;

* __Id__, which is a numerical value, is used to keep the order of the consumers. Each user has a unique id value.
* __Income__ represents the annual income of the consumers. It is a numerical value.
* __Age__ represents the ages of consumers. It is a numerical value.
* __Experience__, which is a numerical value, indicates the experiences of consumers in years.
* __Married__, a categorical value, is used to keep whether consumers are married or not. There are two categories: single or married. In this dataset, 90% of the consumers are categorized as single and 10% as married.
* __House Ownership__ keeps whether consumers own a home. It is a categorical value and there are three categories. These categories are rented, owned and no rent & no own. In this dataset, 92% of the consumers are categorized as rented, 5% as owned and 3% as no rent & no own.
* __Car Ownership__ keeps whether consumers own a car. It is a categorical value and there are two categories, yes and no. 30% of the consumers are categorized as yes and 70% as no.
* __Profession__ is the area of expertise of the consumers. It is a categorical value.
* __City__, which is a categorical value, is the city where consumers live.
* __State__, which is a categorical value, is the state where consumers live.
* __Current Job Years__ represents the years in which consumers are in their current job. It is a numerical value.
* __Current House Years__ represents the number of years consumers have been in their current residence. It is a numerical value.
* __Risk Flag__ is a target attribute. It consists of binary values.

## Data Preprocess and Algorithms
### Missing Values
While checking whether there is missing data in the rows of our data, it is seen that the data of all the attributes are complete.

### Distribution in Dataset
In order to apply models on this used dataset and make estimates, first of all, the data set should be processed. First of all, an uneven distribution was observed in the risk flag, which is the target feature in the data set. There are 221004 samples from “0” values and 30996 samples from “1” values belonging to the risk flag.

Considering the possibility that this will cause a problem in the future, it was decided to create two different data sets. In fact, the only difference between the two data sets created is the number of samples used. There is no change in the sample number of the first data set, but under-sampling was done in the second data set. In the under-sampling data set, firstly the number of the value of the target attribute "1" was determined, it was randomly taken from the value "0" as much as the detected number. Then, the two arrays were combined and turned into a dataframe. Samples were shuffled as a final process.

Before the model development process, some attributes that were expected not to affect the classification were dropped from the dataset. Since the Id attribute will be ineffective in classifying, it has been dropped. State and City attributes have been dropped from the dataset because they have too many unique values and have no effect on classification. When trying to convert these attributes to dummy attributes, it is thought that it will slow down the algorithms and have a negative effect on accuracy as they have too many unique values.

### Outlier Detection
A data point that varies greatly from other observations is referred to as an outlier. At the stage of checking whether there is an outlier value in a data set, there should not be any missing value first. Boxplots are preferred for outlier detection processes. Boxplots are often used to determine outliers on numerical values. Boxplots contain the quartiles and IQR (interquartile range) values of an attribute in the dataset. The points outside of whiskers in the boxplot represent outlier values. Boxplots were created with the attributes of Income, Age, Current House Years and Current Job Years in order to check whether there is an outlier value in this dataset. These attributes are chosen because they have numerical values. No outlier value was reached in the boxplots formed.

### Transformation of Features
Each value in the data set is of a different type, so each variable must be converted to a common type before applying the model.
Preprocess made in this direction is the transformation of categorical data. First of all, "married" and "car_ownership" features have been converted into binary. These two variables are not converted to dummy variables because otherwise, they fall into a dummy trap. The remaining "house_ownership" and "profession" are converted to dummy variables. Thus, it appears as new features in the dataset.

The last conversion was applied to numerical values. Numerical values are indifferent value ranges, so normalization was applied and each was drawn between 1-0.

### Classification Models
Preprocess operations have been performed for both data sets and the next step is the algorithm selection. The algorithm chosen first is the K-Nearest Neighbor (KNN). KNN, both the classification and regression algorithm, makes predictions based on the distance between the features of each sample. For the new sample to be estimated, the operation is performed with the distance calculation formula selected, which is Minkowski, Euclidean or Manhattan, with each sample in the existing dataset. Afterwards, the closest neighbors as many as the given number of "k" are found and the estimation process is performed.

The second algorithm chosen for classification is the Naive Bayes Classifier. The Naive Bayes Classifier is based on the Bayes theorem. It is a lazy algorithm that can run on unbalanced datasets. The way the algorithm works calculates the probability of each state for an element and classifies it according to the one with the highest probability value. With only a small amount of training data, this algorithm performs admirably.

Another algorithm chosen is the Decision Tree Classifier. This algorithm can also be used when both the features that define the data set and the target feature are categorical and continuous values. Decision Tree structure, which starts with the feature that best parses the dataset in the training phase, is put on the first node and continues to form by branching. Thus, the features selected for prediction form nodes. In the classical tree structure logic, the samples passed through the nodes are estimated by reaching the leaves.

## Models and Classification
### Train - Test Models and Results
All three models were tested on the two datasets we have. First of all, to create train and test datasets, the datasets were divided into %90 train data and %10 test data with the help of a method in the sklearn library called Train-Test Split. In order to execute the models with different features and to achieve the best results, we determined some features of the models and ran each model according to these features. Four different neighbors for the KNN model, two types and eight different tree depths were determined for the Decision Tree, and the models were run with these parameters. The highest accuracy from these trials was determined. The model features with the highest accuracy were recorded and the training times of each model were calculated. A classification report was created for each model, and as a result, the best models were found on each dataset. Since all three models were executed on two datasets, six different results were obtained. 

![image](https://user-images.githubusercontent.com/37415162/136179115-d1247c7e-cb78-43b9-9397-10e104ddad93.png)

The parameters that give the best accuracy value shown in the figure above are listed in the table below.

![image](https://user-images.githubusercontent.com/37415162/136179363-d0c22719-e596-4f06-b616-64bad2901e31.png)

The Decision tree algorithm gave the best results in both data sets, while the KNN gave results close to the Decision tree and the Naive Bayes algorithm could not obtain very high accuracy results in our data sets. Although the accuracy of all models in the undersampled dataset decreased compared to our dataset of 250000 data, it is seen that the values of metrics such as precision, recall and f1 score increased. Precision is the percentage of positive classes that are predicted correctly. Recall is the percentage of actual positives that our model predicted to be positive. Since these metrics are important, their increase has a positive effect on the models. We can conclude that the KNN model runs slower than other models in both datasets and Naive Bayes is the fastest running model.

### Cross-validation Models and Results
Similar to the previous stage, cross-validation was performed on two data sets in this stage. The purpose of cross-validation is to train the dataset in order to use the dataset more efficiently and to give better results. As in the train-test phase, the data set was divided into 90% and 10%. In the first stage, 10 fold cross-validation was applied on the set containing 90% of the data. For this, the GridSearchCV function was used on python. Thus, the train was performed in 10 different parts. The parameters required for each machine learning algorithm are also used during the training of each fold with the mentioned GridSearchCV function. For the K-nearest neighbor algorithm, 4 different neighbors and different algorithm types for the Decision Tree were tried as 8 different depth parameters.

After training both datasets with cross-validation, the predicted operation was performed on the test set with the most optimal result. The first value comparison is as seen in the figure below.

![image](https://user-images.githubusercontent.com/37415162/136179995-320eb2fb-7fcc-425b-96aa-024de4258ea9.png)

In the table shown below, the best parameters used for the estimation of each model are shown.

![image](https://user-images.githubusercontent.com/37415162/136180062-bf792048-a7a2-4874-a72c-622f6280a472.png)

KNN and Decision Tree applied on the data set with 250,000 data reached 89% similar accuracies, but also known that accuracy isn't the only thing to look for. Comparisons between the two with other metrics are required. Precision shows how many of the positive predictions are actually positive, recall shows how many of the positive values are predicted positive. It is observed that the metrics of KNN have better precision and recall values than Decision Tree. However, it has been observed that KNN completes its work in a much longer time when it is necessary to make comments in order to use the resources efficiently. Also, when the six of them are compared again within the metrics, it is observed that the results of the Decision Tree applied on the undersampled dataset are more optimal with all 86%. DecisionTree_undersampled also completed its execution in an optimal time as 1.3 minutes.

## Conclusion
KNN, Naive Bayes and Decision Tree were run on the dataset consisting of 250,000 data and an undersampled dataset consisting of 61,992 data used in this project, using both cross-validation and train-test split method. Looking at the results obtained from these, it can be observed that the precision, recall and f1 score values of the models executed on the undersampled dataset are nearly the same. The reason for this is the distribution in the normal dataset has been changed and the newly created undersampled dataset contains equal numbers of 0 and 1 values, which means 50% distribution. In the normal data set, the difference between precision and recall values is higher. In this dataset, the incorrect estimation rate is higher than the models in the undersampled dataset due to the large difference between the numbers of data.

When the Naive Bayes model was executed on datasets, it was the model that gave the worst metric results in all four cases. These results show that Naive Bayes does not fit the Credit Risk dataset. The KNN model provided good metric results, but even though it provided results very close to the Decision Tree when the training times are taken into account it cannot be considered as the best model in these datasets, since the training periods are too long. Decision Tree has given in both two models the best metric results and can be shown as the most optimal model working in these datasets because it executes in a short period.

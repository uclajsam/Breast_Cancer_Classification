# Breast Cancer Classification Project

## Introduction
Cancer is a disease that affects millions of people around the world.  Machine learning has been used as a technique to help classify cancerous tumors as 'benign' or 'malignant' based on certain attributes.  For this project, we will analyze the Wisconsin Breast Cancer dataset.  The set has 569 entries with 32 variables.  The data can be obtained from the UCI Repository found [here](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)) or imported through the pre-programmed datasets in Python.  The goal of this project is to create the best model in order to classify a randomized tumor as benign (no cancer) or malignant (cancer present).  The following Python libraries will be used for this analysis:

- Numpy
- Pandas
- Seaborn
- Matplotlib
- Sklearn

## Data Dictionary
The dataset has 32 variables which are defined below:
| Variable | Description |
| -------- | ----------- |
| id | Unique identifier for each entry |
| radius* | Distance from the center point to the perimeter |
| texture* | Standard deviation of gray-scaled values |
| perimeter* | Closed boundary distance of the cell |
| area* | Area of the cell |
| smoothness* | Local variation in radius lengths | 
| compactness* | Defined as the square perimeter per unit area |
| concavity* | Severity of concave portions of the countour |
| concave points* | Number of concave points of the contour |
| symmetry* | Not defined |
| fractal dimension* | Coastline approximation |
| target | Cell classification: 0 - Malignant, 1 - Benign |

* = Variables have 'mean', 'error', and 'worst' classifications (i.e. `mean_radius`, `radius_error`, `worst_radius` for the radius variable)

## Exploratory Data Analysis (EDA)
Our initial analysis shows that there are no missing values for any of the variables.  We also drop the `id` column because this is a unique identifier for each cell and will not have any bearing with modeling or analysis.  The distribution of the target variable (cell classification) can be seen below:

<p align = "center">
<img src = "https://user-images.githubusercontent.com/60159655/88986581-3c766a00-d288-11ea-9add-014eec27dfdc.png" />
</p>

From our data, approximately 63% of the cells are benign and 37% are malignant.

### Visualizing Feature Variables
We utilized swarmplots to visualize each feature variable with respect to cancer type to see if we can clearly separate the Maligantn and Benign groups.  To deal with potential outliers, we used the StandardScaler method to define a smaller range of potential values for each variable.  

#### "Mean" Features Swarmplot
![Mean Swarmplot](https://user-images.githubusercontent.com/60159655/89219716-2d900000-d585-11ea-80d0-a17688561ca3.png)

#### "Error" Features Swarmplot
![Error Swarmplot](https://user-images.githubusercontent.com/60159655/89220407-5cf33c80-d586-11ea-95a0-653ab2bf5364.png)

#### "Worst" Features Swarmplot
![Worst Swarmplot](https://user-images.githubusercontent.com/60159655/89220468-78f6de00-d586-11ea-9c32-af9228d9966e.png)

**Observations:** From these three plots, there are some variables that can be separated in terms of cancer type.  For example, the following variables show swarmplots where the blue and orange dots are clearly separated:

`mean radius`
`mean area` 
`mean perimter`
`mean concavity`
`radius error`
`perimeter error`
`area error`
`worst radius`
`worst perimter`
`worst area`
`worst concave points`

We can assume that these variables are clearl important in defining whether or not a random cell is classified as malignant or benign.  We will take these values into consideration when creating our classification model below. 

### Correlation Matrix
![Breast Cancer Correlation Matrix](https://user-images.githubusercontent.com/60159655/89221759-b0ff2080-d588-11ea-9a37-f55ecc2348dd.png)

From the correlation matrix, we see that there are feature groups that are highly correlated with one another (i.e. `mean_radius`, `mean_perimeter`, `mean_area`).  We need to these relationships into account when creating our model because overfitting could possibly occur.  From the matrix, only 6 out of 30 features show no correlation to the target variable (-0.10 < r < 0.10).

## Modeling
For our baseline model, we will train with all of the features and do a 60/40 train-test split on the data.  We allocated a greater percentage for the "test" dataset because we have a small sample size (569 entries).  In addition, we will utilize the following common classification machine learning models with default parameters to achieve a baseline:

1) Logistic Regression
2) Naive Bayes Classifier
3) Stochiastic Gradient Descent
4) K-Neighbors Classifier
5) Decision Tree Classifier
6) Random Forest Classifier
7) Support Vector Classifier

Below is a table of our baseline results:

<p align = "center">
<img src = "https://user-images.githubusercontent.com/60159655/89239959-98eec780-d5af-11ea-9b5c-21a6f476dd40.png" />
</p>

**Observation:** From the table, it seems like the data fits well for several classification models.  Out of models tested, the Random Forest Classifier has the highest metrics across the board.  We will choose this model for furuther analysis, although other models works just as well for this dataset.

### Baseline Random Forest Confusion Matrix (Test Accuracy = 95.2%)
From our test data, our Random Forest Classifier model correctly predicted 217 out of 228 values.  
<p align = "center">
<img src = "https://user-images.githubusercontent.com/60159655/89253558-d2382f00-d5d1-11ea-9246-90d32db93f3b.png" />
</p>

### Random Forest Classifier Feature Importances
When we plot the "feature importances" for this model we determine the top 5 most important features are:

1) worst perimeter
2) worst concave points
3) worst area
4) mean concave points
5) worst radius

<p align = "center">
<img src = "https://user-images.githubusercontent.com/60159655/89254437-f72da180-d5d3-11ea-8edd-14ecb29d6ca7.png" />
</p>

The importance of these features is not a surprise since these features are highly correlated with the target variable.  Since our data fits well with our baseline model, we will try to improve our model by selecting the most relevant features while trying to improve accuracy.

## Simplifying Model Using RFE
RFE (Recursive Feature Elimination) is an algorithm used to select the most relevant features in a machine learning model.  We will use RFE to determine if we can simplify our original model with 30 features and see if we can get an even better accuracy with less features.  We start by using RFE to determine the best model with one feature, train a Random Forest Classifier model, and predict evaluation metrics on the test data.  We increase the number of features by one and repeat the same process until we have created a model using all of the features.  A graph of metric values is shown below which illustrates that the best accuracy and F1 score is achieved with **_14 features_**.

<p align = "center">
<img src = "https://user-images.githubusercontent.com/60159655/89316537-1870ab00-d631-11ea-9e36-7fdad19e8dac.png" />
</p>

RFE also allows us to extract which features were used for each iteration of the algorithm.  For 14 features, RFE chose the following features for the optimal model:

`mean_radius`, `mean_texture`, `mean_perimeter`, `mean_area`, `mean_concavity`, `mean_concave_points`, `area_error`, `worst_radius`, `worst_texture`, `worst_perimeter`, `worst_area`, `worst_smoothness`, `worst_concavity`, `worst_concave_points`

### Correlation Matrix for 14-Feature RFE Model
<p align = "center">
<img src = "https://user-images.githubusercontent.com/60159655/89318924-4efbf500-d634-11ea-98fa-ac378cd3ee01.png" />
</p>

**Observation:**  RFE chose features that are highly correlated with one another.  Although chosing this set of features reduced the number of features of the original model by 50% and slightly increased accuracy, overfitting our model could possibly happen.  We will try and create a better model by eliminating redundant features while achieving similar accuracy scores. 

## Feature Selection (Removing Redundancy)
To improve our baseline model, we will try to remove highly correlated and redundant features from the original dataset.  For example, we have features that include the radius, perimeter, and area.  However, these quantities are directly related to one another.  The perimeter of a circle is directly proportional to the radius and the area of a circle is directly proprotational to the radius^2.  Incorporating all three features might be redundant, so we will only choose the area to include in the improved model.  Intuitively, we will keep 'area' attributes and drop 'perimeter' and 'radius' attributes.  In addition, the attribute groups containing 'concavity', 'compactness', and 'concave points' are highly correlated.  We will keep one of the three from this group (concave points).  Therefore, the following attributes will be dropped from our available list of features:

`mean_radius`, `mean_perimeter`, `mean_compactness`, `mean_concavity`, `perimeter_error`, `radius_error`, `compactness_error`, `concavity_error`, `worst_radius`, `worst_perimeter`, `worst_compactness`, `worst_concavity`

Using RFE in the manner above, we determine that from this subset of features, the model with **_11 features_** generates the best Accuracy and F1 scores:
<p align = "center">
<img src = "https://user-images.githubusercontent.com/60159655/89320103-e6ae1300-d635-11ea-8cca-ca25e4bc351a.png" />
</p>

<p align = "center">
<img src = "https://user-images.githubusercontent.com/60159655/89321053-448f2a80-d637-11ea-84ac-8c9a3f237ccd.png" />
</p>

For the model with 11 features, RFE chose the following attributes:

`mean_texture`, `mean_area`, `mean_concave_points`, `area_error`, `concave_points_error`, `worst_texture`, `worst_area`, `worst_smoothness`, `worst_concave_points`, `worst_symmetry`, `worst_fractal_dimension`

### Correlation Matrix for 11-Feature RFE Model
<p align = "center">
<img src = "https://user-images.githubusercontent.com/60159655/89324069-84f0a780-d63b-11ea-967e-a6c0de662884.png" />
</p>

**Observation:** Although there is still high correlation with some of the features, this groups of features will be less prone to overfitting compared to the 14-feature RFE model defined above.  

## Accuracy and F1 Summary of Random Forest Classifier Models
| Model | Accuracy | F1 |
| -------- | -------- | -------- |
| Baseline Model (30 features) | 0.9518 | 0.9617 |
| RFE Model (14 features) | 0.9693 | 0.9754 |
| RFE Model (11 features) | 0.9737 | 0.9790 |

**Observation:**  By using RFE, we were able to achieve slightly beter accuracy and F1 scores from the baseline model.  We increased a 2% increase in accuracy and F1 scores while reducing the number of features in the best model by more than 50% from the original model.

## Conclusion
The goal of this project was to analyze a dataset associated with breast cancer.  Using data analysis and machine learning algorithms, we were able to create a Random Forest Classifier model which could detect the correct classificaiton of a tumor by 98%.  Having a high accuracy is critical in the health care industry because cancer is a life-altering diagnosis.  Having models that correctly predict and classify cancerous cells is critical.  This model can be improved upon by using a larger dataset or even tuning the hyperparameters of the Random Forest Classifier even further. 

# Breast Cancer Classification Project

## Introduction
Cancer is a disease that affects millions of people around the world.  Machine learning has been used as a technique to help classify cancerous tumors as 'benign' or 'malignant' based on certain attributes.  For this project, I will analyze the Wisconsin Breast Cancer dataset.  The set has 569 entries with 32 variables.  The data can be obtained from the UCI Repository found [here](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)) or imported through the pre-programmed datasets in Python.  The goal of this project is to create the best model in order to classify a randomized tumor as benign (no cancer) or malignant (cancer present).  The following Python libraries will be used for this analysis:

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

## Reading the Data

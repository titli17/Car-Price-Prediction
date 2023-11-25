# Car-Price-Prediction

## **Introduction:**  

In this project, we will be predicting the prices of cars based on various features like year, owner, fuel type, transmission, etc. using Machine Learning model. Companies are using this technology to determine the prices of new cars that they produce which will help them to set the most accurate prices for their cars based on the market value of cars. As a result, optimal prices for cars are being set leading to better growth and outcomes for car manufacturers respectively.

## **Reading the libraries:**

![image](https://github.com/titli17/Car-Price-Prediction/assets/96014974/7f61a4f3-6ae2-47a8-b1d9-1d08b0cdfbbf)


Pandas is a library that is used for data analysis. It also acts as a wrapper over Matplotlib and NumPy libraries. For instance, .plot() from Pandas performs a similar operation to Matplotlib plot operations.

Seaborn provides a high-level interface that is used to draw informative statistical plots.

NumPy is used for performing a wide variety of mathematical operations for arrays and matrices. In addition to this, the steps taken for computation from the NumPy library are effective and time-efficient.

Sklearn (Scikit-learn) is one of the most popular and useful libraries that is used for machine learning in python. It provides a list of efficient techniques and tools for machine learning and statistical modelling including classification, regression, clustering, and dimensionality reduction. As could be seen from the code above, there are various sub-packages from Sklearn used for machine learning purposes respectively.


#### Note: The data was taken from 
https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho


## **Data Exploration and Visualization:**  

We will perform exploratory data analysis for the data to get insights from it.

![image](https://github.com/titli17/Car-Price-Prediction/assets/96014974/50bc661a-66e0-431b-ab24-43b540e5d88c)

![image](https://github.com/titli17/Car-Price-Prediction/assets/96014974/5dfd8ce1-009f-494d-894c-7fc843fa032c)

 We have grouped the cars based on the years of manufacture and then took the average price of the cars after grouping them. We see that as the year progresses, there is a steady average increase in the prices of cars from our data respectively.
 

![image](https://github.com/titli17/Car-Price-Prediction/assets/96014974/9ea8c647-7226-48bf-a614-600f4abc458a)

![image](https://github.com/titli17/Car-Price-Prediction/assets/96014974/41d56729-5893-47fa-8f3a-c1db33f14401)

Price of Test Drive car is the maximum followed by ascending order of the number of owners owning a particular car.


![image](https://github.com/titli17/Car-Price-Prediction/assets/96014974/0ed6b56a-5ba3-4f1f-8371-1dcc27753332)

![image](https://github.com/titli17/Car-Price-Prediction/assets/96014974/3dbded0e-0859-4a7d-a3db-45f1151cdbe9)

Automatic cars have significantly higher prices than manual ones which is being reflected in the data.


## **Data Preprocessing:**

There are no null values in our data set, so there is no room for that error rectification. 
Target encoding is a useful data preprocessing method where the categorical features are converted to numerical features so that machine learning operations could be performed. Numeric values are better understood by the computer than strings.

![image](https://github.com/titli17/Car-Price-Prediction/assets/96014974/f70b6470-b5cf-4458-afe8-95c0364a0f2d)

![image](https://github.com/titli17/Car-Price-Prediction/assets/96014974/d928187c-5ac7-4944-973d-a8c35aa289ff)

 We can see that the datatype of the categorical features have been changed to integer type after encoding.
 

## **Correlation Analysis:**

The value of the correlation coefficient ranges between -1 to 1 respectively. The higher the positive correlation between the features, the more would be correlation coefficient value would move to 1. The higher the negative correlation between the features, the more would the correlation coefficient value move to -1 respectively.

![image](https://github.com/titli17/Car-Price-Prediction/assets/96014974/585a79a7-73b5-42d5-8ba0-ad5cc91464f4)

The correlation coefficient for year and selling price is a positive one which means with increase in year the price also increases.
The correlation coefficient for kilometres driven and selling price is a negative one which means as the value of km driven increases the price of the car decreases.


## **Identifying and storing the target variable and the features separately:**

![image](https://github.com/titli17/Car-Price-Prediction/assets/96014974/603fc38d-f038-46a5-88dc-56adf715415b)

X stores the features and Y stores the target variable.

![image](https://github.com/titli17/Car-Price-Prediction/assets/96014974/9914d2bc-52e1-48bc-9c7c-9e78eab83683)

We can see that X stores all the columns except selling price and name. Y stores selling price.


## **Splitting data set into Training Data and Test Data:**

It is important to divide the data into training and test set. The training set is used to train the machine learning models under consideration. After successfully training the models, we take a look at their performance over the test set where the output is already known. After the machine learning models predict the outcome for the test set, we take those values and compare them with the known test outputs to evaluate the performance of our model.

![image](https://github.com/titli17/Car-Price-Prediction/assets/96014974/8468b384-86cc-4f4d-8e22-aa83151d7265)


## **Model Building:**

After successfully converting the values into numerical vectors, it is now time to use our machine learning models for predicting the prices of cars. We will use Linear Regression.

![image](https://github.com/titli17/Car-Price-Prediction/assets/96014974/cc2f9d5f-790b-40c7-85db-f23a3e7b425a)

Now we need to see how our model performs on the test set.
![image](https://github.com/titli17/Car-Price-Prediction/assets/96014974/568e8c88-2f43-4ed5-a04e-322cb9c2e18d)


## **Model Evaluation:**

For evaluating the performance of our model we will plot the Actual Price as x-axis and Predicted Price as y-axis. 
![image](https://github.com/titli17/Car-Price-Prediction/assets/96014974/790b920e-f463-4ef0-be7a-4fc2641de606)


The closer the values are together; it means our model is doing well. If there is a big scatter in the output plot, it means that our model is not doing well on the test set.

![image](https://github.com/titli17/Car-Price-Prediction/assets/96014974/8aea148f-fb50-440f-9ef4-fe2258cd5212)

We can see that Linear Regression is doing quite well in terms of its prediction of the prices.

We will now estimate the R-squared error and Mean Square error of our model for better evaluation.
R-squared shows how well the data fits the regression model. It can take values between 0 and 1. A low R-squared figure is generally a bad sign for predictive models.
Mean Square error measures the square of errors. Lower the value, closer is the prediction to the actual one.

![image](https://github.com/titli17/Car-Price-Prediction/assets/96014974/56b144c0-2cb9-44a9-a819-b12b6a88f825)

The model is doing quite well based on these values.


## **Strengths and Weaknesses of the Model:**

The model is doing quite well with the prediction since the target variable has a linear relationship with the features and it becomes less complex.

Over-fitting might occur because the training data set is too small and does not contain enough data samples to accurately represent all possible input data values.


## **Modifications to enhance the Model:**

•	We need more data for better prediction. Here, we had around 3906 data in training set and 434 data in test set.

•	Building more complex models for better accuracy.

•	Better processing of the data like checking the distribution of cars based upon selling price and excluding those rows which lie in the minimum distribution.

•	Over-fitting can be avoided using some dimensionality reduction techniques, regularization and cross-validation.


## **For Non-Technical Audience:**

In thus project, we have predicted the car prices using data science and machine learning models. Our data set had various features like year, owner, fuel type, transmission, seller type, kilometres driven upon which the car price was predicted.

We processed our data to avoid any null value (if present) and changed all the words like “manual”, “diesel”, etc. to numerical information for better understanding by the computer.

We checked the dependency between the car prices and the various features present in our data to find out those features upon which car prices actually depended.

The average price of cars have significantly increased over the past few years. If the number of kilometres driven by a car is more compared to others, then the price of that car is less. Test drive cars have maximum price followed by first hand, second hand, third hand, and so on.

Then we applied Linear Regression model (a machine learning model) to predict the car prices and used some metrics like Mean Squared error and R-Squared error to check the accuracy of our model.
Linear Regression model worked quite well in terms of accuracy but there are more complex models in machine learning which can help us give better results.




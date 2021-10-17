# Churn Modelling - How to predict if a bank’s customer will stay or leave the bank

Using a source of 10,000 bank records from Kaggle, I have created an app to demonstrate the ability to apply machine learning models to predict the likelihood of customer churn. We accomplished this using the following steps:

## 1. Cleaning the data

By reading the dataset into a dataframe using pandas,  I removed unnecessary data fields including individual customer IDs and names. This will leave us with a list of columns for Credit Score, Geography, Gender, Age, Length of time as a Bank customer, Balance, Number Of Bank Products Used, Has a Credit Card, Is an Active Member, Estimated Salary and Exited. 

## 2. Analyzing initial DataFrame

Utilizing Matplotlib, Seaborn and Pandas, I next analyzed the data. We can see that the dataset given was imbalanced. The majority class, "Stays" (0), has around 80% data points and the minority class, "Exits" (1), has around 20% datapoints. To address this, I utilized SMOTE in the machine learning algorithms (Synthetic Minority Over-sampling Technique). 

In percentage, female customers are more likely to leave the bank at 25%, compared to 16% of males.

The smallest number of customers are from Germany, and they are also the most likely to leave the bank. Almost one in three German customers in our sample left the bank.

## 3. Machine Learning using 7 different models

I then tested seven different machine learning models (and used six in the final application) to predict customer churn, including Logistic Regression, Decision Tree, Random Forest, Deep Learning (TensorFlow), K-Nearest Neighbor, Support Vector Machine and XGBoost. 

As mentioned earlier, I also used SMOTE to handle issues with the imbalanced data on the Support Vector Machine model. SMOTE (Synthetic Minority Over-sampling Technique) is an over-sampling method that creates new (synthetic) samples based on the samples in our minority classes. It finds the k-nearest-neighbors of each member of the minority classes. The new samples should be generated only in the training set to ensure our model generalizes well to unseen data. Using SMOTE gave better recall results which is a general goal for customer churning tasks.

## 4. Loading models to display predictions on app

Finally, using Flask and HTML/CSS, I created the user-facing app to add information to the data set matching the initial dataframe to predict the likelihood of a customer departing the bank. This was then deployed to Heroku, and can be found at this URL: https://bank-customer-churn-156.herokuapp.com/
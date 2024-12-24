# Module 18 Challenge

## Student Loan Risk Advisor

This notebook assess an applicant's risk to issue a loan to.

### Requirements

#### Prepare the Data for Use on a Neural Network Model

* Two datasets were created: a target (y) dataset, which includes the "credit_ranking" column, and a features (X) dataset, which includes the other columns.

``` python
X = loans_df.drop(columns="credit_ranking")
y = loans_df["credit_ranking"]
```

* The features and target sets have been split into training and testing datasets.

``` python
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
```

* Scikit-learn's StandardScaler was used to scale the features data.

``` python
s = StandardScaler()
X_scaler = s.fit(X_train)
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)
```

#### Compile and Evaluate a Model Using a Neural Network

* A deep neural network was created with appropriate parameters. 

``` python
num_inputs = len(X_train_scaled[0])
hidden_nodes_layer1 = 2*num_inputs+1
hidden_nodes_layer2 = 2*num_inputs+1
output_nodes = 1
nn = tf.keras.models.Sequential()
nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer1, input_dim=num_inputs, activation="relu"))
nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer2, activation="relu"))
nn.add(tf.keras.layers.Dense(units=output_nodes, activation="sigmoid"))
```

* The model was compiled and fit using the accuracy loss function, the adam optimizer, the accuracy evaluation metric, and a small number of epochs, such as 50 or 100. 

``` python
nn.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]) 
fit_model = nn.fit(X_train_scaled,y_train,epochs=50) 
```

* The model was evaluated using the test data to determine its loss and accuracy. 

``` python
model_loss, model_accuracy = nn.evaluate(X_test_scaled,y_test, verbose=3)
print(f"Loss: {model_loss}, Accuracy: {model_accuracy}") 
```

* The model was saved and exported to a keras file named student_loans.keras.

``` python
file_path = Path("student_loans.keras")
nn.save(file_path) 
```

#### Predict Loan Repayment Success by Using your Neural Network Model

* The saved model was reloaded.

``` python
file_path = Path("student_loans.keras")
nn_imported = tf.keras.models.load_model(file_path)
```

* The reloaded model was used to make binary predictions on the testing data.

``` python
predictions = nn_imported.predict(X_test)
p = pd.DataFrame({"Predictions":list(predictions)})
p['Predictions'] = [round(x[0]) for x in p["Predictions"]]
p
```

* A classification report is generated for the predictions and the testing data.

``` python
pp=list(p["Predictions"])
print(f"Classification Report:\n {classification_report(y_test,pp)}")
```

#### Discuss creating a recommendation system for student loans

##### Question 1

* The response describes the data that should be collected to build a recommendation system for student loan options.
* The response explains why they think that data should be collected.
* The type of data described is appropriate for a recommendation system for student loan options.

``` python
'''
The data that should be collected is the data that is the most closely related to financial habits. I would focus more on features such as current debt (credit and other), income (scholarships, job, etc), delinquent debt, credit history, and employment history. These features are better suited for loan repayment calculations because they are more closely tied with an individuals financial habits. Current debt and income will give a good idea about what other loan obligations does this individual have and will they be able to cover this new loan on top of the other debt. Delinquent debt payments and credit history will show the individual's ability to pay on time and consistency. Employment history will show if this individual has a stable source of income that can be relied on to pay back the loan for the life of the loan.
'''
```

##### Question 2

* The response chose a filtering method.
* The student justified the choice of their filtering method.
* The choice of filtering method was appropriate for the data selected in the previous question.

``` python
'''
The model would use collaborative filtering. This is because we are looking for similarities between users on what each feature is and what interest rate they took and if they repay it or not. It is reasonable to think that users with similar financial habits would make better candidtates for repaying the loan and so we can offer better rates to sell the loan. In other words, there is a higher probability that a new applicant will pay back the loan if their features are similar to the existing students who hve paid back their loans.
'''
```

##### Question 3

* The response lists two real-world challenges with building a recommendation system for student loans. (4 points)
* The response explains why these challenges would be of concern for a student loan recommendation system. (6 points)

``` python
'''
One challenge would be the inherent unpredictability of a student's situation to remain a good candidate for the life of the loan. Meaning, even though all the features at the time of the application indicated they were a good candidate, something could happen that dramatically changes their features and causes them to become delinquent on the loan. Another challenge is the quality of the data. This is very confidential information that is required and there is a risk of false data which could adversly effect other applicants' ability to qualify for a better loan.
'''
```
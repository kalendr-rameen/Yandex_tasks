# client_bank_prediction

Clients began to leave Beta Bank. Every month. Not much, but noticeable. The bank's marketing experts figured it was cheaper to keep current customers than to attract new ones.
We need to predict whether the client will leave the bank in the near future or not. We are provided with historical data on customer behavior and termination of contracts with the bank.
Construct a model with an extremely high value of F1-measure. To pass the project successfully, you need to bring the metric to 0.59. Test F1-measure on the test sample by yourself.
Additionally measure AUC-ROC, compare its value with F1-measure.

```
Attributes 

  RowNumber - index of the row in the data
  CustomerId - unique identifier of the client
  Surname - last name
  CreditScore - credit rating
  Geography - country of residence
  Gender - gender
  Age - age
  Tenure - how many years the person has been the client of the bank
  Balance - account balance
  NumOfProducts - number of bank products used by the client
  HasCrCard - availability of credit card
  IsActiveMember - activity of the client
  EstimatedSalary - expected salary
  
Target attribute

  Exited - the fact that the client left
```

import numpy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Append.csv')
df = df[['GrLivArea', 'BedroomAbvGr', 'FullBath', 'SalePrice']]

print('DATAFRAME NULL INFO:-')
print(df.isnull().sum())
df = df.dropna()

print('DATAFRAME:-')
print(df)
print('DATAFRAME INFO:-')
print(df.info())
print('DATAFRAME DESCRIPTION:-')
print(df.describe())


plt.title('Feature Correlation Matrix')
sns.heatmap(df.corr(), annot=True)
plt.show()

x = df[['GrLivArea', 'BedroomAbvGr', 'FullBath']]
y = df[['SalePrice']]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.17, random_state=42)
model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print('Mean Squared Error is: ', mean_squared_error(y_test, y_pred))
print('R^2 Score is: ', r2_score(y_test, y_pred))

plt.title('Actual vs Predicted Sale Prices')
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Sale Prices')
plt.ylabel('Predicted Sale Prices')
plt.show()

s = int(input('Enter the square foot: '))
be = int(input('Enter the number of bedrooms: '))
ba = int(input('Enter the number of bathrooms: '))
user_input = pd.DataFrame([[s, be, ba]], columns=['GrLivArea', 'BedroomAbvGr', 'FullBath'])
user_input_prediction = model.predict(user_input)
print('Predicted Sale Price is: ', user_input_prediction)
# STOCK PRICE PREDICTION USING MACHINE LEARNING MODELS 
# NAME: RUTH OKOSUN 
# SCHOOL: GRAND CANYON UNIVERSITY 

#LIBRARIES LOADING 
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression 
from  sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")
from pylab import rcParams
rcParams['figure.figsize'] = 10, 6
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import math
import sklearn.metrics as metrics
%matplotlib inline
from sklearn.metrics import mean_squared_error, mean_absolute_error

#.................DATA COLLECTION...........

# reading stock data from yahoo AND USING THE API TO GET THE STOCK DATA 
from pandas_datareader.data import DataReader
import yfinance as yf
from pandas_datareader import data as pdr
yf.pdr_override()

# FOR TIME STAMP
from datetime import datetime

#THE TICKER IS USED TO ENTER THE SYMBOL OF THE DESIRED STOCK COMPANY 
stocks = input("Enter your stock ticker: ")
company_name = []
company_list = []
tech_list = [item.strip() for item in stocks.split(',')]
stockCount = len(tech_list)
start = datetime(2014, 1, 1)
end = datetime.now()


for stock in tech_list:
    stockData = yf.download(stock, start, end)
    companyStockInfo = yf.Ticker(stock)
    company_name.append(companyStockInfo.info['longName'])
    globals()["stockKey"] = stockData
    company_list.append(stockData)  
df = pd.concat(company_list, axis=0)

#VIEWING THE HISTORICAL STOCK PRICE OF THE SELECTED COMPANY 
df.head(10)

#CHOOSING THE INDEPENDENT VARIABLE AND VISUALING IT 

df = df[["Adj Close"]]
print (df)
plt.figure(figsize=(15, 10))
plt.subplots_adjust(top=1.25, bottom=1.2)

for i, company in enumerate(company_list, 1):
    plt.subplot(stockCount, 1, i)
    company['Adj Close'].plot()
    plt.ylabel('Adj Close')
    plt.xlabel(None)
    plt.title(f"Closing Price of {tech_list[i - 1]}")
print(company_list)
    
plt.tight_layout()

ma_day = [10, 20, 50]

for ma in ma_day:
    for company in company_list:
        column_name = f"MA for {ma} days"
        company[column_name] = company['Adj Close'].rolling(ma).mean()

fig, axes = plt.subplots(1)
fig.set_figheight(10)
fig.set_figwidth(15)

stockKey[['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot(ax = axes)
axes.set_title(stocks)
fig.tight_layout()

# CALCUALTING THE PERCENTAGE CHANGE OF EACH DAY 
for company in company_list:
    company['Daily Return'] = company['Adj Close'].pct_change()
# PLOTING THE RESULT 
fig, axes = plt.subplots(1)
fig.set_figheight(10)
fig.set_figwidth(15)
stockKey['Daily Return'].plot(ax=axes, legend=True, linestyle='--', marker='o')
axes.set_title(stocks)
fig.tight_layout()
plt.figure(figsize=(12, 9))

for i, company in enumerate(company_list, 1):
    plt.subplot(stockCount, 1, i)
    company['Daily Return'].hist(bins=50)
    plt.xlabel('Daily Return')
    plt.ylabel('Counts')
    plt.title(f'{company_name[i - 1]}')  
plt.tight_layout()

# Grab all the closing prices for the tech stock list into one DataFrame

closing_df = pdr.get_data_yahoo(tech_list, start=start, end=end)['Adj Close']

# Make a new tech returns DataFrame
tech_rets = closing_df.pct_change()
tech_rets.head()

tech_rets = tech_rets.to_frame().reset_index()
#print(type(tech_rets))# Grab all the closing prices for the tech stock list into one DataFrame

closing_df = pdr.get_data_yahoo(tech_list, start=start, end=end)['Adj Close']

# Make a new tech returns DataFrame
tech_rets = closing_df.pct_change()
tech_rets.head()

tech_rets = tech_rets.to_frame().reset_index()
#print(type(tech_rets))
rets = tech_rets.dropna()
print(rets.mean())
print(rets.std())
area = np.pi * 20


#.......................... DATA PREPROCESSING................................
# CREATING A NEW DATAFRAME WITH THE ADJ CLOSE 
data = df.filter(['Adj Close'])
# Convert the dataframe to a numpy array
dataset = data.values
# Get the number of rows to train the model on
training_data_len = int(np.ceil( len(dataset) * .95 ))
training_data_len

#inputing a variable to predict n number of days 
forecast_out = 30
#since we need a dependent variable too, we are going to be using the shift up function to get the target variable from the Adjusted close, that will be used for this prediction 
df ["Prediction"] = df [["Adj Close"]].shift(-forecast_out)
#view the Adj close abf prediction (target)dataset 
print(df)

#i am going to be creating the independent row 
#first we will have to converg the 
X = np.array(df.drop(["Prediction"],axis=1))
#i will need to remove the n forecast_out rows 
X = X[:-forecast_out]
print (X)
#create the dependent dataset (y)
# i will have to convert all the varible to numpy arrary including the Nan variables 
y = np.array(df["Prediction"])
#getting al the values of y excluding only the n row 
y = y[:-forecast_out]
#view the data output 
print(y)

#split my dataset into 70% traning set  and 30% test set
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# performing preprocessing part, Scale the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


#.....................BUILDING THE various MACHINE LEARNING MODELS ................
# for this program we are going to start our prediction using the SVM model 

#BUILD AND TEST THE SVR MODEL SVR 
svr_rbf = SVR(kernel="rbf", C=1e3, gamma=0.1)
svr_rbf.fit(x_train, y_train)
#model testing: this will simply return the coefficient of determination R^2 of the prediction 
#the best possible score should be 1.0
svm_confidence = svr_rbf.score(x_test, y_test)
print ("svm confidence:" , svm_confidence )
#BUILD AND TEST THE LINEAR REGRESSION MODEL
#the second model will will create and train is the linear regression model 
lr = LinearRegression()
#to train the model 
lr.fit(x_train , y_train)
#model testing: this will simply return the coefficient of determination R^2 of the prediction 
#the best possible score should be 1.0
lr_confidence = lr.score(x_test, y_test)
print ("lr confidence:" , lr_confidence )

#PREDICTING THE NEXT N DAYS
#lets set forecast to predict the next 30 rows of our original datasetfrom the Ajusted Close column
x_forecast = np.array(df.drop(["Prediction"], axis=1))[-forecast_out:]
print(x_forecast)
#print the SVr model prediction for the next "n" (30) days
svr_prediction = svr_rbf.predict(x_test)
print(svr_prediction)


#PERFORMANCE EVALUATION USING METRICS 

#PERFORMANCE EVALUATION FOR SVR

from sklearn.metrics import mean_squared_error

mse = np.sqrt(mean_squared_error(y_test, svr_prediction))
print('MSE:', mse)
rmse = metrics.mean_squared_error(y_test,svr_prediction, squared=False)
print('RMSE:', rmse)
mae = metrics.mean_absolute_percentage_error(y_test, svr_prediction)
print('MAE:', mae)

#PLOTING THE RESULT FOR SVR
plt.figure(figsize=(16,6))
plt.title('SVR')
plt.plot(df['Prediction'])
plt.xlabel('DATE', fontsize=18)
plt.ylabel('Adj Close', fontsize=18)
plt.show()

#print the linear regression model prediction for the next "n" (30) days
lr_prediction = lr.predict(x_test)
print(lr_prediction)

#PERFORMANCE EVALUATION OF LINEAR REGRESSION 
mse = metrics.mean_squared_error(y_test,lr_prediction)
print('MSE:', mse)
rmse = metrics.mean_squared_error(y_test,lr_prediction, squared=False)
print('RMSE:', rmse)
mae = metrics.mean_absolute_percentage_error(y_test, lr_prediction)
print('MAE:', mae)

#PLOTTING THE LINEAR REGRESSION 
plt.figure(figsize=(16,6))
plt.title('LinearRegression')
plt.plot(df['Prediction'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Adj Close', fontsize=18)
plt.show()

#BUIDING THE LSTM MODEL 
#I HAVE TO RUN THIS MODEL SEPERATELY, I WILL ATTACH THE FULL CODE SEPERATLY 

#COLLECT DATA FROM YFINANCE 

df = pdr.get_data_yahoo(stocks, start='2012-01-01', end=datetime.now())
# DATA VISUALIZATION 
print(df)

# PLOTING THE DATA ON A GRAPH 
plt.figure(figsize=(16,6))
plt.title('Close Price History')
plt.plot(df['Adj Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
st.pyplot()

#DATA CLEANING AND PREPROCESSING 
# Create a new dataframe with only the 'ADJ Close column 
data = df.filter(['Adj Close'])

# Convert the dataframe to a numpy array
dataset = data.values

#FEATURE EXTRACTION 
# Get the number of rows to train the model on
training_data_len = int(np.ceil( len(dataset) * .95 ))
training_data_len

# SCALING THE DATA 
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)
scaled_data

# CREATING THE TRAINING SET 
# Create the scaled training data set
train_data = scaled_data[0:int(training_data_len), :]
# Split the data into x_train and y_train data sets
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
    if i<= 61:
        print(x_train)
        print(y_train)
        print()
        
# Convert the x_train and y_train to numpy arrays 
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
#x_train.shape
from tensorflow.keras.layers import Dense, LSTM
from keras.models import Sequential
#keras.layers import Dense, lstm

# BUILDING THE LSTM MODEL 
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape= (x_train.shape[1], 1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
mode.summary()

# COMPILING THE MODEL 
model.compile(optimizer='adam', loss='mean_squared_error')

# TRAINING THE MODEL 
model.fit(x_train, y_train, batch_size=1, epochs=5)
# Create the testing data set
# Create a new array containing scaled values from index 1543 to 2002 
test_data = scaled_data[training_data_len - 60: , :]
# Create the data sets x_test and y_test
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])
    
# Convert the data to a numpy array
x_test = np.array(x_test)

# Reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))
print(x_test)

# PREDICTING N DAYS WITH THE MODEL 
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
#USING PERFORMANCE METRIC FOR EVALAUTING THE MODEL 
# Get the root mean squared error (RMSE)
rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
rmse

# PLOTTING THE PREDICTION MODEL
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions
# Visualize the data
plt.figure(figsize=(16,6))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Adj Close'])
plt.plot(valid[['Adj Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
print(predictions)
#st.pyplot()


# Show the valid and predicted prices
valid

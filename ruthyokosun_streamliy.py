
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
#sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")


st.title("Stock Price Trend Prediction ")
st.header("Capstone Milestone 3")
st.sidebar.header("Ruth Okosun ")

# For reading stock data from yahoo
from pandas_datareader.data import DataReader
import yfinance as yf
from pandas_datareader import data as pdr

yf.pdr_override()

# For time stamps

from datetime import datetime
stocks = st.text_input("Enter your stock ticker: ")
while stocks == "":
    continue
company_name = []
company_list = []

tech_list = [item.strip() for item in stocks.split(',')]
stockCount = len(tech_list)

end = datetime.now()
start = datetime(end.year - 1, end.month, end.day).strftime("%Y-%m-%d")

for stock in tech_list:
    stockData = yf.download(stock, start, end)
    companyStockInfo = yf.Ticker(stock)
    st.write(company_name.append(companyStockInfo.info['longBusinessSummary']))
    st.write('Historical Stock Price')
    globals()["stockKey"] = stockData
    company_list.append(stockData)


n_years = st.slider("Year of Prediction:", 1 , 5)
period = n_years * 365

    
df = pd.concat(company_list, axis=0)
st.write(df.tail(10))

# Let's view a historical view of the closing price
st.set_option('deprecation.showPyplotGlobalUse', False)
plt.figure(figsize=(15, 10))
plt.subplots_adjust(top=1.25, bottom=1.2)

for i, company in enumerate(company_list, 1):
    plt.subplot(stockCount, 1, i)
    company['Adj Close'].plot()
    plt.ylabel('Adj Close')
    plt.xlabel(None)
    plt.title(f"Closing Price of {tech_list[i - 1]}" )
    plt.legend([tech_list[i - 1]])
   
plt.tight_layout()
st.pyplot()
# Create a new dataframe with only the 'Close column 
data = df.filter(['Adj Close'])
# Convert the dataframe to a numpy array
dataset = data.values
# Get the number of rows to train the model on
training_data_len = int(np.ceil( len(dataset) * .95 ))
training_data_len

# Scale the data
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

scaled_data

# Create the training data set 
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
# x_train.shape
from tensorflow.keras.layers import Dense, LSTM
from keras.models import Sequential
#keras.layers import Dense, lstm

# Build the LSTM model
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape= (x_train.shape[1], 1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
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
# Get the models predicted price values 
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)



# Plot the data
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
#st.subheader('Prediction')
#st.write(predictions)
st.write(f'Prediction plot for {n_years} years')
st.pyplot()


# Get the root mean squared error (RMSE)
st.subheader("Root Mean Squared Error")
rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
st.write(rmse)

# Show the valid and predicted prices
st.subheader('Predicted Stock Price')
st.write(valid)



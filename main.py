import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import joblib
import os

#title
app_name = "Stock Price Prediction App"
st.write("# " + app_name)
st.subheader("This app predicts the stock price of a company using machine learning techniques.")

#add image from online
st.image("https://media.istockphoto.com/id/1487894858/photo/candlestick-chart-and-data-of-financial-market.jpg?s=612x612&w=0&k=20&c=wZ6vVmbm4BV2JOePSnNNz-0aFVOJZ0P9nhdeOMGUg5I=")

#take input from user about the start and end date and create side bar

st.sidebar.title("select the parameters")
start_date = st.sidebar.date_input("Start Date", datetime.date(2024, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date.today())

#add ticker symbol list
ticker_list = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 
               'JPM', 'JNJ', 'V', 'PG', 'MA', 'UNH', 'HD', 
               'BAC', 'WMT', 'DIS', 'KO', 'XOM', 'CVX', 'PFE', 
               'MRK', 'NFLX', 'NVDA', 'INTC', 'CSCO', 'T', 'CMCSA', 
               'PEP', 'ABBV', 'ABT', 'MCD', 'WBA', 'COST', 'TMO', 'ORCL', 
               'PYPL', 'TXN', 'QCOM', 'AMGN', 'AMAT', 'ADBE', 'AVGO', 'BKNG', 
               'ADI', 'LMT', 'ISRG', 'MDLZ', 'TMUS', 'MU', 'NXPI', 'TM', 'AMD', 
               'KLAC', 'LRCX', 'KLAC', 'LRCX', 'KLAC', 'LRCX', 'KLAC', 'LRCX', 
               'KLAC', 'LRCX', 'KLAC', 'LRCX',
               # Adding top NSE stocks
               'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS',
               'ICICIBANK.NS', 'HDFC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'KOTAKBANK.NS',
               'LT.NS', 'ITC.NS', 'AXISBANK.NS', 'ASIANPAINT.NS', 'MARUTI.NS']

ticker = st.sidebar.selectbox("Select a ticker symbol", ticker_list)

#fetch data from user inputs using yfinance library

data = yf.download(ticker, start=start_date, end=end_date)

# add date as a column to dataframe

data.insert(0, 'Date', data.index, True)
data.reset_index(drop=True, inplace=True)
st.write("#### Data from", start_date, "to", end_date)
st.write(data)

#Plot the data
st.write("#### Data Visualization")
st.subheader("Plot of the data")
fig = px.line(data, x="Date", y=data.columns, title="closing price", width=1000, height=500)
st.plotly_chart(fig)

# add a column to select column from data

column = st.selectbox("Select a column for forecast", data.columns[1:])

data = data[['Date',column]]
st.write("Selected Data")
st.write(data)

#ADF test check stationary

st.header("Is Data Stationary ?")
st.write("####",adfuller(data[column])[1] < 0.05)
decomposition = seasonal_decompose(data[column], model='additive', period=12)

# make same plot in plotly
st.plotly_chart(px.line(x=data["Date"], y=decomposition.trend, title="Trend", width=1200, height=400, labels={"x": "Date", "y": "Price"}).update_traces(line_color='red'))
st.plotly_chart(px.line(x=data["Date"], y=decomposition.seasonal, title="Seasonal", width=1200, height=400, labels={"x": "Date", "y": "Price"}).update_traces(line_color='green'))
st.plotly_chart(px.line(x=data["Date"], y=decomposition.resid, title="Residual", width=1200, height=400, labels={"x": "Date", "y": "Price"}).update_traces(line_color='blue', line_dash='dot'))    

#user input for three parameters of the model and seasonal order

# p: The order of the autoregressive term (AR)
p = st.slider("Select the value of p (AR order)", 0, 5, 1)
# d: The degree of differencing (I)
d = st.slider("Select the value of d (Differencing)", 0, 5, 1)
# q: The order of the moving average term (MA)
q = st.slider("Select the value of q (MA order)", 0, 5, 1)

seasonal_order = st.number_input("Select the value of seasonal p", 0, 24, 12)


# Create a unique filename for the model based on parameters
model_filename = f"training_models/sarimax_model_{ticker}_{column}_{p}_{d}_{q}_{seasonal_order}.joblib"
st.write("## *******************************************")
# Check if the model file exists
if os.path.exists(model_filename):
    # Load the existing model
    model = joblib.load(model_filename)
    st.write("## Loaded existing model.")
    st.write("** Note: ** Any change in values will create a new model.")
else:
    # Create and fit a new model
    model = sm.tsa.statespace.SARIMAX(data[column], order=(p, d, q), seasonal_order=(seasonal_order, 0, 0, 12)).fit()
    # Save the model
    joblib.dump(model, model_filename, compress=8)
    st.write("## Created and saved new model.")
st.write("## *******************************************")
#print model summary
st.header("Model Summary")
st.write(model.summary())
st.write("----------")


# predict future values (Forecasting)
st.write("Forecasting the Data")
forecast_period = st.slider("Select the number of days to forecast", 1, 365, 30)

predictions = model.get_prediction(start=len(data), end=len(data)+forecast_period)
predictions = predictions.predicted_mean

    #add index to results dataframe as dates
predictions.index = pd.date_range(start=end_date, periods=len(predictions), freq='D')
predictions = pd.DataFrame(predictions)
predictions.insert(0, 'Date', predictions.index, True)
predictions.reset_index(drop=True, inplace=True)
st.write("## Predictions", predictions)
st.write("## Actual Data", data)
st.write("---")

#plot the data

fig = go.Figure()
fig.add_trace(go.Scatter(x=data["Date"], y=data[column], name="Actual", mode="lines", line=dict(color='blue')))
fig.add_trace(go.Scatter(x=predictions["Date"], y=predictions["predicted_mean"], name="Predicted", mode="lines", line=dict(color='red')))
fig.update_layout(title="Actual vs Predicted", xaxis_title="Date", yaxis_title="Price", width=1200, height=400)

st.plotly_chart(fig)
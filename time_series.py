import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import io
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib
import plotly.graph_objs as go

# Function to adjust forecast dates based on selected frequency
def adjust_forecast_dates(start_date, periods, frequency):
    if frequency == 'Daily':
        return pd.date_range(start=start_date, periods=periods, freq='D')
    elif frequency == 'Monthly':
        return pd.date_range(start=start_date, periods=periods, freq='M')
    elif frequency == 'Quarterly':
        return pd.date_range(start=start_date, periods=periods, freq='Q')
    elif frequency == 'Yearly':
        return pd.date_range(start=start_date, periods=periods, freq='A')

# Load data function
@st.cache_data
# def load_data():
#     data = pd.read_csv('tsa-train.csv', parse_dates=['date'])
#     return data

def load_data():
    # Path to the ZIP file containing the CSV
    zip_path = './ezyzip.zip'
    
    # Open the ZIP file
    with zipfile.ZipFile(zip_path, 'r') as z:
        # Open the CSV file within the ZIP
        with z.open('tsa-train.csv') as csv_file:
            # Read the CSV file directly into pandas
            data = pd.read_csv(csv_file, parse_dates=['date'])
    
    return data

# Load model and scaler function
@st.cache_data
def load_model_and_scaler():
    scaler = joblib.load('scaler.pkl')
    lstm_model = load_model('my_model.h5')
    return scaler, lstm_model

# SNAIVE Forecasting function
def snaive_forecasting(data, periods):
    season_length = 12  # Adjust based on your seasonal cycle
    last_season = data[-season_length:]
    forecast_dates = adjust_forecast_dates(data['date'].iloc[-1] + pd.DateOffset(days=1), periods, frequency_selection)
    forecast_sales = np.tile(last_season['sales'].values, int(np.ceil(periods/season_length)))[:periods]
    forecast_df = pd.DataFrame({'date': forecast_dates, 'sales': forecast_sales})
    return forecast_df

# LSTM Forecasting function
def lstm_forecasting(data, periods, scaler, lstm_model):
    last_inputs = data['sales'].values[-1]  # Adjust based on your model's requirements
    last_inputs_scaled = scaler.transform(np.array([last_inputs]).reshape(-1, 1))
    forecast_dates = adjust_forecast_dates(data['date'].iloc[-1] + pd.DateOffset(days=1), periods, frequency_selection)
    forecast_sales = []
    current_input = last_inputs_scaled
    for _ in range(periods):
        pred = lstm_model.predict(current_input.reshape(1, 1, -1))
        forecast_sales.append(scaler.inverse_transform(pred)[0][0])
        current_input = pred
    forecast_df = pd.DataFrame({'date': forecast_dates, 'sales': forecast_sales})
    return forecast_df

# Plot sales over time function
def plot_sales_over_time(data, title='Sales Over Time'):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['date'], y=data['sales'], mode='lines', name='Sales'))
    fig.update_layout(title=title, xaxis_title='Date', yaxis_title='Sales')
    return fig

# Streamlit app layout
st.title('Sales Forecasting in Grocery Stores')
df = load_data()
scaler, lstm_model = load_model_and_scaler()
st.write("Here's a glimpse of the dataset:", df.head())

# Sidebar options
st.sidebar.header('Settings')
family_selection = st.sidebar.selectbox('Select Product', df['family'].unique())
days_to_predict = st.sidebar.slider('Days to Predict', 1, 30, 2)
frequency_selection = st.sidebar.selectbox('Select Prediction Frequency', ['Daily', 'Monthly', 'Quarterly', 'Yearly'])
model_selection = st.sidebar.radio('Forecasting Model Applied', ['Naive', 'ARIMA', 'SNAIVE', 'LSTM'])
filtered_data = df[df['family'] == family_selection]

# Main content
fig = plot_sales_over_time(filtered_data)
st.plotly_chart(fig)

if model_selection == 'Naive':
    forecast_dates = adjust_forecast_dates(filtered_data['date'].iloc[-1] + pd.DateOffset(days=1), days_to_predict, frequency_selection)
    last_value = filtered_data['sales'].iloc[-1]
    forecast_values = [last_value for _ in range(len(forecast_dates))]
    forecast_df = pd.DataFrame({'date': forecast_dates, 'sales': forecast_values})
    fig_forecast = plot_sales_over_time(forecast_df, title='Naive Forecasted Sales')
    st.plotly_chart(fig_forecast)

elif model_selection == 'SNAIVE':
    forecast_df = snaive_forecasting(filtered_data, days_to_predict)
    fig_forecast = plot_sales_over_time(forecast_df, title='SNAIVE Forecasted Sales')
    st.plotly_chart(fig_forecast)

elif model_selection == 'LSTM':
    forecast_df = lstm_forecasting(filtered_data, days_to_predict, scaler, lstm_model)
    fig_forecast = plot_sales_over_time(forecast_df, title='LSTM Forecasted Sales')
    st.plotly_chart(fig_forecast)

elif model_selection == 'ARIMA':
    arima_order = (5, 1, 0)
    arima_model = ARIMA(filtered_data['sales'], order=arima_order)
    arima_result = arima_model.fit()
    forecast_values = arima_result.forecast(steps=days_to_predict)
    forecast_dates = adjust_forecast_dates(filtered_data['date'].iloc[-1] + pd.DateOffset(days=1), days_to_predict, frequency_selection)
    forecast_df = pd.DataFrame({'date': forecast_dates, 'sales': forecast_values})
    fig_forecast = plot_sales_over_time(forecast_df, title='ARIMA Forecasted Sales')
    st.plotly_chart(fig_forecast)

import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import altair as alt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose

# Function to fetch stock data and make predictions
def get_stock_data(ticker, start_date, end_date, forecast_days=30):
    data = yf.download(ticker, start=start_date, end=end_date)
    df = data.reset_index()[['Date', 'Close']]
    df.columns = ['ds', 'y']

    # Use SARIMAX model for forecasting
    model = SARIMAX(df['y'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 30))
    results = model.fit(disp=False)
    future = pd.date_range(start=df['ds'].iloc[-1] + pd.Timedelta(days=1), periods=forecast_days, freq='D')
    forecast = results.predict(start=len(df), end=len(df)+forecast_days-1)

    forecast_df = pd.DataFrame({'ds': future, 'yhat': forecast.values})

    return data, forecast_df

# Function to get N-day historical indices performance
def get_indices_performance(indices, end_date, days=10):
    performance = []
    for index in indices:
        data = yf.download(index, start=end_date - timedelta(days=days+1), end=end_date)
        change_pct = (data['Close'] - data['Close'].shift(1)) / data['Close'].shift(1) * 100
        performance.append(change_pct)
    return performance

# App title
st.title("Stock App")

# Input for stock ticker
ticker = st.text_input("Enter the stock ticker:", "AAPL")

# Date range input
start_date = st.date_input("Start date:", datetime.today() - timedelta(days=365))
end_date = st.date_input("End date:", datetime.today())

# Fetch stock data and predictions
stock_data, forecast = get_stock_data(ticker, start_date, end_date)

# Combine the actual and forecast data
stock_forecast = stock_data[['Close']].merge(forecast[['ds', 'yhat']], left_on=stock_data.index, right_on=forecast['ds'], how='outer')

# Display line chart
st.subheader("Stock Price and Forecast")
chart = alt.Chart(stock_forecast).transform_fold(
    ['Close', 'yhat'],
    as_=['Variable', 'Value']
).mark_line().encode(
    x=alt.X('ds:T', title='Date'),
    y=alt.Y('Value:Q', title='Price'),
    color=alt.Color('Variable:N', legend=alt.Legend(title='Series')),
    tooltip=['ds', 'Value']
).interactive()

st.altair_chart(chart, use_container_width=True)


st.altair_chart(chart, use_container_width=True)

# Indices list
indices = ['^GSPC', '^DJI', '^IXIC', '^RUT', '^N225', '^FTSE', '^GDAXI', '^FCHI', '^HSI', '^AXJO']

# Fetch 10 indices performance
performance = get_indices_performance(indices, end_date)

# Modify the layout and display historical data
col1, col2 = st.columns(2)

with col1:
    st.subheader("Indices Performance")
    performance_df = pd.DataFrame({"Index": indices, "Performance (%)": performance})
    for idx, row in performance_df.iterrows():
        st.write(f"{row['Index']}: {row['Performance (%)'][-1]:.2f}%")

with col2:
    st.subheader("Stock Prediction vs Actual Price")
    st.write(f"Actual Price: {stock_data['Close'].iloc[-1]:.2f}")
    st.write(f"Predicted Price: {forecast['yhat'].iloc[-1]:.2f}")

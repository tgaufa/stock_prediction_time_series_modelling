import streamlit as st
import altair as alt

import src.util as util
import pandas as pd
import numpy as np

params = util.load_config()

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

actual_price = pd.DataFrame(util.pickle_load(params["raw_dataset_path"])[params['target_stock']])
predict_price = pd.DataFrame(util.pickle_load(params["predict_dataset_path"]))

actual_price['symbol'] = params["target_stock"] + ' Actual'
predict_price['symbol'] = params["target_stock"] + ' Prediction'
predict_price.rename(columns={predict_price.columns[0]: 'price'}, inplace=True)
actual_price.rename(columns={actual_price.columns[0]: 'price'}, inplace=True)

stock_price = pd.concat([actual_price, predict_price], axis=0)
stock_price.index.name = 'date'
stock_price.reset_index(inplace=True)



latest_price = stock_price[stock_price['symbol'] == params["target_stock"] + ' Actual'].sort_values(by='date', ascending=False).iloc[0]['price']
latest_price_lag = stock_price[stock_price['symbol'] == params["target_stock"] + ' Actual'].sort_values(by='date', ascending=False).iloc[1]['price']
change = (latest_price-latest_price_lag)/latest_price_lag

latest_pred = stock_price[stock_price['symbol'] == params["target_stock"] + ' Prediction'].sort_values(by='date', ascending=False).iloc[0]['price']
#latest_pred_lag = stock_price[stock_price['symbol'] == params["target_stock"] + ' Prediction'].sort_values(by='date', ascending=False).iloc[1]['price']
change_pred = (latest_pred-latest_price_lag)/latest_price_lag

change_var = change_pred-change
price_var = latest_pred-latest_price
# first row
a1, a2, a3 = st.columns(3)
with a1:
    a1.metric("Actual", f'{latest_price: .0f}', f'{change: .1%}')

with a2:
    a2.metric("Prediction", f'{latest_pred: .0f}', f'{change_pred: .1%}')

with a3:
    a3.metric("Difference", f'{price_var: .0f}', f'{change_var: .1%}')


# second row

st.subheader("Stock Price and Forecast")
base = alt.Chart(stock_price).encode(
).properties(
    width=600,
    height= 300
)

line = base.mark_line().encode(
    x="date", 
    y="price",
    color=alt.Color('symbol:N', scale=alt.Scale(domain=['BMRI.JK Actual', 'BMRI.JK Prediction'], range=['white', 'green']), legend=None)

    )

nearest = alt.selection(type='single', nearest=True, on='mouseover', fields=['date'], empty='none')

# Add a vertical line that follows the cursor
vertical_line = base.mark_rule(color='gray').encode(
    x='date:T'
).transform_filter(
    nearest
)

band = alt.Chart(stock_price).mark_errorband(extent='ci').encode(
    x='date',
    y=alt.Y('price')
)

last_price = base.mark_circle().encode(
    alt.X("last_date['date']:T"),
    alt.Y("last_date['price']:Q")
).transform_aggregate(
    last_date="argmax(date)",
    groupby=["symbol"]
)
company_name = last_price.mark_text(align="left", dx=4).encode(
    text="symbol",
    color=alt.Color("symbol:N", scale=alt.Scale(domain=['BMRI.JK Actual', 'BMRI.JK Prediction'], range=['white', 'green']), legend=None)
)


# Add a text layer to display the data labels at the cursor position
text = base.mark_text(align='left', dx=4, dy=-4).encode(
    x='date:T',
    y='price:Q',
    detail='symbol:N',
    text=alt.condition(nearest, 'price:Q', alt.value(' '))
).add_selection(
    nearest
)

chart1 = (line + band + vertical_line + company_name + text + last_price).encode(
    x=alt.X('date:T', title="date"),
    y=alt.Y('price:Q', title="price")
).interactive()

chart1




indices = util.pickle_load(params["clean_dataset_path"])



# Get the last 10 dates
last_5_dates = indices.tail(5).index[::-1]


for date in last_5_dates:
    st.write(f"Date: {date.strftime('%Y-%m-%d')}")  # Display the date as a subheader

    # Extract the data for the current date
    date_data = indices.loc[date].to_frame().T

    # Transform the DataFrame into a long format
    long_format_data = date_data.melt(var_name='index_name', value_name='price')

    # Get the minimum and maximum values of the 'price' column
    min_value = long_format_data['price'].min()
    max_value = long_format_data['price'].max()

    # Calculate the symmetric range around zero
    symmetric_range = max(abs(min_value), abs(max_value))


# Create the Altair bar chart using the transformed DataFrame
    target_index_chart = alt.Chart(long_format_data).transform_filter(
        alt.datum.index_name == 'BMRI.JK'
    ).mark_bar().encode(
        x=alt.X('index_name:N', title='Index'),
        y=alt.Y('price:Q', title='Price', scale=alt.Scale(domain=[-symmetric_range, symmetric_range])),
        color=alt.condition(
            alt.datum.price > 0,
            alt.value("seagreen"),  # The positive color
            alt.value('#e06666')     # The negative color
        )
    ).properties(width=600, height=200)

    other_indices_chart = alt.Chart(long_format_data).transform_filter(
        alt.datum.index_name != 'BMRI.JK'
    ).mark_bar().encode(
        x=alt.X('index_name:N', title='Index'),
        y=alt.Y('price:Q', title='Price', scale=alt.Scale(domain=[-symmetric_range, symmetric_range])),
        color=alt.value("gray")  # The color for other indices
    ).properties(width=600, height=200)

    # Combine the charts
    final_chart = target_index_chart + other_indices_chart
    final_chart





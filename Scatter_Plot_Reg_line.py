import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

results_dir = '../Results/'
data_dir = '../Data/Ext_Data/'

temp_path = results_dir + 'volumes_prepped.csv'
ext_data_us_eu_path = data_dir + "Us_Eu.csv"
ext_data_oil_path = data_dir + "Oil.csv"
ext_data_health_path = data_dir + "Health_Index.csv"


df_SPA_vol = pd.read_csv(temp_path, sep='|')
df_euro_us = pd.read_csv(ext_data_us_eu_path, sep=',')
df_OIL = pd.read_csv(ext_data_oil_path, sep=';')
df_health_index = pd.read_csv(ext_data_health_path, sep=';')

print(df_SPA_vol)
print(df_health_index)
print(df_OIL)
print(df_euro_us)

df_test_health = pd.merge(df_SPA_vol, df_health_index, how='left', on=['Date']).sort_values('Date', ascending=False)

df_test_oil = pd.merge(df_SPA_vol, df_OIL, how='left', on=['Date']).sort_values('Date', ascending=False)

import plotly.express as px

fig = px.scatter(df_test_oil, x="SPA REINE", y="Brent Oil Price", trendline="ols")
fig.update_layout(
    title=go.layout.Title(
        text="Spa Reine Volume vs Brent Oil Price",
        xref="paper",
        x=0
    ),
    xaxis=go.layout.XAxis(
        title=go.layout.xaxis.Title(
            text="Spa Reine Vol",
            font=dict(
                family="Courier New, monospace",
                size=18,
                color="#7f7f7f"
            )
        )
    ),
    yaxis=go.layout.YAxis(
        title=go.layout.yaxis.Title(
            text="Brent Oil Price",
            font=dict(
                family="Courier New, monospace",
                size=18,
                color="#7f7f7f"
            )
        )
    )
)

fig.show()
# -------------------------------------------------------------------------------------------------------------------- #
#                                                                                                                      #
#                                                   Deloitte Belgium                                                   #
#                                   Gateway building Luchthaven Nationaal 1J 1930 Zaventem                              #
#                                                                                                                      #
# -------------------------------------------------------------------------------------------------------------------- #
#
# Author list (Alphabetical Order):
#    Name                                       Username                                     Email
#    Ugo Leonard                                uleonard                                     uleonard@deloitte.com
# -------------------------------------------------------------------------------------------------------------------- #
# ###                                               Program Description                                            ### #
# -------------------------------------------------------------------------------------------------------------------- #
# To be written
# -------------------------------------------------------------------------------------------------------------------- #
#                                             Parameters & Libraries                                                   #
# -------------------------------------------------------------------------------------------------------------------- #
import plotly.express as px
import numpy as np
import datetime
import statsmodels.api as sm

import pandas as pd
import matplotlib.pyplot as plt

data_dir = '../Data/'
results_dir = '../Results/'

volumes_path = data_dir + "Spadel.xlsx"
vol_out_path = results_dir + 'volumes_prepped.csv'
# -------------------------------------------------------------------------------------------------------------------- #
#                                             Functions                                                                #
# -------------------------------------------------------------------------------------------------------------------- #

# -------------------------------------------------------------------------------------------------------------------- #
#                                                 Main                                                                 #
# -------------------------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':
    print('Starting')

    # Import data
    file = pd.read_excel(volumes_path, None)
    df_volumes_in = pd.DataFrame()
    for key in file.keys():
        df = file[key]
        df = df[['Material Code & Description', 'Date', 'Sales Volume (L)','Sales Value Gross incl. Taxes (Loc. Cur.)','Sub Brand Code & Description']]
        df_volumes_in = df_volumes_in.append(df)

    df_volumes = df_volumes_in.copy()

    df_volumes['Label'] = df_volumes['Material Code & Description'].str.split(r'\)\s').str[1].str.replace(r'\s+', ' ')
    df_volumes['SKU'] = df_volumes['Material Code & Description'].str.extract(r'\((.*)\)')
    df_volumes['Brand'] = df_volumes['Sub Brand Code & Description'].str.split(r'\)\s').str[1].str.replace(r'\s+', ' ')

    df_volumes = df_volumes.rename(columns={'Sales Volume (L)': 'Volume',
                                            'Sales Value Gross incl. Taxes (Loc. Cur.)': 'GTS_Local'})
    df_volumes['Date'] = pd.to_datetime(df_volumes['Date'])

    df_volumes = df_volumes.loc[(~df_volumes['SKU'].isnull()),
                                ['Brand', 'SKU', 'Label', 'Date', 'Volume', 'GTS_Local']]

    df_volumes['Volume'] = df_volumes['Volume'].astype(float)

    df_volumes_M = df_volumes.groupby(['Brand', pd.Grouper(key='Date', freq='M')]).sum().reset_index()
    df_yearly = df_volumes_M.groupby(['Brand', pd.Grouper(key='Date', freq='A')])['Volume', 'GTS_Local'].sum().reset_index()

    df_volumes_M['Volatility'] = df_volumes_M.sort_values(['Brand', 'Date']).groupby('Brand')['Volume'].pct_change()
    df_std = df_volumes_M.groupby(['Brand', pd.Grouper(key='Date', freq='A')])['Volatility'].std().reset_index()
    df_std["Volatility"] = df_std["Volatility"].abs()
    df_std["Volatility"] *= 100

    df_summary = pd.merge(df_yearly, df_std, how='left', on=['Brand', 'Date']).sort_values('Volume', ascending=False)
    df_summary = df_summary[df_summary['Brand'] != 'SPA WATER MIX']
    df_summary["Year"] = df_summary["Date"].dt.year

    df_summary = df_summary[df_summary["Year"] < 2019]

    df_summary = df_summary.sort_values("Year")





    fig =  px.scatter(df_summary, x="Volume", y="Volatility", animation_frame="Year",
           size="GTS_Local", color="Brand", hover_name="Brand", size_max=55, range_x=[100,150000000], range_y=[0,100])

     #layout=go.Layout(
     #    title=go.layout.Title(
     #        text="Spadel Brand Demand Evolution 2008-2018 ",
     #        xref="paper",
     #        x=0
     #    ),
     #    xaxis=go.layout.XAxis(
     #        title=go.layout.xaxis.Title(
     #            text="Volume (Liters)",
     #            font=dict(
     #                family="Courier New, monospace",
     #                size=18,
     #                color="#7f7f7f"
     #            )
     #        )
     #    ),
     #    yaxis=go.layout.YAxis(
     #        title=go.layout.yaxis.Title(
     #            text="Volatility (%)",
     #            font=dict(
     #                family="Courier New, monospace",
     #                size=18,
     #                color="#7f7f7f"
     #            )
     #        )
     #    )
     #)

    fig.show()



# -------------------------------------------------------------------------------------------------------------------- #
#                                                 Export                                                               #
# -------------------------------------------------------------------------------------------------------------------- #


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
import plotly
import json
import urllib
import urllib.request
import pandas as pd
from tqdm import tqdm

pd.set_option('display.width', 320)
pd.set_option('display.max_colwidth', -1)

# If want to run for a specific year, change:
# - results_dir
# - budget_path
# - in read_budget(): pd.to_datetime(df_budget_pivot['index'] + '-18', format='%b-%y')
# - in prepare_R(): (df_product['Date'] < pd.Timestamp('2019-01-01'))
# - in prepare_R(): comment (df_product['SKU_and_Label'] != '1425AG.SPA ORANGE SP 4*6X 500PET') &

data_dir = '../Data/'
results_dir = '../Results/SKU/MARKET/2018/'

volumes_path = data_dir + "Spadel.xlsx"
budget_path = data_dir + "Budgeted_Market2018.xlsx"
promo1416_path = data_dir + 'RetailPromotions1416.csv'
promo1719_path = data_dir + 'RetailPromotions1719.csv'

weather_temp_path = results_dir + 'weather_temp.csv'
temp_path = results_dir + 'volumes_temp.csv'

volumes_outpath = results_dir + 'volumes_prepped.csv'
weather_outpath = results_dir + 'weather.csv'
internal_outpath = results_dir + 'internal_factors.csv'
budget_outpath = results_dir + 'budget_crosscheck.csv'
# -------------------------------------------------------------------------------------------------------------------- #
#                                             Functions                                                                #
# -------------------------------------------------------------------------------------------------------------------- #


def read_in(volumes_path, test=True):
    # Import data
    if test == True:
        df_volumes = pd.read_csv(temp_path, sep='|')
        df_volumes['Date'] = pd.to_datetime(df_volumes['Date'])
        df_volumes['Volume'] = df_volumes['Volume'].astype(float)

    else:
        file = pd.read_excel(volumes_path, None)
        df_volumes_in = pd.DataFrame()
        for key in file.keys():
            df = file[key]
            df = df[['Generic Code & Description', 'Date', 'Sales Volume (L)', 'Sales Value Gross incl. Taxes (Loc. Cur.)',
                     'Sub Brand Code & Description', 'Sales Organization Distribution Channel']]
            df_volumes_in = df_volumes_in.append(df)

        df_volumes = df_volumes_in.copy()
        #df_volumes = df_volumes_in[df_volumes_in['Sales Organization Distribution Channel']=='HOME']

        df_volumes['Label'] = df_volumes['Generic Code & Description'].str.split(r'\)\s').str[1].str.replace(r'\s+', ' ')
        df_volumes['SKU'] = df_volumes['Generic Code & Description'].str.extract(r'\((.*)\)')
        df_volumes['Brand'] = df_volumes['Sub Brand Code & Description'].str.split(r'\)\s').str[1].str.replace(r'\s+', ' ')
        df_volumes['SKU_and_Label'] = df_volumes['SKU'] + '.' + df_volumes['Label']

        df_volumes = df_volumes.rename(columns={'Sales Volume (L)': 'Volume',
                                                'Sales Value Gross incl. Taxes (Loc. Cur.)': 'GTS_Local',
                                                'Sales Organization Distribution Channel': 'Market'})

        df_volumes = df_volumes.loc[~df_volumes['SKU'].isnull()]
        df_volumes['Date'] = pd.to_datetime(df_volumes['Date'])
        df_volumes['Volume'] = df_volumes['Volume'].astype(float)

        df_volumes.to_csv(temp_path, sep='|', header=True, index=False)
    print('Read data in')

    return df_volumes[['Brand', 'SKU_and_Label', 'Date', 'Volume', 'GTS_Local', 'Market']]


def read_external_weather(test=True):
    if test==True:
        df_weather = pd.read_csv(weather_temp_path, sep='|')
    else:
        start_dates = [pd.date_range('2008-07-01', '2019-09-01', freq='1M') - pd.offsets.MonthBegin(1)]
        end_dates = [pd.date_range('2008-08-01', '2019-10-01', freq='1M') - pd.offsets.MonthEnd(1)]

        start_dates_str = [item.strftime("%Y-%m-%d") for sublist in start_dates for item in sublist]
        end_dates_str = [item.strftime("%Y-%m-%d") for sublist in end_dates for item in sublist]

        df_weather = pd.DataFrame(columns=['Date', 'Max_Temp_C', 'Min_Temp_C', 'Avg_Temp_C'])
        url_start = 'http://api.worldweatheronline.com/premium/v1/past-weather.ashx?key=6ab63dcb65a04ebf928104352190209&q=Brussels&format=json&date=2008-07-01&enddate=2008-07-31&tp=24'

        for i in tqdm(range(0, len(start_dates_str))):
            start = start_dates_str[i]
            end = end_dates_str[i]
            url_temp = url_start.replace('2008-07-01', start)
            url = url_temp.replace('2008-07-31', end)

            with urllib.request.urlopen(url) as url_open:
                data = json.loads(url_open.read().decode())
                for j in data['data']['weather']:
                    df_temp = pd.DataFrame({'Date': j['date'],
                                            'Max_Temp_C': np.float(j['maxtempC']),
                                            'Min_Temp_C': np.float(j['mintempC']),
                                            'Avg_Temp_C': np.float(j['avgtempC'])}, index=[0])
                    df_weather = df_weather.append(df_temp)

        df_weather.to_csv(weather_temp_path, sep='|', header=True, index=False)

    df_weather['Date'] = pd.to_datetime(df_weather['Date'], format="%Y-%m-%d")
    df_weather = df_weather.sort_values('Date')
    # Flag days with more than 25 degrees
    df_weather.loc[df_weather['Max_Temp_C'] >= 25, 'TwentyFive'] = 1
    df_weather['HotDays'] = df_weather['TwentyFive'].groupby((df_weather['TwentyFive'] != df_weather['TwentyFive'].shift()).cumsum()).cumcount()+1
    df_weather.loc[df_weather['HotDays'] > 2, 'Canicule'] = 1
    df_weather['Canicule'] = df_weather['Canicule'].fillna(0)

    # Flag days with more than 30 degrees
    df_weather.loc[df_weather['Max_Temp_C'] >= 30, 'VHotDays'] = 1
    df_weather['VHotDays'] = df_weather['VHotDays'].fillna(0)
    df_weather_gb = df_weather.groupby([pd.Grouper(key='Date', freq='MS')]).agg(
        {'Canicule': 'sum', 'VHotDays': 'sum', 'Avg_Temp_C': 'mean'})

    df_weather_gb['Canicule'] = np.sqrt(df_weather_gb['Canicule'])
    df_weather_gb.to_csv(weather_outpath, sep='|', header=True, index=True)
    print('Weather')
    return df_weather_gb


def read_promos(promo1416_path, promo1719_path, brands_tokeep):
    def correct_frequency(df_product):
        months_range = pd.date_range(df_product['Date'].min(), df_product['Date'].max(), freq='M')
        df_time = pd.DataFrame(index=months_range)
        df_months = pd.merge(df_time, df_product, how='left', left_index=True, right_on='Date')
        df_months['SKU_and_Label'] = df_months['SKU_and_Label'].fillna(df_months['SKU_and_Label'].unique()[0])
        cols = ['Promo']
        df_months[cols] = df_months[cols].fillna(0)
        return df_months

    # Promotions from 2014 to 2016
    df_promo14 = pd.read_csv(promo1416_path, sep='|', encoding='latin-1')

    months = {'March': 'Mar', 'April': 'Apr', 'June': 'Jun', 'July': 'Jul', 'Sept': 'Sep'}
    df_promo14['Maand Name'] = df_promo14['Maand Name'].map(months).fillna(df_promo14['Maand Name'])
    df_promo14['Date'] = pd.to_datetime(df_promo14['Maand Name'] + '-' + df_promo14['Year'].astype(str), format='%b-%Y')

    brands = {'SOURCE_DE_BRU': 'BRU'}
    df_promo14['Segment'] = df_promo14['Brand'].map(brands).fillna(df_promo14['Brand'])
    product = {'SOURCE_DE_BRU': 'BRU'}
    df_promo14['Type'] = df_promo14['Variant'].map(product).fillna(df_promo14['Variant'])
    df_promo14['Format'] = df_promo14['Pack Size'].str.extract(r'(\d+)').astype(int).astype(str)

    df_promo14 = df_promo14.loc[((df_promo14['Date'] > pd.Timestamp('2013-12-31')) &
                                 (df_promo14['Date'] < pd.Timestamp('2017-01-01'))),
                                 ['Segment', 'Type', 'Format', 'Date', 'Concept name']].drop_duplicates()
    df_promo14gb = df_promo14.groupby(['Segment', 'Type', 'Format', pd.Grouper(key='Date', freq='M')]).size().reset_index(name='Promo')

    df_promo17 = pd.read_csv(promo1719_path, sep='|', encoding='latin-1')
    df_promo17 = df_promo17.loc[(df_promo17['Segment'].str.contains(r'BRU|SPA')) &
                            (~df_promo17['Promo Week'].isnull()),
                            ['Segment', 'Type', 'Full name', 'Promo Week', 'Mechanism']].drop_duplicates()

    df_promo17['Date'] = pd.to_datetime(df_promo17['Promo Week'].str.split('W ').str[1] + "0", format="%Y %U%w")
    df_promo17['Format'] = df_promo17['Full name'].str.extract(r'(\d+)').astype(int).astype(str)
    df_promo17gb = df_promo17.groupby(['Segment', 'Type', 'Format', pd.Grouper(key='Date', freq='M')]).size().reset_index(name='Promo')

    df_promotions = df_promo14gb.append(df_promo17gb, sort='False')

    df_promotions.loc[((df_promotions['Segment'] == 'BRU') &
                    (df_promotions['Format'] == '125')), 'SKU_and_Label'] = brands_tokeep[0]

    df_promotions.loc[((df_promotions['Segment'] == 'SPA') &
                    (df_promotions['Type'] == 'TOUCH OF') &
                    (df_promotions['Format'] == '50')), 'SKU_and_Label'] = brands_tokeep[1]

    df_promotions.loc[((df_promotions['Segment'] == 'SPA') &
                    (df_promotions['Type'] == 'REINE') &
                    (df_promotions['Format'] == '150')), 'SKU_and_Label'] = brands_tokeep[2]

    df_promotions.loc[((df_promotions['Segment'] == 'SPA') &
                    (df_promotions['Type'] == 'CARBONATED') &
                    (df_promotions['Format'] == '50')), 'SKU_and_Label'] = brands_tokeep[3]

    df_promotions.loc[((df_promotions['Segment'] == 'SPA') &
                    (df_promotions['Type'] == 'INTENSE') &
                    (df_promotions['Format'] == '50')), 'SKU_and_Label'] = brands_tokeep[4]

    df_promotions = df_promotions[~(df_promotions['SKU_and_Label'].isnull())]
    df_promotions = df_promotions[['Date', 'SKU_and_Label', 'Promo']]
    df_promotions['SKU_and_Label'] = df_promotions['SKU_and_Label'].str.replace(r'[^a-zA-Z0-9]', '.')

    df_promo_full = df_promotions.groupby('SKU_and_Label').apply(correct_frequency).reset_index(drop=True)

    df_promo_full.to_csv(internal_outpath, header=True, index=False, sep='|')
    return None


def read_budget(budget_path, brands_tokeep):
    df_budget = pd.read_excel(budget_path)
    #df_budget = df_budget[df_budget['Version'] == 'Budget']
    df_budget = df_budget[df_budget['Version'] != 'Others']

    brands_tokeep_short = [string.split('.')[0] for string in brands_tokeep]

    df_brands = pd.DataFrame({'Product': brands_tokeep_short,
                              'SKU_and_Label': brands_tokeep})

    df_budget = pd.merge(df_budget, df_brands, how='inner', on='Product')
    df_budget['Version'] = df_budget['Version'].str.upper()
    df_budget['SKU_and_Label'] = df_budget['SKU_and_Label'] + '///' + df_budget['Version']
    df_budget_pivot = df_budget.T
    df_budget_pivot.columns = df_budget_pivot.iloc[-1]
    df_budget_pivot = df_budget_pivot.reset_index()
    df_budget_pivot = df_budget_pivot.loc[~(df_budget_pivot['index'].isin(['Product', 'Version', 'Full Year', 'SKU_and_Label']))]

    months = {'Mrt': 'Mar', 'Mei': 'May', 'Okt': 'Oct'}
    df_budget_pivot['index'] = df_budget_pivot['index'].map(months).fillna(df_budget_pivot['index'])
    df_budget_pivot['Date'] = pd.to_datetime(df_budget_pivot['index'] + '-18', format='%b-%y')

    brands_tokeep_update = ['1038AG.SPA INTENSE 4*6X 500PET///HOME', '1038AG.SPA INTENSE 4*6X 500PET///OUT OF HOME',
                            '1038AG.SPA INTENSE 4*6X 500PET///OTHERS', '3008AG.BRU 6X1250PET///HOME',
                            '3008AG.BRU 6X1250PET///OUT OF HOME', '3008AG.BRU 6X1250PET///OTHERS',
                            '1098AG.SPA REINE 6X1500PET///HOME', '1098AG.SPA REINE 6X1500PET///OUT OF HOME',
                            '1098AG.SPA REINE 6X1500PET///OTHERS', '1425AG.SPA ORANGE SP 4*6X 500PET///HOME',
                            '1425AG.SPA ORANGE SP 4*6X 500PET///OUT OF HOME', '1425AG.SPA ORANGE SP 4*6X 500PET///OTHERS',
                            '1366AG.SPA TO LEMON 4*6X 500PET///HOME', '1366AG.SPA TO LEMON 4*6X 500PET///OUT OF HOME',
                            '1366AG.SPA TO LEMON 4*6X 500PET///OTHERS']
    cols = ['Date'] + brands_tokeep_update
    df_budget_pivot[cols].to_csv(budget_outpath, header=True, index=False, sep='|')
    print('Budget')
    return None


def prepare_bubble(df_volumes):
    df_volumes_M = df_volumes.groupby(['Brand', 'SKU_and_Label', pd.Grouper(key='Date', freq='M')]).sum().reset_index()
    df_yearly = df_volumes_M.groupby(['Brand', 'SKU_and_Label', pd.Grouper(key='Date', freq='A')])['Volume', 'GTS_Local'].sum().reset_index()

    df_volumes_M['Volatility'] = df_volumes_M.sort_values(['Brand', 'SKU_and_Label', 'Date']).groupby('SKU_and_Label')['Volume'].pct_change()
    df_std = df_volumes_M.groupby(['Brand', 'SKU_and_Label', pd.Grouper(key='Date', freq='A')])['Volatility'].std().reset_index()
    df_std["Volatility"] = df_std["Volatility"].abs()
    df_std["Volatility"] *= 100

    df_summary = pd.merge(df_yearly, df_std, how='left', on=['Brand', 'SKU_and_Label', 'Date']).sort_values('Volume', ascending=False)

    df_summary["Year"] = df_summary["Date"].dt.year
    df_summary = df_summary[df_summary["Year"] < 2019]
    df_summary = df_summary.sort_values("Year")

    df_summary.loc[df_summary['GTS_Local'] > 0, 'Gross_Trade_Sales'] = df_summary['GTS_Local']
    df_summary['Gross_Trade_Sales'] = df_summary['Gross_Trade_Sales'].fillna(1)

    fig = px.scatter(df_summary, x="Volume", y="Volatility", animation_frame="Year", animation_group='SKU_and_Label',
          size="Gross_Trade_Sales", color="Brand", hover_name="SKU_and_Label", size_max=55,
          range_x=[100, 25000000], range_y=[0, 150])
    plotly.offline.plot(fig, filename=results_dir + 'Spadel_PoC_Brand_evolution.html')
    print('Plot')
    return None


def prepare_R(df_volumes, brands_tokeep, volumes_outpath):
    df_product = df_volumes.groupby(['SKU_and_Label', 'Market', pd.Grouper(key='Date', freq='MS')])['Volume'].sum().reset_index()
    df_product_size = (df_product.groupby(['SKU_and_Label', 'Market']).size() > 36).reset_index()
    product_list = df_product_size.loc[df_product_size[0] == True, 'SKU_and_Label'].tolist()
    df_product = df_product[(df_product['SKU_and_Label'].isin(brands_tokeep)) &
                            (df_product['SKU_and_Label'] != '1425AG.SPA ORANGE SP 4*6X 500PET') &
                            (df_product['SKU_and_Label'].isin(product_list)) &
                            (df_product['Date'] < pd.Timestamp('2019-01-01'))]

    df_product['SKU_and_Label'] = df_product['SKU_and_Label'] + '///' + df_product['Market']

    df_product = df_product.pivot(index='Date', columns='SKU_and_Label', values='Volume')
    df_product.to_csv(volumes_outpath, header=True, index=True, sep='|')
    print('Subset for R')
    return None

# -------------------------------------------------------------------------------------------------------------------- #
#                                                 Main                                                                 #
# -------------------------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':
    print('Starting')
    #'1366AG.SPA TO LEMON 4*6X 500PET'
    #'1365AG.SPA TO MINT 4*6X 500PET'

    brands_tokeep = ['3008AG.BRU 6X1250PET', '1366AG.SPA TO LEMON 4*6X 500PET',
                     '1098AG.SPA REINE 6X1500PET', '1425AG.SPA ORANGE SP 4*6X 500PET',
                     '1038AG.SPA INTENSE 4*6X 500PET']

    # Read the data in
    df_volumes = read_in(volumes_path, test=True)
    #df_weather = read_external_weather(test=False)
    #df_promo = read_promos(promo1416_path, promo1719_path, brands_tokeep)
    read_budget(budget_path, brands_tokeep)

    # Create the bubble plot
    #prepare_bubble(df_volumes)

    # Create sample for R
    df_output = prepare_R(df_volumes, brands_tokeep, volumes_outpath)

# -------------------------------------------------------------------------------------------------------------------- #
#                                                 Export                                                               #
# -------------------------------------------------------------------------------------------------------------------- #
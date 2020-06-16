import plotly.express as px
import numpy as np
import datetime
import plotly
import json
import urllib
import urllib.request
import pandas as pd
import plotly.graph_objects as go


data_dir = '../Data/'
results_dir = '../Results/'

volumes_path = data_dir + "Spadel.xlsx"

volumes_outpath_1 = results_dir + "3008AG.BRU 6X1250PET PROMOS.csv"
volumes_outpath_2 = results_dir + "1365AG.SPA TO MINT 4*6X 500PET PROMOS.csv"
volumes_outpath_3 = results_dir + "1098AG.SPA REINE 6X1500PET PROMOS.csv"
volumes_outpath_4 = results_dir + "1038AG.SPA INTENSE 4*6X 500PET PROMOS.csv"
volumes_outpath_5 = results_dir + "1102AG.SPA REINE 4*6X 500PET PROMOS.csv"

temp_path = results_dir + "Discounts.csv"

# -------------------------------------------------------------------------------------------------------------------- #
#                                             Functions                                                                #
# -------------------------------------------------------------------------------------------------------------------- #




# -------------------------------------------------------------------------------------------------------------------- #
#                                                 Main                                                                 #
# -------------------------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':

    # Data Preparation -> taken from Data Preparation.py

    file = pd.read_excel(volumes_path, None)
    df_discounts_in = pd.DataFrame()
    for key in file.keys():
        df = file[key]
        df = df[['Generic Code & Description', 'Date', 'Sales Value Gross incl. Taxes (Loc. Cur.)',
                 'Sub Brand Code & Description', 'Total On Invoice Discount Value (Loc. Cur.)',
                 'Total Permanent Discount Value On Invoice (Loc. Cur.)',
                 'Total Temporary Discount Value On Invoice (Loc. Cur.)', 'Flashed price packs (Loc. Cur.)',
                 'Free goods: discount (Loc. Cur.)', 'Sales Value Net On Invoice (Loc. Cur.)', 'Sales Volume (L)']]

    df_discounts_in = df_discounts_in.append(df)
    df_discounts = df_discounts_in.copy()
    df_discounts['Label'] = df_discounts['Generic Code & Description'].str.split(r'\)\s').str[1].str.replace(r'\s+',
                                                                                                             ' ')
    df_discounts['SKU'] = df_discounts['Generic Code & Description'].str.extract(r'\((.*)\)')
    df_discounts['Brand'] = df_discounts['Sub Brand Code & Description'].str.split(r'\)\s').str[1].str.replace(r'\s+',
                                                                                                               ' ')
    df_discounts['SKU_and_Label'] = df_discounts['SKU'] + '.' + df_discounts['Label']
    df_discounts = df_discounts.rename(columns={'Total On Invoice Discount Value (Loc. Cur.)': 'Total Discount Value',
                                                'Sales Value Gross incl. Taxes (Loc. Cur.)': 'GTS_Local',
                                                'Total Permanent Discount Value On Invoice (Loc. Cur.)': 'Total Permanent Discount Value',
                                                'Total Temporary Discount Value On Invoice (Loc. Cur.)': 'Total Temporary Discount Value',
                                                'Flashed price packs (Loc. Cur.)': 'Flashed Price',
                                                'Free goods: discount (Loc. Cur.)': 'Free goods',
                                                'Sales Value Net On Invoice (Loc. Cur.)': 'Net Sales',
                                                'Sales Volume (L)': 'Volume'})
    df_discounts = df_discounts.loc[~df_discounts['SKU'].isnull()]
    df_discounts['Date'] = pd.to_datetime(df_discounts['Date'])
    df_discounts['GTS_Local'] = df_discounts['GTS_Local'].astype(float)

    # Create columns w/ Base Price and % change

    df_discounts['Volume'] = df_discounts['Volume'].astype(float)
    df_discounts['Base_Price'] = df_discounts['GTS_Local'] / df_discounts['Volume']
    df_discounts['Total_Discount_Value_pct'] = df_discounts['Total Discount Value'].abs() / df_discounts['GTS_Local']
    df_discounts['Total_Permanent_Discount_Value_pct'] = df_discounts['Total Permanent Discount Value'].abs() / \
                                                         df_discounts['GTS_Local'].abs()
    df_discounts['Total_Temporary_Discount_Value_Pct'] = df_discounts['Total Temporary Discount Value'].abs() / \
                                                         df_discounts['GTS_Local'].abs()
    df_discounts['Flashed_Price_Packs_pct'] = df_discounts['Flashed Price'].abs() / df_discounts['GTS_Local'].abs()
    df_discounts['Free_goods_pct'] = df_discounts['Free goods'].abs() / df_discounts['GTS_Local'].abs()

    # Create R dataset

    brands_tokeep = ['1098AG.SPA REINE 6X1500PET', '3008AG.BRU 6X1250PET',
                     '1102AG.SPA REINE 4*6X 500PET', '1038AG.SPA INTENSE 4*6X 500PET', '1211AG.SPA CITRON 4*6X 500PET']

    df_product = df_discounts.groupby(['SKU_and_Label', pd.Grouper(key='Date', freq='MS')])[
        'Base_Price', 'Total_Discount_Value_pct', 'Total_Permanent_Discount_Value_pct',
        'Total_Temporary_Discount_Value_Pct', 'Flashed_Price_Packs_pct', 'Free_goods_pct'].sum().reset_index()
    df_product_size = (df_product.groupby('SKU_and_Label').size() > 36).reset_index()
    product_list = df_product_size.loc[df_product_size[0] == True, 'SKU_and_Label'].tolist()
    df_product = df_product[(df_product['SKU_and_Label'].isin(brands_tokeep)) &
                            (df_product['SKU_and_Label'].isin(product_list)) &
                            (df_product['Date'] < pd.Timestamp('2019-09-01'))]

    Brand_Bru = ['3008AG.BRU 6X1250PET']
    df_product_bru = df_product[df_product['SKU_and_Label'].isin(Brand_Bru)]

    Brand_SPA_Citron = ['1365AG.SPA TO MINT 4*6X 500PET']
    df_product_mint = df_product[df_product['SKU_and_Label'].isin(Brand_SPA_Citron)]

    Brand_SPA_Reine_1_5 = ['1098AG.SPA REINE 6X1500PET']
    df_product_reine_1_5 = df_product[df_product['SKU_and_Label'].isin(Brand_SPA_Reine_1_5)]

    Brand_Spa_Intense = ['1038AG.SPA INTENSE 4*6X 500PET']
    df_product_orange = df_product[df_product['SKU_and_Label'].isin(Brand_Spa_Intense)]

    Brand_SPA_Reine_5 = ['1102AG.SPA REINE 4*6X 500PET']
    df_product_reine_5 = df_product[df_product['SKU_and_Label'].isin(Brand_SPA_Reine_5)]



    df_product_bru.to_csv(volumes_outpath_1, header=True, index=True, sep='|')
    df_product_mint.to_csv(volumes_outpath_2, header=True, index=True, sep='|')
    df_product_reine_1_5.to_csv(volumes_outpath_3, header=True, index=True, sep='|')
    df_product_reine_5.to_csv(volumes_outpath_4, header=True, index=True, sep='|')
    df_product.to_csv(volumes_outpath_5, header=True, index=True, sep='|')

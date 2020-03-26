import pandas as pd
import numpy as np
import datetime

def dateProcess(start_date_dt, end_date_dt):
    diff = end_date_dt - start_date_dt
    return start_date_dt, diff

# 1. Determine and return the top 3 retailers (not stores) in terms of revenue for
# the requested time period

def dictTopRetailersRev(path_str, start_date_dt, end_date_dt):

    sales_data = pd.read_csv("/".join([path_str,'sales_data.csv']), delimiter=',')
    stores = pd.read_csv("/".join([path_str,'stores.csv']), delimiter=',')

    # Add retailer to sales_data
    sales_data_retailer = sales_data.drop(['product_id', 'sales'], axis=1) \
            .join(stores[['store_id', 'retailer_id']], on='store_id', how='left', lsuffix='_left', rsuffix='_right') \
            .drop(['store_id_left'], axis=1) \
            .rename(columns={"store_id_right":"store_id"})

    # Group by date and retailer
    df_rev = sales_data_retailer[['store_id','date','revenue','retailer_id']].groupby(['date','retailer_id']).agg({'revenue':np.sum}).unstack() # group by retailer first??
    df_rev.columns = list(df_rev.columns.levels[1])

    # Fill dict with required dates and amounts
    dict_rev = {}

    for retailer in df_rev.columns:
        dict_rev[retailer]=0.0 # Initialize dictionary

    def fillDict(row):
        for retailer in row.index:
            dict_rev[retailer]=dict_rev[retailer] + row[retailer]

    start_date, diff = dateProcess(start_date_dt, end_date_dt)

    for i in range(diff.days+1):
        date = start_date + datetime.timedelta(days=i)
        date_str = date.strftime('%Y-%m-%d')
        df_rev[df_rev.index==date_str].apply(fillDict, axis=1)

    # Sort dict
    dict_rev_sorted = {k: v for k, v in sorted(dict_rev.items(), key=lambda item: item[1], reverse=True)}

    return dict_rev_sorted

def topRetailersRev(path_str, start_date_dt, end_date_dt):
    dict_rev_sorted = dictTopRetailersRev(path_str, start_date_dt, end_date_dt)
    for idx, retailer in enumerate(dict_rev_sorted):
        print("retailer {} is rank {} with revenue {}".format(retailer, idx+1, "{0:,.2f}".format(dict_rev_sorted[retailer])))

# 2. Determine the top product family in terms of sales for the requested time
# period, and return its daily sales

def dictTopFamilySales(path_str, start_date_dt, end_date_dt):
    products = pd.read_csv("/".join([path_str,'products.csv']), delimiter=',')
    sales_data = pd.read_csv("/".join([path_str,'sales_data.csv']), delimiter=',')

    # Add family id to sales_data_family
    sales_data_family = sales_data.drop(['revenue', 'store_id'], axis=1) \
            .join(products[['product_id', 'family_id']], on='product_id', how='left', lsuffix='_left', rsuffix='_right') \
            .drop(['product_id_left', 'product_id_right'], axis=1)

    # Group by date and family_id
    sales_data_grouped = sales_data_family.groupby(['date','family_id']).agg({'sales':np.sum}).unstack()
    sales_data_grouped.columns = list(sales_data_grouped.columns.levels[1])

    # Fill dict with required dates and amounts
    dict_sales = {}

    start_date, diff = dateProcess(start_date_dt, end_date_dt)

    for family in sales_data_grouped.columns:
        dict_sales[str(family)]=0 # Initialize dictionary

    def fillDict(row):
        for family in row.index:
            dict_sales[str(family)]=dict_sales[str(family)] + row[family]

    for i in range(diff.days+1):
            date = start_date + datetime.timedelta(days=i)
            date_str = date.strftime('%Y-%m-%d')
            sales_data_grouped[sales_data_grouped.index==date_str].apply(fillDict, axis=1)

    dict_sales_sorted = {k: v for k, v in sorted(dict_sales.items(), key=lambda item: item[1], reverse=True)}

    top_family = list(dict_sales_sorted)[0]

    # Dict for sales by dates
    dict_top_sales = {}

    for i in range(diff.days+1):
        date = start_date + datetime.timedelta(days=i)
        date_str = date.strftime('%Y-%m-%d')
        sales_daily=sales_data_grouped[sales_data_grouped.index==date_str][int(top_family)][0]
        dict_top_sales[date_str]=sales_daily

    return dict_sales_sorted, dict_top_sales

def topFamilySales(path_str, start_date_dt, end_date_dt):
    dict_sales_sorted, dict_top_sales = dictTopFamilySales(path_str, start_date_dt, end_date_dt)
    top_family = list(dict_sales_sorted)[0]
    print("family {} has the highest sales".format(top_family))
    for date, sales in dict_top_sales.items():
        print("{}: sales={}".format(date, "{0:,.2f}".format(sales)))


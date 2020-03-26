import pandas as pd
import datetime

product_list = []
store_list = []
sales_list = []

def dateProcess(start_date_dt, end_date_dt):
    diff = end_date_dt - start_date_dt
    return start_date_dt, diff

class Product:
    product_dict = {}
    def __init__(self, product_id, name="", family="", price=None):
        self.product_id = product_id
        self.name = name
        self.family = family
        self.price = price
        Product.product_dict[product_id] = price
    @classmethod
    def find_price(cls, product_id):
        return Product.product_dict[product_id]

class Store:
    def __init__(self, store_id, name, retailer):
        self.store_id = store_id
        self.name = name
        self.retailer = retailer

class Sales:
    sales_dict = {}
    def __init__(self, date, product_id, store_id, sales_amount):
        self.date = date
        self.product_id = product_id
        self.store_id = store_id
        self.sales_amount = sales_amount
        Sales.sales_dict[date, store_id, product_id] = sales_amount
    @classmethod
    def find_amount(cls, date, store_id, product_id):
        return Sales.sales_dict[date, store_id, product_id]

def init(path_str):
    product_list.clear()
    store_list.clear()
    sales_list.clear()
    products = pd.read_csv("/".join([path_str,'products.csv']), delimiter=',')
    stores = pd.read_csv("/".join([path_str,'stores.csv']), delimiter=',')
    sales_data = pd.read_csv("/".join([path_str,'sales_data.csv']), delimiter=',')
    products.apply(createProductFromDF, axis=1)
    stores.apply(createStoreFromDF, axis=1)
    print('fetching sales...')
    sales_data.apply(createSalesFromDF, axis=1)
    print('done fetching sales...')

def createProductFromDF(row):
    product = Product(row['product_id'], row['product_name'], row['family_id'], row['price'])
    product_list.append(product)

def createStoreFromDF(row):
    store = Store(row['store_id'], row['store_name'], row['retailer_id'])
    store_list.append(store)

def createSalesFromDF(row):
    sales = Sales(row['date'], row['product_id'], row['store_id'], row['sales'])
    sales_list.append(sales)

def salesFromStores(start_date_dt, end_date_dt, stores, sales_list):
    product_store_list = []
    start_date_dt, diff = dateProcess(start_date_dt, end_date_dt)
    for i in range(diff.days+1):
        date = start_date_dt + datetime.timedelta(days=i)
        date_str = date.strftime('%Y-%m-%d')
        for sales in sales_list:
            if(sales.store_id in stores and sales.date == date_str):
                product_store_list.append([sales.date, sales.store_id, sales.product_id])
    return product_store_list

def salesFromProducts(start_date_dt, end_date_dt, products, sales_list):
    sales_products_list = []
    start_date, diff = dateProcess(start_date_dt, end_date_dt)
    for i in range(diff.days+1):
        date = start_date + datetime.timedelta(days=i)
        date_str = date.strftime('%Y-%m-%d')
        for sales in sales_list:
            if(sales.product_id in products and sales.date == date_str):
                sales_products_list.append([sales.date, sales.store_id, sales.product_id])
    return sales_products_list

def dailySalesFromFamily(start_date_dt, end_date_dt, family, sales_list):
    dict_daily = {}
    start_date, diff = dateProcess(start_date_dt, end_date_dt)
    products = [x.product_id for x in product_list if x.family == family]
    for i in range(diff.days+1):
        date = start_date + datetime.timedelta(days=i)
        date_str = date.strftime('%Y-%m-%d')
        dict_daily[date_str] = 0
        for sales in sales_list:
            if(sales.product_id in products and sales.date == date_str):
                dict_daily[date_str] = dict_daily[date_str] + sales.sales_amount
    return dict_daily

def topRetailersRev(path_str, start_date_dt, end_date_dt):
    init(path_str)
    retailers = list(set([x.retailer for x in store_list]))
    dict_rev = {}
    for retailer in retailers:
        print('retrieving sales for retailer {}'.format(retailer))
        dict_rev[retailer]=0.0
        stores = [x.store_id for x in store_list if x.retailer == retailer]
        sales_retailer = salesFromStores(start_date_dt, end_date_dt, stores, sales_list)
        for date, store_id, product_id in sales_retailer:
             sales_amount = Sales.find_amount(date, store_id, product_id)
             price = Product.find_price(product_id)
             dict_rev[retailer] = dict_rev[retailer] + (sales_amount * price)
    dict_rev_sorted = {k: v for k, v in sorted(dict_rev.items(), key=lambda item: item[1], reverse=True)}
    for idx, retailer in enumerate(dict_rev_sorted):
        print("retailer {} is rank {} with revenue {}".format(retailer, idx+1, "{0:,.2f}".format(dict_rev_sorted[retailer])))

def topFamilySales(path_str, start_date_dt, end_date_dt):
    init(path_str)
    families = list(set([x.family for x in product_list]))
    dict_sales = {}
    for family in families:
        print('retrieving sales for family {}'.format(family))
        dict_sales[family]=0
        products = [x.product_id for x in product_list if x.family == family]
        sales_family = salesFromProducts(start_date_dt, end_date_dt, products, sales_list)
        for date, store_id, product_id in sales_family:
             sales_amount = Sales.find_amount(date, store_id, product_id)
             dict_sales[family] = dict_sales[family] + sales_amount
        dict_sales_sorted = {k: v for k, v in sorted(dict_sales.items(), key=lambda item: item[1], reverse=True)}
    top_family = list(dict_sales_sorted)[0]
    print("family {} has the highest sales".format(top_family))
    top_daily_sales = dailySalesFromFamily(start_date_dt, end_date_dt, top_family, sales_list)
    for date, amount in top_daily_sales.items():
        print("{}: sales={}".format(date, "{0:,.2f}".format(amount)))
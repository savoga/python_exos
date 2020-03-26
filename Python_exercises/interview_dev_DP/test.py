import unittest
import pandas as pd
import data_analysis_v2 as da
import datetime

ds = datetime.date(2000, 1, 1)
de = datetime.date(2000, 1, 1)
PATH_TEST = '/home/savoga/Bureau/DataPred/data_test'

def simpleDF():
    stores = {'store_id': [0, 1, 2, 3],
              'store_name': ['store_1', 'store_2', 'store_3', 'store_4'],
              'retailer_id': [0, 1, 0, 1]}
    stores_df = pd.DataFrame(data=stores)
    stores_df.to_csv(PATH_TEST + '/stores.csv')
    products = {'product_id': [0, 1, 2, 3],
                'product_name': ['product_0', 'product_1', 'product_2', 'product_4'],
                'family_id': [0, 1, 0, 1],
                'price': [0, 0, 0, 0]}
    products_df = pd.DataFrame(data=products)
    products_df.to_csv(PATH_TEST + '/products.csv')
    sales = {'product_id': [0, 1],
             'store_id': [0, 1],
             'date': ['2000-01-01', '2000-01-01'],
             'sales': [100, 50],
             'revenue': [20, 0]}
    sales_df = pd.DataFrame(data=sales)
    sales_df.to_csv(PATH_TEST + '/sales_data.csv')
    return stores_df, products_df, sales_df

class SalesTest(unittest.TestCase):

    def testTopRetailersRev(self):
        stores_df, _, sales_df = simpleDF()
        dict_test = {0:20, 1:0}
        self.assertEqual(da.dictTopRetailersRev(PATH_TEST, ds, de), dict_test)

    def testTopFamilySales(self):
        _, products_df, sales_df = simpleDF()
        dict_test_1 = {'0':100, '1':50}
        dict_test_2 = {'2000-01-01':100}
        self.assertEqual(da.dictTopFamilySales(PATH_TEST, ds, de)[0], dict_test_1)
        self.assertEqual(da.dictTopFamilySales(PATH_TEST, ds, de)[1], dict_test_2)

def run_tests():
    test_suite = unittest.makeSuite(SalesTest)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(test_suite)

if __name__ == '__main__':
    run_tests()

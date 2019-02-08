import pandas as pd
import pylab as pl
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori

class ParseRetail:
    'Class to read transactions csv and return support, confidence and lift dataframes'
    
    def __init__(self, csv_file):
        print('Reading Files...')
        self.DF = pd.read_csv(csv_file)
        if 'Value' not in self.DF.columns:
            print('Calculating total price of bills...')
            self.DF['Value'] = self.DF.apply(lambda x: x['UnitPrice']*x['Quantity'], axis = 1)
            
        #Create lists of all items present in each individual invoice
        list_sku = self.DF.groupby('InvoiceNo')['Description'].apply(list)
        self.list_sku = list_sku.values.tolist()
        
        sum_value = self.DF.groupby(['InvoiceNo']).sum()['Value']
        self.fulfilled_DF = self.DF[self.DF['Value'] > 0]
        self.sum_value_fulfilled = sum_value[sum_value > 0]
        self.support_DF = self.return_support()
        self.support_1_length = self.return_support_n_length('=', 1)
        self.support_more_length = self.return_support_n_length('>', 1)
        self.confidence_DF = self.return_confidence()
        self.lift_DF = self.return_lift()
       
    def return_support(self):
        print('Calculating support values...')
        te = TransactionEncoder()
        te_array = te.fit(self.list_sku).transform(self.list_sku)
        df_list = pd.DataFrame(te_array, columns = te.columns_)
        support = apriori(df_list, min_support = 0.05, use_colnames = True)
        support = support.sort_values(['support'], ascending = False)
        support['length'] = support.apply(lambda x: len(x['itemsets']), axis = 1)
        return support

#Function to calculate support for invoices containing n# items
    def return_support_n_length(self, comparison = '=', n = 1):
        if comparison == '=':
            return self.support_DF[self.support_DF.length == n]
        elif comparison == '>':
            return self.support_DF[self.support_DF.length > n]
        elif comparison == '<':
            return self.support_DF[self.support_DF.length < n]
        else:
            print('Error: Invalid parameters in call to \'return_support_n_length()\'')
	
#Function to calculate confidence
    def return_confidence(self):
        print('Calculating confidence values...')
        itemsets_list = []
        confidence_list = []
        dict_confidence = {}
        
        for ii in range(len(self.support_1_length)):
            for jj in range(len(self.support_more_length)):
                if self.support_1_length.iloc[ii]['itemsets'].issubset(self.support_more_length.iloc[jj]['itemsets']):
                    itemsets_list.append(self.support_more_length.iloc[jj]['itemsets'])
                    antecedent = list(self.support_1_length.iloc[ii]['itemsets'])[0]
                    confidence = self.support_more_length.iloc[jj]['support']/self.support_1_length.iloc[ii]['support']
                    confidence_list.append(confidence)
        dict_confidence = {'itemsets': itemsets_list, 'confidence': confidence_list}
        confidence_DF = pd.DataFrame(dict_confidence)
        confidence_DF = confidence_DF.sort_values(['confidence'], ascending = False)
        return confidence_DF

#Function to calculate lift
    def return_lift(self):
        print('Calculating lift values...')
        lift_items = []
        lift_values = []
        lift_support = []
        core_support = []
        dict_lift = {}

        for jj in range(len(self.support_more_length)):
            support_lis = []
            for ii in range(len(self.support_1_length)):
                if self.support_more_length.iloc[jj]['itemsets'].issuperset(self.support_1_length.iloc[ii]['itemsets']):
                    support_lis.append(self.support_1_length.iloc[ii]['support'])
            lift_items.append(self.support_more_length.iloc[jj]['itemsets'])
            lift = self.support_more_length.iloc[jj]['support']
            for ii in support_lis:
                lift = lift/ii
            lift_values.append(lift)
            lift_support.append(support_lis)
            core_support.append(self.support_more_length.iloc[jj]['support'])

        dict_lift = {'itemsets': lift_items, 'lift': lift_values, 'supports': lift_support, 'coresupport': core_support}
        lift_DF = pd.DataFrame(dict_lift)
        lift_DF = lift_DF.sort_values(['lift'], ascending = False)

        return lift_DF		

if __name__ == '__main__':
    retailObj = ParseRetail('retail.csv')
    print('\n Support Dataframe head:\n', retailObj.support_DF.head())
    print('\n Confidence Dataframe head:\n', retailObj.confidence_DF.head())
    print('\n Lift Dataframe head:\n', retailObj.lift_DF.head())

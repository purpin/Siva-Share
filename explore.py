#Code to analyse the online retail data
import dask
import pandas as pd 
import pylab as pl 
import seaborn as sbn 
import dask.dataframe as dd 
from dask.delayed import delayed
import numpy as np
import time
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
start_time = time.time()

print "Reading files..."
excel_file = '../data/organic_world.csv'
# parts = dask.delayed(pd.read_excel)(excel_file, sheet_name=0)
# df = dd.from_delayed(parts)

df = pd.read_csv(excel_file)
df = df[:-1]
#Function to combine unit prices and quantity

value_lis = [float(ii) for ii in df.Total.values]

def func(number1,number2):
    return number1 * number2

print ("Creating price values...")
# df['Value'] = df.apply(lambda x: func(x['UnitPrice'], x['Quantity']), axis=1)
df['Value'] = value_lis

#Create a dataframe with the mean and standard deviations of the bills
#Groupby invoice number to get the mean and standard deviation of value of each bill - (both in unit price and value of items)

sum_value = df.groupby(['InvoiceNo']).sum()['Value']

sum_value_ful = sum_value[sum_value > 0]
sum_value_can = sum_value[sum_value < 0]

# pl.hist(sum_value_ful, range(0,5001, 100), alpha = 0.4)
# pl.figure()
# pl.hist(abs(sum_value_can), range(0,5001, 100), alpha = 0.4)
# pl.show()

#Analyse the cancelled orders in full
can_df = df[df['Value'] < 0]
can_df = can_df[can_df['Value'] > -70000]
cana, canb = np.histogram(abs(can_df['UnitPrice']), range(0,100,2))

ful_df = df[df['Value'] > 0]
ful_df = ful_df[ful_df['Value'] < 70000]
fula, fulb = np.histogram(ful_df['UnitPrice'], range(0,100,2))

#Convert the items present in each invoice into one-hot encoding
list_sku = df.groupby('InvoiceNo')['Description'].apply(list)
list_sku = list_sku.values.tolist()

def return_support(list_sku  = list_sku):
	te = TransactionEncoder()
	te_ary = te.fit(list_sku).transform(list_sku)
	df_list = pd.DataFrame(te_ary, columns=te.columns_)
	support = apriori(df_list, min_support=0.05, use_colnames=True)
	return support

support_df = return_support(list_sku)
support_df = support_df.sort_values(['support'], ascending = False)

#Search for an element in lists of length > 1
support_df['length'] = support_df.apply(lambda x: len(x['itemsets']), axis = 1)

support_1_length = support_df[support_df.length == 1]
support_more_length = support_df[support_df.length > 1]

dict_confidence = {}
itemsets_list = []
confidence_list = []

for ii in range(len(support_1_length)):
	for jj in range(len(support_more_length)):
		if support_1_length.iloc[ii]['itemsets'].issubset(support_more_length.iloc[jj]['itemsets']):
			itemsets_list.append(support_more_length.iloc[jj]['itemsets'])
			antecedent = list(support_1_length.iloc[ii]['itemsets'])[0]
			confidence = support_more_length.iloc[jj]['support']/support_1_length.iloc[ii]['support']
			confidence_list.append(confidence)

dict_confidence = {'itemsets': itemsets_list, 'confidence': confidence_list}
df_confidence = pd.DataFrame(dict_confidence)
df_confidence = df_confidence.sort_values(['confidence'],ascending = False)


#Get the lift of an association
lift_items = []
lift_values = []
lift_support = []
core_support = []
for jj in range(len(support_more_length)):
	support_lis = []
	for ii in range(len(support_1_length)):
		if support_more_length.iloc[jj]['itemsets'].issuperset(support_1_length.iloc[ii]['itemsets']):
			support_lis.append(support_1_length.iloc[ii]['support'])
	lift_items.append(support_more_length.iloc[jj]['itemsets'])
	lift = support_more_length.iloc[jj]['support']
	for ii in support_lis:
		lift = lift/ii
	lift_values.append(lift)
	lift_support.append(support_lis)
	core_support.append(support_more_length.iloc[jj]['support'])

dict_lift = {'itemsets': lift_items, 'lift': lift_values, 'supports': lift_support, 'coresupport': core_support}
df_lift = pd.DataFrame(dict_lift)
df_lift = df_lift.sort_values(['lift'],ascending = False)



######################################################################
# Association Rule Learning using the Dataset Online Retail II
######################################################################

# Goal: Suggesting products to users at the basket stage

# InvoiceNo: Invoice Number (If this code starts with C, it means that the transaction has been cancelled)
# StockCode: Product code (unique for each product)
# Description: Product name
# Quantity: Number of products (How many of the products on the invoices were sold)
# InvoiceDate: Invoice date
# UnitPrice: Invoice price ( Sterling )
# CustomerID: Unique customer number
# Country: Country name

######################################################################
# 1. Data Preparation
######################################################################

# importing the libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from mlxtend.frequent_patterns import apriori, association_rules
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.expand_frame_repr', False)   # ensures that the output is on a single line

# read the dataset
df1 = pd.read_excel('datasets/online_retail_II.xlsx', sheet_name='Year 2010-2011')
df = df1.copy()
df.columns = [col.lower() for col in df.columns]
df.head()

# get the descriptive_statistics
df.describe().T

#                 count     mean     std       min      25%      50%      75%      max
# quantity    541910.00     9.55  218.08 -80995.00     1.00     3.00    10.00 80995.00
# price       541910.00     4.61   96.76 -11062.06     1.25     2.08     4.13 38970.00
# customer id 406830.00 15287.68 1713.60  12346.00 13953.00 15152.00 16791.00 18287.00
#
# the variables quantity and price have negative values because of the C expressions in the variable invoice.
# C means 'Cancel' of the order.
# they should remove from the dataset.


# get the missing data
df.isnull().sum().sort_values(ascending=False)

#
# customer id    135080
# description      1454
# invoice             0
# stockcode           0
# quantity            0
# invoicedate         0
# price               0
# country             0
#
# customer id and description have missing data. we can remove them since there are many data in the dataset.


# to solve the above problems, define the folloowing function named 'retail_data_prep'.
def retail_data_prep(dataframe):
    dataframe.dropna(inplace=True)  # to drop the missing data
    dataframe[dataframe['invoice'].str.contains('C', na=False)] # to remove the data that include 'C' expressions
    dataframe = dataframe[dataframe['price'] > 0] # to remove the negative price data
    dataframe = dataframe[dataframe['quantity'] > 0] # to remove the negative quantity data
    
    return dataframe

df = retail_data_prep(df)
df.isnull().sum()
df.describe().T


# invoice        0
# stockcode      0
# description    0
# quantity       0
# invoicedate    0
# price          0
# customer id    0
# country        0
#
#                 count     mean     std      min      25%      50%      75%      max
# quantity    397885.00    12.99  179.33     1.00     2.00     6.00    12.00 80995.00
# price       397885.00     3.12   22.10     0.00     1.25     1.95     3.75  8142.75
# customer id 397885.00 15294.42 1713.14 12346.00 13969.00 15159.00 16795.00 18287.00
#
# now there is no missng data and negative values
#
# however, there is also another problem with outliers. we can show them using seaborn library as follows:

def check_outliers(dataframe, variable):
    sns.boxplot(x=dataframe[variable])
    plt.show(block=True)

check_outliers(df, 'price')
check_outliers(df, 'quantity')

# therefore, we should remove some outliers from the dataset.
# firts, we should determine the outlier thresholds,
# then we should replace the outliers with thresholds as follows:

def outlier_thresholds(dataframe, variable, q1=0.01, q3=0.99):
    quantile1 = dataframe[variable].quantile(q1)
    quantile3 = dataframe[variable].quantile(q3)
    iqr = quantile3 - quantile1
    low_limit = quantile1 - 1.5 * iqr
    up_limit = quantile3 + 1.5 * iqr
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[dataframe[variable] < low_limit, variable] = low_limit
    dataframe.loc[dataframe[variable] > up_limit, variable] = up_limit

replace_with_thresholds(df, 'price')
replace_with_thresholds(df, 'quantity')

# Now we changed the outliers with the thresholds. by calling the check_aoutlier function,
# we can check whether the process of ok or not
# or we can analyze the descriptive statistics as follows:

check_outliers(df, 'price')
check_outliers(df, 'quantity')
df.describe().T

#                 count     mean     std      min      25%      50%      75%      max
# quantity    397885.00    11.83   25.52     1.00     2.00     6.00    12.00   298.50
# price       397885.00     2.89    3.23     0.00     1.25     1.95     3.75    37.06
# customer id 397885.00 15294.42 1713.14 12346.00 13969.00 15159.00 16795.00 18287.00

######################################################################
# 2. Preparing the Association Rule Learning Data Structures
######################################################################

# let's move forward by reducing the dataset to a specific country, for instance United Kingdom
# So here the association rules of UK customers are derived.

df_fr = df[df['country'] == 'France']
df_fr.head()
df_fr.shape     # (8342, 8)
df.shape        # (397885, 8)


# Let's create a matrix named 'invoice_product' and define a function

def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(['invoice', 'stockcode'])['quantity'].sum().unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['invoice', 'description'])['quantity'].sum().unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0)

fr_inv_pro_df = create_invoice_product_df(df_fr, id=True)
fr_inv_pro_df.head()

# Now we created the pivot table usign stockcode. if id = False, we could create the pivot table accroding to the description
# Let's write a function named 'check_id' to check which product the id values are.

def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe['stockcode'] == stock_code][['description']].values[0].tolist()
    print(product_name)

check_id(df_fr, 11001) # ['ASSTD DESIGN RACING CAR PEN']
check_id(df_fr, 16218) # ['CARTOON  PENCIL SHARPENERS']

######################################################################
# 3. Association Rule Analysis
######################################################################

# After creating the pivot table, now we can make the association rule analysis using apriori and association_rules methods.
# Finding the support values, i.e. probabilities, of all possible visual associations with the apriori method

frequnt_items = apriori(fr_inv_pro_df, min_support=0.01, use_colnames=True)
frequnt_items.sort_values('support', ascending=False)
# Thus, product pairs and their corresponding support values are obtained.

# Association rules are deduced using this obtained data as follows:
rules = association_rules(frequnt_items, metric='support', min_threshold=0.01)

# reducing the rules according to some thresholds of support, confidence and lift values
reduced_rules = rules[(rules['support'] > 0.05) & (rules['confidence'] > 0.01) & (rules['lift'] > 5)]
reduced_rules   # [84 rows x 9 columns]
# thus the dataframe is reduced from 137270t products to 84 products

# now check the id of some products
check_id(df_fr, 21080)  # ['SET/20 RED RETROSPOT PAPER NAPKINS ']
check_id(df_fr, 22727)  # ['ALARM CLOCK BAKELIKE RED ']

# functionalization of the above processes
def create_rules(dataframe, id=True, country='France'):
    dataframe = dataframe[dataframe['country'] == country]
    dataframe = create_invoice_product_df(dataframe, id)
    frequnt_items = apriori(dataframe, min_support=0.01, use_colnames=True)
    rules = association_rules(frequnt_items, metric='support', min_threshold=0.01)

    return rules

create_rules(df_fr)

######################################################################
# 4. Product Recommendation Application
######################################################################
def arl_recommender(rules_df, product_id, recommend_count=1):
    sorted_rules = rules_df.sort_values('lift', ascending=False)
    recommendation_list = []
    for i, product in enumerate(sorted_rules['antecedents']):
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]['consequents'])[0])
    return recommendation_list[0: recommend_count]

arl_recommender(rules, 21791, 1)    # [10002]
arl_recommender(rules, 21791, 2)    # [10002, 22549]
arl_recommender(rules, 21558, 3)    # [22352, 22352, 22728]

# hocam apriori algoritmasi bolumunde support degerini hesaplama formulu var.. (Freq(X, Y) /N seklinde. bu deger cok yüksekse online ve ofline tarafta farkli aksiyonlarin alindigindan bahsetmisiz..bende oraya binanen bu farkli aksiyonlar nasil belirleniyor ve ne gibi farkliliklar yapilmaktadir?
# Burada şu tarz bir yaklaşım olabilir, tamamen mantıksal gidiyorum. Örneğin meşhur bebek bezi-bira örneğini düşünün. Burada bir ilişki yakaladıktan sonra Walmart raf dizilişini değiştirdi ve buna offline bir müdahale gözüyle bakabiliriz, benzer durumda internet alışverişinde bebek bezini sepete ekleyenlere öneri olarak bira gelse online olurdu. Veya şarkı örneği üzerinden gidelim, A grubunu dinleyenler B grubunu da çokça dinliyor olsun. Spotify gibi bir şirket online olarak A grubunu dinleyenlere B’yi önerir açık bir şekilde, benzer şekilde kaset satan yerler de buna göre dizim yapar. Hatta hayal gücünü biraz daha öteye taşıyalım. Belki de A ve B gruplarını bir araya fiziksel bir şekilde getirecek bir kayıt şirketi ciddi paralar kazanabilir

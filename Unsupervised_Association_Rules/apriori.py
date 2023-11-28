import mlxtend
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules 

data_set = [['Milk', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
            ['Dill', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
            ['Milk', 'Apple', 'Kidney Beans', 'Eggs'],
            ['Milk', 'Unicorn', 'Corn', 'Kidney Beans', 'Yogurt'],
            ['Corn', 'Onion', 'Onion', 'Kidney Beans', 'Ice cream', 'Eggs']]
te=TransactionEncoder

te=TransactionEncoder()
te_ary=te.fit(data_set).transform(data_set)    #Apply one-hot-encoding on our dataset
df=pd.DataFrame(te_ary, columns=te.columns_)  # type: ignore #Creating a new DataFrame from our Numpy array

print(df)
print(apriori(df, min_support=0.6))

frequent_itemsets=apriori(df, min_support=0.6, use_colnames=True) #Instead of column indices we can use column names.
print(frequent_itemsets)

confidance=association_rules(frequent_itemsets,metric="confidence",min_threshold=0.7) # associate items
print(confidance)

lift= association_rules(frequent_itemsets,metric="lift",min_threshold=1.25)
print(lift)

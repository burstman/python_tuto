from mlxtend.preprocessing import TransactionEncoder
import pandas as pd
from mlxtend.frequent_patterns import apriori

data = pd.read_csv(
    'Unsupervised_Association_Rules/Market_Basket_Optimisation.csv', header=None)

data.info()

list_of_lists = data.apply(lambda row: row.dropna().tolist(), axis=1).tolist()


te = TransactionEncoder()

# Apply one-hot-encoding on our dataset
te_ary = te.fit(list_of_lists).transform(list_of_lists)
# Creating a new DataFrame from our Numpy array
df = pd.DataFrame(te_ary, columns=te.columns_) # type: ignore
df.info()

from mlxtend.frequent_patterns import apriori
print(apriori(df, min_support=0.1,use_colnames=True))

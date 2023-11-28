from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd
from mlxtend.frequent_patterns import apriori

toy_dataset = [['Skirt', 'Sneakers', 'Scarf', 'Pants', 'Hat'],

               ['Sunglasses', 'Skirt', 'Sneakers', 'Pants', 'Hat'],

               ['Dress', 'Sandals', 'Scarf', 'Pants', 'Heels'],

               ['Dress', 'Necklace', 'Earrings', 'Scarf', 'Hat', 'Heels', 'Hat'],

               ['Earrings', 'Skirt', 'Skirt', 'Scarf', 'Shirt', 'Pants']]

te = TransactionEncoder()

# Apply one-hot-encoding on our dataset
te_ary = te.fit(toy_dataset).transform(toy_dataset)
# Creating a new DataFrame from our Numpy array
df = pd.DataFrame(te_ary, columns=te.columns_)  # type: ignore


frequent_itemsets = apriori(df, min_support=0.6, use_colnames=True)

assoc = association_rules(
    frequent_itemsets, metric="confidence", min_threshold=0.7)


assoc = association_rules(frequent_itemsets, metric="lift", min_threshold=1.25)
print(assoc)

# We have confidance 1 that's mean for every 'Skirt' bought their is 'Pants' bought.
# But we have a lift 1.25 is slightly > 1 that 's mean their is a moderate level association. the presenve
# of one may not affect the other. 

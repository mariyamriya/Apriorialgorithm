import numpy as np  # Importing NumPy for numerical operations
import pandas as pd  # Importing pandas for handling datasets
from sklearn.model_selection import train_test_split  # Importing function for dataset splitting
from sklearn.metrics import mean_squared_error  # Importing error metric function
from scipy.cluster.hierarchy import linkage, dendrogram  # Importing hierarchical clustering tools
import matplotlib.pyplot as plt  # Importing Matplotlib for visualization
from mlxtend.frequent_patterns import apriori, association_rules  # Importing functions for market basket analysis

# Sample transaction data for association rule mining
transaction_data = [
    ['bread', 'milk', 'apple', 'peanuts'],
    ['bread', 'milk', 'peanuts'],
    ['bread', 'milk'],
    ['bread', 'milk', 'apple'],
    ['bread', 'apple', 'peanuts'],
    ['bread', 'apple'],
    ['bread', 'milk', 'apple', 'peanuts'],
    ['milk', 'apple']
]

# Extract unique items from transactions
unique_items = sorted(set(item for basket in transaction_data for item in basket))

# Construct a DataFrame representing item occurrence in transactions
binary_matrix = pd.DataFrame([{item: (item in basket) for item in unique_items} for basket in transaction_data])

# Run the Apriori algorithm to identify frequent itemsets
frequent_sets = apriori(binary_matrix, min_support=0.3, use_colnames=True)

# Generate association rules from the frequent itemsets
association_results = association_rules(frequent_sets, metric="lift", min_threshold=1.0)

# Display frequent itemsets
print("Frequent Itemsets:")
print(frequent_sets)

# Display association rules
print("\nAssociation Rules:")
print(association_results)

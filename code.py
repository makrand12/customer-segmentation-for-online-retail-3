# --------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# code starts here
df = pd.read_csv(path)
df.isnull().sum()
df.dropna(inplace = True)
df = df[df["Country"] == "United Kingdom"]
df[df["InvoiceDate"] >="2010-12-09"]
df["Return"] = df.InvoiceNo.str.contains('C')
df["Purchase"]= np.where(df["Return"]==True,0,1)
print(df.head(5))

# code ends here


# --------------
# create new dataframe customer
customers = pd.DataFrame({'CustomerID': df['CustomerID'].unique()},dtype=int)

# calculate the recency
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['Recency'] = pd.to_datetime("2011-12-10") - (df['InvoiceDate'])

# remove the time factor
df.Recency = df.Recency.dt.days

# purchase equal to one 
temp = df[df['Purchase']==1]

# customers latest purchase day
recency=temp.groupby(by='CustomerID',as_index=False).min()
customers=customers.merge(recency[['CustomerID','Recency']],on='CustomerID')




# --------------
# code stars here
temp_1= df[['CustomerID','InvoiceNo','Purchase']]
temp_1.drop_duplicates(subset = ['InvoiceNo'], inplace = True)
annual_invoice = temp_1.groupby(by='CustomerID',as_index=False).sum()
temp_1.rename(columns = {'Purchase' : 'Frequency'}, inplace =True)
print(temp_1.head())
customers = customers.merge(annual_invoice,on='CustomerID')
# code ends here


# --------------
# Removed frequency == 0
customers=customers[customers['Frequency']>0]
# Create amount and groupby based on customer id
df['Amount']=df['Quantity'] * df['UnitPrice']
annual_sales=df.groupby(by='CustomerID',as_index=False).sum()
annual_sales.rename(columns={'Amount':'monetary'},inplace=True)

# added in the customer dataframe
customers=customers.merge(annual_sales[['CustomerID','monetary']],on='CustomerID')




# --------------
# code ends here
np.where(customers['monetary']<0)
customers["Recency_log"] = np.log(customers['Recency']+0.1)
customers["Frequency_log"] = np.log(customers['Frequency']+0.1)
customers["Monetary_log"] = np.log(customers['monetary']+0.1)
# code ends here


# --------------
# import packages
from sklearn.cluster import KMeans

# Code starts here

# Empty list for storing WCSS across all values of k
dist = []

# Iterate from 1-9
for i in range(1,10):
    # Initialize KMeans algorithm
    km=KMeans(n_clusters=i,init='k-means++', max_iter=300, n_init=10, random_state=0)
    # Fit on data
    km.fit(customers.iloc[:,1:7])
    # Append WCSS to list storing WCSS
    dist.append(km.inertia_)

# Initialize figure
plt.figure(figsize=(10,10))

# Line plot # clusters on X-axis and WCSS on Y-axis 
plt.plot(range(1,10),dist)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('wcss')
plt.show()

# Code ends here


# --------------

# Code starts here

# initialize KMeans object
cluster = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)

# create 'cluster' column
customers['cluster'] = cluster.fit_predict(customers.iloc[:,1:7])

# plot the cluster

customers.plot.scatter(x= 'Frequency_log', y= 'Monetary_log', c='cluster', colormap='viridis')
plt.show()

# Code ends here



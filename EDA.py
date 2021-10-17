# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# # Bank Customer Churn Prediction - EDA
# %% [markdown]
# ## Dependencies

# %%
import pandas as pd
import numpy as np

# Matplotlib for visualization
from matplotlib import pyplot as plt
# display plots in the notebook
get_ipython().run_line_magic('matplotlib', 'inline')

# Seaborn for easier visualization
import seaborn as sns
sns.set_style('darkgrid')

# store elements as dictionary keys and their counts as dictionary values
from collections import Counter

# scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Classification metrics
from sklearn.metrics import confusion_matrix, classification_report


# %%
# If you are runnig "Dark Reader" on Chrome,install "jupyterthemes 
#  and run these lines so the legends and axes become visible
## from jupyterthemes import jtplot
## jtplot.style(theme='monokai', context='notebook', ticks=True, grid=False)

# To reset default matplotlib rcParams
## jtplot.reset()

# %% [markdown]
# # Data Source
# %% [markdown]
# [Kaggle - Churn Modelling Calssification Data Set](https://www.kaggle.com/shrutimechlearn/churn-modelling)
# 
# This data set contains details of a bank's customers and the target variable is a binary variable reflecting the fact whether the customer left the bank (closed his account) or he continues to be a customer.
# 
# It consists of 10,000 records  with demographic and bank history information from customers from three countries, France, Germany and Spain,
# 
# 
# %% [markdown]
# # Exploratory Analysis
# %% [markdown]
# ### Reading the CSV and Performing Basic Data Cleaning

# %%
# Loading the dataset
df = pd.read_csv("Resources/Churn_Modelling.csv")
print(f"Dataframe dimensions: {df.shape}")
df.head()


# %%
df.info()

# %% [markdown]
# There are no "nulls" in our dataframe.

# %%
# Listing number of unique customer IDs
df.CustomerId.nunique()

# %% [markdown]
# All Cusuomer IDs are unique --> that also means no duplicates

# %%
df.duplicated().sum()

# %% [markdown]
# #### Unused Features
# %% [markdown]
# To make dataframe easily readable we will drop features not needed for machine learning:
# 
# * RowNumber
# * CustomerId
# * Surname

# %%
# Drop unused features
df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)
print(f"Dataframe dimensions: {df.shape}")
df.head()

# %% [markdown]
# ### Distributions of Numeric Features
# %% [markdown]
# #### Plotting Histogram grid

# %%
# Plot histogram grid
df.hist(figsize=(14,14))

plt.show()

# %% [markdown]
# #### Summary statistics for the numeric features

# %%
# Summarizing numerical features
df.describe()

# %% [markdown]
# From the summary statistics and the histograms we can conclude that all features look OK. We do not see any extreme values for any feature.
# %% [markdown]
# ### Distributions of Categorical Features

# %%
# Summarizing categorical features
df.describe(include=['object'])

# %% [markdown]
# This shows us the number of unique classes for each feature. For example, there are more males (5457) than females. And France is most common of 3 geographies in our dataframe. There are no sparse classes.
# %% [markdown]
# Let's visualize this information.

# %%
# Bar plot for "Gender"
plt.figure(figsize=(4,4))
df['Gender'].value_counts().plot.bar(color=['b', 'g'])
plt.ylabel('Count')
plt.xlabel('Gender')
plt.xticks(rotation=0)
plt.show()

# Display count of each class
Counter(df.Gender)

# %% [markdown]
# In our data sample there are more males than females.

# %%
# Bar plot for "Geography"
plt.figure(figsize=(6,4))
df['Geography'].value_counts().plot.bar(color=['b', 'g', 'r'])
plt.ylabel('Count')
plt.xlabel('Geography')
plt.xticks(rotation=0)
plt.show()

# Display count of each class
Counter(df.Geography)

# %% [markdown]
# Majority of customers are from France, about 50%, and from Germany and Spain around 25% each.
# %% [markdown]
# ### Churn Segmentation by Gender¶

# %%
# Segment "Exited" by gender and display the frequency and percentage within each class
grouped = df.groupby('Gender')['Exited'].agg(Count='value_counts')
grouped


# %%
# Reorganize dataframe for plotting count
dfgc = grouped
dfgc = dfgc.pivot_table(values='Count', index='Gender', columns=['Exited'])
dfgc


# %%
# Calculate percentage within each class
dfgp = grouped.groupby(level=[0]).apply(lambda g: round(g * 100 / g.sum(), 2))
dfgp.rename(columns={'Count': 'Percentage'}, inplace=True)
dfgp


# %%
# Reorganize dataframe for plotting percentage
dfgp = dfgp.pivot_table(values='Percentage', index='Gender', columns=['Exited'])
dfgp


# %%
# Churn distribution by gender, count + percentage

labels= ['Stays', 'Exits']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

dfgc.plot(kind='bar',
          color=['g', 'r'],
          rot=0, 
          ax=ax1)
ax1.legend(labels)
ax1.set_title('Churn Risk per Gender (Count)', fontsize=14, pad=10)
ax1.set_ylabel('Count',size=12)
ax1.set_xlabel('Gender', size=12)


dfgp.plot(kind='bar',
          color=['g', 'r'],
          rot=0, 
          ax=ax2)
ax2.legend(labels)
ax2.set_title('Churn Risk per Gender (Percentage)', fontsize=14, pad=10)
ax2.set_ylabel('Percentage',size=12)
ax2.set_xlabel('Gender', size=12)

plt.show()

# %% [markdown]
# 
# In percentage females are more likely to leave the bank; 25% comparing to males, 16%.
# %% [markdown]
# ### Churn Segmentation by Geography¶

# %%
# Segment "Exited" by geography and display the frequency and percentage within each class
grouped = df.groupby('Geography')['Exited'].agg(Count='value_counts')
grouped


# %%
# Reorganize dataframe for plotting count
dfgeoc = grouped
dfgeoc = dfgeoc.pivot_table(values='Count', index='Geography', columns=['Exited'])
dfgeoc


# %%
# Calculate percentage within each class
dfgeop = grouped.groupby(level=[0]).apply(lambda g: round(g * 100 / g.sum(), 2))
dfgeop.rename(columns={'Count': 'Percentage'}, inplace=True)
dfgeop


# %%
# Reorganize dataframe for plotting percentage
dfgeop = dfgeop.pivot_table(values='Percentage', index='Geography', columns=['Exited'])
dfgeop


# %%
# Churn distribution by geography, count + percentage

labels= ['Stays', 'Exits']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

dfgeoc.plot(kind='bar',
          color=['g', 'r'],
          rot=0, 
          ax=ax1)
ax1.legend(labels)
ax1.set_title('Churn Risk per Geography (Count)', fontsize=14, pad=10)
ax1.set_ylabel('Count',size=12)
ax1.set_xlabel('Geography', size=12)


dfgeop.plot(kind='bar',
          color=['g', 'r'],
          rot=0, 
          ax=ax2)
ax2.legend(labels)
ax2.set_title('Churn Risk per Geography (Percentage)', fontsize=14, pad=10)
ax2.set_ylabel('Percentage',size=12)
ax2.set_xlabel('Geography', size=12)

plt.show()

# %% [markdown]
# 
# The smallest number of customers are from Germany but it looks that they are most likely to leave the bank. Almost one third of German customers in our sample left the bank.
# %% [markdown]
# ## Correlations

# %%
# Calculate correlations between numeric features
correlations = df.corr()

# sort features in order of their correlation with "Exited"
sort_corr_cols = correlations.Exited.sort_values(ascending=False).keys()
sort_corr = correlations.loc[sort_corr_cols,sort_corr_cols]
sort_corr

# %% [markdown]
# Let's use Seaborn's .heatmap() function to visualize the correlation grid.

# %%
# Generate a mask for the upper triangle
corr_mask = np.zeros_like(correlations)
corr_mask[np.triu_indices_from(corr_mask)] = 1

# Make the figsize 9x9
plt.figure(figsize=(9,9))

# Plot heatmap of annotated correlations; change background to white
##with sns.axes_style('white'):
sns.heatmap(sort_corr*100, 
                cmap='RdBu', 
                annot=True,
                fmt='.0f',
                mask=corr_mask,
                cbar=False)
    
plt.title('Correlations by Exited', fontsize=14)
plt.yticks(rotation=0)
plt.show()

# %% [markdown]
# Very weak correlations in general. Only weak positive correlation with age, very weak positive correlation with balance, and very weak negative correlations with number of products and membership.
# %% [markdown]
# ## Pairplot

# %%
# Plot Seaborn's pairplot
g = sns.pairplot(df, hue='Exited',
                 palette={1 : 'green',
                          0 : 'red'},
                 plot_kws={'alpha' : 0.8, 'edgecolor' : 'b', 'linewidth' : 0.5})

fig = g.fig
fig.subplots_adjust(top=0.95, wspace=0.2)
fig.suptitle('Plot by "Exited" Classes',
             fontsize=26,
             fontweight='bold')


# Update the legend
new_title = 'Churn Risk'
g._legend.set_title(new_title)
# replace labels
new_labels = ['Stays', 'Exits']
for t, l in zip(g._legend.texts, new_labels): t.set_text(l)

plt.show()

# %% [markdown]
# The *density plots* on the diagonal make it easier to compare these distributions. We can notice that only few features have slightly different distributions. For example, from the density plot for <code style="color:steelblue">Age</code>, it could be seen that older people have slightly higher tendecy to leave the bank.
# %% [markdown]
# Let’s reduce the clutter by plotting only four features: <code style="color:steelblue">Age</code>, <code style="color:steelblue">IsActiveMember</code>, <code style="color:steelblue">NumOfProducts</code> and <code style="color:steelblue">Balance</code>.

# %%
# Plot Seaborn's pairplot
g = sns.pairplot(df, hue='Exited',
                 vars=['Age', 'IsActiveMember', 'NumOfProducts', 'Balance'], # reduce to less features
                 palette={0 : 'green',
                          1 : 'red'},
                 plot_kws={'alpha' : 0.8, 'edgecolor' : 'b', 'linewidth' : 0.5})

fig = g.fig
fig.subplots_adjust(top=0.95, wspace=0.2)
fig.suptitle('Reduced Plot by "Exited" Classes',
             fontsize=14,
             fontweight='bold')

# Update the legend
new_title = 'Churn Risk'
g._legend.set_title(new_title)
# replace labels
new_labels = ['Stays', 'Exits']
for t, l in zip(g._legend.texts, new_labels): t.set_text(l)

plt.show()

# %% [markdown]
# From density plots we can see that older customers and customer with more products more often leaving the bank.
# %% [markdown]
# ## Violin Plots

# %%
# Segment age by Exited and plot distributions
#  “categorical” variable Exited is a numeric
#  for plotting purposes only we will change it to real categorical variable

# Define palette
my_pal = {'Stays': 'green', 'Exits': 'red'}
# Convert to categorical
hr = {0: 'Stays', 1: 'Exits'}
churn = df['Exited'].map(hr)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Churn Risk vs. Different Attributes', fontsize=16)
fig.subplots_adjust(top=0.92, wspace=0.3, hspace=0.3)

sns.violinplot(x=churn,
               y=df['Age'],
               order=['Stays', 'Exits'], 
               palette=my_pal,
               ax=ax1)

ax1.set_title('Churn Risk vs. Age', fontsize=14, pad=10)
ax1.set_ylabel('Age',size=12)
ax1.set_xlabel('Churn Risk ("Exited")', size=12)

sns.violinplot(x=churn,
               y=df['Balance'],
               order=['Stays', 'Exits'], 
               palette=my_pal,
               ax=ax2)

ax2.set_title('Churn Risk vs. Balance', fontsize=14, pad=10)
ax2.set_ylabel('Balance',size=12)
ax2.set_xlabel('Churn Risk ("Exited")', size=12)

sns.violinplot(x=churn,
               y=df['NumOfProducts'],
               order=['Stays', 'Exits'], 
               palette=my_pal,
               ax=ax3)

ax3.set_title('Churn Risk vs. Number of Products', fontsize=14, pad=10)
ax3.set_ylabel('NumOfProducts',size=12)
ax3.set_xlabel('Churn Risk ("Exited")', size=12)

sns.violinplot(x=churn,
               y=df['IsActiveMember'],
               order=['Stays', 'Exits'], 
               palette=my_pal,
               ax=ax4)

ax4.set_title('Churn Risk vs. Active Membership', fontsize=14, pad=10)
ax4.set_ylabel('IsActiveMember',size=12)
ax4.set_xlabel('Churn Risk ("Exited")', size=12)
plt.show()

# %% [markdown]
# Violin plots are confirming the earlier statement that older customers and customer with more products are more likely to leave the bank.
# %% [markdown]
# ### Distributions of the Target Feature

# %%
# Define our target variable
y = df.Exited


# %%
y.shape

# %% [markdown]
# Let's define a small helper funtcion which displays count and percentage per class of the target feature.

# %%
# Function to display count and percentage per class of target feature
def class_count(a):
    counter=Counter(a)
    kv=[list(counter.keys()),list(counter.values())]
    dff = pd.DataFrame(np.array(kv).T, columns=['Exited','Count'])
    dff['Count'] = dff['Count'].astype('int64')
    dff['%'] = round(dff['Count'] / a.shape[0] * 100, 2)
    return dff.sort_values('Count',ascending=False)


# %%
# Let's use the function
dfcc = class_count(y)
dfcc


# %%
# Plot distribution of target variable, Exited column

labels=['Stays', 'Exits']
dfcc.plot.bar(x='Exited', y='Count', color=['g', 'r'], legend=False)
plt.xticks(dfcc['Exited'], labels, rotation=0)
plt.ylabel('Count')
plt.show()

# %% [markdown]
# We can see that our dataset is imbalanced. The majority class, "Stays" (0), has around 80% data points and the minority class, "Exits" (1), has around 20% datapoints.
# 
# To address this, in our machine learning algorithms we will use SMOTE (Synthetic Minority Over-sampling Technique).
# %% [markdown]
# ## Finalizing the Dataframe

# %%
df.head()


# %%
df.info()

# %% [markdown]
# Our dataframe looks good and it is ready to be saved.
# %% [markdown]
# #### Save the dataframe as the analytical base table

# %%
# Save analytical base table
### df.to_csv('Resources/analytical_base_table.csv', index=None)


# %%




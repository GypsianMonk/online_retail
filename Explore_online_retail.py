#!/usr/bin/env python
# coding: utf-8

# # Portfolio Project: Online Retail Exploratory Data Analysis with Python

# ## Overview
# 
# In this project, you will step into the shoes of an entry-level data analyst at an online retail company, helping interpret real-world data to help make a key business decision.

# ## Case Study
# In this project, you will be working with transactional data from an online retail store. The dataset contains information about customer purchases, including product details, quantities, prices, and timestamps. Your task is to explore and analyze this dataset to gain insights into the store's sales trends, customer behavior, and popular products. 
# 
# By conducting exploratory data analysis, you will identify patterns, outliers, and correlations in the data, allowing you to make data-driven decisions and recommendations to optimize the store's operations and improve customer satisfaction. Through visualizations and statistical analysis, you will uncover key trends, such as the busiest sales months, best-selling products, and the store's most valuable customers. Ultimately, this project aims to provide actionable insights that can drive strategic business decisions and enhance the store's overall performance in the competitive online retail market.
# 
# ## Prerequisites
# 
# Before starting this project, you should have some basic knowledge of Python programming and Pandas. In addition, you may want to use the following packages in your Python environment:
# 
# - pandas
# - numpy
# - seaborn
# - matplotlib
# 
# These packages should already be installed in Coursera's Jupyter Notebook environment, however if you'd like to install additional packages that are not included in this environment or are working off platform you can install additional packages using `!pip install packagename` within a notebook cell such as:
# 
# - `!pip install pandas`
# - `!pip install matplotlib`

# ## Project Objectives
# 1. Describe data to answer key questions to uncover insights
# 2. Gain valuable insights that will help improve online retail performance
# 3. Provide analytic insights and data-driven recommendations

# ## Dataset
# 
# The dataset you will be working with is the "Online Retail" dataset. It contains transactional data of an online retail store from 2010 to 2011. The dataset is available as a .xlsx file named `Online Retail.xlsx`. This data file is already included in the Coursera Jupyter Notebook environment, however if you are working off-platform it can also be downloaded [here](https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx).
# 
# The dataset contains the following columns:
# 
# - InvoiceNo: Invoice number of the transaction
# - StockCode: Unique code of the product
# - Description: Description of the product
# - Quantity: Quantity of the product in the transaction
# - InvoiceDate: Date and time of the transaction
# - UnitPrice: Unit price of the product
# - CustomerID: Unique identifier of the customer
# - Country: Country where the transaction occurred

# ## Tasks
# 
# You may explore this dataset in any way you would like - however if you'd like some help getting started, here are a few ideas:
# 
# 1. Load the dataset into a Pandas DataFrame and display the first few rows to get an overview of the data.
# 2. Perform data cleaning by handling missing values, if any, and removing any redundant or unnecessary columns.
# 3. Explore the basic statistics of the dataset, including measures of central tendency and dispersion.
# 4. Perform data visualization to gain insights into the dataset. Generate appropriate plots, such as histograms, scatter plots, or bar plots, to visualize different aspects of the data.
# 5. Analyze the sales trends over time. Identify the busiest months and days of the week in terms of sales.
# 6. Explore the top-selling products and countries based on the quantity sold.
# 7. Identify any outliers or anomalies in the dataset and discuss their potential impact on the analysis.
# 8. Draw conclusions and summarize your findings from the exploratory data analysis.

# ## Task 1: Load the Data

# In[10]:


# your code here 1.
import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[11]:


df = pd.read_excel('Online Retail.xlsx')


print(df.head(10))


# In[12]:


#shape of our dataset
df.shape


# In[13]:


#check the head of the dataset
df.head(10)


# In[14]:


df.tail(10)


# In[15]:


df.info()


# In[16]:


#exploring the unique values of each attribute
print("Number of transactions: ", df['InvoiceNo'].nunique())
print("Number of products: ",df['StockCode'].nunique())
print("Number of customers:", df['CustomerID'].nunique() )
print("Percentage of customers NA: ", round(df['CustomerID'].isnull().sum() * 100 / len(df),2),"%" )
print('Number of countries: ',df['Country'].nunique())


# In[17]:


df.describe()


# In[18]:


#get cancelled transactions
cancelled_orders = df[df['InvoiceNo'].astype(str).str.contains('C')]
cancelled_orders.head()


# In[19]:


cancelled_orders[cancelled_orders['Quantity']>0]


# In[20]:


#search for transaction where quantity == -80995
cancelled_orders[cancelled_orders['Quantity']==-80995]


# In[21]:


#2 Perform Data Cleaning


# In[22]:


# Check for missing values
missing_values = df.isnull().sum()
print("Missing Values:\n", missing_values)

# Handle missing values (if any)
# Example: Drop rows with any missing values
df.dropna(inplace=True)

# Check the cleaned dataset
print("Cleaned Data:\n", df.head())


# In[23]:


# Identify redundant or unnecessary columns
# Example: Remove 'Description' column
df.drop(columns=['Description'], inplace=True)

# Check the dataset after removing columns
print("Data after removing unnecessary columns:\n", df.head())


# In[24]:


#3. Explore Basic Statistics


# In[26]:


# Display basic statistics using describe() method
basic_stats = df.describe()
print("Basic Statistics:\n", basic_stats)


# In[ ]:


# Calculate measures of central tendency
mean_values = df.mean()
median_values = df.median()
mode_values = df.mode()

# Calculate measures of dispersion
std_deviation = df.std()
range_values = df.max() - df.min()

print("\nMeasures of Central Tendency:")
print("Mean:\n", mean_values)
print("\nMedian:\n", median_values)
print("\nMode:\n", mode_values)

print("\nMeasures of Dispersion:")
print("Standard Deviation:\n", std_deviation)
print("\nRange:\n", range_values)


# In[ ]:


# 4 Data Visualization 


# In[34]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



# Set the style for seaborn plots
sns.set(style="whitegrid")

# Histogram for Quantity distribution
plt.figure(figsize=(8, 6))
sns.distplot(df['Quantity'], bins=30, kde=True, color='skyblue')
plt.title('Distribution of Quantity')
plt.xlabel('Quantity')
plt.ylabel('Density')
plt.show()


# Scatter plot for Quantity vs. UnitPrice
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Quantity', y='UnitPrice', data=df, color='green')
plt.title('Scatter plot of Quantity vs. Unit Price')
plt.xlabel('Quantity')
plt.ylabel('Unit Price')
plt.show()

# Bar plot for top-selling products (based on Quantity)
top_products = df.groupby('Description')['Quantity'].sum().nlargest(10)
plt.figure(figsize=(10, 6))
top_products.plot(kind='bar', color='orange')
plt.title('Top 10 Selling Products')
plt.xlabel('Product')
plt.ylabel('Total Quantity Sold')
plt.xticks(rotation=45)
plt.show()

# Line plot for sales trends over time
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['YearMonth'] = df['InvoiceDate'].dt.to_period('M')
sales_trends = df.groupby('YearMonth')['Quantity'].sum()
plt.figure(figsize=(10, 6))
sales_trends.plot(kind='line', marker='o', color='red')
plt.title('Sales Trends Over Time')
plt.xlabel('Year-Month')
plt.ylabel('Total Quantity Sold')
plt.xticks(rotation=45)
plt.show()


# In[35]:


# 5 Analyzing Sales Trends Over Time


# In[36]:


import pandas as pd
import matplotlib.pyplot as plt


# Convert 'InvoiceDate' to datetime format
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Extract month and day of the week from 'InvoiceDate'
df['Month'] = df['InvoiceDate'].dt.month
df['DayOfWeek'] = df['InvoiceDate'].dt.day_name()

# Group by month and day of the week to analyze sales trends
monthly_sales = df.groupby('Month')['Quantity'].sum()
daywise_sales = df.groupby('DayOfWeek')['Quantity'].sum()

# Plotting sales trends over time (monthly)
plt.figure(figsize=(10, 6))
monthly_sales.plot(kind='line', marker='o', color='blue')
plt.title('Monthly Sales Trends')
plt.xlabel('Month')
plt.ylabel('Total Quantity Sold')
plt.xticks(ticks=range(1, 13), labels=[
    'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
])
plt.show()

# Displaying the busiest days of the week in terms of sales
busiest_days = daywise_sales.sort_values(ascending=False)
print("Busiest Days of the Week in Terms of Sales:")
print(busiest_days)


# In[37]:


#6 Explore the top-selling products and countries based on the quantity sold.


# In[38]:


#Exploring Top-Selling Products:

# Calculate top-selling products based on the total quantity sold
top_products = df.groupby('Description')['Quantity'].sum().nlargest(10)

# Bar plot for top-selling products
plt.figure(figsize=(10, 6))
top_products.plot(kind='bar', color='orange')
plt.title('Top 10 Selling Products')
plt.xlabel('Product')
plt.ylabel('Total Quantity Sold')
plt.xticks(rotation=45)
plt.show()


# In[39]:


#Exploring Top Selling Countries:

# Calculate total sales quantity per country
sales_by_country = df.groupby('Country')['Quantity'].sum().sort_values(ascending=False)

# Select top countries based on total sales quantity
top_countries = sales_by_country.head(10)

# Bar plot for top-selling countries
plt.figure(figsize=(12, 8))
top_countries.plot(kind='bar', color='green')
plt.title('Top 10 Countries by Total Sales Quantity')
plt.xlabel('Country')
plt.ylabel('Total Quantity Sold')
plt.xticks(rotation=90)
plt.show()


# In[40]:


#7. Identify any outliers or anomalies in the dataset and discuss their potential impact on the analysis.


# In[41]:




# Set the style for seaborn plots
sns.set(style="whitegrid")

# Box plot for 'Quantity' column to identify outliers
plt.figure(figsize=(6, 6))
sns.boxplot(y='Quantity', data=df, color='skyblue')
plt.title('Box plot for Quantity')
plt.ylabel('Quantity')
plt.show()

# Box plot for 'UnitPrice' column
plt.figure(figsize=(6, 6))
sns.boxplot(y='UnitPrice', data=df, color='lightgreen')
plt.title('Box plot for UnitPrice')
plt.ylabel('Unit Price')
plt.show()


# In[43]:


from scipy import stats

# Calculate z-scores for 'Quantity' column
z_scores_quantity = stats.zscore(df['Quantity'])

# Calculate z-scores for 'UnitPrice' column
z_scores_unit_price = stats.zscore(df['UnitPrice'])

# Define threshold for outliers (e.g., z-score greater than 3 or less than -3)
threshold = 3

# Filter outliers based on the threshold
outliers_quantity = df[abs(z_scores_quantity) > threshold]
outliers_unit_price = df[abs(z_scores_unit_price) > threshold]

print("Outliers in Quantity column:")
print(outliers_quantity)

print("\nOutliers in UnitPrice column:")
print(outliers_unit_price)


# Impact of Outliers:
# Skewed Analysis: Outliers can distort the distribution, leading to skewed statistical measures such as mean and standard deviation.
# Misinterpretation of Trends: Outliers might affect trend analysis, making trends less clear or even creating false trends.
# Influence on Relationships: Outliers can influence correlations or relationships between variables, impacting the accuracy of predictions or models.

# In[ ]:


#8 Draw conclusions and summarize your findings from the exploratory data analysis.


# # Here is a summary of findings and conclusions drawn from the exploratory data analysis (EDA) performed on the "Online Retail" dataset:
# 
# 1. **Sales Trends Over Time:**
#    - Monthly sales show variations throughout the year, with certain months having higher sales than others.
#    - Peaks and troughs in sales might be influenced by seasonal trends or promotional activities.
# 
# 2. **Busiest Days of the Week:**
#    - Identified the days of the week with the highest sales volume.
#    - Understanding these patterns can help in planning promotions or scheduling operations efficiently.
# 
# 3. **Top-Selling Products:**
#    - Analyzed the top-selling products based on total quantity sold.
#    - Identified the products that contribute the most to the overall sales volume.
# 
# 4. **Top Selling Countries:**
#    - Explored sales quantities across different countries to identify top-performing markets.
#    - Understanding sales distribution across countries can aid in targeting marketing efforts or expanding operations.
# 
# 5. **Outlier Detection:**
#    - Identified outliers, if any, in the 'Quantity' and 'UnitPrice' columns using box plots and z-scores.
#    - Evaluated potential impacts of outliers on statistical measures and analysis outcomes.
# 
# 6. **Data Cleaning Insights:**
#    - Explored basic statistics and handled missing values or unnecessary columns during data cleaning.
#    - Cleaned data is essential for accurate analysis and prevents biased results.
# 
# 7. **Potential Impact on Analysis:**
#    - Outliers might skew statistical measures, affect distribution, and distort relationships between variables.
#    - Understanding the impact of outliers is crucial for accurate interpretation and modeling.
# 
# 8. **Further Analysis and Recommendations:**
#    - Further analysis can involve deeper segmentation of sales data by customer demographics, products, or regions for targeted marketing strategies.
#    - Recommendations may include adjusting pricing strategies, promoting popular products, or focusing on high-sales periods.
# 
# Overall, the EDA provided valuable insights into sales trends, popular products, and sales distribution across various dimensions. Further analysis and targeted strategies based on these findings can potentially enhance business decisions, improve operational efficiency, and drive revenue growth for the online retail store.

# In[ ]:





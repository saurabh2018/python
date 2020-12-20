
import numpy as np 
import pandas as pd
import matplotlib.pyplot as  plt
from matplotlib.pyplot import ion, plot
from pylab import*

#PLOT AND SHOW
"""plot([10,20,30])
show()"""


#ARRAY
"""a = np.array([1, 2,3 ,4])
print (a)"""


#REPEAT
"""print (np.repeat('2015', 10))
sales_2017 = pd.DataFrame([['chair',20],['sofa',24],['table',15]],columns=['product','sales_units'])
sales_2018 = pd.DataFrame([['chair',25],['sofa',10],['shelf',10]],columns=['product','sales_units'])
print(sales_2017)
print(sales_2018)

sales_2017['year'] = np.repeat(2017,sales_2017.shape[0])
sales_2018['year'] = np.repeat(2018,sales_2018.shape[0])
sales = pd.concat([sales_2017,sales_2018], ignore_index=True)
print(sales)"""


#HISTOGRAM
"""rng = np.random.default_rng(2)
mu, sigma = 2, 0.5
v = rng.normal(mu,sigma,10000)
plt.hist(v, bins=50, density=1)
(n, bins) = np.histogram(v, bins=50, density=True)
plt.plot(.5*(bins[1:]+bins[:-1]), n)
show()"""

#NUMPY HISTOGRAM
"""df=pd.read_csv('height_and_weight.csv')
df.head()
df.Height.plot(kind='hist',color='blue',edgecolor='black',figsize=(10,2))
plt.title('Distribution of Height', size=24)
plt.xlabel('Height (inches)', size=9)
plt.ylabel('Frequency', size=9);
show()"""

#SCATTER CHART
df=pd.read_csv('height_and_weight.csv')
df.head()
df.plot(kind='scatter', x='Height',y='Weight', color='blue',alpha=0.3, figsize=(9,6))
plt.title('Relationship between Height and Weight', size=24)
plt.xlabel('Height (inches)', size=18)
plt.ylabel('Weight (pounds)', size=18);






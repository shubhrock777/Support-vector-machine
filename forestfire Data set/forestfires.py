import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#loading the dataset
forest = pd.read_csv("D:/BLR10AM/Assi/26.SVM/Datasets_SVM/forestfires.csv")

#2.	Work on each feature of the dataset to create a data dictionary as displayed in the below image
#######feature of the dataset to create a data dictionary

#######feature of the dataset to create a data dictionary




data_details =pd.DataFrame({"column name":forest.columns,
                            "data type(in Python)": forest.dtypes})

            #3.	Data Pre-forestcessing
          #3.1 Data Cleaning, Feature Engineering, etc
          
          
#details of forest 
forest.info()
forest.describe()          

forest.nunique()

#for targe checking data is balanced 
forest.size_category.value_counts()



#data types        
forest.dtypes


#checking for na value
forest.isna().sum()
forest.isnull().sum()


#checking unique value for each columns
forest.nunique()


"""	Exploratory Data Analysis (EDA):
	Summary
	Univariate analysis
	Bivariate analysis """

    


EDA ={"column ": forest.columns,
      "mean": forest.mean(),
      "median":forest.median(),
      "mode":forest.mode(),
      "standard deviation": forest.std(),
      "variance":forest.var(),
      "skewness":forest.skew(),
      "kurtosis":forest.kurt()}

EDA





# covariance for data set 
covariance = forest.cov()
covariance

# Correlation matrix 
Correlation = forest.corr()
Correlation

# according to correlation coefficient no correlation of  Administration & State with model_dffit
#According scatter plot strong correlation between model_dffit and rd_spend and 
#also some relation between model_dffit and m_spend.

#variance for each column
forest.var()                   #rain column has low variance 

forest.rain.value_counts()     #rain column has maximum 0 

#droping rain colunm
forest.drop(["rain"], axis = 1, inplace = True)


sns_df=pd.concat([forest.iloc[:,2:10],forest.iloc[:,[29]]],axis=1)
####### graphidf repersentation 

##historgam and scatter plot
import seaborn as sns
sns.pairplot(sns_df,hue='size_category')


#boxplot for every columns
forest.columns
forest.nunique()

#boxplot for every column

# Boxplot of independent variable distribution for each category of size_category

sns.boxplot(x = "size_category", y = "FFMC", data =forest)
sns.boxplot(x = "size_category", y = "DMC", data = forest)
sns.boxplot(x = "size_category", y = "DC", data = forest)
sns.boxplot(x = "size_category", y = "ISI", data = forest)
sns.boxplot(x = "size_category", y = "temp", data =forest)
sns.boxplot(x = "size_category", y = "RH", data = forest)
sns.boxplot(x = "size_category", y = "wind", data = forest)
sns.boxplot(x = "size_category", y = "area", data = forest)

forest.boxplot(column=['FFMC', 'DMC', 'DC', 'ISI','temp', 'RH', 'wind', 'area'])  

# Create dummy variables on categorcal columns
#or 
from sklearn.preprocessing import OneHotEncoder
# creating instance of one-hot-encoder
enc = OneHotEncoder(handle_unknown='ignore')
sta=forest.iloc[:,[0,1]]

enc_df = pd.DataFrame(enc.fit_transform(sta).toarray())



#normal

# Normalization function using z std. all are continuous data.
def norm_func(i):
    x = (i-i.mean())/(i.std())
    return (x)

# Normalized data frame (considering the numerical part of data)
df = norm_func(forest.iloc[:,2:10])
df.describe()


#final dataframe
model_df = pd.concat([forest.iloc[:,[29]],enc_df,df,forest.iloc[:,10:29] ], axis =1)


##################################
###upport Vector Machines MODEL###
"""5.	Model Building
5.1	Build the model on the scaled data (try multiple options)
5.2	Perform Support Vector Machines.
5.3	Train and Test the data and compare accuracies by Confusion Matrix and use different Hyper Parameters
5.4	Briefly explain the model output in the documentation

"""


from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

train,test = train_test_split(model_df, test_size = 0.20,random_state=77)

train_X = train.iloc[:, 1:]
train_y = train.iloc[:, 0]
test_X  = test.iloc[:, 1:]
test_y  = test.iloc[:, 0]


#checking ratio for train and test 
test_y.value_counts()
train_y.value_counts()


# kernel = linear
model_linear = SVC(kernel = "linear")

model_linear.fit(train_X, train_y)

pred_test_linear = model_linear.predict(test_X)

np.mean(pred_test_linear == test_y)

# kernel = rbf
model_rbf = SVC(kernel = "rbf")
model_rbf.fit(train_X, train_y)
pred_test_rbf = model_rbf.predict(test_X)

np.mean(pred_test_rbf==test_y)


#liner kernel is performing well 

# Constructing the confusion matrix for liner kernel
from sklearn.metrics import confusion_matrix
confusion_matrix(test_y,pred_test_linear)

# Constructing the confusion matrix for liner rbf
from sklearn.metrics import confusion_matrix
confusion_matrix(test_y,pred_test_rbf)
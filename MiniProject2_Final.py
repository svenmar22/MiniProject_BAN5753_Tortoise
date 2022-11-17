#!/usr/bin/env python
# coding: utf-8

# In[2]:


import findspark
findspark.init()


# In[3]:


from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
from pyspark.sql.types import * 
import pyspark.sql.functions as F
from pyspark.sql.functions import col, asc,desc
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pyspark.sql import SQLContext
from pyspark.mllib.stat import Statistics
import pandas as pd
from pyspark.sql.functions import udf
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler,StandardScaler
from pyspark.ml import Pipeline
from sklearn.metrics import confusion_matrix

spark=SparkSession.builder .master ("local[*]").appName("part3").getOrCreate()


# In[4]:


sc=spark.sparkContext
sqlContext=SQLContext(sc)


# In[5]:


import os
os.getcwd()


# In[7]:


from platform import python_version

print(python_version())


# In[8]:


sc.version #spark version


# ## Read File

# In[6]:


df=spark.read  .option("header","True") .option("inferSchema","True") .option("sep",";") .csv("C:/Users/dsiri/OneDrive/Desktop/BAN 5753/Exercises/miniproject2/XYZ_Bank_Deposit_Data_Classification.csv")


# ### Sample Data

# In[7]:


df.toPandas().head(5)


# ### 1. Exploratory Data Analysis

# In[8]:


print("There are",df.count(),"rows",len(df.columns),
      "columns" ,"in the data.") 


# In[9]:


import re
cols =[re.sub("|\.","",i) for i in df.columns]
df= df.toDF(*cols)


# In[11]:


df.groupby("y").count().show()


# #### Data Types of Columns

# In[12]:


df.printSchema()


# #### Null Values

# In[13]:


from pyspark.sql.functions import isnan, when, count, col
df.select([count(when(isnan(c), c)).alias(c) for c in df.columns]).toPandas().head()


# In[145]:


print("Numerical Variables")
df.toPandas().describe(include=['int','float'])


# In[146]:


print("Categorical Variables")
df.toPandas().describe(include=['object'])


# In[147]:


#Cardinality of all variables
df.describe().toPandas().nunique()


# In[151]:


#Distribution of Features

from matplotlib import cm

fig = plt.figure(figsize=(25,15)) ## Plot Size 
st = fig.suptitle("Numeric Variables", fontsize=50,
                  verticalalignment='center') # Plot Main Title 

for col,num in zip(df.toPandas().describe().columns, range(1,11)):
    ax = fig.add_subplot(3,4,num)
    ax.hist(df.toPandas()[col])
    plt.style.use('fast') 
    plt.grid(False)
    plt.xticks(rotation=45,fontsize=20)
    plt.yticks(fontsize=15)
    plt.title(col.upper(),fontsize=20)
plt.tight_layout()
st.set_y(0.95)
fig.subplots_adjust(top=0.85,hspace = 0.4)
plt.show()


# In[152]:


#Distribution of Features

from matplotlib import cm

fig = plt.figure(figsize=(25,15)) ## Plot Size 
st = fig.suptitle("Categorical Variables", fontsize=50,
                  verticalalignment='center') # Plot Main Title 

for col,num in zip(df.toPandas().describe(include=['object']).columns, range(1,11)):
    ax = fig.add_subplot(3,4,num)
    ax.hist(df.toPandas()[col])
    plt.style.use('fast') 
    plt.grid(False)
    plt.xticks(rotation=45,fontsize=18)
    plt.yticks(fontsize=15)
    plt.title(col.upper(),fontsize=20)
plt.tight_layout()
st.set_y(0.95)
fig.subplots_adjust(top=0.85,hspace = 1.2)
plt.show()


# In[154]:


#Pearson Correlation
numeric_features = [t[0] for t in df.dtypes if t[1] != 'string']
numeric_features_df=df.select(numeric_features)

col_names =numeric_features_df.columns
features = numeric_features_df.rdd.map(lambda row: row[0:])
corr_mat=Statistics.corr(features, method="pearson")
corr_df = pd.DataFrame(corr_mat)
corr_df.index, corr_df.columns = col_names, col_names
corr_df


# In[170]:


success = df.filter(df['y']=="yes").count()/ df.count() * 100
print("Marketing campaign success percent = "+ str(success))
Failure = df.filter(df['y']=="no").count()/ df.count() * 100
print("Marketing campaign failure percent = "+ str(Failure))


# In[191]:


df1=df.toPandas()
piefreq=df1['y'].value_counts()
labels = ['Unsubscribed','Subscribed']
piefreq.plot.pie(autopct='%1.1f%%', labels=labels,title =' Meassure of Success rate')


# In[177]:


df.filter(df['y']=="yes").groupBy("marital").count().orderBy('count', ascending=[False]).show()


# In[178]:


# marital status and deposit
j_df = pd.DataFrame()

j_df['yes'] = df1[df1['y'] == 'yes']['marital'].value_counts()
j_df['no'] = df1[df1['y'] == 'no']['marital'].value_counts()

j_df.plot.bar(title = 'Marital status and deposit')


# In[179]:


#age and deposit
df.filter(df['y']=="yes").groupBy("age").count().orderBy('age').show(200)


# In[180]:


#job and deposit
df.filter(df['y']=="yes").groupBy("job").count().orderBy('count', ascending=[False]).show(200)


# In[181]:


#job and deposit
df1=df.toPandas()

j_df = pd.DataFrame()

j_df['yes'] = df1[df1['y'] == 'yes']['job'].value_counts()
j_df['no'] = df1[df1['y'] == 'no']['job'].value_counts()

j_df.plot.bar(title = 'Job and deposit')


# In[182]:


#education and deposit
df.filter(df['y']=="yes").groupBy("education").count().orderBy('count', ascending=[False]).show(200)


# In[183]:


df1=df.toPandas()
#education and deposit
j_df = pd.DataFrame()

j_df['yes'] = df1[df1['y'] == 'yes']['education'].value_counts()
j_df['no'] = df1[df1['y'] == 'no']['education'].value_counts()

j_df.plot.bar(title = 'Education and deposit')


# In[184]:


#Contact and deposit
df.filter(df['y']=="yes").groupBy("contact").count().orderBy('count', ascending=[False]).show(200)


# In[19]:


#Create a new column - Age Group
df2=df
def udf_multiple(age):
      if (age <= 25):
        return 'Under 25'
      elif (age >= 25 and age <= 35):
        return 'Between 25 and 35'
      elif (age > 35 and age < 50):
        return 'Between 36 and 49'
      elif (age >= 50):
        return 'Over 50'
      else: return 'N/A'

age_udf = udf(udf_multiple)
df2=df2.withColumn("age_group", age_udf('age'))


# In[185]:


#type of contact and deposit
j_df = pd.DataFrame()

j_df['yes'] = df1[df1['y'] == 'yes']['contact'].value_counts()
j_df['no'] = df1[df1['y'] == 'no']['contact'].value_counts()

j_df.plot.bar(title = 'Type of contact and deposit')


# In[ ]:





# In[ ]:





# In[175]:





# In[20]:


#Create a new column - pdays_contact

def udf_pdays(pdays):
      if (pdays == 999):
        return '0'
      else: return '1'

pdays_udf = udf(udf_pdays)
df2=df2.withColumn("pdays_contact", pdays_udf('pdays'))


# In[31]:


df2=df2.drop("age","pdays")
df2.printSchema()
df2.toPandas().head(5)


# In[99]:


df3=df2

job_indexer = StringIndexer(inputCol="job", outputCol="job_index")
marital_indexer = StringIndexer(inputCol="marital", outputCol="marital_index")
education_indexer = StringIndexer(inputCol="education", outputCol="education_index")
default_indexer = StringIndexer(inputCol="default", outputCol="default_index")
housing_indexer = StringIndexer(inputCol="housing", outputCol="housing_index")
loan_indexer = StringIndexer(inputCol="loan", outputCol="loan_index")
contact_indexer = StringIndexer(inputCol="contact", outputCol="contact_index")
month_indexer = StringIndexer(inputCol="month", outputCol="month_index")
day_of_week_indexer = StringIndexer(inputCol="day_of_week", outputCol="day_of_week_index")
poutcome_indexer = StringIndexer(inputCol="poutcome", outputCol="poutcome_index")
y_indexer = StringIndexer(inputCol="y", outputCol="label")
age_indexer = StringIndexer(inputCol="age_group", outputCol="age_group_index")
pdays_contact_indexer = StringIndexer(inputCol="pdays_contact", outputCol="pdays_contact_index")

pipeline = Pipeline(stages=[job_indexer, marital_indexer,education_indexer,default_indexer,housing_indexer,loan_indexer,contact_indexer,month_indexer,day_of_week_indexer, poutcome_indexer,y_indexer, age_indexer,pdays_contact_indexer])
index_df = pipeline.fit(df2).transform(df2)
index_df.toPandas().tail(10)


# In[100]:


## One-hot encoder


job_encoder = OneHotEncoder(inputCol="job_index", outputCol="job_encoded")
marital_encoder = OneHotEncoder(inputCol="marital_index", outputCol="marital_encoded")
education_encoder = OneHotEncoder(inputCol="education_index", outputCol="education_encoded")
default_encoder = OneHotEncoder(inputCol="default_index", outputCol="default_encoded")
housing_encoder = OneHotEncoder(inputCol="housing_index", outputCol="housing_encoded")
loan_encoder = OneHotEncoder(inputCol="loan_index", outputCol= "loan_encoded")
contact_encoder = OneHotEncoder(inputCol="contact_index", outputCol= "contact_encoded")
month_encoder = OneHotEncoder(inputCol="month_index", outputCol= "month_encoded")
day_of_week_encoder = OneHotEncoder(inputCol="day_of_week_index", outputCol= "day_of_week_encoded")
poutcome_encoder = OneHotEncoder(inputCol="poutcome_index", outputCol= "poutcome_encoded")
y_encoder = OneHotEncoder(inputCol="label", outputCol= "y_encoded")
age_encoder = OneHotEncoder(inputCol="age_group_index", outputCol= "age_encoded")
pdays_encoder = OneHotEncoder(inputCol="pdays_contact_index", outputCol= "pdays_encoded")

pipeline = Pipeline(stages=[job_encoder,marital_encoder,education_encoder,default_encoder,housing_encoder,                            loan_encoder,contact_encoder,month_encoder,day_of_week_encoder,                            poutcome_encoder,y_encoder,age_encoder,pdays_encoder ])
encoder_model = pipeline.fit(index_df).transform(index_df)
encoder_model.toPandas().head(10)


# In[101]:


encoder_model.printSchema()


# In[102]:



## Vector assembler

import pandas as pd
pd.set_option('display.max_colwidth', 80)
pd.set_option("display.max_columns", 12)

assembler = VectorAssembler()         .setInputCols (["age_encoded","job_encoded","marital_encoded","education_encoded",
                         "default_encoded","housing_encoded","loan_encoded","contact_encoded",\
                         "month_encoded","day_of_week_encoded","duration",\
                         "campaign","previous","poutcome_encoded",\
                         "empvarrate", "conspriceidx","consconfidx","euribor3m","nremployed","pdays_encoded"])\
         .setOutputCol ("vectorized_features")
        
# In case of missing you can skip the invalid ones
assembler_df=assembler.setHandleInvalid("skip").transform(encoder_model)
assembler_df.toPandas().head()


# In[164]:


## Standard Scaler

scaler = StandardScaler()         .setInputCol ("vectorized_features")         .setOutputCol ("features")
        
scaler_model=scaler.fit(assembler_df)
scaler_df=scaler_model.transform(assembler_df)
scaler_df.select("vectorized_features","features").toPandas().head(5)


# In[104]:


# weighted col since the target variable is unbalanced
balancingRatio = scaler_df.filter(col('label') == 1).count() / scaler_df.count()
calculateWeights = udf(lambda x: 1 * balancingRatio if x == 0 else (1 * (1.0 - balancingRatio)), DoubleType())

weightedDataset = scaler_df.withColumn("classWeightCol", calculateWeights('label'))

weightedDataset.toPandas().head(10)


# In[106]:


train, test = weightedDataset.randomSplit([0.8, 0.2], seed = 2018)
print("Training Dataset Count: " + str(train.count()))
print("Test Dataset Count: " + str(test.count()))


# In[107]:


train.groupby("label").count().show()


# In[ ]:


## Logistic Regression Model


# In[199]:


from pyspark.ml.classification import LogisticRegression
lr = LogisticRegression(featuresCol = 'features', labelCol = 'label', weightCol="classWeightCol",maxIter=5)
lrModel = lr.fit(train)
path = "C:/Users/dsiri/OneDrive/Desktop/BAN 5753/Exercises/miniproject2/LR"
lrModel.save(path)
predictions = lrModel.transform(test)
#predictions_train = lrModel.transform(train)
predictions.select('label', 'features',  'rawPrediction', 'prediction', 'probability').toPandas().head(5)


# In[ ]:


## Logistic Regression Model- Evaluation


# In[132]:


accuracy = predictions.filter(predictions.label == predictions.prediction).count() / float(predictions.count())
print("LR Model Accuracy : ",accuracy)


# In[144]:


trainingSummary = lrModel.summary
roc = trainingSummary.roc.toPandas()
plt.plot(roc['FPR'],roc['TPR'])
plt.ylabel('False Positive Rate')
plt.xlabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
print('Training set areaUnderROC: ' + str(trainingSummary.areaUnderROC))


# In[134]:


from pyspark.ml.evaluation import BinaryClassificationEvaluator
evaluator = BinaryClassificationEvaluator()
print('Test Area Under ROC', evaluator.evaluate(predictions))


# In[135]:


class_names=[1.0,0.0]
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[136]:


y_true = predictions.select("label")
y_true = y_true.toPandas()

y_pred = predictions.select("prediction")
y_pred = y_pred.toPandas()

cnf_matrix = confusion_matrix(y_true, y_pred,labels=class_names)
#cnf_matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix')
plt.show()


# In[155]:


#Best Model Feature Weights
weights = lrModel.coefficients
weights = [(float(w),) for w in weights]
weightsDF = sqlContext.createDataFrame(weights, ["Feature Weight"])
weightsDF.toPandas()


# In[ ]:





# In[ ]:


def ExtractFeatureImp(featureImp, dataset, featuresCol):
    list_extract = []
    for i in dataset.schema[featuresCol].metadata["ml_attr"]["attrs"]:
        list_extract = list_extract + dataset.schema[featuresCol].metadata["ml_attr"]["attrs"][i]
    varlist = pd.DataFrame(list_extract)
    varlist['score'] = varlist['idx'].apply(lambda x: featureImp[x])
    return(varlist.sort_values('score', ascending = False))


# In[ ]:


## Decision Tree Model


# In[113]:



from pyspark.ml.classification import DecisionTreeClassifier
dt = DecisionTreeClassifier(featuresCol = 'features', labelCol = 'label',weightCol="classWeightCol", maxDepth = 3)
dtModel = dt.fit(train)
predictions = dtModel.transform(test)
predictions.select('label', 'features',  'rawPrediction', 'prediction', 'probability').show(10)


# In[ ]:


## Decision Tree Model Evaluation


# In[114]:


accuracy = predictions.filter(predictions.label == predictions.prediction).count() / float(predictions.count())
print("DT Model Accuracy : ",accuracy)


# In[116]:


from pyspark.ml.evaluation import BinaryClassificationEvaluator
evaluator = BinaryClassificationEvaluator()
print('Test Area Under ROC', evaluator.evaluate(predictions))


# In[ ]:


## Random Forest Model 


# In[167]:


from pyspark.ml.classification import RandomForestClassifier
rf = RandomForestClassifier(featuresCol = 'features', labelCol = 'label', weightCol="classWeightCol")
rfModel = rf.fit(train)
predictions = rfModel.transform(test)
predictions.select('label', 'features',  'rawPrediction', 'prediction', 'probability').show(10)


# In[168]:


rfModel.featureImportances


# In[ ]:


## Random Forest Model Evaluation


# In[118]:


accuracy = predictions.filter(predictions.label == predictions.prediction).count() / float(predictions.count())
print("RF Model Accuracy : ",accuracy)


# In[119]:


from pyspark.ml.evaluation import BinaryClassificationEvaluator
evaluator = BinaryClassificationEvaluator()
print('Test Area Under ROC', evaluator.evaluate(predictions))


# In[80]:


y_true = predictions.select("y_index")
y_true = y_true.toPandas()

y_pred = predictions.select("prediction")
y_pred = y_pred.toPandas()

cnf_matrix = confusion_matrix(y_true, y_pred,labels=class_names)
#cnf_matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix')
plt.show()


# In[ ]:


## k-Means Model


# In[122]:


from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
silhouette_score=[]
evaluator = ClusteringEvaluator(predictionCol='prediction', featuresCol='features',                                 metricName='silhouette', distanceMeasure='squaredEuclidean')
for i in range(2,10):
    
    KMeans_algo=KMeans(featuresCol='features', k=i)
    
    KMeans_fit=KMeans_algo.fit(train)
    
    KMeans_transform=KMeans_fit.transform(train)
    
    
    
    score=evaluator.evaluate(KMeans_transform)
    
    silhouette_score.append(score)
    
    print("Silhouette Score:",score)


# In[123]:


#Visualizing the silhouette scores in a plot
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1,1, figsize =(8,6))
ax.plot(range(2,10),silhouette_score)
ax.set_xlabel('k')
ax.set_ylabel('cost')
plt.show()


# In[126]:


KMeans_=KMeans(featuresCol='features', k=2) 
KMeans_Model=KMeans_.fit(train)
predictions=KMeans_Model.transform(test)

predictions.show(10)


# In[ ]:


## kMeans Model Evaluation


# In[127]:


accuracy = predictions.filter(predictions.label == predictions.prediction).count() / float(predictions.count())
print("kMeans Model Accuracy : ",accuracy)


# In[87]:


y_true = KMeans_Assignments.select("label")
y_true = y_true.toPandas()

y_pred = KMeans_Assignments.select("prediction")
y_pred = y_pred.toPandas()

cnf_matrix = confusion_matrix(y_true, y_pred,labels=class_names)
#cnf_matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix')
plt.show()


# In[88]:


accuracy = KMeans_Assignments.filter(KMeans_Assignments.y_index == KMeans_Assignments.prediction).count() / float(KMeans_Assignments.count())
print("Accuracy : ",accuracy)


# In[83]:


from pyspark.ml.feature import PCA as PCAml
pca = PCAml(k=2, inputCol="features", outputCol="pca")
pca_model = pca.fit(scaler_df)
pca_transformed = pca_model.transform(scaler_df)
import numpy as np
x_pca = np.array(pca_transformed.rdd.map(lambda row: row.pca).collect())


# In[84]:


cluster_assignment = np.array(KMeans_Assignments.rdd.map(lambda row: row.prediction).collect()).reshape(-1,1)
import seaborn as sns
import matplotlib.pyplot as plt

pca_data = np.hstack((x_pca,cluster_assignment))

pca_df = pd.DataFrame(data=pca_data, columns=("1st_principal", "2nd_principal","cluster_assignment"))
sns.FacetGrid(pca_df,hue="cluster_assignment", height=6).map(plt.scatter, '1st_principal', '2nd_principal' ).add_legend()

plt.show()


# In[ ]:





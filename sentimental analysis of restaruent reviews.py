#!/usr/bin/env python
# coding: utf-8

# In[1]:


#  importing required primary packages
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


# reading the csv file
df = pd.read_csv('Restaurant_Reviews.tsv',sep='\t')


# In[3]:


df.head()


# In[4]:


# getting the info  of the data set
df.info()


# In[5]:


# removing the repeated or duplicate values
df = df.drop_duplicates(keep = 'last')
df


# In[6]:


df.info()


# In[7]:


df['Review'][900]


# In[8]:


df['Liked'][500]


# In[9]:


# finding number of unique values
data_frame=df['Liked'].value_counts().to_frame()
data_frame


# In[10]:


# pandas plotting
data_frame['Liked'].value_counts().plot(kind='bar')


# In[11]:


# pandas plotting
data_frame['Liked'].value_counts().plot(kind='bar')


# In[12]:


#  data visualization using matplotlib
# this piechart is showing the all 997 values with their respective values (0&1)
import matplotlib.pyplot as plt
x = df['Review'].values
y = df['Liked'].values
plt.pie(y)
plt.show()


# In[13]:


# rows - 996 and columns - 2
df.shape


# In[14]:


# splitting the date into training data and testing data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0)


# In[15]:


x_train.shape


# In[16]:


x_test.shape


# In[17]:


# using countvectorizer coverting the text data to numerical values
# that is it involves removing of stop words and counting the occurence of each useful word and storing them
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(stop_words='english')
x_train_vect = vect.fit_transform(x_train)
x_test_vect = vect.transform(x_test)


# In[18]:


# using countvectorizer coverting the text data to numerical values
# that is it involves removing of stop words and counting the occurence of each useful word and storing them
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(stop_words='english')
x_train_vect = vect.fit_transform(x_train)


# In[19]:


x_test_vect = vect.transform(x_test)


# In[20]:


x_train_vect.toarray()


# In[21]:


x_test_vect.toarray()


# In[22]:


x_test_vect.shape


# In[23]:


#METHOD 1
# using support vector classifier
from sklearn.svm import SVC
model1 = SVC()
model1.fit(x_train_vect,y_train)


# In[24]:


# predicted values
y_pred1 = model1.predict(x_test_vect)
y_pred1


# In[25]:


# finding accuracy of the model
from sklearn.metrics import accuracy_score
accuracy_score(y_pred1,y_test)


# In[26]:


# METHOD 2
# combining countvectorizer and support vector classifier using pipeline
from sklearn.pipeline import make_pipeline
model2 = make_pipeline(CountVectorizer(),SVC())
model2.fit(x_train,y_train)


# In[27]:


# predicted values
y_pred2 = model2.predict(x_test)
y_pred2


# In[28]:


# finding accuracy of the model
accuracy_score(y_pred2,y_test)


# In[29]:


# METHOD 3
# using Naive Bayes
from sklearn.naive_bayes import MultinomialNB
model3 = MultinomialNB()
model3.fit(x_train_vect,y_train)


# In[30]:


# predicted values
y_pred3 = model3.predict(x_test_vect)
y_pred3


# In[31]:


# finding accuracy of the model
accuracy_score(y_pred3,y_test)


# In[32]:


# METHOD 4
# combining count vectorizer and Naive Bayes using pipeline
from sklearn.pipeline import make_pipeline
model4 = make_pipeline(CountVectorizer(),MultinomialNB())
model4.fit(x_train,y_train)


# In[33]:


# predicted values
y_pred4 = model4.predict(x_test)
y_pred4


# In[34]:


#  finding accuracy of the model
accuracy_score(y_pred4,y_test)


# In[35]:


# THE ACCURACIES OF ALL MODELS CREATED
import pandas as pd
accuracy = [['SVC',0.7269076305220884],['SVC Pipeline',0.8152610441767069]
,['MultinomialNB',0.7469879518072289],['MultinomialNB Pipeline',0.7791164658634538]]
accuracy_df = pd.DataFrame(accuracy,columns = ['Model','Accuracy'])
accuracy_df = accuracy_df.style.set_properties(**{'text-align':'left'})
accuracy_df = accuracy_df.set_table_styles([dict(selector = 'th',props=[('text-align','center')])])
accuracy_df


# In[36]:


# using joblib for saving the model having highest accuracy
# pickling
import joblib
joblib.dump(model2,'restaurant_reviews')


# In[37]:


# loading the model
import joblib
reload_model = joblib.load('restaurant_reviews')


# In[38]:


# predicting the values with highest accuracy model after pickling
reload_model.predict(['good'])


# In[39]:


reload_model.predict(['not so good'])


# In[40]:


reload_model.predict(['not worth'])


# In[41]:


# creating the webapp using streamlit
# installing streamlit
get_ipython().system('pip install streamlit --quiet')


# In[ ]:


# using file handiling in python
# write mode for writing the reviews
# button - on click displays the output as 0 or 1
%%writefile reviews.py
import streamlit as st
import joblib
from PIL import Image
image = Image.open('OLP.jpeg')
st.image(image, caption='Artificial Intelligence and Machine learning')
print()
st.title('SENTIMENT ANALYSIS')
reload_model = joblib.load('restaurant_reviews')
ip = st.text_input("Give your expensive review : ")
op = reload_model.predict([ip])
if st.button('PREDICT'):
  st.text('OUTPUT ')
  st.text('1 - POSITIVE REVIEW')
  st.text('0 - NEGATIVE REVIEW')
  st.title(op[0])


# In[ ]:


# using local tunnel tool for creating a URL to webapp
get_ipython().system('streamlit run reviews.py & npx localtunnel --port 8501')


#!/usr/bin/env python
# coding: utf-8

# # Project Background
# 
# In fall 2020 the film, The Social Dilemma, released on Netflix to millions of viewers around the globe. This powerful documentary explores how social media has damaged society by manipulating and addicting its users. The spread of social media has caused profound changes to our mental health and has lead many to question previously universally held truths. Such an eye opening film has caught the attention of researchers who are interestead in analyzing the reaction of the film's audience. Film producers are also interested in audience reaction as it will be used to position any follow up work to future audiences. Using a dataset that extracted thousands of tweets in response to the #TheSocialDilemma hashtag, the goal is to identify any interesting trends and present findings to a team of researchers and film producers. Questions to consider are 1) what is the general mood or subjective opinions the audience has after viewing this film? 2) what abstract topics occur and can be gleamed from these tweets? 
# 
# Dataset Source: https://www.kaggle.com/kaushiksuresh147/the-social-dilemma-tweets

# ## Objectives
# 
# 
# 
# *   Perform detailed exploratory data analysis
# *   Create a Topic Model and analyze results
# *   Build a model that captures audience sentiment
# *   Issue Conclusions
# 

# ## Dataset Description
# 
# The dataset was extracted using TwitterAPI, and consits of 10,526 Tweets from around the globe (14 columns, 20,067 rows). 
# 
# 
# 
# 
# 
# 
# 
# ### Input Variables
# 
# **User Name** - The name of the user, as they’ve defined it.
# 
# **User Location** - The user-defined location for this account’s profile.
# 
# **User Description** - The user-defined UTF-8 string describing their account
# 
# **User Created** - Time and date, when the account was created.
# 
# **User Followers** - The number of followers an account currently has.
# 
# **User Friends** - The number of friends an account currently has.
# 
# **User Favorites** - The number of favorites a account currently has 
# 
# **User Verified** - When true, indicates that the user has a verified account
# 
# **Date** - UTC time and date when the Tweet was created
# 
# **Text** - The actual UTF-8 text of the Tweet
# 
# **Hashtags** - All the other hashtags posted in the tweet along with #TheSocialDilemma
# 
# **Source** - 	Utility used to post the Tweet, Tweets from the Twitter website have a source value - web
# 
# **Is Retweet** - Indicates whether this Tweet has been Retweeted by the authenticating user.
# 
# 
# 
# 
# 
# 
# 
# 
# ### Target Variable 
# 
# **Sentiment** - Indicates the sentiment of the tweet, consists of three categories: Positive, neutral, and negative
# 
# 
# 
# 

# ## Project Setup

# In[1]:


import numpy as np
import pandas as pd
#pd.set_option('display.max_rows', None)
#pd.set_option('display.max_columns', None)


# In[4]:


import os
os.getcwd()


# In[7]:


os.chdir('C:\\Users\nsold\OneDrive\Desktop\Advanced Data Mining')


# In[ ]:


# import using Google Colab
from google.colab import files
uploaded = files.upload()


# In[2]:


social_d = pd.read_csv('TheSocialDilemma.csv')


# In[3]:


social_d.head()


# In[4]:


social_d['user_created']= social_d['user_created'].apply(pd.to_datetime)


# ## Exploratory Data Analysis

# In[ ]:


pip install -U dataprep


# In[ ]:


# Overall Analysis of the dataset
from dataprep.eda import create_report
from dataprep.eda import plot, plot_correlation, plot_missing
create_report(social_d)


# ### Observations from Exploratory Data Analysis
# 
# 
# *   The top 3 countries that tweets are coming from are US, India, UK
# *   All tweets are original, (no retweets)
# *   User accounts have been created with a date range of July 2006 to October 2020, with roughly 3% created in Q3 2020
# *   User followers range from 0 to 15624426
# *   User friends range from 0 to 	288625
# *   Only 3.54% of user accounts are verified
# *   60.92% of the hashtags are 'TheSocialDilemma' 
# *   47.40% of the tweets are classified as positive, 34.79% as neutral, and 17.80% as negative
# *   Missing values found in User Location, User Description, and Hastags
# 
# 
# 

# ## Topic Model Preprocessing
# 
# The thousands of tweets found in this dataset are in response to the SocialDilemma hashtag, but what topics stand out? A topic model will be created to find hidden semantic structure in these tweets.

# In[ ]:


get_ipython().system('pip install pycaret-nightly')


# In[5]:


# Setting up environment for Topic Modeling
from pycaret.nlp import *
import nltk
import matplotlib.pyplot as plt
#import pandas_profiling
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from nltk.corpus import stopwords


# In[6]:


nltk.download('stopwords')
nltk.download('wordnet')


# ### Stopwords
# 
# In addition to the common stopwords that must be removed for the analysis there are words having to due with the title of the film and platforms that do not provide further insight.

# In[7]:


stop_words = stopwords.words('english')


# In[48]:


print(stop_words)


# In[30]:


words_to_remove = ['https', 'co', 'watch', 'netflix', 'social', 'dilemma', 'documentary']
stop_words.extend(words_to_remove)


# ## Topic Model Fitting

# In[31]:


social_tnlp = setup(data = social_d, target= 'text', custom_stopwords=stop_words, session_id= 123)


# In[49]:


lda = create_model('lda', num_topics= 2)


# In[50]:


print(lda)


# In[51]:


lda_results = assign_model(lda)
lda_results.head()


# ## Model and Corpus Analysis

# In[52]:


plot_model(plot = 'frequency')


# In[46]:


plot_model(plot = 'distribution')


# In[34]:


plot_model(plot = 'bigram')


# In[35]:


plot_model(plot = 'trigram')


# In[36]:


plot_model(plot = 'wordcloud')


# In[38]:


plot_model(lda, plot = 'frequency', topic_num = 'Topic 1')


# In[44]:


plot_model(lda, plot = 'frequency', topic_num = 'Topic 0')


# In[54]:


plot_model(lda, plot = 'topic_distribution')


# In[40]:


plot_model(lda, plot = 'tsne')


# In[42]:


plot_model(lda, plot = 'umap')


# In[41]:


plot_model(lda, plot = 'topic_model')


# In[ ]:





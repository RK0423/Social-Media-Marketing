#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
import plotly.express as px
from matplotlib import pyplot
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.figure_factory as ff


# In[2]:


df_t=pd.read_csv("C:/Users/ushan/Downloads/tiktok.csv")
df_i=pd.read_csv("C:/Users/ushan/Downloads/instagram.csv")
df_y=pd.read_csv("C:/Users/ushan/Downloads/youtube.csv")


# # Tiktok Dataset Analysis

# In[3]:


df_t.info()


# In[4]:


df_t.describe()


# In[5]:


df_t.head(20)


# In[6]:


df_t.tail(20)


# In[7]:


df_t.isnull().sum()


# In[8]:


df_t.shape


# In[9]:


df_t['Subscribers count'].str[-1].value_counts()


# In[10]:


import re
def convert(x):
    return re.findall('\d+\.?\d*',x)


# In[11]:


def change(df,list1):
    for i in list1:
        df['new'+i]=df[i].apply(convert)
        df['new'+i]=df['new'+i].apply(lambda x: "".join(x))
        df['new'+i]=pd.to_numeric(df['new'+i])
        df['new'+i]=np.where(['M' in j for j in df[i]],df['new'+i]*1000000,
                             np.where(['K' in j1 for j1 in df[i]],df['new'+i]*1000,df['new'+i]))
    return df


# In[12]:


change(df_t,['Subscribers count'])


# In[13]:


df_t.sort_values(by='newSubscribers count',ascending=False,ignore_index=True).iloc[0:10,[1,2]]


# In[14]:


zip_doc=zip(df_t['Subscribers count'],df_t['Views avg.'],df_t['Likes avg'],df_t['Comments avg.'],df_t['Shares avg'])


# In[15]:


subscribers_count= []
views_avg=[]
likes_avg=[]
comments_avg=[]
shares_avg=[]
for subscribers, views, likes, comments, shares in zip_doc:
    if 'K' in subscribers :
        subscrib=subscribers.strip('K')
        subscrib=float(subscrib)*1000
        subscribers_count.append(round(subscrib))
    if 'M' in subscribers :
        subscrib=subscribers.strip('M')
        subscrib=float(subscrib)*1000000
        subscribers_count.append(round(subscrib))
    if 'K' in views :
        view=views.strip('K')
        view=float(view)*1000
        views_avg.append(round(view))
    if 'M' in views :
        view=views.strip('M')
        view=float(view)*1000000
        views_avg.append(round(view))
    if 'K' in likes :
        like=likes.strip('K')
        like=float(like)*1000
        likes_avg.append(round(like))
    if 'M' in likes :
        like=likes.strip('M')
        like=float(like)*1000000
        likes_avg.append(round(like))
    if 'K' in comments :
        comment=comments.strip('K')
        comment=float(comment)*1000
        comments_avg.append(round(comment))
    elif 'M' in comments :
        comment=comments.strip('M')
        comment=float(comment)*1000000
        comments_avg.append(round(comment))
    else:
        comment=float(comment)
        comments_avg.append(round(comment))        
    if 'K' in shares :
        share=shares.strip('K')
        share=float(share)*1000
        shares_avg.append(round(share))
    elif 'M' in shares :
        share=shares.strip('M')
        share=float(share)*1000000
        shares_avg.append(round(share))
    else:
        share=float(share)
        shares_avg.append(round(share)) 
# len(subscribers_count)
# len(views_avg)
# len(likes_avg)
# len(comments_avg)
# len(shares_avg)


# In[16]:


df_t['Subscribers']=subscribers_count
df_t['Views']=views_avg
df_t['Likes']=likes_avg
df_t['Comments']=comments_avg
df_t['Shares']=shares_avg


# In[17]:


sorted_by_subscribers_df=df_t.sort_values(by='Subscribers')
fig = px.histogram(sorted_by_subscribers_df.tail(10), 
                   y="Tiktoker name", 
                   x='Subscribers',
                   text_auto=True,
                   title='Top Most Followed Tiktokers')
fig.show()


# In[18]:


sorted_by_views_df=df_t.sort_values(by='Views')
fig = px.histogram(sorted_by_views_df.tail(10), 
                   y="Tiktoker name", 
                   x='Views',
                   text_auto=True,
                   title='Top Most Viewed Tiktokers')
fig.show()


# In[19]:


sorted_by_likes_df=df_t.sort_values(by='Likes')
fig = px.histogram(sorted_by_likes_df.tail(10), 
                   y="Tiktoker name", 
                   x='Likes',
                   text_auto=True,
                   title='Top Most Liked Tiktokers')
fig.show()


# In[20]:


sorted_by_comments_df=df_t.sort_values(by='Comments')
fig = px.histogram(sorted_by_subscribers_df.tail(), 
                   y="Tiktoker name", 
                   x='Comments',
                   text_auto=True,
                   title='Top Most Commented Tiktokers')
fig.show()


# In[21]:


sorted_by_shares_df=df_t.sort_values(by='Shares')
fig = px.histogram(sorted_by_shares_df.tail(), 
                   y="Tiktoker name", 
                   x='Shares',
                   text_auto=True,
                   title='Top Most Shared Tiktokers')
fig.show()


# In[22]:


fig = px.imshow(round(df_t.corr(),1),
                text_auto=True,
                title='Correlation Heatmap')
fig.show()


# # Instagram Dataset Analysis

# In[23]:


df_i.info


# In[24]:


df_i.describe()


# In[25]:


df_i.head(20)


# In[26]:


df_i.tail(20)


# In[27]:


df_i.isnull().sum()


# In[28]:


df_i.rename({'category_1':'Category','category_2':'category','Audience country(mostly)':'Audience Country'},axis=1,inplace=True)


# In[29]:


df_i


# In[30]:


df_i.drop(labels=['Influencer insta name','category','Authentic engagement\r\n'],axis=1,inplace=True)


# In[31]:


df_i.head(20)


# In[32]:


df_i.dropna()


# In[33]:


df_i.isnull().sum()


# In[34]:


df_i.shape


# In[35]:


df_i.Category.value_counts()


# In[36]:


pd.unique(df_i["Category"])


# In[37]:


df_i.isnull()


# In[38]:


df_i.Category.isnull()


# In[39]:


df_i.Category.isnull().sum().sum()


# In[40]:


ndf_i=df_i.Category.dropna()


# In[41]:


df_i


# In[42]:


ndf_i.head(20)


# In[43]:


li=['Followers','Engagement avg\r\n']


# In[44]:


change(df_i,li)


# In[45]:


print(df_i['Followers'].str[-1].unique())


# In[46]:


df_i['newFollowers']=df_i['newFollowers']/1000000


# In[47]:


df_i.drop(labels=['Engagement avg\r\n','newEngagement avg\r\n'],axis=1,inplace=True)


# In[48]:


df_i.head(5)


# In[49]:


df_i.sort_values(by='newFollowers',ascending=False,ignore_index=True).iloc[0:15,[0,1,3,-1]]


# In[50]:


print("Shape of Dataset:- ",df_i.shape)
df_i.head().style.background_gradient(cmap='YlOrRd')


# In[51]:


plt.title('Top 15 most followed celebrity on instagram')
plt.xlabel('Followers in Million')
sns.barplot(y='instagram name',x='newFollowers',data=df_i.sort_values(by='newFollowers',ascending=False).head(15))


# In[52]:


pallete=['red','green','yellow','salmon','cyan','blue','orange']


# In[53]:


def plot(df):
    plt.figure(figsize=(8,6))
    plt.xlabel('number of times category occured')
    plt.ylabel('Category')
    df['Category'].value_counts().sort_values(ascending=True).plot.barh(color=pallete)


# In[54]:


plot(df_i)


# In[55]:


plt.title('Top 15 most followed celebrity on instagram')
plt.xlabel('Followers in Million')
sns.barplot(y='Audience Country',x='newFollowers',data=df_i.sort_values(by='newFollowers',ascending=False).head(100))


# In[56]:


chart = sns.countplot(x="Audience Country", data=df_i, order = df_i['Audience Country'].value_counts().index)
chart.set_xticklabels(chart.get_xticklabels(), rotation=50);


# # Youtube Dataset Analysis

# In[57]:


df_y


# In[58]:


df_y.info


# In[59]:


df_y.describe()


# In[60]:


df_y.isnull().sum()


# In[61]:


df_y.drop_duplicates(subset=['channel name'],inplace=True)


# In[62]:


plot(df_y)


# ## TOP consumer countries of the influencers content on YOUTUBE

# In[63]:


def plot_c(df):
    plt.figure(figsize=(10,8))
    plt.xlabel('number of times category occured')
    df['Audience Country'].value_counts().sort_values().plot.barh(color=pallete)


# In[64]:


plot_c(df_y)


# ### (TARGET COUNTRY FOR BUISNESS)Checking the demand for categories by Country wise

# In[65]:


def demand(data,category):
    return data[data['Category']==category]['Audience Country'].value_counts().sort_values(ascending=True).plot.barh(color=pallete)


# In[66]:


demand(df_y,'Education')


# In[67]:


df_y.iloc[0:10,[1,2,3]]


# In[68]:


ly=['Followers','avg views', 'avg likes', 'avg comments']


# In[69]:


df_i['newFollowers'].describe()


# In[70]:


df_i['newFollowers'].quantile(0.50)


# ### I am taking 60M as a threshold means for instagram celebrity havning above 60M followers are considerd to be mega celebrity

# In[71]:


df_y.head(50)


# In[ ]:






# coding: utf-8

# In[11]:


import pandas as pd
import numpy as np
import sys
sys.path.append('/usr/local/lib/python2.7/dist-packages')


# In[12]:


df_A = pd.read_csv("classA.csv")
df_B = pd.read_csv("classB.csv")


# In[13]:


#df = df_A


# In[14]:


#df = df_A.set_index(['source', 'target'])
df = df_A
df


# In[15]:


df_A


# In[16]:


relations = ['BEFORE',
             'CONTAINS',
             'OVERLAP',
             'BEGINS-ON',
             'ENDS-ON'
            ]


# In[17]:


invRelations = ['BEFORE_INV',
             'CONTAINS_INV',
             'OVERLAP_INV',
             'ENDS-ON',
             'BEGINS-ON'
            ]


# In[18]:


def invert(relation) :
    return invRelations[relations.index(relation)]
    


# In[19]:


df_B = df_B[(df_B['source'] != 'e6')]
df_B


# In[87]:


def addClassifier(df_B, className) :
    global df
    df[className] = np.nan
    for ix, row in df_B.iterrows() :
        findKey = (row['source'], row['target'])
        if (findKey in df.index) :
            #print findKey, className, row[className]
            df.loc[findKey][className] = row[className]
        else :
            findKey = (row['target'], row['source'])
            if (findKey in df.index) :
                df.loc[findKey][className] = invert(row[className])
            else :
                newRow = df_B[(df_B['source'] == findKey[1]) & (df_B['target'] == findKey[0])]
                newRow.set_index(['source', 'target'], inplace=True)
                print newRow
                df = df.append(newRow)
    print df
    return df


# In[20]:


df_inner = pd.merge(df, 
                    df_B, 
                    left_on=['source', 'target'], right_on=['target','source'],
                    how='inner',
                    suffixes=('', '_y'))[df_B.columns]
df_inner['classB'] = df_inner['classB'].map(lambda x : invert(x))
df_inner


# In[23]:


df


# In[24]:


df_B


# In[21]:


df_B = pd.merge(df_B, 
                    df_inner, 
                    left_on=['source', 'target'], right_on=['target','source'],
                    how='left',
                    suffixes=('', '_y'))
#df_inner['classB'] = df_inner['classB'].map(lambda x : invert(x))
df_B


# In[22]:


df_B = df_B[(pd.isnull(df_B['source_y']))][['source', 'target', 'classB']]
df_B


# In[23]:


# Concatenate df_B_x with df_inner
df_B = pd.concat([df_B, df_inner])
df_B


# In[24]:


df_outer= pd.merge(df.reset_index(), 
                    df_B, 
                    left_on=['source', 'target'], right_on=['source','target'],
                    how='outer',
                    left_index=False,
                    right_index=False,
                    suffixes=('', '_y'))
df_outer


# In[73]:


df_outer.drop('index', axis=1, inplace=True)
df_outer


# In[88]:


df = addClassifier(df_B, "classB")
df


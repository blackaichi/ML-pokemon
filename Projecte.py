
# coding: utf-8

# ### David Hernandez i Eric Casanovas

# # Processar dades

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')
import cufflinks as cf
cf.go_offline()


# In[2]:


combats = pd.read_csv('combats.csv')
pokemons = pd.read_csv('pokemon.csv')


# In[3]:


combats.head()


# In[4]:


pokemons.head()


# In[5]:


pokemons.info()


# In[6]:


combats.info()


# In[7]:


combats.describe()


# In[8]:


pokemons.describe()


# In[9]:


sns.pairplot(pokemons)


# In[10]:


sns.pairplot(combats)


# In[11]:


sns.distplot(pokemons['Generation'])


# In[12]:


sns.heatmap(combats.corr())


# In[13]:


sns.heatmap(pokemons.corr())


# In[14]:


sns.heatmap(pokemons.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# # Hem de treure els NaN i convertir les dades categoriques en numeriques

# In[15]:


print('files totals = ', end=""); print(len(pokemons))


# In[16]:


print('files amb Type2 nul = ', end=""); print(pokemons['Type 2'].isnull().values.sum())


# In[17]:


print('files amb Type 2 no nul = ', end=""); print(len(pokemons)-pokemons['Type 2'].isnull().values.sum())


# In[18]:


sns.heatmap(combats.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[19]:


def changetype(x):
    if pd.isna(x):
        return 'notype'
    else:
        return x
pokemons["Type 2"]=pokemons["Type 2"].apply(changetype)


# In[20]:


pokemons.head()


# In[21]:


sns.heatmap(pokemons.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[22]:


print('files amb Type 2 no nul = ', end=""); print(len(pokemons)-pokemons['Type 2'].isnull().values.sum())


# In[23]:


print('files totals = ', end=""); print(len(pokemons))


# # Observem com ja hem tret tots els NaN i ara observarem les dades mitjançant diferents grafics

# In[24]:


pokemons['Type 1'].iplot(kind='hist',bins=30)


# In[25]:


pokemons['Type 2'].iplot(kind='hist',bins=30)


# In[26]:


a=sns.FacetGrid(data=pokemons,col_wrap=4,col="Type 1");
a=a.map(plt.scatter,"Attack","Speed")
plt.show();


# In[27]:


a=sns.FacetGrid(data=pokemons,col_wrap=4,col="Type 1");
a=a.map(plt.scatter,"Attack","Defense")
plt.show();


# In[28]:


a=sns.FacetGrid(data=pokemons,col_wrap=4,col="Type 2");
a=a.map(plt.scatter,"Attack","Speed")
plt.show();


# In[29]:


a=sns.FacetGrid(data=pokemons,col_wrap=4,col="Type 2");
a=a.map(plt.scatter,"Attack","Defense")
plt.show();


# In[30]:


sns.set_color_codes("pastel")
ax = sns.countplot(x="Generation", hue="Legendary", data=pokemons)
plt.xticks(rotation= 90)
plt.xlabel('Type 1')
plt.ylabel('Total ')
plt.title("Total Pokemon by Generation")


# In[31]:


pokemons2=pokemons.drop(["#","Generation"],axis=1)
sns.factorplot(data=pokemons2,size=7,kind="box",aspect=2);
plt.show();


# # Despres d'observar varies coses sobre les dades les acabarem de tractar per a juntar-ho tot en un únic DataSet

# In[32]:


numberwins = combats.groupby('Winner').count()


# In[33]:


pokemons.set_index('#', inplace=True)


# In[34]:


pokemons['timeswin'] = numberwins['First_pokemon']


# In[35]:


countfirst = combats.groupby('Second_pokemon').count()
countsecond = combats.groupby('First_pokemon').count()


# In[36]:


pokemons['timesfirst'] = countfirst['First_pokemon']
pokemons['timessecond'] = countsecond['Second_pokemon']


# In[37]:


pokemons['%win']= pokemons['timeswin']/(pokemons['timesfirst']+pokemons['timessecond'])


# In[38]:


pokemons.head()


# # Un cop ho tenim tot a un DataSet hem de canviar els valors que no siguin enters o floats a aquests

# In[39]:


type1 = pd.get_dummies(pokemons['Type 1'],drop_first=True)
type2 = pd.get_dummies(pokemons['Type 2'],drop_first=True)


# In[40]:


a = list(type1)
type1.rename(columns={type_pokemon: type_pokemon+'1' for type_pokemon in a}, inplace=True)
a = list(type2)
type2.rename(columns={type_pokemon: type_pokemon+'2' for type_pokemon in a}, inplace=True)


# In[41]:


pokemons.drop(['Type 1','Type 2','Name'],axis=1,inplace=True)


# In[42]:


pokemons = pd.concat([pokemons, type1, type2],axis=1)


# In[43]:


pokemons['Legendary']=pokemons['Legendary'].astype(int)


# In[44]:


pokemons.head()


# # Un cop finalitzat ja podem començar a fer prediccions

# # Linear regression

# In[45]:


pokemons=pokemons.dropna()


# In[46]:


X = pokemons.drop(['%win', 'Generation'], axis=1)
y = pokemons['%win']


# In[47]:


from sklearn.model_selection import train_test_split


# In[48]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)


# In[49]:


from sklearn.linear_model import LinearRegression


# In[50]:


lm = LinearRegression()


# In[51]:


lm.fit(X_train,y_train)


# In[52]:


predictions = lm.predict(X_test)


# In[53]:


plt.scatter(y_test,predictions)


# In[54]:


sns.distplot((y_test-predictions),bins=50);


# In[55]:


from sklearn import metrics


# In[56]:


print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# # Camins (KMeans)

# In[57]:


from sklearn.cluster import KMeans


# In[58]:


camins = KMeans(n_clusters=4)


# In[59]:


camins.fit(pokemons)


# In[60]:


camins.cluster_centers_


# In[61]:


camins.labels_


# # Decision Trees

# In[62]:


pokemons.head()


# In[63]:


pokemons['aprox%win']=0
for x in pokemons.index:
    pokemons.at[x, 'aprox%win']= 0 if pokemons.at[x, '%win'] < 0.1 else 1 if pokemons.at[x, '%win'] < 0.2 else 2 if pokemons.at[x, '%win'] < 0.3 else 3 if pokemons.at[x, '%win'] < 0.4 else 4 if pokemons.at[x, '%win'] < 0.5 else 5 if pokemons.at[x, '%win'] < 0.6 else 6 if pokemons.at[x, '%win'] < 0.7 else 7 if pokemons.at[x, '%win'] < 0.8 else 8 if pokemons.at[x, '%win'] < 0.9 else 10


# In[64]:


from sklearn.model_selection import train_test_split


# In[65]:


X = pokemons.drop(['aprox%win', '%win'],axis=1)
y = pokemons['aprox%win']


# In[66]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)


# In[67]:


from sklearn.tree import DecisionTreeClassifier


# In[68]:


dtree = DecisionTreeClassifier()


# In[69]:


dtree.fit(X_train,y_train)


# In[70]:


predictions = dtree.predict(X_test)


# In[71]:


from sklearn.metrics import classification_report,confusion_matrix


# In[72]:


print(classification_report(y_test,predictions))


# In[73]:


print(confusion_matrix(y_test,predictions))


# # Random Forest

# In[74]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)


# In[75]:


rfc_pred = rfc.predict(X_test)


# In[76]:


print(confusion_matrix(y_test,rfc_pred))


# In[77]:


print(classification_report(y_test,rfc_pred))


# ## FI

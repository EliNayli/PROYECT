
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

get_ipython().magic('matplotlib notebook')


# In[2]:


import random
def get_random_ops(rows=100):
    data = []
    for i in range(0,rows):
        a = random.randint(1,100)
        b = random.randint(1,100)
        suma, resta, multi, div = random.choice([
            [1, 0, 0, 0], # 1 es suma y 0 las demas operaciones
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
   

        ])

        if suma == 1: y = a+b #Y es algo, es lo que le damos al modelo para que aprenda
        if resta == 1 : y = a-b ## estamos usando Y para entrenar al modelo
        if multi == 1 : y = a*b
        if div == 1: y = a/b
    

        data.append({ ##anadiremos a data la estructura que queremos crear
            "a" : a,
            "b" : b,
            "suma" : suma,
            "resta" : resta,
            "multi" : multi,
            "div" : div,
            "y" : round(y,2)
            })

    return data


# In[32]:


##Paso 1 y 2
#data = pd.DataFrame(get_random_ops(25000))
#Paso3
#data = pd.DataFrame(get_random_ops(100000))
#Paso4
data = pd.DataFrame(get_random_ops(1000000))
data[["a" ,"b", "suma" ,"resta", "multi", "div", "y"]].head()


# In[20]:


get_random_ops(rows=3)


# In[33]:


data.hist()


# In[34]:


from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split


# In[35]:


x_train,x_test,y_train,y_test = train_test_split(
    data[["a" ,"b", "suma" ,"resta", "multi", "div"]], data["y"],
    test_size = 0.3, random_state = 42    
)


# In[36]:


x_train[:1]


# In[37]:


y_train[:1]


# In[38]:


model = MLPRegressor(
    ##Paso1
    max_iter = 800,
    ##paso2
    hidden_layer_sizes=(100,100,100),
    learning_rate_init = 0.0001,
)
model.fit(x_train,y_train)


# In[39]:


print(x_test.iloc[3000])
print(y_test.iloc[3000])
print(model.predict([x_test.iloc[3000]]))


# In[40]:


predict = model.predict(x_test)
print("Predict: %s" % list(predict[:5]))


# In[41]:


data_check = pd.DataFrame(predict, columns = ["predict"])
data_check["y"] = list(y_test)
data_check.set_index(["y"], drop= False, inplace= True)
data_check.sort_values(by=["y"], inplace=True)


# In[42]:


data_check.head()


# In[172]:


##Paso1
data_check.plot()


# In[104]:


##Paso1
data_check.plot()


# In[118]:


#PruebaFinal
data_check.plot()


# In[18]:


#Prueba2
data_check.plot()


# In[31]:


#Prueba3
data_check.plot()


# In[44]:


#PruebaFinal
data_check.plot()


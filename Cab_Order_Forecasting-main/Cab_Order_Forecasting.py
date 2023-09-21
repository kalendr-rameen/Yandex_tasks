#!/usr/bin/env python
# coding: utf-8

# <h1>Содержание<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Подготовка" data-toc-modified-id="Подготовка-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Подготовка</a></span></li><li><span><a href="#Анализ" data-toc-modified-id="Анализ-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Анализ</a></span></li><li><span><a href="#Обучение" data-toc-modified-id="Обучение-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Обучение</a></span></li><li><span><a href="#Тестирование" data-toc-modified-id="Тестирование-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Тестирование</a></span></li><li><span><a href="#Чек-лист-проверки" data-toc-modified-id="Чек-лист-проверки-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Чек-лист проверки</a></span></li></ul></div>

# #  Прогнозирование заказов такси

# Компания «Чётенькое такси» собрала исторические данные о заказах такси в аэропортах. Чтобы привлекать больше водителей в период пиковой нагрузки, нужно спрогнозировать количество заказов такси на следующий час. Постройте модель для такого предсказания.
# 
# Значение метрики *RMSE* на тестовой выборке должно быть не больше 48.
# 
# Вам нужно:
# 
# 1. Загрузить данные и выполнить их ресемплирование по одному часу.
# 2. Проанализировать данные.
# 3. Обучить разные модели с различными гиперпараметрами. Сделать тестовую выборку размером 10% от исходных данных.
# 4. Проверить данные на тестовой выборке и сделать выводы.
# 
# 
# Данные лежат в файле `taxi.csv`. Количество заказов находится в столбце `num_orders` (от англ. *number of orders*, «число заказов»).

# ## Подготовка

# In[1]:


import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
from collections import Counter
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import time
from lightgbm import LGBMRegressor
import lightgbm as lgb
from catboost import CatBoostRegressor
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.stattools import acf
from itertools import product
from tqdm import tqdm_notebook


# In[2]:


df = pd.read_csv('/datasets/taxi.csv', parse_dates=[0], index_col=0)


# In[3]:


df.plot()


# **Из данного графика ничего сильно не понятно, поэтому стоит ресемлировать данные по часу таким образом сократить кол-во наблюдений**

# In[4]:


df_resample = df.resample('1H').mean()


# In[5]:


df_resample = df.resample('1H').sum()


# In[88]:


df_resample


# ## Анализ

# Чтобы проанализировать данные выполним декомпозицию данных. 

# In[8]:


d = seasonal_decompose(df_resample)


# In[9]:


d.trend


# In[10]:


plt.figure(figsize=(10,15))
plt.subplot(311)
d.trend.plot(ax=plt.gca())
plt.title('Trend')
plt.subplot(312)
# < напишите код здесь >
d.seasonal.plot(ax=plt.gca())
plt.title('Seasonality')
plt.subplot(313)
# < напишите код здесь >
d.resid.plot(ax=plt.gca())
plt.title('Residuals')
plt.tight_layout()


# **Рассмотрим получше на сезонность**

# In[11]:


d.seasonal.iloc[100:150].plot()


# In[12]:


d.seasonal['2018-03-12':'2018-03-26'].plot()


# In[13]:


d.seasonal['2018-03':'2018-04'].plot()


# **Видно что присутствует `ежедневная сезонность` с возникающими пиками возникающие достигающие пиков в 00:00 и около 17:00. Рассмотрим данные за неделю с 2018-03-12 : 2018-03-18, а также за две недели** 

# In[14]:


df_resample['2018-03-12':'2018-03-18'].plot()


# In[15]:


df_resample['2018-03-12':'2018-03-25'].plot()


# In[16]:


df_resample['2018-04-01':'2018-04-15'].plot()


# In[17]:


df_resample['2018-04-06':'2018-04-20'].plot()


# # **Общий вывод**
# 
# Видно что присутсвует тенденция к росту кол-ва заказов такси, также наблюдается отчётливая сезонность в данных

# По нашим данным  можно скзазать что присутсвует внутрисуточная сезонность потому что видно что есть пики возникающие в различные периоды дня, а вот присутвие остальным видов сезонности как мне кажется отсутсвует<br>
# 

# ## Обучение

# Сделает фичи для наших данных чтобы мы могли обучить модель

# In[18]:


df_resample['num_orders']


# In[19]:


def make_features(data, max_lag, rolling_mean_size):
    data['year'] = data.index.year
    data['month'] = data.index.month
    data['day'] = data.index.day
    data['dayofweek'] = data.index.dayofweek
    data['hour'] = data.index.hour
    
    for lag in range(1, max_lag + 1):
        data['lag_{}'.format(lag)] = data['num_orders'].shift(lag)

    # < напишите код здесь >
    data['rolling_mean'] = data['num_orders'].shift().rolling(rolling_mean_size).mean()


# In[20]:


def make_features(data, max_lag, rolling_mean_size):
    data['dayofweek'] = data.index.dayofweek
    data['hour'] = data.index.hour
    
    for lag in range(1, max_lag + 1):
        data['lag_{}'.format(lag)] = data['num_orders'].shift(lag)

    # < напишите код здесь >
    data['rolling_mean'] = data['num_orders'].shift().rolling(rolling_mean_size).mean()


# In[22]:


make_features(df_resample, 1, 24)


# In[23]:


train, oth = train_test_split(df_resample, test_size=0.2, shuffle=False)


# In[24]:


test, valid = train_test_split(oth, test_size = 0.5, shuffle=False)


# In[25]:


features_train = train.drop(columns='num_orders', axis=1)
target_train = train['num_orders']


# In[26]:


features_test = test.drop(columns='num_orders', axis=1)
target_test = test['num_orders']


# In[27]:


features_valid = valid.drop(columns='num_orders', axis=1)
target_valid = valid['num_orders']


# In[28]:


features_train


# In[29]:


model = CatBoostRegressor(iterations=2000, learning_rate=0.05, depth=10, loss_function = 'MAE', eval_metric = 'RMSE')


# In[30]:


model.fit(features_train, target_train, use_best_model=True, silent=False, eval_set=(features_valid, target_valid))


# In[31]:


mse_catboost = mean_squared_error(target_test, model.predict(features_test))
rmse_catboost = mse_catboost ** 0.5


# In[32]:


rmse_catboost


# **Проверим вторую модель на наших данных**

# In[33]:


params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'l2', 'l1'},
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0,
    "num_iterations": 1000,
    "n_estimators": 10000
}


# In[34]:


lgb_train = lgb.Dataset(features_train, target_train)
lgb_eval = lgb.Dataset(features_test, target_test, reference=lgb_train)


# In[35]:


start_time = time.time()
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=20,
                valid_sets=lgb_eval,
                early_stopping_rounds=5)
lgbm_exec_time = time.time() - start_time


# In[36]:


y_pred = gbm.predict(features_test, num_iteration=gbm.best_iteration)


# In[37]:


mse_lgbm = mean_squared_error(target_test, y_pred)
rmse_lgbm = mse_lgbm ** 0.5
rmse_lgbm


# **Теперь запустим модель ARIMA для наших данных**

# In[38]:


df_resample


# In[39]:


plot_pacf(df_resample['num_orders']);
plot_acf(df_resample['num_orders']);


# **По характеру полученных графиков можно заметить что данные скорее похожи на стационарные, при этом видны пики на 24 лаге, возможно что это некая `"месячная корреляция"` данных?**

# **Проверим на стационарность тестом Дикей-Фулера**

# In[40]:


ad_fuller_result = adfuller(df_resample['num_orders'])
print(f'ADF Statistic: {ad_fuller_result[0]}')
print(f'p-value: {ad_fuller_result[1]}')


# **p-value < 0.05 => Отклоняем H0 о нестационарности, следовательно приводить к стационарной форме для проведения теста не требуется**

# **Затем с помощью Akaike’s Information Criterion (AIC) беремерём все значения p,d,q, чтобы найти модель с наименьшим AIC**

# In[41]:


def optimize_ARIMA(order_list, exog):
    """
        Return dataframe with parameters and corresponding AIC
        
        order_list - list with (p, d, q) tuples
        exog - the exogenous variable
    """
    
    results = []
    
    for order in tqdm_notebook(order_list):
        try: 
            model = SARIMAX(exog, order=order).fit(disp=-1)
        except:
            continue
            
        aic = model.aic
        results.append([order, model.aic])
        
    result_df = pd.DataFrame(results)
    result_df.columns = ['(p, d, q)', 'AIC']
    #Sort in ascending order, lower AIC is better
    result_df = result_df.sort_values(by='AIC', ascending=True).reset_index(drop=True)
    
    return result_df


# In[42]:


ps = range(22, 26, 1)
d = 1
qs = range(22, 26, 1)


# In[43]:


parameters = product(ps, qs)
parameters_list = list(parameters)


# In[62]:


order_list = []
for each in parameters_list:
    each = list(each)
    each.insert(1, 1)
    each = tuple(each)
    order_list.append(each)
    
result_df = optimize_ARIMA(order_list, exog=df_resample['num_orders'])
result_df


# In[63]:


result_df.iloc[0,0]


# In[64]:


result_df.iloc[result_df['AIC'].idxmin()]


# In[65]:


train, test = train_test_split(df_resample, test_size=0.1, shuffle=False)


# In[69]:


best_model = SARIMAX(df_resample['num_orders'], order=(24,1,25)).fit()
print(best_model.summary())


# In[71]:


best_model.resid


# In[72]:


ljung_box, p_value = acorr_ljungbox(best_model.resid)

print(f'Ljung-Box test: {ljung_box[:10]}')
print(f'p-value: {p_value[:10]}')


# In[73]:


plot_pacf(best_model.resid);
plot_acf(best_model.resid);


# In[74]:


df_resample.iloc[4300:4401]['num_orders']


# In[75]:


best_model.predict(4300,4400)


# In[76]:


target_test.index.day


# In[77]:


a = mean_squared_error(df_resample.iloc[4300:4401]['num_orders'], best_model.predict(4300,4400))


# In[78]:


a ** 0.5


# ## Тестирование

# **Первая модель**

# In[107]:


list_of_tuples = list(zip(target_test,model.predict(features_test)))


# In[111]:


model1_data = pd.DataFrame(list_of_tuples,
                  columns = ['true', 'predict'], index = target_test.index)


# In[182]:


plt.figure(figsize=(10,15))
model1_data.loc['2018-08-01':'2018-08-09'].plot(figsize=(15,5), ylabel='Число заказов', title='Результат Catboost(7дней)', xlabel = 'Дата и время')


# **Вторая модель**

# In[118]:


y_pred = gbm.predict(features_test, num_iteration=gbm.best_iteration)


# In[119]:


list_of_tuples = list(zip(target_test,y_pred))


# In[120]:


model2_data = pd.DataFrame(list_of_tuples,
                  columns = ['true', 'predict'], index = target_test.index)


# In[181]:


model2_data.loc['2018-08-01':'2018-08-09'].plot(figsize=(15,5), ylabel='Число заказов', title='Результат LGBM(7 дней)', xlabel = 'Дата и время')


# **ARIMA**

# In[139]:


best_model.predict(start = 1000, end = 1300)


# In[140]:


df_resample.iloc[1000:1301]


# In[149]:


list_of_tuples = list(zip(df_resample.iloc[1000:1301]['num_orders'], best_model.predict(start = 1000, end = 1300)))


# In[152]:


model3_data = pd.DataFrame(list_of_tuples,
                  columns = ['true', 'predict'], index = df_resample.iloc[1000:1301].index)


# In[153]:


model3_data


# In[180]:


model3_data.plot(figsize=(15,5), ylabel='Число заказов', title='Результат ARIMA (14 дней)', xlabel = 'Дата и время')


# ![image.png](attachment:image.png)

#!/usr/bin/env python
# coding: utf-8

# # Выбор локации для скважины

# Допустим, вы работаете в добывающей компании «ГлавРосГосНефть». Нужно решить, где бурить новую скважину.
# 
# Вам предоставлены пробы нефти в трёх регионах: в каждом 10 000 месторождений, где измерили качество нефти и объём её запасов. Постройте модель машинного обучения, которая поможет определить регион, где добыча принесёт наибольшую прибыль. Проанализируйте возможную прибыль и риски техникой *Bootstrap.*
# 
# Шаги для выбора локации:
# 
# - В избранном регионе ищут месторождения, для каждого определяют значения признаков;
# - Строят модель и оценивают объём запасов;
# - Выбирают месторождения с самым высокими оценками значений. Количество месторождений зависит от бюджета компании и стоимости разработки одной скважины;
# - Прибыль равна суммарной прибыли отобранных месторождений.

# # 1. Загрузка и подготовка данных

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
from tempfile import mkdtemp
import os
import numpy as np


# In[2]:


data0 = pd.read_csv('/datasets/geo_data_0.csv')
data1 = pd.read_csv('/datasets/geo_data_1.csv')
data2 = pd.read_csv('/datasets/geo_data_2.csv')
print(data0.info())
print()
print(data1.info())
print()
print(data2.info())


# # 2. Обучение и проверка модели

# ### Разобьём данные на обучающую и валидационную выборки в соотношении 75:25

# In[3]:


data0_train, data0_valid = train_test_split(data0, test_size=0.25, random_state=12345)
data1_train, data1_valid = train_test_split(data1, test_size=0.25, random_state=12345)
data2_train, data2_valid = train_test_split(data2, test_size=0.25, random_state=12345)


# Т.к в задание просят не дублировать код, поэтому снизу и практически во всем задание использую globals() для обращения к словарю глобальных переменных и создания новых. Собственно снизу применяю метод reset_index, чтобы можно было делать срезы данных по индексам

# In[4]:


for i in range(3):
    globals()[f'data{i}_train'].reset_index(drop=True, inplace=True)
    globals()[f'data{i}_valid'].reset_index(drop=True, inplace=True)


# ### Обучите модель и сделайте предсказания на валидационной выборке.

# Сохраняю все модели в отдельных путях, чтобы можно было легко к ним обратиться

# In[5]:


save_dir = mkdtemp()


# Делю данные (train, valid) для трёх регионов на target, features

# In[6]:


for i in range(3):
    globals()[f'features_train_{i}'] = globals()[f'data{i}_train'].drop(['product', 'id'], axis=1)
    globals()[f'target_train_{i}'] = globals()[f'data{i}_train']['product']
    globals()[f'features_valid_{i}'] = globals()[f'data{i}_valid'].drop(['product', 'id'], axis=1)
    globals()[f'target_valid_{i}'] = globals()[f'data{i}_valid']['product']


# Обучение моделей и их сохранение по указанному пути

# In[7]:


for i in range(3):
    model = LinearRegression()
    model.fit(globals()[f'features_train_{i}'], globals()[f'target_train_{i}'])
    
    globals()[f'predict_data{i}'] = model.predict(globals()[f'features_valid_{i}'])
    
    globals()[f'filename_model_data{i}'] = os.path.join(save_dir, f'model.data{i}')
    joblib.dump(model, globals()[f'filename_model_data{i}'])


# ### Напечатайте на экране средний запас предсказанного сырья и RMSE модели 

# In[8]:


print("Средний запас предсказанного сырья для 'geo_data_0' =", predict_data0.mean())
print("RMSE модели для предсказаний полученных из данных 'geo_data_0' =",
      mean_squared_error(target_valid_0, predict_data0) ** 0.5)
print()
print("Средний запас предсказанного сырья для 'geo_data_1' =", predict_data1.mean())
print("RMSE модели для предсказаний полученных из данных 'geo_data_1' =",
      mean_squared_error(target_valid_1, predict_data1) ** 0.5)
print()
print("Средний запас предсказанного сырья для 'geo_data_2' =", predict_data2.mean())
print("RMSE модели для предсказаний полученных из данных 'geo_data_2' =",
      mean_squared_error(target_valid_2, predict_data2) ** 0.5)


# ### Анализ результатов

# В целом можно заметить, что модель для `geo_data_1` содержит значительно меньшее значения RMSE модели в отличие от остальных моделей, где RMSE достигает значения порядка 40.0. При этом значения предсказанного сырья для `geo_data1` значительно ниже в отличие от остальных регионов что в целом согласуется с полученными результатами

# # 3. Подготовка к расчёту прибыли

# ### Все ключевые значения для расчётов сохраните в отдельных переменных.

# Кол-во всего выделенных денег и доход с каждой единицы продукта сохраняем в отдельных переменных

# In[10]:


ALL_MONEY = 10 ** 10


# In[11]:


UNIT_INCOME = 450 * (10**3)


# ### Рассчитайте достаточный объём сырья для безубыточной разработки новой скважины. Сравните полученный объём сырья со средним запасом в каждом регионе.

# Мы знаем что у нас выделяют 10 млрд на 200 скважин, также знаем доход от единицы продукта = > нужно выделенные деньги поделить на доход от единицы (т.о получим кол-во единиц необходимых для безубыточной разработки на всех скважинах ) а затем делим на число скважин

# In[12]:


(10 ** 9/450000)/200


# ### Анализ результатов

# Объём сырья для безубыточной разработки новой скважины намного меньше средних значений для каждого региона => по логике получается что скорее всего какие бы вы не выбрали скважины доход будет, ну либо вы очень 'удачливый' и выбираете самые худшие скважины и тогда можете уйти в убыток, но для этого собственно и мы занимается обработкой данных, чтобы такой ситуации не произошло

# # 4. Расчёт прибыли и рисков 

# ### Функция для расчёта прибыли по выбранным скважинам и предсказаниям модели

# In[13]:


def profit(target, probabilities, count):
    probs_sorted = probabilities.sort_values(ascending=False)
    selected = target[probs_sorted.index][:count]
    return UNIT_INCOME * selected.sum() - ALL_MONEY


# ### Выбираем скважины с максимальными значениями предсказаний и считаем значения целевое значение сырья для 200 наилучших предсказаний и прибыль по этим 200-м точкам

# In[14]:


for i in range(3):
    globals()[f'max_predict_{i}'] = pd.Series(
        globals()[f'predict_data{i}']).sort_values(ascending=False)
    
    globals()[f'max_target_{i}'] = globals()[f'target_valid_{i}'][globals()[f'max_predict_{i}'].index[:200]]
    
    print(f'Целевое значение сырья для 200 наилучших предсказний из geo_data{i} =',
          globals()[f'max_target_{i}'].sum())
    
    print(f'Прибыль от 200 с максимальными значениями предсказаний из geo_data{i} =',
          profit(globals()[f'max_target_{i}'], globals()[f'max_predict_{i}'][:200], 200)) 
    print()


# ### Считаем риски и прибыль для каждого региона техникой Bootstrap с 1000 выборок, чтобы найти распределение прибыли. 

# Находим среднюю прибыль, 95%-й доверительный интервал и риск убытков

# In[15]:


for j in range(3):
    globals()[f'values{j}'] = []
    state = np.random.RandomState(42)
    for i in range(1000):
        target_subsample = pd.Series(globals()[f'target_valid_{j}']).sample(500, replace=False, random_state=state)
        probs_subsample = pd.Series(globals()[f'predict_data{j}'])[target_subsample.index]
        globals()[f'values{j}'].append(profit(target_subsample, probs_subsample, 200))
 
    globals()[f'values{j}'] = pd.Series(globals()[f'values{j}'])
    mean = globals()[f'values{j}'].mean()
    print(f'Риск убытков для geo_data{j} =',
          (globals()[f'values{j}'] < 0).mean() * 100, 
         'процента')
    #print(globals()[f'values{j}'][globals()[f'values{j}'] < 0 ]/globals()[f'values{j}'].shape[0])
    print(f"Средняя прибыль для geo_data{j}:", mean)
    print()


# In[16]:


for i in range(3):
    print(f'95% доверительный интервал для региона{i+1}, составляет:(', 
          globals()[f'values{i}'].quantile(0.025), ":",globals()[f'values{i}'].quantile(0.975), ')' )
print()


# **Вывод**
# 
# Самым лучшим регионом для добычи оказался 2-ой регион, т.к вероятность убытка на нем получается самой минимальной(вследствие видимо высокого RMSE самой модели для этого региона), и доверительный интервал получается также самым наилучшим, что опять таки склоняет нас к выбору этого региона. Регион 2 самый перспективный для нефтедобычи!

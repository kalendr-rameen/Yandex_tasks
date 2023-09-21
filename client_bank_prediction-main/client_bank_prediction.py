#!/usr/bin/env python
# coding: utf-8

# # Отток клиентов

# Из «Бета-Банка» стали уходить клиенты. Каждый месяц. Немного, но заметно. Банковские маркетологи посчитали: сохранять текущих клиентов дешевле, чем привлекать новых.
# 
# Нужно спрогнозировать, уйдёт клиент из банка в ближайшее время или нет. Нам предоставлены исторические данные о поведении клиентов и расторжении договоров с банком. 
# 
# Нужно постройте модель с предельно большим значением *F1*-меры. Чтобы сдать проект успешно, нужно довести метрику до 0.59. Также необходимо проверить *F1*-меру на тестовой выборке самостоятельно.
# 
# Дополнительно измеряйте *AUC-ROC*, сравнивайте её значение с *F1*-мерой.
# 
# Источник данных: [https://www.kaggle.com/barelydedicated/bank-customer-churn-modeling](https://www.kaggle.com/barelydedicated/bank-customer-churn-modeling)

# <h1>Содержание<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Подготовка-данных" data-toc-modified-id="Подготовка-данных-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Подготовка данных</a></span><ul class="toc-item"><li><span><a href="#Вывод" data-toc-modified-id="Вывод-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span><strong>Вывод</strong></a></span></li></ul></li><li><span><a href="#Исследование-задачи" data-toc-modified-id="Исследование-задачи-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Исследование задачи</a></span><ul class="toc-item"><li><ul class="toc-item"><li><span><a href="#Исследуем-1-ю-модель-(Логистическая-регрессия)" data-toc-modified-id="Исследуем-1-ю-модель-(Логистическая-регрессия)-2.0.1"><span class="toc-item-num">2.0.1&nbsp;&nbsp;</span>Исследуем <code>1-ю модель (Логистическая регрессия)</code></a></span></li><li><span><a href="#Исследуем-2-ю-модель-(Дерево-решений)" data-toc-modified-id="Исследуем-2-ю-модель-(Дерево-решений)-2.0.2"><span class="toc-item-num">2.0.2&nbsp;&nbsp;</span>Исследуем <code>2-ю модель (Дерево решений)</code></a></span></li><li><span><a href="#Исследуем-3-ю-модель-(Случайный-лес)" data-toc-modified-id="Исследуем-3-ю-модель-(Случайный-лес)-2.0.3"><span class="toc-item-num">2.0.3&nbsp;&nbsp;</span>Исследуем <code>3-ю модель (Случайный лес)</code></a></span></li></ul></li></ul></li><li><span><a href="#Борьба-с-дисбалансом" data-toc-modified-id="Борьба-с-дисбалансом-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Борьба с дисбалансом</a></span><ul class="toc-item"><li><span><a href="#Первый-метод-борьбы-с-дисбалансом-мы-выбрали-применение-параметра-class_weight='balanced'." data-toc-modified-id="Первый-метод-борьбы-с-дисбалансом-мы-выбрали-применение-параметра-class_weight='balanced'.-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>Первый метод борьбы с дисбалансом мы выбрали применение параметра <code>class_weight='balanced'</code>.</a></span><ul class="toc-item"><li><span><a href="#Сначала-применяем-к-1-ой-модели-(логистическая-регрессия)" data-toc-modified-id="Сначала-применяем-к-1-ой-модели-(логистическая-регрессия)-3.1.1"><span class="toc-item-num">3.1.1&nbsp;&nbsp;</span>Сначала применяем к <code>1-ой модели (логистическая регрессия)</code></a></span></li><li><span><a href="#Исследуем-2-ю-модель-(Дерево-решений)" data-toc-modified-id="Исследуем-2-ю-модель-(Дерево-решений)-3.1.2"><span class="toc-item-num">3.1.2&nbsp;&nbsp;</span>Исследуем <code>2-ю модель (Дерево решений)</code></a></span></li><li><span><a href="#Исследуем-3-ю-модель-(Случайный-лес)" data-toc-modified-id="Исследуем-3-ю-модель-(Случайный-лес)-3.1.3"><span class="toc-item-num">3.1.3&nbsp;&nbsp;</span>Исследуем <code>3-ю модель (Случайный лес)</code></a></span></li></ul></li><li><span><a href="#Попытаемся-применить-второй-метод-борьбы-с-дисбалансом-а-именно-upsampling,-т.е-мы-будем-искуственно-увеличивать-размер-выборки." data-toc-modified-id="Попытаемся-применить-второй-метод-борьбы-с-дисбалансом-а-именно-upsampling,-т.е-мы-будем-искуственно-увеличивать-размер-выборки.-3.2"><span class="toc-item-num">3.2&nbsp;&nbsp;</span>Попытаемся применить <strong>второй метод борьбы с дисбалансом</strong> а именно <code>upsampling</code>, т.е мы будем искуственно увеличивать размер выборки.</a></span><ul class="toc-item"><li><span><a href="#Теперь-применяем-полученные-данные-для-обучения-1-ой-модели-(логистическая-регрессия)" data-toc-modified-id="Теперь-применяем-полученные-данные-для-обучения-1-ой-модели-(логистическая-регрессия)-3.2.1"><span class="toc-item-num">3.2.1&nbsp;&nbsp;</span>Теперь применяем полученные данные для обучения <code>1-ой модели (логистическая регрессия)</code></a></span></li><li><span><a href="#Теперь-применяем-полученные-данные-для-обучения-2-ой-модели-(Дерево-решений)" data-toc-modified-id="Теперь-применяем-полученные-данные-для-обучения-2-ой-модели-(Дерево-решений)-3.2.2"><span class="toc-item-num">3.2.2&nbsp;&nbsp;</span>Теперь применяем полученные данные для обучения <code>2-ой модели (Дерево решений)</code></a></span></li><li><span><a href="#Теперь-применяем-полученные-данные-для-обучения-3-ей-модели-(Дерево-решений)" data-toc-modified-id="Теперь-применяем-полученные-данные-для-обучения-3-ей-модели-(Дерево-решений)-3.2.3"><span class="toc-item-num">3.2.3&nbsp;&nbsp;</span>Теперь применяем полученные данные для обучения 3-ей модели (Дерево решений)</a></span></li></ul></li></ul></li><li><span><a href="#Тестирование-модели" data-toc-modified-id="Тестирование-модели-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Тестирование модели</a></span></li><li><span><a href="#Чек-лист-готовности-проекта" data-toc-modified-id="Чек-лист-готовности-проекта-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Чек-лист готовности проекта</a></span></li></ul></div>

# ## 1. Подготовка данных

# In[1]:


import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('/datasets/Churn.csv')
df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)


# Первый делом я решил избавиться от ненужных данных при обучение поскольку такие данные будут лишь успложнять построенную модель и скорее всего приведут к ошибкам

# In[2]:


df.info()


# Видно, что у нас есть пропущенные данные в колонке `tenure`, посмотрим на данные поближе чтобы понять если ли какие либо различия у этих обьектов от остальных, но прежде привёдём названия столбцов к нижнему регистру.

# In[3]:


df.columns = df.columns.str.lower()


# In[4]:


df.describe()


# In[5]:


df.head(10)


# In[6]:


df[df.isna().any(axis=1)].head(10)


# In[7]:


df[df.isna().any(axis=1)].describe()


# In[8]:


df.describe()


# Данные с NaN в колонке `tenure` выглядят вполне нормально, поэтому было решено заменить их на рандомные значения чтобы не изменить распреление самих данных, и не потерять лишние данные. Посмотрим на распределение величин до и после

# In[9]:


df['tenure'].hist()


# In[10]:


import random


# In[11]:


index_nan = df[df['tenure'].isna()]['tenure'].index


# In[12]:


rand_list = []
for i in range(len(index_nan)):
    rand_list.append(random.randint(df['tenure'].min(), df['tenure'].max()))


# In[13]:


rand_series = pd.Series(rand_list, index=index_nan)


# In[14]:


df.loc[index_nan,'tenure'] = rand_series


# In[15]:


df['tenure'].hist()


# Распределение на глаз почти не изменилось!

# In[16]:


df.info()


# **Добавил сверху код с заменой NaN на рандомные значения, само распределение не сильно(либо вовсе не) изменилось!** 

# Заменим тип данных в колонке `tenure` с 'object' на 'int64'

# In[17]:


df['tenure'] = df['tenure'].astype('int64')


# In[18]:


df.info()


# Применили технику OHE для прямого декодирования данных

# In[19]:


df = pd.get_dummies(df, drop_first=True)


# In[20]:


df.head()


# In[21]:


df.info()


# Теперь маштабируем данные для количественных признаков 

# In[22]:


numeric = ['creditscore','age','tenure', 'balance', 'numofproducts', 'estimatedsalary']


# In[23]:


scaler = StandardScaler()
scaler.fit(df[numeric])


# In[24]:


df[numeric] = scaler.transform(df[numeric])


# ### **Вывод** 
# 
# Теперь после обработки данных можно их использовать для дальнейшего построения модели. Можно заметить что в данных присутствуют как категориальные признаки так и численные, поэтому применили технику OHE для прямого декодирования категориальных данных. Приступим к построению моделей
# 

# ## 2. Исследование задачи

# #### Исследуем `1-ю модель (Логистическая регрессия)`

# In[25]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score


# Используем joblib для сохранения моделей

# In[26]:


from tempfile import mkdtemp
import joblib
save_dir = mkdtemp()
import os 


# In[27]:


features = df.drop(['exited'], axis=1)


# In[28]:


target = df['exited']


# In[29]:


from sklearn.model_selection import train_test_split


# In[30]:


df_train, df_valid_n_test = train_test_split(df, test_size=0.4, random_state=12345)


# In[31]:


df_valid, df_test = train_test_split(df_valid_n_test, test_size=0.5, random_state=12345)


# In[32]:


features_train = df_train.drop(['exited'], axis=1)
target_train = df_train['exited']
features_valid = df_valid.drop(['exited'], axis=1)
target_valid = df_valid['exited']


# In[33]:


print(df_train['exited'].value_counts())
print()
df_valid['exited'].value_counts()


# Можно заметить сразу **несбалансированность данных**, с которой мы будем бороться в след. пункте!

# In[34]:


model = LogisticRegression(random_state=12345, solver='liblinear')
model.fit(features_train,target_train) 
predict_valid = model.predict(features_valid)
probabilities_one_valid = model.predict_proba(features_valid)[:,1]

print(f1_score(target_valid, predict_valid))
print('roc_auc_score =',roc_auc_score(target_valid, probabilities_one_valid))


# Сохраняем модель

# In[35]:


filename_logistic_reg = os.path.join(save_dir, 'model.joblib.log.reg')
joblib.dump(model, filename_logistic_reg)


# Итак модель создали, поиграем теперь со значением threshold 

# In[36]:


probabilities_valid = joblib.load(filename_logistic_reg).predict_proba(features_valid)
probabilities_one_valid = probabilities_valid[:, 1]

best_f1 = 0
best_threshold = 0

for threshold in np.arange(0, 0.7, 0.005):
    predicted_valid = probabilities_one_valid > threshold
    f1_score_val = f1_score(target_valid, predicted_valid)
    print("Порог = {:.2f} | f1 = {:.3f}".format(
        threshold, f1_score_val))
    if best_f1 < f1_score_val:
        best_threshold = threshold
        best_f1 = f1_score_val


# In[37]:


print('best f1_score: {} ,\nbest threshold : {}'.format(best_f1, best_threshold))


# Видно что при изменение порога увеличивается значение f1 метрики!

# #### Исследуем `2-ю модель (Дерево решений)`

# Подбираем оптимальную модель перебирая гиперпараметры, в данном случае max_depth

# In[38]:


best_res = 0
best_model = None
for i in range(1,20):
    model = DecisionTreeClassifier(random_state = 12345,max_depth=i)
    model.fit(features_train, target_train)
    predictions = model.predict(features_valid)
    res = f1_score(target_valid, predictions)
    print('max_depth = {} :'.format(i), res)
    if res > best_res:
        best_res = res
        best_model = model


# In[39]:


print('best f1_score for DecisionTree =', best_res)


# In[40]:


filename_decision_tree = os.path.join(save_dir, 'model.joblib.decision_tree')
joblib.dump(best_model, filename_decision_tree)


# In[41]:


probabilities_valid = joblib.load(filename_decision_tree).predict_proba(features_valid)
probabilities_one_valid = probabilities_valid[:, 1]

best_f1 = 0
best_threshold = 0

for threshold in np.arange(0, 0.7, 0.005):
    predicted_valid = probabilities_one_valid > threshold
    f1_score_val = f1_score(target_valid, predicted_valid)
    print("Порог = {:.2f} | f1 = {:.3f}".format(
        threshold, f1_score_val))
    if best_f1 < f1_score_val:
        best_threshold = threshold
        best_f1 = f1_score_val


# In[42]:


print('best f1_score: {} ,\nbest threshold : {}'.format(best_f1, best_threshold))


# Опять как и в предыдущем случае значение f1_score увеличивается при снижение порога!

# #### Исследуем `3-ю модель (Случайный лес)`

# Подбираем оптимальную модель перебирая гиперпараметры, в данном случае n_estimators

# In[43]:


best_model = None
best_result = 0
for est in range(1, 30):
    model = RandomForestClassifier(random_state=12345, n_estimators=est) 
    model.fit(features_train, target_train) 
    predictions = model.predict(features_valid)
    res = f1_score(target_valid, predictions) 
    print('number_estimator = {} :'.format(est), res)
    if res > best_result:
        best_model = model 
        best_result = res


# In[44]:


print('best f1_score for Random Forest = ', best_result)


# In[45]:


filename_random_forest = os.path.join(save_dir, 'model.joblib.random_forest')
joblib.dump(best_model, filename_random_forest)


# In[46]:


joblib.load(filename_random_forest)


# In[47]:


probabilities_valid = joblib.load(filename_random_forest).predict_proba(features_valid)
probabilities_one_valid = probabilities_valid[:, 1]

best_f1 = 0
best_threshold = 0

for threshold in np.arange(0, 0.7, 0.005):
    predicted_valid = probabilities_one_valid > threshold
    f1_score_val = f1_score(target_valid, predicted_valid)
    print("Порог = {:.2f} | f1 = {:.3f}".format(
        threshold, f1_score_val))
    if best_f1 < f1_score_val:
        best_threshold = threshold
        best_f1 = f1_score_val


# In[48]:


print('best f1_score: {} ,\nbest threshold : {}'.format(best_f1, best_threshold))


# ## 3. Борьба с дисбалансом

# ### Первый метод борьбы с дисбалансом мы выбрали применение параметра `class_weight='balanced'`.
# 
# #### Сначала применяем к `1-ой модели (логистическая регрессия)`

# In[49]:


model = LogisticRegression(random_state=12345, solver='liblinear', class_weight='balanced')
model.fit(features_train,target_train) 
predict_valid = model.predict(features_valid)
probabilities_one_valid = model.predict_proba(features_valid)[:,1]

print('f1_score =',f1_score(target_valid, predict_valid))
print('roc_auc_score =',roc_auc_score(target_valid, probabilities_one_valid))


# Метрика достаточно сильно изменилась в сравенение с метрикой (без изменения threshold) у модели без данного параметра, но такого значения всё равно не достаточно в условиях данной задачи.

# In[50]:


filename_logistic_reg_weighted = os.path.join(save_dir, 'model.joblib.log.reg.weight')
joblib.dump(model, filename_logistic_reg_weighted)


# Перебираем значения threshold

# In[51]:


best_f1 = 0
best_threshold = 0
best_roc = 0

for threshold in np.arange(0, 0.7, 0.005):
    predicted_valid = probabilities_one_valid > threshold
    f1_score_val = f1_score(target_valid, predicted_valid)
    print("Порог = {:.2f} | f1 = {:.3f}".format(
        threshold, f1_score_val))
    if best_f1 < f1_score_val:
        best_threshold = threshold
        best_f1 = f1_score_val


# In[52]:


print('best f1_score: {} ,\nbest threshold : {}'.format(best_f1, best_threshold))


# #### Исследуем `2-ю модель (Дерево решений)`
# 
# Подбираем оптимальную модель перебирая гиперпараметры, в данном случае max_depth

# In[53]:


best_res = 0
best_model = None
for i in range(1,20):
    model = DecisionTreeClassifier(random_state = 12345,max_depth=i, class_weight='balanced')
    model.fit(features_train, target_train)
    predictions = model.predict(features_valid)
    res = f1_score(target_valid, predictions)
    print('max_depth = {} :'.format(i), res)
    if res > best_res:
        best_res = res
        best_model = model

        


# In[54]:


predict_valid = best_model.predict(features_valid)
probabilities_one_valid = best_model.predict_proba(features_valid)[:,1]
print('f1_score =',f1_score(target_valid, predict_valid))
print('roc_auc_score =',roc_auc_score(target_valid, probabilities_one_valid))


# В целом значения метрики уже удовлетворяет условиям задачи но посмотрим еще на другие методы, может найдем что-то получше)

# In[55]:


filename_decision_tree_weighted = os.path.join(save_dir, 'model.joblib.decision.tree.weight')
joblib.dump(best_model, filename_decision_tree_weighted)


# In[56]:


from sklearn.metrics import f1_score
model = DecisionTreeClassifier(random_state=12345, max_depth=5, class_weight='balanced')
model.fit(features_train, target_train)
probabilities_valid = model.predict_proba(features_valid)
probabilities_one_valid = probabilities_valid[:, 1]

best_f1 = 0
best_threshold = 0

for threshold in np.arange(0, 0.7, 0.005):
    predicted_valid = probabilities_one_valid > threshold
    f1_score_val = f1_score(target_valid, predicted_valid)
    print("Порог = {:.2f} | f1 = {:.3f}".format(
        threshold, f1_score_val))
    if best_f1 < f1_score_val:
        best_threshold = threshold
        best_f1 = f1_score_val


# In[57]:


print('best f1_score: {} ,\nbest threshold : {}'.format(best_f1, best_threshold))


# #### Исследуем `3-ю модель (Случайный лес)`
# 
# Подбираем оптимальную модель перебирая гиперпараметры, в данном случае `n_estimators`

# In[58]:


from sklearn.ensemble import RandomForestRegressor
best_model = None
best_result = 0
for est in range(1, 30):
    model = RandomForestClassifier(random_state=12345, n_estimators=est, class_weight='balanced') 
    model.fit(features_train, target_train) 
    predictions = model.predict(features_valid)
    res = f1_score(target_valid, predictions) 
    print('number_estimator = {} :'.format(est), res)
    if res > best_result:
        best_model = model 
        best_result = res


# In[59]:


predict_valid = best_model.predict(features_valid)
probabilities_one_valid = best_model.predict_proba(features_valid)[:,1]
print('f1_score =',f1_score(target_valid, predict_valid))
print('roc_auc_score =',roc_auc_score(target_valid, probabilities_one_valid))


# In[60]:


filename_random_forest_weighted = os.path.join(save_dir, 'model.joblib.random.forest.weighted')
joblib.dump(best_model, filename_random_forest_weighted)


# In[61]:


probabilities_valid = joblib.load(filename_random_forest_weighted).predict_proba(features_valid)
probabilities_one_valid = probabilities_valid[:, 1]

best_f1 = 0
best_threshold = 0

for threshold in np.arange(0, 0.7, 0.005):
    predicted_valid = probabilities_one_valid > threshold
    f1_score_val = f1_score(target_valid, predicted_valid)
    print("Порог = {:.2f} | f1 = {:.3f}".format(
        threshold, f1_score_val))
    if best_f1 < f1_score_val:
        best_threshold = threshold
        best_f1 = f1_score_val


# In[62]:


print('best f1_score: {} ,\nbest threshold : {}'.format(best_f1, best_threshold))


# ### Попытаемся применить **второй метод борьбы с дисбалансом** а именно `upsampling`, т.е мы будем искуственно увеличивать размер выборки. 

# In[63]:


from sklearn.utils import shuffle


# In[64]:


def upsample(features, target, repeat):
    features_zeros = features[target == 0]
    features_ones = features[target == 1]
    target_zeros = target[target == 0]
    target_ones = target[target == 1]

    features_upsampled = pd.concat([features_zeros] + [features_ones] * repeat)
    target_upsampled = pd.concat([target_zeros] + [target_ones] * repeat)
    
    features_upsampled, target_upsampled = shuffle(
        features_upsampled, target_upsampled, random_state=12345)
    
    return features_upsampled, target_upsampled


# In[65]:


features_upsampled, target_upsampled = upsample(features_train, target_train, 4)


# #### Теперь применяем полученные данные для обучения `1-ой модели (логистическая регрессия)`

# In[66]:


model = LogisticRegression(random_state=12345, solver='liblinear')
model.fit(features_upsampled,target_upsampled) 
predict_valid = model.predict(features_valid)
probabilities_one_valid = model.predict_proba(features_valid)[:,1]

print(f1_score(target_valid, predict_valid))
print('roc_auc_score =',roc_auc_score(target_valid, probabilities_one_valid))


# In[67]:


filename_logistic_reg_upsampling = os.path.join(save_dir, 'model.joblib.log.reg.up')
joblib.dump(model, filename_logistic_reg_upsampling)


# In[68]:


probabilities_valid = joblib.load(filename_logistic_reg_upsampling).predict_proba(features_valid)
probabilities_one_valid = probabilities_valid[:, 1]

best_f1 = 0
best_threshold = 0

for threshold in np.arange(0, 0.7, 0.005):
    predicted_valid = probabilities_one_valid > threshold
    f1_score_val = f1_score(target_valid, predicted_valid)
    print("Порог = {:.2f} | f1 = {:.3f}".format(
        threshold, f1_score_val))
    if best_f1 < f1_score_val:
        best_threshold = threshold
        best_f1 = f1_score_val


# In[69]:


print('best f1_score: {} ,\nbest threshold : {}'.format(best_f1, best_threshold))


# #### Теперь применяем полученные данные для обучения `2-ой модели (Дерево решений)`

# In[70]:


best_res = 0
best_model = None
for i in range(1,20):
    model = DecisionTreeClassifier(random_state = 12345,max_depth=i)
    model.fit(features_upsampled, target_upsampled)
    predictions = model.predict(features_valid)
    res = f1_score(target_valid, predictions)
    print('max_depth = {} :'.format(i), res)
    if res > best_res:
        best_res = res
        best_model = model


# In[71]:


predict_valid = best_model.predict(features_valid)
probabilities_one_valid = best_model.predict_proba(features_valid)[:,1]
print('f1_score =',f1_score(target_valid, predict_valid))
print('roc_auc_score =',roc_auc_score(target_valid, probabilities_one_valid))


# In[72]:


filename_decision_tree_upsample = os.path.join(save_dir, 'model.joblib.decision_tree.up')
joblib.dump(best_model, filename_decision_tree_upsample)


# In[73]:


probabilities_valid = joblib.load(filename_decision_tree_upsample).predict_proba(features_valid)
probabilities_one_valid = probabilities_valid[:, 1]

best_f1 = 0
best_threshold = 0

for threshold in np.arange(0, 0.7, 0.005):
    predicted_valid = probabilities_one_valid > threshold
    f1_score_val = f1_score(target_valid, predicted_valid)
    print("Порог = {:.2f} | f1 = {:.3f}".format(
        threshold, f1_score_val))
    if best_f1 < f1_score_val:
        best_threshold = threshold
        best_f1 = f1_score_val


# In[74]:


print('best f1_score: {} ,\nbest threshold : {}'.format(best_f1, best_threshold))


# #### Теперь применяем полученные данные для обучения 3-ей модели (Дерево решений)

# In[75]:


best_model = None
best_result = 0
for est in range(1, 30):
    model = RandomForestClassifier(random_state=12345, n_estimators=est) 
    model.fit(features_upsampled, target_upsampled) 
    predictions = model.predict(features_valid)
    res = f1_score(target_valid, predictions) 
    print('number_estimator = {} :'.format(est), res)
    if res > best_result:
        best_model = model 
        best_result = res


# In[76]:


predict_valid = best_model.predict(features_valid)
probabilities_one_valid = best_model.predict_proba(features_valid)[:,1]
print('f1_score =',f1_score(target_valid, predict_valid))
print('roc_auc_score =',roc_auc_score(target_valid, probabilities_one_valid))


# In[77]:


filename_random_forest_upsample = os.path.join(save_dir, 'model.joblib.random_forest.up')
joblib.dump(best_model, filename_random_forest_upsample)


# In[78]:


probabilities_valid = joblib.load(filename_random_forest_upsample).predict_proba(features_valid)
probabilities_one_valid = probabilities_valid[:, 1]

best_f1 = 0
best_threshold = 0

for threshold in np.arange(0, 0.7, 0.005):
    predicted_valid = probabilities_one_valid > threshold
    f1_score_val = f1_score(target_valid, predicted_valid)
    print("Порог = {:.2f} | f1 = {:.3f}".format(
        threshold, f1_score_val))
    if best_f1 < f1_score_val:
        best_threshold = threshold
        best_f1 = f1_score_val


# In[79]:


print('best f1_score: {} ,\nbest threshold : {}'.format(best_f1, best_threshold))


# **Вывод**
# 
# Самой лучшей моделью **в случае с применением параметра class_weight = 'balanced'**, оказался алгоритм DecisionTreeClassifier, значения f1_score и aug_roc которого оказались максимальными среди трёх представленных алгоритмов и равны соответсвенно **f1_score = 0.5963791267305644, roc_auc_score = 0.8310244134068074**, также при изменение threshold эти значения становятся чуть лучше. **В случае с upsampling-ом** лучшей моделью оказался алгоритм RandomForest, значения которого превышают DecisionTreeClassifier, и являются лучшими среди всех методов борьбы с дисбалансом и равны соответственно **f1_score = 0.609597924773022, roc_auc_score = 0.8284286742600669**. Значения f1_score в обоих случаях удовлетворяют условиям задачи, поэтому теперь стоит проверить как наша модель справляется с тестовой выборкой.

# ## 4. Тестирование модели

# Протестируем 2 лидирующие модели дабы удостовериться в лидерстве одной из них(или в их равенстве)

# In[80]:


features_test = df_test.drop(['exited'], axis=1)
target_test = df_test['exited']


# Обьединяю train и test для DecisionTreeClassifier, чтобы обучить ее с уже известными параметрами на обьединенной выборке

# In[81]:


features_train_valid =  pd.concat([features_train] + [features_valid])
target_train_valid = pd.concat([target_train] + [target_valid])


# Также создаю **upsampling** train и test, чтобы применить RandomForestClassifier, который у нас испльзуется с подобранными параметрами, но без параметра class_weight='balanced'

# In[82]:


features_upsampled_train_valid, target_upsampled_train_valid = upsample(
    features_train_valid, target_train_valid, 4)


# Теперь обучим DecisionTree, о прежде посмотрим на параметры уже обученной модели, чтобы их задать в новой модели

# In[83]:


joblib.load(filename_decision_tree_weighted)


# In[84]:


model = DecisionTreeClassifier(class_weight='balanced', max_depth=5, random_state=12345)


# In[85]:


model.fit(features_train_valid, target_train_valid)


# In[86]:


predict_test = model.predict(features_test)
probabilities_one_valid = model.predict_proba(features_test)[:,1]
print('f1_score =',f1_score(target_test, predict_test))
print('roc_auc_score =',roc_auc_score(target_test, probabilities_one_valid))


# Посмотрим на важность факторов для нашей модели

# In[87]:


import matplotlib.pyplot as plt
importances = model.feature_importances_
feature_names = [f'{i}' for i in features_test.columns]
forest_importances = pd.Series(importances, index=feature_names)

fig, ax = plt.subplots()
forest_importances.plot.bar(ax=ax)
ax.set_title("Feature importances")
ax.set_ylabel("Mean decrease in impurity")данных
fig.tight_layout()


# Метрика получилась лучше после обьединения данных

# Теперь обучим RandomForest

# In[88]:


joblib.load(filename_random_forest_upsample)


# In[89]:


model = RandomForestClassifier(random_state=12345, n_estimators=17)


# In[90]:


model.fit(features_upsampled_train_valid, target_upsampled_train_valid)


# In[91]:


predict_test = model.predict(features_test)
probabilities_one_valid = model.predict_proba(features_test)[:,1]
print('f1_score =',f1_score(target_test, predict_test))
print('roc_auc_score =',roc_auc_score(target_test, probabilities_one_valid))


# Посмотрим на важность параметров модели

# In[92]:


importances = model.feature_importances_
feature_names = [f'{i}' for i in features_test.columns]
forest_importances = pd.Series(importances, index=feature_names)

fig, ax = plt.subplots()
forest_importances.plot.bar(ax=ax)
ax.set_title("Feature importances")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout() 


# Предсказания стали чуть хуже нежели которые были получены в предыдущем пунке, также стоит отметить что в обеих случая метрики оказались равны, поэтому в данном случае обе модели оказались равноценны.

# **Вывод**
# 
# Обе модели ведут себя одинаково 'в бою' и применение любой из них к тестовым данным равноценно. Также можно заметить значительную разницу в важности параметров для разных моделей, в случае RandomForest принимаются во внимание все параметры в отличие от DecisionTree, что скорее всего не может не скзаать на скорости работы самой модели
# 

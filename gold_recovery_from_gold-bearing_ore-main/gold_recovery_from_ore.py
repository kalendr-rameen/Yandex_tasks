#!/usr/bin/env python
# coding: utf-8

# <h1>Содержание<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Подготовка-данных" data-toc-modified-id="Подготовка-данных-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Подготовка данных</a></span><ul class="toc-item"><li><span><a href="#Открытие-файлов-и-их-изучение" data-toc-modified-id="Открытие-файлов-и-их-изучение-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Открытие файлов и их изучение</a></span></li><li><span><a href="#Проверьте,-что-эффективность-обогащения-рассчитана-правильно.-Вычислите-её-на-обучающей-выборке-для-признака-rougher.output.recovery.-Найдите-MAE-между-вашими-расчётами-и-значением-признака.-Опишите-выводы." data-toc-modified-id="Проверьте,-что-эффективность-обогащения-рассчитана-правильно.-Вычислите-её-на-обучающей-выборке-для-признака-rougher.output.recovery.-Найдите-MAE-между-вашими-расчётами-и-значением-признака.-Опишите-выводы.-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>Проверьте, что эффективность обогащения рассчитана правильно. Вычислите её на обучающей выборке для признака rougher.output.recovery. Найдите MAE между вашими расчётами и значением признака. Опишите выводы.</a></span></li><li><span><a href="#Проанализируйте-признаки,-недоступные-в-тестовой-выборке.-Что-это-за-параметры?-К-какому-типу-относятся?" data-toc-modified-id="Проанализируйте-признаки,-недоступные-в-тестовой-выборке.-Что-это-за-параметры?-К-какому-типу-относятся?-1.3"><span class="toc-item-num">1.3&nbsp;&nbsp;</span>Проанализируйте признаки, недоступные в тестовой выборке. Что это за параметры? К какому типу относятся?</a></span></li><li><span><a href="#Проведите-предобработку-данных" data-toc-modified-id="Проведите-предобработку-данных-1.4"><span class="toc-item-num">1.4&nbsp;&nbsp;</span>Проведите предобработку данных</a></span></li></ul></li><li><span><a href="#Анализ-данных" data-toc-modified-id="Анализ-данных-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Анализ данных</a></span><ul class="toc-item"><li><span><a href="#Посмотрите,-как-меняется-концентрация-металлов-(Au,-Ag,-Pb)-на-различных-этапах-очистки.-Опишите-выводы." data-toc-modified-id="Посмотрите,-как-меняется-концентрация-металлов-(Au,-Ag,-Pb)-на-различных-этапах-очистки.-Опишите-выводы.-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Посмотрите, как меняется концентрация металлов (Au, Ag, Pb) на различных этапах очистки. Опишите выводы.</a></span></li><li><span><a href="#Сравните-распределения-размеров-гранул-сырья-на-обучающей-и-тестовой-выборках.-Если-распределения-сильно-отличаются-друг-от-друга,-оценка-модели-будет-неправильной." data-toc-modified-id="Сравните-распределения-размеров-гранул-сырья-на-обучающей-и-тестовой-выборках.-Если-распределения-сильно-отличаются-друг-от-друга,-оценка-модели-будет-неправильной.-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Сравните распределения размеров гранул сырья на обучающей и тестовой выборках. Если распределения сильно отличаются друг от друга, оценка модели будет неправильной.</a></span></li><li><span><a href="#Исследуйте-суммарную-концентрацию-всех-веществ-на-разных-стадиях:-в-сырье,-в-черновом-и-финальном-концентратах." data-toc-modified-id="Исследуйте-суммарную-концентрацию-всех-веществ-на-разных-стадиях:-в-сырье,-в-черновом-и-финальном-концентратах.-2.3"><span class="toc-item-num">2.3&nbsp;&nbsp;</span>Исследуйте суммарную концентрацию всех веществ на разных стадиях: в сырье, в черновом и финальном концентратах.</a></span></li></ul></li><li><span><a href="#Модель" data-toc-modified-id="Модель-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Модель</a></span><ul class="toc-item"><li><span><a href="#Напишите-функцию-для-вычисления-итоговой-sMAPE." data-toc-modified-id="Напишите-функцию-для-вычисления-итоговой-sMAPE.-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>Напишите функцию для вычисления итоговой sMAPE.</a></span></li><li><span><a href="#Обучите-разные-модели-и-оцените-их-качество-кросс-валидацией.-Выберите-лучшую-модель-и-проверьте-её-на-тестовой-выборке.-Опишите-выводы." data-toc-modified-id="Обучите-разные-модели-и-оцените-их-качество-кросс-валидацией.-Выберите-лучшую-модель-и-проверьте-её-на-тестовой-выборке.-Опишите-выводы.-3.2"><span class="toc-item-num">3.2&nbsp;&nbsp;</span>Обучите разные модели и оцените их качество кросс-валидацией. Выберите лучшую модель и проверьте её на тестовой выборке. Опишите выводы.</a></span></li></ul></li><li><span><a href="#Чек-лист-готовности-проекта" data-toc-modified-id="Чек-лист-готовности-проекта-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Чек-лист готовности проекта</a></span></li></ul></div>

# # Восстановление золота из руды

# Подготовьте прототип модели машинного обучения для «Цифры». Компания разрабатывает решения для эффективной работы промышленных предприятий.
# 
# Модель должна предсказать коэффициент восстановления золота из золотосодержащей руды. Используйте данные с параметрами добычи и очистки. 
# 
# Модель поможет оптимизировать производство, чтобы не запускать предприятие с убыточными характеристиками.
# 
# Вам нужно:
# 
# 1. Подготовить данные;
# 2. Провести исследовательский анализ данных;
# 3. Построить и обучить модель.
# 
# Чтобы выполнить проект, обращайтесь к библиотекам *pandas*, *matplotlib* и *sklearn.* Вам поможет их документация.

# ## 1. Подготовка данных

# ### Открытие файлов и их изучение

# In[1]:


import sklearn
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.datasets import make_regression
from sklearn.metrics import make_scorer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.dummy import DummyRegressor
from tempfile import mkdtemp
import joblib
import os 


# In[2]:


save_dir = mkdtemp()


# In[3]:


df_train = pd.read_csv('/datasets/gold_recovery_train_new.csv')
df_test = pd.read_csv('/datasets/gold_recovery_test_new.csv')
df = pd.read_csv('/datasets/gold_recovery_full_new.csv')


# In[4]:


df_train[df_train.isna().any(axis=1)].head(20)


# In[5]:


df_train.info()


# In[6]:


df_test.info()


# In[7]:


df_train[df_train.columns[df_train.isnull().any()]].isnull().sum().head(50)


# **Вывод**
# 
# Видно что:
# 1) в `df_train` кол-во колонок больше чем в `df_test`
# 2) Данные пропущены в обоих датасетах
# 3) Коол-во пропусков варьируется в зависимости от колонки в достаточно больших диапазонах от 1 до 1000 и более 

# ### Проверьте, что эффективность обогащения рассчитана правильно. Вычислите её на обучающей выборке для признака rougher.output.recovery. Найдите MAE между вашими расчётами и значением признака. Опишите выводы.

# In[8]:


Rec_train = df_train['rougher.output.recovery']


# In[9]:


C = df_train['rougher.output.concentrate_au']
F = df_train['rougher.input.feed_au']
T = df_train['rougher.output.tail_au']


# In[10]:


Rec_measured = (C * (F - T))/ (F * (C - T)) * 100


# In[11]:


mean_absolute_error(Rec_train, Rec_measured)


# In[12]:


df_train[~df_train.isna()]


# **Комментарий**
# 
# Значение MAE достаточно маленькое чтобы говорить о каких лиюо значительных различиях между расчитанной нами и значениями в таблице величины `rougher.output.recovery`.

# ### Проанализируйте признаки, недоступные в тестовой выборке. Что это за параметры? К какому типу относятся?

# In[13]:


col_only_train = df_train.columns[~df_train.columns.isin(df_test.columns)]
print(col_only_train)


# **Комментарий**
# 
# Видно что колонки отсутсвутющие в `df_test` в основном касаются выходных данных(т.е c типом параметра output), что в целом наверно логично ибо насколько я понял в тестовом наборе данных указаны лишь параметры которые известны лишь в начале, соответственно итоговые значения нам не должны быть известны(так как мы их по сути предсказываем) 

# ### Проведите предобработку данных

# Заполняем методом ffill т.к соседние по времени параметры часто похожи(так было сказано в условие)

# In[14]:


# for i in df_train[df_train.isna().any(axis=1)].index:
#     print(df_train.loc[i][df_train[df_train.isna().any(axis=1)].loc[i].isna()], '\n')


# In[15]:


df_train.fillna(method='ffill', inplace=True)


# In[16]:


df_test.fillna(method='ffill', inplace=True)


# ## 2. Анализ данных

# ### Посмотрите, как меняется концентрация металлов (Au, Ag, Pb) на различных этапах очистки. Опишите выводы.

# **Флотация**

# In[17]:


for i in ['ag', 'au', 'pb']:
    globals()[f'{i}_flotation'] = df_train[f'rougher.output.concentrate_{i}'] - df_train[f'rougher.input.feed_{i}']
    print(
        f'Разница концентраций для {i}-элемента между конечным и исходными концентрациями равна','\n',
        globals()[f'{i}_flotation'])


# **Первичным этап очистки**

# In[18]:


for i in ['ag', 'au', 'pb']:
    print(
        f'Разница концентраций для {i}-элемента между конечным и исходными концентрациями во время первичной очистки равна','\n',
        df_train[f'primary_cleaner.output.concentrate_{i}'] - df_train[f'rougher.output.concentrate_{i}'])


# **Вторичный этап очистки**

# In[19]:


for i in ['ag', 'au', 'pb']:
    print(
        f'Разница концентраций для {i}-элемента между конечным и исходными концентрациями во время первичной очистки равна','\n',
        df_train[f'final.output.concentrate_{i}'] - df_train[f'primary_cleaner.output.concentrate_{i}'])


# ![image.png](attachment:image.png)

# Получилось сделать, но не нашел способов как сделать это без повторений кода..

# In[20]:


sns.set(rc={'figure.figsize':(11.7,8.27)})


# In[21]:


metals = ['ag', 'au', 'pb']


# In[22]:


from matplotlib.pyplot import figure


# In[23]:


metal = metals[0]
columns = ['rougher.output.concentrate_', 'primary_cleaner.output.concentrate_', 'final.output.concentrate_']
for j in columns:
    subset = df_train[j+metal]
    sns.distplot(subset, hist = False, kde = True,
            kde_kws = {'linewidth': 1, 'shade': True},
            label = j+i)
        
    # Plot formatting
    plt.legend(prop={'size': 11}, title = 'Этап')
    plt.title(f'Распределение концетрации {metal} на разных этапах очистки')
    plt.xlabel('Процент веществ')
    plt.ylabel('Плотность')


# In[24]:


metal = metals[1]
columns = ['rougher.output.concentrate_', 'primary_cleaner.output.concentrate_', 'final.output.concentrate_']
for j in columns:
    subset = df_train[j+metal]
    sns.distplot(subset, hist = False, kde = True,
            kde_kws = {'linewidth': 1, 'shade': True},
            label = j+i)
        
    # Plot formatting
    plt.legend(prop={'size': 11}, title = 'Этап')
    plt.title(f'Распределение концетрации {metal} на разных этапах очистки')
    plt.xlabel('Процент веществ')
    plt.ylabel('Плотность')


# In[25]:


metal = metals[2]
columns = ['rougher.output.concentrate_', 'primary_cleaner.output.concentrate_', 'final.output.concentrate_']
for j in columns:
    subset = df_train[j+metal]
    sns.distplot(subset, hist = False, kde = True,
            kde_kws = {'linewidth': 1, 'shade': True},
            label = j+i)
        
    # Plot formatting
    plt.legend(prop={'size': 11}, title = 'Этап')
    plt.title(f'Распределение концетрации {metal} на разных этапах очистки')
    plt.xlabel('Процент веществ')
    plt.ylabel('Плотность')


# Можно заметить что концетрация целевого металла а именно золота, повышается по мере прохождения через стадии обработки, в отличие от других металлов коцентрация которых в конце либо становится меньше либо немного повышается в стравнение с начальными концентрациями

# ### Сравните распределения размеров гранул сырья на обучающей и тестовой выборках. Если распределения сильно отличаются друг от друга, оценка модели будет неправильной.

# In[26]:


ax = df_train.plot(kind='hist', y='rougher.input.feed_size',range=(0, 150), bins=30,linewidth=5, alpha=0.7, label='train')
df_test.plot(kind='hist', y='rougher.input.feed_size',range=(0, 150), bins=30 ,linewidth=5, alpha=0.7, label='test' , ax=ax)


# In[27]:


# ax = df_train.sample(frac=0.36, random_state=12345).plot(kind='hist', y='rougher.input.feed_size',range=(0, 150), bins=30,linewidth=5, alpha=0.7, label='train')
# df_test.plot(kind='hist', y='rougher.input.feed_size',range=(0, 150), bins=30 ,linewidth=5, alpha=0.7, label='test' , ax=ax)


# Видно что распределения похожи(т.к концы одинаковые и разница именно в середине распределения куда из за того что в тестовой выборке просто меньше данных, а то что пик меньше именно в середине, ем более подтвержает что распределения похожи)

# ### Исследуйте суммарную концентрацию всех веществ на разных стадиях: в сырье, в черновом и финальном концентратах.

# In[28]:


df_train['rougher.input.sum'] = (df_train['rougher.input.feed_ag']
 + df_train['rougher.input.feed_au']
 + df_train['rougher.input.feed_pb']
 + df_train['rougher.input.feed_sol'])


# In[29]:


df_train['rougher.input.sum'].plot(kind='hist', alpha=0.5, grid=True, bins=40)


# In[30]:


df_train['rougher.output.sum'] = (df_train['rougher.output.concentrate_ag'] 
 + df_train['rougher.output.concentrate_au'] 
 + df_train['rougher.output.concentrate_sol']
 + df_train['rougher.output.concentrate_pb'])


# In[31]:


df_train['rougher.output.sum'].plot(kind='hist', alpha=0.5, grid=True, bins=40)


# In[32]:


df_train['primary_cleaner.output.sum'] = (df_train['primary_cleaner.output.concentrate_ag'] 
 + df_train['primary_cleaner.output.concentrate_au'] 
 + df_train['primary_cleaner.output.concentrate_pb'] 
 + df_train['primary_cleaner.output.concentrate_sol'])


# In[33]:


df_train['primary_cleaner.output.sum'].plot(kind='hist', alpha=0.5, grid=True, bins=40)


# In[34]:


df_train['final.output.sum'] = (df_train['final.output.concentrate_ag'] 
 + df_train['final.output.concentrate_au'] 
 + df_train['final.output.concentrate_pb'] 
 + df_train['final.output.concentrate_sol'])


# In[35]:


df_train['final.output.sum'].plot(kind='hist', alpha=0.5, grid=True, bins=40)


# In[36]:


df_train[df_train['final.output.sum'] < 7]


# Можно заметить что в данных есть странные выбросы которые не совсем понятно откуда взялись, попробуем посмотреть что же с этими данными не так возможно сравнение нормальных данных с аномальными даст нам некую картину происходящего и возможно получится что это вовсе не выбросы а нормальные значения. Сперва посмотрим на данные после флотации (колонка `rougher.output.sum`), для этого сделаем отсечку на значениях из этой таблицы < 5 (т.е аномальные значения), и значения > 60 (т.е нормальные значения). После этого вычтем для каждого обьекта соответвующие признаки, перед этим применив метод `.reset_index`, чтобы не возникало проблем с индексами, и затем посмотрим на среднее для полученных значений, возможно в данных изначально 'затесалось', что то непонятное

# In[37]:


roug = [i for i in df_train.columns if 'rougher' in i]


# In[38]:


less = df_train[df_train['rougher.output.sum'] < 5][roug].reset_index(drop=True)


# In[39]:


less.shape


# In[40]:


more = df_train[df_train['rougher.output.sum'] > 60][roug].reset_index(drop=True).loc[0:302]


# In[41]:


for i in roug:
    print(f'{i} : ', (more-less)[i].mean())


# Можно заметить что присутсвутют огромные разницы в некоторых признаках, достигающие порой 1000 и более, в то же время видна заметная разница между значениями `tail`где по сути и должны быть по сути те металлы что не попали в черновой концентрат, но видно что разница мала. Получается что у нас просто пропали наши металлы куда то, поэтому мне кажется что возникновение таких аномалий ни что иное как лаги самих датчиков. В следствие этого стоит удалить такие данные

# In[42]:


pr = [x for x in df_train.columns if 'primary_cleaner' in x]


# In[43]:


less = df_train[df_train['primary_cleaner.output.sum'] < 1][[x for x in df_train.columns if 'primary_cleaner' in x]].reset_index(drop=True)


# In[44]:


more = df_train[df_train['primary_cleaner.output.sum'] > 60][[x for x in df_train.columns if 'primary_cleaner' in x]].reset_index(drop=True).loc[0:193]


# In[45]:


for i in [x for x in df_train.columns if 'primary_cleaner' in x]:
    print(f'{i} : ', (more-less)[i].mean())


# Наблюдаем что отклонения в похожих колонках присутсвуют, при этом опять же в tail незначительные отличия, поэтому решим что скорее всего нужно выбросить хвосты, т.к уж очень они подозрительные и неправдаподобные

# In[46]:


df_train[['rougher.output.sum', 'final.output.sum', 'primary_cleaner.output.sum']].boxplot()


# In[47]:


id1 = df_train[df_train['rougher.output.sum'] < 40].index


# In[48]:


id2 = df_train[df_train['primary_cleaner.output.sum'] < 40].index


# In[49]:


id3 = df_train[df_train['final.output.sum'] < 40].index


# In[50]:


all_id = id1.union(id2).union(id3)


# In[51]:


df_train.drop(all_id, inplace=True)


# **Комментарий** 
# 
# Смотря на графики boxplot-ов я в целом убеждаюсь в правильности того как я сделал срезы. Т.е то что я срезал значения <5 в целом не сильно портит хвосты и мы избавляемся от нулей

# ## 3. Модель

# ### Напишите функцию для вычисления итоговой sMAPE.

# In[52]:


def smape(true_val, predict):
    rougher, final = ((abs(true_val - predict) / ((abs(true_val) + abs(predict))/2)).mean() * 100)
    res = 0.25 * rougher + 0.75 * final
    return res 


# In[53]:


custom_scorer = make_scorer(smape, greater_is_better=False)


# In[54]:


target_train = df_train[['rougher.output.recovery', 'final.output.recovery']]
features_train = df_train[df_train.columns[df_train.columns.isin(df_test.columns)]].drop(['date'], axis=1)
target_test = df.set_index('date').loc[df_test.set_index('date').index,
                                       ['rougher.output.recovery','final.output.recovery']].reset_index(drop=True)
features_test=df_test.drop(['date'], axis=1)


# In[55]:


features_test


# In[56]:


features_train.shape, features_test.shape


# ### Обучите разные модели и оцените их качество кросс-валидацией. Выберите лучшую модель и проверьте её на тестовой выборке. Опишите выводы.

# In[60]:


pipe_lr = Pipeline([('mm',MinMaxScaler()),('lr', LinearRegression(n_jobs=-1))])
pipe_rfr = Pipeline([('mm',MinMaxScaler()),('rfr', RandomForestRegressor(n_estimators=15, random_state=12345, n_jobs=-1))])
pipe_knn = Pipeline([('mm',MinMaxScaler()),('knn', KNeighborsRegressor(n_jobs=-1))])

params_lr = dict(lr__fit_intercept=['True', 'False'])

params_rfr = dict(rfr__n_estimators=[20, 50, 100, 200],
                  rfr__min_samples_leaf=range(1,4))

params_knn = dict(knn__n_neighbors=range(5,10),
                  knn__metric=['minkowski', 'manhattan'],
                  knn__leaf_size=range(30,50,5),
                  knn__algorithm = ['auto', 'ball_tree', 'kd_tree'])


pipes = [pipe_lr, pipe_rfr, pipe_knn]
params = [params_lr, params_rfr, params_knn]
models = ['linear_regression',
          'random_forest',
          'nearest_neighbors']



for model, pipe, param in zip(models, pipes, params):
    
    search = RandomizedSearchCV(pipe, param, verbose=5, scoring=custom_scorer, cv=3, n_jobs=-1)
    
    search.fit(features_train, target_train)
    
    globals()[f'filename_{model}'] = os.path.join(save_dir, f'model.joblib.{model}')
    joblib.dump(search, globals()[f'filename_{model}'])
    
    print(model)
    print(search.best_score_)
    print(search.best_params_)
    print()


# In[61]:


dummy_regressor_rougher = DummyRegressor(strategy="median")
dummy_regressor_rougher.fit(features_train, target_train)
dummy_rougher_pred = dummy_regressor_rougher.predict(features_test)
smape_dummy_rougher = smape(target_test, dummy_rougher_pred)
print(smape_dummy_rougher)


# In[62]:


joblib.load(filename_random_forest)


# In[63]:


smape(target_test, joblib.load(filename_random_forest).predict(features_test))


# **Вывод**
# 
# Проверку на адекватность лучшая модель не прошла, по всей видимости либо задача просто не выполнима, в рамках машинного обучения, но как мне кажется модель оказалась просто переполнена не целевыми фичами, которые не влияют на целевой показатель, поэтому испытывает сложности в предсказаниях.

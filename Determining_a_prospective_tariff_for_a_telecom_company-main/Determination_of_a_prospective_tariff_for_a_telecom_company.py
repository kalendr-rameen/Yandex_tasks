#!/usr/bin/env python
# coding: utf-8

# # Определение перспективного тарифа для телеком компании
# 
# 

# **Описание проекта**
# 
# 
# Вы аналитик компании «Мегалайн» — федерального оператора сотовой связи. Клиентам предлагают два тарифных плана: «Смарт» и «Ультра». Чтобы скорректировать рекламный бюджет, коммерческий департамент хочет понять, какой тариф приносит больше денег.
# Вам предстоит сделать предварительный анализ тарифов на небольшой выборке клиентов. В вашем распоряжении данные 500 пользователей «Мегалайна»: кто они, откуда, каким тарифом пользуются, сколько звонков и сообщений каждый отправил за 2018 год. Нужно проанализировать поведение клиентов и сделать вывод — какой тариф лучше.
# 
# 1. [Открытие данных](#task1)
# 2. [Предобраюотка данных](#task2)
#    * [Обработка calls_df](#call_prep)
#    * [Обработка internet_df](#internat_prep)
#    * [Обработка messages_df](#messages_prep)
#    * [Обработка tariff_df](#tariff_prep)
#    * [Обработка users_df](#users_prep)
#    * [Получение значений кол-ва звонков и мунут разговора помесячно ](#calls_num)
#    * [Считаю кол-во отправленных сообщений по месяцам](#sms_num)
#    * [Считаю кол-во потраченных мб интернета в месяц](#internet_sum)
# 3. [Анализ данных](#data_analysis)
# 4. [Тестирование гипотез](#hyp_test)
# 5. [Общий вывод](#conc)

# <a id = "step1"></a>
# 
# # Шаг 1. Откройте файл с данными и изучите общую информацию
# Путь к файлам:
#     /datasets/calls.csv. Скачать датасет
#     /datasets/internet.csv. Скачать датасет
#     /datasets/messages.csv. Скачать датасет
#     /datasets/tariffs.csv. Скачать датасет
#     /datasets/users.csv. Скачать датасет
# 

# In[1]:


import pandas as pd 
import numpy
import numpy as np 
import matplotlib.pyplot as plt
from scipy import stats as st
import math
from matplotlib import pyplot


# In[2]:


calls_df = pd.read_csv("./calls.csv")
internet_df = pd.read_csv("./internet.csv")
messages_df = pd.read_csv("./messages.csv")
tariffs_df = pd.read_csv("./tariffs.csv")
users_df = pd.read_csv("./users.csv")


# In[3]:


calls_df.info()


# In[4]:


internet_df.info()


# In[5]:


messages_df.info()


# In[6]:


tariffs_df.info()


# In[7]:


users_df.info()


# **Комментарий(Шаг 1)** 
# 
# Все ошибки данные касаются в основном как можно увидеть либо даты неправильно поставлены и их нужно перевести в формат "datetime64" либо же нужно перевести тип float64 в int64 чтобы всё привексти к одному типу данных, либо же наоборот как в случае с таблцицей calls_df где колонка "user_id" в формате int64 => нужно привести к object . Так в случае с tarifffs_ds данные не нужны в предварительной обработке. В случае со столбом churn_date в таблице users_df то такое маленькое кол-во ненулевых значений можно обьяснить лишь тем что лишь малое число пользователей прекратило пользоваться услугами оператора "Мегалайн"

# <a id="task2"></a>
# 
# # 2. Подготовка данных
# 
# Необоходимо:<br> 
# 
#     1.Привести данные к нужным типам;<br>    
#     2.Найти и исправить ошибки в данных.<br>
# 
# Поясните, какие ошибки мы нашли и как их исправили. Обратите внимание, что длительность многих звонков — 0.0 минут. Это могут быть пропущенные звонки. Обрабатывать ли эти нулевые значения, решать вам — оцените, как их отсутствие повлияет на результаты анализа.
# 
#     
# 
# Необходимо посчитать для каждого пользователя:
# 
#     1.количество сделанных звонков и израсходованных минут разговора по месяцам;
#     2. количество отправленных сообщений по месяцам;
#     3.объем израсходованного интернет-трафика по месяцам;
#     4. помесячную выручку с каждого пользователя (вычтите бесплатный лимит из суммарного количества звонков, сообщений и интернет-трафика.

# <a id="call_prep"></a>
# ## Обработка calls_df

# In[8]:


calls_df.info()


# In[9]:


calls_df.head()


# In[10]:


calls_df['call_date'].head()


# Перевожу колонку `call_date` в формат datetime64(данные расположены в формате YY:mm:dd)
# 
# колонку `user_id` перевожу в формат object

# In[11]:


calls_df['call_date'] = pd.to_datetime(calls_df['call_date'], format='%Y-%m-%d')


# In[12]:


calls_df['user_id'] = calls_df['user_id'].astype('object')


# In[13]:


calls_df['duration'] = numpy.ceil(calls_df['duration'])


# Оставил колнку `duration` не изменяя её т.к int нельзя

# **Комментарий**
# 
# Округлил колонку `duration` методом numpy.ceil

# In[14]:


calls_df.head(100)


# In[15]:


calls_df


# На этом этапе я пытаюсь понять откуда взялись эти нули в наших данных потому, поэтому я смотрю на лидеров по  количеству нулей

# In[16]:


calls_df[calls_df['duration'] == 0].groupby('user_id')['duration'].count().sort_values(ascending=False).head(20)


# In[17]:


calls_df[calls_df['duration'] == 0].groupby('user_id')['duration'].count().sort_values(ascending=False).head(50).index


#  Здесь я просто смотрю на лидеров по кол-ву звонков любых включая нули

# In[18]:


calls_df.groupby('user_id')['duration'].count().sort_values(ascending=False).head(20)


# In[19]:


calls_df.groupby('user_id')['duration'].count().sort_values(ascending=False).head(50).index


# Здесь я подсчитываю сколько из 50 лидеров с нулями и 50 лидеров всего по продолжительности и сравниванию не одни и те же ли это пользователи.

# In[20]:


sum = 0 
for i in calls_df[calls_df['duration'] == 0].groupby('user_id')['duration'].count().sort_values(ascending=False).head(50).index:
    if i in calls_df.groupby('user_id')['duration'].count().sort_values(ascending=False).head(50).index:
        sum +=1


# In[21]:


sum


# **Вывод**

# 43 из 50 совпадают => Всё достаточно логично чем больше звонков тем больше непринятых звонков, так как в следующих заданиях требуется посчитать количество израсходованных минут то замена как мне кажется не совсем будет корректа так как мы получим искаженные данные  ведь если всё заменить нулями то мы будем анализировать данные дальше о продолжительности звонков, когда будем считать сколько прибыли пришло и замена этих данных на медианную исказит конечный результат. Поэтом нули я решил оставить

# <a id="internat_prep"></a>
# ## Обработка internet_df

# Можно заметить после вывода `internet_df.head(10)` есть колонка Unnamed:0, которая нам ни к чему позже в коде я её удалю

# In[22]:


internet_df.head(10)


# In[23]:


internet_df.info()


# In[24]:


del internet_df['Unnamed: 0']


# Видно что данные в колонке `session_date` расположены в формате YY:mm:dd, поэтому нужно заменить эту колонку на соответсвуютщий формат datetime64

# In[25]:


internet_df['session_date'] =  pd.to_datetime(internet_df['session_date'], format="%Y-%m-%d")


# In[26]:


#internet_df['mb_used'] = internet_df['mb_used'].astype('int64')
internet_df['mb_used'] = numpy.ceil(internet_df['mb_used'])


# Колонку `mb_used` оставил тогда как есть ибо на int нельзя т.к округление в меньшую сторону. Заменил `user_id` на тип object, хотя это совсем не обязательно, но так лучше понять что же именно содежится в колонке есть забыл ее значение

# In[27]:


internet_df['user_id'] = internet_df['user_id'].astype('object')


# In[28]:


internet_df.info()


# **Комментарий** 
# 
# Перевел колонки в int64 которые требовали этого, удалил колонку Unnamed:0 т.к по сути она являлась индексом данных, который уже есть. И перевел даты в формат datetime64.

# <a id="messages_prep"></a>
# ## Обработка messages_df

# In[29]:


messages_df.info()


# In[30]:


messages_df.head(10)


# In[31]:


messages_df['user_id'] = messages_df['user_id'].astype('object')


# Переведём колонку `message_date` к формату datetime64, форматированием YY:mm:dd

# In[32]:


messages_df['message_date'] = pd.to_datetime(messages_df['message_date'], format='%Y-%m-%d')


# In[33]:


messages_df.info()


# **Комментарий** 
# Дату перевел в нужный формат. Заменил не числовую колонку user_id, на тип данных object

# <a id = "tariff_prep"></a>
# ## Обработка Tarrifs_df

# In[34]:


tariffs_df.info()


# In[35]:


tariffs_df.head(10)


# Как я уже упоминал эти данные править не нужно

# <a id = "users_prep"></a>
# ## Обработка Users_df

# In[36]:


users_df.info()


# In[37]:


users_df['user_id'] = users_df['user_id'].astype('object')


# Заполняю колонку `churn_date` нулями чтобы потом перевести эту колонку в datetime64, где нули заменятся очень древней датой

# In[38]:


users_df['churn_date'] = users_df['churn_date'].fillna(0)


# In[39]:


users_df['churn_date']


# Сделал замену в колонках `churn_date` и `reg_date` в формат datetime64, форматированием типа YY:dd:mm. Все значения где были 0-и заполнилось значением 1970-01-01 таким образом мы сможем делать сравнения т.к получается что эта дата всегда будет раньше чем дата регистрации 

# In[40]:


# Все значения где были 0-и заполнилось значением 1970-01-01 таким образом мы сможем делать сравнения 
# т.к получается что эта дата всегда будет раньше чем дата регистрации 
users_df['churn_date'] = pd.to_datetime(users_df['churn_date'], format='%Y-%m-%d')
users_df['reg_date'] = pd.to_datetime(users_df['reg_date'], format="%Y-%m-%d")


# In[41]:


users_df['reg_date'] = pd.to_datetime(users_df['reg_date'], format="%Y-%m-%d")


# In[42]:


users_df.info()


# **Комментарий**
# 
# Заменил нулевые значений в колонке churn_date на 0-ли а затем перевел эту колонку в нужный тип datedime64, также было сделано с колонкой reg_date. User_id колонка переведена к типу object

# In[43]:


internet_df.info()


# **Комментарий к v3**
# 
# Исправил правда я думал что если мы оставим данные типа float без округления это как то изменит наши данные? Заранее извиняюсь если это совсем глупый вопрос но не очень понимаю обоснованность такого действия. Разве что только если округлять вверх то работодатель всегда будет чуть больше доволен?)

# <a id ="calls_num"></a>
# ## Количество сделанных звонков и израсходованных минут разговора по месяцам

# In[44]:


calls_df.info()


# Создаю колонку `month` для того чтобы в дальнейшем сделать сводную таблцицу базируясь на этой колонке

# In[45]:


calls_df['month'] = pd.DatetimeIndex(calls_df['call_date']).month


# In[46]:


calls_df.head()


# Делаю сводную талицу где использую как раз новую колонку `month`, с применением метода 'count' т.о получаю таблицу с кол-ом звонков в месяц

# In[47]:


calls_df


# In[48]:


call_dur_count = pd.pivot_table(
    calls_df, index='user_id', columns='month', values='duration', aggfunc='count', #fill_value=0, dropna=False
)


# In[49]:


call_dur_count.head(10)


# In[50]:


call_dur_count.info()


# Делаю сводную таблицу со столбцом `month` только здесь использую метод `.sum` т.о получаю продолжительность суммарную звонков за месяц для каждого пользователя

# In[51]:


call_dur_sum = pd.pivot_table(
    calls_df, index='user_id', columns='month', values='duration', aggfunc='sum', #fill_value=0, dropna=False
)


# In[52]:


call_dur_sum


# In[53]:


call_dur_sum.info()


# In[54]:


# Видно что NaN возникает в определенные месяцы в следствие того что подключение связи происходи позже
users_df[users_df['user_id'] == 1000]


# **Вывод**
# 
# Посчитали кол-ва звонков и сумму продолжительности звонков построив две сводные таблицы, опираясь на данные полученные в колонке `month` и `duration`

# <a id = 'sms_num'></a>
# ## Количество отправленных сообщений по месяцам

# In[55]:


messages_df


# Также как и в прошлом случае добавляю колонку `month` чтобы можно было дальше строить замечательные сводные таблицы)

# In[56]:


messages_df['month'] = pd.DatetimeIndex(messages_df['message_date']).month


# Строю сводную таблицу на слолбцах `user_id` и `month` методом `.count`

# In[57]:


messages_count = pd.pivot_table(
    messages_df, index='user_id', columns='month', values='id', aggfunc='count', #fill_value=0, dropna=False
)


# In[58]:


messages_count.head(10)


# In[59]:


messages_count.info()


# **Вывод**
# 
# Сделали сводную таблицу в которой посчитали число отправленных смс помесячно для каждого пользователя

# <a id='internet_sum'></a>
# ## Объем израсходованного интернет-трафика по месяцам

# Добавляю колонку `month`, чтобы дальше построить свобную таблицу для расчёта израсходованного трафика помесячно

# In[60]:


internet_df['month'] = pd.DatetimeIndex(internet_df['session_date']).month


# Составляю своднуб таблицу на основание полученной колонки `month` методом `.sum`

# In[61]:


internet_mb_sum = pd.pivot_table(
    internet_df, index='user_id', columns='month', values='mb_used', aggfunc='sum', #fill_value=0
)


# In[62]:


internet_mb_sum.head(10)


# In[63]:


internet_mb_sum.info()


# **Вывод**
# 
# Посчитали кол-во потраченных мб интернета в месяц с помощью сводной таблицы, как видно на всех предыдущих шагах в том числе число пользователей растёт по мере прохождения года.

# ## Помесячную выручка с каждого пользователя 
# Нужно вычесть бесплатный лимит из суммарного количества звонков, сообщений и интернет-трафика; умножить остаток на значение из тарифного плана; прибавьте абонентскую плату, соответствующую тарифному плану.

# In[64]:


tariffs_df


# In[65]:


all_col = []


# In[66]:


# Я не знаю как сделать это проще =( Суть его в том чтобы пройтись по всем колонками каждого индекса из таблицы 
# users_df и попытаться посчитать кол-во потраченных смс, минут, интернета в месяц а затем расчитать для 
# каждого месяцы какая выручка была от каждого клиета
for i in users_df.index:
    
    user_id = users_df.loc[i]['user_id'] 
    #print(user_id)
    tariff = users_df.loc[i]['tariff']
    tariff_data = tariffs_df[tariffs_df['tariff_name'] == tariff]
    col = []
    # на этом этапе записываю значения колонок для tariff и колнку с данным тарифом для каждого юзера отдельно
    # эти занчения нам пригодятся в дальнейшем
    if tariff == 'smart':
# проверка тарифа, т.к у меня проблема возникала с инкесацией поэтому здесь две идентичных части кода
        minutes_included = tariff_data['minutes_included'][0]# - call_dur_sum.loc[user_id][0]
        minutes_price = tariffs_df['rub_per_minute'][0]
        messages_included = tariff_data['messages_included'][0]
        messages_price = tariffs_df['rub_per_message'][0]
        mb_included = tariff_data['mb_per_month_included'][0]
        gb_price = tariffs_df['rub_per_gb'][0]
        month_fee = tariffs_df['rub_monthly_fee'][0]
        # Здесь я считаю всевозможные значения цен на доп услуги и кол-во предоставляемых услуг в пакете
        # Всё это нужно посчитать чтобы потом просто вычитать из кол-ва потраченного кол-во предоставленного
        # и перемножать на значение цен за доп услуга
        for i in range(1,13):# Здесь прохожусь по месяцам
            try:
                spent_minutes = call_dur_sum.loc[user_id][i]
            except:
                spent_minutes = 0  
            try:
                spent_messages = messages_count.loc[user_id][i]
            except:
                spent_messages = 0
            try:
                spent_mb = internet_mb_sum.loc[user_id][i]
            except:
                spent_mb = 0
            #часть кода с исключениями сделана потому что там возникает такая ситуация что порой бывает что юзер
            # не пользовался той или иной услугой и поэтому его нет в таблице такие значения я брал за нули(0)
            #print(spent_mb, spent_messages, spent_minutes)
            if spent_mb == 0 and spent_messages == 0 and spent_minutes == 0:
                col.append(0)
            # Если пользователь не пользовалься ничем в течение месяца то скорее всего он не подключен ещё был
            # на тот месяц
            else:
                # Здесь происходит расчёты кол-ва потраченного/недотраченного, в общем то считается
                # прибыль от клиента
                excess_minutes = spent_minutes - minutes_included
                if excess_minutes < 0 : 
                    # цикл проверок если потраченное меньше включенного в тариф то мы не сможем
                    # списать с него доп плату кроме как по тарифу поэтому ставим 0
                    excess_minutes = 0
                excess_messages = spent_messages - messages_included
                if excess_messages < 0 :
                    excess_messages = 0
                excess_gb = (spent_mb - mb_included) / 1024# деление произвожу т.к указана цена за каждый ГБ
                if excess_gb < 0 :
                    excess_gb = 0
                price =  (
                          (excess_minutes * minutes_price) + 
                          (excess_gb * gb_price) + 
                          (excess_messages * messages_price) +
                          (month_fee)  
                         )
                col.append(price)
                # Добавляем значение для каждого юзера (одного) сколько на нём заработали в i-ый месяц
                # Вторая часть кода как я уже писал идентична, поэтому я не стал её комментировать
                # Далее как только по всем мясяцам прошлись то добавляем полученный список в список all_col, где
                # буду храниться значния для каждого юзера в порядке который был в таблице users_df
    elif tariff == 'ultra':
        minutes_included = tariff_data['minutes_included'][1]# - call_dur_sum.loc[user_id][0]
        minutes_price = tariffs_df['rub_per_minute'][1]
        messages_included = tariff_data['messages_included'][1]
        messages_price = tariffs_df['rub_per_message'][1]
        mb_included = tariff_data['mb_per_month_included'][1]
        gb_price = tariffs_df['rub_per_gb'][1]
        month_fee = tariffs_df['rub_monthly_fee'][1]
        for i in range(1,13):
            try:
                spent_minutes = call_dur_sum.loc[user_id][i]
            except:
                spent_minutes = 0
            try:
                spent_messages = messages_count.loc[user_id][i]
            except:
                spent_messages = 0
            try:
                spent_mb = internet_mb_sum.loc[user_id][i]
            except:
                spent_mb = 0
            if spent_mb == 0 and spent_messages == 0 and spent_minutes == 0:
                col.append(0)
            else:
                excess_minutes = spent_minutes - minutes_included
                if excess_minutes < 0 :
                    excess_minutes = 0
                excess_messages = spent_messages - messages_included
                if excess_messages < 0 :
                    excess_messages = 0
                excess_gb = (spent_mb - mb_included) / 1024 
                if excess_gb < 0 :
                    excess_gb = 0
                price =  (
                          (excess_minutes * minutes_price) + 
                          (excess_gb * gb_price) + 
                          (excess_messages * messages_price) +
                          (month_fee)  
                         )
                col.append(price)
        
    print(col, "| user_id:{}".format(user_id))
    all_col.append(col)# Вот добавления для каждого юзера


# In[67]:


# Вроде всё сходится и данные похожи на правду 
users_df[users_df['user_id'] == 1003]


# In[68]:


len(all_col)


# In[69]:


np.array(all_col)


# Так как у нас в полученном списке all_col значения потраченного для каждого юзера отдельно значит нам можно составить из этого датафрейм где колонками будут значения от 1 до 12 и с индексами из таблица колонкой `user_id` из таблицы users_df

# In[70]:


profit_per_client_per_month = pd.DataFrame(np.array(all_col), columns=range(1,13), index= users_df['user_id'])


# In[71]:


profit_per_client_per_month


# <a id='data_analysis'></a>
# # Шаг 3. Проанализируйте данные

# In[72]:


struct_calls = calls_df.pivot_table(index=["month","user_id"], values='duration', aggfunc='sum')


# In[73]:


struct_internet = internet_df.pivot_table(index=['month', 'user_id'], values = 'mb_used', aggfunc='sum')


# In[74]:


struct_messages = messages_df.pivot_table(index=['month', 'user_id'], values= 'message_date', aggfunc='count')


# In[75]:


struct_table = struct_calls.join(struct_internet).join(struct_messages)


# In[76]:


struct_table = struct_table.reset_index()


# In[77]:


struct_table


# Добавляю тарифы для среза дальнейшего

# In[78]:


def tariff(row):
    user_id = row['user_id']
    return users_df[users_df['user_id'] == user_id].reset_index()['tariff'][0]


# In[79]:


struct_table['tariff'] = struct_table.apply(tariff, axis=1)


# In[80]:


struct_table


# Создаём срезы

# In[81]:


smart_struct = struct_table[struct_table['tariff'] == "smart"]


# In[82]:


ultra_struct = struct_table[struct_table['tariff'] == 'ultra']


# Строем как повышается среднее значение параметра параметр от месяца

# In[83]:


struct_table.pivot_table(index='month', values='duration').reset_index().plot(x='month', y='duration', kind='bar')


# In[84]:


struct_table.pivot_table(index='month', values='mb_used').reset_index().plot(x='month', y='mb_used', kind='bar')


# In[85]:


struct_table.pivot_table(index='month', values='message_date').reset_index().plot(
    x='month', y='message_date', kind='bar'
)


# **Вывод**
# 
# Видно что с течением времени кол-во обьём используемых услуг растёт

# # Считаем значения для smart

# Расчёты по продолжительности разговора

# In[86]:


smart_struct


# **Итак здесь я считаю средние значения, дисперсию и стандартное отклонение для тарифа `smart` и `ultra` чтобы охарактеризователь такие показатели как длительность разговора, кол-во потраченных мб, число отправленных смс. Для этого я беру нужные соответствующие столюцы и рассчитываю все показатели**

# In[87]:


smart_struct['duration'].var()


# In[88]:


math.sqrt(smart_struct['duration'].var())


# In[89]:


smart_struct['duration'].median()


# Расчёты по кол-ву использованного трафика

# In[90]:


smart_struct['mb_used'].var()


# In[91]:


math.sqrt(smart_struct['mb_used'].var())


# In[92]:


smart_struct['mb_used'].median()


# Расчёты по числу использованных смс

# In[93]:


smart_struct['message_date'].var()


# In[94]:


math.sqrt(smart_struct['message_date'].var())


# In[95]:


smart_struct['message_date'].median()


# # Ultra

# Расчёты по продолжительности разговора

# In[96]:


ultra_struct


# In[97]:


ultra_struct['duration'].var()


# In[98]:


math.sqrt(ultra_struct['duration'].var())


# In[99]:


ultra_struct['duration'].median()


# Расчёты по кол-ву используемого траффика

# In[100]:


ultra_struct['mb_used'].var()


# In[101]:


math.sqrt(ultra_struct['mb_used'].var())


# In[102]:


ultra_struct['mb_used'].median()


# Расчёты по числу смс-ок

# In[103]:


ultra_struct['message_date'].var()


# In[104]:


math.sqrt(ultra_struct['message_date'].var())


# In[105]:


ultra_struct['message_date'].median()


# Вывод:
# 
# Ultra:
#   
#   **duration**(Продолжительность разговора для Ultra)
#       среднее: 94203
#       
#       стандартное отклонение:307
#       
#       дисперсия:528
#       
#   **mb_used**(Число потраченных мегабайт для Ultra)
#       среднее: 19446
#       
#       стандартное отклонение:9988
#       
#       дисперсия:99750820
#   
#   **message_date**(Кол-во потраченных смс для Ultra)
#   
#       среднее:52
#       
#       стандартное отклонение:44
#       
#       дисперсия:2000
#       
#           
# Smart:
#   
#   **duration**(Продолжительность разговора для Smart)
# 
#       среднее: 423
#       
#       стандартное отклонение: 189
#       
#       дисперсия: 35844
#       
#       
#       
#   **mb_used**(Число потраченных мегабайт для Smart)
#   
#       среднее: 16629
#       
#       стандартное отклонение:5875
#       
#       дисперсия:34510889
#   
#   **message_date**(Кол-во потраченных смс для Smart)
#   
#       среднее: 34
#       
#       стандартное отклонение:26
#       
#       дисперсия: 719
#       
#           
# Таким образом видно что все средние значения больше у ultra, в следствие этого предпогаю что расходуют они больше нежели smart

# Сделаем для каждого вида услуг сводную таблицу. Всё сделаем как указано выше. 

# In[106]:


struct_table.pivot_table(index='tariff', values='duration', aggfunc=['median', np.std, np.var])


# In[107]:


struct_table.pivot_table(index='tariff', values='mb_used', aggfunc=['median', np.std, np.var])


# In[108]:


struct_table.pivot_table(index='tariff', values='message_date', aggfunc=['median', np.std, np.var])


# Вывод:
# 
# Ultra:
#   
#   **duration**(Продолжительность разговора для Ultra)
#   
#       среднее: 528
#       
#       стандартное отклонение:307
#       
#       дисперсия:94203
#       
#   **mb_used**(Число потраченных мегабайт для Ultra)
#   
#       среднее: 19446
#       
#       стандартное отклонение:9988
#       
#       дисперсия:99750820
#   
#   **message_date**(Кол-во потраченных смс для Ultra)
#   
#       среднее:52
#       
#       стандартное отклонение:44
#       
#       дисперсия:2000
#       
#           
# Smart:
#   
#   **duration**(Продолжительность разговора для Smart)
# 
#       среднее: 423
#       
#       стандартное отклонение: 189
#       
#       дисперсия: 35844
#       
#       
#       
#   **mb_used**(Число потраченных мегабайт для Smart)
#   
#       среднее: 16629
#       
#       стандартное отклонение:5875
#       
#       дисперсия:34510889
#   
#   **message_date**(Кол-во потраченных смс для Smart)
#   
#       среднее: 34
#       
#       стандартное отклонение:26
#       
#       дисперсия: 719
#       
#           
# Таким образом видно что все средние значения больше у ultra, в следствие этого предпогаю что расходуют они больше нежели smart

# **Графики**

# In[109]:


fig, ax0 = plt.subplots()
ax0.hist(ultra_struct['message_date'], alpha=0.5, label='ultra', bins=30, density=True)
ax0.hist(smart_struct['message_date'], alpha=0.5, label='smart', bins=30, density=True)
ax0.legend(loc='upper right')
ax0.set_title("Кол-во отправленных смс")
pyplot.show()


# In[110]:


fig, ax1 = plt.subplots()
ax1.hist(ultra_struct['mb_used'], alpha=0.5, label='ultra', bins=30, density=True)
ax1.hist(smart_struct['mb_used'], alpha=0.5, label='smart', bins=30, density=True)
ax1.legend(loc='upper right')
ax1.set_title("Кол-во использованных мб(мегабайт)")
pyplot.show()


# In[111]:


fig, ax2 = plt.subplots()
ax2.hist(ultra_struct['duration'], alpha=0.5, label='ultra', bins=30, density=True)
ax2.hist(smart_struct['duration'], alpha=0.5, label='smart', bins=30, density=True)
ax2.legend(loc='upper right')
ax2.set_title("Продолжительность звонка")
pyplot.show()


# Видно что в целом каждый раз пользователи пользователи тарифа `ultra` намного чаще пользуются большими значениями всех исследуемых параметров(т.е дольше говорям, больше отправляют смс и т.д) чем пользователи тарифа `smart`. Таким образом среднее и медиана будут смещаться для тарифа `ultra`в большую строну нежели для `smart`. Получается что пользователи `ultra` в среднем больше используют предоставляемого им трафика, несмотря даже порой на наличие выбросов (например как при расчёте кол-ва мб). Таким образом `ultra` выбирают те кто просто больше пользуется услугами мобильной связи 

# In[112]:


struct_table


# In[113]:


def profit(row):
    user_id = row['user_id']
    month = row['month']
    return profit_per_client_per_month.loc[user_id,month]


# In[114]:


struct_table['profit'] = struct_table.apply(profit, axis=1)


# In[115]:


struct_table


# In[116]:


struct_table.pivot_table(index='tariff', values='profit', aggfunc=[np.mean,np.std,np.var])


# Средняя выручка по smart и по ultra отличается, причём у ультра среднее больше, а стандартное отклонение меньше, поэтому здесь даже без теста можно сказать что среднее ген.сов-ти больше у ультра. Но вообще общий вывод что ultra тратит больше денег

# <a id='hyp_test'></a>
# # Шаг 4. Проверьте гипотезы

# Проверка первой гипотезы, которая гласит о том что средняя выручка пользователей тарифов «Ультра» и «Смарт» различается; Чтобы ее проверить возьмём срезы для выручки по тарифу. Далее посчитаем медианное значение от всех чтобы понять что всё же сколько в среднем приходит денег от каждого пользователся а далее делаем t-test

# In[117]:


users_df


# In[118]:


profit_per_client_per_month['tariff'] = users_df['tariff']


# In[119]:


users_df['tariff']


# Добавляем колонку `tariff` чтобы дальше по ней делать срез

# In[120]:


profit_per_client_per_month['tariff'] = users_df.set_index("user_id")['tariff']


# In[121]:


profit_per_client_per_month


# Считаем значения для каждого юзера помесячно сколько он тратит медианно за все 12 месяцев

# In[122]:


median_profit= []
for i in profit_per_client_per_month.index:
    count = []
    for j in range(1,13):
        if profit_per_client_per_month.isna().loc[i, j] == False:
            count.append(profit_per_client_per_month.loc[i, j])
    count.sort()
    mid = len(count) // 2
    res = (count[mid] + count[~mid]) / 2
    median_profit.append(res)
    #duration.append(sum/count)            


# Я сделал такую таблицу но сделать привести ее к такому виду все равно использует цикл в моём случае, и даже если сделать такую форму то группировку всё равно не получится вроде сделать, есть ли методы как проще привести таблицу к такому виду? Дальше я буду использовать этот датафрейм `profit_per_client_per_month_2` при расчёте t-test'ов
# 
# Часть кода где я это делаю помечена снизу и сверху вот так: -------------

# добавляем в таблицу `"profit_per_client_per_month"` колонку `med_profit`содержащую значения из median_profit, где хранится медианные значения для каждого юзера(т.е это медиана для одной строки по сути)

# # -------------------------------------------------------------------------------------------

# In[123]:


all_rows = []
num = 0 

for i in range(1,13):
    all_rows.append(profit_per_client_per_month[i].reset_index(drop=True))


# In[124]:


profit_per_client_per_month_2 = pd.DataFrame(all_rows)


# In[125]:


profit_per_client_per_month_2.columns = range(1000,1500)


# In[126]:


profit_per_client_per_month_2


# # -------------------------------------------------------------------------------------------

# In[127]:


profit_per_client_per_month['med_profit'] = pd.Series(median_profit, index=profit_per_client_per_month.index)


# # Вывод 1-го теста

# Исходя из данных теста разцица между сравниваемыми значениями статистически значима(p-value < 0.05) а значит разница между пользователями смарт и ультра существует

# Проверяем гипотезу: **средняя выручка пользователей из Москвы отличается от выручки пользователей из других регионов**
# 
# В данном случае мы проверяем значима ли разница медиан выручки между жителями столцы и нет соответственно. Альтернативная же гипотеза состоит в том что разница есть. Здесь мы также делаем срез по гододам. Далее мы снова проводим тест о равенстве средних генеральных совокупностей, но так как у нас лишь выборка то пользуемся t-test'ом. Тест двухсторонний так как нам не важно кто больше важно лишь признать факт разницы
# 
# H0: med1 - med2 = 0
# 
# H1: |med1 - med2| > 0
# 
# Формулировка гипотезы H0: Среднее выручка от двух генеральных совокупностей для жителей Москвы и из окраин(!=Москва) не отличаются и равны друг другу
# 
# Формулировка гипотезы H1: Средняя выручка двух генеральных совокупностей для жителей Москвы и из окраин(!=Москва) отличаются и не равны друг другу

# Делаем колонку город на основание места проживания пользователя где разделяем их на москвичей и не москвичей соотв. когда будем делать срез

# In[128]:


profit_per_client_per_month['city'] = users_df.set_index('user_id')['city']


# Уровень значимости берем как 0.05/2 т.к гипотеза друхсторонняя. Проводим t-test на полученных данный. Передаю на вход выборку из медианных значений для каждого юзера за 12 месяцев.

# Собираю все данные в один список, так все москвичи в all_data_moscow и с окраин all_data_non_moscow. Там хранятся все выручки с каждого месяца для каждого пользователя принадлежащего к условиям задачи

# In[129]:


all_data_moscow = []
for i in profit_per_client_per_month[profit_per_client_per_month['city'] == 'Москва'].index:
    for j in profit_per_client_per_month_2[i].dropna(axis=0):
        all_data_moscow.append(j)


# In[130]:


all_data_non_moscow = []
for i in profit_per_client_per_month[profit_per_client_per_month['city'] != 'Москва'].index:
    for j in profit_per_client_per_month_2[i].dropna(axis=0):
        all_data_non_moscow.append(j)


# In[131]:


results_2 = st.ttest_ind(all_data_moscow,all_data_non_moscow)


# In[132]:


results_2


# In[133]:


if 0.05 > results_2.pvalue/2:
    print( 'p-value =',results.pvalue , ",Отвергаем H0 гипотезу")
else:
    print("Принимаем H0")


# # Вывод 2-го теста

# Исходя из данных теста разцица между сравниваемыми значениями статистически не значима(p-value > 0.025) а значит разница между пользователями из столицы и из других город отсутствует.

# <a id ='conc'></a>
# # Шаг 5. Напишите общий вывод

# Выручка по `smart` и по `ultra` отличается, причём в большую строну у `ultra`, к тому же стандартное отклонение по выручке у `ultra` меньше, чем у `smart`, что даёт ещё больше поводов утверждать что `ultra` тратит ежемесячно в среднем больше чем пользователь `smart`

# По данным видно что с течением времени а именно с 1 по 12 месяцы количество клиентов растёт и к 12 меяцу достигает
# максимума поэтому распреления выглядят более нормальными.
# 
# Вывод по различиям между тарифами `smart` и `ultra`:
# 
# Ultra:
#   
#   **duration**(Продолжительность разговора для Ultra)
#       среднее: 94203
#       
#       стандартное отклонение:307
#       
#       дисперсия:528
#       
#   **mb_used**(Число потраченных мегабайт для Ultra)
#       среднее: 19446
#       
#       стандартное отклонение:9988
#       
#       дисперсия:99750820
#   
#   **message_date**(Кол-во потраченных смс для Ultra)
#   
#       среднее:52
#       
#       стандартное отклонение:44
#       
#       дисперсия:2000
#       
#           
# Smart:
#   
#   **duration**(Продолжительность разговора для Smart)
# 
#       среднее: 423
#       
#       стандартное отклонение: 189
#       
#       дисперсия: 35844
#       
#       
#       
#   **mb_used**(Число потраченных мегабайт для Smart)
#   
#       среднее: 16629
#       
#       стандартное отклонение:5875
#       
#       дисперсия:34510889
#   
#   **message_date**(Кол-во потраченных смс для Smart)
#   
#       среднее: 34
#       
#       стандартное отклонение:26
#       
#       дисперсия: 719
#       
#           
# Таким образом видно что все средние значения больше у ultra, в следствие этого предпогаю что расходуют они больше нежели smart
# 
# Даже здесь видно разброс меньше среднее больше, тест видимо не врёт!
# 
# **После проведения двух гипотез оказалось что на выручку влияет выбранный тариф клиентом, но не влияет город проживания абонента** 
# 
#     Таким образом отвечая на главный вопрос данного задания:"Какой тариф лучше для того что правильно скорректировать бюджет рекламной компании?" Получается что тариф ультра приносит больше прибыль поэтому нам нужно лишь более серьёзно афишировать этот тариф чтобы всё больше людей приходило и брало этот тариф и как следствие платило больше денег, либо же больше вложиться в  рекламу тарифа смарт чтобы увеличить больше охват клиентов. Но сам по себе для компании выгоднее привлекать на ультра так как клиенты на нём больше платят денег!

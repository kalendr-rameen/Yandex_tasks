#!/usr/bin/env python
# coding: utf-8

# # Задача: Исследование объявлений о продаже квартир
# 
# В вашем распоряжении данные сервиса Яндекс.Недвижимость — архив объявлений о продаже квартир в Санкт-Петербурге и соседних населённых пунктах за несколько лет. Нужно научиться определять рыночную стоимость объектов недвижимости. Ваша задача — установить параметры. Это позволит построить автоматизированную систему: она отследит аномалии и мошенническую деятельность. 
# 
# По каждой квартире на продажу доступны два вида данных. Первые вписаны пользователем, вторые получены автоматически на основе картографических данных. Например, расстояние до центра, аэропорта, ближайшего парка и водоёма.

# **Описание данных**<br>
# 
#     airports_nearest — расстояние до ближайшего аэропорта в метрах (м)
#     balcony — число балконов
#     ceiling_height — высота потолков (м)
#     cityCenters_nearest — расстояние до центра города (м)
#     days_exposition — сколько дней было размещено объявление (от публикации до снятия)
#     first_day_exposition — дата публикации
#     floor — этаж
#     floors_total — всего этажей в доме
#     is_apartment — апартаменты (булев тип)
#     kitchen_area — площадь кухни в квадратных метрах (м²)
#     last_price — цена на момент снятия с публикации
#     living_area — жилая площадь в квадратных метрах(м²)
#     locality_name — название населённого пункта
#     open_plan — свободная планировка (булев тип)
#     parks_around3000 — число парков в радиусе 3 км
#     parks_nearest — расстояние до ближайшего парка (м)
#     ponds_around3000 — число водоёмов в радиусе 3 км
#     ponds_nearest — расстояние до ближайшего водоёма (м)
#     rooms — число комнат
#     studio — квартира-студия (булев тип)
#     total_area — площадь квартиры в квадратных метрах (м²)
#     total_images — число фотографий квартиры в объявлении

# <h1>Содержание<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Изучение-данных-из-файла" data-toc-modified-id="Изучение-данных-из-файла-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Открытие файла с данными и изучение общей информации</a></span></li><li><span><a href="#Предобработка-данных" data-toc-modified-id="Предобработка-данных-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Предобработка данных</a></span></li><li><span><a href="#Расчёты-и-добавление-результатов-в-таблицу" data-toc-modified-id="Расчёты-и-добавление-результатов-в-таблицу-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Расчёты и добавление результатов в таблицу</a></span></li><li><span><a href="#Исследовательский-анализ-данных" data-toc-modified-id="Исследовательский-анализ-данных-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Исследовательский анализ данных</a></span></li><li><span><a href="#Общий-вывод" data-toc-modified-id="Общий-вывод-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Общий вывод</a></span></li><li><span><a href="#Чек-лист-готовности-проекта" data-toc-modified-id="Чек-лист-готовности-проекта-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Чек-лист готовности проекта</a></span></li></ul></div>

# ## 1. Открытие файла с данными и изучение общей информации

# In[1]:


import pandas as pd
df = pd.read_csv("./real_estate_data.csv", sep="\t")


# In[2]:


#Сделал переиндексирование потому что просто удобнее 
#смотреть на описание данных
df = df.reindex(sorted(df.columns), axis=1)


# In[3]:


df['is_apartment'].value_counts()


# In[4]:


df.info()


# In[5]:


#Сразу бросается в глаза огромное кол-во NaN
df.head()


# **Вывод**

# После первого осмотра данных бросается в глаза, что очень много значений NaN, и очень соответсвенно много колонок типа float => Нужно заменить данные на более логичные чтобы все привести к одному типу а именно int64

# In[6]:


# a = df.select_dtypes(include='float64')
# df.isna().any()
# df[df['kitchen_area'].isna()].info()
# df[df['living_area'].isna()].info()
# df[df['floors_total'].isna()].isna().any()
# df = df.reindex(sorted(df.columns), axis=1)


# ## 2. Предобработка данных

# In[7]:


#Данные далеко не идеальны нужно многое заменять на что то правдоподо-
#бное для определнных групп

df.isna().any()


# In[8]:


df[df['airports_nearest'].isna()].info()


# In[9]:


df.query('locality_name == "Санкт-Петербург"')['airports_nearest'].median()


# In[10]:


# Видно что медиана не слишком отличается от медианы только 
#лишь по питеру, видимо те кто живут за городом не хотят заполнять
# по сути не нужную информацию, т.к те кто покупает дом за городом
# вряд ли смотрят расстояние до аэропорта из за того что до него 
# просто по дефолту далеко
df['airports_nearest'].median()


# In[11]:


all_places = df[df['airports_nearest'].isna()]['locality_name'].value_counts().index


# In[12]:


for i in all_places:
    print(i)


# Как я уже писал выше видно что не заполняют лишь люди живущие на окраине области т.е не в городе в следствие этого я решил ввести колонку с типом местности area_type, который как мне кажется в дальнейшем анализе может сильно помочь

# In[13]:


def area_type(row):
    area = row['locality_name']
    if area == "Санкт-Петербург":
        return "город"
    else:
        return "область"


# In[14]:


df['area_type'] = df.apply(area_type, axis = 1)


# In[15]:


df.groupby('area_type')['locality_name'].count().sum()


# In[16]:


df.info()


# In[17]:


# Видно что данных больше чем 23650 поэтому обьявления где даже не названы
# места их нахождения можно отбросить мне кажется без всякого
# чувства вины)
df.dropna(subset=['locality_name'], inplace=True)


# In[18]:


df.info()


# In[19]:


df[df['floors_total'].isna()]['floor'].head(15)
    


# In[20]:


# Раз человек не заполнил информацию о балконе то скорее всего его
# и нет
df['balcony'] = df['balcony'].fillna(value=0)


# In[21]:


median_from_city = df[df['locality_name'] == "Санкт-Петербург"]['airports_nearest'].median()
median_from_village = median_from_city + 102090
median_to_city_center = df[df['locality_name'] == "Санкт-Петербург"]['cityCenters_nearest'].median()
median_to_city_center_from_village = median_to_city_center + 102090


# https://ru.distance.to/%D0%A1%D0%B0%D0%BD%D0%BA%D1%82-%D0%9F%D0%B5%D1%82%D0%B5%D1%80%D0%B1%D1%83%D1%80%D0%B3/%D0%9B%D0%B5%D0%BD%D0%B8%D0%BD%D0%B3%D1%80%D0%B0%D0%B4%D1%81%D0%BA%D0%B0%D1%8F-%D0%BE%D0%B1%D0%BB%D0%B0%D1%81%D1%82%D1%8C Взял среднее расстояние из области до Питера и заменяю все NaN в колонах airport_nearest и cityCenters_nearest как сумму расстояния из области + медианы из этих колонок т.к все эти колонки по большей части заполнены для питера

# In[22]:


df['airports_nearest'] = df['airports_nearest'].fillna(value=median_from_village)
df['cityCenters_nearest'] = df['cityCenters_nearest'].fillna(value=median_to_city_center_from_village)


# In[23]:


df.info()


# In[24]:


# высота потолков мне кажется можно изменить по медиане в зависимости 
# от типа местности(колонка area_type)
df['ceiling_height'] = df['ceiling_height'].fillna(
    df.groupby('area_type')['ceiling_height'].transform('median')
)


# In[25]:


df.info()


# In[26]:


#apartament
df[df['is_apartment'] == False]['open_plan'].value_counts()


# In[27]:


df['is_apartment'].value_counts()


# In[28]:


df[df['is_apartment'] == False]['kitchen_area'].max()


# In[29]:


df[df['is_apartment'] == True]['kitchen_area'].max()


# In[30]:


# Делаю замену на False т.к видно что этот вид данных итак заполнен False-ами
#и он слабозаполнен и не хочется терять данные, поэтому вроде логичная замена
df['is_apartment'] = df['is_apartment'].fillna(value=False)


# In[31]:


df[df['is_apartment'] == False]['total_area'].value_counts()


# In[32]:


df.info()


# In[33]:


#floors_total
df[df['floors_total'].isna()]


# In[34]:


#floors_total был  заменен на медианное значание от area_type
df['floors_total'] = df['floors_total'].fillna(df.groupby('area_type')['floors_total'].transform('median'))


# In[35]:


#living_area был  заменен на медианное значание от area_type
df['living_area'] = df['living_area'].fillna(df.groupby('area_type')['living_area'].transform('median'))


# In[36]:


#kitchen_area был  заменен на медианное значание от area_type
df['kitchen_area'] = df['kitchen_area'].fillna(df.groupby('area_type')['kitchen_area'].transform('median'))


# In[37]:


df.info()


# In[38]:


df[df['parks_nearest'].isna()]


# In[39]:


for i in df.dropna(subset=['parks_nearest'])['locality_name']:
    print(i)


# Видно что заполняли значения для расстояния до парков в основном только жители города, но есть и все же жители областти заполнившие эту колонку, поэтому будем заполнять по типу местности(area_type) а не по living area отдельно

# In[40]:


df['parks_nearest'] = df['parks_nearest'].fillna(
    df.groupby('area_type')['parks_nearest'].transform('median')
)


# In[41]:


df.info()


# In[42]:


# Можно заполнять нулями т.к скорее всего леса то есть может быть
# но парков в нынешнем понимание скорее всего в области таковой нет
df['parks_around3000'] = df['parks_around3000'].fillna(value=0)


# In[43]:


df.info()


# In[44]:


#Расстояние до прудов делаем аналогично как и для парков
df['ponds_nearest'] = df['ponds_nearest'].fillna(
    df.groupby('area_type')['ponds_nearest'].transform('median')
)


# In[45]:


df.info()


# In[46]:


df[df['area_type'] == 'область']['ponds_around3000'].median()


# In[47]:


#Не заполнили кол-во прудов значит нет
df['ponds_around3000'] = df['ponds_around3000'].fillna(value=0)


# In[48]:


df.info()


# In[49]:


df[df['days_exposition'].isna()]


# In[50]:


# Все дни которые не заполнены скорее всего просто еще в продаже
# поэтому давайте заполним все это нулями чтобы 
#позже это было удобнее срезать
df['days_exposition'] = df['days_exposition'].fillna(value=0)


# In[51]:


df.info()


# In[52]:


a = df.select_dtypes(include='float64').columns


# In[53]:


df.isna().any()


# In[54]:


df.info()


# In[55]:


df[df['parks_around3000'].isna()]['parks_around3000']


# In[56]:


df['parks_around3000'] = df['parks_around3000'].fillna(value=0)


# In[57]:


df.info()


# In[58]:


#заменяем float на int64
for i in a:
    df['{}'.format(i)] = df['{}'.format(i)].astype('int64')


# In[59]:


df.info()


# Сначала посмотрел какие же всё таки столбцы у нас содержат NaN, оказалось достаточно много. Затем сделал срез по всем NaN содердащимся в колонке airport_nearest и увидел что существует некая зависимость между парками прудами и расстоянием до города. И да оказалось что существует некая причина этим явлениям а именно люди с NaN в этих колонках жили в пригороде. В следствие этого явления я решил что хорошо бы создать новую колонку "area_type" где если колонка locality_name равна питеру то она относится к городу и наоборот соответсвенно. NaN в кол-ве балконов были заменены на 0 потому что скорее всего они появляются вследствие того что данные просто не заполняются. ceiling_height были заполнены методом transform('median') сгруппированную по area_type, таким же образом было сделано со столбцами floors_total, living_area, kitchen_area. is_apartment были заменены на False там где NaN. 

# ## 3. Расчёты и добавление результатов в таблицу

# In[60]:


df['floors_total']


# In[61]:


#добавляем цену за метр
df['price_for_meter'] = df['last_price']/df['total_area']


# In[62]:


df.head()


# In[63]:


df['first_day_exposition'].head()


# In[64]:


df['first_day_exposition'].value_counts().index[0:11]


# In[65]:


df['first_day_exposition'] = pd.to_datetime(df['first_day_exposition'], format='%Y-%m-%dT%H:%M:%S')


# In[66]:


def floor_type(row):
    floor = row['floor']
    last_floor = row['floors_total']
    if floor == 1:
        return "первый"
    elif last_floor==floor:
        return "последний"
    else:
        return "другой"


# In[67]:


df['floor_type'] = df.apply(floor_type, axis=1)


# In[68]:


df['floor_type'].value_counts()


# In[69]:


df['living_area/total_area'] = df['living_area'] / df['total_area']


# In[70]:


df['kitchen_area/total_area'] = df['kitchen_area'] / df['total_area']


# In[71]:


df


# **Вывод**

# Все нижеперечисленные параметры были подсчитаны
#    - цену квадратного метра;
#    - день недели, месяц и год публикации объявления;
#    - этаж квартиры; варианты — первый, последний, другой;
#    - соотношение жилой и общей площади, а также отношение площади кухни к общей.

# ## 4. Исследовательский анализ данных

# In[72]:


import matplotlib.pyplot as plt


# In[73]:


df.info()


# In[74]:


min=df['total_area'].min()
max=df['total_area'].max()


# In[75]:


df['total_area'].hist(bins=50)


# In[76]:


df['last_price'].hist(bins=30)


# In[77]:


df['last_price'].hist(bins=30, range=(100000, 70000000))


# In[78]:


df['rooms'].hist(bins=30)


# In[79]:


df['days_exposition'].hist(bins=30, range=(1,50))


# In[80]:


print(df['days_exposition'].mean())
print(df['days_exposition'].median())


# In[81]:


df.columns


# In[82]:


df['days_exposition'].quantile(q=0.14)


# In[83]:


df.drop(df[df['days_exposition'] == 0].index)['days_exposition'].quantile()


# In[84]:


df.drop(df[df['days_exposition'] == 0].index)['days_exposition'].quantile(q=0.95)


# In[85]:


a = df.drop(df[(df['days_exposition'] == 0)].index)
boxplot = a.boxplot(column = 'days_exposition')
plt.show()


# In[86]:


a = df.drop(df[(df['days_exposition'] == 0) | (df['days_exposition'] > 500)].index)
boxplot = a.boxplot(column = 'days_exposition')
plt.show()


# In[87]:


a = df.drop(df[df['days_exposition'] == 0].index)
boxplot = a.boxplot(column = 'days_exposition', return_type='axes')
plt.show()


# In[88]:


num = (a['days_exposition'].quantile(q=0.75) - a['days_exposition'].quantile(q=0.25))*1.5


# In[89]:


num


# In[90]:


a['days_exposition'].quantile(0.75) + num


# In[91]:


a['days_exposition'].quantile(0.25) 


# In[92]:


a['days_exposition'].quantile(0.5)


# In[93]:


(a['days_exposition'] < 40).sum()


# In[94]:


new_df = a[a['days_exposition'] <= 510]


# Необычно долгой продажей будет значение больше 510, а необычно быстрой продажи просто нет получается исходя из значений боксплота, но в целом это логично у тебя могут купить очень быстро твой товар потому что это рынок и всё зависит от цены, зато есть обьявления висящие очень долго и скорее всего за этими обьявлениями стоят кем то созданные непонятно зачем обьявления, обычно я так понимаю можно допинговать цены на рынке, поэтому вроде всё логично. Также обычно продажа составляет 95 дней

# In[95]:


df['cityCenters_nearest']


# In[96]:


print(new_df['last_price'].corr(new_df['total_area']))
print(new_df['last_price'].corr(new_df['rooms']))
print(new_df['last_price'].corr(new_df['cityCenters_nearest']))


# In[97]:


print(df['last_price'].corr(df['total_area']))
print(df['last_price'].corr(df['rooms']))
print(df['last_price'].corr(df['cityCenters_nearest']))


# Видно слабую корреляцию цены и площади, в остальных случаях кореляций точно нет, также видно что в случае когда мы отбрасываем лишние данные корреляции становятся меньше, что также подтвердает предполодение лишь о слабой корреляции

# In[98]:


df.plot(x='rooms', y ='last_price', kind='scatter')


# In[99]:


df.plot(x='cityCenters_nearest', y = 'last_price', kind='scatter')


# In[100]:


df.total_area


# In[101]:


df.plot(x='total_area', y = 'last_price', kind='scatter')


# В целом видно что есть корреляции для всех этих величин (площади, числа комнат, удалённости от центра) с ценой но всё равно есть еще много видимо других факторов которые влияют на цену

# In[102]:


df.groupby('floor_type')['last_price'].sum() / df.groupby('floor_type')['last_price'].count()


# In[103]:


df.groupby('floor_type')['last_price'].median()


# In[104]:


df.groupby('floor_type')['last_price'].mean()


# In[105]:


df['floor_type'].value_counts()


# Видно что 'первый' этаж всегда самый дешевый, а 'последний' этаж являются самым дорогим в среднем, а в медианном значение это обьяснимо разницей в кол-ве значений между 'другой' и 'последний' этаж, а именно 'другой' этаж намного больше

# In[106]:


df['days'] = pd.DatetimeIndex(df['first_day_exposition']).day


# In[107]:


df['month'] = pd.DatetimeIndex(df['first_day_exposition']).month
df['year'] = pd.DatetimeIndex(df['first_day_exposition']).year


# In[108]:


df['days'].corr(df['last_price'])


# In[109]:


df['month'].corr(df['last_price'])


# In[110]:


df['year'].corr(df['last_price'])


# In[111]:


df.plot(x='last_price', y='year', kind='scatter')


# In[112]:


df.plot(x='last_price', y='month', kind='scatter')


# In[113]:


df.plot(x='last_price', y='month', kind='scatter')


# Корреляции между датой не замечено

# In[114]:


df.groupby('locality_name')['last_price'].count().sort_values(ascending=False).head(10)


# In[115]:


index = df.groupby('locality_name')['last_price'].count().sort_values(ascending=False).head(10).index


# In[116]:


len(index)


# In[117]:


for i in index:
    new_df = df[df['locality_name'] == "{}".format(i)]
    price = (new_df['last_price']/new_df['total_area']).median()
    print(price, i)
    
    #print("Средняя цена квадратного метра равна {}".format(price))


# Самая дешевая цена в Выборге, самая дорогая в Питере, что логично

# In[118]:


second_df = df[(df['locality_name'] == 'Санкт-Петербург') & (df['cityCenters_nearest'] < 100000)]
second_df.plot(x='cityCenters_nearest', y='last_price', kind='scatter', grid=True)


# Видно, что центр начинается между 5000 и 10000, а именно где то около 7500-7600 

# In[119]:


centr = df[(df['locality_name'] == 'Санкт-Петербург') & (df['cityCenters_nearest'] < 7600)]


# In[120]:


centr['total_area'].hist(bins=30)


# In[121]:


centr.boxplot(column='total_area')


# In[122]:


centr['last_price'].hist(bins=60, range=(0,2*(10**7)))


# In[123]:


centr.boxplot(column='last_price')


# In[124]:


centr[centr['last_price'] < 20000000].boxplot(column='last_price')


# In[125]:


centr['rooms'].hist(bins=30)


# In[126]:


centr['is_apartment'].sum()


# In[127]:


centr.boxplot(column='rooms')


# In[128]:


centr['ceiling_height'].hist(bins=30)


# In[129]:


centr['ceiling_height'].hist(bins=30, range=(2,5))


# In[130]:


centr.boxplot(column='ceiling_height')


# In[131]:


centr[centr['ceiling_height'] < 5].boxplot(column='ceiling_height')


# In[132]:


centr.info()


# In[133]:


centr.plot(x='rooms',y='last_price', kind='scatter')


# In[134]:


centr['last_price'].corr(centr['rooms'])


# In[135]:


centr.plot(x='floor',y='last_price', kind='scatter')


# In[136]:


centr['last_price'].corr(centr['floor'])


# In[137]:


centr.plot(x='ceiling_height',y='last_price', kind="scatter")


# In[138]:


centr['last_price'].corr(centr['ceiling_height'])


# In[139]:


centr.plot(x="cityCenters_nearest", y="last_price", kind='scatter')


# In[140]:


centr.plot(x='days', y='last_price', kind='scatter')


# In[141]:


centr.plot(x='month', y='last_price', kind='scatter')


# In[142]:


centr.plot(x='year', y='last_price', kind='scatter')


# In[143]:


centr['cityCenters_nearest'].corr(centr['last_price'])


# In[144]:


centr['days'].corr(centr['last_price'])


# In[145]:


centr['month'].corr(centr['last_price'])


# In[146]:


centr['year'].corr(centr['last_price'])


# In[147]:


centr['last_price'].corr(centr['total_area'])


# **Вывод**

# 
# Необычно долгой продажей будет значение больше 510, а необычно быстрой продажи просто нет получается исходя из значений боксплота, но в целом это логично у тебя могут купить очень быстро твой товар потому что это рынок и всё зависит от цены, зато есть обьявления висящие очень долго и скорее всего за этими обьявлениями стоят кем то созданные непонятно зачем обьявления, обычно я так понимаю можно допинговать цены на рынке, поэтому вроде всё логично. Также обычно продажа составляет 95 дней. 
# 
# 
# Видно слабую корреляцию цены и площади, в остальных случаях кореляций точно нет, также видно что в случае когда мы отбрасываем лишние данные корреляции становятся меньше, что также подтвердает предполодение лишь о том что скорее всего те данные которые мы отбросили не настоящие ибо они усиливают слабую корреляцию
# 
# В целом видно что есть корреляции для всех этих величин (площади, числа комнат, удалённости от центра) с ценой но всё равно есть еще много видимо других факторов которые влияют на цену
# 
# Видно что 'первый' этаж всегда самый дешевый, а 'последний' этаж являются самым дорогим в среднем, а в медианном значение это обьяснимо разницей в кол-ве значений между 'другой' и 'последний' этаж, а именно 'другой' этаж намного больше
# 
# Цена не зависит от дня,месяца или года
# 
# Самая дешевая цена в Выборге, самая дорогая в Питере, что логично
# 
# Видно, что центр начинается между 5000 и 10000, а именно где то около 7500-7600 
# 
# у перечисленных колонок(число комнат, этаж, удалённость от центра, дата размещения объявления) нет корреляции с ценой
# 

# ## 5. Общий вывод

# Исходя из данного анализа данных становится видно что работа с данными очень сложная вещь и очень многие  параметры не очен понятно как оценивать. Данные очень часто загрязнены и часто их нужно очищать и искать достаточно логичное обьяснение своим действиям. Я заметил что мне показалось странно это то что цена на рынке далеко не зависит от таких субьективных понятий как площадь, что у меня например вызывает недоумение небольшое ведь казалось бы больше площадь помещения больше цена, но видимо жизнь вносит свои коррективы и не даёт нам порой продавать слишком выгодно.Вообще самая сильная корреляция как раз у площади но мне кажется что есть возможно что еще сильнее влияет, но я не исключаю того что я не прав. Вообще если я прав то получается что цена очень вариабельная штута.

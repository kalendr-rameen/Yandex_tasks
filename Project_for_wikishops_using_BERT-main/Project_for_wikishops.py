#!/usr/bin/env python
# coding: utf-8

# Привет, меня зовут Люман Аблаев. Сегодня я проверю твой проект.
# <br> Дальнейшее общение будет происходить на "ты" если это не вызывает никаких проблем.
# <br> Желательно реагировать на каждый мой комментарий ('исправил', 'не понятно как исправить ошибку', ...)
# <br> Пожалуйста, не удаляй комментарии ревьюера, так как они повышают качество повторного ревью.
# 
# Комментарии будут в <font color='green'>зеленой</font>, <font color='blue'>синей</font> или <font color='red'>красной</font> рамках:
# 
# <div class="alert alert-block alert-success">
# <b>Успех:</b> Если все сделано отлично
# </div>
# 
# <div class="alert alert-block alert-info">
# <b>Совет: </b> Если можно немного улучшить
# </div>
# 
# <div class="alert alert-block alert-danger">
# <b>Ошибка:</b> Если требуются исправления. Работа не может быть принята с красными комментариями.
# </div>
# 
# 
# <font color='orange' style='font-size:24px; font-weight:bold'>Общее впечатление</font>
# * Большое спасибо за проделанную работу. Видно, что приложено много усилий.
# * Радует, что ноутбук хорошо структурирован. Приятно проверять такие работы.
# - Отлично, что стоп-слова были исключены при векторизации!
# - Качественное и лаконичное написание выводов!
# - Ты успешно справился с задачей машинного обучения для текстов, поздравляю!
# * Отправляю проект назад, чтобы ты дополнил работу(по желанию) иначе можешь просто отправить проект еще раз и я его зачту.
# 
# 
# 
# <font color='green'><b>Полезные (и просто интересные) материалы:</b> \
# Для работы с текстами используют и другие подходы. Например, сейчас активно используются RNN (LSTM) и трансформеры (BERT и другие с улицы Сезам, например, ELMO). НО! Они не являются панацеей, не всегда они нужны, так как и TF-IDF или Word2Vec + модели из классического ML тоже могут справляться. \
# BERT тяжелый, существует много его вариаций для разных задач, есть готовые модели, есть надстройки над библиотекой transformers. Если, обучать BERT на GPU (можно в Google Colab или Kaggle), то должно быть побыстрее.\
# https://huggingface.co/transformers/model_doc/bert.html \
# https://t.me/renat_alimbekov \
# https://colah.github.io/posts/2015-08-Understanding-LSTMs/ - Про LSTM \
# https://web.stanford.edu/~jurafsky/slp3/10.pdf - про энкодер-декодер модели, этеншены\
# https://pytorch.org/tutorials/beginner/transformer_tutorial.html - официальный гайд
# по трансформеру от создателей pytorch\
# https://transformer.huggingface.co/ - поболтать с трансформером \
# Библиотеки: allennlp, fairseq, transformers, tensorflow-text — множествореализованных
# методов для трансформеров методов NLP \
# Word2Vec https://radimrehurek.com/gensim/models/word2vec.html 
# 
# <font color='green'>Пример BERT с GPU:
# ```python
# %%time
# from tqdm import notebook
# batch_size = 2 # для примера возьмем такой батч, где будет всего две строки датасета
# embeddings = [] 
# for i in notebook.tqdm(range(input_ids.shape[0] // batch_size)):
#         batch = torch.LongTensor(input_ids[batch_size*i:batch_size*(i+1)]).cuda() # закидываем тензор на GPU
#         attention_mask_batch = torch.LongTensor(attention_mask[batch_size*i:batch_size*(i+1)]).cuda()
#         
#         with torch.no_grad():
#             model.cuda()
#             batch_embeddings = model(batch, attention_mask=attention_mask_batch)
#         
#         embeddings.append(batch_embeddings[0][:,0,:].cpu().numpy()) # перевод обратно на проц, чтобы в нумпай кинуть
#         del batch
#         del attention_mask_batch
#         del batch_embeddings
#         
# features = np.concatenate(embeddings) 
# ```
# Можно сделать предварительную проверку на наличие GPU.\
# Например, так: ```device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")```\
# Тогда вместо .cuda() нужно писать .to(device)

# <h1>Содержание<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Подготовка" data-toc-modified-id="Подготовка-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Подготовка</a></span></li><li><span><a href="#Обучение" data-toc-modified-id="Обучение-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Обучение</a></span><ul class="toc-item"><li><span><a href="#1.LogisticRegression" data-toc-modified-id="1.LogisticRegression-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>1.LogisticRegression</a></span></li><li><span><a href="#2.-SGDClassifier" data-toc-modified-id="2.-SGDClassifier-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>2. SGDClassifier</a></span></li><li><span><a href="#3.-Catboost" data-toc-modified-id="3.-Catboost-2.3"><span class="toc-item-num">2.3&nbsp;&nbsp;</span>3. Catboost</a></span></li><li><span><a href="#4.-LGBMClassifier" data-toc-modified-id="4.-LGBMClassifier-2.4"><span class="toc-item-num">2.4&nbsp;&nbsp;</span>4. LGBMClassifier</a></span></li><li><span><a href="#5.-BERT" data-toc-modified-id="5.-BERT-2.5"><span class="toc-item-num">2.5&nbsp;&nbsp;</span>5. BERT</a></span></li></ul></li><li><span><a href="#Выводы" data-toc-modified-id="Выводы-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Выводы</a></span></li><li><span><a href="#Чек-лист-проверки" data-toc-modified-id="Чек-лист-проверки-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Чек-лист проверки</a></span></li></ul></div>

# # Проект для «Викишоп» c BERT

# Интернет-магазин «Викишоп» запускает новый сервис. Теперь пользователи могут редактировать и дополнять описания товаров, как в вики-сообществах. То есть клиенты предлагают свои правки и комментируют изменения других. Магазину нужен инструмент, который будет искать токсичные комментарии и отправлять их на модерацию. 
# 
# Обучите модель классифицировать комментарии на позитивные и негативные. В вашем распоряжении набор данных с разметкой о токсичности правок.
# 
# Постройте модель со значением метрики качества *F1* не меньше 0.75. 
# 
# **Инструкция по выполнению проекта**
# 
# 1. Загрузите и подготовьте данные.
# 2. Обучите разные модели. 
# 3. Сделайте выводы.
# 
# Для выполнения проекта применять *BERT* необязательно, но вы можете попробовать.
# 
# **Описание данных**
# 
# Данные находятся в файле `toxic_comments.csv`. Столбец *text* в нём содержит текст комментария, а *toxic* — целевой признак.

# <div class="alert alert-block alert-success">
# <b>Успех:</b> Спасибо за описание проекта
# </div>
# 

# ## Подготовка

# <div style="background: #FAB007; padding: 20px; border: 5px double black; border-radius: 20px;">
# <font color='white'> 
# <b><u>Пояснения студента</u></b>
# </font>
# <font color='white'><br>
# И так чтобы подготовить данные нам необходимо провести лемматизацию предоставленных нам данных, и трансоформировать полученные лемматизированные данные с помощью трансформера Tf-id с использованием стоп слов чтобы отбросить предлоги. Таким образом полученные данные можно будет использовать для обучения модели и предсказания целевого признака<br>
# 

# In[1]:


get_ipython().system(' pip install transformers')


# In[2]:


get_ipython().system(' pip install nltk')


# In[3]:


get_ipython().system(' pip install pymystem3')


# In[4]:


get_ipython().system(' pip install torch')


# In[5]:


import pandas as pd
from pymystem3 import Mystem
import nltk
from nltk.stem import WordNetLemmatizer
import re
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier, LGBMRegressor
import lightgbm as lgbm
import catboost as cb
from sklearn.linear_model import SGDClassifier
import torch
import torch
import transformers 
import numpy as np


# In[6]:


df = pd.read_csv('/datasets/toxic_comments.csv')


# In[7]:


nltk.download('wordnet')


# <div class="alert alert-block alert-info">
# <b>Совет: </b> Не забывай про первичный осмотр данных. Также в задачах классификации важно проверять дисбаланс классов
# </div>

# In[8]:


lemmatizer = WordNetLemmatizer()


# In[ ]:


all_lemm_sentences = []
for i in range(df.shape[0]):
    sentence = df.loc[i]['text']
    word_list = re.sub(r'[^a-zA-Z ]', ' ', df.loc[i]['text']).split()
    lemmatized_output =  ' '.join([lemmatizer.lemmatize(w) for w in word_list])
    all_lemm_sentences.append(lemmatized_output)


# <div class="alert alert-block alert-success">
# 
# <b>Успех:</b> Очистка данных и лемматизация  проведены корректно. 
# </div>
# 

# <div class="alert alert-block alert-info">
# <b>Совет: </b> Неплохо было бы видеть результаты до/после, а не только результат, для наглядности =)
# </div>
# 
# 

# In[ ]:


df['lemm'] = pd.Series(all_lemm_sentences)


# In[8]:


train,oth = train_test_split(df, test_size=0.4)


# In[9]:


valid, test = train_test_split(oth, test_size=0.5)


# In[10]:


train_val = pd.concat([train, valid])


# In[11]:


corpus = train['lemm'].values


# In[12]:


nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


# In[13]:


text_transformer = TfidfVectorizer(stop_words=stop_words, ngram_range=(1, 2), lowercase=True)


# In[14]:


X_train_text = text_transformer.fit_transform(corpus)


# In[15]:


X_test_text = text_transformer.transform(test['lemm'].values)


# <div class="alert alert-block alert-success">
# <b>Успех:</b> Хорошо, что обучил векторизатор только на тренировочной чатси данных. Это уменьшает переобучение.
# </div>

# <div class="alert alert-block alert-info">
# <b>Совет: </b> Приводить английские тексты к юникоду не имеет смысла, так как это приводит только к увеличению занимаемой памяти.
# </div>

# ## Обучение

# <div style="background: #FAB007; padding: 20px; border: 5px double black; border-radius: 20px;">
# <font color='white'> 
# <b><u>Пояснения студента</u></b>
# </font>
# <font color='white'><br>
# Итак данные готовы, теперь применим 4 модели(LogRegressing, Catboost, SGDClassifier, LGBMClassifier), и посмотрим кто же лучше всего может описать и предсказать по нашим данным<br>
# 

# ### 1.LogisticRegression

# In[18]:


logit = LogisticRegression(C=5e1, solver='lbfgs', multi_class='multinomial', random_state=17, n_jobs=-1)    


# In[19]:


logit.fit(X_train_text, train['toxic'])


# In[20]:


logit.predict(X_test_text)


# In[21]:


accuracy_score(test['toxic'], logit.predict(X_test_text))


# In[22]:


f1_score(test['toxic'], logit.predict(X_test_text))


# <div class="alert alert-block alert-info">
# <b>Совет: </b> Тут можно было подобрать параметр C.
# </div>

# ### 2. SGDClassifier

# In[23]:


from sklearn.linear_model import SGDClassifier


# In[24]:


clf = SGDClassifier(loss = "hinge", penalty = "l1")


# In[25]:


clf.fit(X_train_text, train['toxic'])


# In[26]:


f1_score(test['toxic'], clf.predict(X_test_text))


# ### 3. Catboost

# In[40]:


model = cb.CatBoostClassifier(iterations=100, depth=10, loss_function='Logloss',random_seed=12345)


# In[42]:


model.fit(X_train_text, train['toxic'])


# In[45]:


f1_score(test['toxic'], model.predict(X_test_text))


# ### 4. LGBMClassifier

# In[27]:


clf_LGBM = lgbm.LGBMClassifier(verbose=-1, learning_rate=0.5, max_depth=20, num_leaves=50, n_estimators=120, max_bin=2000)


# In[28]:


clf_LGBM.fit(X_train_text, train['toxic'])


# In[29]:


f1_score(test['toxic'], clf_LGBM.predict(X_test_text))


# <div class="alert alert-block alert-success">
# <b>Успех:</b> Этот шаг был сделан очень хорошо. Радует, что ты попробовал разные модели. 
# </div>

# ### 5. BERT

# <div style="background: #FAB007; padding: 20px; border: 5px double black; border-radius: 20px;">
# <font color='white'> 
# <b><u>Пояснения студента</u></b>
# </font>
# <font color='white'><br>
# И так приступим, для начала я удаляю все переменные использованные ранее чтобы наше ядро не крашнулось, и памяти хватило. Затем токенизирую все текста(ну почти все).<br>
# 

# In[1]:


from IPython import get_ipython
get_ipython().magic('reset -sf') 


# In[65]:


import pandas as pd
import torch
import transformers 
import numpy as np
from tqdm import notebook
from tokenizers import BertWordPieceTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


# In[3]:


tokenizer = BertWordPieceTokenizer(
  clean_text=False,
  handle_chinese_chars=False,
  strip_accents=False,
  lowercase=True,
)


# In[4]:


df = pd.read_csv('/datasets/toxic_comments.csv')


# In[5]:


#df = df.iloc[:10000]


# In[6]:


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


# In[7]:


tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-cased', model_max_len=800)


# In[23]:


tokenized = df['text'].apply(
  lambda x: tokenizer.encode(x, add_special_tokens=True))


# <div style="background: #FAB007; padding: 20px; border: 5px double black; border-radius: 20px;">
# <font color='white'> 
# <b><u>Пояснения студента</u></b>
# </font>
# <font color='white'><br>
# Так как длина в нашей макс модели 512 то нам нужно ограничить наш диапазон на 512, что я ниже и делаю. Если существуют методы как избежать этого то скажите как<br>
# 

# 

# In[24]:


index = []
for i in range(len(tokenized)):
    if len(tokenized.iloc[i]) > 512:
        index.append(i)


# In[25]:


tokenized.drop(index=index, inplace=True)


# In[26]:


tokenized


# <div style="background: #FAB007; padding: 20px; border: 5px double black; border-radius: 20px;">
# <font color='white'> 
# <b><u>Пояснения студента</u></b>
# </font>
# <font color='white'><br>
# В целом не так уж и много данных ушло<br>
# 

# In[11]:


max_len = 0
for i in tokenized.values:
    if len(i) > max_len:
        max_len = len(i)

padded = np.array([i + [0]*(max_len - len(i)) for i in tokenized.values])
attention_mask = np.where(padded != 0, 1, 0)


# <div class="alert alert-block alert-success">
# <b>Успех[2]:</b>  padding и attention получены корректно
# </div>

# In[29]:


del tokenized


# In[13]:


model = transformers.BertModel.from_pretrained('bert-base-cased')


# **Embedding**

# In[15]:


get_ipython().run_cell_magic('time', '', 'from tqdm import notebook\nbatch_size = 100 # для примера возьмем такой батч, где будет всего две строки датасета\nembeddings = [] \nfor i in notebook.tqdm(range(padded.shape[0] // batch_size)):\n        batch = torch.LongTensor(padded[batch_size*i:batch_size*(i+1)]).to(device) # закидываем тензор на GPU\n        attention_mask_batch = torch.LongTensor(attention_mask[batch_size*i:batch_size*(i+1)]).to(device)\n        \n        with torch.no_grad():\n            model.to(device)\n            batch_embeddings = model(batch, attention_mask=attention_mask_batch)\n        \n        embeddings.append(batch_embeddings[0][:,0,:].cpu().numpy()) # перевод обратно на проц, чтобы в нумпай кинуть\n        del batch\n        del attention_mask_batch\n        del batch_embeddings\n        \nfeatures = np.concatenate(embeddings)')


# <div class="alert alert-block alert-success">
# <b>Успех[2]:</b>  Хороший выбран батч-сайз
# </div>

# In[19]:


features.shape


# In[28]:


df = df.drop(index=index).iloc[:155700]


# In[38]:


df


# In[47]:


features[:124560]


# In[48]:


df.iloc[:124560]


# <div style="background: #FAB007; padding: 20px; border: 5px double black; border-radius: 20px;">
# <font color='white'> 
# <b><u>Пояснения студента</u></b>
# </font>
# <font color='white'><br>
# После embedding'a настало время научить модель на одной из моделей, например LogRegression<br>
# 

# In[69]:


log_model = LogisticRegression(max_iter=124560, C=5e1, solver='lbfgs', multi_class='multinomial', random_state=17, n_jobs=-1)
log_model.fit(features[:124560], df.iloc[:124560]['toxic'])


# In[70]:


features[124560:].shape


# In[71]:


features.shape


# In[72]:


log_model.predict(features[124560:])


# In[73]:


f1_score(df[124560:]['toxic'], log_model.predict(features[124560:]))


# <div style="background: #FAB007; padding: 20px; border: 5px double black; border-radius: 20px;">
# <font color='white'> 
# <b><u>Пояснения студента</u></b>
# </font>
# <font color='white'><br>
# f1 < 0.75 ((( Обидно даже. Возможно ли как то улучшить? А также если сделать обственный vocab file, то может ли быть результат лучше?<br>
# 

# <div class="alert alert-block alert-success">
# <b>Успех[2]:</b>  Возможно подобрав параметры получше, и учесть дисбаланс классов
# </div>

# ## Выводы

# In[74]:


pd.DataFrame([0.7850659359479363,0.6533066132264529,0.7532690984170681,0.7664162038549494, 0.712411347517730], columns=['f1_score'], 
             index=['LogisticRegression','SGDClassifier','CatBoostClassifier','LGBMClassifier', 'BERT'])


# <div style="background: #FAB007; padding: 20px; border: 5px double black; border-radius: 20px;">
# <font color='white'> 
# <b><u>Пояснения студента</u></b>
# </font>
# <font color='white'><br>
# Полученные результаты свидетельствут о том что в целом все модели предсказывают достаточно неплохо, но в рамках данного проекта нужно преодолеть значение f1 > 0.75 с чем справились все, за исключением стохастического градиентого спуска. А также BERT не достиг значения 0,75<br>
# 

# <div class="alert alert-block alert-success">
# <b>Успех:</b> Приятно видеть структурированный вывод и таблицу в конце проекта! 
# </div>

# ## Чек-лист проверки

# - [x]  Jupyter Notebook открыт
# - [x]  Весь код выполняется без ошибок
# - [x]  Ячейки с кодом расположены в порядке исполнения
# - [x]  Данные загружены и подготовлены
# - [x]  Модели обучены
# - [x]  Значение метрики *F1* не меньше 0.75
# - [x]  Выводы написаны

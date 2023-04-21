#словарь с методами пандаса и связанные с ним библиотеки
pandas_methods = {
    "json.dump()": "Записывает объект Python в файл в формате JSON",
    "json.dumps()": "Создает JSON-строку из переданного в нее объекта",
    "json.load()": "Считывает файл в формате JSON и возвращает объекты Python",
    "json.loads()": "Преобразует JSON-строку в объекты Python",
    "df.to_json()": "Запись датафрейма в формат JSON",
    "pd.read_json()": "Чтение файла формата JSON",
    "eval()": "Функция, выполняющая строку-выражение, переданную ей в качестве обязательного аргумента, и возвращающая результат выполнения этой строки.",
    "replace()": "Возвращает копию строки, в которой все вхождения подстроки заменяются другой подстрокой",
    "split()": "Разбивает строку по указанному разделителю и возвращает список строк",
    "strip()": "Возвращает копию строки, удаляя как начальные, так и конечные символы (в зависимости от переданного строкового аргумента; по умолчанию - пробелы)",
    "apply()": "Позволяет пользователю передать функцию и применить ее к каждому отдельному значению серии Pandas",
    "rename(columns={'costs_sum': 'costs'}, inplace=True)": "Переименовывает столбец датафрейма 'costs_sum‘ в 'costs‘ (параметр inplace=True перезаписывает датафрейм)",
    "df.country.str.contains('platform')": "Проверяет все значения столбца 'country‘ датафрейма на вхождение подстроки 'platform'",
    "df.drop([3334, 3335, 3336,3337],inplace=True)": "Удаляет наблюдения датафрейма по наименованию индексов ([3334, 3335, 3336, 3337]; параметр inplace=True перезаписывает датафрейм)",

    "pd.DataFrame(data)": "Создание объекта DataFrame",
    "!gdown –id 'id файла'": "одгрузка файла в среду выполнения кода Colab с google диска",
    "from google.colab import files": " Импорт класса files из библиотеки google.colab.",
    "uploaded = files.upload()": "Импорт класса files из библиотеки google.colab. Подгрузка файла в среду выполнения кода с персонального компьютера",
    "pd.read_csv('shop_users.csv')": "Чтение файла формата csv (tsv)",
    "pd.read_excel('shop_users.xlsx')": "Чтение файла формата xlsx (xls)",
    "df.head()": "Показывает наблюдения с начала датасета (по умолчанию - 5 наблюдений)",
    "df.tail(3)": "Показывает наблюдения с конца датасета (указано 3 строки)",
    "df.sample(4)": "Показывает случайно выбранные наблюдения (указано 4 строки)",
    "type(df)": "Показывает тип данных передаваемого объекта",
    "df.size": "Показывает размеры датасета (количество ячеек: строки Х столбцы)",
    "len(df)": "Показывает длину датасета (количество наблюдений)",
    "len(df.axes[1])": "Показывает количество признаков в датасете (количество столбцов)",
    "df.shape": "Показывает размеры датасета (количество наблюдений и признаков)",
    "df.dtypes": "Показывает информацию о типах данных по каждому признаку",
    "df.info()": "Показывает общую информацию о датасете",

    "df.copy()": "Создание копии датафрейма df",
    "df.columns": "Вывод списка колонок (столбцов) датафрейма",
    "df.drop(columns=columns, inplace=True)": "Удаление столбцов датафрейма (параметр inplace=True перезаписывает датафрейм)",
    "df.duplicated()": "Проверка на дубликаты: выдает объект Series с булевыми значениями True/False",
    "df.duplicated().sum()": "Количество полных дубликатов (по умолчанию параметр keep='first‘ присваивает значение True первому из совпадений)",
    "df.duplicated(subset=['bdate', 'sex'], keep=False).sum()": "Количество дубликатов по столбцам 'bdate', 'sex’ (параметр keep=False присваивает значения True всем найденным дубликатам)",
    "df.drop_duplicates(inplace=True)": "Удаление полных дубликатов (по умолчанию параметр keep='first‘ удаляет все дубликаты, кроме первого; параметр inplace=True перезаписывает датафрейм)",
    "df.drop_duplicates(subset=['bdate', 'sex'],keep=False, inplace=True)": "Удаление дубликатов по столбцам 'bdate', 'sex’ (параметр keep=False удаляет все найденные дубликаты; параметр inplace=True перезаписывает датафрейм)",

    "df. isna() / df. isnull()": "Проверка датафрейма на пропуски",
    "df. notna() / df. notnull()": "Проверка датафрейма на заполненные значения",
    "df. isna().sum()": "Количество пропущенных значений в датафрейме",
    "df.isnull().mean()*100": "Процент пропущенных значений в датафрейме",
    "df.notna().sum()": "Количество заполненных значений в датафрейме",
    "df['costs']. isna().sum()": "Количество пропущенных значений по столбцу ‘costs’",
    "df.dropna(inplace=True)": "Удалению всех наблюдений датафрейма, где есть хотя бы одно пустое значение (параметр inplace=True перезаписывает датафрейм)",
    "df.dropna(subset=['costs', 'games'], inplace=True)": "Удаление наблюдений, у которых пропущены значения по столбцам 'costs‘ и 'games' (параметр inplace=True перезаписывает датафрейм)",
    "df['city'].fillna('Moscow‘, inplace=True)": "Заполнение пропущенных значений по столбцу 'city‘ значением 'Moscow‘ (параметр inplace=True перезаписывает датафрейм)",
    "mean = df['followers_count'].mean()": "empty",
    "df['followers_count'].fillna(mean, inplace=True)": "Заполнение пропущенных значений по столбцу 'followers_count‘ средним значением по этому столбцу (параметр inplace=True перезаписывает датафрейм)",
    "df.reset_index(drop=True, inplace=True)": "Назначение нового индексного столбца датафрейма (параметр drop=True автоматически удаляет исходный индексный столбец; параметр inplace=True перезаписывает датафрейм)",
    "df.index": "Проверка диапазона индексов",
    "pd.Series.astype(‘int’)":" переводит признак в тип данных int, используемый в Python (ошибка при наличии пропущенных значений)",
    "pd.Series.astype(‘Int64’)":" переводит признак в тип данных Int64, используемый в Pandas  (работает при наличии пропущенных значений)",
    "pd.to_numeric()":" переводит признак в числовой тип данных int64 или float64  в зависимости от подаваемых значений",
    "import datetime":" Импорт модуля datetime",
    "from datetime import datetime": "Импорт класса datetime модуля datetime",
    "pd.to_datetime()": "Переводит объект Pandas в формат datetime64",
    "pd.to_datetime(df.Series, unit='s', origin='unix')": "Переводит объект Series, значения которого закодированы в Unix time в секундах, в формат datetime64",
    "pd.Timestamp.utcfromtimestamp( df.Series)": "Переводит объект Series, значения которого закодированы в Unix time в секундах, в формат datetime64",
    "df.Series.dt.strftime('%d.%m.%Y')": "Переводит объект Series формата datetime в строковое значение в виде ДД.ММ.ГГГГ",
    "pd.Series.astype()": "Переводит признаки в типы данных, используемых в Python или Numpy (тип данных необходимо указывать в качестве параметра)",
    "pd.to_numeric(downcast=‘float’)": "Переводит признак в числовой тип данных (параметр downcast указывает на перевод в тип данных float64; без указания параметра переводит в числовой тип данных int64 или float64",
    "pd.Timestamp.today()": "Получение текущих даты и времени",
    "DatetimeObject.year": "Извлечение года из объекта формата datetime (существует большое число атрибутов)",
    "current_year - df.bdate.apply(lambda x: x.year)": "Расчет количества лет как текущий год за минусом года, извлеченного из даты рождения",
    "df['costs_to_rur'] = df['currency'].map(rates) * df['costs']": "Расчет дополнительного признака со стоимостью в единой валюте – рублях на основании словаря с  курсами валют от существующих признаков 'currency‘ и 'costs'",
    "df.sex.apply(lambda x: 1 if x == 'женский'  else 2)": "Кодирование пола (при условии, что все значения заполнены) sex_dict = {1: 'женский'  2: 'мужской', 0: 'не указан'}",
    "df.sex.apply(lambda x: sex_dict[x])": "Декодирование пола на основании словарярасшифровки",

    "df.Series.unique()": "Получение уникальных значений объекта Series  датафрейма",
    "df.explode(‘Series’, ignore_index=True)": "Трансформирует датафрейм в разрезе списков значений по признаку (параметр ignore_index=True сбрасывает исходные индексы) columns = […]",
    "df = df[columns]": "Сохраняет датафрейм, состоящий из переданного списка признаков, при этом учитывая их порядок в списке",
    "df.to_csv(‘FileName.csv’)": "Записывает датафрейм в файл формата csv",
    "df.to_excel(‘FileName.xlsx’, index=False)": "Записывает датафрейм в файл формата xlsx (параметр index=False игнорирует индексный столбец при записи)  ",

    "df.Series.str.len()": "Возвращает длины строковых значений объекта Series",
    "df.Series.str.count(‘s’)": "Возвращает количество вхождений подстроки ‘s’ в строку объекта Series",
    "df.Series.str.lower()": "Приведение строковых значений объекта Series в нижний регистр",
    "df.Series.str.upper()": "Приведение строковых значений объекта Series в верхний регистр",
    "df.Series.str.capitalize()": "Приведение первых символов строковых значений объекта Series в верхний регистр, а остальных символов – в нижний регистр",
    "df.Series.str.strip()": "Возвращает копии строки объекта Series, удаляя как начальные, так и конечные символы (в зависимости от переданного строкового аргумента; по умолчанию - пробелы)",
    "df.Series.str.lstrip()": "Возвращает копии строки объекта Series, удаляя как начальные символы (в зависимости от переданного строкового аргумента; по умолчанию - пробелы)",
    "df.Series.str.rstrip()": "Возвращает копии строки объекта Series, удаляя конечные символы (в зависимости от переданного строкового аргумента; по умолчанию - пробелы)",
    "df.Series.str.replace(‘old’, ‘new’)": "Заменяет указанную подстроку строковых значений объекта Series на новую",
    "df.Series.str.split(sep=‘sep’)": "Разделяет строки объекта Series на список подстрок по указанному разделителю (по  умолчанию – пробел)",
    "df.Series.str.cat(sep=‘sep’)": "Объединяет строки объектов Series в единое строковое значение через указанный разделитель (по умолчанию – пробел)",
    "df.Series.str. get_dummies()": "Кодирует значения объекта Series, для каждого создавая отдельный столбец",
    "df.loc[df['column_name'] == 'some_value', 'column_name'] = 'значение'": "Заменяет значение объекта Series по условию (для любого типа данных)",

    "df.loc[df.city.str.contains('petersb urg'), 'city'] = 'санкт-петербург'": "Заменяет значение признака 'city' на 'санкт-петербург' при условии, что изначальное значение по этому же признаку – содержит подстроку 'petersburg",
    "df1[df1['player_surname'] == 'GOLOVIN']": "поиск строк с Головиным в ПЗ №6)",
    "df.set_index('id', inplace=True)":  "меняем колонку index на id, inplace сохраняет изменения в датафрейме",
    "pd.read_json('partner_data_records_cp1251.json', encoding='cp1251')": "чтение файла в кодировке Windows-1251 (cp1251),encoding='UTF-8'",
    "df = pd.read_excel('partner_data.xlsx', sheet_name='partner_data1', header=0, index_col='id')": "header — строка с заголовками, index_col — колонка с индексами",
    ".iloc": "выбор элементов по индексу;",
    ".loc": "выбор элементов по названию. ",
    "df.iloc[0,0]": "элемент на позиции [0,0]",
    "df.iloc[0:3, 0]": "элементы 0, 1, 2 нулевого столбца",
    "df.iloc[:, 0]": "полный срез столбца 0",
    "df.iloc[0, :]": "полный срез строки 0",
    "df.loc[1, 'age']": "строка с id 1, столбец age",
    "df.loc[5:6, 'marital':'education']": "обе границы включаются в диапазон",
    "df.loc[[5,10], ['marital','default']]": "можно передавать списки",
    "df['age']": "один столбец",
    "df[['age', 'marital']]": "несколько столбцов"

}
# словарь с шаблонами eda
eda = {
"Шаблон применения какой-либо функции ко всем значениям колонки": '''import re
def clear_price(price):
    return int(re.sub(\'\\D\', \'\', price))
dataset[\'Очищенная цена\'] = dataset[\'Цена\'].apply(clear_price)''',
    "Шаблон сортировки по значению": '''sorted_dataset = dataset.sort_values('Очищенная цена')
sorted_dataset''',
    "Импортируем библиотеку визуализации":'''import matplotlib.pyplot as plt''',
    "Для визуального представления данных в том числе медианы, квартилей и выбросов используем boxplot": '''plt.boxplot(dataset['Номер этажа'])
plt.show()''',
    "Альтернативный способ визуализировать колонку - гистограмма":'''plt.hist(dataset['Номер этажа'], bins = 50)
plt.xlim((None,55)) # для "отрезания" от графика неинформативного
plt.show()''',
    "Создание переменной с Series":'''year_of_construction = dataset['Год постройки']''',
    "Расчет минимального и максимального значения Series":'''max_value = year_of_construction.max()
min_value = year_of_construction.min()
print('Максимальный год постройки: ', max_value, 'Минимальный год постройки: ', min_value)''',
    "Расчет среднего значения Series":'''mean_value = year_of_construction.mean()
print('Средний год постройки: ', mean_value)''',
    "Расчет медианы Series":'''median_value = year_of_construction.median()
print('Медиана года постройки: ', median_value)''',
    "Расчет 10-го и 25-го процентилей Series":'''percentile_10_value = year_of_construction.quantile(0.10)
percentile_25_value = year_of_construction.quantile(0.25)
print('10-й процентиль года постройки: ', percentile_10_value)
print('25-й процентиль года постройки: ', percentile_25_value)''',
    "Для расчета всех основных статистических показателей сразу":'''year_of_construction.describe()''',
    "Неграфический способ проанализировать колонку":'''dataset['Класс жилья'].describe()''',
    "Более полный неграфический способ":'''class_counts = dataset['Класс жилья'].value_counts()
class_counts''',
    "Круговая диаграмма":'''plt.pie(class_counts.values, labels = class_counts.index)
plt.title('Круговая диаграмма распределения классов квартир') # Добавление подписи к графику
plt.show()''',
    "Стобцевая диаграмма":'''type_counts = dataset['Тип здания'].value_counts()
plt.barh(type_counts.index, type_counts.values)
plt.title('Столбцовая диаграмма распределения типов здания')
plt.show()''',
    "Шаблона первичного анализа взаимосвязи категориальной и числовой колонок":'''dataset.groupby('Класс жилья')["Очищенная цена за м²"].describe()''',
    "Шаблон визуального анализа взаимосвязи категориальной и числовой колонок":'''import seaborn as sns
sns.boxplot(x='Класс жилья', y="Очищенная цена за м²", data=dataset)
plt.axis(ymin=0, ymax=1100000) 
plt.show()''',
    "Шаблона анализа взаимосвязи двух и более числовых колонок":'''g = sns.PairGrid(new_dataset[columns])
g.map(sns.scatterplot, alpha=0.2)''',
    "Шаблона анализа взаимосвязи двух категориальных колонок":'''import matplotlib.pyplot as plt
flat_types = dataset['Класс жилья'].unique()
flat_types = flat_types[1:]
labels = dataset['Тип здания'].unique()
colors = dict(zip(labels, plt.cm.tab20.colors[:len(labels)]))
for flat_type in flat_types: 
  df = dataset[dataset['Класс жилья']==flat_type] 
  class_counts = df['Тип здания'].value_counts()
  labels =class_counts.index
  plt.title(flat_type)
  patches, texts = plt.pie(
      class_counts.values, 
      labels = labels, 
      colors = [colors[key] for key in labels],
      textprops=dict(color="w")
      )
  plt.legend(patches, labels, title="Типы здания", loc="upper center", bbox_to_anchor=(1, 0, 0.5, 1)) 
  plt.show()''',
    "Шаблон фильтрации по одному значению в колонке":'''filtered_dataset = dataset[dataset['Этап строительства']=='Котлован']''',
    "Шаблон фильтрации числовой колонки":'''filtered_dataset = dataset[dataset['Год постройки']>2010]''',
    "Шаблон фильтрации по нескольким значениям в колонке":'''districts = ['Строгино', 'Щукино', 'Хорошёвский']
filtered_dataset = dataset[dataset['Район'].isin(districts)]''',
    "Шаблон фильтрации по нескольким колонкам":'''districts = ['Строгино', 'Щукино', 'Хорошёвский']
filtered_dataset = dataset[(dataset['Район'].isin(districts)) & (dataset['Год постройки']>2010)]''',
    "Шаблон фильтрации по всем значениям кроме заданных в фильтре":'''districts = ['Строгино', 'Щукино', 'Хорошёвский']
filtered_dataset = dataset[~(dataset['Район'].isin(districts))]''',
    "Шаблон взятия элемента по индексу":'''dataset.loc[100]''',
    "Шаблон взятия среза по промежутку":'''dataset.loc[1000:1002]''',
    "Шаблон создания нового датасета списку индексов ":'''index_list = [354,1,2]
dataset.loc[index_list].head()''',
    "Для не числовых индексов все работает аналогично":'''describe_data = dataset.describe()
describe_data.loc['max']''',
    "Шаблон для взятия среза в хаотично пронумеровнном датасете ":'''sorted_dataset = dataset.sort_values('Количество комнат', ascending=False)
sorted_dataset.iloc[:100]''',
    "Нестандартная фильтрация с помощью дополнительный функции":'''def is_address_correct(address):
  return 'Тагильская' in address
mask = dataset['Адрес'].apply(is_address_correct)
filtred_dataset = dataset[mask]''',
    "Шаблон фильтрации числовых колонок":'''dtypes = dataset.dtypes
num_dtypes = dtypes[dtypes!='object']
new_columns = num_dtypes.index
filtred_dataset = dataset[num_dtypes.index]''',
    "Шаблон взятия среднего значения цены квартир сгруппированных по значениям Этапа строительства  ":'''groups_mean_price = dataset.groupby('Этап строительства')['Очищенная цена'].mean()''',
    "Шаблон взятия среднего значения цены квартир сгруппированных по значениям колонок Этапа строительства и Класса жилья":'''
groups_mean_price = dataset.groupby(['Этап строительства', 'Класс жилья'], as_index=False)['Очищенная цена'].mean()
groups_mean_price['Очищенная цена'] = groups_mean_price['Очищенная цена'].astype(int)''',
    "Размер шрифта на диаграмме":'''params = {
          'axes.titlesize': 15,   Размер шрифта главной подписи
          'xtick.labelsize': 12,  Размер шрифта подписей тикетов оси X
          'axes.labelsize': 14    Размер шрифта подписей осей
          }
plt.rcParams.update(params)''',
    "Шаблон улучшения графика":'''import matplotlib.pyplot as plt
type_counts = dataset['Тип здания'].value_counts()
other_types = type_counts[type_counts<500]
type_counts = type_counts[type_counts>500]
type_counts['Другое'] = other_types.sum() # Замена редких значений категорией Другое
plt.style.use('seaborn') # Смена стиля всех графиков matplotlib
params = {
          'axes.titlesize': 15,   # Размер шрифта главной подписи
          'xtick.labelsize': 12,  # Размер шрифта подписей тикетов оси X
          'axes.labelsize': 14    # Размер шрифта подписей осей
          }
plt.rcParams.update(params) # Фиксация параметров
plt.ylabel('Количество квартир') # Подпись оси Y
plt.title('Диаграмма распределения типов здания') # Подпись всего графика
plt.xticks(rotation=30, ha = 'right') # Поворот на 30 градусов подписей оси X и выравнивание по правому краю
plt.bar(type_counts.index, type_counts.values, color='#03A9F4') # Кастомизация цвета
plt.show()''',
    "50 разных способов создать боксплот/диаграмму/гистаграмму":'''https://habr.com/ru/articles/468295/''',
                                                                  }
def search_in(string, dct=pandas_methods):
    '''Функция поиска совпадений по подстроке, можно писать часть названия метода/функции
    или слово/часть слова из описания этой функции/метода'''
    return_list = []
    for name, info in dct.items():
        if string.lower() in name.lower() or string.lower() in info.lower():
            return_list.append({name: info})
    print('*'*156)
    for i in return_list:
        print(*i)
        print(*(i.values()), end='\n'+156 * '-'+'\n')
# end='\n'+ max(map(lambda x: len(str(x)), return_list))*'-'+'\n'
# запросы разделены *, примеры разделены -
#пример поиска в разведочном анализе
search_in('срез',eda)
#пример поиска в пандасе
search_in('чтен')


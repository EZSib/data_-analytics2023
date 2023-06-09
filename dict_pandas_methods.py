from random import shuffle as sh
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

def search_in(string, dct=pandas_methods):
    '''Функция поиска совпадений по подстроке, можно писать часть названия метода/функции
    или слово/часть слова из описания этой функции/метода'''
    return_list = []
    for name, info in dct.items():
        if string.lower() in name.lower() or string.lower() in info.lower():
            return_list.append({name: info})
    print(*(return_list), sep='\n'+ max(map(lambda x: len(str(x)), return_list))*'-'+'\n')

#пример поиска по части метода
search_in('isn')
#пример поиска по части описания
search_in('объе')


more_color = """#2F4F4F
#696969
#708090
#778899
#BEBEBE
#D3D3D3
#191970
#000080
#6495ED
#483D8B
#6A5ACD
#7B68EE
#8470FF
#0000CD
#4169E1
#0000FF
#1E90FF
#00BFFF
#87CEEB
#87CEFA
#B0C4DE
#ADD8E6
#B0E0E6
#AFEEEE
#00CED1
#48D1CC
#40E0D0
#E0FFFF
#66CDAA
#7FFFD4
#006400
#556B2F
#8FBC8F
#2E8B57
#3CB371
#20B2AA
#98FB98
#00FF7F
#7CFC00
#00FF00
#7FFF00
#00FA9A
#ADFF2F
#32CD32
#228B22
#6B8E23
#BDB76B
#EEE8AA
#FAFAD2
#FFFFE0
#FFFF00
#FFD700
#EEDD82
#DAA520
#B8860B
#BC8F8F
#CD5C5C
#8B4513
#A0522D
#CD853F
#DEB887
#F5F5DC
#F5DEB3
#F4A460
#D2B48C
#D2691E
#B22222
#A52A2A
#E9967A
#FA8072
#FFA500
#FF8C00
#FF7F50
#F08080
#FF4500
#FF0000
#FF69B4
#FF1493
#FFC0CB
#FFB6C1
#DB7093
#B03060
#C71585
#D02090
#FF00FF
#DDA0DD
#DA70D6
#BA55D3
#9932CC
#9400D3
#8A2BE2
#A020F0
#9370DB
#D8BFD8
#FFFAFA
#EEE9E9"""
more_color = more_color.split()

def give_color(my_list, colours= more_color):
    sh(more_color)
    my_list = dict(zip(more_color, my_list))
    return my_list
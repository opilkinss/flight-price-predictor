import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
from catboost import CatBoostRegressor
from pathlib import Path

# Определяем путь к папке скрипта
SCRIPT_DIR = Path(__file__).parent

# ------------------------------
# Функция для расчёта расстояния по координатам (гаверсинус)
# ------------------------------
def haversine(lat1, lon1, lat2, lon2):
    """
    Вычисляет расстояние в километрах между двумя точками на Земле
    по формуле гаверсинуса.
    """
    R = 6371  # радиус Земли в км
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = np.sin(delta_phi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(delta_lambda/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

# ------------------------------
# Функция для оценки длительности по расстоянию (в минутах)
# ------------------------------
def estimate_duration(distance_km, speed_kmh=850, ground_time_minutes=30):
    """
    Оценивает длительность полёта в минутах на основе расстояния,
    средней крейсерской скорости и добавочного времени на взлёт/посадку.
    """
    duration_hours = distance_km / speed_kmh + (ground_time_minutes / 60)
    return duration_hours * 60

# ------------------------------
# Загрузка модели
# ------------------------------
@st.cache_resource
def load_model():
    model_path = SCRIPT_DIR / 'catboost_model.cbm'
    if not model_path.exists():
        st.error(f"Файл модели не найден: {model_path}")
        return None
    model = CatBoostRegressor()
    model.load_model(str(model_path))
    return model

# ------------------------------
# Загрузка исторических данных
# ------------------------------
@st.cache_data
def load_data():
    data_path = SCRIPT_DIR / 'flight_prices_week.csv'
    if not data_path.exists():
        st.error(f"Файл данных не найден: {data_path}")
        return None
    df = pd.read_csv(data_path)
    df['depart_date'] = pd.to_datetime(df['depart_date'])
    df['collection_date'] = pd.to_datetime(df['collection_date'])
    return df

# ------------------------------
# Загрузка координат аэропортов из airports.csv
# ------------------------------
@st.cache_data
def load_airport_coords():
    coords_path = SCRIPT_DIR / 'airports.csv'
    if not coords_path.exists():
        st.error("Файл с координатами аэропортов не найден.")
        return None
    df = pd.read_csv(coords_path)
    # Оставляем только строки с непустыми координатами и кодом IATA
    df = df[df['code'].notna() & df['latitude'].notna() & df['longitude'].notna()].copy()
    df['code'] = df['code'].astype(str).str.strip().str.upper()
    # Убираем дубликаты по коду
    df = df.drop_duplicates(subset=['code'], keep='first')
    # Создаём словарь: code -> (lat, lon)
    coords_dict = {}
    for _, row in df.iterrows():
        coords_dict[row['code']] = {'latitude': row['latitude'], 'longitude': row['longitude']}
    return coords_dict

# ------------------------------
# Загрузка справочника названий аэропортов (уже с кодом в скобках)
# ------------------------------
@st.cache_data
def load_airport_names():
    names_path = SCRIPT_DIR / 'airport_names.csv'
    if not names_path.exists():
        st.error("Файл с названиями аэропортов не найден.")
        return None, None
    df_names = pd.read_csv(names_path)
    df_names['code'] = df_names['code'].astype(str).str.strip().str.upper()
    df_names['display_name'] = df_names['display_name'].astype(str).str.strip()
    # Удаляем возможные дубликаты по отображаемому имени (на всякий случай)
    df_names = df_names.drop_duplicates(subset=['display_name'], keep='first')
    code_to_display = dict(zip(df_names['code'], df_names['display_name']))
    display_to_code = dict(zip(df_names['display_name'], df_names['code']))
    return code_to_display, display_to_code

# ------------------------------
# Загрузка средних расстояний и длительностей из исторических данных
# (на случай отсутствия координат)
# ------------------------------
@st.cache_data
def get_route_distances(df):
    return df.groupby(['origin', 'destination'], as_index=False)['distance'].mean()

@st.cache_data
def get_route_durations(df):
    return df.groupby(['origin', 'destination'], as_index=False)['duration'].mean()

# ------------------------------
# Основная часть приложения
# ------------------------------

# Загружаем исторические данные
df_main = load_data()
if df_main is None:
    st.stop()

# Загружаем координаты и справочник названий
coords_dict = load_airport_coords()
code_to_display, display_to_code = load_airport_names()
if code_to_display is None or display_to_code is None:
    st.stop()

# Формируем список отображаемых имён для выпадающего списка
display_options = sorted(display_to_code.keys())

# Загружаем модель
model = load_model()
if model is None:
    st.stop()

# Вычисляем средние расстояния и длительности по историческим данным (резерв)
route_distances = get_route_distances(df_main)
route_durations = get_route_durations(df_main)

# ------------------------------
# Интерфейс
# ------------------------------
st.title('✈️ Предсказание цены авиабилета')
st.markdown('Введите параметры рейса, и модель CatBoost предскажет стоимость билета (в рублях).')

col1, col2 = st.columns(2)
with col1:
    origin_display = st.selectbox('Город вылета', display_options)
with col2:
    destination_display = st.selectbox('Город назначения', display_options)

origin = display_to_code[origin_display]
destination = display_to_code[destination_display]

depart_date = st.date_input('Дата вылета', min_value=date.today())
stops = st.selectbox('Количество пересадок', [0, 1, 2, 3])

if st.button('Предсказать цену'):
    if origin == destination:
        st.error('Города вылета и назначения должны отличаться!')
    else:
        # ---------- Расстояние ----------
        origin_coords = coords_dict.get(origin) if coords_dict else None
        dest_coords = coords_dict.get(destination) if coords_dict else None
        if origin_coords and dest_coords:
            # Рассчитываем по координатам
            distance = haversine(
                origin_coords['latitude'], origin_coords['longitude'],
                dest_coords['latitude'], dest_coords['longitude']
            )
            st.info("Расстояние рассчитано по координатам.")
        else:
            # Используем среднее из исторических данных
            dist_row = route_distances[(route_distances['origin'] == origin) & (route_distances['destination'] == destination)]
            if dist_row.empty:
                distance = route_distances['distance'].mean()
                st.warning('Точное расстояние для этого маршрута не найдено, используется среднее.')
            else:
                distance = dist_row.iloc[0]['distance']

        # ---------- Длительность ----------
        dur_row = route_durations[(route_durations['origin'] == origin) & (route_durations['destination'] == destination)]
        if not dur_row.empty:
            duration = dur_row.iloc[0]['duration']
        else:
            # Оцениваем по расстоянию (параметры можно подобрать, здесь эмпирические)
            duration = estimate_duration(distance, speed_kmh=850, ground_time_minutes=30)
            st.info("Длительность оценена по расстоянию.")

        # ---------- Признаки из даты ----------
        collection_date = date.today()
        days_before = (depart_date - collection_date).days
        depart_weekday = depart_date.weekday()
        depart_month = depart_date.month

        # ---------- Создаём DataFrame для предсказания ----------
        input_df = pd.DataFrame([{
            'distance': distance,
            'duration': duration,
            'stops': stops,
            'days_before': days_before,
            'depart_weekday': depart_weekday,
            'depart_month': depart_month,
            'origin': origin,
            'destination': destination
        }])

        # ---------- Предсказание ----------
        try:
            prediction = model.predict(input_df)[0]
            st.success(f'## Предсказанная цена: {prediction:.0f} руб.')
        except Exception as e:
            st.error(f"Ошибка при предсказании: {e}")

        with st.expander('Показать детали'):
            st.write('**Входные параметры:**')
            st.dataframe(input_df)
            st.write(f'**Средняя абсолютная ошибка модели:** около 4776 руб.')
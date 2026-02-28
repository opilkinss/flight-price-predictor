import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
from catboost import CatBoostRegressor
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent

# ------------------------------
# Функция гаверсинуса
# ------------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = np.sin(delta_phi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(delta_lambda/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

# ------------------------------
# Оценка длительности
# ------------------------------
def estimate_duration(distance_km, speed_kmh=850, ground_time_minutes=30):
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
# Загрузка исторических данных (нужны для средних значений)
# ------------------------------
@st.cache_data
def load_data():
    data_path = SCRIPT_DIR / 'flight_prices_week.csv'
    if not data_path.exists():
        st.error(f"Файл данных не найден: {data_path}")
        return None
    df = pd.read_csv(data_path)
    df['depart_date'] = pd.to_datetime(df['depart_date'])
    df['return_date'] = pd.to_datetime(df['return_date'])
    df['collection_date'] = pd.to_datetime(df['collection_date'])
    return df

# ------------------------------
# Загрузка координат аэропортов
# ------------------------------
@st.cache_data
def load_airport_coords():
    coords_path = SCRIPT_DIR / 'airports.csv'
    if not coords_path.exists():
        st.error("Файл с координатами аэропортов не найден.")
        return None
    df = pd.read_csv(coords_path)
    df = df[df['code'].notna() & df['latitude'].notna() & df['longitude'].notna()].copy()
    df['code'] = df['code'].astype(str).str.strip().str.upper()
    df = df.drop_duplicates(subset=['code'], keep='first')
    coords_dict = {}
    for _, row in df.iterrows():
        coords_dict[row['code']] = {'latitude': row['latitude'], 'longitude': row['longitude']}
    return coords_dict

# ------------------------------
# Загрузка справочника названий аэропортов
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
    df_names = df_names.drop_duplicates(subset=['display_name'], keep='first')
    code_to_display = dict(zip(df_names['code'], df_names['display_name']))
    display_to_code = dict(zip(df_names['display_name'], df_names['code']))
    return code_to_display, display_to_code

# ------------------------------
# Средние расстояния и длительности (резерв)
# ------------------------------
@st.cache_data
def get_route_distances(df):
    return df.groupby(['origin', 'destination'], as_index=False)['distance'].mean()

@st.cache_data
def get_route_durations(df):
    return df.groupby(['origin', 'destination'], as_index=False)['duration'].mean()

# ------------------------------
# Основная часть
# ------------------------------

df_main = load_data()
if df_main is None:
    st.stop()

coords_dict = load_airport_coords()
code_to_display, display_to_code = load_airport_names()
if code_to_display is None or display_to_code is None:
    st.stop()

display_options = sorted(display_to_code.keys())

model = load_model()
if model is None:
    st.stop()

route_distances = get_route_distances(df_main)
route_durations = get_route_durations(df_main)

st.title('✈️ Предсказание цены авиабилета (туда-обратно)')
st.markdown('Введите параметры рейса, и модель CatBoost предскажет стоимость билета (в рублях).')

col1, col2 = st.columns(2)
with col1:
    origin_display = st.selectbox(
        'Город вылета',
        options=display_options,
        index=None,
        placeholder='Впишите город или выберите из списка'
    )
with col2:
    destination_display = st.selectbox(
        'Город назначения',
        options=display_options,
        index=None,
        placeholder='Впишите город или выберите из списка'
    )

# Если пользователь ещё не выбрал города, переменные будут None
if origin_display and destination_display:
    origin = display_to_code[origin_display]
    destination = display_to_code[destination_display]
else:
    origin = destination = None

depart_date = st.date_input('Дата вылета', min_value=date.today())
return_date = st.date_input('Дата возвращения', min_value=depart_date)

stops = st.selectbox('Количество пересадок', [0, 1, 2, 3])

if st.button('Предсказать цену'):
    if not origin_display or not destination_display:
        st.error('Пожалуйста, выберите города вылета и назначения.')
    elif origin == destination:
        st.error('Города вылета и назначения должны отличаться!')
    elif return_date <= depart_date:
        st.error('Дата возвращения должна быть позже даты вылета.')
    else:
        # ----- Расстояние -----
        origin_coords = coords_dict.get(origin) if coords_dict else None
        dest_coords = coords_dict.get(destination) if coords_dict else None
        if origin_coords and dest_coords:
            distance = haversine(
                origin_coords['latitude'], origin_coords['longitude'],
                dest_coords['latitude'], dest_coords['longitude']
            )
            st.info("Расстояние рассчитано по координатам.")
        else:
            dist_row = route_distances[(route_distances['origin'] == origin) & (route_distances['destination'] == destination)]
            if dist_row.empty:
                distance = route_distances['distance'].mean()
                st.warning('Точное расстояние для этого маршрута не найдено, используется среднее.')
            else:
                distance = dist_row.iloc[0]['distance']

        # ----- Длительность полёта -----
        dur_row = route_durations[(route_durations['origin'] == origin) & (route_durations['destination'] == destination)]
        if not dur_row.empty:
            duration = dur_row.iloc[0]['duration']
        else:
            duration = estimate_duration(distance)
            st.info("Длительность полёта оценена по расстоянию.")

        # ----- Признаки на основе дат -----
        collection_date = date.today()
        days_before = (depart_date - collection_date).days
        days_before_return = (return_date - collection_date).days  # ключевой признак
        depart_weekday = depart_date.weekday()
        depart_month = depart_date.month

        # ----- DataFrame для модели -----
        # Порядок колонок должен точно соответствовать обучению:
        # числовые: distance, duration, stops, days_before, depart_weekday, depart_month, days_before_return
        # категориальные: origin, destination
        input_df = pd.DataFrame([{
            'distance': distance,
            'duration': duration,
            'stops': stops,
            'days_before': days_before,
            'depart_weekday': depart_weekday,
            'depart_month': depart_month,
            'days_before_return': days_before_return,
            'origin': origin,
            'destination': destination
        }])

        try:
            prediction = model.predict(input_df)[0]
            st.success(f'## Предсказанная цена: {prediction:.0f} руб.')
        except Exception as e:
            st.error(f"Ошибка при предсказании: {e}")

        with st.expander('Показать детали'):
            st.write('**Входные параметры:**')
            st.dataframe(input_df)
            st.write('**Средняя абсолютная ошибка модели:** около 4776 руб.')
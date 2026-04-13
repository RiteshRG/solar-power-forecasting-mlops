import pandas as pd, numpy as np
import matplotlib.pyplot as plt

df_p1_G = pd.read_csv('data/raw/Plant_1_Generation_Data.csv')

first_22 = set(df_p1_G['SOURCE_KEY'].head(22))
all_unique = set(df_p1_G['SOURCE_KEY'].unique())

df_p1_w = pd.read_csv('data/raw/Plant_1_Weather_Sensor_Data.csv')

df_p1_G['DATE_TIME'] = pd.to_datetime(
    df_p1_G['DATE_TIME'],
    format='%d-%m-%Y %H:%M'
)

df_p1_w['DATE_TIME'] = pd.to_datetime(
    df_p1_w['DATE_TIME'],
)

df_p1_merged = pd.merge(df_p1_G, df_p1_w, on = ['DATE_TIME'], how='left')

df = df_p1_merged[['AC_POWER','IRRADIATION', 'AMBIENT_TEMPERATURE','DC_POWER', 'DATE_TIME']]

df_plant = df.groupby('DATE_TIME').agg({
    'AC_POWER':'sum',
    'DC_POWER':'sum',
    'IRRADIATION':'mean',
    'AMBIENT_TEMPERATURE':'mean'
}).reset_index()

df_hourly = df_plant.resample('1H', on='DATE_TIME').mean().reset_index()

df_hourly.drop(labels = ['DC_POWER'], axis=1, inplace=True, errors='ignore')

#Extract Time Features
df_hourly['HOUR'] = df_hourly['DATE_TIME'].dt.hour

df_hourly['HOUR_SIN'] = np.sin(2*np.pi*df_hourly['HOUR']/24)
df_hourly['HOUR_COS'] = np.cos(2*np.pi*df_hourly['HOUR']/24)

df_hourly = df_hourly.drop(columns=['HOUR'])

df_hourly = df_hourly[df_hourly['IRRADIATION'] > 0]

df = df_hourly.drop(columns=['DATE_TIME'])

df.to_csv('data/processed/train.csv', index=False)

# ALSO push to feature store

print("Preprocessing completed. Processed data saved to 'data/processed/train.csv'.")
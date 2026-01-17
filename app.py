
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os # For managing file paths

# --- Data Loading and Preprocessing (as in the notebook) ---
@st.cache_data # Cache the data loading to improve performance
def load_and_preprocess_data():
    # Construct the full path to the CSV file
    beer_df = pd.read_csv("./Consumo_cerveja.csv")

    # Drop rows with any missing values
    beer_df = beer_df.dropna()

    # Rename columns
    beer_df.columns = ['date', 'avg_temp', 'min_temp', 'max_max', 'precipitation', 'is_weekend', 'consumption']

    # Convert types
    beer_df['avg_temp'] = beer_df['avg_temp'].str.replace(',', '.').astype(float)
    beer_df['min_temp'] = beer_df['min_temp'].str.replace(',', '.').astype(float)
    beer_df['max_max'] = beer_df['max_max'].str.replace(',', '.').astype(float)
    beer_df['precipitation'] = beer_df['precipitation'].str.replace(',', '.').astype(float)
    beer_df['is_weekend'] = beer_df['is_weekend'] > 0.5
    beer_df['date'] = pd.to_datetime(beer_df['date'])

    return beer_df

beer_df = load_and_preprocess_data()

# --- Streamlit App Layout ---
st.set_page_config(layout="wide")

st.title("Análise do Consumo de Cerveja em São Paulo")

st.markdown("""
Esta aplicação Streamlit apresenta uma análise exploratória do consumo de cerveja em São Paulo,
baseada nos dados de 2015.
""")

# Display the raw data
st.header("Dados Brutos e Pré-processados")
st.write("Amostra do DataFrame após carregamento e limpeza:")
st.dataframe(beer_df.head())

st.write("Informações gerais do DataFrame:")
st.write(f"Número de linhas: {beer_df.shape[0]}")
st.write(f"Número de colunas: {beer_df.shape[1]}")
st.write("Tipos de dados:")
st.dataframe(beer_df.dtypes.astype(str).reset_index().rename(columns={
    'index': 'Coluna',
    0: 'Tipo de Dado'
}))

# --- Visualizations (reproducing some from the notebook) ---
st.header("Visualizações Chave")

# Monthly Average Precipitation Plot
st.subheader("Precipitação Média Mensal")
monthly_df_precip = beer_df.copy()
monthly_df_precip.set_index('date', inplace=True)
monthly_avg_precip = monthly_df_precip['precipitation'].resample('ME').mean()

fig_precip, ax_precip = plt.subplots(figsize=(12, 6))
ax_precip.bar([d.strftime('%b %Y') for d in monthly_avg_precip.index], monthly_avg_precip.values, color='skyblue', edgecolor='navy')
ax_precip.set_title('Média de Precipitação Mensal', fontsize=16)
ax_precip.set_xlabel('Mês', fontsize=12)
ax_precip.set_ylabel('Precipitação Média (mm)', fontsize=12)
plt.xticks(rotation=45)
ax_precip.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
st.pyplot(fig_precip)

# Consumption Distribution by Weekday Boxplot
st.subheader("Distribuição do Consumo por Dia da Semana")
weekdays_df = beer_df.copy()
weekdays_df['weekday'] = beer_df['date'].dt.day_name()
weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

fig_weekday, ax_weekday = plt.subplots(figsize=(12, 6))
sns.boxplot(x='weekday', y='consumption', data=weekdays_df, order=weekday_order, ax=ax_weekday)
ax_weekday.set_title('Distribuição do Consumo por Dia da Semana', fontsize=15)
ax_weekday.set_xlabel('Dia da Semana', fontsize=12)
ax_weekday.set_ylabel('Consumo', fontsize=12)
ax_weekday.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
st.pyplot(fig_weekday)

# Monthly Consumption Comparison (Year over Year)
st.subheader("Comparação do Consumo Mensal (Ano a Ano)")
yearly_consumption_df = beer_df.copy()
yearly_consumption_df['year'] = yearly_consumption_df['date'].dt.year
yearly_consumption_df['month_num'] = yearly_consumption_df['date'].dt.month
monthly_data = yearly_consumption_df.groupby(['year', 'month_num'])['consumption'].sum().unstack(level=0)

fig_monthly_comp, ax_monthly_comp = plt.subplots(figsize=(12, 6))
monthly_data.plot(kind='line', marker='o', ax=ax_monthly_comp)
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
ax_monthly_comp.set_xticks(range(1, 13))
ax_monthly_comp.set_xticklabels(month_names)
ax_monthly_comp.set_title('Comparação do Consumo Mensal (Ano a Ano)')
ax_monthly_comp.set_ylabel('Consumo Total')
ax_monthly_comp.set_xlabel('Mês')
ax_monthly_comp.legend(title='Ano')
ax_monthly_comp.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
st.pyplot(fig_monthly_comp)

# Daily Consumption Over Time (with Rolling Average)
st.subheader("Consumo Diário ao Longo do Tempo")
daily_consumption_df = beer_df.copy()
daily_consumption_df = daily_consumption_df.sort_values('date')
daily_consumption_df['rolling_avg'] = daily_consumption_df['consumption'].rolling(window=7).mean()

fig_daily, ax_daily = plt.subplots(figsize=(15, 7))
ax_daily.plot(daily_consumption_df['date'], daily_consumption_df['consumption'], label='Consumo Diário',
              color='steelblue', alpha=0.3, linewidth=1)
ax_daily.plot(daily_consumption_df['date'], daily_consumption_df['rolling_avg'], label='Média Móvel de 7 Dias',
              color='darkred', linewidth=2)
ax_daily.set_title('Consumo Diário ao Longo do Tempo (2015)', fontsize=16)
ax_daily.set_xlabel('Data', fontsize=12)
ax_daily.set_ylabel('Consumo', fontsize=12)
ax_daily.grid(True, linestyle='--', alpha=0.5)
ax_daily.legend()
fig_daily.autofmt_xdate()
plt.tight_layout()
st.pyplot(fig_daily)

# Low-Pass Filtered Consumption Trend
st.subheader("Tendência de Consumo (Filtro Passa-Baixa)")
softened_daily_consumption = beer_df.copy()
softened_daily_consumption = softened_daily_consumption.sort_values('date')
softened_daily_consumption['soft_trend'] = softened_daily_consumption['consumption'].ewm(span=30, adjust=False).mean()

fig_lp, ax_lp = plt.subplots(figsize=(15, 7))
ax_lp.plot(softened_daily_consumption['date'], softened_daily_consumption['consumption'], color='gray', alpha=0.2, label='Dados Diários Brutos')
ax_lp.plot(softened_daily_consumption['date'], softened_daily_consumption['soft_trend'], color='royalblue', linewidth=3, label='Tendência Passa-Baixa (EWMA)')
ax_lp.set_title('Análise de Tendência de Consumo (Filtro Passa-Baixa)', fontsize=16)
ax_lp.set_xlabel('Data')
ax_lp.set_ylabel('Consumo')
ax_lp.legend()
ax_lp.grid(True, alpha=0.3)
fig_lp.autofmt_xdate()
plt.tight_layout()
st.pyplot(fig_lp)

import streamlit as st
import pandas as pd
import os

# --- Data Loading and Preprocessing ---
@st.cache_data
def load_and_preprocess_data():
    # Simulando o carregamento do CSV
    # Em um cenário real, certifique-se que o arquivo existe no caminho abaixo
    try:
        beer_df = pd.read_csv("./Consumo_cerveja.csv")
    except FileNotFoundError:
        # Fallback para demonstração caso o arquivo não exista localmente
        st.error("Arquivo 'Consumo_cerveja.csv' não encontrado. Verifique o caminho.")
        return pd.DataFrame()

    beer_df = beer_df.dropna()
    beer_df.columns = ['date', 'avg_temp', 'min_temp', 'max_max', 'precipitation', 'is_weekend', 'consumption']

    # Conversão de tipos
    for col in ['avg_temp', 'min_temp', 'max_max', 'precipitation']:
        if beer_df[col].dtype == 'object':
            beer_df[col] = beer_df[col].str.replace(',', '.').astype(float)
    
    beer_df['is_weekend'] = beer_df['is_weekend'] > 0.5
    beer_df['date'] = pd.to_datetime(beer_df['date'])

    return beer_df

beer_df = load_and_preprocess_data()

if not beer_df.empty:
    # --- Streamlit App Layout ---
    st.set_page_config(layout="wide", page_title="Consumo de Cerveja SP")

    st.title("🍺 Análise do Consumo de Cerveja em São Paulo")

    st.markdown("""
    Esta aplicação utiliza **Streamlit Charts** nativos para apresentar uma análise exploratória do consumo de cerveja em São Paulo (2015).
    """)

    # Display the raw data
    st.header("Dados Processados")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Amostra do DataFrame")
        st.dataframe(beer_df.head(), use_container_width=True)

    with col2:
        st.subheader("Metadados")
        st.write(f"**Linhas:** {beer_df.shape[0]}")
        st.write(f"**Colunas:** {beer_df.shape[1]}")
        st.dataframe(beer_df.dtypes.astype(str).reset_index().rename(columns={
            'index': 'Coluna', 0: 'Tipo'
        }), use_container_width=True)

    # --- Visualizations with Streamlit Native Charts ---
    st.header("Visualizações Interativas")

    # 1. Monthly Average Precipitation Plot
    st.subheader("Precipitação Média Mensal")
    monthly_precip = beer_df.copy()
    monthly_precip.set_index('date', inplace=True)
    monthly_avg_precip = monthly_precip['precipitation'].resample('ME').mean()
    
    # st.bar_chart usa o índice como eixo X automaticamente
    st.bar_chart(monthly_avg_precip, color="#0077b6")

    # 2. Consumption Distribution by Weekday (Using Vega-Lite for Boxplot)
    st.subheader("Distribuição do Consumo por Dia da Semana")
    weekdays_df = beer_df.copy()
    weekdays_df['weekday'] = weekdays_df['date'].dt.day_name()
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    # Streamlit usa Vega-Lite por baixo para gráficos complexos como Boxplots
    st.vega_lite_chart(weekdays_df, {
        'mark': {'type': 'boxplot', 'extent': 'min-max'},
        'encoding': {
            'x': {'field': 'weekday', 'type': 'nominal', 'sort': weekday_order, 'title': 'Dia da Semana'},
            'y': {'field': 'consumption', 'type': 'quantitative', 'title': 'Consumo (Litros)'},
            'color': {'field': 'weekday', 'type': 'nominal', 'legend': None}
        },
        'config': {'view': {'stroke': 'transparent'}}
    }, use_container_width=True)

    # 3. Monthly Consumption Comparison
    st.subheader("Comparação do Consumo Mensal (Sazonalidade)")
    yearly_df = beer_df.copy()
    yearly_df['month_name'] = yearly_df['date'].dt.strftime('%b')
    yearly_df['month_num'] = yearly_df['date'].dt.month
    
    # Agrupando para linha
    seasonal_data = yearly_df.groupby(['month_num', 'month_name'])['consumption'].sum().reset_index()
    seasonal_data = seasonal_data.set_index('month_name')['consumption']
    
    st.line_chart(seasonal_data, color="#ff4b4b")

    # 4. Daily Consumption + Rolling Average
    st.subheader("Consumo Diário e Média Móvel (7 dias)")
    daily_df = beer_df.copy().sort_values('date')
    daily_df['Média Móvel'] = daily_df['consumption'].rolling(window=7).mean()
    daily_df = daily_df.rename(columns={'consumption': 'Consumo Real'})
    
    # Passando múltiplas colunas para o line_chart
    st.line_chart(daily_df.set_index('date')[['Consumo Real', 'Média Móvel']], color=["#3282b8", "#be3144"])

    # 5. Trend Analysis (EWMA)
    st.subheader("Tendência de Longo Prazo (Filtro Passa-Baixa)")
    trend_df = beer_df.copy().sort_values('date')
    trend_df['Tendência (EWMA)'] = trend_df['consumption'].ewm(span=30, adjust=False).mean()
    trend_df = trend_df.rename(columns={'consumption': 'Consumo Bruto'})

    st.area_chart(trend_df.set_index('date')[['Consumo Bruto', 'Tendência (EWMA)']], color=["#dddddd", "#004a7c"])

else:
    st.warning("Aguardando carregamento de dados...")

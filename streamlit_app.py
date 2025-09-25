import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from st_aggrid import AgGrid
from streamlit_option_menu import option_menu
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="📊 Dashboard Shopping Avançado", layout="wide")

COL_AGE = "Age"
COL_GENDER = "Gender"
COL_CATEGORY = "Category"
COL_PURCHASE = "Purchase Amount (USD)"
COL_LOCATION = "Location"
COL_RATING = "Review Rating"
COL_SUB_STATUS = "Subscription Status"
COL_PAY_METHOD = "Payment Method"
COL_DISCOUNT = "Discount Applied"
COL_PROMO = "Promo Code Used"
COL_PREV_PURCHASE = "Previous Purchases"
COL_FREQ = "Frequency of Purchases"
COL_SIZE = "Size"
COL_COLOR = "Color"
COL_SEASON = "Season"
COL_SHIPPING = "Shipping Type"

st.markdown(
    """
    <style>
    /* Força a cor do texto principal para escuro (corrige o tema dark) */
    body {
        color: #333;
    }

    .css-18e3th9 {background-color: #f0f2f6;}
    .stButton>button {background-color: #0072C6; color: white;}
    
    /* --- CORREÇÃO DEFINITIVA PARA AS MÉTRICAS --- */

    /* Estilo do container branco da métrica */
    div.stMetric {
        background-color: #FFFFFF !important;
        border: 1px solid #E0E0E0;
        border-radius: 8px;
        padding: 15px;
    }

    /* FORÇA a cor do TÍTULO (label) dentro da métrica */
    div.stMetric label {
        color: #555555 !important;
    }

    /* FORÇA a cor do VALOR (div) dentro da métrica */
    div.stMetric div {
        color: #0072C6 !important;
    }

    /* ------------------------------------ */

    h2, h3 {color: #0072C6;}
    </style>
    """,
    unsafe_allow_html=True
)
# ---------------------------
# Funções Auxiliares
# ---------------------------
@st.cache_resource
def train_model(X_train, y_train):
    """Treina o modelo e o armazena em cache para evitar reprocessamento."""
    numeric_features = [f for f in [COL_AGE] if f in X_train.columns]
    categorical_features = [f for f in [COL_GENDER, COL_CATEGORY, COL_LOCATION] if f in X_train.columns]

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop"
    )
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42))
    ])
    model.fit(X_train, y_train)
    return model

def generate_decision_card_content(df, boolean_col, target_col, positive_msg, negative_msg, neutral_msg):
    """Gera o conteúdo para um card de recomendação com base em uma coluna booleana."""
    if boolean_col not in df.columns or target_col not in df.columns:
        return neutral_msg, "#f1c40f", "⚠️"

    group_yes = df[df[boolean_col].astype(str).str.lower().isin(["yes", "true", "1", "subscribed"])]
    group_no = df[~df.index.isin(group_yes.index)]
    
    mean_yes = group_yes[target_col].mean() if not group_yes.empty else np.nan
    mean_no = group_no[target_col].mean() if not group_no.empty else np.nan

    if not np.isnan(mean_yes) and not np.isnan(mean_no):
        if mean_yes > mean_no:
            msg = positive_msg.format(mean_yes=mean_yes, mean_no=mean_no)
            return msg, "#2ecc71", "✅"
        else:
            msg = negative_msg.format(mean_yes=mean_yes, mean_no=mean_no)
            return msg, "#e74c3c", "❌"
    else:
        return neutral_msg, "#f1c40f", "⚠️"

@st.cache_data
def geocode_locations_with_cache(locations):
    """Geocodifica uma lista de locais, usando um cache em arquivo para persistência."""
    geolocator = Nominatim(user_agent="streamlit_app")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
    
    CACHE_FILE = "location_cache.csv"
    if os.path.exists(CACHE_FILE):
        cache_df = pd.read_csv(CACHE_FILE)
    else:
        cache_df = pd.DataFrame(columns=["Location", "Latitude", "Longitude"])

    latitudes, longitudes = [], []
    new_entries = []
    location_map = cache_df.set_index('Location').to_dict('index')

    for loc in locations:
        if loc in location_map:
            latitudes.append(location_map[loc]['Latitude'])
            longitudes.append(location_map[loc]['Longitude'])
        else:
            try:
                geo = geocode(loc)
                lat, lon = (geo.latitude, geo.longitude) if geo else (None, None)
            except Exception:
                lat, lon = None, None
            latitudes.append(lat)
            longitudes.append(lon)
            if loc not in [d['Location'] for d in new_entries]:
                new_entries.append({"Location": loc, "Latitude": lat, "Longitude": lon})
    
    if new_entries:
        new_entries_df = pd.DataFrame(new_entries)
        cache_df = pd.concat([cache_df, new_entries_df], ignore_index=True)
        cache_df.to_csv(CACHE_FILE, index=False)
        
    return latitudes, longitudes

# ---------------------------
# Menu lateral
# ---------------------------
with st.sidebar:
    selected = option_menu(
        "📊 Menu Principal",
        ["Dashboard", "Estatísticas", "Análises Avançadas", "Predição", "Insights", "Decisões"],
        icons=["bar-chart", "clipboard-data", "graph-up", "activity", "lightbulb", "check2-circle"],
        menu_icon="cast", default_index=0)

# ---------------------------
# Carregar e preparar dados
# ---------------------------
DATA_PATH = "shopping_behavior_updated.csv"
if not os.path.exists(DATA_PATH):
    st.error(f"Arquivo não encontrado: {DATA_PATH}.")
    st.stop()
df = pd.read_csv(DATA_PATH)
df.columns = [c.strip() for c in df.columns]

# ---------------------------
# Filtros da Sidebar
# ---------------------------
with st.sidebar.expander("Filtros"):
    if COL_GENDER in df.columns:
        generos = st.multiselect("Gênero", list(df[COL_GENDER].dropna().unique()), default=list(df[COL_GENDER].dropna().unique()))
    if COL_AGE in df.columns:
        age_min, age_max = int(df[COL_AGE].min()), int(df[COL_AGE].max())
        idade_min, idade_max = st.slider("Faixa etária", age_min, age_max, (age_min, age_max))
    if COL_CATEGORY in df.columns:
        categorias = st.multiselect("Categorias", list(df[COL_CATEGORY].dropna().unique()), default=list(df[COL_CATEGORY].dropna().unique()))

df_filtrado = df.copy()
if COL_GENDER in df.columns and generos:
    df_filtrado = df_filtrado[df_filtrado[COL_GENDER].isin(generos)]
if COL_AGE in df.columns:
    df_filtrado = df_filtrado[df_filtrado[COL_AGE].between(idade_min, idade_max)]
if COL_CATEGORY in df.columns and categorias:
    df_filtrado = df_filtrado[df_filtrado[COL_CATEGORY].isin(categorias)]
df_filtrado = df_filtrado.copy()

if COL_LOCATION in df_filtrado.columns:
    with st.spinner("🔍 Processando localizações..."):
        unique_locations = pd.Series(df_filtrado[COL_LOCATION].unique()).dropna().tolist()
        latitudes, longitudes = geocode_locations_with_cache(unique_locations)
        location_coords = dict(zip(unique_locations, zip(latitudes, longitudes)))
        df_filtrado["Latitude"] = df_filtrado[COL_LOCATION].map(lambda x: location_coords.get(x, (None, None))[0] if pd.notna(x) else None)
        df_filtrado["Longitude"] = df_filtrado[COL_LOCATION].map(lambda x: location_coords.get(x, (None, None))[1] if pd.notna(x) else None)

# ---------------------------
# ABA DASHBOARD (COM MAIS MÉTRICAS)
# ---------------------------
if selected == "Dashboard":
    st.markdown("## 📊 Dashboard de Performance")
    
    # --- Linha 1 de Métricas ---
    col1, col2, col3 = st.columns(3)
    col1.metric("🛒 Total de Compras", f"{len(df_filtrado):,}")
    
    if COL_PURCHASE in df_filtrado.columns:
        faturamento_total = df_filtrado[COL_PURCHASE].sum()
        gasto_medio_val = df_filtrado[COL_PURCHASE].mean()
        col2.metric("💰 Faturamento Total", f"${faturamento_total:,.2f}")
        col3.metric("💸 Ticket Médio", f"${gasto_medio_val:.2f}")

    # --- Linha 2 de Métricas ---
    col4, col5, col6 = st.columns(3)
    if COL_RATING in df_filtrado.columns:
        avaliacao_media = df_filtrado[COL_RATING].mean()
        col4.metric("⭐ Avaliação Média", f"{avaliacao_media:.2f}")
    
    if COL_PREV_PURCHASE in df_filtrado.columns:
        compras_anteriores = df_filtrado[COL_PREV_PURCHASE].sum()
        col5.metric("🔄 Total Compras Anteriores", f"{compras_anteriores:,}")
    
    if COL_CATEGORY in df_filtrado.columns:
        col6.metric("🏷️ Categorias Distintas", f"{df_filtrado[COL_CATEGORY].nunique():,}")

    st.markdown("---")
    st.markdown("### 📋 Tabela Interativa de Dados")
    AgGrid(df_filtrado)
    
    c1, c2 = st.columns(2)
    if COL_PURCHASE in df_filtrado.columns:
        c1.markdown("### 📈 Distribuição do Valor de Compras")
        c1.plotly_chart(px.histogram(df_filtrado, x=COL_PURCHASE, nbins=30), use_container_width=True)
    if COL_GENDER in df_filtrado.columns and COL_PURCHASE in df_filtrado.columns:
        c2.markdown("### 💵 Gasto Médio por Gênero")
        media_por_genero = df_filtrado.groupby(COL_GENDER)[COL_PURCHASE].mean().reset_index()
        c2.plotly_chart(px.bar(media_por_genero, x=COL_GENDER, y=COL_PURCHASE, color=COL_GENDER), use_container_width=True)

# ---------------------------
# ABA ESTATÍSTICAS
# ---------------------------
if selected == "Estatísticas":
    st.markdown("## 📌 Estatísticas Avançadas")
    st.markdown("### Estatísticas de Variáveis Numéricas")
    numeric_df = df_filtrado.select_dtypes(include=np.number)
    if not numeric_df.empty: st.write(numeric_df.describe())
    st.markdown("### Estatísticas de Variáveis Categóricas (Texto)")
    categorical_df = df_filtrado.select_dtypes(include=['object', 'category'])
    if not categorical_df.empty: st.write(categorical_df.describe())
    st.markdown("---")
    st.markdown("### 📉 Matriz de Correlação (numéricas)")
    if not numeric_df.empty:
        corr = numeric_df.corr()
        fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale="Viridis", aspect="auto")
        st.plotly_chart(fig_corr, use_container_width=True)
    st.markdown("### 📊 Boxplot de Compras por Categoria")
    if COL_CATEGORY in df_filtrado.columns and COL_PURCHASE in df_filtrado.columns:
        fig_box = px.box(df_filtrado, x=COL_CATEGORY, y=COL_PURCHASE, points="all", color=COL_CATEGORY)
        st.plotly_chart(fig_box, use_container_width=True)

# ---------------------------
# ABA ANÁLISES AVANÇADAS
# ---------------------------
if selected == "Análises Avançadas":
    st.markdown("## 📊 Análises e Cruzamentos Avançados")
    st.markdown("### 🔹 Distribuições Individuais")
    c1, c2, c3 = st.columns(3)
    if COL_AGE in df_filtrado.columns: c1.plotly_chart(px.histogram(df_filtrado, x=COL_AGE, nbins=30, title="Distribuição de Idade"), use_container_width=True)
    if COL_PURCHASE in df_filtrado.columns: c2.plotly_chart(px.histogram(df_filtrado, x=COL_PURCHASE, nbins=30, title="Distribuição de Compras"), use_container_width=True)
    if COL_RATING in df_filtrado.columns: c3.plotly_chart(px.histogram(df_filtrado, x=COL_RATING, nbins=5, title="Distribuição de Avaliações"), use_container_width=True)
    st.markdown("### 🔹 Cruzamentos entre Categorias")
    cat_cruzamentos = [(COL_GENDER, COL_PAY_METHOD), (COL_GENDER, COL_CATEGORY), (COL_LOCATION, COL_CATEGORY), (COL_CATEGORY, COL_SEASON), (COL_SHIPPING, COL_DISCOUNT)]
    for col_a, col_b in cat_cruzamentos:
        if col_a in df_filtrado.columns and col_b in df_filtrado.columns:
            st.markdown(f"#### 📌 {col_a} × {col_b}")
            cruz = df_filtrado.groupby([col_a, col_b]).size().reset_index(name="Count")
            st.plotly_chart(px.bar(cruz, x=col_a, y="Count", color=col_b, barmode="group", title=f"{col_a} × {col_b}"), use_container_width=True)

# ---------------------------
# ABA PREDIÇÃO
# ---------------------------
if selected == "Predição":
    st.markdown("## 🤖 Predição Avançada")
    numeric_features = [f for f in [COL_AGE] if f in df_filtrado.columns]
    categorical_features = [f for f in [COL_GENDER, COL_CATEGORY, COL_LOCATION] if f in df_filtrado.columns]
    if not numeric_features or COL_PURCHASE not in df_filtrado.columns:
        st.warning("Colunas necessárias para predição ausentes.")
    else:
        X = df_filtrado[numeric_features + categorical_features].dropna()
        y = df_filtrado.loc[X.index, COL_PURCHASE]
        if X.shape[0] < 10:
            st.warning("Dados insuficientes para treinar modelo.")
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            with st.spinner("🔧 Treinando modelo..."):
                model = train_model(X_train, y_train)
            y_pred = model.predict(X_test)
            st.write(f"**MSE:** {mean_squared_error(y_test, y_pred):.2f} | **R²:** {r2_score(y_test, y_pred):.2f}")
            st.markdown("### Faça sua previsão")
            idade = st.number_input("Idade", min_value=18, max_value=100, value=30)
            genero = st.selectbox("Gênero", df_filtrado[COL_GENDER].unique())
            categoria = st.selectbox("Categoria", df_filtrado[COL_CATEGORY].unique())
            localizacao = st.selectbox("Localização", df_filtrado[COL_LOCATION].unique())
            input_df = pd.DataFrame({COL_AGE: [idade], COL_GENDER: [genero], COL_CATEGORY: [categoria], COL_LOCATION: [localizacao]})
            predicao = model.predict(input_df)[0]
            st.success(f"💰 Valor estimado de compra: ${predicao:.2f}")

# ---------------------------
# ABA INSIGHTS
# ---------------------------
if selected == "Insights":
    st.markdown("## 💡 Insights Automáticos")
    if not df_filtrado.empty:
        st.write(f"- Total de compras (filtrado): {len(df_filtrado)}")
        if COL_PURCHASE in df_filtrado.columns: st.write(f"- Gasto médio: ${df_filtrado[COL_PURCHASE].mean():.2f}")
        if COL_CATEGORY in df_filtrado.columns: st.write(f"- Categoria mais popular: {df_filtrado[COL_CATEGORY].mode().iloc[0]}")
        if COL_AGE in df_filtrado.columns: st.write(f"- Idade predominante: {int(df_filtrado[COL_AGE].median())} anos (mediana)")
    else:
        st.warning("Sem dados para exibir.")

# ---------------------------
# ABA DECISÕES (COM MAIS OPÇÕES)
# ---------------------------
if selected == "Decisões":
    st.markdown("## 🎯 Recomendações de Negócio Estratégico")
    st.write("Com base nos filtros e dados selecionados, considere as seguintes ações:")

    def card(texto, cor="#2ecc71", emoji=""):
        st.markdown(f'<div style="padding:15px; margin:8px 0; border-radius:10px; background-color:{cor}; color:white; font-size:16px;">{emoji} {texto}</div>', unsafe_allow_html=True)

    # --- Cards de Análise ---
    
    msg, cor, emoji = generate_decision_card_content(df_filtrado, COL_SUB_STATUS, COL_PURCHASE,
        "Assinantes gastam MAIS (${mean_yes:.2f} vs ${mean_no:.2f}) → **Ação:** Fortalecer e promover o programa de assinaturas.",
        "Assinantes gastam MENOS (${mean_yes:.2f} vs ${mean_no:.2f}) → **Ação:** Revisar e agregar valor aos benefícios da assinatura.",
        "Dados insuficientes para analisar o impacto das assinaturas.")
    card(msg, cor, emoji)

    msg, cor, emoji = generate_decision_card_content(df_filtrado, COL_DISCOUNT, COL_PURCHASE,
        "Descontos estão associados a um ticket médio MAIOR (${mean_yes:.2f} vs ${mean_no:.2f}) → **Ação:** Continuar com promoções estratégicas.",
        "Descontos estão associados a um ticket médio MENOR (${mean_yes:.2f} vs ${mean_no:.2f}) → **Ação:** Revisar política de cupons para evitar canibalização de receita.",
        "Dados insuficientes para avaliar impacto de descontos.")
    card(msg, cor, emoji)

    msg, cor, emoji = generate_decision_card_content(df_filtrado, COL_PROMO, COL_PURCHASE,
        "O uso de cupons eleva o gasto médio (${mean_yes:.2f} vs ${mean_no:.2f}) → **Ação:** Incentivar o uso de cupons como estratégia de upsell.",
        "O uso de cupons está ligado a um gasto menor (${mean_yes:.2f} vs ${mean_no:.2f}) → **Ação:** Avaliar se os cupons atraem apenas caçadores de ofertas.",
        "Dados insuficientes para analisar o uso de cupons.")
    card(msg, cor, emoji)
    
    if COL_SEASON in df_filtrado.columns and COL_PURCHASE in df_filtrado.columns:
        try:
            sazonal_revenue = df_filtrado.groupby(COL_SEASON)[COL_PURCHASE].sum()
            top_season = sazonal_revenue.idxmax()
            top_revenue = sazonal_revenue.max()
            card(f"**Sazonalidade:** A estação '{top_season}' gera o maior faturamento (${top_revenue:,.2f}) → **Ação:** Planejar campanhas e estoque para este período.", "#3498db", "📅")
        except: pass

    if COL_FREQ in df_filtrado.columns and COL_PURCHASE in df_filtrado.columns:
        try:
            freq_spending = df_filtrado.groupby(COL_FREQ)[COL_PURCHASE].mean()
            top_freq_group = freq_spending.idxmax()
            top_avg_spend = freq_spending.max()
            card(f"**Frequência:** Clientes com frequência '{top_freq_group}' possuem o maior ticket médio (${top_avg_spend:.2f}) → **Ação:** Criar programas de fidelidade para este segmento.", "#9b59b6", "❤️")
        except: pass

    if COL_SHIPPING in df_filtrado.columns and COL_PURCHASE in df_filtrado.columns:
        try:
            shipping_spending = df_filtrado.groupby(COL_SHIPPING)[COL_PURCHASE].mean()
            top_shipping_group = shipping_spending.idxmax()
            top_avg_spend = shipping_spending.max()
            card(f"**Frete:** Compras com frete '{top_shipping_group}' têm o maior gasto médio (${top_avg_spend:.2f}) → **Ação:** Oferecer este frete como benefício para compras acima de um certo valor.", "#e67e22", "🚚")
        except: pass

    if COL_AGE in df_filtrado.columns and COL_PURCHASE in df_filtrado.columns:
        try:
            age_mean = df_filtrado.groupby(COL_AGE)[COL_PURCHASE].mean()
            faixa, valor = int(age_mean.idxmax()), age_mean.max()
            card(f"**Faixa Etária Chave:** Clientes com ≈{faixa} anos possuem o maior ticket médio (${valor:.2f}) → **Ação:** Direcionar marketing para este perfil demográfico.", "#2ecc71", "🎯")
        except: pass

    if COL_LOCATION in df_filtrado.columns and COL_PURCHASE in df_filtrado.columns:
        try:
            loc = df_filtrado.groupby(COL_LOCATION)[COL_PURCHASE].sum().idxmax()
            total_loc = df_filtrado.groupby(COL_LOCATION)[COL_PURCHASE].sum().max()
            card(f"**Mercado Principal:** A região de '{loc}' representa o maior faturamento (${total_loc:,.2f}) → **Ação:** Otimizar logística e campanhas regionais.", "#1abc9c", "📍")
        except: pass
        
    if COL_CATEGORY in df_filtrado.columns and COL_RATING in df_filtrado.columns:
        try:
            cat_rating = df_filtrado.groupby(COL_CATEGORY)[COL_RATING].mean()
            top_cat = cat_rating.idxmax()
            top_rating_avg = cat_rating.max()
            card(f"**Categoria com Melhor Avaliação:** '{top_cat}' (nota média {top_rating_avg:.2f}) → **Ação:** Promover produtos desta categoria e usá-los como referência de qualidade.", "#f39c12", "🏆")
        except: pass

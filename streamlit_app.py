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

# --- Configuração ---
st.set_page_config(page_title="📊 Dashboard Shopping Avançado", layout="wide")

# --- Estilo customizado ---
st.markdown("""
<style>
.css-18e3th9 {background-color: #f0f2f6;}
.stButton>button {background-color: #0072C6; color: white;}
.stMetric {
    background-color: #0072C6 !important;
    color: white !important;
    border-radius: 8px;
    padding: 15px;
    font-size: 18px;
}
h2, h3 {color: #0072C6;}
</style>
""", unsafe_allow_html=True)

# --- Menu lateral ---
with st.sidebar:
    selected = option_menu(
        "📊 Menu Principal",
        ["Dashboard", "Estatísticas", "Análises Avançadas", "Predição", "Insights"],
        icons=["bar-chart", "clipboard-data", "graph-up", "activity", "lightbulb"],
        menu_icon="cast",
        default_index=0,
    )

# --- Carregar dataset ---
df = pd.read_csv("shopping_behavior_updated.csv")

# --- Filtros ---
with st.sidebar.expander("Filtros"):
    generos = st.multiselect("Gênero", df["Gender"].unique(), default=df["Gender"].unique())
    idade_min, idade_max = st.slider("Faixa etária", int(df["Age"].min()), int(df["Age"].max()), (int(df["Age"].min()), int(df["Age"].max())))
    categorias = st.multiselect("Categorias", df["Category"].unique(), default=df["Category"].unique())

df_filtrado = df[
    (df["Gender"].isin(generos)) &
    (df["Age"].between(idade_min, idade_max)) &
    (df["Category"].isin(categorias))
]

# --- Geocoding com cache ---
CACHE_FILE = "location_cache.csv"

@st.cache_data
def geocode_locations_with_cache(locations):
    geolocator = Nominatim(user_agent="streamlit_app")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
    if os.path.exists(CACHE_FILE):
        cache_df = pd.read_csv(CACHE_FILE)
    else:
        cache_df = pd.DataFrame(columns=["Location", "Latitude", "Longitude"])
    latitudes, longitudes = [], []
    new_entries = []
    for loc in locations:
        cached = cache_df[cache_df["Location"] == loc]
        if not cached.empty:
            latitudes.append(cached.iloc[0]["Latitude"])
            longitudes.append(cached.iloc[0]["Longitude"])
        else:
            try:
                geo = geocode(loc)
                lat, lon = (geo.latitude, geo.longitude) if geo else (None, None)
            except:
                lat, lon = None, None
            latitudes.append(lat)
            longitudes.append(lon)
            new_entries.append({"Location": loc, "Latitude": lat, "Longitude": lon})
    if new_entries:
        cache_df = pd.concat([cache_df, pd.DataFrame(new_entries)], ignore_index=True)
        cache_df.to_csv(CACHE_FILE, index=False)
    return latitudes, longitudes

if "Location" in df_filtrado.columns:
    with st.spinner("🔍 Processando localizações..."):
        unique_locations = df_filtrado["Location"].unique()
        latitudes, longitudes = geocode_locations_with_cache(unique_locations)
        location_coords = dict(zip(unique_locations, zip(latitudes, longitudes)))
        df_filtrado["Latitude"] = df_filtrado["Location"].map(lambda x: location_coords.get(x, (None, None))[0])
        df_filtrado["Longitude"] = df_filtrado["Location"].map(lambda x: location_coords.get(x, (None, None))[1])

# === ABA DASHBOARD ===
if selected == "Dashboard":
    st.markdown("## 📊 Dashboard Avançado")
    col1, col2, col3 = st.columns(3)
    col1.metric("🛒 Total Compras", len(df_filtrado))
    col2.metric("💰 Gasto Médio", f"${df_filtrado['Purchase Amount (USD)'].mean():.2f}")
    col3.metric("🏷️ Categorias Distintas", df_filtrado["Category"].nunique())

    st.markdown("### 📋 Tabela Interativa")
    AgGrid(df_filtrado)

    st.markdown("### 📈 Distribuição do Valor de Compras")
    fig_hist = px.histogram(df_filtrado, x="Purchase Amount (USD)", nbins=30, color_discrete_sequence=["#1f77b4"])
    st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("### 💵 Gasto Médio por Gênero")
    media_por_genero = df_filtrado.groupby("Gender")["Purchase Amount (USD)"].mean().reset_index()
    fig_bar = px.bar(media_por_genero, x="Gender", y="Purchase Amount (USD)", color="Gender", color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("### 🏷️ Top Categorias Compradas")
    top_categorias = df_filtrado["Category"].value_counts().reset_index()
    top_categorias.columns = ["Category", "Count"]
    fig_cat = px.bar(top_categorias.head(10), x="Category", y="Count", color="Count", color_continuous_scale="Blues")
    st.plotly_chart(fig_cat, use_container_width=True)

    st.markdown("### 📊 Relação Idade x Valor de Compra")
    media_idade = df_filtrado.groupby("Age")["Purchase Amount (USD)"].mean().reset_index()
    fig_line = px.line(media_idade, x="Age", y="Purchase Amount (USD)", markers=True, line_shape="spline")
    st.plotly_chart(fig_line, use_container_width=True)

# === ABA ESTATÍSTICAS ===
if selected == "Estatísticas":
    st.markdown("## 📌 Estatísticas Avançadas")
    st.write(df_filtrado.describe())

    st.markdown("### 📉 Matriz de Correlação")
    numeric_df = df_filtrado.select_dtypes(include=np.number)
    corr = numeric_df.corr()
    fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale="Viridis")
    st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("### 📊 Boxplot de Compras por Categoria")
    fig_box = px.box(df_filtrado, x="Category", y="Purchase Amount (USD)", points="all", color="Category")
    st.plotly_chart(fig_box, use_container_width=True)

# === ABA ANÁLISES AVANÇADAS ===
if selected == "Análises Avançadas":
    st.markdown("## 📊 Análises e Cruzamentos Avançados")

    st.markdown("### 🔹 Distribuições Individuais")
    col1, col2, col3 = st.columns(3)
    col1.plotly_chart(px.histogram(df_filtrado, x="Age", nbins=30, title="Distribuição de Idade"))
    col2.plotly_chart(px.histogram(df_filtrado, x="Purchase Amount (USD)", nbins=30, title="Distribuição de Compras"))
    col3.plotly_chart(px.histogram(df_filtrado, x="Review Rating", nbins=5, title="Distribuição de Avaliações"))

    st.markdown("### 🔹 Cruzamentos entre Categorias")
    cat_cruzamentos = [
        ("Gender", "Payment Method"),
        ("Gender", "Category"),
        ("Location", "Payment Method"),
        ("Location", "Category"),
        ("Category", "Payment Method"),
        ("Category", "Season"),
        ("Size", "Color"),
        ("Shipping Type", "Discount Applied"),
        ("Subscription Status", "Payment Method"),
        ("Promo Code Used", "Discount Applied")
    ]
    for col_a, col_b in cat_cruzamentos:
        st.markdown(f"#### 📌 {col_a} × {col_b}")
        st.markdown(f"Análise da relação entre **{col_a}** e **{col_b}**, mostrando como as categorias interagem e influenciam o comportamento de compra.")
        cruz = df_filtrado.groupby([col_a, col_b]).size().reset_index(name="Count")
        fig_bar = px.bar(cruz, x=col_a, y="Count", color=col_b, barmode="group", title=f"{col_a} × {col_b}")
        st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown(f"Boxplot mostrando a distribuição dos valores de compra para cada categoria de **{col_a}** e **{col_b}**.")
        fig_box = px.box(df_filtrado, x=col_a, y="Purchase Amount (USD)", color=col_b, title=f"Boxplot: {col_a} × {col_b}")
        st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("### 🔹 Cruzamentos entre Numéricos e Categorias")
    num_cat_cruzamentos = [
        ("Age", "Payment Method"),
        ("Age", "Category"),
        ("Age", "Gender"),
        ("Purchase Amount (USD)", "Category"),
        ("Purchase Amount (USD)", "Payment Method"),
        ("Review Rating", "Category"),
        ("Frequency of Purchases", "Payment Method")
    ]
    for num_col, cat_col in num_cat_cruzamentos:
        st.markdown(f"#### 📌 {num_col} × {cat_col}")
        st.markdown(f"Analisando como **{num_col}** varia entre as categorias de **{cat_col}**.")
        fig_box = px.box(df_filtrado, x=cat_col, y=num_col, color=cat_col, title=f"Boxplot: {num_col} × {cat_col}")
        st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("### 🔹 Matriz de Correlação Avançada")
    numeric_df = df_filtrado.select_dtypes(include=np.number)
    try:
        cat_df_encoded = pd.get_dummies(df_filtrado.select_dtypes(include="object"))
        corr_df = pd.concat([numeric_df, cat_df_encoded], axis=1).corr()
        st.markdown("A matriz abaixo mostra a correlação entre todas as variáveis numéricas e categorias transformadas em numéricas.")
        fig_corr = px.imshow(corr_df, text_auto=True, color_continuous_scale="Viridis", title="Matriz de Correlação Completa")
        st.plotly_chart(fig_corr, use_container_width=True)
    except Exception as e:
        st.warning(f"Erro ao gerar matriz de correlação: {e}")

    st.markdown("### 🔹 Heatmaps Detalhados")
    for col_a, col_b in cat_cruzamentos:
        st.markdown(f"#### Heatmap: {col_a} × {col_b}")
        st.markdown(f"Visualização da média do valor de compra para combinações de **{col_a}** e **{col_b}**.")
        pivot_table = pd.pivot_table(df_filtrado, index=col_a, columns=col_b, values="Purchase Amount (USD)", aggfunc="mean")
        if pivot_table.shape[0] > 0 and pivot_table.shape[1] > 0:
            fig_heatmap = px.imshow(pivot_table, text_auto=True, color_continuous_scale="Viridis", title=f"Heatmap {col_a} × {col_b}")
            st.plotly_chart(fig_heatmap, use_container_width=True)

    st.markdown("### 🔹 Scatter Plots de Relações Numéricas")
    scatter_pairs = [
        ("Age", "Purchase Amount (USD)"),
        ("Age", "Review Rating"),
        ("Previous Purchases", "Purchase Amount (USD)"),
        ("Frequency of Purchases", "Purchase Amount (USD)")
    ]
    for x_col, y_col in scatter_pairs:
        st.markdown(f"#### Scatter Plot: {x_col} × {y_col}")
        st.markdown(f"Observando a relação entre **{x_col}** e **{y_col}** para detectar padrões ou tendências.")
        fig_scatter = px.scatter(df_filtrado, x=x_col, y=y_col, color="Category", title=f"{x_col} × {y_col}", opacity=0.7)
        st.plotly_chart(fig_scatter, use_container_width=True)

# === ABA PREDIÇÃO ===
if selected == "Predição":
    st.markdown("## 🤖 Predição Avançada")
    numeric_features = ["Age"]
    categorical_features = ["Gender", "Category", "Location"]
    X = df_filtrado[numeric_features + categorical_features].dropna()
    y = df_filtrado.loc[X.index, "Purchase Amount (USD)"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42))
    ])
    with st.spinner("🔧 Treinando modelo..."):
        model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    st.write(f"**MSE:** {mse:.2f} | **R²:** {r2:.2f}")

    st.markdown("### Faça sua previsão")
    idade = st.number_input("Idade", min_value=10, max_value=100, value=30)
    genero = st.selectbox("Gênero", df["Gender"].unique())
    categoria = st.selectbox("Categoria", df["Category"].unique())
    localizacao = st.selectbox("Localização", df["Location"].unique())

    input_df = pd.DataFrame({
        "Age": [idade],
        "Gender": [genero],
        "Category": [categoria],
        "Location": [localizacao]
    })
    predicao = model.predict(input_df)[0]
    st.success(f"💰 Valor estimado de compra: ${predicao:.2f}")

# === ABA INSIGHTS ===
if selected == "Insights":
    st.markdown("## 💡 Insights Automáticos")
    st.write(f"- Total de compras: {len(df_filtrado)}")
    st.write(f"- Gasto médio: ${df_filtrado['Purchase Amount (USD)'].mean():.2f}")
    st.write(f"- Categoria mais popular: {df_filtrado['Category'].mode()[0]}")
    st.write(f"- Faixa etária predominante: {df_filtrado['Age'].mode()[0]}")
    st.write(f"- Gênero predominante: {df_filtrado['Gender'].mode()[0]}")

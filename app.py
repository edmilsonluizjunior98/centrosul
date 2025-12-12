import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import unicodedata
import itertools
import requests
import json
import concurrent.futures
import urllib3

# ==============================================================================
# CONFIGURAÇÕES TÉCNICAS (Bypass de Segurança & API Key)
# ==============================================================================
# Sua chave de API
API_KEY_FIXA = "AIzaSyDDuL_uFtxEWZRmYDwXlZ_z4Vv31lLu--U"

# Desabilita avisos de segurança para rodar liso na rede corporativa
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ==============================================================================
# 1. CONFIGURAÇÃO E ESTILO (EXECUTIVO)
# ==============================================================================
st.set_page_config(
    page_title="Painel Executivo - Cartão de Todos",
    layout="wide",
    page_icon="📊"
)

# Cores
COR_VENDAS = "#F1C40F"       # Amarelo Ouro
COR_ADIMPLENCIA = "#2980B9"  # Azul Forte
COR_DESFILIACAO = "#C0392B"  # Vermelho Escuro
COR_QIA = "#27AE60"          # Verde

# CSS Personalizado
st.markdown(f"""
<style>
    .kpi-card {{
        background-color: #ffffff;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 10px;
        text-align: center;
        height: 160px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }}
    .kpi-title {{
        font-size: 14px;
        color: #666;
        text-transform: uppercase;
        font-weight: 600;
        margin-bottom: 5px;
    }}
    .kpi-value {{
        font-size: 26px;
        font-weight: 700;
        color: #333;
    }}
    .kpi-sub {{
        font-size: 13px;
        color: #555;
        margin-top: 5px;
        font-weight: 500;
    }}
    .kpi-evo {{
        font-size: 12px;
        color: #888;
        margin-top: 2px;
        font-style: italic;
    }}
    /* Estilo para a caixa da IA */
    .ia-box {{
        background-color: #f0f7ff;
        border-left: 5px solid #4285F4;
        padding: 15px;
        border-radius: 6px;
        margin-top: 15px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }}
    .border-qia {{ border-top: 4px solid {COR_QIA}; }}
    .border-vendas {{ border-top: 4px solid {COR_VENDAS}; }}
    .border-adimp {{ border-top: 4px solid {COR_ADIMPLENCIA}; }}
    .border-desf {{ border-top: 4px solid {COR_DESFILIACAO}; }}
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. CARREGAMENTO DE DADOS (ETL)
# ==============================================================================
@st.cache_data(ttl=600)
def carregar_dados():
    sheet_id = '1qnJ6yOB_iYZbKM-qYkqtS5Q7NvlLOONw35sacfMMBPo'
    url = f'https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid=0'

    try:
        # Tenta ler com bypass de SSL também na planilha, caso precise
        df = pd.read_csv(url, header=[0, 1], dtype=str, index_col=0, storage_options={'ssl': False})
    except:
        try:
            df = pd.read_csv(url, header=[0, 1], dtype=str, index_col=0)
        except:
            df = pd.read_csv(url, sep=';', header=[0, 1], dtype=str, index_col=0)

    df_empilhado = df.stack(level=0)
    df_empilhado.index.names = ['franquia', 'data']
    df_final = df_empilhado.reset_index()

    def limpar_texto(texto):
        if not isinstance(texto, str): return str(texto)
        texto = unicodedata.normalize('NFKD', texto).encode('ASCII', 'ignore').decode('utf-8')
        return texto.lower().strip().replace(' ', '_').replace('(', '').replace(')', '').replace('%', 'perc')

    df_final.columns = [limpar_texto(c) for c in df_final.columns]

    def converter_numero(valor):
        if isinstance(valor, str):
            valor = valor.strip()
            if valor in ['-', '', 'nan']: return np.nan
            valor = valor.replace('%', '').replace('.', '').replace(',', '.')
            try:
                return float(valor)
            except:
                return np.nan
        return valor

    # Aplica conversão em todas as colunas exceto as de identificação
    for col in [c for c in df_final.columns if c not in ['franquia', 'data']]:
        df_final[col] = df_final[col].apply(converter_numero)

    df_final['data'] = pd.to_datetime(df_final['data'], format='%Y/%m', errors='coerce')
    
    return df_final.dropna(subset=['data']).sort_values(by='data')

try:
    df_completo = carregar_dados()
except Exception as e:
    st.error(f"Erro ao carregar dados: {e}")
    st.stop()

# ==============================================================================
# 3. FILTROS
# ==============================================================================
st.sidebar.header("Filtros")

todas_franquias = sorted(df_completo['franquia'].unique())
pre_selecao = [f for f in todas_franquias if 'total' in f.lower()]
if not pre_selecao: pre_selecao = [todas_franquias[0]]

franquias_selecionadas = st.sidebar.multiselect(
    "Selecione Franquias:",
    options=todas_franquias,
    default=pre_selecao
)

datas = sorted(df_completo['data'].unique())
data_inicial, data_final = st.sidebar.select_slider(
    "Período de Análise:",
    options=datas,
    value=(datas[0], datas[-1]),
    format_func=lambda x: x.strftime("%m/%Y")
)

df_filtrado = df_completo[
    (df_completo['franquia'].isin(franquias_selecionadas)) &
    (df_completo['data'] >= data_inicial) &
    (df_completo['data'] <= data_final)
]

if df_filtrado.empty:
    st.warning("Sem dados para os filtros selecionados.")
    st.stop()

# ==============================================================================
# 4. CÁLCULOS KPI
# ==============================================================================
df_agregado = df_filtrado.groupby('data')[['vendas', 'qia', 'adimplencia', 'desfiliacao']].sum().reset_index()
# Correção da média da adimplência
df_agregado['adimplencia'] = df_filtrado.groupby('data')['adimplencia'].mean().values

# --- QIA ---
qia_inicial = df_agregado.iloc[0]['qia']
qia_final = df_agregado.iloc[-1]['qia']
evolucao_qia = qia_final - qia_inicial

# --- VENDAS ---
total_vendas = df_agregado['vendas'].sum()
media_vendas = df_agregado['vendas'].mean()
vendas_ini = df_agregado.iloc[0]['vendas']
vendas_fim = df_agregado.iloc[-1]['vendas']
evo_vendas = vendas_fim - vendas_ini

# --- DESFILIAÇÃO ---
total_desf = df_agregado['desfiliacao'].sum()
media_desf = df_agregado['desfiliacao'].mean()
desf_ini = df_agregado.iloc[0]['desfiliacao']
desf_fim = df_agregado.iloc[-1]['desfiliacao']
evo_desf = desf_fim - desf_ini

# --- ADIMPLÊNCIA ---
media_adimp_periodo = df_agregado['adimplencia'].mean()
adimp_ini = df_agregado.iloc[0]['adimplencia']
adimp_fim = df_agregado.iloc[-1]['adimplencia']
evo_adimp = adimp_fim - adimp_ini

# ==============================================================================
# 5. VISUALIZAÇÃO
# ==============================================================================
st.title("Painel de Resultados")
st.markdown(f"**Período:** {data_inicial.strftime('%m/%Y')} a {data_final.strftime('%m/%Y')}")

# --- CARTÕES KPI ---
c1, c2, c3, c4 = st.columns(4)

def card_html(titulo, valor, sub1, sub2, classe):
    return f"""
    <div class="kpi-card {classe}">
        <div class="kpi-title">{titulo}</div>
        <div class="kpi-value">{valor}</div>
        <div class="kpi-sub">{sub1}</div>
        <div class="kpi-evo">{sub2}</div>
    </div>
    """

with c1:
    st.markdown(card_html(
        "QIA (Saldo)", 
        f"{qia_final:,.0f}".replace(',', '.'), 
        f"Inicial: {qia_inicial:,.0f}",
        f"Evolução: {evolucao_qia:+,.0f}".replace(',', '.'),
        "border-qia"
    ), unsafe_allow_html=True)

with c2:
    st.markdown(card_html(
        "Vendas (Total)", 
        f"{total_vendas:,.0f}".replace(',', '.'), 
        f"Média Mensal: {media_vendas:,.0f}",
        f"Evolução: {evo_vendas:+,.0f}".replace(',', '.'),
        "border-vendas"
    ), unsafe_allow_html=True)

with c3:
    st.markdown(card_html(
        "Adimplência (Média)", 
        f"{media_adimp_periodo:.2f}%", 
        f"Final: {adimp_fim:.2f}%",
        f"Evolução: {evo_adimp:+.2f}%",
        "border-adimp"
    ), unsafe_allow_html=True)

with c4:
    st.markdown(card_html(
        "Desfiliações (Total)", 
        f"{total_desf:,.0f}".replace(',', '.'), 
        f"Média Mensal: {media_desf:,.0f}",
        f"Evolução: {evo_desf:+,.0f}".replace(',', '.'),
        "border-desf"
    ), unsafe_allow_html=True)

st.markdown("---")

# --- GRÁFICOS ---
line_styles = ['solid', 'dash', 'dot', 'dashdot', 'longdash']
line_cycler = itertools.cycle(line_styles)
franquia_styles = {f: next(line_cycler) for f in franquias_selecionadas}

# 1. GRÁFICO COMBINADO: VENDAS VS DESFILIAÇÃO
st.subheader("Entradas (Vendas) vs Saídas (Desfiliação)")
fig_comb = go.Figure()

for f in franquias_selecionadas:
    dados_f = df_filtrado[df_filtrado['franquia'] == f]
    estilo = franquia_styles[f]
    
    fig_comb.add_trace(go.Scatter(
        x=dados_f['data'], y=dados_f['vendas'],
        mode='lines+markers', name=f'{f} - Vendas',
        line=dict(color=COR_VENDAS, width=3, dash=estilo),
        marker=dict(symbol='circle')
    ))
    
    fig_comb.add_trace(go.Scatter(
        x=dados_f['data'], y=dados_f['desfiliacao'],
        mode='lines+markers', name=f'{f} - Desfiliação',
        line=dict(color=COR_DESFILIACAO, width=3, dash=estilo),
        marker=dict(symbol='x')
    ))

fig_comb.update_layout(template="plotly_white", hovermode="x unified", legend=dict(orientation="h", y=1.1))
st.plotly_chart(fig_comb, use_container_width=True)


# 2. QIA E ADIMPLÊNCIA
col_g1, col_g2 = st.columns(2)

with col_g1:
    st.subheader("Evolução do QIA")
    fig_qia = go.Figure()
    for f in franquias_selecionadas:
        dados_f = df_filtrado[df_filtrado['franquia'] == f]
        fig_qia.add_trace(go.Scatter(
            x=dados_f['data'], y=dados_f['qia'],
            mode='lines+markers', name=f,
            line=dict(color=COR_QIA, dash=franquia_styles[f], width=3, shape='spline')
        ))
    fig_qia.update_layout(template="plotly_white", hovermode="x unified", legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig_qia, use_container_width=True)

with col_g2:
    st.subheader("Histórico de Adimplência")
    fig_adimp = go.Figure()
    for f in franquias_selecionadas:
        dados_f = df_filtrado[df_filtrado['franquia'] == f]
        fig_adimp.add_trace(go.Scatter(
            x=dados_f['data'], y=dados_f['adimplencia'],
            mode='lines+markers', name=f,
            line=dict(color=COR_ADIMPLENCIA, dash=franquia_styles[f], width=3, shape='spline')
        ))
    fig_adimp.update_layout(template="plotly_white", hovermode="x unified", legend=dict(orientation="h", y=1.1))
    fig_adimp.update_yaxes(ticksuffix="%")
    st.plotly_chart(fig_adimp, use_container_width=True)

# ==============================================================================
# 6. ANÁLISE COM GEMINI AI (CORRIGIDO PARA REDE CORPORATIVA & MODELO NOVO)
# ==============================================================================
st.markdown("---")
st.header("🤖 Inteligência Artificial (Gemini)")

# Função robusta que tenta vários modelos e ignora SSL
def chamada_api_robusta(prompt):
    # Lista de prioridade de modelos (Novo -> Antigo)
    modelos = ["gemini-1.5-flash", "gemini-1.5-flash-latest", "gemini-1.0-pro", "gemini-pro"]
    headers = {'Content-Type': 'application/json'}
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    
    erros = []
    
    for modelo in modelos:
        # Monta a URL para o modelo específico
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{modelo}:generateContent?key={API_KEY_FIXA}"
        
        try:
            # verify=False é o SEGREDO para funcionar na sua rede
            response = requests.post(url, headers=headers, json=data, verify=False, timeout=8)
            
            if response.status_code == 200:
                # Sucesso! Retorna o texto
                return response.json()['candidates'][0]['content']['parts'][0]['text']
            else:
                erros.append(f"{modelo}: Status {response.status_code}")
                
        except Exception as e:
            erros.append(f"{modelo}: Erro Conexão")
            
    return f"Não foi possível conectar. Detalhes: {', '.join(erros)}"

def analisar_franquia(franquia, df_f):
    # Prepara os dados para a IA
    vendas = df_f['vendas'].sum()
    cancel = df_f['desfiliacao'].sum()
    saldo = vendas - cancel
    
    q_ini = df_f.iloc[0]['qia']
    q_fim = df_f.iloc[-1]['qia']
    delta_q = q_fim - q_ini
    
    adimp = df_f['adimplencia'].mean()
    
    # Prompt curto e direto
    prompt = f"""
    Analise a franquia '{franquia}' do Cartão de Todos.
    DADOS:
    - Saldo de Vidas: {saldo:+,.0f} (Vendas {vendas} vs Saídas {cancel})
    - QIA (Base Pagante): {q_ini:,.0f} -> {q_fim:,.0f} (Var: {delta_q:+,.0f})
    - Adimplência: {adimp:.1f}%
    
    TAREFA: Responda em 1 frase curta com um veredito (Positivo/Negativo) e o motivo.
    """
    
    texto = chamada_api_robusta(prompt)
    cor = "green" if delta_q > 0 else "red"
    
    return f"""
    <div class="ia-box">
        <strong>{franquia}</strong> <span style="font-size:12px;color:#555">(QIA: <span style="color:{cor}">{delta_q:+,.0f}</span>)</span><br>
        <i>{texto}</i>
    </div>
    """

if st.button("🚀 Gerar Análise com IA", type="primary", use_container_width=True):
    progresso = st.progress(0, text="Iniciando IA...")
    resultados_html = []
    
    # Processamento Paralelo (4 ao mesmo tempo)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(analisar_franquia, f, df_filtrado[df_filtrado['franquia'] == f]): f for f in franquias_selecionadas}
        
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            resultados_html.append(future.result())
            progresso.progress((i + 1) / len(franquias_selecionadas), text=f"Analisando {i+1}/{len(franquias_selecionadas)}...")
            
    progresso.empty()
    
    # Exibe em colunas
    cols = st.columns(2)
    for i, html in enumerate(resultados_html):
        with cols[i % 2]:
            st.markdown(html, unsafe_allow_html=True)

# --- TABELA DE DADOS (AJUSTADA) ---
st.markdown("### 📋 Base de Dados Detalhada")

colunas_para_mostrar = ['franquia', 'data', 'qia', 'vendas', 'adimplencia', 'desfiliacao', 'churn_rate']
cols_existentes = [c for c in colunas_para_mostrar if c in df_filtrado.columns]

st.dataframe(
    df_filtrado[cols_existentes]
    .sort_values(by=['franquia', 'data'], ascending=[True, False])
    .style.format({
        'data': lambda x: x.strftime('%m/%Y'),
        'qia': '{:,.0f}',
        'vendas': '{:,.0f}',
        'desfiliacao': '{:,.0f}',
        'adimplencia': '{:.2f}%',
        'churn_rate': '{:.2f}%'
    }),
    use_container_width=True
)
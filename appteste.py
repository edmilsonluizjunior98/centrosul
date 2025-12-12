import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import unicodedata
import itertools
import urllib3

# ==============================================================================
# CONFIGURAÇÕES TÉCNICAS
# ==============================================================================
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ==============================================================================
# 1. ESTILO
# ==============================================================================
st.set_page_config(page_title="Gestão Cartão de Todos",  layout="wide", page_icon="💼")

COR_VENDAS = "#F1C40F"       # Amarelo
COR_ADIMPLENCIA = "#2980B9"  # Azul
COR_DESFILIACAO = "#C0392B"  # Vermelho
COR_QIA = "#27AE60"          # Verde
COR_ALERTA = "#E74C3C"

st.markdown(f"""
<style>
    .kpi-card {{ background-color: #ffffff; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center; height: 140px; display: flex; flex-direction: column; justify-content: center; }}
    .kpi-value {{ font-size: 24px; font-weight: 700; color: #333; }}
    .kpi-title {{ font-size: 13px; color: #666; text-transform: uppercase; font-weight: 600; }}
    .record-box {{ background-color: #f0f9ff; border-left: 5px solid #333; padding: 15px; border-radius: 5px; margin-bottom: 10px; }}
    .alert-box {{ background-color: #fff5f5; border-left: 5px solid {COR_ALERTA}; padding: 15px; border-radius: 5px; margin-bottom: 10px; }}
    .border-qia {{ border-top: 4px solid {COR_QIA}; }}
    .border-vendas {{ border-top: 4px solid {COR_VENDAS}; }}
    .border-adimp {{ border-top: 4px solid {COR_ADIMPLENCIA}; }}
    .border-desf {{ border-top: 4px solid {COR_DESFILIACAO}; }}
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. DADOS
# ==============================================================================
@st.cache_data(ttl=600)
def carregar_dados():
    url = 'https://docs.google.com/spreadsheets/d/1qnJ6yOB_iYZbKM-qYkqtS5Q7NvlLOONw35sacfMMBPo/export?format=csv&gid=0'
    try: df = pd.read_csv(url, header=[0, 1], dtype=str, index_col=0, storage_options={'ssl': False})
    except: 
        try: df = pd.read_csv(url, header=[0, 1], dtype=str, index_col=0)
        except: df = pd.read_csv(url, sep=';', header=[0, 1], dtype=str, index_col=0)
    
    df = df.stack(level=0).reset_index()
    df.columns = ['franquia', 'data'] + [unicodedata.normalize('NFKD', c).encode('ASCII', 'ignore').decode('utf-8').lower().strip().replace(' ', '_').replace('(', '').replace(')', '').replace('%', 'perc') for c in df.columns[2:]]
    for c in df.columns[2:]:
        df[c] = df[c].apply(lambda x: float(str(x).strip().replace('%','').replace('.','').replace(',','.')) if str(x).strip() not in ['','-','nan'] else np.nan)
    df['data'] = pd.to_datetime(df['data'], format='%Y/%m', errors='coerce')
    return df.dropna(subset=['data']).sort_values('data')

try: df_completo = carregar_dados()
except Exception as e: st.error(f"Erro dados: {e}"); st.stop()

# ==============================================================================
# 3. FILTROS
# ==============================================================================
st.sidebar.header("🔍 Configuração")
todas = sorted(df_completo['franquia'].unique())
pre_sel = [f for f in todas if 'total' in f.lower()] or [todas[0]]

datas = sorted(df_completo['data'].unique())
dt_ini, dt_fim = st.sidebar.select_slider("Período:", options=datas, value=(datas[0], datas[-1]), format_func=lambda x: x.strftime("%m/%Y"))

# ==============================================================================
# 4. DASHBOARD INTELIGENTE
# ==============================================================================
tab_visual, tab_resumo = st.tabs(["📊 Dashboard Visual", "📑 Resumo dos Números"])

# ------------------------------------------------------------------------------
# ABA 1: VISUAL
# ------------------------------------------------------------------------------
with tab_visual:
    franquias = st.multiselect("Selecione para o Gráfico:", todas, default=pre_sel)
    df_filt = df_completo[(df_completo['franquia'].isin(franquias)) & (df_completo['data'] >= dt_ini) & (df_completo['data'] <= dt_fim)]
    
    if df_filt.empty:
        st.warning("Sem dados.")
    else:
        # Lógica de Cores Automática
        modo_comparacao = len(franquias) > 1
        if modo_comparacao:
            cores_ciclo = itertools.cycle(px.colors.qualitative.Bold)
            mapa_cores = {f: next(cores_ciclo) for f in franquias}
        
        # KPIs
        agg = df_filt.groupby('data')[['vendas', 'qia', 'adimplencia', 'desfiliacao']].sum().reset_index()
        agg['adimplencia'] = df_filt.groupby('data')['adimplencia'].mean().values
        q_fim, q_ini = agg.iloc[-1]['qia'], agg.iloc[0]['qia']
        v_tot, d_tot = agg['vendas'].sum(), agg['desfiliacao'].sum()
        adm_med = agg['adimplencia'].mean()

        st.markdown(f"### Visão Geral: {dt_ini.strftime('%m/%Y')} a {dt_fim.strftime('%m/%Y')}")

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f"<div class='kpi-card border-qia'><div class='kpi-title'>QIA (Saldo)</div><div class='kpi-value'>{q_fim:,.0f}</div><div style='color:#777;font-size:12px'>Var: {q_fim-q_ini:+,.0f}</div></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='kpi-card border-vendas'><div class='kpi-title'>Vendas</div><div class='kpi-value'>{v_tot:,.0f}</div><div style='color:#777;font-size:12px'>Total</div></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='kpi-card border-adimp'><div class='kpi-title'>Adimplência</div><div class='kpi-value'>{adm_med:.1f}%</div><div style='color:#777;font-size:12px'>Média</div></div>", unsafe_allow_html=True)
        c4.markdown(f"<div class='kpi-card border-desf'><div class='kpi-title'>Desfiliações</div><div class='kpi-value'>{d_tot:,.0f}</div><div style='color:#777;font-size:12px'>Total</div></div>", unsafe_allow_html=True)

        st.markdown("---")
        
        # Gráficos
        fig1 = go.Figure()
        for f in franquias:
            d = df_filt[df_filt['franquia'] == f]
            cor_vendas = mapa_cores[f] if modo_comparacao else COR_VENDAS
            cor_desf = mapa_cores[f] if modo_comparacao else COR_DESFILIACAO
            
            fig1.add_trace(go.Scatter(x=d['data'], y=d['vendas'], name=f'{f} Vendas' if modo_comparacao else 'Vendas', line=dict(color=cor_vendas, width=3), legendgroup=f))
            fig1.add_trace(go.Scatter(x=d['data'], y=d['desfiliacao'], name=f'{f} Saída' if modo_comparacao else 'Desfiliação', line=dict(color=cor_desf, width=3, dash='dot'), legendgroup=f))
            
        fig1.update_layout(title="Comparativo: Entradas vs Saídas", template="plotly_white", hovermode="x unified", height=400, margin=dict(t=40,b=20))
        st.plotly_chart(fig1, use_container_width=True)

        cg1, cg2 = st.columns(2)
        with cg1:
            fig2 = go.Figure()
            for f in franquias:
                d = df_filt[df_filt['franquia'] == f]
                c = mapa_cores[f] if modo_comparacao else COR_QIA
                fig2.add_trace(go.Scatter(x=d['data'], y=d['qia'], name=f if modo_comparacao else 'QIA', line=dict(color=c, width=3)))
            fig2.update_layout(title="Evolução QIA", template="plotly_white", hovermode="x unified", height=300, margin=dict(t=30,b=20))
            st.plotly_chart(fig2, use_container_width=True)

        with cg2:
            fig3 = go.Figure()
            for f in franquias:
                d = df_filt[df_filt['franquia'] == f]
                c = mapa_cores[f] if modo_comparacao else COR_ADIMPLENCIA
                fig3.add_trace(go.Scatter(x=d['data'], y=d['adimplencia'], name=f if modo_comparacao else 'Adimplência', line=dict(color=c, width=3)))
            fig3.update_layout(title="Histórico Adimplência", template="plotly_white", hovermode="x unified", height=300, margin=dict(t=30,b=20))
            st.plotly_chart(fig3, use_container_width=True)
            
        st.markdown("### Detalhamento")
        st.dataframe(df_filt[['franquia', 'data', 'qia', 'vendas', 'adimplencia', 'desfiliacao']].sort_values(['franquia', 'data'], ascending=[True, False]), use_container_width=True)

# ------------------------------------------------------------------------------
# ABA 2: RESUMO
# ------------------------------------------------------------------------------
with tab_resumo:
    st.header("📑 Resumo de Performance")
    franquia_resumo = st.selectbox("Selecione a Franquia:", options=todas)
    
    df_r = df_completo[(df_completo['franquia'] == franquia_resumo) & (df_completo['data'] >= dt_ini) & (df_completo['data'] <= dt_fim)].copy()
    
    if df_r.empty:
        st.warning("Sem dados para o período.")
    else:
        # Cálculos de QIA e Pico
        q_start = df_r.iloc[0]['qia']
        q_end = df_r.iloc[-1]['qia']
        q_evo = q_end - q_start
        
        # Pico de QIA
        idx_max_qia = df_r['qia'].idxmax()
        q_pico_val = df_r.loc[idx_max_qia, 'qia']
        q_pico_data = df_r.loc[idx_max_qia, 'data'].strftime('%m/%Y')
        
        cor_evo = "green" if q_evo > 0 else "red"
        
        # Bloco Verde com Pico Incluído
        st.markdown(f"""
        <div style="background-color: #f1f8e9; padding: 20px; border-radius: 10px; border: 1px solid #c5e1a5; margin-bottom: 30px;">
            <h3 style="color: #2e7d32; margin:0; text-align:center;">🌱 Evolução da Base (QIA)</h3>
            <div style="display: flex; justify-content: space-around; margin-top: 15px; text-align:center;">
                <div><small>Início</small><br><span style="font-size: 24px; font-weight:bold;">{q_start:,.0f}</span></div>
                <div><small>Fim</small><br><span style="font-size: 24px; font-weight:bold;">{q_end:,.0f}</span></div>
                <div><small>Crescimento</small><br><span style="font-size: 24px; font-weight:bold; color: {cor_evo};">{q_evo:+,.0f}</span></div>
            </div>
            <div style="margin-top: 15px; text-align: center; border-top: 1px dashed #a5d6a7; padding-top: 10px;">
                <span style="font-size: 16px; color: #1b5e20;">🏆 <b>Pico no Ano:</b> {q_pico_val:,.0f} contratos em {q_pico_data}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Rankings
        idx_max_v = df_r['vendas'].idxmax()
        idx_max_a = df_r['adimplencia'].idxmax()
        idx_min_d = df_r['desfiliacao'].idxmin()
        idx_min_v = df_r['vendas'].idxmin()
        idx_min_a = df_r['adimplencia'].idxmin()
        idx_max_d = df_r['desfiliacao'].idxmax()

        c_melhores, c_piores = st.columns(2)

        with c_melhores:
            st.markdown("### 🏆 Melhores Momentos")
            st.markdown(f"""
            <div class="record-box"><b>Maior Venda:</b> {df_r.loc[idx_max_v, 'vendas']:,.0f} <br><small>{df_r.loc[idx_max_v, 'data'].strftime('%m/%Y')}</small></div>
            <div class="record-box"><b>Maior Adimplência:</b> {df_r.loc[idx_max_a, 'adimplencia']:.2f}% <br><small>{df_r.loc[idx_max_a, 'data'].strftime('%m/%Y')}</small></div>
            <div class="record-box"><b>Menor Desfiliação:</b> {df_r.loc[idx_min_d, 'desfiliacao']:,.0f} <br><small>{df_r.loc[idx_min_d, 'data'].strftime('%m/%Y')}</small></div>
            """, unsafe_allow_html=True)

        with c_piores:
            st.markdown("### ⚠️ Pontos de Atenção")
            st.markdown(f"""
            <div class="alert-box"><b>Pior Venda:</b> {df_r.loc[idx_min_v, 'vendas']:,.0f} <br><small>{df_r.loc[idx_min_v, 'data'].strftime('%m/%Y')}</small></div>
            <div class="alert-box"><b>Pior Adimplência:</b> {df_r.loc[idx_min_a, 'adimplencia']:.2f}% <br><small>{df_r.loc[idx_min_a, 'data'].strftime('%m/%Y')}</small></div>
            <div class="alert-box"><b>Maior Desfiliação:</b> {df_r.loc[idx_max_d, 'desfiliacao']:,.0f} <br><small>{df_r.loc[idx_max_d, 'data'].strftime('%m/%Y')}</small></div>
            """, unsafe_allow_html=True)
            
        st.markdown("---")
        texto_copiar = f"""
RESUMO DE PERFORMANCE - {franquia_resumo}
Período: {dt_ini.strftime('%m/%Y')} a {dt_fim.strftime('%m/%Y')}

🔹 EVOLUÇÃO BASE (QIA):
Início: {q_start:,.0f} | Fim: {q_end:,.0f} | Saldo: {q_evo:+,.0f}
🏆 PICO DE QIA: {q_pico_val:,.0f} ({q_pico_data})

🏆 MELHORES MOMENTOS:
Max Vendas: {df_r.loc[idx_max_v, 'vendas']:,.0f} ({df_r.loc[idx_max_v, 'data'].strftime('%m/%Y')})
Max Adimplência: {df_r.loc[idx_max_a, 'adimplencia']:.2f}% ({df_r.loc[idx_max_a, 'data'].strftime('%m/%Y')})
Menor Desfiliação: {df_r.loc[idx_min_d, 'desfiliacao']:,.0f} ({df_r.loc[idx_min_d, 'data'].strftime('%m/%Y')})

⚠️ PONTOS DE ATENÇÃO:
Min Vendas: {df_r.loc[idx_min_v, 'vendas']:,.0f} ({df_r.loc[idx_min_v, 'data'].strftime('%m/%Y')})
Min Adimplência: {df_r.loc[idx_min_a, 'adimplencia']:.2f}% ({df_r.loc[idx_min_a, 'data'].strftime('%m/%Y')})
Pico de Saída: {df_r.loc[idx_max_d, 'desfiliacao']:,.0f} ({df_r.loc[idx_max_d, 'data'].strftime('%m/%Y')})
        """
        st.text_area("📋 Texto Pronto (Copie e Cole):", value=texto_copiar, height=300)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import unicodedata
import itertools 
import requests
import io
import urllib3
import streamlit.components.v1 as components
from datetime import datetime

# Desabilita avisos de SSL
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ==============================================================================
# CONFIGURAÇÕES E ESTILO
# ==============================================================================
st.set_page_config(page_title="Gestão Cartão de Todos", layout="wide", page_icon="💼")

# CORES PADRÃO (Visão Única)
COR_VENDAS_PADRAO = "#F59E0B"       # Amarelo/Laranja
COR_DESFILIACAO_PADRAO = "#DC2626"  # Vermelho
COR_SALDO_PADRAO = "#4ade80"        # Verde Suave Sólido (para texto/marcadores)
COR_SALDO_TRANSPARENTE = "rgba(74, 222, 128, 0.5)" # Verde Suave Transparente (para a linha)
COR_QIA = "#16A34A"                 # Verde Folha
COR_ADIMPLENCIA = "#2563EB"         # Azul Royal

# PALETA DE CORES (Comparação)
PALETA_CORES = [
    "#2563EB", # Azul
    "#D97706", # Laranja
    "#7C3AED", # Roxo
    "#059669", # Verde
    "#DB2777", # Rosa
    "#0891B2"  # Ciano
]

st.markdown(f"""
<style>
    /* Estilo dos KPIs */
    .kpi-card {{ background-color: #ffffff; padding: 15px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); text-align: center; height: 160px; display: flex; flex-direction: column; justify-content: center; border: 1px solid #eee; }}
    
    .kpi-value {{ font-size: 26px; font-weight: 700; color: #1e293b; margin-bottom: 5px; }}
    .kpi-title {{ font-size: 12px; color: #64748b; text-transform: uppercase; font-weight: 600; margin-bottom: 10px; letter-spacing: 0.5px; }}
    .kpi-sub {{ font-size: 12px; color: #94a3b8; border-top: 1px solid #f1f5f9; padding-top: 8px; margin-top: 5px; }}
    
    .border-qia {{ border-top: 4px solid {COR_QIA}; }}
    .border-vendas {{ border-top: 4px solid {COR_VENDAS_PADRAO}; }}
    .border-adimp {{ border-top: 4px solid {COR_ADIMPLENCIA}; }}
    .border-desf {{ border-top: 4px solid {COR_DESFILIACAO_PADRAO}; }}

    /* Cards de Destaque */
    .destaque-card {{ background-color: #f8fafc; border-left: 4px solid #334155; padding: 15px; border-radius: 4px; margin-bottom: 10px; }}
    .atencao-card {{ background-color: #fef2f2; border-left: 4px solid #ef4444; padding: 15px; border-radius: 4px; margin-bottom: 10px; }}
    .card-label {{ font-weight: 600; color: #334155; font-size: 14px; display: block; }}
    .card-value {{ font-size: 13px; color: #64748b; }}
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# FUNÇÕES AUXILIARES
# ==============================================================================
def to_scalar(val):
    if isinstance(val, (pd.Series, np.ndarray, pd.DataFrame)):
        try: return val.item()
        except: return val.iloc[0] if len(val) > 0 else 0
    return val

def fmt(valor):
    try:
        if isinstance(valor, (float, int)):
            return "{:,.0f}".format(valor).replace(",", "X").replace(".", ".").replace("X", ".")
        return str(valor)
    except:
        return str(valor)

def fmt_dec(valor):
    try:
        return "{:,.2f}".format(valor).replace(",", "X").replace(".", ",").replace("X", ".")
    except:
        return str(valor)

def formatar_mes(data):
    meses = {1: 'Jan', 2: 'Fev', 3: 'Mar', 4: 'Abr', 5: 'Mai', 6: 'Jun', 
             7: 'Jul', 8: 'Ago', 9: 'Set', 10: 'Out', 11: 'Nov', 12: 'Dez'}
    try:
        return f"{meses[data.month]}/{data.year}"
    except:
        return str(data)

# ==============================================================================
# 1. CARREGAMENTO DE DADOS (ONLINE)
# ==============================================================================
@st.cache_data(ttl=300) 
def carregar_dados():
    url = 'https://docs.google.com/spreadsheets/d/1qnJ6yOB_iYZbKM-qYkqtS5Q7NvlLOONw35sacfMMBPo/export?format=csv&gid=0'
    try:
        response = requests.get(url, verify=False, timeout=15)
        response.raise_for_status()
        
        try: df = pd.read_csv(io.StringIO(response.text), header=[0, 1], dtype=str, index_col=0)
        except: df = pd.read_csv(io.StringIO(response.text), sep=';', header=[0, 1], dtype=str, index_col=0)
        
        df = df.stack(level=0, future_stack=True).reset_index()
        
        new_cols = ['franquia', 'data']
        for c in df.columns[2:]:
            clean_c = unicodedata.normalize('NFKD', str(c)).encode('ASCII', 'ignore').decode('utf-8')
            clean_c = clean_c.lower().strip().replace(' ', '_').replace('(', '').replace(')', '').replace('%', 'perc')
            new_cols.append(clean_c)
        df.columns = new_cols
        
        for c in df.columns[2:]:
            df[c] = df[c].apply(lambda x: float(str(x).strip().replace('%','').replace('.','').replace(',','.')) if str(x).strip() not in ['','-','nan'] else 0.0)
            
        df['data'] = pd.to_datetime(df['data'], format='%Y/%m', errors='coerce')
        df = df.dropna(subset=['data']).sort_values('data')
        df['mes_exibicao'] = df['data'].apply(formatar_mes)
        
        # Cálculo da Taxa de Churn (%)
        df['churn_rate'] = df.apply(lambda row: (row['desfiliacao'] / row['qia'] * 100) if row['qia'] > 0 else 0.0, axis=1)

        return df
    except Exception as e:
        st.error(f"⚠️ Erro de Conexão com a Planilha: {e}")
        return pd.DataFrame()

df_completo = carregar_dados()
if df_completo.empty: st.stop()

# ==============================================================================
# 2. FILTROS
# ==============================================================================
st.sidebar.header("🔍 Configuração")
datas = sorted(df_completo['data'].unique())
dt_ini, dt_fim = st.sidebar.select_slider("Período:", options=datas, value=(datas[0], datas[-1]), format_func=formatar_mes)

df_periodo = df_completo[(df_completo['data'] >= dt_ini) & (df_completo['data'] <= dt_fim)]

# Filtro de Volume (excluindo Centrosul do calculo)
df_calc_vol = df_periodo[df_periodo['franquia'] != 'CENTROSUL']
vendas_por_franquia = df_calc_vol.groupby('franquia')['vendas'].sum()
franquias_ativas = vendas_por_franquia[vendas_por_franquia >= 100].index.tolist()

if 'CENTROSUL' in df_completo['franquia'].unique() and 'CENTROSUL' not in franquias_ativas:
    franquias_ativas.insert(0, 'CENTROSUL')

if not franquias_ativas:
    st.error("Sem dados suficientes no período.")
    st.stop()

todas_opcoes = sorted(franquias_ativas)
if 'CENTROSUL' in todas_opcoes: 
    todas_opcoes.remove('CENTROSUL')
    todas_opcoes.insert(0, 'CENTROSUL')

franquias_sel = st.sidebar.multiselect("Selecione Unidade(s):", todas_opcoes, default=[todas_opcoes[0]])

if not franquias_sel:
    st.warning("Selecione ao menos uma unidade.")
    st.stop()

df_viz = df_periodo[df_periodo['franquia'].isin(franquias_sel)].copy()

# Nome do contexto
if len(franquias_sel) == 1:
    nome_relatorio = franquias_sel[0]
elif "CENTROSUL" in franquias_sel and len(franquias_sel) == 1:
    nome_relatorio = "Regional CENTROSUL"
else:
    nome_relatorio = "Comparativo Selecionado"

# ==============================================================================
# 3. DASHBOARD UNIFICADO
# ==============================================================================
tab_dash, tab_ia = st.tabs(["📊 Visão Tática", "🧠 Relatório IA"])

with tab_dash:
    if df_viz.empty:
        st.info("Sem dados.")
    else:
        # --- BLOCO 1: KPIs ---
        if len(franquias_sel) > 1 and "CENTROSUL" in franquias_sel:
            df_kpi = df_viz[df_viz['franquia'] != "CENTROSUL"]
        else:
            df_kpi = df_viz

        agg = df_kpi.groupby('data')[['vendas', 'qia', 'adimplencia', 'desfiliacao']].sum().reset_index()
        agg['adimplencia'] = df_kpi.groupby('data')['adimplencia'].mean().values
        agg['churn_rate'] = df_kpi.groupby('data')['churn_rate'].mean().values 
        
        q_fim = to_scalar(agg.iloc[-1]['qia'])
        q_ini = to_scalar(agg.iloc[0]['qia'])
        q_var = q_fim - q_ini
        
        v_tot = to_scalar(agg['vendas'].sum())
        v_med = to_scalar(agg['vendas'].mean())
        
        d_tot = to_scalar(agg['desfiliacao'].sum())
        d_med = to_scalar(agg['desfiliacao'].mean())
        
        churn_medio_periodo = agg['churn_rate'].mean()
        adm_med = to_scalar(agg['adimplencia'].mean())

        st.markdown(f"### Visão: {nome_relatorio}")
        c1, c2, c3, c4 = st.columns(4)
        
        c1.markdown(f"""<div class='kpi-card border-qia'><div class='kpi-title'>QIA (Base Ativa)</div><div class='kpi-value'>{fmt(q_fim)}</div><div class='kpi-sub'>Var: <span style='color:{'green' if q_var>0 else 'red'}'>{fmt(q_var)}</span></div></div>""", unsafe_allow_html=True)
        c2.markdown(f"""<div class='kpi-card border-vendas'><div class='kpi-title'>Vendas</div><div class='kpi-value'>{fmt(v_tot)}</div><div class='kpi-sub'>Média: {fmt(v_med)}/mês</div></div>""", unsafe_allow_html=True)
        c3.markdown(f"""<div class='kpi-card border-adimp'><div class='kpi-title'>Adimplência</div><div class='kpi-value'>{fmt_dec(adm_med)}%</div><div class='kpi-sub'>Média do Período</div></div>""", unsafe_allow_html=True)
        c4.markdown(f"""<div class='kpi-card border-desf'><div class='kpi-title'>Desfiliações</div><div class='kpi-value'>{fmt(d_tot)}</div><div class='kpi-sub'>Méd: {fmt(d_med)} | Churn Médio: {fmt_dec(churn_medio_periodo)}%</div></div>""", unsafe_allow_html=True)

        st.markdown("---")

        # --- BLOCO 2: GRÁFICOS ---
        modo_comparacao = len(franquias_sel) > 1
        
        # LAYOUT CLEAN
        layout_clean = dict(
            separators=",.", 
            template="plotly_white", 
            hovermode="x unified",
            yaxis=dict(showgrid=False, showticklabels=False, visible=False),
            xaxis=dict(showgrid=False, tickfont=dict(color="#64748b")), 
            margin=dict(t=40, b=40, l=10, r=10),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )

        # 1. GRÁFICO PRINCIPAL
        if not modo_comparacao:
            # === MODO 1: VISÃO ÚNICA (BARRAS LADO A LADO + SALDO) ===
            st.markdown("##### 📉 Fluxo de Contratos (Vendas vs Saídas)")
            fig1 = go.Figure()
            
            for f in franquias_sel:
                d = df_viz[df_viz['franquia'] == f].copy()
                d['saldo'] = d['vendas'] - d['desfiliacao']
                
                # Vendas (Amarelo)
                fig1.add_trace(go.Bar(
                    x=d['mes_exibicao'], y=d['vendas'], name=f"{f} - Vendas", 
                    marker_color=COR_VENDAS_PADRAO, opacity=0.9, 
                    text=d['vendas'].apply(fmt), textposition='outside',
                    textfont=dict(size=10, color=COR_VENDAS_PADRAO), # Cor do texto = Cor da barra
                    hovertemplate='Vendas: %{y:,.0f}<extra></extra>'
                ))
                
                # Desfiliação (Vermelho)
                fig1.add_trace(go.Bar(
                    x=d['mes_exibicao'], y=d['desfiliacao'], name=f"{f} - Desfiliação", 
                    marker_color=COR_DESFILIACAO_PADRAO, opacity=0.9,
                    text=d['desfiliacao'].apply(fmt), textposition='outside',
                    textfont=dict(size=10, color=COR_DESFILIACAO_PADRAO), # Cor do texto = Cor da barra
                    hovertemplate='Saídas: %{y:,.0f}<extra></extra>'
                ))

                # Saldo (Linha Verde Suave com Transparência)
                fig1.add_trace(go.Scatter(
                    x=d['mes_exibicao'], y=d['saldo'], name=f"{f} - Saldo Líquido", 
                    # Use a cor transparente para a linha
                    line=dict(color=COR_SALDO_TRANSPARENTE, width=3, shape='spline'), 
                    text=d['saldo'].apply(lambda x: f"{x:+,.0f}"), 
                    mode='lines+markers+text',
                    textposition="top center",
                    # Mantenha o texto e os marcadores sólidos para leitura
                    textfont=dict(size=11, color=COR_SALDO_PADRAO, weight="bold"), 
                    marker=dict(size=6, color=COR_SALDO_PADRAO), 
                    hovertemplate='Saldo: %{y:+,.0f}<extra></extra>'
                ))
            
            fig1.update_layout(
                height=450, 
                **layout_clean, 
                barmode='group', # Lado a lado
                legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5, font=dict(size=11, color="#555"))
            )

        else:
            # === MODO 2: VISÃO COMPARATIVA (CORES + HACHURA + TEXTO DISCRETO) ===
            st.markdown("##### 📈 Comparativo de Vendas vs Desfiliações")
            fig1 = go.Figure()
            
            cores_ciclo = itertools.cycle(PALETA_CORES)
            
            for f in franquias_sel:
                d = df_viz[df_viz['franquia'] == f].copy()
                cor_franquia = next(cores_ciclo)
                
                # Vendas (Cor Sólida)
                fig1.add_trace(go.Bar(
                    x=d['mes_exibicao'], y=d['vendas'], name=f"{f} - Vendas", 
                    marker_color=cor_franquia, opacity=1.0, 
                    text=d['vendas'].apply(fmt), textposition='outside', 
                    textfont=dict(size=9, color='#333'),
                    hovertemplate=f'{f}<br>Vendas: %{{y:,.0f}}<extra></extra>'
                ))
                
                # Desfiliação (Mesma Cor + HACHURA)
                fig1.add_trace(go.Bar(
                    x=d['mes_exibicao'], y=d['desfiliacao'], name=f"{f} - Desfiliação", 
                    marker=dict(color=cor_franquia, pattern_shape="/"), # Hachura
                    opacity=0.6,
                    text=d['desfiliacao'].apply(fmt), textposition='outside',
                    textfont=dict(size=9, color='#333'),
                    hovertemplate=f'{f}<br>Saídas: %{{y:,.0f}}<extra></extra>'
                ))
            
            fig1.update_layout(
                height=450, 
                **layout_clean, 
                barmode='group', # Barras lado a lado
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5, font=dict(size=11, color="#555"))
            )
            
        st.plotly_chart(fig1, use_container_width=True)

        # 2. Gráfico QIA (Full Width)
        st.markdown("##### 🌱 Evolução da Base (QIA)")
        fig2 = go.Figure()
        cores_ciclo_qia = itertools.cycle(PALETA_CORES if modo_comparacao else [COR_QIA])
        
        for f in franquias_sel:
            d = df_viz[df_viz['franquia'] == f].copy()
            c = next(cores_ciclo_qia)
            
            fig2.add_trace(go.Scatter(
                x=d['mes_exibicao'], y=d['qia'], name=f, 
                line=dict(color=c, width=3, shape='spline'),
                text=d['qia'].apply(fmt),
                mode='lines+markers+text' if not modo_comparacao else 'lines',
                textposition="top center",
                textfont=dict(size=10, color='#444'),
                marker=dict(size=6),
                cliponaxis=False,
                hovertemplate='QIA: %{y:,.0f}<extra></extra>' 
            ))
        fig2.update_layout(height=400, **layout_clean)
        st.plotly_chart(fig2, use_container_width=True)
            
        # 3. Gráfico Adimplência (Full Width)
        st.markdown("##### 💰 Qualidade Financeira (Adimplência %)")
        fig3 = go.Figure()
        cores_ciclo_adimp = itertools.cycle(PALETA_CORES if modo_comparacao else [COR_ADIMPLENCIA])
        
        for f in franquias_sel:
            d = df_viz[df_viz['franquia'] == f]
            c = next(cores_ciclo_adimp)
            
            fig3.add_trace(go.Scatter(
                x=d['mes_exibicao'], y=d['adimplencia'], name=f, 
                line=dict(color=c, width=3, shape='spline'),
                text=d['adimplencia'].apply(lambda x: fmt_dec(x) + "%"),
                mode='lines+markers+text' if not modo_comparacao else 'lines',
                textposition="top center",
                textfont=dict(size=10, color='#444'),
                marker=dict(size=6),
                cliponaxis=False,
                hovertemplate='Adimp: %{y:,.2f}%<extra></extra>' 
            ))
        
        fig3.update_layout(height=400, **layout_clean)
        st.plotly_chart(fig3, use_container_width=True)

        # --- BLOCO 3: DESTAQUES ---
        st.markdown("---")
        
        if len(franquias_sel) > 1:
             df_comp = df_viz[df_viz['franquia'] != 'CENTROSUL'] if 'CENTROSUL' in franquias_sel else df_viz
             df_grp = df_comp.groupby('franquia').agg({'vendas':'sum', 'adimplencia':'mean', 'desfiliacao':'sum'}).reset_index()
             
             if not df_grp.empty:
                best_v = df_grp.loc[df_grp['vendas'].idxmax()]
                best_a = df_grp.loc[df_grp['adimplencia'].idxmax()]
                best_d = df_grp.loc[df_grp['desfiliacao'].idxmin()] 
                
                worst_v = df_grp.loc[df_grp['vendas'].idxmin()]
                worst_a = df_grp.loc[df_grp['adimplencia'].idxmin()]
                worst_d = df_grp.loc[df_grp['desfiliacao'].idxmax()]
                
                l_col1, l_col2 = "🏆 Melhores Unidades", "⚠️ Unidades em Atenção"
             else: st.stop()
        else:
             df_grp = df_viz.reset_index(drop=True)
             best_v = df_grp.loc[df_grp['vendas'].idxmax()]
             best_a = df_grp.loc[df_grp['adimplencia'].idxmax()]
             best_d = df_grp.loc[df_grp['desfiliacao'].idxmin()]
             
             worst_v = df_grp.loc[df_grp['vendas'].idxmin()]
             worst_a = df_grp.loc[df_grp['adimplencia'].idxmin()]
             worst_d = df_grp.loc[df_grp['desfiliacao'].idxmax()]
             l_col1, l_col2 = "🏆 Melhores Momentos", "⚠️ Pontos de Atenção"

        col_l, col_r = st.columns(2)
        with col_l:
            st.subheader(l_col1)
            txt_v = f"{best_v['franquia']}: {fmt(best_v['vendas'])}" if len(franquias_sel)>1 else f"Maior Venda: {fmt(best_v['vendas'])}"
            sub_v = "Total" if len(franquias_sel)>1 else best_v['mes_exibicao']
            st.markdown(f"""<div class="destaque-card"><span class="card-label">{txt_v}</span><span class="card-value">{sub_v}</span></div>""", unsafe_allow_html=True)
            
            txt_a = f"{best_a['franquia']}: {fmt_dec(best_a['adimplencia'])}%" if len(franquias_sel)>1 else f"Maior Adimplência: {fmt_dec(best_a['adimplencia'])}%"
            sub_a = "Média" if len(franquias_sel)>1 else best_a['mes_exibicao']
            st.markdown(f"""<div class="destaque-card"><span class="card-label">{txt_a}</span><span class="card-value">{sub_a}</span></div>""", unsafe_allow_html=True)

            txt_d = f"{best_d['franquia']}: {fmt(best_d['desfiliacao'])}" if len(franquias_sel)>1 else f"Menor Desfiliação: {fmt(best_d['desfiliacao'])}"
            sub_d = "Total" if len(franquias_sel)>1 else best_d['mes_exibicao']
            st.markdown(f"""<div class="destaque-card"><span class="card-label">{txt_d}</span><span class="card-value">{sub_d}</span></div>""", unsafe_allow_html=True)

        with col_r:
            st.subheader(l_col2)
            txt_vb = f"{worst_v['franquia']}: {fmt(worst_v['vendas'])}" if len(franquias_sel)>1 else f"Pior Venda: {fmt(worst_v['vendas'])}"
            sub_vb = "Total" if len(franquias_sel)>1 else worst_v['mes_exibicao']
            st.markdown(f"""<div class="atencao-card"><span class="card-label">{txt_vb}</span><span class="card-value">{sub_vb}</span></div>""", unsafe_allow_html=True)
            
            txt_ab = f"{worst_a['franquia']}: {fmt_dec(worst_a['adimplencia'])}%" if len(franquias_sel)>1 else f"Pior Adimplência: {fmt_dec(worst_a['adimplencia'])}%"
            sub_ab = "Média" if len(franquias_sel)>1 else worst_a['mes_exibicao']
            st.markdown(f"""<div class="atencao-card"><span class="card-label">{txt_ab}</span><span class="card-value">{sub_ab}</span></div>""", unsafe_allow_html=True)

            txt_db = f"{worst_d['franquia']}: {fmt(worst_d['desfiliacao'])}" if len(franquias_sel)>1 else f"Maior Desfiliação: {fmt(worst_d['desfiliacao'])}"
            sub_db = "Total" if len(franquias_sel)>1 else worst_d['mes_exibicao']
            st.markdown(f"""<div class="atencao-card"><span class="card-label">{txt_db}</span><span class="card-value">{sub_db}</span></div>""", unsafe_allow_html=True)

        # --- DADOS DETALHADOS ---
        st.markdown("### 📋 Dados Detalhados")
        df_sorted = df_viz.sort_values(['franquia', 'data'], ascending=[True, False]).copy()
        
        df_show = df_sorted[['franquia', 'mes_exibicao', 'vendas', 'desfiliacao', 'churn_rate', 'qia', 'adimplencia']].copy()
        df_show['vendas'] = df_show['vendas'].apply(fmt)
        df_show['desfiliacao'] = df_show['desfiliacao'].apply(fmt)
        df_show['qia'] = df_show['qia'].apply(fmt)
        df_show['adimplencia'] = df_show['adimplencia'].apply(lambda x: fmt_dec(x) + "%")
        df_show['churn_rate'] = df_show['churn_rate'].apply(lambda x: fmt_dec(x) + "%")
        
        st.dataframe(
            df_show[['franquia', 'mes_exibicao', 'vendas', 'desfiliacao', 'churn_rate', 'qia', 'adimplencia']],
            use_container_width=True,
            column_config={
                "mes_exibicao": "Mês/Ano",
                "franquia": "Unidade",
                "vendas": "Vendas",
                "desfiliacao": "Desfiliações (Abs)",
                "churn_rate": "Churn Rate (%)",
                "qia": "QIA",
                "adimplencia": "Adimplência"
            },
            hide_index=True
        )

# ==============================================================================
# 4. RELATÓRIO IA
# ==============================================================================
with tab_ia:
    st.header(f"🧠 Análise Estratégica: {nome_relatorio}")
    
    def gerar_html_profundo(df_input, nome_contexto, eh_regional):
        vendas_tot = to_scalar(df_input['vendas'].sum())
        churn_tot = to_scalar(df_input['desfiliacao'].sum())
        ratio = (vendas_tot / churn_tot) if churn_tot > 0 else 0
        adimp_media = to_scalar(df_input['adimplencia'].mean())
        
        try:
            df_s = df_input.sort_values('data')
            base_ini = df_s.groupby('data')['qia'].sum().iloc[0]
            base_fim = df_s.groupby('data')['qia'].sum().iloc[-1]
            crescimento_abs = base_fim - base_ini
        except:
            crescimento_abs = 0
            base_fim = 0
        
        html_content = f"""
        <!DOCTYPE html>
        <html lang="pt-BR">
        <head>
            <meta charset="UTF-8">
            <style>
                body {{ font-family: 'Segoe UI', sans-serif; padding: 30px; background: #fff; color: #333; }}
                h1 {{ color: #2c3e50; border-bottom: 2px solid #eee; padding-bottom: 15px; }}
                .summary-box {{ background: #f9f9f9; padding: 20px; border-radius: 8px; margin: 20px 0; display: flex; justify-content: space-between; }}
                .kpi-box {{ text-align: center; }}
                .kpi-val {{ display: block; font-size: 24px; font-weight: bold; color: #2c3e50; }}
                .kpi-lbl {{ font-size: 12px; text-transform: uppercase; color: #777; }}
                .insight-card {{ border-left: 4px solid #ccc; padding: 15px; margin: 10px 0; background: #fff; }}
                .danger {{ border-color: #c0392b; background: #fff5f5; }}
                .success {{ border-color: #27ae60; background: #f0f9f0; }}
            </style>
        </head>
        <body>
            <h1>Relatório: {nome_contexto}</h1>
            <div class="summary-box">
                <div class="kpi-box"><span class="kpi-val">{fmt(base_fim)}</span><span class="kpi-lbl">QIA Atual</span></div>
                <div class="kpi-box"><span class="kpi-val" style="color:{'green' if crescimento_abs>0 else 'red'}">{fmt(crescimento_abs)}</span><span class="kpi-lbl">Evolução</span></div>
                <div class="kpi-box"><span class="kpi-val">{fmt(vendas_tot)}</span><span class="kpi-lbl">Vendas</span></div>
                <div class="kpi-box"><span class="kpi-val">{fmt_dec(adimp_media)}%</span><span class="kpi-lbl">Adimplência</span></div>
            </div>
            
            <div class="insight-card {'success' if ratio > 1.5 else 'danger'}">
                 <strong>Diagnóstico:</strong> A relação Vendas/Churn é de {fmt_dec(ratio)}. 
                 (A cada 1 cancelamento, entram {fmt_dec(ratio)} novos contratos).
            </div>
        </body>
        </html>
        """
        return html_content

    html_final = gerar_html_profundo(df_viz, nome_relatorio, len(franquias_sel)>1)
    
    c_btn, c_prev = st.columns([1, 4])
    with c_btn:
        st.download_button("📥 Baixar Relatório", data=html_final, file_name="Relatorio.html", mime="text/html")
    with c_prev:
        components.html(html_final, height=500, scrolling=True)
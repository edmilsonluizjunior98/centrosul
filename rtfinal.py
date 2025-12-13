import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import unicodedata
import itertools
import requests
import io
import streamlit.components.v1 as components
from datetime import datetime

# ==============================================================================
# CONFIGURAÇÕES E ESTILO
# ==============================================================================
st.set_page_config(page_title="Gestão Cartão de Todos", layout="wide", page_icon="💼")

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
    .record-box {{ background-color: #f8f9fa; border-left: 5px solid #27AE60; padding: 15px; border-radius: 5px; margin-bottom: 10px; }}
    .alert-box {{ background-color: #fff5f5; border-left: 5px solid {COR_ALERTA}; padding: 15px; border-radius: 5px; margin-bottom: 10px; }}
    .border-qia {{ border-top: 4px solid {COR_QIA}; }}
    .border-vendas {{ border-top: 4px solid {COR_VENDAS}; }}
    .border-adimp {{ border-top: 4px solid {COR_ADIMPLENCIA}; }}
    .border-desf {{ border-top: 4px solid {COR_DESFILIACAO}; }}
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

def formatar_mes(data):
    meses = {1: 'Jan', 2: 'Fev', 3: 'Mar', 4: 'Abr', 5: 'Mai', 6: 'Jun', 
             7: 'Jul', 8: 'Ago', 9: 'Set', 10: 'Out', 11: 'Nov', 12: 'Dez'}
    return f"{meses[data.month]}/{data.year}"

# ==============================================================================
# 1. CARREGAMENTO DE DADOS
# ==============================================================================
@st.cache_data(ttl=600)
def carregar_dados():
    url = 'https://docs.google.com/spreadsheets/d/1qnJ6yOB_iYZbKM-qYkqtS5Q7NvlLOONw35sacfMMBPo/export?format=csv&gid=0'
    try:
        response = requests.get(url, verify=False)
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
        
        # Cria coluna auxiliar Mês/Ano para exibição
        df['mes_exibicao'] = df['data'].apply(formatar_mes)
        
        return df
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return pd.DataFrame()

df_completo = carregar_dados()
if df_completo.empty: st.stop()

# ==============================================================================
# 2. FILTROS
# ==============================================================================
st.sidebar.header("🔍 Configuração")
datas = sorted(df_completo['data'].unique())
dt_ini, dt_fim = st.sidebar.select_slider("Período:", options=datas, value=(datas[0], datas[-1]), format_func=formatar_mes)

# Filtro de Franquias Ativas
df_periodo = df_completo[(df_completo['data'] >= dt_ini) & (df_completo['data'] <= dt_fim)]
vendas_por_franquia = df_periodo.groupby('franquia')['vendas'].sum()
franquias_ativas = vendas_por_franquia[vendas_por_franquia >= 100].index.tolist()

# Garante que CENTROSUL apareça se existir no CSV, mesmo que a venda seja diferente
if "CENTROSUL" in df_completo['franquia'].unique():
    if "CENTROSUL" not in franquias_ativas:
        franquias_ativas.insert(0, "CENTROSUL")

df_final = df_periodo[df_periodo['franquia'].isin(franquias_ativas)].copy()
todas_opcoes = sorted(franquias_ativas)

# Seletor
franquias_sel = st.sidebar.multiselect("Selecione Unidade(s) ou Regional:", todas_opcoes, default=[todas_opcoes[0]])

if not franquias_sel:
    st.warning("Selecione ao menos uma franquia.")
    st.stop()

# Filtra o DF Final para o que foi selecionado
df_viz = df_final[df_final['franquia'].isin(franquias_sel)]

# Lógica de Contexto (Regional vs Unidade)
is_regional = "CENTROSUL" in franquias_sel or len(franquias_sel) > 1
nome_relatorio = "CENTROSUL / REGIONAL" if is_regional else franquias_sel[0]

# ==============================================================================
# 3. DASHBOARD (Aba Única de Análise)
# ==============================================================================
tab_dash, tab_ia = st.tabs(["📊 Visão Tática (Dashboard)", "🧠 Relatório Estratégico (IA)"])

with tab_dash:
    if df_viz.empty:
        st.info("Sem dados para exibir.")
    else:
        # --- BLOCO 1: KPIs ---
        # Se for regional, removemos a linha "CENTROSUL" da soma para não duplicar, 
        # a menos que só ela esteja selecionada.
        if "CENTROSUL" in franquias_sel and len(franquias_sel) > 1:
            df_kpi = df_viz[df_viz['franquia'] != "CENTROSUL"]
        else:
            df_kpi = df_viz

        agg = df_kpi.groupby('data')[['vendas', 'qia', 'adimplencia', 'desfiliacao']].sum().reset_index()
        agg['adimplencia'] = df_kpi.groupby('data')['adimplencia'].mean().values # Média simples
        
        q_fim, q_ini = to_scalar(agg.iloc[-1]['qia']), to_scalar(agg.iloc[0]['qia'])
        v_tot, d_tot = to_scalar(agg['vendas'].sum()), to_scalar(agg['desfiliacao'].sum())
        adm_med = to_scalar(agg['adimplencia'].mean())

        st.markdown(f"### Visão Consolidada: {nome_relatorio}")
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f"<div class='kpi-card border-qia'><div class='kpi-title'>QIA (Saldo)</div><div class='kpi-value'>{q_fim:,.0f}</div><div style='color:#777;font-size:12px'>Var: {q_fim-q_ini:+,.0f}</div></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='kpi-card border-vendas'><div class='kpi-title'>Vendas</div><div class='kpi-value'>{v_tot:,.0f}</div><div style='color:#777;font-size:12px'>Total Período</div></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='kpi-card border-adimp'><div class='kpi-title'>Adimplência</div><div class='kpi-value'>{adm_med:.1f}%</div><div style='color:#777;font-size:12px'>Média</div></div>", unsafe_allow_html=True)
        c4.markdown(f"<div class='kpi-card border-desf'><div class='kpi-title'>Desfiliações</div><div class='kpi-value'>{d_tot:,.0f}</div><div style='color:#777;font-size:12px'>Total Período</div></div>", unsafe_allow_html=True)

        st.markdown("---")

        # --- BLOCO 2: GRÁFICOS ---
        mapa_cores = {f: c for f, c in zip(franquias_sel, itertools.cycle(px.colors.qualitative.Bold))}
        modo_comparacao = len(franquias_sel) > 1

        fig1 = go.Figure()
        for f in franquias_sel:
            d = df_viz[df_viz['franquia'] == f]
            c_v = mapa_cores[f] if modo_comparacao else COR_VENDAS
            fig1.add_trace(go.Scatter(x=d['data'], y=d['vendas'], name=f, line=dict(color=c_v, width=3)))
        
        fig1.update_layout(title="Volume de Vendas", template="plotly_white", hovermode="x unified", height=350)
        st.plotly_chart(fig1, use_container_width=True)

        col_g2, col_g3 = st.columns(2)
        with col_g2:
            fig2 = go.Figure()
            for f in franquias_sel:
                d = df_viz[df_viz['franquia'] == f]
                c = mapa_cores[f] if modo_comparacao else COR_QIA
                fig2.add_trace(go.Scatter(x=d['data'], y=d['qia'], name=f, line=dict(color=c, width=3)))
            fig2.update_layout(title="Evolução da Base (QIA)", template="plotly_white", height=300)
            st.plotly_chart(fig2, use_container_width=True)
            
        with col_g3:
            fig3 = go.Figure()
            for f in franquias_sel:
                d = df_viz[df_viz['franquia'] == f]
                c = mapa_cores[f] if modo_comparacao else COR_ADIMPLENCIA
                fig3.add_trace(go.Scatter(x=d['data'], y=d['adimplencia'], name=f, line=dict(color=c, width=3)))
            fig3.update_layout(title="Qualidade (Adimplência %)", template="plotly_white", height=300)
            st.plotly_chart(fig3, use_container_width=True)

        # --- BLOCO 3: DESTAQUES E RESUMO (Antiga Aba 2) ---
        st.markdown("### 📌 Destaques do Período")
        
        # Agrupamento para destaques
        df_resumo = df_viz.groupby('franquia').agg({'vendas':'sum', 'desfiliacao':'sum', 'adimplencia':'mean', 'qia':'last'}).reset_index()
        
        if is_regional:
            # Destaques Comparativos
            melhor_venda = df_resumo.loc[df_resumo['vendas'].idxmax()]
            pior_churn = df_resumo.loc[df_resumo['desfiliacao'].idxmax()]
            
            c_dest1, c_dest2 = st.columns(2)
            c_dest1.markdown(f"""
            <div class="record-box">
                <b>🏆 Unidade Líder em Vendas:</b> {melhor_venda['franquia']}<br>
                <span style="font-size:20px">{melhor_venda['vendas']:,.0f} contratos</span>
            </div>
            """, unsafe_allow_html=True)
            c_dest2.markdown(f"""
            <div class="alert-box">
                <b>⚠️ Maior Volume de Saída:</b> {pior_churn['franquia']}<br>
                <span style="font-size:20px">{pior_churn['desfiliacao']:,.0f} contratos</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Destaques Temporais (Unidade Única)
            idx_max_v = df_viz['vendas'].idxmax()
            idx_min_a = df_viz['adimplencia'].idxmin()
            
            c_dest1, c_dest2 = st.columns(2)
            c_dest1.markdown(f"""
            <div class="record-box">
                <b>📅 Melhor Mês de Vendas:</b> {df_viz.loc[idx_max_v, 'mes_exibicao']}<br>
                <span style="font-size:20px">{df_viz.loc[idx_max_v, 'vendas']:,.0f} vendas</span>
            </div>
            """, unsafe_allow_html=True)
            c_dest2.markdown(f"""
            <div class="alert-box">
                <b>📉 Pior Adimplência:</b> {df_viz.loc[idx_min_a, 'mes_exibicao']}<br>
                <span style="font-size:20px">{df_viz.loc[idx_min_a, 'adimplencia']:.2f}%</span>
            </div>
            """, unsafe_allow_html=True)

        # --- BLOCO 4: DADOS DETALHADOS ---
        st.markdown("### 📋 Dados Detalhados")
        # Seleciona colunas e renomeia
        cols_show = ['franquia', 'mes_exibicao', 'vendas', 'desfiliacao', 'qia', 'adimplencia']
        df_show = df_viz[cols_show].sort_values(['franquia', 'data'], ascending=[True, False]).copy()
        
        st.dataframe(
            df_show.style.format({'vendas':'{:.0f}', 'desfiliacao':'{:.0f}', 'qia':'{:.0f}', 'adimplencia':'{:.2f}%'}),
            use_container_width=True,
            column_config={
                "mes_exibicao": "Mês/Ano",
                "franquia": "Unidade",
                "vendas": "Vendas",
                "qia": "Base Ativa",
                "adimplencia": "Adimplência"
            },
            hide_index=True
        )

# ==============================================================================
# 4. RELATÓRIO IA (Profundo e Contextual)
# ==============================================================================
with tab_ia:
    st.header(f"🧠 Análise Estratégica: {nome_relatorio}")
    
    # Função para gerar insights profundos
    def gerar_html_profundo(df_input, nome_contexto, eh_regional):
        
        # Cálculos Base
        vendas_tot = to_scalar(df_input['vendas'].sum())
        churn_tot = to_scalar(df_input['desfiliacao'].sum())
        ratio = (vendas_tot / churn_tot) if churn_tot > 0 else 0
        adimp_media = to_scalar(df_input['adimplencia'].mean())
        base_ini = to_scalar(df_input.sort_values('data')['qia'].iloc[0])
        base_fim = to_scalar(df_input.sort_values('data')['qia'].iloc[-1])
        crescimento_abs = base_fim - base_ini
        
        # --- HTML Generator ---
        html_content = f"""
        <!DOCTYPE html>
        <html lang="pt-BR">
        <head>
            <meta charset="UTF-8">
            <style>
                body {{ font-family: 'Segoe UI', sans-serif; padding: 30px; background: #fff; color: #333; }}
                h1 {{ color: #2c3e50; border-bottom: 2px solid #eee; padding-bottom: 15px; }}
                h2 {{ color: #2980b9; margin-top: 30px; font-size: 18px; }}
                .summary-box {{ background: #f9f9f9; padding: 20px; border-radius: 8px; margin: 20px 0; display: flex; justify-content: space-between; }}
                .kpi-box {{ text-align: center; }}
                .kpi-val {{ display: block; font-size: 24px; font-weight: bold; color: #2c3e50; }}
                .kpi-lbl {{ font-size: 12px; text-transform: uppercase; color: #777; }}
                .insight-card {{ border-left: 4px solid #ccc; padding: 15px; margin: 10px 0; background: #fff; }}
                .danger {{ border-color: #c0392b; background: #fff5f5; }}
                .success {{ border-color: #27ae60; background: #f0f9f0; }}
                .neutral {{ border-color: #f1c40f; background: #fffae6; }}
            </style>
        </head>
        <body>
            <h1>Relatório de Gestão: {nome_contexto}</h1>
            <p>Período de Análise: {dt_ini.strftime('%m/%Y')} a {dt_fim.strftime('%m/%Y')}</p>
            
            <div class="summary-box">
                <div class="kpi-box"><span class="kpi-val">{base_fim:,.0f}</span><span class="kpi-lbl">Base Atual</span></div>
                <div class="kpi-box"><span class="kpi-val" style="color:{'green' if crescimento_abs>0 else 'red'}">{crescimento_abs:+,.0f}</span><span class="kpi-lbl">Evolução Líquida</span></div>
                <div class="kpi-box"><span class="kpi-val">{vendas_tot:,.0f}</span><span class="kpi-lbl">Vendas Totais</span></div>
                <div class="kpi-box"><span class="kpi-val">{adimp_media:.1f}%</span><span class="kpi-lbl">Adimplência Média</span></div>
            </div>
        """
        
        # --- Lógica Diferenciada: Regional vs Unidade ---
        
        if eh_regional:
            # ANÁLISE COMPARATIVA (CENTROSUL)
            # Remove a linha CENTROSUL para calcular ranking real
            df_rank = df_input[df_input['franquia'] != 'CENTROSUL'].groupby('franquia').agg(
                {'vendas':'sum', 'desfiliacao':'sum', 'qia': lambda x: x.iloc[-1] - x.iloc[0]}
            ).sort_values('vendas', ascending=False)
            
            top_vendas = df_rank.index[0]
            top_vendas_val = df_rank.iloc[0]['vendas']
            
            pior_cresc_nome = df_rank['qia'].idxmin()
            pior_cresc_val = df_rank.loc[pior_cresc_nome, 'qia']
            
            html_content += f"""
            <h2>1. Análise de Competitividade (Regional)</h2>
            <div class="insight-card success">
                <strong>🏆 A Locomotiva:</strong> A unidade <strong>{top_vendas}</strong> foi responsável por liderar as vendas com {top_vendas_val:,.0f} contratos.
                Isso representa um benchmarking interno que deve ser estudado.
            </div>
            <div class="insight-card danger">
                <strong>🚨 Ponto Crítico de Retenção:</strong> A unidade <strong>{pior_cresc_nome}</strong> teve o pior saldo líquido ({pior_cresc_val:+,.0f} vidas).
                Apesar do esforço de vendas, o balde está furado nesta unidade específica.
            </div>
            <h3>Ranking de Vendas (Top 3):</h3>
            <ul>
                {''.join([f'<li><strong>{idx}:</strong> {row.vendas:,.0f} vendas</li>' for idx, row in df_rank.head(3).iterrows()])}
            </ul>
            """
            
        else:
            # ANÁLISE PROFUNDA (UNIDADE)
            # Tendência
            df_input = df_input.sort_values('data')
            tendencia_vendas = "crescente" if df_input['vendas'].iloc[-1] > df_input['vendas'].iloc[0] else "decrescente"
            pico_churn = df_input['desfiliacao'].max()
            mes_pico_churn = df_input.loc[df_input['desfiliacao'].idxmax(), 'mes_exibicao']
            
            analise_saude = ""
            if ratio < 1.2:
                analise_saude = "A saúde da base está <strong>CRÍTICA</strong>. O esforço comercial está apenas repondo o churn, sem gerar crescimento real."
                classe_saude = "danger"
            elif ratio < 2:
                analise_saude = "A operação exige <strong>ATENÇÃO</strong>. Para crescer, é necessário vender o dobro do que se perde."
                classe_saude = "neutral"
            else:
                analise_saude = "A operação está <strong>SAUDÁVEL</strong>, gerando valor real acima da perda natural."
                classe_saude = "success"

            html_content += f"""
            <h2>1. Diagnóstico de Saúde da Franquia</h2>
            <div class="insight-card {classe_saude}">
                {analise_saude} (Ratio de Eficiência: {ratio:.1f}x)
            </div>
            
            <h2>2. Tendências e Comportamento</h2>
            <ul>
                <li><strong>Ritmo Comercial:</strong> As vendas apresentam uma tendência <strong>{tendencia_vendas}</strong> comparando o início e fim do período.</li>
                <li><strong>Alerta de Saída:</strong> O momento de maior estresse na base foi em <strong>{mes_pico_churn}</strong>, com {pico_churn:,.0f} cancelamentos. Vale investigar ações ou problemas ocorridos neste mês.</li>
                <li><strong>Qualidade Financeira:</strong> A adimplência média de {adimp_media:.1f}% impacta diretamente o fluxo de caixa recorrente.</li>
            </ul>
            """

        html_content += """
        <div style="margin-top:30px; font-size:12px; color:#999; text-align:center;">
            Relatório gerado automaticamente por Inteligência de Dados.
        </div>
        </body>
        </html>
        """
        return html_content

    # Gera HTML
    html_final = gerar_html_profundo(df_viz, nome_relatorio, is_regional)
    
    # Exibe
    c_btn, c_prev = st.columns([1, 4])
    with c_btn:
        st.download_button(
            label="📥 Baixar PDF/HTML",
            data=html_final,
            file_name=f"Relatorio_{nome_relatorio.replace(' ','_')}.html",
            mime="text/html"
        )
    
    with c_prev:
        components.html(html_final, height=600, scrolling=True)
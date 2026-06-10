# -*- coding: utf-8 -*-
"""
Gestao de Riscos Geopoliticos em Ativos de Commodities - B3
Versao metodologicamente corrigida

Disciplina: Inteligencia de Mercado e Analytics
Autora: Gabriela Alvim De Nigris
Data: Junho de 2026

CORRECOES IMPLEMENTADAS EM RELACAO A VERSAO ORIGINAL
-----------------------------------------------------
1. Agregacao da carteira com RETORNOS SIMPLES (log-retornos nao sao
   aditivos no corte transversal). Pesos fixos = rebalanceamento diario
   implicito, agora declarado explicitamente.
2. VaR parametrico com media zero (pratica padrao em horizonte diario),
   alem de variantes t-Student e Cornish-Fisher para caudas pesadas,
   coerentes com a curtose documentada (teste de Jarque-Bera incluido).
3. VaR condicional via EWMA (lambda = 0.94, RiskMetrics), capturando o
   clustering de volatilidade que o VaR incondicional ignora.
4. Intervalos de confianca via bootstrap para o VaR historico em
   janelas curtas (o percentil de 5% com ~65 obs. usa 3-4 pontos da
   cauda; sem IC o "multiplicador de risco" nao e interpretavel).
5. Correlacoes sob estresse com ajuste de Forbes-Rigobon (2002) para o
   vies de heterocedasticidade, e teste de Fisher (z) para a diferenca
   de correlacoes entre periodos.
6. Backtesting formal do VaR: testes de Kupiec (POF) e de
   Christoffersen (independencia) sobre previsoes out-of-sample.
7. Sharpe com taxa livre de risco (CDI medio do periodo, parametro
   configuravel), em vez de rf = 0.
8. Canal cambial testado empiricamente (regressao dos retornos da
   carteira contra a variacao do USD/BRL), em vez de afirmado.
9. Analise de robustez das janelas de volatilidade movel (21/42/63d).
10. Linguagem das conclusoes: evidencia DESCRITIVA consistente com a
    hipotese, nao afirmacao causal. As janelas de evento sao escolhidas
    ex post; sem estrategia de identificacao (event study com retornos
    anormais ou indice continuo de risco geopolitico, ex.: GPR de
    Caldara e Iacoviello), o desenho nao permite inferencia causal.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import norm, t as t_dist, jarque_bera, chi2
import yfinance as yf
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (12, 5)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['font.size'] = 11

# =====================================================================
# 0. PARAMETROS GERAIS
# =====================================================================
ATIVOS = ['VALE3.SA', 'PETR4.SA', 'PRIO3.SA', 'GGBR4.SA']
NOMES = {'VALE3.SA': 'Vale', 'PETR4.SA': 'Petrobras',
         'PRIO3.SA': 'PetroRio', 'GGBR4.SA': 'Gerdau'}
DATA_INI = '2022-01-01'
DATA_FIM = '2026-06-04'

PESOS = np.array([0.35, 0.30, 0.20, 0.15])   # Vale, Petrobras, PetroRio, Gerdau
VALOR_PORTFOLIO = 1_000_000                   # R$
CONFIANCA = 0.95
ALPHA = 1 - CONFIANCA

# Taxa livre de risco: CDI medio anualizado do periodo (ajustar se
# necessario; idealmente baixar a serie 12 do SGS/BCB e compor).
RF_ANUAL = 0.1115

# Janelas de analise. ATENCAO: escolhidas ex post (limitacao declarada).
JANELA_NORMAL = ('2024-07-01', '2024-12-31')
JANELA_UCRANIA = ('2022-02-24', '2022-04-30')
JANELA_IRA = ('2026-02-28', None)             # None = ate o fim da amostra

EVENTOS = {
    'Invasao\nUcrania': '2022-02-24',
    'Conflito\nIsrael-Hamas': '2023-10-07',
    'Tarifas\nTrump': '2025-04-02',
    'Ataque\nEUA-Israel\nao Ira': '2026-02-28',
}

CORES = {'Vale': '#003087', 'Petrobras': '#009c3b',
         'PetroRio': '#e8a020', 'Gerdau': '#c0392b'}

# =====================================================================
# 1. COLETA DE DADOS
# =====================================================================
print('Baixando precos dos ativos...')
df_precos = yf.download(ATIVOS, start=DATA_INI, end=DATA_FIM,
                        auto_adjust=True, progress=False)['Close']
df_precos.columns.name = None
df_precos.index.name = 'Data'
df_precos = df_precos.rename(columns=NOMES)[list(CORES.keys())]
df_precos = df_precos.dropna(how='all').ffill()

print('Baixando USD/BRL para o teste do canal cambial...')
usdbrl = yf.download('BRL=X', start=DATA_INI, end=DATA_FIM,
                     auto_adjust=True, progress=False)['Close']
if isinstance(usdbrl, pd.DataFrame):
    usdbrl = usdbrl.iloc[:, 0]
usdbrl.name = 'USDBRL'

print(f'Periodo: {df_precos.index[0].date()} a {df_precos.index[-1].date()}')
print(f'Observacoes: {len(df_precos)} dias uteis')
print(f'Dados ausentes por ativo:\n{df_precos.isnull().sum()}')

# =====================================================================
# 2. RETORNOS
# ---------------------------------------------------------------------
# Retornos SIMPLES para agregacao da carteira (aditivos no corte
# transversal). Log-retornos mantidos apenas para estatisticas
# descritivas por ativo (aditivos no tempo).
# =====================================================================
ret_simples = df_precos.pct_change().dropna()
ret_log = np.log(df_precos / df_precos.shift(1)).dropna()

print('\n=== ESTATISTICAS DESCRITIVAS DOS RETORNOS DIARIOS (log) ===\n')
stats = ret_log.describe().T
stats['skewness'] = ret_log.skew()
stats['kurtosis'] = ret_log.kurtosis()   # excesso de curtose
print(stats[['mean', 'std', 'min', 'max', 'skewness', 'kurtosis']].round(5))

print('\n=== TESTE DE NORMALIDADE (Jarque-Bera) ===')
for ativo in ret_log.columns:
    jb, pval = jarque_bera(ret_log[ativo])
    veredito = 'REJEITA normalidade' if pval < 0.05 else 'nao rejeita'
    print(f'  {ativo:<10} JB = {jb:>10.1f}  p-valor = {pval:.4f}  -> {veredito}')
print('  Implicacao: o VaR parametrico Normal e inadequado isoladamente;')
print('  reportamos tambem t-Student e Cornish-Fisher.')

# =====================================================================
# 3. CARTEIRA
# ---------------------------------------------------------------------
# r_p(t) = soma_i w_i * r_i(t), com r_i SIMPLES.
# Pesos fixos a cada dia equivalem a uma carteira rebalanceada
# diariamente (hipotese declarada; custos de transacao ignorados).
# =====================================================================
assert abs(PESOS.sum() - 1) < 1e-9, 'Pesos devem somar 1'
retorno_carteira = ret_simples.dot(PESOS)
retorno_carteira.name = 'Carteira'

rf_diaria = (1 + RF_ANUAL) ** (1 / 252) - 1
excesso = retorno_carteira - rf_diaria
sharpe = (excesso.mean() * 252) / (retorno_carteira.std() * np.sqrt(252))

print('\n=== CARTEIRA (retornos simples, rebalanceamento diario) ===')
print(f'  Retorno medio anualizado:  {retorno_carteira.mean()*252*100:.2f}%')
print(f'  Volatilidade anualizada:   {retorno_carteira.std()*np.sqrt(252)*100:.2f}%')
print(f'  Sharpe (rf = CDI {RF_ANUAL*100:.2f}% a.a.): {sharpe:.2f}')
print('  Obs.: anualizacao por raiz de 252 assume retornos iid;')
print('  com clustering de volatilidade, e uma aproximacao.')

# =====================================================================
# 4. GRAFICO DE PRECOS NORMALIZADOS (descritivo)
# =====================================================================
df_norm = df_precos / df_precos.iloc[0] * 100
fig, ax = plt.subplots(figsize=(16, 6))
for ativo, cor in CORES.items():
    ax.plot(df_norm.index, df_norm[ativo], label=ativo, color=cor, linewidth=1.6)
alturas = [0.98, 0.88, 0.78, 0.98]
for (nome, data), alt in zip(EVENTOS.items(), alturas):
    dt = pd.to_datetime(data)
    if dt <= df_norm.index[-1]:
        ax.axvline(dt, color='gray', linestyle='--', alpha=0.6, linewidth=1)
        ax.text(dt, df_norm.max().max() * alt, nome, fontsize=7.5,
                ha='center', va='top', color='#444',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.75))
ax.axvspan(pd.to_datetime(JANELA_IRA[0]), df_norm.index[-1],
           color='#e74c3c', alpha=0.07, label='Conflito EUA-Ira (2026)')
ax.axhline(100, color='black', linestyle=':', linewidth=0.8, alpha=0.5)
ax.set_xlabel('Data')
ax.set_ylabel('Indice de Preco (Base 100)')
ax.set_title('Precos normalizados. Linhas verticais marcam eventos: '
             'coincidencia temporal, nao causalidade.', fontsize=10)
ax.legend(loc='upper left')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b/%y'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('fig1_precos_normalizados.png', dpi=150)
plt.show()

# =====================================================================
# 5. VOLATILIDADE: ROLLING (com robustez de janela) + EWMA
# =====================================================================
fig, ax = plt.subplots(figsize=(16, 6))
for janela, estilo in zip([21, 42, 63], ['-', '--', ':']):
    vol = retorno_carteira.rolling(janela).std() * np.sqrt(252) * 100
    ax.plot(vol.index, vol, linestyle=estilo, linewidth=1.4,
            label=f'Rolling {janela}d')

# EWMA (RiskMetrics, lambda = 0.94): captura clustering de volatilidade
LAMBDA = 0.94
var_ewma = pd.Series(index=retorno_carteira.index, dtype=float)
var_ewma.iloc[0] = retorno_carteira.iloc[:21].var()
r2 = retorno_carteira ** 2
for i in range(1, len(retorno_carteira)):
    var_ewma.iloc[i] = LAMBDA * var_ewma.iloc[i - 1] + (1 - LAMBDA) * r2.iloc[i - 1]
vol_ewma = np.sqrt(var_ewma)
ax.plot(vol_ewma.index, vol_ewma * np.sqrt(252) * 100, color='black',
        linewidth=1.2, alpha=0.8, label='EWMA (lambda = 0.94)')

ax.axvspan(pd.to_datetime(JANELA_IRA[0]), retorno_carteira.index[-1],
           color='#e74c3c', alpha=0.08)
ax.set_ylabel('Volatilidade anualizada da carteira (%)')
ax.set_title('Robustez: a leitura qualitativa dos picos nao depende da janela escolhida',
             fontsize=10)
ax.legend()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b/%y'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('fig2_volatilidade_robustez.png', dpi=150)
plt.show()

# =====================================================================
# 6. CORRELACOES SOB ESTRESSE: FORBES-RIGOBON + TESTE DE FISHER
# ---------------------------------------------------------------------
# Correlacoes amostrais sobem mecanicamente quando a volatilidade sobe,
# mesmo sem mudanca na estrutura de dependencia (vies de
# heterocedasticidade). Ajuste de Forbes-Rigobon (2002):
#   rho_adj = rho / sqrt(1 + delta * (1 - rho^2)),
#   delta = var_estresse / var_normal - 1 (variancia do ativo
#   condicionante, aqui o de maior aumento de variancia no par).
# =====================================================================
def janela(df, ini, fim):
    return df.loc[ini:fim] if fim else df.loc[ini:]

per_normal = janela(ret_simples, *JANELA_NORMAL)
per_ira = janela(ret_simples, *JANELA_IRA)

corr_n = per_normal.corr()
corr_e = per_ira.corr()
n1, n2 = len(per_normal), len(per_ira)

print('\n=== CORRELACOES: NORMAL vs ESTRESSE (EUA-IRA) ===')
print(f'  Obs. normal: {n1} | Obs. estresse: {n2}')
print(f'\n{"Par":<22}{"rho_norm":>9}{"rho_estr":>9}{"rho_FR":>8}'
      f'{"z Fisher":>10}{"p-valor":>9}')

ativos = list(ret_simples.columns)
resultados_corr = []
for i in range(len(ativos)):
    for j in range(i + 1, len(ativos)):
        a, b = ativos[i], ativos[j]
        r1, r2_ = corr_n.loc[a, b], corr_e.loc[a, b]

        # ajuste FR: condicionar no ativo com maior aumento de variancia
        razoes = [per_ira[a].var() / per_normal[a].var(),
                  per_ira[b].var() / per_normal[b].var()]
        delta = max(razoes) - 1
        r2_adj = r2_ / np.sqrt(1 + delta * (1 - r2_ ** 2))

        # teste de Fisher para diferenca de correlacoes (sem ajuste)
        z = (np.arctanh(r2_) - np.arctanh(r1)) / np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
        p = 2 * (1 - norm.cdf(abs(z)))

        resultados_corr.append((f'{a} x {b}', r1, r2_, r2_adj, z, p))
        print(f'  {a + " x " + b:<20}{r1:>9.2f}{r2_:>9.2f}{r2_adj:>8.2f}'
              f'{z:>10.2f}{p:>9.4f}')

print('\n  Leitura: se rho_FR (ajustada) volta para perto de rho_norm, o')
print('  aumento aparente decorre do vies de heterocedasticidade, nao de')
print('  contagio genuino. O teste de Fisher avalia a correlacao bruta.')

# heatmaps
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, corr, titulo in zip(
        axes, [corr_n, corr_e],
        ['Correlacao - Periodo Normal\n(Jul-Dez 2024)',
         'Correlacao - Conflito EUA-Ira\n(Fev-Jun 2026, sem ajuste FR)']):
    im = ax.imshow(corr, cmap='RdYlGn', vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)))
    ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=30)
    ax.set_yticklabels(corr.columns)
    ax.set_title(titulo, fontweight='bold', fontsize=10)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f'{corr.iloc[i, j]:.2f}', ha='center', va='center',
                    fontsize=10, fontweight='bold')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig('fig3_correlacoes.png', dpi=150)
plt.show()

# =====================================================================
# 7. FUNCOES DE VaR
# =====================================================================
def var_historico(ret, valor, alpha=ALPHA):
    """VaR historico: |percentil alpha| x valor."""
    return abs(np.percentile(ret, alpha * 100)) * valor


def var_historico_bootstrap(ret, valor, alpha=ALPHA, n_boot=5000, seed=42):
    """VaR historico com IC de 95% via bootstrap (essencial em janelas
    curtas, nas quais o percentil usa pouquissimas observacoes)."""
    rng = np.random.default_rng(seed)
    ret = np.asarray(ret)
    boots = np.empty(n_boot)
    for k in range(n_boot):
        amostra = rng.choice(ret, size=len(ret), replace=True)
        boots[k] = abs(np.percentile(amostra, alpha * 100)) * valor
    ponto = var_historico(ret, valor, alpha)
    ic = np.percentile(boots, [2.5, 97.5])
    return ponto, ic[0], ic[1]


def var_parametrico_normal(ret, valor, alpha=ALPHA, media_zero=True):
    """VaR Normal. media_zero=True e a pratica padrao em horizonte
    diario (a media diaria e ruido e pode mascarar o risco)."""
    mu = 0.0 if media_zero else np.mean(ret)
    sigma = np.std(ret, ddof=1)
    q = mu + norm.ppf(alpha) * sigma
    return max(-q, 0.0) * valor


def var_parametrico_t(ret, valor, alpha=ALPHA):
    """VaR t-Student: graus de liberdade estimados por MV; adequado a
    caudas pesadas documentadas pelo Jarque-Bera."""
    ret = np.asarray(ret)
    nu, loc, scale = t_dist.fit(ret)
    nu = max(nu, 2.1)
    q = loc + scale * t_dist.ppf(alpha, nu)
    return max(-q, 0.0) * valor


def var_cornish_fisher(ret, valor, alpha=ALPHA):
    """VaR Cornish-Fisher: expansao do quantil Normal com assimetria e
    curtose amostrais."""
    ret = pd.Series(np.asarray(ret))
    s, k = ret.skew(), ret.kurtosis()  # excesso de curtose
    z = norm.ppf(alpha)
    z_cf = (z + (z**2 - 1) * s / 6 + (z**3 - 3 * z) * k / 24
            - (2 * z**3 - 5 * z) * s**2 / 36)
    sigma = ret.std(ddof=1)
    q = z_cf * sigma
    return max(-q, 0.0) * valor


# =====================================================================
# 8. VaR POR JANELA, COM INTERVALOS DE CONFIANCA
# =====================================================================
print('\n' + '=' * 72)
print('  VaR DIARIO A 95% POR JANELA (carteira de R$ 1.000.000)')
print('  IC de 95% do VaR historico via bootstrap (5.000 reamostras)')
print('=' * 72)

ret_normal = per_normal.dot(PESOS)
ret_ucrania = janela(ret_simples, *JANELA_UCRANIA).dot(PESOS)
ret_ira = per_ira.dot(PESOS)

janelas = [
    ('Periodo Normal (Jul-Dez 2024)', ret_normal),
    ('Invasao da Ucrania (Fev-Abr 2022)', ret_ucrania),
    ('Conflito EUA-Ira (Fev-Jun 2026)', ret_ira),
    ('Periodo Completo (2022-2026)', retorno_carteira),
]

tabela = []
for label, r in janelas:
    vh, lo, hi = var_historico_bootstrap(r, VALOR_PORTFOLIO)
    vn = var_parametrico_normal(r, VALOR_PORTFOLIO)
    vt = var_parametrico_t(r, VALOR_PORTFOLIO)
    vcf = var_cornish_fisher(r, VALOR_PORTFOLIO)
    tabela.append((label, len(r), vh, lo, hi, vn, vt, vcf))
    print(f'\n  {label}  ({len(r)} obs.)')
    print(f'    Historico:      R$ {vh:>10,.0f}   IC95% [{lo:,.0f} ; {hi:,.0f}]')
    print(f'    Normal (mu=0):  R$ {vn:>10,.0f}')
    print(f'    t-Student:      R$ {vt:>10,.0f}')
    print(f'    Cornish-Fisher: R$ {vcf:>10,.0f}')
    if len(r) < 100:
        print(f'    ATENCAO: janela curta ({len(r)} obs.); o percentil de 5%')
        print(f'    usa ~{max(int(len(r)*0.05), 1)} observacoes. Interpretar com o IC.')

vh_n = tabela[0][2]
vh_u = tabela[1][2]
vh_i = tabela[2][2]
print(f'\n  Multiplicadores de risco vs. periodo normal (VaR historico):')
print(f'    Ucrania: {vh_u/vh_n:.2f}x | EUA-Ira: {vh_i/vh_n:.2f}x')
print('  Leitura honesta: janelas de estresse foram selecionadas ex post')
print('  por serem turbulentas; os multiplicadores DESCREVEM a elevacao')
print('  do risco realizado, nao testam causalidade do choque geopolitico.')

# grafico comparativo com barras de erro (IC bootstrap)
fig, ax = plt.subplots(figsize=(12, 6))
labels = [t[0].replace(' (', '\n(') for t in tabela]
vhs = [t[2] for t in tabela]
err_lo = [t[2] - t[3] for t in tabela]
err_hi = [t[4] - t[2] for t in tabela]
vts = [t[6] for t in tabela]
x = np.arange(len(labels))
w = 0.35
ax.bar(x - w/2, vhs, w, yerr=[err_lo, err_hi], capsize=5,
       label='VaR Historico (IC 95%)', color='#2c7bb6', alpha=0.85)
ax.bar(x + w/2, vts, w, label='VaR t-Student', color='#d7191c', alpha=0.85)
ax.set_ylabel('VaR Diario (R$)')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8)
ax.legend()
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'R$ {v:,.0f}'))
plt.tight_layout()
plt.savefig('fig4_var_janelas.png', dpi=150)
plt.show()

# =====================================================================
# 9. VaR CONDICIONAL (EWMA) E ROLLING VaR HISTORICO
# =====================================================================
var_ewma_serie = norm.ppf(ALPHA) * vol_ewma * -1 * VALOR_PORTFOLIO  # mu = 0
JANELA_ROLL = 252
rolling_var = retorno_carteira.rolling(JANELA_ROLL).apply(
    lambda r: abs(np.percentile(r, ALPHA * 100)) * VALOR_PORTFOLIO)

fig, ax = plt.subplots(figsize=(16, 6))
ax.plot(rolling_var.index, rolling_var / 1000, color='#2c7bb6', linewidth=1.4,
        label=f'VaR Historico rolling ({JANELA_ROLL}d)')
ax.plot(var_ewma_serie.index, var_ewma_serie / 1000, color='#c0392b',
        linewidth=1.2, alpha=0.85, label='VaR condicional EWMA (mu = 0)')
ax.axvspan(pd.to_datetime(JANELA_IRA[0]), retorno_carteira.index[-1],
           color='#e74c3c', alpha=0.10)
ax.set_ylabel('VaR Diario (R$ mil)')
ax.set_title('O VaR condicional reage ao choque em dias; o historico de '
             'janela longa reage com atraso e dilui o estresse', fontsize=10)
ax.legend()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b/%y'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('fig5_var_condicional.png', dpi=150)
plt.show()

# =====================================================================
# 10. BACKTESTING: KUPIEC (POF) E CHRISTOFFERSEN (INDEPENDENCIA)
# ---------------------------------------------------------------------
# Previsao one-step-ahead, out-of-sample: o VaR previsto para o dia t
# usa apenas informacao ate t-1. Comparamos dois modelos:
#   (a) Historico rolling de 252 dias
#   (b) EWMA parametrico Normal (mu = 0)
# =====================================================================
def teste_kupiec(violacoes, total, p=ALPHA):
    x = violacoes.sum()
    pi_hat = x / total
    if pi_hat in (0, 1):
        return np.nan, np.nan, x
    lr = -2 * ((total - x) * np.log(1 - p) + x * np.log(p)
               - (total - x) * np.log(1 - pi_hat) - x * np.log(pi_hat))
    return lr, 1 - chi2.cdf(lr, df=1), x


def teste_christoffersen(violacoes):
    v = violacoes.astype(int).values
    n00 = n01 = n10 = n11 = 0
    for k in range(1, len(v)):
        if v[k-1] == 0 and v[k] == 0: n00 += 1
        elif v[k-1] == 0 and v[k] == 1: n01 += 1
        elif v[k-1] == 1 and v[k] == 0: n10 += 1
        else: n11 += 1
    if (n01 + n11) == 0 or (n00 + n10) == 0:
        return np.nan, np.nan
    pi01 = n01 / (n00 + n01)
    pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    if pi in (0, 1) or pi01 in (0, 1):
        return np.nan, np.nan
    def safe_log(x):
        return np.log(x) if x > 0 else 0.0
    l0 = (n00 + n10) * safe_log(1 - pi) + (n01 + n11) * safe_log(pi)
    l1 = (n00 * safe_log(1 - pi01) + n01 * safe_log(pi01)
          + n10 * safe_log(1 - pi11) + (n11 * safe_log(pi11) if n11 > 0 else 0.0))
    lr = -2 * (l0 - l1)
    return lr, 1 - chi2.cdf(lr, df=1)


# (a) historico rolling: VaR de t calculado com a janela que termina em t-1
var_hist_fc = rolling_var.shift(1).dropna()
real_a = retorno_carteira.loc[var_hist_fc.index] * VALOR_PORTFOLIO
viol_a = (real_a < -var_hist_fc)

# (b) EWMA: var_ewma ja usa apenas r_{t-1} na recursao (one-step-ahead)
var_ewma_fc = var_ewma_serie.iloc[JANELA_ROLL:]   # mesmo out-of-sample
real_b = retorno_carteira.loc[var_ewma_fc.index] * VALOR_PORTFOLIO
viol_b = (real_b < -var_ewma_fc)

print('\n' + '=' * 72)
print('  BACKTESTING DO VaR 95% (out-of-sample, one-step-ahead)')
print('=' * 72)
for nome, viol in [('Historico rolling 252d', viol_a),
                   ('EWMA parametrico (mu = 0)', viol_b)]:
    total = len(viol)
    lr_k, p_k, x = teste_kupiec(viol, total)
    lr_c, p_c = teste_christoffersen(viol)
    print(f'\n  Modelo: {nome}')
    print(f'    Obs.: {total} | Violacoes: {x} ({x/total*100:.2f}%; esperado {ALPHA*100:.0f}%)')
    print(f'    Kupiec POF:        LR = {lr_k:6.2f}  p-valor = {p_k:.4f}'
          f'  -> {"REJEITA cobertura correta" if p_k < 0.05 else "nao rejeita"}')
    if not np.isnan(lr_c):
        print(f'    Christoffersen:    LR = {lr_c:6.2f}  p-valor = {p_c:.4f}'
              f'  -> {"violacoes AGRUPADAS no tempo" if p_c < 0.05 else "independencia nao rejeitada"}')
    else:
        print('    Christoffersen:    nao computavel (sem violacoes consecutivas suficientes)')
print('\n  Leitura: violacoes agrupadas em torno dos choques indicam que o')
print('  modelo nao se adapta a mudancas de regime, exatamente o ponto da')
print('  pergunta de pesquisa. Agora isso e TESTADO, nao apenas afirmado.')

# =====================================================================
# 11. CANAL CAMBIAL: TESTE EMPIRICO
# ---------------------------------------------------------------------
# A recomendacao de hedge cambial precisa de evidencia: regressao OLS
# dos retornos da carteira contra a variacao diaria do USD/BRL.
# =====================================================================
ret_fx = usdbrl.pct_change().dropna()
base = pd.concat([retorno_carteira, ret_fx], axis=1, join='inner').dropna()
base.columns = ['carteira', 'fx']

X = np.column_stack([np.ones(len(base)), base['fx'].values])
y = base['carteira'].values
beta, res_ss, _, _ = np.linalg.lstsq(X, y, rcond=None)
y_hat = X @ beta
ss_res = float(((y - y_hat) ** 2).sum())
ss_tot = float(((y - y.mean()) ** 2).sum())
r2_fx = 1 - ss_res / ss_tot
sigma2 = ss_res / (len(y) - 2)
se_beta = np.sqrt(sigma2 * np.linalg.inv(X.T @ X)[1, 1])
t_stat = beta[1] / se_beta

print('\n=== CANAL CAMBIAL (OLS: r_carteira ~ var. USD/BRL) ===')
print(f'  Beta cambial: {beta[1]:.3f}  (t = {t_stat:.2f})  |  R2 = {r2_fx:.3f}')
if abs(t_stat) > 1.96:
    direcao = 'negativa' if beta[1] < 0 else 'positiva'
    print(f'  Sensibilidade cambial estatisticamente significativa e {direcao}.')
    print('  A recomendacao de hedge cambial passa a ter base empirica.')
else:
    print('  Sem sensibilidade cambial significativa no periodo: a')
    print('  recomendacao de hedge cambial NAO se sustenta nos dados.')
print('  Obs.: erros-padrao OLS simples; para o relatorio final, usar')
print('  erros robustos (HAC/Newey-West) dada a autocorrelacao da volatilidade.')

# =====================================================================
# 12. SINTESE
# =====================================================================
print('\n' + '=' * 72)
print('  SINTESE METODOLOGICA')
print('=' * 72)
print("""
  1. Os resultados por janela sao EVIDENCIA DESCRITIVA de elevacao do
     risco realizado em periodos turbulentos. Como as janelas foram
     escolhidas ex post, o desenho nao identifica efeito causal de
     choques geopoliticos. Extensao natural: event study com retornos
     anormais ou regressao contra o indice GPR (Caldara e Iacoviello).

  2. A rejeicao da normalidade (Jarque-Bera) justifica reportar VaR
     t-Student e Cornish-Fisher ao lado do historico; o VaR Normal
     incondicional subestima a cauda.

  3. O backtesting (Kupiec e Christoffersen) substitui a afirmacao de
     "subestimacao" por um teste formal de cobertura e independencia.

  4. O ajuste de Forbes-Rigobon separa contagio genuino de aumento
     mecanico de correlacao por heterocedasticidade.

  5. O IC bootstrap explicita a incerteza do VaR historico em janelas
     curtas; multiplicadores de risco sem IC nao sao interpretaveis.

  6. O canal cambial deixou de ser conjectura e foi testado por OLS.
""")

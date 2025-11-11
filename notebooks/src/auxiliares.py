import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import shapiro
from scipy.stats import levene
from scipy.stats import ttest_ind
from scipy.stats import ttest_rel
from scipy.stats import f_oneway
from scipy.stats import wilcoxon
from scipy.stats import mannwhitneyu
from scipy.stats import friedmanchisquare
from scipy.stats import kruskal


def tab_dist_freq(df,coluna, coluna_frequencia = False):
    '''
    cria uma tabela de distribuição de frequencias para uma coluna de um df
    espera uma coluna categorica
    parameters
    ----------
    dataframe : pd.DataFrame
        DataFrame com os dados
    coluna: str
        Nome da coluna categorica
    coluna_freq : bool
        Informa se a coluna passada já é com os valores de frequencia ou não. Padrão: False

    Returns
    ----------
    pd.DataFrame
        DataFrame com a tabela de distribuição de frequencias
    '''
    
    df_estatistica = pd.DataFrame()

    if coluna_frequencia:
        df_estatistica['frequencia'] = df[coluna]
        df_estatistica['freq_relativa'] = df_estatistica['frequencia'] / df_estatistica['frequencia'].sum()
    else:
        df_estatistica['frequencia']=df[coluna].value_counts().sort_index()
        df_estatistica['freq_relativa'] =df[coluna].value_counts(normalize = True).sort_index()
    
    df_estatistica['freq_acumulada']= df_estatistica['frequencia'].cumsum()
    df_estatistica['freq_relativa_acumulada']=df_estatistica['freq_relativa'].cumsum()
    
    return df_estatistica


def composicao_hist_boxplot(dataframe, coluna, intervalos = 'auto'):
    fig, (ax1, ax2) = plt.subplots(
    nrows =2, 
    ncols = 1,
    sharex = True,
    gridspec_kw = {'height_ratios': (0.15, 0.85),
                  'hspace': 0.02
                  }
    )
    
    
    
    sns.boxplot(data = dataframe,
                x= coluna, 
                showmeans= True,
                meanline = True,
                meanprops = {'color':'C1', 'linewidth': 1.5, 'linestyle' : '--'},
                medianprops = {'color':'C2', 'linewidth': 1.5, 'linestyle' : '--'},
                ax = ax1
    )
    
    sns.histplot( data = dataframe, x= coluna, kde = True, bins = intervalos, ax = ax2)
    
    for ax in (ax1, ax2):
        ax.grid(True, linestyle = '--', color = 'gray', alpha = 0.5)
        ax.set_axisbelow(True)
    
    ax2.axvline(dataframe[coluna].mean(), color = 'C1', linestyle = '--', label = 'Média')
    ax2.axvline(dataframe[coluna].median(), color = 'C2', linestyle = '--', label = 'Mediana')
    ax2.axvline(dataframe[coluna].mode()[0], color = 'C3', linestyle = '--', label = 'Moda')
    
    
    ax2.legend()
    plt.show()


def analise_shapiro(dataframe, alfa = 0.05):
    print('Teste de Shapiro-Wilk')
    for coluna in dataframe.columns:
        estatistica_sw, valor_p_sw = shapiro(dataframe[coluna], nan_policy = 'omit')
        print(f'{estatistica_sw = :.3f}')
        if valor_p_sw > alfa:
            print(f'{coluna} segue uma distribuição normal (valor p: {valor_p_sw:.3f})')
        else:
            print(f'{coluna} não segue uma distribuição normal (valor p: {valor_p_sw:.3f})')

def analise_levene(dataframe, alfa = 0.05, centro = 'mean'):
    print('Teste de Levene')
    
    estatistica_levene, valor_p_levene = levene(
        *[dataframe[coluna] for coluna in dataframe.columns], 
        center = centro, 
        nan_policy = 'omit'
    )
    
    print(f'{estatistica_levene = :.3f}')
    
    if valor_p_levene > alfa:
        print(f'Variâncias iguais. (valor p: {valor_p_levene:.3f})')
    else:
        print(f'Ao menos uma variância é diferente. (valor p: {valor_p_levene:.3f})')

def analises_shapiro_levene(dataframe, alfa = 0.05, centro = 'mean'):
    analise_shapiro(dataframe, alfa)

    print()

    analise_levene(dataframe, alfa, centro)


def analise_ttest_ind(
    dataframe,
    alfa = 0.05,
    variancias_iguais = True,
    alternativa = 'two-sided',
):
    print('Teste t de Student')
    estatistica_ttest, valor_p_ttest = ttest_ind(
        *[dataframe[coluna] for coluna in dataframe.columns], 
        equal_var = variancias_iguais,
        alternative = alternativa,
        nan_policy = 'omit'
    )
    
    print(f'{estatistica_ttest = :.3f}')
    
    if valor_p_ttest > alfa:
        print(f'Não rejeita a hipótese nula (valor p: {valor_p_ttest:.3f})')
    else:
        print(f'Rejeita a hhipótese nula (valor p: {valor_p_ttest:.3f})')
    

def analise_ttest_rel(
    dataframe,
    alfa = 0.05,
    alternativa = 'two-sided',
):
    print('Teste t de Student')
    estatistica_ttest, valor_p_ttest = ttest_rel(
        *[dataframe[coluna] for coluna in dataframe.columns], 
        alternative = alternativa,
        nan_policy = 'omit'
    )
    
    print(f'{estatistica_ttest = :.3f}')
    
    if valor_p_ttest > alfa:
        print(f'Não rejeita a hipótese nula (valor p: {valor_p_ttest:.3f})')
    else:
        print(f'Rejeita a hhipótese nula (valor p: {valor_p_ttest:.3f})')

def analise_anova_one_way(
    dataframe,
    alfa = 0.05
):
    print('Teste ANOVA one way')
    estatistica_f, valor_p_f = f_oneway(
        *[dataframe[coluna] for coluna in dataframe.columns], 
        nan_policy = 'omit'
    )
    
    print(f'{estatistica_f = :.3f}')
    
    if valor_p_f > alfa:
        print(f'Não rejeita a hipótese nula (valor p: {valor_p_f:.3f})')
    else:
        print(f'Rejeita a hhipótese nula (valor p: {valor_p_f:.3f})')


def analise_wilcoxon(
    dataframe,
    alfa = 0.05,
    alternativa = 'two-sided'
):
    print('Teste de Wilcoxon')
    estatistica_wilcoxon, valor_p_wilcoxon = wilcoxon(
        *[dataframe[coluna] for coluna in dataframe.columns], 
        nan_policy = 'omit',
        alternative = alternativa
    )
    
    print(f'{estatistica_wilcoxon = :.3f}')
    
    if valor_p_wilcoxon > alfa:
        print(f'Não rejeita a hipótese nula (valor p: {valor_p_wilcoxon:.3f})')
    else:
        print(f'Rejeita a hipótese nula (valor p: {valor_p_wilcoxon:.3f})')

def analise_mannwhitneyu(
    dataframe,
    alfa = 0.05,
    alternativa = 'two-sided'
):
    print('Teste de Mann-Whitney')
    estatistica_mw, valor_p_mw = mannwhitneyu(
        *[dataframe[coluna] for coluna in dataframe.columns], 
        nan_policy = 'omit',
        alternative = alternativa
    )
    
    print(f'{estatistica_mw = :.3f}')
    
    if valor_p_mw > alfa:
        print(f'Não rejeita a hipótese nula (valor p: {valor_p_mw:.3f})')
    else:
        print(f'Rejeita a hipótese nula (valor p: {valor_p_mw:.3f})')


def analise_friedman(
    dataframe,
    alfa = 0.05,
):
    print('Teste de Friedman')
    estatistica_friedman, valor_p_friedman = friedmanchisquare(
        *[dataframe[coluna] for coluna in dataframe.columns], 
        nan_policy = 'omit',
    )
    
    print(f'{estatistica_friedman = :.3f}')
    
    if valor_p_friedman > alfa:
        print(f'Não rejeita a hipótese nula (valor p: {valor_p_friedman:.3f})')
    else:
        print(f'Rejeita a hipótese nula (valor p: {valor_p_friedman:.3f})')


def analise_kruskal(
    dataframe,
    alfa = 0.05,
):
    print('Teste de Kruskal')
    estatistica_kruskal, valor_p_kruskal = kruskal(
        *[dataframe[coluna] for coluna in dataframe.columns], 
        nan_policy = 'omit',
    )
    
    print(f'{estatistica_kruskal = :.3f}')
    
    if valor_p_kruskal > alfa:
        print(f'Não rejeita a hipótese nula (valor p: {valor_p_kruskal:.3f})')
    else:
        print(f'Rejeita a hipótese nula (valor p: {valor_p_kruskal:.3f})')
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2, norm


def check_srm(df, ratios=[0.5, 0.5]):
    group_sizes = df['variable'].value_counts().to_dict()
    group_sizes = np.array([group_sizes['control'], group_sizes['treatment']])
    ratios = np.array(ratios)
    total_num = group_sizes.sum()
    expected_sizes = total_num * ratios
    statistics = (((group_sizes - expected_sizes) ** 2) / (expected_sizes)).sum()
    p_value = 1 - chi2.cdf(statistics, group_sizes.shape[0] - 1)
    return p_value


def bootstrap(df, B):
    ctrs = []
    sample_size = df.shape[0]
    for _ in range(B):
        chosen_instances = np.random.choice(sample_size, size=sample_size, replace=True)
        n_imps = df.iloc[chosen_instances].imps.sum()
        n_clicks = df.iloc[chosen_instances].clicks.sum()
        ctrs.append(n_clicks / n_imps)
    return np.var(ctrs)


st.title('ABテストツール')

st.header('ファイルアップロード')
st.write('以下のような形式のCSVファイルをアップロードする')

st.table(
    pd.DataFrame(
        [
            [1, 'control', 13, 3],
            [2, 'treatment', 20, 4],
            [3, 'control', 25, 2],
            [4, 'treatment', 35, 6],
            [5, 'control', 22, 1],
        ],
        columns=['user_id', 'variable', 'imps', 'clicks']
    )
)

uploaded_file = st.file_uploader('Choose a file')
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.table(df.head())


st.header('SRMチェック')
if uploaded_file is not None:
    pvalue = check_srm(df, ratios=[0.5, 0.5])
    if pvalue > 0.01:
        st.write(f'SRMは検出されませんでした.(p-value: {pvalue})')
    else:
        st.write('SRMが検出されました．テストを終了します.(p-value: {pvalue})')

st.header('ヒストグラム')
if uploaded_file is not None:
    fig = plt.figure(figsize=(12, 9))
    ax = plt.axes()
    ax = sns.histplot(
        x='ctr',
        hue='variable',
        data=df.assign(
            ctr=lambda tdf: tdf['clicks'] / tdf['imps']
        ),
        ax=ax,
        kde=True
    )
    st.pyplot(fig)

st.header('差の検定')
if uploaded_file is not None:
    control_ctr = df.loc[df['variable'] == 'control', 'clicks'].sum() / df.loc[df['variable'] == 'control', 'imps'].sum()
    treatment_ctr = df.loc[df['variable'] == 'treatment', 'clicks'].sum() / df.loc[df['variable'] == 'treatment', 'imps'].sum()
    diff = treatment_ctr - control_ctr
    control_var = bootstrap(df[df['variable'] == 'control'], 1000)
    treatment_var = bootstrap(df[df['variable'] == 'treatment'], 1000)
    diff_se = np.sqrt(control_var + treatment_var)
    pvalue = 1 - norm.cdf(diff / diff_se)
    st.table(pd.DataFrame(
        [[
            treatment_ctr, control_ctr, diff, diff_se, pvalue
        ]],
        columns=['treatmentのCTR', 'controlのCTR', '差', '差の標準誤差', 'P値']
    ))

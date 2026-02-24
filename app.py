import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import warnings
import io

warnings.filterwarnings("ignore")

# ==========================================
# FUNZIONI STATISTICHE
# ==========================================
def check_solidity(df, target, group):
    solidity = []
    for g in df[group].dropna().unique():
        sub = df[df[group] == g][target].dropna()
        n = len(sub)
        out = 0
        if n > 0:
            Q1, Q3 = sub.quantile(0.25), sub.quantile(0.75)
            out = ((sub < (Q1 - 1.5*(Q3-Q1))) | (sub > (Q3 + 1.5*(Q3-Q1)))).sum()
        dist = "Normal" if n >= 3 and stats.shapiro(sub)[1] >= 0.05 else ("Skewed" if n >= 3 else "n<3")
        solidity.append([g, n, out, dist])
    return pd.DataFrame(solidity, columns=['Group', 'N', 'Out', 'Dist'])

def get_stats(df, target, group):
    s = df.groupby(group)[target].agg(['mean', 'std', 'count'])
    s['se'] = s['std'] / np.sqrt(s['count'])
    s['cv%'] = (s['std'] / s['mean']) * 100
    return s[['mean', 'std', 'se', 'cv%']].round(3).reset_index().rename(columns={'mean':'Mean', 'std':'SD', 'se':'SE', 'cv%':'CV%'})

def get_anova(df, target, group):
    grps = [g[target].dropna() for _, g in df.groupby(group)]
    if len(grps) < 2: return "N/A", 0, 1
    f, p = stats.f_oneway(*grps)
    sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns"
    return sig, f, p

def get_tukey(df, target, group):
    df_clean = df.dropna(subset=[target, group])
    if df_clean[group].nunique() < 2: return pd.DataFrame()
    try:
        t = pairwise_tukeyhsd(df_clean[target], df_clean[group])
        res = pd.DataFrame(data=t.summary().data[1:], columns=t.summary().data[0])
        return res.rename(columns={'group1':'A', 'group2':'B', 'meandiff':'Diff', 'p-adj':'p', 'reject':'Sig'})[['A','B','Diff','p','Sig']]
    except: return pd.DataFrame()

# NUOVO ALGORITMO ROBUSTO CLD (Compact Letter Display) PER IL TUKEY
def get_letters(df, target, group):
    df_clean = df.dropna(subset=[target, group])
    grps = df_clean[group].unique()
    
    if len(grps) < 2: return {g:'a' for g in grps}
    try:
        tukey = pairwise_tukeyhsd(df_clean[target], df_clean[group])
        res = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])
    except: 
        return {g:'a' for g in grps}

    means = df_clean.groupby(group)[target].mean().sort_values(ascending=False)
    sorted_groups = means.index.tolist()
    
    # Crea matrice di significatività
    sig_matrix = pd.DataFrame(False, index=sorted_groups, columns=sorted_groups)
    for idx, row in res.iterrows():
        g1, g2 = row['group1'], row['group2']
        if row['reject']: 
            if g1 in sorted_groups and g2 in sorted_groups:
                sig_matrix.loc[g1, g2] = True
                sig_matrix.loc[g2, g1] = True

    letters_map = {g: [] for g in sorted_groups}
    current_letter = 'a'

    # Assegnazione raggruppata delle lettere
    for i in range(len(sorted_groups)):
        cluster = [sorted_groups[i]]
        for j in range(i + 1, len(sorted_groups)):
            can_join = True
            for member in cluster:
                if sig_matrix.loc[sorted_groups[j], member]:
                    can_join = False
                    break
            if can_join:
                cluster.append(sorted_groups[j])
        
        for member in cluster:
            if current_letter not in letters_map[member]:
                letters_map[member].append(current_letter)
        current_letter = chr(ord(current_letter) + 1)
    
    # Pulizia delle lettere ridondanti
    letter_to_groups = {}
    for g, letters in letters_map.items():
        for l in letters:
            if l not in letter_to_groups:
                letter_to_groups[l] = set()
            letter_to_groups[l].add(g)
            
    letters_to_remove = set()
    all_letters = list(letter_to_groups.keys())
    for l1 in all_letters:
        for l2 in all_letters:
            if l1 != l2 and l1 not in letters_to_remove and l2 not in letters_to_remove:
                if letter_to_groups[l1].issubset(letter_to_groups[l2]):
                    letters_to_remove.add(l1)
                    
    final_letters = {}
    for g, letters in letters_map.items():
        final_letters[g] = [l for l in letters if l not in letters_to_remove]
        
    # Rinomina le lettere partendo da 'a' in ordine pulito
    used_letters = set()
    for letters in final_letters.values():
        used_letters.update(letters)
    
    sorted_used = sorted(list(used_letters))
    remap = {old: chr(97+i) for i, old in enumerate(sorted_used)}
    
    for g in final_letters:
        final_letters[g] = "".join(sorted([remap[l] for l in final_letters[g]]))
        
    return final_letters

# ==========================================
# SETUP STREAMLIT APP E STATO SESSIONE
# ==========================================
st.set_page_config(page_title="Data Viz & Stats App", layout="wide")
st.title("📊 Piattaforma Avanzata di Analisi Dati")

# Inizializza array per il report in PDF
if 'report_figs' not in st.session_state:
    st.session_state.report_figs = []

# --- SIDEBAR GLOBALE ---
st.sidebar.header("1. Stile Globale")
font_choice = st.sidebar.selectbox("Font Base", ['sans-serif', 'serif', 'monospace'])
global_font_scale = st.sidebar.slider("Scala Font Globale", 0.5, 3.0, 1.2, 0.1)
palette_choice = st.sidebar.selectbox("Palette Colori", ['deep', 'muted', 'pastel', 'bright', 'dark', 'colorblind', 'Set1', 'Set2'])

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=global_font_scale)
plt.rcParams.update({'font.family': font_choice})

# --- DATI ---
st.header("📝 1. Gestione Dati")
uploaded_file = st.file_uploader("Carica un file CSV o Excel (Opzionale)", type=['csv', 'xlsx'])

if uploaded_file is not None:
    try:
        df_init = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Errore caricamento: {e}")
        df_init = pd.DataFrame({"Categoria": ["A", "A", "B", "B"], "Valore": [10, 12, 8, 9]})
else:
    # Dati fittizi di esempio basati sul tuo script
    df_init = pd.DataFrame({
        "Materia Prima": ["orange", "orange", "lemon", "lemon", "orange", "lemon"],
        "Tempo di Cottura": [45, 75, 45, 75, 45, 75], 
        "Umidità %": [52.4, 39.8, 44.0, 31.5, 53.0, 42.5],
        "Opacità": [60.5, 79.5, 55.1, 76.9, 66.3, 76.1],
        "tanδ": [0.243, 0.202, 0.215, 0.207, 0.264, 0.215]
    })

if 'data' not in st.session_state or uploaded_file is not None:
    st.session_state.data = df_init.copy()

# TABELLA E STRUTTURA
st.markdown("**Compila o modifica i tuoi dati qui sotto:**")
edited_df = st.data_editor(st.session_state.data, num_rows="dynamic", use_container_width=True, key="data_editor")
st.session_state.data = edited_df.copy()

with st.expander("🛠️ Aggiungi o Rinomina Colonne"):
    col_add, col_ren = st.columns(2)
    with col_add:
        st.markdown("**➕ Aggiungi una nuova colonna**")
        new_col_name = st.text_input("Nome della nuova colonna")
        new_col_type = st.selectbox("Tipo di dato", ["Numerico", "Testo (Categoria)"])
        if st.button("Aggiungi Colonna", type="secondary"):
            if new_col_name and new_col_name not in st.session_state.data.columns:
                st.session_state.data[new_col_name] = 0.0 if new_col_type == "Numerico" else "Nuovo Valore"
                st.rerun()
            elif new_col_name in st.session_state.data.columns:
                st.warning("Esiste già una colonna con questo nome!")

    with col_ren:
        st.markdown("**✏️ Rinomina una colonna esistente**")
        old_name = st.selectbox("Seleziona la colonna da rinominare", st.session_state.data.columns)
        new_name = st.text_input("Inserisci il nuovo titolo")
        if st.button("Rinomina Colonna", type="secondary"):
            if new_name and new_name not in st.session_state.data.columns:
                st.session_state.data = st.session_state.data.rename(columns={old_name: new_name})
                st.rerun()
            elif new_name in st.session_state.data.columns:
                st.warning("Il nuovo nome è già in uso!")

df = st.session_state.data.copy()

if not df.empty:
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    all_cols = df.columns.tolist()
    
    st.markdown("---")
    st.header("📈 2. Analisi e Visualizzazione")
    
    # 5 SCHEDE INVECE DI 3
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Categoriale (1 Fattore)", 
        "📊 Categoriale Raggruppato (2 Fattori)", 
        "📈 Correlazione Singoli Punti", 
        "📈 Scatterplot Mediati (Centroidi)", 
        "📑 Report & Export PDF"
    ])
    
    # ==========================================
    # TAB 1: ANALISI CATEGORIALE (1 Fatto)
    # ==========================================
    with tab1:
        st.markdown("👉 **Ideale per:** Confrontare gruppi semplici su un'unica variabile X.")
        col_x, col_y = st.columns(2)
        with col_x: x_cat = st.selectbox("Asse X (Categoria)", all_cols, key="cat_x1")
        with col_y: y_num = st.selectbox("Asse Y (Numerico)", numeric_cols, key="cat_y1") if numeric_cols else None
        
        if y_num:
            df_clean = df.dropna(subset=[x_cat, y_num])
            with st.expander("⚙️ Parametri Avanzati del Grafico"):
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    plot_type = st.selectbox("Tipo di Grafico", ['Bar Plot', 'Box Plot', 'Scatter/Swarm'], key="ptype1")
                    custom_title = st.text_input("Titolo Grafico", f"Analisi di {y_num} per {x_cat}", key="title1")
                    title_size1 = st.slider("Dimensione Titolo", 10, 30, 16, key="ts1")
                with c2:
                    custom_xlab = st.text_input("Nome Asse X", x_cat, key="xlab1")
                    xlab_size1 = st.slider("Dimens. Asse X", 8, 24, 12, key="xsize1")
                    show_tukey = st.checkbox("Mostra Lettere Tukey", value=True, key="tukey1")
                with c3:
                    custom_ylab = st.text_input("Nome Asse Y", y_num, key="ylab1")
                    ylab_size1 = st.slider("Dimens. Asse Y", 8, 24, 12, key="ysize1")
                    auto_scale1 = st.checkbox("Scala Y Auto", value=True, key="ascale1")
                    y_max1 = st.number_input("Y Max (se manuale)", value=float(df_clean[y_num].max()*1.2) if not df_clean.empty else 10.0, disabled=auto_scale1, key="ymax1")
                with c4:
                    fig_w1 = st.number_input("Larghezza Fig.", value=8.0, key="fw1")
                    fig_h1 = st.number_input("Altezza Fig.", value=6.0, key="fh1")
                    bar_w = st.slider("Spessore Elementi", 0.1, 1.0, 0.4, key="bw1")
                    err_w = st.slider("Spessore Bordi/Errori", 0.5, 3.0, 1.5, key="ew1")

            fig1, ax1 = plt.subplots(figsize=(fig_w1, fig_h1), dpi=300)
            order = df_clean.groupby(x_cat)[y_num].mean().sort_values(ascending=False).index
            letters = get_letters(df_clean, y_num, x_cat) if show_tukey else {}
            
            if plot_type == 'Bar Plot':
                sns.barplot(data=df_clean, x=x_cat, y=y_num, order=order, ax=ax1, palette=palette_choice, edgecolor=".2", linewidth=err_w, width=bar_w, capsize=0.1, err_kws={'linewidth': err_w}, errorbar='sd')
            elif plot_type == 'Box Plot':
                sns.boxplot(data=df_clean, x=x_cat, y=y_num, order=order, ax=ax1, palette=palette_choice, width=bar_w, linewidth=err_w, showfliers=False)
                sns.stripplot(data=df_clean, x=x_cat, y=y_num, order=order, ax=ax1, color=".2", size=5, alpha=0.6)
            else:
                sns.swarmplot(data=df_clean, x=x_cat, y=y_num, order=order, ax=ax1, palette=palette_choice, size=6)
                sns.pointplot(data=df_clean, x=x_cat, y=y_num, order=order, ax=ax1, color='black', join=False, markers='_', scale=1.5, errorbar=None)

            if not auto_scale1: ax1.set_ylim(0, y_max1)
            else: ax1.set_ylim(0, ax1.get_ylim()[1] * 1.15)

            if show_tukey:
                y_limit = ax1.get_ylim()[1]
                for i, cat in enumerate(order):
                    l = letters.get(cat, "")
                    val = df_clean[df_clean[x_cat]==cat][y_num].dropna()
                    h = (val.mean() + val.std()) if plot_type == 'Bar Plot' else val.max()
                    if pd.isna(h): h = val.mean() if not pd.isna(val.mean()) else 0
                    ax1.text(i, h + (y_limit * 0.05), l, ha='center', fontsize=12*global_font_scale, weight='bold')

            ax1.set_title(custom_title, fontsize=title_size1, fontweight='bold', pad=15)
            ax1.set_xlabel(custom_xlab, fontsize=xlab_size1, fontweight='bold')
            ax1.set_ylabel(custom_ylab, fontsize=ylab_size1, fontweight='bold')
            sns.despine(ax=ax1, trim=True)
            
            st.pyplot(fig1)
            
            if st.button("➕ Aggiungi al Report PDF", key="add_cat1", type="primary"):
                st.session_state.report_figs.append(fig1)
                st.success("✅ Grafico aggiunto con successo al Report!")

            st.subheader("Tabelle Statistiche")
            st.dataframe(get_stats(df_clean, y_num, x_cat), use_container_width=True)
            st.dataframe(get_tukey(df_clean, y_num, x_cat), use_container_width=True)


    # ==========================================
    # TAB 2: ANALISI CATEGORIALE A 2 FATTORI (NUOVO)
    # ==========================================
    with tab2:
        st.markdown("👉 **Ideale per:** Confrontare due variabili incrociate (es. Materia Prima x Tempo di Cottura).")
        col_x2, col_h2, col_y2 = st.columns(3)
        with col_x2: x_cat2 = st.selectbox("Asse X (Fattore Principale)", all_cols, key="cat_x2")
        with col_h2: hue_cat2 = st.selectbox("Colore / Gruppo (Fattore 2)", all_cols, key="cat_h2")
        with col_y2: y_num2 = st.selectbox("Asse Y (Numerico)", numeric_cols, key="cat_y2")

        if y_num2 and x_cat2 and hue_cat2:
            df_clean2 = df.dropna(subset=[x_cat2, hue_cat2, y_num2]).copy()
            # Creazione condizione combinata
            df_clean2['Condition'] = df_clean2[hue_cat2].astype(str) + '_' + df_clean2[x_cat2].astype(str)
            
            with st.expander("⚙️ Parametri Avanzati del Barplot Raggruppato"):
                c1, c2, c3 = st.columns(3)
                with c1:
                    custom_title2 = st.text_input("Titolo Grafico", f"{y_num2} grouped by {x_cat2} & {hue_cat2}", key="t2")
                    show_tukey2 = st.checkbox("Mostra Lettere Tukey (sui 2 Fattori)", value=True, key="tukey2")
                with c2:
                    auto_scale2 = st.checkbox("Scala Y Auto", value=True, key="ascale2")
                    y_max2 = st.number_input("Y Max", value=float(df_clean2[y_num2].max()*1.2), disabled=auto_scale2, key="ymax2")
                with c3:
                    fig_w2 = st.number_input("Larghezza", value=8.0, key="fw2")
                    fig_h2 = st.number_input("Altezza", value=6.0, key="fh2")

            fig2_b, ax2_b = plt.subplots(figsize=(fig_w2, fig_h2), dpi=300)
            
            sns.barplot(data=df_clean2, x=x_cat2, y=y_num2, hue=hue_cat2, palette=palette_choice, 
                        capsize=.1, errorbar='sd', ax=ax2_b, edgecolor=".2", linewidth=1.0)
            
            if not auto_scale2: ax2_b.set_ylim(0, y_max2)
            else:
                y_max_calc = (df_clean2.groupby('Condition')[y_num2].mean() + df_clean2.groupby('Condition')[y_num2].std().fillna(0)).max()
                ax2_b.set_ylim(0, y_max_calc * 1.3)

            if show_tukey2 and len(df_clean2['Condition'].unique()) > 1:
                letters_dict = get_letters(df_clean2, y_num2, 'Condition')
                x_levels = sorted(df_clean2[x_cat2].unique())
                hue_levels = sorted(df_clean2[hue_cat2].unique())
                n_hues = len(hue_levels)
                
                # Calcolo offset per posizionare le lettere al centro di ciascuna barra
                offsets = np.linspace(-0.4 + (0.4/n_hues), 0.4 - (0.4/n_hues), n_hues)
                
                for idx_x, x_val in enumerate(x_levels):
                    for idx_h, h_val in enumerate(hue_levels):
                        cond = f"{h_val}_{x_val}"
                        subset = df_clean2[df_clean2['Condition'] == cond][y_num2]
                        if not subset.empty:
                            m, s = subset.mean(), subset.std()
                            if pd.isna(s): s = 0
                            x_pos = idx_x + offsets[idx_h]
                            letter = letters_dict.get(cond, "")
                            ax2_b.text(x_pos, m + s + (ax2_b.get_ylim()[1] * 0.05), letter, ha='center', fontweight='bold', fontsize=12*global_font_scale)

            ax2_b.set_title(custom_title2, fontweight='bold', pad=15)
            ax2_b.legend(title=hue_cat2, loc='upper right')
            sns.despine(ax=ax2_b, trim=True)
            
            st.pyplot(fig2_b)
            
            if st.button("➕ Aggiungi al Report PDF", key="add_cat2", type="primary"):
                st.session_state.report_figs.append(fig2_b)
                st.success("✅ Grafico aggiunto con successo al Report!")


    # ==========================================
    # TAB 3: CORRELAZIONE & REGRESSIONE
    # ==========================================
    with tab3:
        st.markdown("👉 **Ideale per:** Trovare la retta di tendenza su tutti i singoli campioni misurati.")
        if len(numeric_cols) < 2:
            st.warning("Servono almeno due colonne numeriche.")
        else:
            col_rx, col_ry, col_rg = st.columns(3)
            with col_rx: reg_x = st.selectbox("Asse X (Numerico)", numeric_cols, index=0)
            with col_ry: reg_y = st.selectbox("Asse Y (Numerico)", numeric_cols, index=1 if len(numeric_cols)>1 else 0)
            with col_rg: reg_group = st.selectbox("Raggruppa per colore (Opzionale)", ["Nessuno"] + all_cols)
            
            df_reg = df.dropna(subset=[reg_x, reg_y])
            
            with st.expander("⚙️ Parametri Avanzati"):
                r1, r2, r3 = st.columns(3)
                with r1:
                    custom_title3 = st.text_input("Titolo", f"{reg_y} vs {reg_x}")
                    show_r2 = st.checkbox("Mostra R²", value=True)
                with r2:
                    show_ci = st.checkbox("Mostra Intervallo di Confidenza (95%)", value=True)
                    line_w = st.slider("Spessore Linea Regressione", 1.0, 5.0, 2.0)
                with r3:
                    fig_w3 = st.number_input("Larghezza Fig.", value=8.0, key="fw3")
                    fig_h3 = st.number_input("Altezza Fig.", value=6.0, key="fh3")

            fig3, ax3 = plt.subplots(figsize=(fig_w3, fig_h3), dpi=300)
            
            r_val = 0
            if len(df_reg) > 1:
                slope, intercept, r_value, p_value, std_err = stats.linregress(df_reg[reg_x], df_reg[reg_y])
                r_val = r_value**2

            sns.regplot(data=df_reg, x=reg_x, y=reg_y, ax=ax3, scatter=False, color='gray', ci=95 if show_ci else None, line_kws={"linestyle": "--", "linewidth": line_w, "alpha": 0.6})
            
            if reg_group != "Nessuno":
                sns.scatterplot(data=df_reg, x=reg_x, y=reg_y, hue=reg_group, ax=ax3, palette=palette_choice, s=80, edgecolor='black', alpha=0.8)
                ax3.legend(title=reg_group, bbox_to_anchor=(1.05, 1), loc='upper left')
            else:
                sns.scatterplot(data=df_reg, x=reg_x, y=reg_y, ax=ax3, color=sns.color_palette(palette_choice)[0], s=80, edgecolor='black', alpha=0.8)

            final_title = f"{custom_title3}\n$R^2 = {r_val:.3f}$" if show_r2 else custom_title3
            ax3.set_title(final_title, fontweight='bold', pad=15)
            sns.despine(ax=ax3)
            
            st.pyplot(fig3)
            
            if st.button("➕ Aggiungi al Report PDF", key="add_reg3", type="primary"):
                st.session_state.report_figs.append(fig3)
                st.success("✅ Grafico aggiunto con successo al Report!")

    # ==========================================
    # TAB 4: SCATTERPLOT MEDIATI (NUOVO)
    # ==========================================
    with tab4:
        st.markdown("👉 **Ideale per:** Disegnare i centroidi (medie) di gruppi combinati, aggiungendo incertezza (barre di errore) sia su X che su Y senza tracciare la regressione.")
        if len(numeric_cols) < 2:
            st.warning("Servono almeno due colonne numeriche.")
        else:
            c_sx, c_sy, c_sc, c_ss = st.columns(4)
            with c_sx: sc_x = st.selectbox("Asse X (Num)", numeric_cols, index=0, key="sc_x")
            with c_sy: sc_y = st.selectbox("Asse Y (Num)", numeric_cols, index=1 if len(numeric_cols)>1 else 0, key="sc_y")
            with c_sc: sc_color = st.selectbox("Fattore Colore", all_cols, key="sc_color")
            with c_ss: sc_style = st.selectbox("Fattore Forma (Marker)", all_cols, key="sc_style")
            
            df_scat = df.dropna(subset=[sc_x, sc_y, sc_color, sc_style])
            
            if not df_scat.empty:
                # Aggrega i dati
                aggr_df = df_scat.groupby([sc_color, sc_style])[[sc_x, sc_y]].agg(['mean', 'std']).reset_index()
                aggr_df.columns = [f"{c[0]}_{c[1]}" if c[1] else c[0] for c in aggr_df.columns]
                
                with st.expander("⚙️ Parametri Avanzati"):
                    st_fw = st.number_input("Larghezza Fig", value=8.0, key="fw4")
                    st_fh = st.number_input("Altezza Fig", value=6.0, key="fh4")
                
                fig4, ax4 = plt.subplots(figsize=(st_fw, st_fh), dpi=300)
                
                color_levels = sorted(aggr_df[sc_color].unique())
                style_levels = sorted(aggr_df[sc_style].unique())
                palette_colors = sns.color_palette(palette_choice, len(color_levels))
                markers = ['o', 'D', 's', '^', 'v', 'p', '*', 'h']
                
                for idx_c, c_val in enumerate(color_levels):
                    for idx_s, s_val in enumerate(style_levels):
                        row = aggr_df[(aggr_df[sc_color] == c_val) & (aggr_df[sc_style] == s_val)]
                        if not row.empty:
                            x_m, y_m = row[f"{sc_x}_mean"].values[0], row[f"{sc_y}_mean"].values[0]
                            x_s, y_s = row[f"{sc_x}_std"].values[0], row[f"{sc_y}_std"].values[0]
                            
                            marker_sym = markers[idx_s % len(markers)]
                            
                            ax4.errorbar(x_m, y_m, xerr=x_s, yerr=y_s, 
                                         fmt=marker_sym, capsize=6, markersize=12, 
                                         color=palette_colors[idx_c], markeredgecolor='black',
                                         label=f'{c_val} - {s_val}')

                ax4.set_title(f"Centroidi: {sc_y} vs {sc_x}", fontweight='bold', pad=15)
                ax4.set_xlabel(sc_x)
                ax4.set_ylabel(sc_y)
                ax4.legend(title="Legenda Combinata", bbox_to_anchor=(1.05, 1), loc='upper left')
                sns.despine(ax=ax4)
                
                st.pyplot(fig4)
                
                if st.button("➕ Aggiungi al Report PDF", key="add_scat4", type="primary"):
                    st.session_state.report_figs.append(fig4)
                    st.success("✅ Grafico aggiunto con successo al Report!")


    # ==========================================
    # TAB 5: REPORT PDF
    # ==========================================
    with tab5:
        st.markdown("### 📚 Generazione Fascicolo PDF")
        st.write("Qui puoi visualizzare tutti i grafici che hai messo da parte e scaricarli in un unico documento PDF pronto per la stampa o la presentazione.")
        
        if not st.session_state.report_figs:
            st.info("📌 Il tuo report è attualmente vuoto. Vai nei tab precedenti e clicca su '➕ Aggiungi al Report PDF'.")
        else:
            st.success(f"Hai **{len(st.session_state.report_figs)}** grafici pronti per l'esportazione nel tuo fascicolo.")
            
            # Genera il PDF multi-pagina in memoria
            pdf_buffer = io.BytesIO()
            with PdfPages(pdf_buffer) as pdf:
                for f in st.session_state.report_figs:
                    pdf.savefig(f, bbox_inches='tight')
            pdf_buffer.seek(0)
            
            c_down, c_clear = st.columns(2)
            with c_down:
                st.download_button(
                    label="📥 SCARICA FASCICOLO PDF COMPLETO",
                    data=pdf_buffer,
                    file_name="Report_Analisi_Completo.pdf",
                    mime="application/pdf",
                    type="primary",
                    use_container_width=True
                )
            with c_clear:
                if st.button("🗑️ Svuota Report (Cancella Tutti)", use_container_width=True):
                    st.session_state.report_figs = []
                    st.rerun()

            st.markdown("---")
            st.markdown("#### 👁️ Anteprima dei grafici nel fascicolo:")
            cols = st.columns(2)
            for i, f in enumerate(st.session_state.report_figs):
                with cols[i % 2]:
                    st.write(f"**Pagina {i+1}**")
                    st.pyplot(f)
                    
else:
    st.info("La tabella dati è vuota. Inserisci dei dati per visualizzare il grafico.")


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

def get_letters(df, target, group):
    df_clean = df.dropna(subset=[target, group])
    grps = df_clean[group].unique()
    if len(grps) < 2: return {g:'a' for g in grps}
    try:
        tukey = pairwise_tukeyhsd(df_clean[target], df_clean[group])
        res = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])
    except: return {g:'a' for g in grps}

    means = df_clean.groupby(group)[target].mean().sort_values(ascending=False)
    chars = "abcdefghijklmnopqrstuvwxyz"
    current_char_idx = 0
    sorted_groups = means.index.tolist()
    
    sig_matrix = pd.DataFrame(False, index=sorted_groups, columns=sorted_groups)
    for _, row in res.iterrows():
        g1, g2 = row['group1'], row['group2']
        if row['reject'] and g1 in sorted_groups and g2 in sorted_groups:
            sig_matrix.loc[g1, g2] = True
            sig_matrix.loc[g2, g1] = True
            
    letters_map = {g: "" for g in sorted_groups}
    for i, g1 in enumerate(sorted_groups):
        if i == 0: letters_map[g1] = chars[0]
        else:
            prev = sorted_groups[i-1]
            if sig_matrix.loc[g1, prev]: current_char_idx += 1
            letters_map[g1] = chars[current_char_idx]
    return letters_map

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
    df_init = pd.DataFrame({
        "Trattamento": ["45 min", "45 min", "45 min", "75 min", "75 min", "75 min"], 
        "Umidità %": [52.4, 53.0, 42.4, 39.8, 42.9, 36.2],
        "Opacità": [60.5, 66.3, 67.4, 79.5, 77.2, 78.0]
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
    
    # 3 SCHEDE INVECE DI 2
    tab1, tab2, tab3 = st.tabs(["📊 Analisi Categoriale (Tukey)", "📈 Correlazione & Regressione", "📑 Report & Export PDF"])
    
    # ==========================================
    # TAB 1: ANALISI CATEGORIALE
    # ==========================================
    with tab1:
        st.markdown("👉 **Ideale per:** Confrontare gruppi o categorie.")
        col_x, col_y = st.columns(2)
        with col_x: x_cat = st.selectbox("Asse X (Categoria)", all_cols, key="cat_x")
        with col_y: y_num = st.selectbox("Asse Y (Numerico)", numeric_cols, key="cat_y") if numeric_cols else None
        
        if y_num:
            df_clean = df.dropna(subset=[x_cat, y_num])
            with st.expander("⚙️ Parametri Avanzati del Grafico Categoriale"):
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    plot_type = st.selectbox("Tipo di Grafico", ['Bar Plot', 'Box Plot', 'Scatter/Swarm'], key="ptype")
                    custom_title = st.text_input("Titolo Grafico", f"Analisi di {y_num} per {x_cat}")
                    title_size1 = st.slider("Dimensione Titolo", 10, 30, 16, key="ts1")
                with c2:
                    custom_xlab = st.text_input("Nome Asse X", x_cat, key="xlab1")
                    xlab_size1 = st.slider("Dimens. Testo Asse X", 8, 24, 12, key="xsize1")
                    show_tukey = st.checkbox("Mostra Lettere Tukey", value=True)
                with c3:
                    custom_ylab = st.text_input("Nome Asse Y", y_num, key="ylab1")
                    ylab_size1 = st.slider("Dimens. Testo Asse Y", 8, 24, 12, key="ysize1")
                    auto_scale1 = st.checkbox("Scala Y Auto", value=True, key="ascale1")
                    y_max1 = st.number_input("Y Max (se manuale)", value=float(df_clean[y_num].max()*1.2) if not df_clean.empty else 10.0, disabled=auto_scale1)
                with c4:
                    fig_w1 = st.number_input("Larghezza Fig.", value=8.0, key="fw1")
                    fig_h1 = st.number_input("Altezza Fig.", value=6.0, key="fh1")
                    bar_w = st.slider("Spessore Elementi", 0.1, 1.0, 0.4, key="bw")
                    err_w = st.slider("Spessore Linee (Bordi)", 0.5, 3.0, 1.5, key="ew")
                    tick_size1 = st.slider("Dimens. Numeri/Etichette Assi", 6, 20, 10, key="ticks1")

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
            ax1.tick_params(axis='both', which='major', labelsize=tick_size1)
            sns.despine(ax=ax1, trim=True)
            
            st.pyplot(fig1)
            
            # Pulsanti di salvataggio
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("➕ Aggiungi al Report PDF", key="add_cat_report", type="primary"):
                    st.session_state.report_figs.append(fig1)
                    st.success("✅ Grafico aggiunto con successo al Report!")
            with col_btn2:
                buf1 = io.BytesIO()
                fig1.savefig(buf1, format="pdf", bbox_inches='tight')
                buf1.seek(0)
                st.download_button("📥 Scarica Solo Questo (PDF)", data=buf1, file_name="Grafico_Categoriale.pdf", mime="application/pdf")

            st.subheader("Tabelle Statistiche Associate")
            st.dataframe(get_stats(df_clean, y_num, x_cat), use_container_width=True)
            st.dataframe(get_tukey(df_clean, y_num, x_cat), use_container_width=True)

    # ==========================================
    # TAB 2: CORRELAZIONE & REGRESSIONE
    # ==========================================
    with tab2:
        st.markdown("👉 **Ideale per:** Trovare relazioni tra due variabili continue.")
        if len(numeric_cols) < 2:
            st.warning("Servono almeno due colonne numeriche per un grafico di correlazione.")
        else:
            col_rx, col_ry, col_rg = st.columns(3)
            with col_rx: reg_x = st.selectbox("Asse X (Numerico)", numeric_cols, index=0)
            with col_ry: reg_y = st.selectbox("Asse Y (Numerico)", numeric_cols, index=1 if len(numeric_cols)>1 else 0)
            with col_rg: reg_group = st.selectbox("Raggruppa per colore (Opzionale)", ["Nessuno"] + all_cols)
            
            df_reg = df.dropna(subset=[reg_x, reg_y])
            with st.expander("⚙️ Parametri Avanzati del Grafico di Correlazione"):
                r1, r2, r3, r4 = st.columns(4)
                with r1:
                    custom_title2 = st.text_input("Titolo", f"{reg_y} vs {reg_x}")
                    title_size2 = st.slider("Dimensione Titolo", 10, 30, 16, key="ts2")
                    show_r2 = st.checkbox("Mostra R² nel Titolo", value=True)
                with r2:
                    custom_xlab2 = st.text_input("Nome Asse X", reg_x, key="xlab2")
                    xlab_size2 = st.slider("Dimens. Testo Asse X", 8, 24, 12, key="xsize2")
                    tick_size2 = st.slider("Dimens. Numeri/Etichette", 6, 20, 10, key="ticks2")
                with r3:
                    custom_ylab2 = st.text_input("Nome Asse Y", reg_y, key="ylab2")
                    ylab_size2 = st.slider("Dimens. Testo Asse Y", 8, 24, 12, key="ysize2")
                    show_ci = st.checkbox("Mostra Intervallo di Confidenza", value=True)
                with r4:
                    fig_w2 = st.number_input("Larghezza Fig.", value=8.0, key="fw2")
                    fig_h2 = st.number_input("Altezza Fig.", value=6.0, key="fh2")
                    marker_s = st.slider("Dimensione Punti", 10, 200, 60)
                    line_w = st.slider("Spessore Linea Regressione", 1.0, 5.0, 2.0)

            fig2, ax2 = plt.subplots(figsize=(fig_w2, fig_h2), dpi=300)
            
            r_val = 0
            if len(df_reg) > 1:
                slope, intercept, r_value, p_value, std_err = stats.linregress(df_reg[reg_x], df_reg[reg_y])
                r_val = r_value**2

            sns.regplot(data=df_reg, x=reg_x, y=reg_y, ax=ax2, scatter=False, color='gray', ci=95 if show_ci else None, line_kws={"linestyle": "--", "linewidth": line_w, "alpha": 0.6})
            
            if reg_group != "Nessuno":
                sns.scatterplot(data=df_reg, x=reg_x, y=reg_y, hue=reg_group, ax=ax2, palette=palette_choice, s=marker_s, edgecolor='black', alpha=0.8)
                ax2.legend(title=reg_group, bbox_to_anchor=(1.05, 1), loc='upper left')
            else:
                sns.scatterplot(data=df_reg, x=reg_x, y=reg_y, ax=ax2, color=sns.color_palette(palette_choice)[0], s=marker_s, edgecolor='black', alpha=0.8)

            final_title = f"{custom_title2}\n$R^2 = {r_val:.3f}$" if show_r2 else custom_title2
            ax2.set_title(final_title, fontsize=title_size2, fontweight='bold', pad=15)
            ax2.set_xlabel(custom_xlab2, fontsize=xlab_size2, fontweight='bold')
            ax2.set_ylabel(custom_ylab2, fontsize=ylab_size2, fontweight='bold')
            ax2.tick_params(axis='both', which='major', labelsize=tick_size2)
            sns.despine(ax=ax2, trim=False)
            
            st.pyplot(fig2)
            
            # Pulsanti di salvataggio
            col_btn3, col_btn4 = st.columns(2)
            with col_btn3:
                if st.button("➕ Aggiungi al Report PDF", key="add_reg_report", type="primary"):
                    st.session_state.report_figs.append(fig2)
                    st.success("✅ Grafico aggiunto con successo al Report!")
            with col_btn4:
                buf2 = io.BytesIO()
                fig2.savefig(buf2, format="pdf", bbox_inches='tight')
                buf2.seek(0)
                st.download_button("📥 Scarica Solo Questo (PDF)", data=buf2, file_name="Grafico_Correlazione.pdf", mime="application/pdf")

    # ==========================================
    # TAB 3: REPORT PDF
    # ==========================================
    with tab3:
        st.markdown("### 📚 Generazione Fascicolo PDF")
        st.write("Qui puoi visualizzare tutti i grafici che hai messo da parte e scaricarli in un unico documento PDF pronto per la stampa o la presentazione.")
        
        if not st.session_state.report_figs:
            st.info("📌 Il tuo report è attualmente vuoto. Vai nei tab 'Analisi Categoriale' o 'Correlazione' e clicca su '➕ Aggiungi al Report PDF' sotto i grafici che vuoi salvare.")
        else:
            st.success(f"Hai **{len(st.session_state.report_figs)}** grafici pronti per l'esportazione nel tuo fascicolo.")
            
            # Genera il PDF multi-pagina in memoria
            pdf_buffer = io.BytesIO()
            with PdfPages(pdf_buffer) as pdf:
                for f in st.session_state.report_figs:
                    pdf.savefig(f, bbox_inches='tight')
            pdf_buffer.seek(0)
            
            # Area Pulsanti (Download & Reset)
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
            
            # Mostra le anteprime in griglia
            cols = st.columns(2)
            for i, f in enumerate(st.session_state.report_figs):
                with cols[i % 2]:
                    st.write(f"**Pagina {i+1}**")
                    st.pyplot(f)
                    
else:
    st.info("La tabella dati è vuota. Inserisci dei dati per visualizzare il grafico.")


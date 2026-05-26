#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ZIPFOVA ANALÝZA TEXTU
=====================
Profesionální lingvistická analýza knih pomocí Zipfových zákonů.
"""

import os
import sys
import re
import requests
from collections import Counter
from importlib.util import find_spec
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress

import plotly.express as px
import plotly.graph_objects as go

import streamlit as st
import spacy
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from ufal.udpipe import Model, Pipeline, ProcessingError

# NLTK nemá nativní seznam pro češtinu, proto použijeme vlastní
CZECH_STOPWORDS = set(
    "a aby ačkoli asi až bez bude budem budeš budete budou by byl byla "
    "byli bylo byly bys být co do jak jako je jeho jej její jemu jen "
    "jestli jsme jsou k kam kde kdo když ke kvůli má mají me mezi mi mě "
    "my na nad nám nás náš ne nebo něco něho něj nějak někdo nic ní od "
    "ode on ona oni ono o po pod podle pokud pro proč při před přes pak "
    "s se si sice so své svůj ta tak také tam te tebe tebou tedy ten ti "
    "to tobě toto ty tvé tvůj u už v ve však všichni z za zda ze že".split()
)

# Stažení stop slov, pokud chybí
nltk.download('stopwords', quiet=True)
try:
    en_stops = set(stopwords.words("english"))
except OSError:
    en_stops = set("i me my myself we our ours ourselves you your yours yourself yourselves he him his himself she her hers herself it its itself they them their theirs themselves what which who whom this that these those am is are was were be been being have has had having do does did doing a an the and but if or because as until while of at by for with about against between into through during before after above below to from up down in out on off over under again further then once here there when where why how all any both each few more most other some such no nor not only own same so than too very s t can will just don should now".split())

STOPWORDS = {
    "Čeština": CZECH_STOPWORDS,
    "Angličtina": en_stops,
}


# ============================================================
# KONFIGURACE
# ============================================================

st.markdown("""
<style>
/* Zamezení ztmavení obrazovky při přepočítávání (rerunu) aplikace */
[data-testid="stAppViewContainer"], [data-testid="stAppViewBlockContainer"] {
    opacity: 1 !important;
    filter: none !important;
    transition: none !important;
}
.stApp {
    opacity: 1 !important;
}
</style>
""", unsafe_allow_html=True)

UDPIPE_DEFAULTS = {
    "Čeština": "czech-pdt-ud-2.5-191206.udpipe",
    "Angličtina": "english-ewt-ud-2.5-191206.udpipe",
}

# Počet slov zobrazených v grafu četnosti (fixní limit kvůli výkonu)
FREQ_GRAPH_LIMIT = 1000

st.set_page_config(
    page_title="Zipfova analýza textu – od počtu slov k duši textu.",
    layout="wide"
)

st.title("📚 Zipfova analýza textu – od počtu slov k duši textu.")
st.caption("Analýza rozložení četnosti slov podle Zipfových zákonů")


# ============================================================
# NAČÍTÁNÍ NLP MODELŮ
# ============================================================

@st.cache_resource
def load_spacy_model(lang: str):
    try:
        model_name = "en_core_web_sm" if lang == "Angličtina" else "cs_core_news_md"
        return spacy.load(model_name)
    except OSError:
        return None


def spacy_available(lang: str) -> bool:
    """Rychlá kontrola bez blokujícího načítání modelu."""
    model_name = "en_core_web_sm" if lang == "Angličtina" else "cs_core_news_md"
    return find_spec(model_name) is not None


# ============================================================
# UDPipe
# ============================================================

@st.cache_resource
def load_udpipe_model(model_path: str):
    # Nejprve zkusíme absolutní cestu ze složky skriptu
    script_dir = os.path.dirname(os.path.abspath(__file__))
    abs_path = os.path.join(script_dir, model_path)
    
    if not os.path.exists(abs_path):
        # Fallback na lokální cestu (pokud by model_path byla absolutní)
        if os.path.exists(model_path):
            abs_path = model_path
        else:
            raise RuntimeError(f"Nelze najít model UDPipe. Hledáno na: {abs_path}")

    # C++ knihovna může mít problém s diakritikou v cestě na Windows.
    # Použijeme short path (8.3)
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes
        _GetShortPathNameW = ctypes.windll.kernel32.GetShortPathNameW
        _GetShortPathNameW.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
        _GetShortPathNameW.restype = wintypes.DWORD

        output_buf_size = _GetShortPathNameW(abs_path, None, 0)
        if output_buf_size > 0:
            output_buf = ctypes.create_unicode_buffer(output_buf_size)
            if _GetShortPathNameW(abs_path, output_buf, output_buf_size) > 0:
                abs_path = output_buf.value

    model = Model.load(abs_path)
    if not model:
        raise RuntimeError(f"Nelze načíst model UDPipe ze souboru: {abs_path}")
    return model


def udpipe_lemmatize(text: str, model) -> list[tuple[str, str]]:
    pipeline = Pipeline(
        model,
        "tokenize",
        Pipeline.DEFAULT,
        Pipeline.DEFAULT,
        "conllu"
    )

    error = ProcessingError()
    processed = pipeline.process(text, error)

    if error.occurred():
        raise RuntimeError(f"Chyba UDPipe: {error.message}")

    pairs = []
    for line in processed.split("\n"):
        if line.startswith("#") or not line.strip():
            continue
        cols = line.split("\t")
        if len(cols) > 2:
            word = cols[1].lower()
            lemma = cols[2].lower()
            pairs.append((word, lemma))

    return pairs


# ============================================================
# ČIŠTĚNÍ TEXTU
# ============================================================

def remove_gutenberg_metadata(text: str) -> str:
    """Odstraní metadata a licenční texty na začátku a konci knih z Project Gutenberg."""
    start_pattern = re.compile(r"\*\*\*\s*START OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE)
    end_pattern = re.compile(r"\*\*\*\s*END OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE)
    
    start_match = start_pattern.search(text)
    end_match = end_pattern.search(text)
    
    if start_match and end_match:
        start_idx = start_match.end()
        end_idx = end_match.start()
        if start_idx < end_idx:
            return text[start_idx:end_idx]
            
    return text

def get_cleaned_text(text: str) -> str:
    text = remove_gutenberg_metadata(text)
    text = text.lower()
    # Zachovej základní latinku + česká a anglická diakritika
    text = re.sub(r"[^a-z\u00c0-\u024f\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ============================================================
# PŘEDZPRACOVÁNÍ spaCy
# ============================================================

def extract_raw_tokens_spacy(text: str, nlp, language: str) -> list[tuple[str, str]]:
    if nlp is None:
        st.error(f"Model spaCy pro jazyk '{language}' není nainstalován. Přepni vlevo na UDPipe.")
        st.stop()

    # Zamezení pádu u velmi dlouhých textů ve spaCy
    nlp.max_length = len(text) + 1
    
    with nlp.select_pipes(disable=["parser", "ner"]):
        doc = nlp(text)
        
    pairs = []
    for token in doc:
        if token.is_punct or token.is_space:
            continue
        pairs.append((token.text.lower(), token.lemma_.lower()))

    return pairs


# ============================================================
# PŘEDZPRACOVÁNÍ UDPipe
# ============================================================

def extract_raw_tokens_udpipe(text: str, udpipe_model) -> list[tuple[str, str]]:
    CHUNK_SIZE = 50_000
    chunks = [text[i:i+CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
    progress = st.progress(0, text="Zpracovávám text (UDPipe)...")
    
    pairs = []
    for i, chunk in enumerate(chunks):
        pairs.extend(udpipe_lemmatize(chunk, udpipe_model))
        progress.progress((i + 1) / len(chunks), text=f"UDPipe zpracování: část {i+1}/{len(chunks)}")
    progress.empty()

    return pairs


# ============================================================
# ZIPFOVA ANALÝZA
# ============================================================

def build_frequency_table(tokens: tuple[str, ...]) -> pd.DataFrame:
    counter = Counter(tokens)
    df = pd.DataFrame(counter.items(), columns=["slovo", "četnost"])
    df = df.sort_values(by="četnost", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    df["log(rank)"] = np.log10(df["rank"])
    df["log(četnost)"] = np.log10(df["četnost"])
    return df


def fit_zipf(df: pd.DataFrame):
    return linregress(df["log(rank)"], df["log(četnost)"])


def compute_zipf_constant(df: pd.DataFrame) -> float:
    """Konstanta k = f × r (1. Zipfův zákon) — medián přes všechna slova."""
    return float(np.median(df["četnost"] * df["rank"]))


def compute_second_zipf_constant(df: pd.DataFrame) -> float:
    """Konstanta k₂ = a × f² (2. Zipfův zákon) — medián přes frekvenční hladiny f ≤ 50.

    Čím více je k₂ konstantní napříč hladinami, tím lépe zákon platí.
    Medián je robustní vůči extrémním hodnotám na okrajích distribuce.
    """
    freq_counts = df["četnost"].value_counts().reset_index()
    freq_counts.columns = ["f", "a"]
    freq_counts = freq_counts[freq_counts["f"] <= 50]
    k2_values = freq_counts["a"] * freq_counts["f"] ** 2
    return float(np.median(k2_values))


def compute_ttr(total_words: int, vocab_size: int) -> float:
    """Type-Token Ratio — míra lexikální bohatosti textu."""
    return vocab_size / total_words if total_words > 0 else 0.0


# ============================================================
# VIZUALIZACE
# ============================================================

def build_frequency_fig(df: pd.DataFrame) -> go.Figure:
    """Graf četnosti: rank vs. četnost (prvních FREQ_GRAPH_LIMIT slov)."""
    fig = px.line(
        df.head(FREQ_GRAPH_LIMIT),
        x="rank",
        y="četnost",
        title=f"Graf četnosti: rank vs. četnost (prvních {FREQ_GRAPH_LIMIT} slov)",
        labels={"rank": "Rank (pořadí)", "četnost": "Četnost (f)"},
    )
    fig.update_layout(height=500)
    return fig


def build_zipf_fig(df: pd.DataFrame, regression) -> go.Figure:
    x = df["log(rank)"]
    y_fit = regression.intercept + regression.slope * x

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["log(rank)"],
        y=df["log(četnost)"],
        mode="markers",
        name="Slova",
        marker=dict(size=4, opacity=0.6)
    ))
    fig.add_trace(go.Scatter(
        x=x,
        y=y_fit,
        mode="lines",
        name=f"Regresní přímka (sklon = {regression.slope:.3f})",
        line=dict(color="crimson", width=2)
    ))
    fig.update_layout(
        title="Log-log graf (1. Zipfův zákon): log(rank) vs. log(četnost)",
        xaxis_title="log(rank) — logaritmus pořadí",
        yaxis_title="log(četnost) — logaritmus četnosti",
        height=600
    )
    return fig


def build_second_zipf_fig(df: pd.DataFrame) -> go.Figure:
    """2. Zipfův zákon: počet slov se stejnou četností vs. četnost."""
    freq_counts = df["četnost"].value_counts().reset_index()
    freq_counts.columns = ["cetnost_f", "pocet_slov_a"]
    freq_counts = freq_counts.sort_values("cetnost_f")
    freq_counts = freq_counts[freq_counts["cetnost_f"] <= 50]

    fig = px.bar(
        freq_counts,
        x="cetnost_f",
        y="pocet_slov_a",
        title="Graf četnostní distribuce (2. Zipfův zákon): počet slov na dané frekvenční hladině",
        labels={
            "cetnost_f": "Frekvenční hladina (f)",
            "pocet_slov_a": "Počet slov s danou četností (a)"
        }
    )
    fig.update_layout(height=450)
    return fig


def build_second_zipf_verification_fig(df: pd.DataFrame) -> go.Figure:
    """Ověření 2. Zipfova zákona: zobrazuje součin a × f² — měl by být přibližně konstantní."""
    freq_counts = df["četnost"].value_counts().reset_index()
    freq_counts.columns = ["f", "a"]
    freq_counts = freq_counts.sort_values("f")
    freq_counts = freq_counts[freq_counts["f"] <= 50]
    freq_counts["af2"] = freq_counts["a"] * freq_counts["f"] ** 2

    fig = px.scatter(
        freq_counts,
        x="f",
        y="af2",
        title="Ověření 2. Zipfova zákona: součin a × f² by měl být přibližně konstantní",
        labels={
            "f": "Frekvenční hladina (f)",
            "af2": "Součin a × f²"
        }
    )
    fig.update_traces(marker=dict(size=6, opacity=0.8))
    fig.update_layout(height=400)
    return fig


# ============================================================
# TEORIE — sdílený expander
# ============================================================

def render_theory(expanded: bool = False) -> None:
    with st.expander("📚 Teorie: Tři Zipfovy zákony", expanded=expanded):
        st.markdown('''
**Zipfovy zákony** jsou formulací základních vztahů mezi frekvencí jednotky a její distribucí v jazyce.
Ačkoli uplatňování Zipfových zákonů nemá povahu exaktních kvantitativních zákonitostí *(spíše než o zákonu
bychom měli mluvit o empirické pravidelnosti)*, na jejichž základě by bylo možné (bez dodatečných úprav)
předvídat hodnoty, které u reálných textů skutečně naměříme, poskytují Zipfovy zákony adekvátní deskriptivní
rámec pro popis rozložení četnosti v populaci (téměř libovolných) jednotek jazyka.
Typicky tak Zipfovy zákony neplatí pro slova nejfrekventovanější a nejméně frekventovaná.

### 1. Zipfův zákon
Nejpoužívanější a nejznámější ze Zipfových zákonů lze formalizovat vzorcem:

**$f \\times r = k$**

kde $f$ je **četnost** slova (frequency), $r$ je jeho **rank** (pořadí) a $k$ je konstanta.
Četnost slova je tedy nepřímo úměrná jeho ranku. Vztah vychází z předpokladu, že existuje tendence po ustavení
rovnováhy mezi počtem slov v jazyce (rozrůzněnost jazyka) a jejich četností (jazyková ekonomie).
Důsledkem tohoto vztahu je fakt, že každý text obsahuje velmi malý počet slov frekventovaných a většinu slov
málo frekventovaných *(viz podíl hapaxů na celkovém počtu typů)*.

### 2. Zipfův zákon
Vztah mezi počtem slov se stejnou četností a touto četností vyjadřil Zipf takto:

**$a \\times f^2 = k$**

kde $a$ je **počet slov s četností** $f$ a $k$ je konstanta.
Čím vyšší frekvenční hladinu zkoumáme, tím méně slov na ní najdeme (přičemž úbytek není lineární).

### 3. Zipfův zákon
Poslední Zipfův zákon se týká vztahu mezi četností slova a počtem jeho významů:

**$m / \\sqrt{f} = k$**

kde $m$ je **počet významů** slova o četnosti $f$ a $k$ je konstanta.
Tento vztah se dá nejobtížněji empiricky ověřit, protože parcelace (rozdělování) významů je vždy značně
subjektivní. Principiálně tento zákon vypovídá o tom, že slova s nejvyšší četností bývají často
**polysémní** (mají více významů), zatímco slova z nižších frekvenčních pásem mají často jen jeden význam.
        ''')
        st.markdown("*— Zdroj: [wiki.korpus.cz — Zipfův zákon](https://wiki.korpus.cz/doku.php/pojmy:zipf)*")
        st.markdown(
            "Většina nástrojů vám řekne, kolik má váš text slov. Náš nástroj vám řekne, zda má váš text duši.\n\n"
            "Pomocí lingvistických Zipfových zákonů a pokročilých algoritmů analyzujeme matematickou "
            "strukturu vašeho obsahu. Během několika sekund zjistíte, zda je text optimalizovaný přirozeně, "
            "zda netrpí chudou slovní zásobou a zda nebude vyhledávači vyhodnocen jako syntetický spam. "
            "Prodávejte kvalitní obsah, který je podložen tvrdými daty."
        )

# ============================================================
# SEKCE VÝSLEDKŮ
# ============================================================

def render_metrics(tokens: list[str], df: pd.DataFrame, regression, k_constant: float) -> None:
    total_words = len(tokens)
    vocab_size = len(df)
    r2 = regression.rvalue ** 2
    hapax_count = int((df["četnost"] == 1).sum())
    hapax_ratio = hapax_count / vocab_size * 100
    ttr = compute_ttr(total_words, vocab_size)

    st.subheader("📊 Základní statistiky textu")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Celkový počet slov (tokenů)", f"{total_words:,}")
    col2.metric("Velikost slovníku (typů)", f"{vocab_size:,}")
    col3.metric(
        "Podíl hapaxů (slov s f=1)", f"{hapax_ratio:.1f} %",
        help="Hapax legomenon = slovo vyskytující se v textu pouze jednou. Vysoký podíl hapaxů je důsledkem 1. Zipfova zákona."
    )
    col4.metric(
        "TTR (Type-Token Ratio)", f"{ttr:.4f}",
        help="Podíl počtu různých slov (typů) k celkovému počtu slov (tokenů). Vyšší hodnota = bohatší slovní zásoba."
    )
    col5.metric(
        "Sklon regresní přímky", f"{regression.slope:.3f}",
        help="Ideální Zipfův sklon je přibližně −1,0. Blízkost k −1 potvrzuje platnost 1. Zipfova zákona."
    )
    col6.metric(
        "R² (přesnost proložení)", f"{r2:.4f}",
        help="Hodnota R² blízká 1,0 znamená, že data odpovídají lineárnímu modelu v log-log prostoru."
    )


def render_first_zipf(df: pd.DataFrame, regression, k_constant: float) -> None:
    st.markdown("---")
    st.subheader("📈 1. Zipfův zákon: $f \\times r = k$")
    st.markdown(
        f"Konstanta **k** (medián součinu f × r) = **{k_constant:,.0f}** "
        f"— frekvence slova je nepřímo úměrná jeho ranku. "
        f"Sklon regresní přímky = **{regression.slope:.3f}** (ideál: −1,0)."
    )
    st.caption("ℹ️ Konstanta k vyjadřuje průměrný součin četnosti a ranku. Čím blíže je sklon k hodnotě −1, tím lépe text odpovídá Zipfově distribuci.")
    st.plotly_chart(build_frequency_fig(df), use_container_width=True)
    st.plotly_chart(build_zipf_fig(df, regression), use_container_width=True)


def render_second_zipf(df: pd.DataFrame) -> None:
    st.markdown("---")
    st.subheader("📉 2. Zipfův zákon: $a \\times f^2 = k$")
    st.markdown(
        "Čím vyšší frekvenční hladinu zkoumáme, tím méně slov na ní najdeme. "
        "Graf zobrazuje počet slov (**a**) pro každou frekvenční hladinu (**f**) — "
        "úbytek je nelineární."
    )

    k2 = compute_second_zipf_constant(df)
    st.markdown(
        f"Konstanta **k₂** (medián součinu a × f², pro f ≤ 50) = **{k2:,.1f}**. "
        f"Čím stabilnější je tento součin napříč frekvenčními hladinami, tím lépe zákon platí."
    )

    st.plotly_chart(build_second_zipf_fig(df), use_container_width=True)

    st.markdown(
        "**Ověření:** Pokud 2. Zipfův zákon platí, měl by být součin $a \\times f^2$ přibližně "
        "konstantní pro všechny frekvenční hladiny. Graf níže ukazuje, jak moc se tato hodnota mění."
    )
    st.plotly_chart(build_second_zipf_verification_fig(df), use_container_width=True)


def render_third_zipf(df: pd.DataFrame) -> None:
    st.markdown("---")
    st.subheader("📖 3. Zipfův zákon: $m / \\sqrt{f} = k$")
    st.markdown(
        "Třetí Zipfův zákon popisuje vztah mezi **četností slova** a **počtem jeho významů** ($m$). "
        "Slova s vysokou četností bývají často **polysémní** (mají více významů), "
        "zatímco málo frekventovaná slova mívají zpravidla jen jeden význam. "
    )
    st.info(
        "⚠️ **Tento zákon nelze empiricky ověřit pouze z textu.** "
        "Konstanta $k$ vyžaduje znalost skutečného počtu významů $m$ pro každé slovo, "
        "která pochází ze slovníku — ne z korpusu. "
        "Níže zobrazená křivka a odhady v tabulce jsou proto čistě **ilustrativní**: "
        "předpokládají $m = 1$ pro slova s nejnižší četností (hapax legomena) "
        "a z toho odvozují tvar křivky. Skutečné hodnoty $m$ se mohou výrazně lišit."
    )

    # Konstanta k₃ je odvozena z hapaxů, kde lze předpokládat m ≈ 1 (jedno-významová slova).
    # Ze vzorce m / √f = k → k = m / √f = 1 / √1 = 1 pro hapax (f=1, m≈1).
    k3 = 1.0

    freq_range = np.linspace(1, df["četnost"].max(), 500)
    m_values = k3 * np.sqrt(freq_range)

    df_third = pd.DataFrame({
        "Četnost slova (f)": freq_range,
        "Odhadovaný počet významů (m)": m_values
    })

    fig_third = px.line(
        df_third,
        x="Četnost slova (f)",
        y="Odhadovaný počet významů (m)",
        title="Ilustrativní křivka 3. Zipfova zákona: m / √f = k  (k = 1)",
    )
    fig_third.update_layout(height=400)
    st.plotly_chart(fig_third, use_container_width=True)

    # Tabulka: 20 nejfrekventovanějších slov s odhadem m
    # Odhad m = k₃ * √f; jde o ilustraci relativního pořadí, nikoliv absolutní hodnotu.
    df_polysemy = df.head(20)[["slovo", "četnost", "rank"]].copy()
    df_polysemy["odhad m (= k*√f, ilustrativní)"] = (k3 * np.sqrt(df_polysemy["četnost"])).round(2)
    st.markdown(
        "**20 nejfrekventovanějších slov s ilustrativním odhadem počtu významů** "
        "(dle 3. Zipfova zákona — skutečné hodnoty vyžadují slovník):"
    )
    st.dataframe(df_polysemy, use_container_width=True)


def render_interpretation(regression) -> None:
    st.markdown("---")
    st.subheader("🔍 Interpretace výsledků")
    slope = regression.slope
    if -1.2 < slope < -0.8:
        st.success(
            f"✅ Text **odpovídá** přirozené Zipfově distribuci. "
            f"Sklon {slope:.3f} je blízký ideální hodnotě −1,0, "
            f"což potvrzuje platnost 1. Zipfova zákona."
        )
    elif -1.5 < slope < -0.5:
        st.warning(
            f"⚠️ Text se **mírně odchyluje** od ideálního Zipfova rozložení. "
            f"Sklon {slope:.3f} je mimo rozmezí ⟨−1,2; −0,8⟩. "
            f"Důvodem může být odstranění stop slov, lemmatizace nebo OCR chyby."
        )
    else:
        st.error(
            f"❌ Text se **výrazně odchyluje** od Zipfovy distribuce. "
            f"Sklon {slope:.3f} je velmi vzdálený od hodnoty −1,0."
        )


# ============================================================
# DASHBOARD — LEVÝ PANEL
# ============================================================

st.sidebar.header("⚙️ Nastavení")

language = st.sidebar.selectbox(
    "Jazyk textu",
    ["Čeština", "Angličtina"],
    index=1
)

# (stemmer definován uvnitř process_and_analyze pro cachování)

st.sidebar.subheader("Způsob načtení textu")
text_source_type = st.sidebar.radio(
    "Vyberte zdroj textu",
    ["Nahrát ze souboru", "Načíst z URL (Gutenberg apod.)"]
)

uploaded_file = None
url_input = ""
load_url_btn = False

if text_source_type == "Nahrát ze souboru":
    uploaded_file = st.sidebar.file_uploader(
        "Nahrát soubor TXT nebo MD",
        type=["txt", "md"]
    )
else:
    url_input = st.sidebar.text_input(
        "URL adresa k TXT souboru",
        placeholder="https://www.gutenberg.org/cache/epub/5200/pg5200.txt"
    )
    load_url_btn = st.sidebar.button("☁️ Stáhnout text z URL", use_container_width=True)

st.sidebar.subheader("Předzpracování textu")

remove_stopwords = st.sidebar.checkbox(
    "Odstranit stop slova (funkční slova)",
    value=False,
    help="Stop slova jsou velmi frekventovaná funkční slova (spojky, předložky, zájmena), která nemají lexikální obsah."
)

use_lemmatization = st.sidebar.checkbox(
    "Lemmatizace (převod na základní tvar)",
    value=False,
    help="Lemmatizace sjednotí různé tvary jednoho slova (např. 'byl, jsem, budou' → 'být'), čímž se zvýší četnost lemmatu."
)

use_stemming = st.sidebar.checkbox(
    "Stemming (odtržení přípon)",
    value=False,
    help="Stemming hrubě ořízne přípony slov. Pro češtinu není plně podporován."
)

# Metoda lemmatizace — spaCy jen pokud je model k dispozici
lemmatization_options = ["UDPipe", "spaCy"] if spacy_available(language) else ["UDPipe"]

default_index = 0
if language == "Angličtina" and "spaCy" in lemmatization_options:
    default_index = lemmatization_options.index("spaCy")

method = st.sidebar.selectbox(
    "Nástroj lemmatizace",
    lemmatization_options,
    index=default_index,
    help="UDPipe je doporučený nástroj pro češtinu. pro angličtinu je výchozí spaCy."
)

# Cesta k modelu UDPipe — zobrazena vždy při výběru UDPipe (ne až po nahrání souboru)
if method == "UDPipe":
    model_path = st.sidebar.text_input(
        "Cesta k modelu UDPipe",
        UDPIPE_DEFAULTS.get(language, ""),
        help="Zadejte absolutní nebo relativní cestu k souboru .udpipe."
    )
else:
    model_path = None

top_n = st.sidebar.slider(
    "Počet zobrazených slov v tabulce",
    10, 100, 50
)

st.sidebar.divider()
st.sidebar.subheader("🛠️ Akce")

if st.sidebar.button("🔄 Přepočítat", use_container_width=True):
    st.rerun()

if st.sidebar.button("🗑️ Vymazat cache", use_container_width=True):
    st.cache_resource.clear()
    if hasattr(st, "cache_data"):
        st.cache_data.clear()
    # Vymažeme i načtený URL text
    if "cached_url_text" in st.session_state:
        del st.session_state["cached_url_text"]
    st.rerun()

if st.sidebar.button("🖨️ Tisknout", use_container_width=True):
    import streamlit.components.v1 as components
    components.html("<script>window.parent.print();</script>", height=0)


# ============================================================
# NAČTENÍ TEXTU
# ============================================================

text = None

if "cached_url_text" not in st.session_state:
    st.session_state.cached_url_text = None

if text_source_type == "Nahrát ze souboru":
    if uploaded_file:
        raw = uploaded_file.getvalue()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("latin-1")
            st.warning("⚠️ Soubor nebyl v kódování UTF-8 — použita náhradní znaková sada latin-1. "
                       "Výsledky mohou obsahovat drobné znakové odchylky.")
        st.session_state.source_name = uploaded_file.name
elif text_source_type == "Načíst z URL (Gutenberg apod.)":
    if load_url_btn and url_input.strip():
        with st.spinner("Stahuji text z URL, prosím čekejte..."):
            try:
                response = requests.get(url_input.strip(), timeout=15)
                response.raise_for_status()
                
                content_type = response.headers.get("Content-Type", "").lower()
                if "text" not in content_type:
                    raise ValueError(f"URL neobsahuje textový soubor (Content-Type: {content_type}).")

                response.encoding = response.apparent_encoding if response.encoding is None else response.encoding
                st.session_state.cached_url_text = response.text
                st.session_state.source_name = url_input.strip()
                st.sidebar.success("✅ Text úspěšně stažen!")
            except Exception as e:
                st.error(f"Nepodařilo se stáhnout text z dané URL. Zkontrolujte připojení a platnost odkazu. Chyba: {e}")
                st.session_state.cached_url_text = None
                st.stop()
    
    # Použijeme text z cache, pokud už byl někdy dříve úspěšně stažen
    if st.session_state.cached_url_text:
        text = st.session_state.cached_url_text

# Pokud text stále nemáme (uživatel nic nenahrál/nestáhl), zkusíme záložní načtení z ENV
if not text:
    env_path = os.environ.get("ZIPF_INPUT_FILE")
    if env_path:
        p = Path(env_path)
        if p.exists():
            text = p.read_text(encoding="utf-8")
            st.sidebar.success(f"Automaticky načteno: {p.name}")
            st.session_state.source_name = p.name
        else:
            st.sidebar.error(f"Soubor z ZIPF_INPUT_FILE nenalezen: {env_path}")


# ============================================================
# HLAVNÍ ČÁST — ZPRACOVÁNÍ A VÝSTUP
# ============================================================

@st.cache_data(show_spinner=False)
def get_base_tokens(text: str, method: str, language: str, model_path: str | None) -> list[tuple[str, str]]:
    text = get_cleaned_text(text)
    
    if method == "spaCy":
        nlp = load_spacy_model(language)
        if nlp is None:
            raise RuntimeError(f"Model spaCy pro jazyk '{language}' není nainstalován.")
        return extract_raw_tokens_spacy(text, nlp, language)
    else:
        try:
            udpipe_model = load_udpipe_model(model_path)
        except Exception as e:
            raise RuntimeError(str(e))
        return extract_raw_tokens_udpipe(text, udpipe_model)

@st.cache_data(show_spinner=False)
def filter_and_analyze(
    base_tokens: list[tuple[str, str]],
    language: str,
    remove_stopwords: bool,
    use_lemma: bool,
    use_stemming: bool
):
    try:
        from nltk.stem.snowball import SnowballStemmer
        stemmer = SnowballStemmer("english" if language == "Angličtina" else "czech")
    except ValueError:
        stemmer = None

    stop_words = STOPWORDS.get(language, set()) if remove_stopwords else set()
    
    filtered_tokens = []
    for word, lemma in base_tokens:
        token = lemma if use_lemma else word
        if len(token) < 2:
            continue
        if remove_stopwords and token in stop_words:
            continue
        
        if use_stemming and stemmer:
            token = stemmer.stem(token)
            
        filtered_tokens.append(token)
        
    df = build_frequency_table(tuple(filtered_tokens))
    regression = fit_zipf(df)
    k_constant = compute_zipf_constant(df)
    
    return filtered_tokens, df, regression, k_constant


if text:
    st.success("✅ Text byl úspěšně načten.")
    source = st.session_state.get("source_name", "Neznámý zdroj")
    st.caption(f"**Zdroj:** {source}")

    with st.spinner("Probíhá zpracování textu (caching aktivní), prosím čekejte..."):
        try:
            base_tokens = get_base_tokens(
                text=text,
                method=method,
                language=language,
                model_path=model_path
            )
            tokens, df, regression, k_constant = filter_and_analyze(
                base_tokens=base_tokens,
                language=language,
                remove_stopwords=remove_stopwords,
                use_lemma=use_lemmatization,
                use_stemming=use_stemming
            )
        except RuntimeError as e:
            st.error(str(e))
            st.stop()

    if not tokens:
        st.error("❌ Po předzpracování nezbyly žádné tokeny. Zkontroluj nastavení (stop slova, délka slov).")
        st.stop()

    # ---- Výstup ----
    render_metrics(tokens, df, regression, k_constant)
    
    render_interpretation(regression)

    st.markdown("---")
    st.subheader(f"📋 Tabulka {top_n} nejčetnějších slov")
    st.markdown(
        "Sloupce: **slovo** (lemma/tvar), **četnost** (f — absolutní počet výskytů), "
        "**rank** (r — pořadí podle četnosti), **log(rank)**, **log(četnost)**."
    )
    st.dataframe(df.head(top_n), use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Stáhnout výsledky jako CSV",
        data=csv,
        file_name="zipfova_analyza.csv",
        mime="text/csv"
    )

    render_first_zipf(df, regression, k_constant)
    render_second_zipf(df)
    render_third_zipf(df)

    st.markdown("---")
    render_theory(expanded=False)

else:
    st.info("👈 V levém menu nahrajte textový soubor knihy nebo vložte odkaz na knihu z projektu [www.gutenberg.org](https://www.gutenberg.org).")
    st.markdown("---")
    render_theory(expanded=True)

# Patička
st.markdown(
    "<div style='text-align: center; color: gray; margin-top: 50px; margin-bottom: 20px; font-size: 0.9em;'>"
    "Objednejte řešení ve vašem brandu na webu "
    "<a href='https://www.milospop.com' target='_blank' style='text-decoration: none;'>www.milospop.com</a>"
    "</div>",
    unsafe_allow_html=True
)

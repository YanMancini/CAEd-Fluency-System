"""Leitura do dataset bruto e fragmentação em instâncias (chunks).

Cada subpasta de data/raw/ é um nível de complexidade (rótulo).
Cada arquivo .txt é um trecho de um mesmo material didático, por isso a
divisão treino/validação/teste é feita POR ARQUIVO (evita vazamento de
dados entre chunks do mesmo material).
"""
import os
import random
import re

import pandas as pd

# Pasta -> rótulo
LABELS = {
    "1_Ensino_Fundamental_I": "Ensino Fundamental I",
    "2_Ensino_Fundamental_II": "Ensino Fundamental II",
    "3_Ensino_Medio": "Ensino Médio",
    "4_Ensino_Superior": "Ensino Superior",
}
LABEL_ORDER = list(LABELS.values())

ABREVIACOES = {"sr", "sra", "dr", "dra", "prof", "profa", "etc", "ex", "obs", "av", "pág", "pag", "cap", "fig", "St", "sta"}
_FIM_SENT = re.compile(r"(?<=[.!?…])\s+(?=[\"“«\(\[—–-]?[A-ZÁÉÍÓÚÂÊÔÃÕÇ0-9])")
_WORD = re.compile(r"\w+", re.UNICODE)


def split_sentences(texto):
    """Divide um texto em sentenças (regex simples + proteção de abreviações)."""
    partes = _FIM_SENT.split(texto)
    sentencas = []
    for p in partes:
        p = p.strip()
        if not p:
            continue
        if sentencas:
            ultima_palavra = re.findall(r"(\w+)\.$", sentencas[-1])
            if ultima_palavra and ultima_palavra[0].lower() in ABREVIACOES:
                sentencas[-1] += " " + p
                continue
        sentencas.append(p)
    return sentencas


def clean_text(raw):
    """Remove linhas que são títulos/caminhos de navegação (curtas e sem pontuação final)."""
    linhas = []
    for linha in raw.splitlines():
        linha = re.sub(r"\s+", " ", linha).strip()
        if not linha:
            continue
        n_pal = len(_WORD.findall(linha))
        termina = linha[-1] in ".!?…\"”:;)"
        if n_pal < 6 and not termina:
            continue  # título / breadcrumb
        if "/" in linha and not termina and n_pal < 12:
            continue
        linhas.append(linha)
    return " ".join(linhas)


def chunk_text(texto, max_palavras=80, min_palavras=20):
    """Janelas de sentenças consecutivas com no máximo `max_palavras` palavras."""
    chunks, atual, n = [], [], 0
    for s in split_sentences(texto):
        w = len(_WORD.findall(s))
        if atual and n + w > max_palavras:
            if n >= min_palavras:
                chunks.append(" ".join(atual))
            atual, n = [], 0
        atual.append(s)
        n += w
    if atual and n >= min_palavras:
        chunks.append(" ".join(atual))
    return chunks


def load_instances(raw_dir, max_palavras=80, min_palavras=20):
    """Varre raw_dir e devolve DataFrame [texto, label, arquivo]."""
    linhas = []
    for pasta, label in LABELS.items():
        d = os.path.join(raw_dir, pasta)
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Pasta não encontrada: {d}")
        for nome in sorted(os.listdir(d)):
            if not nome.endswith(".txt"):
                continue
            caminho = os.path.join(d, nome)
            with open(caminho, encoding="utf-8", errors="ignore") as f:
                texto = clean_text(f.read())
            for c in chunk_text(texto, max_palavras, min_palavras):
                linhas.append({"texto": c, "label": label, "arquivo": f"{pasta}/{nome}"})
    return pd.DataFrame(linhas)


def split_by_file(df, val=0.15, test=0.15, seed=42):
    """Divisão estratificada por arquivo: todos os chunks de um arquivo ficam no mesmo conjunto."""
    rng = random.Random(seed)
    destino = {}
    for label, g in df.groupby("label"):
        arquivos = sorted(g["arquivo"].unique())
        rng.shuffle(arquivos)
        n = len(arquivos)
        n_test, n_val = int(round(n * test)), int(round(n * val))
        for i, a in enumerate(arquivos):
            destino[a] = "test" if i < n_test else ("val" if i < n_test + n_val else "train")
    df = df.copy()
    df["split"] = df["arquivo"].map(destino)
    return df


def build_processed(raw_dir, out_dir, **kw):
    """Gera data/processed/{train,val,test}.csv e devolve o DataFrame completo."""
    df = split_by_file(load_instances(raw_dir, **kw))
    os.makedirs(out_dir, exist_ok=True)
    for s in ("train", "val", "test"):
        df[df["split"] == s].to_csv(os.path.join(out_dir, f"{s}.csv"), index=False)
    return df


if __name__ == "__main__":
    base = os.path.join(os.path.dirname(__file__), "..", "data")
    d = build_processed(os.path.join(base, "raw"), os.path.join(base, "processed"))
    print(d.groupby(["split", "label"]).size().unstack())

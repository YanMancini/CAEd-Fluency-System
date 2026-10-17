"""Fluxo integrado: texto -> mutação -> modelo -> impacto na complexidade prevista."""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .loader import LABEL_ORDER
from .model import predict_label, predict_score
from .mutations import OPERADORES, apply_mutation

OPS = list(OPERADORES) + ["todas"]


def mutate(nome, texto, seed=None):
    """Aplica um operador, ou 'todas' (palavras -> gênero -> paráfrase, em sequência)."""
    if nome != "todas":
        return apply_mutation(nome, texto, seed)
    mudancas = []
    for op in OPERADORES:
        texto, m = apply_mutation(op, texto, seed)
        mudancas += m
    return texto, mudancas


def analyze_text(modelo, texto, seed=42):
    """Para um texto: classe prevista original e depois de cada mutação."""
    base = {"classe": predict_label(modelo, texto), "score": predict_score(modelo, texto)}
    saida = {"original": base, "mutacoes": {}}
    for op in OPS:
        novo, m = mutate(op, texto, seed)
        saida["mutacoes"][op] = {
            "texto": novo, "mudancas": m, "alterou": novo != texto,
            "classe": predict_label(modelo, novo), "score": predict_score(modelo, novo),
        }
    return saida


def evaluate_impact(modelo, df, outputs_dir, seed=42, max_textos=None, log=print):
    """Aplica cada operador a todos os textos de `df` (coluna texto/label) e mede o impacto."""
    if max_textos and len(df) > max_textos:
        df = df.sample(max_textos, random_state=seed)
    linhas, exemplos = [], []
    for k, (_, r) in enumerate(df.iterrows()):
        t = r["texto"]
        c0, s0 = predict_label(modelo, t), predict_score(modelo, t)
        for op in OPS:
            novo, m = mutate(op, t, seed + k)
            if novo == t:
                c1, s1 = c0, s0
            else:
                c1, s1 = predict_label(modelo, novo), predict_score(modelo, novo)
            linhas.append({"operador": op, "label": r["label"], "alterou": novo != t,
                           "classe_antes": c0, "classe_depois": c1,
                           "score_antes": s0, "score_depois": s1})
            if novo != t and c0 != c1 and len(exemplos) < 40:
                exemplos.append({"operador": op, "antes": t, "depois": novo, "classe_antes": c0,
                                 "classe_depois": c1, "mudancas": m[:6]})
    res = pd.DataFrame(linhas)
    os.makedirs(outputs_dir, exist_ok=True)
    res.to_csv(os.path.join(outputs_dir, "impact_results.csv"), index=False)
    with open(os.path.join(outputs_dir, "impact_examples.json"), "w", encoding="utf-8") as f:
        json.dump(exemplos, f, ensure_ascii=False, indent=2)

    idx = {l: i for i, l in enumerate(LABEL_ORDER)}
    resumo = []
    for op, g in res.groupby("operador", sort=False):
        a = g[g["alterou"]]
        d_cls = a["classe_depois"].map(idx) - a["classe_antes"].map(idx)
        resumo.append({
            "operador": op,
            "textos": len(g),
            "textos_alterados": int(len(a)),
            "porcentagem_alterados": round(100 * len(a) / len(g), 1),
            "porcentagem_mudou_classe(entre alterados)": round(100 * (d_cls != 0).mean(), 1) if len(a) else 0.0,
            "subiu_nivel": int((d_cls > 0).sum()),
            "desceu_nivel": int((d_cls < 0).sum()),
            "delta_score_medio": round(float((a["score_depois"] - a["score_antes"]).mean()), 4) if len(a) else 0.0,
            "delta_score_abs_medio": round(float((a["score_depois"] - a["score_antes"]).abs().mean()), 4) if len(a) else 0.0,
        })
    resumo = pd.DataFrame(resumo)
    resumo.to_csv(os.path.join(outputs_dir, "impact_summary.csv"), index=False)
    log("\n=== Impacto das mutações na classe prevista ===")
    log(resumo.to_string(index=False))

    ax = resumo.set_index("operador")["porcentagem_mudou_classe(entre alterados)"].plot.bar(rot=20, color="#4c72b0")
    ax.set_ylabel("% de textos alterados cuja classe mudou")
    ax.set_title("Impacto das mutações na complexidade prevista")
    plt.tight_layout()
    plt.savefig(os.path.join(outputs_dir, "impact_summary.png"), dpi=120)
    plt.close()
    return res, resumo

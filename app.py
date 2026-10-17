"""Interface web (Flask) para o sistema de complexidade textual do CAEd.

Uso:
    python app.py            # http://127.0.0.1:5000
    FLASK_PORT=8080 python app.py

Rotas:
    GET  /            -> página única (templates/index.html)
    GET  /api/saude   -> verifica se o modelo está carregado
    POST /api/analisar -> {"texto": "...", "seed": 42}
"""
import difflib
import os
import re

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

from src.features import features_dict
from src.loader import LABEL_ORDER
from src.model import TEXT_COL, load_model
from src.pipeline import OPS, mutate

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "outputs")
MAX_CHARS = 5000

app = Flask(__name__)
app.json.ensure_ascii = False  # acentos legíveis no JSON

_modelo = None


def modelo():
    """Carrega o modelo uma única vez (lazy)."""
    global _modelo
    if _modelo is None:
        _modelo = load_model(OUT)
    return _modelo


def probabilidades(pipe, texto):
    """Probabilidade por classe, na ordem Fund. I -> Superior (softmax das margens do SVM)."""
    X = pd.DataFrame({TEXT_COL: [texto]})
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        p = pipe.predict_proba(X)[0]
    else:
        d = pipe.decision_function(X)[0]
        e = np.exp(d - d.max())
        p = e / e.sum()
    ordem = [list(clf.classes_).index(l) for l in LABEL_ORDER]
    return p[ordem]


def avaliar(pipe, texto):
    p = probabilidades(pipe, texto)
    return {
        "classe": LABEL_ORDER[int(p.argmax())],
        "score": float(np.dot(p, np.arange(len(LABEL_ORDER)))),
        "probs": {l: round(float(v), 4) for l, v in zip(LABEL_ORDER, p)},
    }


_TOKEN = re.compile(r"\s+|[^\s]+")


def diff_tokens(antes, depois):
    """Segmentos do texto mutado marcados como alterados (diff por palavra)."""
    a, b = _TOKEN.findall(antes), _TOKEN.findall(depois)
    segs = []
    for op, _, _, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        trecho = "".join(b[j1:j2])
        if trecho:
            segs.append({"t": trecho, "mudou": op != "equal"})
    return segs


@app.get("/")
def index():
    return render_template("index.html", labels=LABEL_ORDER)


@app.get("/api/saude")
def saude():
    try:
        modelo()
        return jsonify(ok=True)
    except Exception as e:  # modelo ausente/corrompido
        return jsonify(ok=False, erro=str(e)), 503


@app.post("/api/analisar")
def analisar():
    dados = request.get_json(silent=True) or {}
    texto = (dados.get("texto") or "").strip()
    if len(texto.split()) < 3:
        return jsonify(erro="Digite um texto com pelo menos 3 palavras."), 400
    if len(texto) > MAX_CHARS:
        return jsonify(erro=f"Texto muito longo (máximo {MAX_CHARS} caracteres)."), 400
    try:
        seed = int(dados.get("seed", 42))
    except (TypeError, ValueError):
        seed = 42

    pipe = modelo()
    original = avaliar(pipe, texto)
    original["atributos"] = {k: round(float(v), 3) for k, v in features_dict(texto).items()}

    mutacoes = {}
    for op in OPS:
        novo, mudancas = mutate(op, texto, seed)
        alterou = novo != texto
        r = avaliar(pipe, novo) if alterou else dict(original)
        r.pop("atributos", None)
        r.update(texto=novo, alterou=alterou, mudancas=mudancas,
                 segmentos=diff_tokens(texto, novo) if alterou else [],
                 delta=round(r["score"] - original["score"], 4),
                 mudou_classe=r["classe"] != original["classe"])
        mutacoes[op] = r

    return jsonify(labels=LABEL_ORDER, original=original, mutacoes=mutacoes)


if __name__ == "__main__":
    app.run(host=os.environ.get("FLASK_HOST", "127.0.0.1"),
            port=int(os.environ.get("FLASK_PORT", 5000)),
            debug=os.environ.get("FLASK_DEBUG") == "1")

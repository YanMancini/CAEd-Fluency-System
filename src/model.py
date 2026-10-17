"""Treinamento, validação e exportação do modelo de complexidade textual.

Representação: TF-IDF (palavras, 1-2 gramas) + atributos estatísticos
(incluindo canonicidade), padronizados e concatenados.
Classificadores comparados: Regressão Logística, Random Forest e SVM linear.
Seleção do melhor modelo pelo F1-macro no conjunto de VALIDAÇÃO; o relatório
final é calculado no conjunto de TESTE (arquivos nunca vistos no treino).
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import LinearSVC

from .features import features_matrix
from .loader import LABEL_ORDER

TEXT_COL = "texto"


def _feats(serie):
    return features_matrix(list(serie))


def build_pipeline(clf):
    pre = ColumnTransformer([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=30000,
                                  sublinear_tf=True, lowercase=True), TEXT_COL),
        ("stats", Pipeline([("f", FunctionTransformer(_feats)), ("sc", StandardScaler())]), TEXT_COL),
    ])
    return Pipeline([("pre", pre), ("clf", clf)])


def candidate_models():
    return {
        "LogisticRegression": LogisticRegression(max_iter=3000, C=3.0, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                               n_jobs=-1, random_state=42),
        "LinearSVC": LinearSVC(C=0.3, class_weight="balanced", max_iter=20000, random_state=42),
    }


def load_splits(processed_dir):
    return {s: pd.read_csv(os.path.join(processed_dir, f"{s}.csv")) for s in ("train", "val", "test")}


def train_and_select(processed_dir, outputs_dir, log=print):
    """Treina todos os candidatos, escolhe o melhor, salva modelo e relatório."""
    sp = load_splits(processed_dir)
    tr, va, te = sp["train"], sp["val"], sp["test"]
    resultados, modelos = {}, {}
    for nome, clf in candidate_models().items():
        log(f"Treinando {nome} ...")
        pipe = build_pipeline(clf).fit(tr[[TEXT_COL]], tr["label"])
        pv = pipe.predict(va[[TEXT_COL]])
        resultados[nome] = {
            "val_accuracy": accuracy_score(va["label"], pv),
            "val_f1_macro": f1_score(va["label"], pv, average="macro"),
        }
        modelos[nome] = pipe
        log(f"  val acc={resultados[nome]['val_accuracy']:.3f}  f1-macro={resultados[nome]['val_f1_macro']:.3f}")

    melhor = max(resultados, key=lambda k: resultados[k]["val_f1_macro"])
    pipe = modelos[melhor]
    pt = pipe.predict(te[[TEXT_COL]])
    rel = classification_report(te["label"], pt, labels=LABEL_ORDER, digits=3)
    cm = confusion_matrix(te["label"], pt, labels=LABEL_ORDER)
    for nome, m in modelos.items():  # métricas de teste de todos, para o relatório
        p = m.predict(te[[TEXT_COL]])
        resultados[nome]["test_accuracy"] = accuracy_score(te["label"], p)
        resultados[nome]["test_f1_macro"] = f1_score(te["label"], p, average="macro")

    os.makedirs(outputs_dir, exist_ok=True)
    joblib.dump(pipe, os.path.join(outputs_dir, "model.joblib"))
    resumo = {"melhor_modelo": melhor, "resultados": resultados, "labels": LABEL_ORDER,
              "confusion_matrix_test": cm.tolist(),
              "n_instancias": {k: int(len(v)) for k, v in sp.items()}}
    with open(os.path.join(outputs_dir, "model_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(resumo, f, indent=2, ensure_ascii=False)
    with open(os.path.join(outputs_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"Modelo escolhido: {melhor}\n\n{rel}\nMatriz de confusão (linhas=real, colunas=previsto):\n")
        f.write(f"{LABEL_ORDER}\n{cm}\n")
    log(f"\nMelhor modelo (val F1-macro): {melhor}\n{rel}\n{cm}")
    return pipe, resumo


def load_model(outputs_dir):
    return joblib.load(os.path.join(outputs_dir, "model.joblib"))


def predict_label(pipe, texto):
    return pipe.predict(pd.DataFrame({TEXT_COL: [texto]}))[0]


def predict_score(pipe, texto):
    """Nível esperado contínuo (0=Fund. I ... 3=Superior) para medir o tamanho do impacto."""
    X = pd.DataFrame({TEXT_COL: [texto]})
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        p = pipe.predict_proba(X)[0]
    else:
        d = pipe.decision_function(X)[0]
        e = np.exp(d - d.max())
        p = e / e.sum()
    ordem = [list(clf.classes_).index(l) for l in LABEL_ORDER]
    return float(np.dot(p[ordem], np.arange(len(LABEL_ORDER))))

import argparse
import os
import sys

import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(BASE, "data", "raw")
PROC = os.path.join(BASE, "data", "processed")
OUT = os.path.join(BASE, "outputs")


def preparar():
    from src.loader import build_processed
    df = build_processed(RAW, PROC)
    print(df.groupby(["split", "label"]).size().unstack().to_string())


def treinar():
    from src.model import train_and_select
    train_and_select(PROC, OUT)


def avaliar(max_textos=None):
    from src.model import load_model
    from src.pipeline import evaluate_impact
    modelo = load_model(OUT)
    teste = pd.read_csv(os.path.join(PROC, "test.csv"))
    evaluate_impact(modelo, teste, OUT, max_textos=max_textos)


def texto(t):
    from src.features import features_dict
    from src.model import load_model
    from src.pipeline import analyze_text
    modelo = load_model(OUT)
    r = analyze_text(modelo, t)
    print("Atributos:", {k: round(v, 3) for k, v in features_dict(t).items()})
    print(f"\nORIGINAL  -> {r['original']['classe']} (nível esperado {r['original']['score']:.2f})\n{t}")
    for op, d in r["mutacoes"].items():
        print(f"\n[{op}] -> {d['classe']} (nível esperado {d['score']:.2f}){'' if d['alterou'] else '  (sem alteração)'}")
        if d["alterou"]:
            print(d["texto"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("comando", nargs="?", default="tudo",
                    choices=["preparar", "treinar", "avaliar", "tudo", "texto"])
    ap.add_argument("valor", nargs="?", help="texto (comando 'texto')")
    ap.add_argument("--max-textos", type=int, default=None, help="limita nº de textos na avaliação")
    a = ap.parse_args()
    if a.comando == "preparar":
        preparar()
    elif a.comando == "treinar":
        treinar()
    elif a.comando == "avaliar":
        avaliar(a.max_textos)
    elif a.comando == "texto":
        if not a.valor:
            sys.exit("Informe o texto: python main.py texto \"Seu texto aqui.\"")
        texto(a.valor)
    else:
        preparar()
        treinar()
        avaliar(a.max_textos)


if __name__ == "__main__":
    main()

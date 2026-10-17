import re

import numpy as np

VOGAIS = set("aeiouáéíóúâêôãõàü")
_WORD = re.compile(r"[^\W\d_]+(?:-[^\W\d_]+)*", re.UNICODE)  # palavras (com hífen)
_SENT_END = re.compile(r"[.!?…]+")

# (artigos, preposições, conjunções, pronomes, auxiliares comuns)
STOPWORDS = set("""
a o as os um uma uns umas de do da dos das em no na nos nas por pelo pela pelos pelas
para pra com sem sob sobre entre até ate ao aos à às e ou mas nem que se como quando
porque pois porém contudo todavia logo então assim já ainda também só mesmo mais menos
muito muita muitos muitas pouco pouca poucos poucas tão tanto tanta tantos tantas
eu tu ele ela nós vós eles elas me te lhe lhes nos vos mim ti si meu minha meus minhas
teu tua teus tuas seu sua seus suas nosso nossa nossos nossas este esta estes estas
esse essa esses essas aquele aquela aqueles aquelas isto isso aquilo qual quais quem
cujo cuja onde aqui ali lá cá não sim
é são era eram foi foram ser sendo sido está estão estava estavam estar esteve
há havia haver tem têm tinha tinham ter teve tenho tem temos
vai vão ia iam ir fui vou
""".split())


def tokenize_words(texto):
    return [w.lower() for w in _WORD.findall(texto)]


def count_sentences(texto):
    n = len([s for s in _SENT_END.split(texto) if s.strip()])
    return max(n, 1)


# ---------------------------------------------------------------- canonicidade
def _to_units(palavra):
    """Converte a palavra em lista de 'C'/'V'. Retorna None se houver símbolo estranho."""
    p = palavra.lower().replace("-", "")
    units, i, n = [], 0, len(p)
    while i < n:
        c = p[i]
        if c in VOGAIS:
            units.append("V")
            i += 1
        elif c.isalpha():
            dig = p[i:i + 2]
            if dig in ("ch", "lh", "nh", "rr", "ss"):
                units.append("C")
                i += 2
            elif dig in ("qu", "gu") and i + 2 < n and p[i + 2] in "eéêií":
                units.append("C")  # que, qui, gue, gui: 'u' mudo
                i += 2
            else:
                units.append("C")
                i += 1
        else:
            return None
    return units


def is_canonical(palavra):
    """True se a palavra só tem sílabas CV (sequência exata CVCV...CV)."""
    u = _to_units(palavra)
    if not u or len(u) % 2:
        return False
    return all(x == ("C" if k % 2 == 0 else "V") for k, x in enumerate(u))


def canonical_ratio(palavras):
    if not palavras:
        return 0.0
    return sum(is_canonical(w) for w in palavras) / len(palavras)


# ---------------------------------------------------------------- atributos
FEATURE_NAMES = [
    "pct_canonicas",       # % de palavras canônicas (CV)
    "tam_medio_palavra",   # nº médio de letras por palavra
    "tam_medio_frase",     # nº médio de palavras por frase
    "densidade_lexical",   # palavras de conteúdo / total de palavras
    "vocabulario",         # nº de palavras distintas (types)
    "ttr",                 # types / tokens
    "pct_palavras_longas", # % de palavras com >= 9 letras
    "virgulas_por_frase",  # proxy de subordinação/apostos
    "n_palavras",
]


def extract_features(texto):
    """Vetor de atributos estatísticos de um texto (ordem = FEATURE_NAMES)."""
    pal = tokenize_words(texto)
    n = len(pal)
    if n == 0:
        return [0.0] * len(FEATURE_NAMES)
    n_sent = count_sentences(texto)
    conteudo = [w for w in pal if w not in STOPWORDS]
    vocab = len(set(pal))
    return [
        canonical_ratio(pal),
        sum(len(w.replace("-", "")) for w in pal) / n,
        n / n_sent,
        len(conteudo) / n,
        float(vocab),
        vocab / n,
        sum(len(w) >= 9 for w in pal) / n,
        texto.count(",") / n_sent,
        float(n),
    ]


def features_matrix(textos):
    return np.array([extract_features(t) for t in textos], dtype=float)


def features_dict(texto):
    return dict(zip(FEATURE_NAMES, extract_features(texto)))


if __name__ == "__main__":
    for w in ["cabeça", "canela", "janela", "luta", "sopa", "abajur", "lápis", "cansado", "asa", "alheio"]:
        print(f"{w:10s} canônica={is_canonical(w)}")

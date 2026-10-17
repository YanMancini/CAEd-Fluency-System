"""Operadores de mutação textual baseados em heurísticas (sem redes neurais).

a) mutate_words      - troca palavras por sinônimos/antônimos (léxico próprio + flexão simples)
b) mutate_gender     - troca o gênero de frases que contêm substantivos de pessoa/animal
c) mutate_paraphrase - reescreve sentenças (adjunto temporal, aposto, voz passiva -> ativa)

Todos recebem um texto e retornam (novo_texto, lista_de_mudanças).
O parâmetro `seed` torna o resultado reprodutível.
Implementado em Python puro: não depende de NLTK/WordNet (que exigem download
de dados e têm cobertura limitada para o português).
"""
import random
import re
from collections import defaultdict

from .loader import split_sentences

_SPLIT = re.compile(r"([^\W\d_]+)", re.UNICODE)  # índices ímpares = palavras


def _tokens(texto):
    return _SPLIT.split(texto)


def _case_like(orig, new):
    if orig.isupper() and len(orig) > 1:
        return new.upper()
    if orig[:1].isupper():
        return new[:1].upper() + new[1:]
    return new


def _lower_first(s):
    return s[:1].lower() + s[1:]


def _upper_first(s):
    return s[:1].upper() + s[1:]


# ------------------------------------------------------------ morfologia simples
def _candidates(w, gmorph=True):
    """Possíveis (lema, feminino?, plural?) de uma forma de superfície."""
    yield w, False, False
    if w.endswith(("ões", "ães")):
        yield w[:-3] + "ão", False, True
    if len(w) > 3 and w[-3:] in ("ais", "eis", "ois", "uis"):
        yield w[:-2] + "l", False, True
    if len(w) > 3 and w.endswith("es") and w[-3] in "rz":
        yield w[:-2], False, True
    if len(w) > 3 and w.endswith("ns"):
        yield w[:-2] + "m", False, True
    if len(w) > 2 and w.endswith("s"):
        b = w[:-1]
        yield b, False, True
        if gmorph and b.endswith("a"):
            yield b[:-1] + "o", True, True
    if gmorph and len(w) > 2 and w.endswith("a"):
        yield w[:-1] + "o", True, False


def _inflect(base, fem, pl):
    if fem and base.endswith("o"):
        base = base[:-1] + "a"
    if pl:
        if base.endswith("ão"):
            base = base[:-2] + "ões"
        elif base.endswith("m"):
            base = base[:-1] + "ns"
        elif len(base) > 2 and base.endswith("l") and base[-2] in "aeou":
            base = base[:-1] + "is"
        elif base.endswith(("r", "z")):
            base += "es"
        elif not base.endswith("s"):
            base += "s"
    return base


# ------------------------------------------------------------ a) troca de palavras
SINONIMOS = [
    # adjetivos / advérbios
    ("bonito", "belo"), ("grande", "enorme"), ("pequeno", "reduzido"), ("feliz", "alegre"),
    ("triste", "abatido"), ("rápido", "veloz"), ("lento", "vagaroso"), ("forte", "robusto"),
    ("bravo", "zangado"), ("esperto", "inteligente"), ("difícil", "complicado"), ("fácil", "simples"),
    ("antigo", "velho"), ("importante", "essencial"), ("cansado", "exausto"), ("rico", "abastado"),
    ("engraçado", "divertido"), ("calmo", "tranquilo"), ("famoso", "célebre"), ("estranho", "esquisito"),
    ("escuro", "sombrio"), ("medroso", "receoso"), ("curioso", "interessado"),
    ("rapidamente", "depressa"), ("logo", "imediatamente"), ("muito", "bastante"),
    ("sempre", "constantemente"), ("talvez", "possivelmente"), ("depois", "posteriormente"),
    ("antes", "anteriormente"), 
    # substantivos (mesmo gênero)
    ("casa", "moradia"), ("carro", "automóvel"), ("menino", "garoto"), ("amigo", "companheiro"),
    ("medo", "pavor"), ("alegria", "felicidade"), ("tristeza", "melancolia"), ("caminho", "trajeto"),
    ("floresta", "mata"), ("barulho", "ruído"), ("história", "narrativa"), ("trabalho", "serviço"),
    ("problema", "impasse"), ("riqueza", "fortuna"), 
    ("viagem", "jornada"), ("tarefa", "atividade"), ("remédio", "medicamento"),
    ("festa", "celebração"), ("briga", "discussão"), ("dinheiro", "recurso"), ("bicho", "animal"),
    # verbos (formas flexionadas comuns)
    ("disse", "falou"), ("olhou", "fitou"), ("viu", "avistou"), ("comprou", "adquiriu"),
    ("começou", "iniciou"), ("terminou", "finalizou"), ("ajudou", "auxiliou"), ("queria", "desejava"),
    ("andou", "caminhou"), ("pegou", "apanhou"), ("gostava", "apreciava"), ("precisa", "necessita"),
    ("começar", "iniciar"), ("terminar", "finalizar"), ("ajudar", "auxiliar"), ("comprar", "adquirir"),
    ("ver", "enxergar"), ("voltou", "retornou"), ("morreu", "faleceu"), ("escolheu", "selecionou"),
    ("mostrou", "exibiu"), ("pediu", "solicitou"), ("respondeu", "retrucou"), ("perguntou", "indagou"),
    ("achou", "encontrou"), ("jogou", "lançou"), ("ganhou", "recebeu"), ("moram", "residem"),
]

ANTONIMOS = [
    ("grande", "pequeno"), ("bonito", "feio"), ("feliz", "triste"), ("rápido", "lento"),
    ("forte", "fraco"), ("alto", "baixo"), ("novo", "velho"), ("bom", "mau"), ("claro", "escuro"),
    ("fácil", "difícil"), ("quente", "frio"), ("cheio", "vazio"), ("rico", "pobre"),
    ("calmo", "agitado"), ("longe", "perto"), ("sempre", "nunca"),
    ("antes", "depois"), ("amor", "ódio"), ("alegria", "tristeza"),
    ("entrou", "saiu"), ("abriu", "fechou"), ("subiu", "desceu"), ("começou", "terminou"),
    ("ganhou", "perdeu"), ("riu", "chorou"), ("chegou", "partiu"), ("acordou", "dormiu"),
    ("amigo", "inimigo"), ("vida", "morte"), ("guerra", "paz"), ("verdade", "mentira"),
    ("gostava", "odiava"), ("gostou", "detestou"), ("lembrou", "esqueceu"), ("achou", "perdeu"),
]


def _build(pares):
    d = defaultdict(list)
    for a, b in pares:
        d[a].append(b)
        d[b].append(a)
    return d


_SIN, _ANT = _build(SINONIMOS), _build(ANTONIMOS)

# adjetivos-alvo que flexionam em gênero (-o/-a)
ADJ_FLEXIVEL = {"pequeno", "belo", "bonito", "robusto", "abatido", "zangado", "esperto", "antigo", "velho",
                "exausto", "abastado", "divertido", "tranquilo", "esquisito", "sombrio", "receoso", "interessado",
                "feio", "fraco", "novo", "alto", "baixo", "frio", "vazio", "cheio", "agitado", "lento",
                "vagaroso", "escuro", "claro", "complicado", "reduzido", "cansado", "calmo", "rico", "pobre",
                "bravo", "engraçado", "curioso", "medroso", "famoso", "estranho", "rápido"}
_DET_FEM = {"a", "as", "uma", "umas", "da", "das", "na", "nas", "essa", "essas", "esta", "estas", "minha",
            "minhas", "sua", "suas", "nossa", "nossas", "aquela", "aquelas", "pela", "pelas"}
_DET_MASC = {"o", "os", "um", "uns", "do", "dos", "no", "nos", "esse", "esses", "este", "estes", "meu",
             "meus", "seu", "seus", "nosso", "nossos", "aquele", "aqueles", "pelo", "pelos"}


def _infer_fem(toks, i):
    """Gênero do contexto: procura determinante nas até 6 palavras anteriores. None se não achar."""
    j = i - 2
    for _ in range(6):
        if j < 1:
            break
        w = toks[j].lower()
        if w in _DET_FEM:
            return True
        if w in _DET_MASC:
            return False
        j -= 2
    return None


def _lookup(w, tabela):
    wl = w.lower()
    for lema, fem, pl in _candidates(wl):
        if lema in tabela:
            return lema, fem, pl
    return None


def mutate_words(texto, mode="both", prob=0.4, seed=None):
    """Troca aleatoriamente ~`prob` das palavras elegíveis por sinônimos/antônimos.

    mode: 'synonym', 'antonym' ou 'both' (escolhe aleatoriamente por palavra).
    """
    rng = random.Random(seed)
    toks = _tokens(texto)
    elegiveis = []
    for i in range(1, len(toks), 2):
        tabelas = []
        if mode in ("synonym", "both") and _lookup(toks[i], _SIN):
            tabelas.append(("sinônimo", _SIN))
        if mode in ("antonym", "both") and _lookup(toks[i], _ANT):
            tabelas.append(("antônimo", _ANT))
        if tabelas:
            elegiveis.append((i, tabelas))
    if not elegiveis:
        return texto, []
    k = max(1, round(prob * len(elegiveis)))
    mudancas = []
    for i, tabelas in rng.sample(elegiveis, k):
        tipo, tab = rng.choice(tabelas)
        lema, fem, pl = _lookup(toks[i], tab)
        alvo = rng.choice(tab[lema])
        if alvo in ADJ_FLEXIVEL and not lema.endswith("o"):
            # origem invariável (ex.: grande): o gênero do alvo vem do contexto
            ctx = _infer_fem(toks, i)
            if ctx is None:
                continue
            fem = ctx
        novo = _case_like(toks[i], _inflect(alvo, fem, pl))
        mudancas.append(f"{tipo}: {toks[i]} -> {novo}")
        toks[i] = novo
    return "".join(toks), mudancas


# ------------------------------------------------------------ b) troca de gênero
SUBST_GENERO = [
    ("menino", "menina"), ("garoto", "garota"), ("homem", "mulher"), ("pai", "mãe"),
    ("avô", "avó"), ("rei", "rainha"), ("irmão", "irmã"), ("tio", "tia"), ("filho", "filha"),
    ("amigo", "amiga"), ("moço", "moça"), ("príncipe", "princesa"), ("ator", "atriz"),
    ("professor", "professora"), ("cantor", "cantora"), ("aluno", "aluna"), ("médico", "médica"),
    ("vizinho", "vizinha"), ("gato", "gata"), ("cachorro", "cachorra"), ("lobo", "loba"),
    ("leão", "leoa"), ("galo", "galinha"), ("boi", "vaca"), ("cavalo", "égua"),
    ("namorado", "namorada"), ("marido", "esposa"), ("senhor", "senhora"), ("escritor", "escritora"),
    ("pintor", "pintora"), ("poeta", "poetisa"), ("jogador", "jogadora"), ("dono", "dona"),
    ("neto", "neta"), ("primo", "prima"), ("sobrinho", "sobrinha"), ("noivo", "noiva"),
    ("herói", "heroína"), ("bruxo", "bruxa"), ("mestre", "mestra"), ("diretor", "diretora"),
    ("pastor", "pastora"), ("cozinheiro", "cozinheira"), ("pescador", "pescadora"),
    ("rapaz", "moça"), ("urso", "ursa"), ("macaco", "macaca"), ("coelho", "coelha"),
    ("pintinho", "pintinha"), ("gigante", "giganta"), ("camponês", "camponesa"),
]
DETERMINANTES = [
    ("a", "o"), ("as", "os"), ("uma", "um"), ("umas", "uns"), ("da", "do"), ("das", "dos"),
    ("na", "no"), ("nas", "nos"), ("pela", "pelo"), ("pelas", "pelos"), ("essa", "esse"),
    ("essas", "esses"), ("esta", "este"), ("estas", "estes"), ("aquela", "aquele"),
    ("aquelas", "aqueles"), ("minha", "meu"), ("minhas", "meus"), ("tua", "teu"), ("tuas", "teus"),
    ("sua", "seu"), ("suas", "seus"), ("nossa", "nosso"), ("nossas", "nossos"), ("dessa", "desse"),
    ("dessas", "desses"), ("nessa", "nesse"), ("nessas", "nesses"), ("à", "ao"), ("às", "aos"),
    ("toda", "todo"), ("todas", "todos"), ("outra", "outro"), ("outras", "outros"),
    ("mesma", "mesmo"), ("mesmas", "mesmos"), ("bela", "belo"), ("belas", "belos"),
    ("boa", "bom"), ("boas", "bons"), ("velha", "velho"), ("velhas", "velhos"),
]
PRONOMES = [("ela", "ele"), ("elas", "eles"), ("dela", "dele"), ("delas", "deles")]
ADJ_O = set("""bonito lindo alto baixo pequeno magro gordo novo bravo calmo esperto rico pobre
sozinho pronto quieto louco ótimo simpático triste_ curioso feio fraco sábio tranquilo
alegre_ querido amado perdido preto branco vermelho amarelo negro cinzento""".split())
ARTIGOS = {"a", "o", "as", "os", "um", "uma", "uns", "umas", "da", "do", "das", "dos", "na", "no",
           "nas", "nos", "pela", "pelo", "pelas", "pelos", "ao", "aos", "à", "às", "essa", "esse",
           "esta", "este", "minha", "meu", "sua", "seu"}

_G_SUBST = {}
for _m, _f in SUBST_GENERO:
    _G_SUBST.setdefault(_m, _f)
    _G_SUBST.setdefault(_f, _m)
_G_DET = {}
for _m, _f in DETERMINANTES + PRONOMES:
    _G_DET[_m] = _f
    _G_DET[_f] = _m


def _gswap(w, tabela):
    """Troca o gênero de w usando a tabela explícita (aceita plural). None se não houver."""
    wl = w.lower()
    for lema, _f, pl in _candidates(wl, gmorph=False):
        if lema in tabela:
            return _case_like(w, _inflect(tabela[lema], False, pl))
    return None


def _adj_swap(w):
    """Troca o gênero de adjetivo/particípio (-o <-> -a) por lista ou sufixo."""
    wl = w.lower()
    for lema, fem, pl in _candidates(wl, gmorph=True):
        if lema.endswith("o") and (lema in ADJ_O or lema.endswith(("oso", "ado", "ido", "ico"))):
            # lema masculino; a forma original era feminina (fem=True) ou masculina
            return _case_like(w, _inflect(lema, not fem, pl))
    return None


def _swap_sentence_gender(sent, force_pron):
    toks = _tokens(sent)
    idx_pal = list(range(1, len(toks), 2))
    mudancas, nomes = [], []
    for i in idx_pal:
        novo = _gswap(toks[i], _G_SUBST)
        if novo:
            nomes.append(i)
            mudancas.append(f"{toks[i]} -> {novo}")
            toks[i] = novo
    if not nomes and not force_pron:
        return sent, [], False
    # determinantes e adjetivos pré-nominais imediatamente antes de cada substantivo trocado
    trocados = set(nomes)
    for n in nomes:
        j, passos = n - 2, 0
        while j >= 1 and passos < 3 and toks[j + 1].strip() == "":
            novo = _gswap(toks[j], _G_DET)
            if not novo:
                break
            trocados.add(j)
            mudancas.append(f"{toks[j]} -> {novo}")
            toks[j] = novo
            j -= 2
            passos += 1
    # adjetivos/particípios depois do primeiro substantivo (sem artigo antes) e pronomes
    primeiro = min(nomes) if nomes else None
    ativo = True  # False dentro de um sintagma nominal que NÃO foi trocado (ex.: 'um carro novo')
    for i in idx_pal:
        w = toks[i]
        if i in nomes:
            ativo = True
        elif i not in trocados and w.lower() in ARTIGOS and i > (primeiro or 0):
            ativo = False
        if w.lower() in ("ela", "ele", "elas", "eles", "dela", "dele", "delas", "deles"):
            novo = _gswap(w, _G_DET)
            mudancas.append(f"{w} -> {novo}")
            toks[i] = novo
        elif primeiro is not None and i > primeiro and ativo:
            ant = toks[i - 2].lower() if i >= 3 else ""
            if ant not in ARTIGOS:
                novo = _adj_swap(w)
                if novo:
                    mudancas.append(f"{w} -> {novo}")
                    toks[i] = novo
    return "".join(toks), mudancas, bool(nomes)


def mutate_gender(texto, seed=None):
    """Troca o gênero das frases com substantivos de pessoa/animal (+ concordância).

    Ex.: 'A cantora era famosa.' -> 'O cantor era famoso.'
    Pronomes ela/ele da frase seguinte também são trocados para manter a coerência.
    """
    sents = split_sentences(texto)
    saida, mudancas, anterior = [], [], False
    for s in sents:
        novo, m, teve_nome = _swap_sentence_gender(s, anterior)
        saida.append(novo)
        mudancas += m
        anterior = teve_nome
    return " ".join(saida), mudancas


# ------------------------------------------------------------ c) paráfrase
_UNID = r"(?:segundos|minutos|horas|dias|semanas|meses|anos|séculos)"
_ADV_FRENTE = (r"(?:Ontem|Hoje|Amanhã|De manhã|De tarde|De noite|À tarde|À noite|Naquele dia|Naquela noite|"
               r"Naquela tarde|Naquela manhã|Certo dia|Um dia|No dia seguinte|Na manhã seguinte|"
               r"Depois d[aeo]s? \w+|Antes d[aeo]s? \w+|Durante [oa]s? \w+)")
_ADV_FUNDO = (r"(?:ontem|hoje|amanhã|de manhã|à tarde|à noite|naquele dia|na semana passada|no ano passado|"
              r"há \w+ " + _UNID + r"(?: atrás)?)")
_SEGUROS = {"a", "o", "as", "os", "um", "uma", "uns", "umas", "ele", "ela", "eles", "elas", "eu", "nós",
            "meu", "minha", "seu", "sua", "essa", "esse", "esta", "este"}
_END = r"[.!?…]*"
_NOME = r"[A-ZÀ-Ú][\wà-ú]+(?:\s+(?:d[aeo]s?\s+)?[A-ZÀ-Ú][\wà-ú]+){0,2}"


def _split_end(s):
    m = re.match(r"^(.*?)([.!?…]*)$", s, re.S)
    return m.group(1), m.group(2)


_CONJ = {"e", "mas", "ou", "que", "porque", "pois", "se", "quando", "como", "por", "para", "em", "entre",
         "no", "na", "de", "com", "onde", "nem"}


def _clausula_simples(t):
    """Cláusula curta, sem vírgulas/aspas/parênteses: segura para mover adjuntos."""
    return not re.search(r"[,:;()\"“”]", t) and len(t.split()) <= 20


def _p_tempo_atras(s):
    m = re.match(rf"^(?P<fr>(?:Há|Faz|Fazia|Havia)\s+\w+\s+{_UNID}(?:\s+atrás)?)\s*,?\s+(?P<rest>.+)$", s, re.S)
    if not m:
        return None
    rest, fim = _split_end(m["rest"])
    if not _clausula_simples(rest):
        return None
    return f"{_upper_first(rest)} {_lower_first(m['fr'])}{fim}"


def _p_adv_frente(s):
    m = re.match(rf"^(?P<adv>{_ADV_FRENTE}),\s+(?P<rest>.+)$", s, re.S)
    if not m:
        return None
    rest, fim = _split_end(m["rest"])
    if not _clausula_simples(rest):
        return None
    return f"{_upper_first(rest)} {_lower_first(m['adv'])}{fim}"


def _p_adv_fundo(s):
    m = re.match(rf"^(?P<main>.+?),?\s+(?P<adv>{_ADV_FUNDO})(?P<fim>{_END})$", s, re.S)
    if not m:
        return None
    main = m["main"]
    if not _clausula_simples(main) or main.split()[0].lower() in _CONJ:
        return None
    if main.split()[0].lower() in _SEGUROS:
        main = _lower_first(main)  # senão mantém maiúscula (possível nome próprio)
    return f"{_upper_first(m['adv'])}, {main}{m['fim']}"


_PESSOAS = set(_G_SUBST) | {"presidente", "governador", "prefeito", "ministro", "cientista", "artista", "atleta",
                            "jornalista", "compositor", "autor", "inventor", "engenheiro", "sargento", "general",
                            "capitão", "doutor", "padre", "papa", "imperador", "menina", "menino", "craque"}


def _p_aposto_criar(s):
    m = re.match(rf"^(?P<art>O|A|Os|As)\s+(?P<noun>[a-zà-ú]+)\s+(?P<name>{_NOME})\s+(?P<rest>[a-zà-ú].*)$", s, re.S)
    if not m or m["noun"] not in _PESSOAS or m["rest"].startswith(("e ", "ou ")) or not re.search("[a-zà-ú]", m["name"]):
        return None
    return f"{m['name']}, {m['art'].lower()} {m['noun']}, {m['rest']}"


def _p_aposto_desfazer(s):
    m = re.match(rf"^(?P<name>{_NOME}),\s+(?P<art>o|a|os|as)\s+(?P<noun>[a-zà-ú]+),\s+(?P<rest>.+)$", s, re.S)
    if not m:
        return None
    return f"{m['art'].upper()} {m['noun']} {m['name']} {m['rest']}"


# particípio -> (pret. perfeito sg/pl, imperfeito sg/pl, presente sg/pl)
_IRREG = {
    "lido": ("leu", "leram", "lia", "liam", "lê", "leem"),
    "escrito": ("escreveu", "escreveram", "escrevia", "escreviam", "escreve", "escrevem"),
    "feito": ("fez", "fizeram", "fazia", "faziam", "faz", "fazem"),
    "visto": ("viu", "viram", "via", "viam", "vê", "veem"),
    "dito": ("disse", "disseram", "dizia", "diziam", "diz", "dizem"),
    "aberto": ("abriu", "abriram", "abria", "abriam", "abre", "abrem"),
    "descoberto": ("descobriu", "descobriram", "descobria", "descobriam", "descobre", "descobrem"),
    "coberto": ("cobriu", "cobriram", "cobria", "cobriam", "cobre", "cobrem"),
    "posto": ("pôs", "puseram", "punha", "punham", "põe", "põem"),
    "eleito": ("elegeu", "elegeram", "elegia", "elegiam", "elege", "elegem"),
    "preso": ("prendeu", "prenderam", "prendia", "prendiam", "prende", "prendem"),
    "aceito": ("aceitou", "aceitaram", "aceitava", "aceitavam", "aceita", "aceitam"),
    "pago": ("pagou", "pagaram", "pagava", "pagavam", "paga", "pagam"),
    "ganho": ("ganhou", "ganharam", "ganhava", "ganhavam", "ganha", "ganham"),
    "gasto": ("gastou", "gastaram", "gastava", "gastavam", "gasta", "gastam"),
    "entregue": ("entregou", "entregaram", "entregava", "entregavam", "entrega", "entregam"),
}
_IR = {"partido", "sentido", "seguido", "ferido", "servido", "vestido", "repetido", "assistido", "definido",
       "decidido", "dividido", "produzido", "conduzido", "traduzido", "conseguido", "permitido",
       "proibido", "admitido", "exigido", "punido", "fugido", "construído", "destruído", "distribuído",
       "atingido", "dirigido", "pedido", "ouvido", "dormido", "surgido", "sorrido", "vindo", "tido"}
_FIM_AGENTE = {"em", "no", "na", "nos", "nas", "durante", "ontem", "hoje", "antes", "depois", "quando",
               "onde", "para", "com", "porque", "e", "mas", "que", "logo", "ainda", "já", "até", "sobre"}
_ADJUNTO_OK = {"em", "no", "na", "nos", "nas", "durante", "ontem", "hoje", "antes", "depois", "até",
               "sobre", "para", "com"}
_PREP_ART = {"pelo": "o", "pela": "a", "pelos": "os", "pelas": "as"}


def _conjuga(part, tempo, plural):
    """tempo: 'perf' | 'imp' | 'pres'. None se o particípio não for reconhecido."""
    p = part.lower()
    if p.endswith(("os", "as")):
        p = p[:-1]
    if p.endswith("a") and p[:-1] + "o" in _IRREG:
        p = p[:-1] + "o"
    elif p.endswith("ada") or p.endswith("ida"):
        p = p[:-1] + "o"
    if p in _IRREG:
        t = _IRREG[p]
        k = {"perf": 0, "imp": 2, "pres": 4}[tempo] + (1 if plural else 0)
        return t[k]
    if p.endswith("ado"):
        r = p[:-3]
        return r + {"perf": ("ou", "aram"), "imp": ("ava", "avam"), "pres": ("a", "am")}[tempo][plural]
    if p.endswith("ido"):
        r = p[:-3]
        if p in _IR:
            return r + {"perf": ("iu", "iram"), "imp": ("ia", "iam"), "pres": ("e", "em")}[tempo][plural]
        if r.endswith("u") and p.endswith("uído"):
            pass
        return r + {"perf": ("eu", "eram"), "imp": ("ia", "iam"), "pres": ("e", "em")}[tempo][plural]
    if p.endswith("ído"):
        r = p[:-3] + "u"
        return r + {"perf": ("iu", "iram"), "imp": ("ia", "iam"), "pres": ("i", "iem")}[tempo][plural]
    return None


def _p_passiva(s):
    m = re.match(r"^(?P<pac>.+?)\s+(?P<aux>foi|foram|era|eram|é|são)\s+(?P<part>\w+)\s+"
                 r"(?P<prep>por|pelo|pela|pelos|pelas)\s+(?P<resto>.+)$", s, re.S)
    if not m:
        return None
    pac0 = m["pac"]
    if (re.search(r"[,:;()\"“”]", pac0) or len(pac0.split()) > 7
            or pac0.split()[0].lower() in _CONJ):
        return None  # paciente deve ser um sintagma simples
    if m["prep"].lower() == "por" and not m["resto"][:1].isupper():
        return None  # 'por' + substantivo comum (ex.: por escambo) raramente é agente
    resto, fim = _split_end(m["resto"])
    if re.search(r"[:;()\"“”]", resto):
        return None
    palavras = resto.split()
    ag = []
    for w in palavras:
        if ag and (w.lower().strip(",;") in _FIM_AGENTE or w.endswith((",", ";"))):
            break
        ag.append(w)
        if len(ag) >= 4:
            break
    resto_adj = " ".join(palavras[len(ag):])
    if resto_adj and resto_adj.split()[0].lower().strip(",;") not in _ADJUNTO_OK:
        return None  # agente não pôde ser delimitado com segurança
    agente = " ".join(ag).rstrip(",;")
    prep = m["prep"].lower()
    plural = prep in ("pelos", "pelas")
    if prep in _PREP_ART:
        agente = f"{_PREP_ART[prep]} {agente}"
    else:  # 'por' + nome próprio: singular
        plural = False
    tempo = {"foi": "perf", "foram": "perf", "era": "imp", "eram": "imp", "é": "pres", "são": "pres"}[m["aux"].lower()]
    verbo = _conjuga(m["part"], tempo, plural)
    if verbo is None:
        return None
    pac = m["pac"]
    if pac.split()[0].lower() in _SEGUROS:
        pac = _lower_first(pac)
    frase = f"{_upper_first(agente)} {verbo} {pac}"
    if resto_adj:
        frase += " " + resto_adj
    return frase + fim


_REGRAS_PARAFRASE = [
    ("adjunto temporal", [_p_tempo_atras, _p_adv_frente, _p_adv_fundo]),
    ("aposto", [_p_aposto_criar, _p_aposto_desfazer]),
    ("voz passiva -> ativa", [_p_passiva]),
]


def paraphrase_sentence(s, rng=None):
    """Aplica UMA regra de paráfrase aplicável (escolhida aleatoriamente). None se nenhuma servir."""
    rng = rng or random.Random()
    aplicaveis = []
    for nome, fns in _REGRAS_PARAFRASE:
        for fn in fns:
            r = fn(s.strip())
            if r and r != s.strip():
                aplicaveis.append((nome, r))
                break
    return rng.choice(aplicaveis) if aplicaveis else None


def mutate_paraphrase(texto, prob=1.0, seed=None):
    """Parafraseia as sentenças às quais alguma regra se aplica (com probabilidade `prob`)."""
    rng = random.Random(seed)
    saida, mudancas = [], []
    for s in split_sentences(texto):
        r = paraphrase_sentence(s, rng) if rng.random() < prob else None
        if r:
            mudancas.append(f"{r[0]}: '{s}' -> '{r[1]}'")
            saida.append(r[1])
        else:
            saida.append(s)
    return " ".join(saida), mudancas


# ------------------------------------------------------------ interface comum
OPERADORES = {
    "troca_palavras": lambda t, seed=None: mutate_words(t, seed=seed),
    "troca_genero": lambda t, seed=None: mutate_gender(t, seed=seed),
    "parafrase": lambda t, seed=None: mutate_paraphrase(t, seed=seed),
}


def apply_mutation(nome, texto, seed=None):
    return OPERADORES[nome](texto, seed=seed)


if __name__ == "__main__":
    demos = {
        "troca_genero": ["A cantora era famosa.", "A professora cansada chegou à escola. Ela abriu a janela."],
        "parafrase": ["Há duas semanas atrás estava chovendo", "O cantor Roberto Carlos fez aniversário",
                      "O texto foi lido pelo aluno", "Ontem, o menino comprou um carro novo."],
        "troca_palavras": ["O menino bonito disse que a casa era grande e queria um carro novo."],
    }
    for op, frases in demos.items():
        for f in frases:
            print(f"[{op}] {f}\n   -> {apply_mutation(op, f, seed=1)}")

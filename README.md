# CAEd · Complexidade textual e mutação de textos narrativos

Sistema desenvolvido para o processo de seleção do CAEd (Prova 024/2026). Ele:

1. **classifica** a complexidade de um pequeno texto em 4 níveis (Ensino Fundamental I, Fundamental II, Médio e Superior);
2. **gera novas versões** do texto com três operadores de mutação (troca de palavras, troca de gênero e paráfrase);
3. **mede o impacto** de cada mutação na complexidade prevista pelo modelo;
4. oferece uma **interface web** para usar tudo isso de forma interativa.

A metodologia completa, os experimentos e as limitações estão no **relatório** que acompanha esta entrega.

## Requisitos

- Python 3.10 ou superior (testado com 3.13)
- Dependências: `numpy`, `pandas`, `scikit-learn`, `joblib`, `matplotlib`, `seaborn`, `flask`

NLTK e spaCy **não são necessários**: a canonicidade, as mutações e os atributos foram implementados em Python puro.

## Instalação

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset

Coloque as quatro pastas do dataset dentro de `data/raw/`:

```
data/raw/
├── 1_Ensino_Fundamental_I/
├── 2_Ensino_Fundamental_II/
├── 3_Ensino_Medio/
└── 4_Ensino_Superior/
```

O rótulo de cada texto vem do nome da pasta. (Os nomes exatos das pastas devem ser os mesmos do dataset recebido.)

## Como executar

### Pipeline completo (preparar, treinar e avaliar, cerca de 2 minutos)

```bash
python main.py tudo
```

Ou cada etapa separadamente:

```bash
python main.py preparar                 # limpa, fragmenta e divide train/val/test em data/processed/
python main.py treinar                  # treina 3 modelos, escolhe o melhor e salva outputs/model.joblib
python main.py avaliar                  # mede o impacto das mutações no conjunto de teste
python main.py avaliar --max-textos 300 # versão rápida da avaliação
```

### Analisar um texto pela linha de comando

```bash
python main.py texto "A cantora era famosa. O texto foi lido pelo aluno."
```

### Interface web

```bash
python app.py                    # http://127.0.0.1:5000
FLASK_PORT=8080 python app.py    # outra porta (PowerShell: $env:FLASK_PORT=8080; python app.py)
```

Na página, cole um texto e clique em **Analisar** (ou use Ctrl+Enter). A interface mostra:

- a classe prevista para o texto original, as probabilidades por nível e os atributos linguísticos (canonicidade, tamanho médio de frase, TTR etc.);
- uma aba para cada mutação (troca de palavras, troca de gênero, paráfrase e todas em sequência), com as **palavras alteradas destacadas**, a nova classe prevista e a variação do nível esperado;
- o campo "semente" torna as mutações aleatórias reproduzíveis.

#### API

`POST /api/analisar` com `{"texto": "...", "seed": 42}` devolve a classificação do original e, para cada operador, o texto mutado, as alterações feitas, a nova classe, as probabilidades e o `delta` do nível esperado. `GET /api/saude` verifica se o modelo foi carregado.

> Se aparecerem avisos de versão do scikit-learn ao carregar o `model.joblib`, rode `python main.py treinar` para regenerá-lo na sua versão.

## Estrutura do projeto

```
.
├── main.py                 # linha de comando (preparar, treinar, avaliar, texto, tudo)
├── app.py                  # servidor web (Flask) e API JSON
├── templates/index.html    # interface web (HTML, CSS e JS puros)
├── requirements.txt
├── src/
│   ├── loader.py           # leitura, limpeza, fragmentação e divisão por arquivo
│   ├── features.py         # canonicidade (CV) e atributos estatísticos
│   ├── model.py            # TF-IDF + atributos, 3 classificadores, exportação
│   ├── mutations.py        # os 3 operadores de mutação
│   └── pipeline.py         # texto -> mutação -> predição -> impacto
├── data/
│   ├── raw/                # dataset original
│   └── processed/          # instâncias de treino, validação e teste (CSV)
└── outputs/
    ├── model.joblib        # modelo escolhido
    ├── model_metrics.json  # métricas dos 3 modelos
    ├── classification_report.txt
    ├── impact_summary.csv / .png
    ├── impact_results.csv
    └── impact_examples.json
```

## Resumo dos resultados

**Modelo (conjunto de teste, com arquivos nunca vistos no treino):**

| Modelo | Acurácia | F1-macro |
|---|---|---|
| Regressão Logística | 0,849 | 0,817 |
| Random Forest | 0,759 | 0,701 |
| **SVM linear (escolhido)** | **0,855** | **0,823** |

**Impacto das mutações (1.885 textos de teste):**

| Operador | Textos alterados | Mudou de classe (entre alterados) |
|---|---|---|
| Troca de palavras | 68,1% | 2,3% |
| Troca de gênero | 16,1% | 1,0% |
| Paráfrase | 0,6% | 0,0% |
| Todas em sequência | 70,9% | 2,3% |

As mutações preservam, em geral, a classe prevista. A paráfrase quase não se aplica a textos reais do corpus porque suas regras são conservadoras; ela foi validada nos três exemplos do edital e em frases construídas.

## Verificações rápidas

```bash
python src/features.py     # testa a canonicidade nos exemplos do edital
python src/mutations.py    # reproduz os exemplos de mutação do edital
```

## Limitações principais

- O léxico de sinônimos e antônimos é próprio e pequeno (cerca de 75 pares de sinônimos e 35 de antônimos); ampliá-lo ou integrar o WordNet/OMW é a melhoria mais direta.
- A canonicidade é ortográfica, não fonológica.
- O impacto das mutações é medido pela predição do modelo; não houve avaliação humana da qualidade dos textos gerados.
- O modelo foi treinado com janelas de até 80 palavras.

Detalhes e discussão completa no relatório.

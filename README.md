# Science Across Languages: Modular Multi-Agent LLM Framework for Document-Level Scientific Translation in Low-Resource Languages

Install:
python3.11 -m venv .venv
uv pip install openai langgraph pandas selenium tqdm lingua-language-detector requests python-dotenv jinja2 pyyaml pymupdf4llm sonar-space googletrans pymupdf-layout

## **Project Summary**

## **Background Information**

## **Question to be Studied**

## **Research Plan**

_Objective I_
_Rationale_

_Objective II_
_Rationale_

_Objective III_
_Rationale_

_Objective IV_
_Rationale_

_Objective V_
_Rationale_

## **Experimental Design**

problem: lack of parallel scientific documents for developing or evaluating scientific translation abilities in low-resource languages

### Data pipeline

High-level overview:

1. Download scientific/academic articles from OpenAlex database as PDFs
2. Convert downloaded PDFs to markdown, preserving document format and images
3. Clean up documents by removing artifacts and select examples (how?)

### Baseline comparison

Test SOTA and commonly used translation models, including commercially available NMT solutions at different payment tiers (google translate free/paid, DeepL) as well as large language models from different providers (chatgpt, gemini, etc). Evaluate using evaluation methods described below.

### Agent roles + Pipeline overview

High-level overview of proposed pipeline:

1. Document planner - build discourse graph of source document (https://arxiv.org/html/2507.03311v1)
2. Terminology agent - extract terms and fetch glossary official translations
3. RAG agent - extract relevatn snippets from monolingual target document corpora
4. Translation agent - Use additional information (e.g. glossary entries, relevant snippets) to translate
5. Document manager - Combine translated segments and perform final review of full document

Document planning agent (required)

- Input: full source document with structure (headers, figures, etc)
- Output: Directed Acyclic Graph
- Tasks:
  - Discourse Agent: Segment into translation/discourse units (sentences/paragraphs) that are internally coherent
  - Edge agent: Identfiy which prior discourses each discourse depends on
  - Note: each discourse will automatically depend on the one that came before to maintain document fluency
  - Construct directed graph

Terminology agent (segment level) (additional module)

- Input: Discourse unit
- Output: Relevant target language term-translation pairs
- Tasks:
  - Extract key domain-specific terms (KeyBERT or specialized LLM)
  - Use language domain corpus to retrieve verified translation for terms
  - Provide source word and target definition/transations in translation prompt

RAG agent (segment level) (additional module)

- Input: Discourse unit
- Output: Relevant snippets from monolingual target language corpus
- Tasks:
  - Calculates relevant snippets from monolingual target language corpus using cross-lingual embedding similarity
  - Provides relevant snippets as few-shot examples of target language scientfic text in translation prompt

Translation agent (segment level) (required)

- Input: Discourse unit + additional module outputs
- Output: Translated discourse unit
- Tasks:
  - Combine additional module outputs with system translation prompt (e.g. "Translate into Swahili" + "The word {src_word} should be translated as {target_word}" + "Example Swahili snippet: {example_sents}" + "Don't provide any additional explanation")
  - Generate translation with contextual prompt (LLM)

Document manager (required)

- Input: Translated discourse units
- Output: Combined translated document
- Tasks:
  - Combine translated discourse units

Final reviwer agent (additional module)

- Input: Translated document
- Output: Refined translated document
- Tasks:
  - Perform pass-over document and note any glaring issues, hallucinations, or missing content

### Evaluation

Lack of parallel scientific document corpora in low-resource languages necessitates reference-free evaluation methods.

Reference-free evaluation:

- Cross-lingual embedding (LASER or LaBSE) similarity between source and target document embeddings
- Facebook sonar BLASER eval
- [COMET-QE](https://aclanthology.org/2022.findings-emnlp.348/) can provide an additional data-point for translation accuracy (biased towards fluency over grammar however, limited language coverage)
- MetricxQE (google version of COMET-QE)
- [GEMBA](https://github.com/MicrosoftTranslator/GEMBA)

Gold-standard:

- Native speaker evaluation through survey with example translation

### Ablation

Evaluate effectiveness of translation pipeline in different configurations, including but not limited to the following:

- Terminology agent + RAG agent + final reviwer
- Terminolgy agent + RAG agent
- Terminology agent + final reviwer

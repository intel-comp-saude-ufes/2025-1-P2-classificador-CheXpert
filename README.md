# 2025-1-P2-classificador-CheXpert

## Objetivo

O intuito desse trabalho é desenvolver um classificador de radiografias de tórax e analisar a sua capacidade de generalização em outra base. Os modelos foram treinandos primeiramente na base do kaggle *Chest X-ray Image* (<https://www.kaggle.com/datasets/alsaniipe/chest-x-ray-image>) e testados posteriormente na base do kaggle *CheXpert-v1.0-small* (<https://www.kaggle.com/datasets/ashery/chexpert>).

## O que o repositório possui?

* Notebook principal (trab.ipynb)
Centraliza os seguintes passos: carregamento, pré-processamento, treinamento, avaliação e inferência do modelo e geração de resultados.

* Módulo `lib/`
Contém utilitários como carregadores de dados, transformações de imagem, definição de modelo e funções auxiliares.

* `requirements.txt`
Lista das dependências necessárias para executar o projeto.

* Arquivo de metadados para a geração de gráficos.


---
# Estrutura do repositório
Este projeto é estruturado para:
1. Armazenar **dados brutos e tratados**.
2. Executar **modelos e análises estatísticas** com scripts organizados.
3. Produzir e armazenar **gráficos e resultados prontos para o artigo**.

## Diretórios

### `lib/`
Biblioteca interna com os módulos do projeto:

* `data.py`: Leitura dos datasets e processamento de dados.

* `early_stopping.py`: Define uma classe que cuida da lógica de *early stopping*.
  
* `history.py`: Define a classe que cuida do gerenciamento do histórico de treinamento.
  
* `metrics.py`: Define uma classe que é um wrapper de métricas, facilitando o cálculo e armazenamento das métricas durante o treinamento.
  
* `models.py`: Define as classes dos modelos que foram utilizados nesse trabalho.
  
* `training.py`: Contém os códigos de treinamento e avalição, bem como outros relacionados a estes processos.

* `utils.py`: Possui algumas funções auxiliares e de *plotting*.

### `experiments/`
Pasta gerada durante o processo de treinamento e geração de resultado dos modelos.


## Como usar

Para reproduzir os resultados apresentados neste código, é necessário ter o **Python** instalado em seu sistema, bem como o **pip** — o gerenciador de pacotes do Python. Também é recomendável utilizar um ambiente virtual (venv) para garantir o isolamento das dependências do projeto. Além disso, você deve ter o código disponível localmente na sua máquina. O código do repositório pode ser obtido de duas formas principais:

### Opção 1: Clonar via Git

```bash
git clone https://github.com/intel-comp-saude-ufes/2025-1-P2-classificador-CheXpert.git
cd 2025-1-P2-classificador-CheXpert
```

### Opção 2: Baixar ZIP

1. Acesse o repositório no GitHub.
2. Clique no botão verde **"Code"**.
3. Selecione **"Download ZIP"**.
4. Extraia o arquivo ZIP em uma pasta da sua preferência.
5. Abra o terminal (ou prompt de comando) dentro da pasta onde extraiu os arquivos para executar os comandos e scripts.

### Executando o código
A seguir, apresentamos as instruções recomendadas para reproduzir os experimentos e gerar os resultados.

1. Instalar dependências do projeto:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2.  Executar o projeto:

Executar as células do notebook python `trab.ipynb`.

3. Verifique o diretório `experiments/` para verificar o resultado dos experimentos. Além do diretório raiz onde serão salvos plots, tabelas latex e arquivos csv's.

---
# Contato
| Autor                 | GitHub               | E-mail               |
| :---------------- | :------: | ----: |
| Pedro Igor Gomes de Morais | [@Pedro2um](https://github.com/Pedro2um) | pedroigorgm@gmail.com |
| Matheus Saick De Martin | [@saick123](https://github.com/saick123) | matheussaick@gmail.com |
| Renzo Henrique Guzzo Leão | [@Renzo-Henrique](https://github.com/seuusuario) | renzolealguzzo@gmail.com |
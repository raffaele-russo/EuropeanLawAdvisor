# EuropeanLawAdvisor


![Build Status](https://github.com/raffaele-russo/EuropeanLawAdvisor/actions/workflows/pylint.yml/badge.svg)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 10](https://img.shields.io/badge/Python%20%7C%203.10-green.svg)](https://shields.io/)

----

EuropeanLawAdvisor allows users to retrieve information about European Laws 

----

## To set up the environment


##### You must have a working [Docker environment] and ollama installed.
Clone the repository:
   ```bash
   git clone https://github.com/raffaele-russo/EuropeanLawAdvisor.git
   cd EuropeanLawAdvisor
```

Set up a Python virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
Create a .env file for Elasticsearch and Kibana credentials:
1. In the root directory of the project, create a `.env` file.
2. Add the following variables:

```bash
ELASTICSEARCH_USERNAME=your_elastic_username
ELASTICSEARCH_PASSWORD=your_elastic_password
KIBANA_PASSWORD=your_kibana_password
```
EuropeanLawAdvisor can use a local LLM for the RAG.

Check out ollama doc for all the models available https://ollama.com/library?sort=popular.

Download the model:
```bash
ollama pull <LLM_MODEL>
```

and set the variable:
```bash
LLM_MODEL
```
Optionally, for better performance you can use an OpenAI model by setting:
```bash
OPENAI_API_KEY
OPENAI_MODEL_NAME
```
## To set up the advisor
```bash
sh setup.sh
```
Elasticsearch will be accessible at http://localhost:9200.

Kibana will be accessible at http://localhost:5601.

Note: You will already be able to start the advisor during the document upload stage of the setup.

## To start the advisor

```bash
cd fast_api_app 
fastapi dev main.py
```

The Law Advisor is now accessible at http://localhost:8000.

## References

<a id="1">[1]</a> 
Chalkidis, I., Fergadiotis, M., Malakasiotis, P., & Androutsopoulos, I. (2019). 
Large-Scale Multi-Label Text Classification on EU Legislation. 
In *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics*, (pp. 6314-6322), Florence, Italy. Association for Computational Linguistics. 
[DOI: 10.18653/v1/P19-1636](https://doi.org/10.18653/v1/P19-1636) [URL](https://www.aclweb.org/anthology/P19-1636)

## Citing this work

@INPROCEEDINGS{10826025,
  author={Russo, Raffaele and Russo, Diego and Orlando, Gian Marco and Romano, Antonio and Riccio, Giuseppe and Gatta, Valerio La and Postiglione, Marco and Moscato, Vincenzo},
  booktitle={2024 IEEE International Conference on Big Data (BigData)}, 
  title={EuropeanLawAdvisor: an open source search engine for European laws}, 
  year={2024},
  volume={},
  number={},
  pages={4751-4756},
  keywords={Accuracy;Law;Large language models;Retrieval augmented generation;Europe;Legislation;Search engines;Nearest neighbor methods;Search problems;Software development management;Retrieval-Augmented Generation (RAG);NLP in Legal Industry;Generative AI;Explainable AI},
  doi={10.1109/BigData62323.2024.10826025}}

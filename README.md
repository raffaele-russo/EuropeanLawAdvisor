# EuropeanLawAdvisor


![Build Status](https://github.com/raffaele-russo/EuropeanLawAdvisor/actions/workflows/pylint.yml/badge.svg)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 10](https://img.shields.io/badge/Python%20%7C%203.10-green.svg)](https://shields.io/)

----

EuropeanLawAdvisor allows users to retrieve information about European Laws 

----

## To set up the environment


##### You must have a working [Docker environment].
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

The Law Advisor is now accessible at http://localhost:8000
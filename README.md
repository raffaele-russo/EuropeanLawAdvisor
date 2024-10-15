# EuropeanLawAdvisor


![Build Status](https://github.com/raffaele-russo/EuropeanLawAdvisor/actions/workflows/pylint.yml/badge.svg)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 10](https://img.shields.io/badge/Python%20%7C%203.10-green.svg)](https://shields.io/)

----

EuropeanLawAdvisor allows users to retrieve information about Euopean Laws 

----

## To set up the advisor


##### You must have a working [Docker environment].


```
git clone https://github.com/raffaele-russo/EuropeanLawAdvisor.git
cd EuropeanLawAdvisor
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
sh setup.sh
```

## To start the advisor

```
cd fast_api_app 
fastapi dev main.py
```

The Law Advisor is now accessible at http://localhost:8000
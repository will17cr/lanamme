# %% [code]
# %% [code]
# coding: utf-8
import os
print(os.environ.get('KAGGLE_KERNEL_RUN_TYPE'))
os.system('pip install -U pip')
os.system('pip install sickle')
os.system('pip install dateparser')
# # os.system('pip install openpyxl')
os.system('pip install python-dateutil')
# # os.system('pip install python-dotenv')
os.system('pip install gspread')
# # os.system('pip install gspread-pandas')
os.system('pip install gspread_dataframe')
os.system('pip install google-api-python-client')
# # os.system('pip install keyring')
# # os.system('pip install --upgrade nbconvert')

# # Procesamiento informes LANAMME desde repositorio
# # Wilmer Ramirez Morera
# # wilmer.ramirez@gmail.com
# # wilmer.ramirez@cgr.go.cr

print("\nScript to request records from LANAMME's repository and update our data")

# ## Importaciones
import datetime as dt
import pandas as pd
from ctypes.util import find_library
from pandas.core.dtypes.inference import is_number
from pandas.core.dtypes.common import is_numeric_v_string_like

import numpy as np
import re

# # Proyecto
# ## Sickle OAI-PHM

from sickle import Sickle
from pprint import pprint

print("Connecting to repository and requesting table of records")

URL = 'https://www.lanamme.ucr.ac.cr/oai/request?'
sickle = Sickle(URL)
records = sickle.ListRecords(metadataPrefix='oai_dc',ignore_deleted=True)
mylist=list()
myDF=pd.DataFrame()

for record in records:
    # pprint(record.header)
    # print()
    # pprint(record.metadata)
    mylist.append(record.metadata)
    tDF=pd.DataFrame([record.metadata])
    myDF=pd.concat([myDF,tDF],ignore_index=True)
    # print(record.metadata)
    # print()
    # print()

print("Processing table of records...")
print(f"{len(myDF)} records in repository")
# ## `matter` boolean para extraer informes
myDF["matter"]=False
myDF.loc[myDF.type.isna(),"type"]=myDF.loc[myDF.type.isna(),"type"].apply(lambda x:[""])
myDF['tipos'] = [', '.join(map(str, l)) for l in myDF['type']]
print(f"\n{myDF.tipos.value_counts()}\n")
# ### metodo para encontrar informes tecnicos
s1=myDF["type"].explode()
cond = s1.str.contains('informe')

cond[cond.isnull()]=True

myDF.loc[s1[cond].index.unique(),"matter"]=True
myDF=myDF[myDF.matter]
myDF.loc[:,'resumen']=myDF.description.apply(lambda x:x[0])
myDF.loc[myDF.subject.isna(),"subject"]=myDF.loc[myDF.subject.isna(),"subject"].apply(lambda x:[""])
myDF['tipos'] = [', '.join(map(str, l)) for l in myDF['type']]
myDF['topicos'] = [', '.join(map(str, l)) for l in myDF['subject']]

myDF.loc[:,"publicado"]=myDF.date.apply(lambda x:x[2])

import dateparser
myDF.loc[:,"fecha_publicado"]=myDF.publicado.apply(lambda x:dateparser.parse(x,settings={'PREFER_DAY_OF_MONTH': 'first',"PREFER_MONTH_OF_YEAR": "first"}))

myDF.loc[:,"fecha"]=myDF.date.apply(lambda x:x[0])
myDF.loc[:,'titulo']=myDF.title.apply(lambda x:x[0])
myDF.loc[:,'title_N']=myDF.title.apply(lambda x:len(x))
myDF.loc[myDF.title_N!=1,"consecutivo"]=myDF.loc[myDF.title_N!=1,"title"].apply(lambda x:x[1])

myDF.loc[~myDF.relation.isna(),"relaciones"]=myDF[~myDF.relation.isna()].relation.apply(lambda x:x[0])
myDF.loc[~myDF.relaciones.isna(),"relaciones"]=myDF[~myDF.relaciones.isna()].relaciones.apply(lambda x:x.replace(";",""))
myDF.loc[(myDF.consecutivo.isna()&(~myDF.relation.isna())),"consecutivo"] =\
   myDF.loc[(myDF.consecutivo.isna()&(~myDF.relation.isna())),"relaciones"]
myDF.loc[:,'autores']=myDF.creator.apply(lambda x:x[0])
myDF['autores'] = ['; '.join(map(str, l)) for l in myDF['creator']]
myDF.loc[myDF.publisher.isna(),"type"]=myDF.loc[myDF.publisher.isna(),"publisher"].apply(lambda x:[""])
myDF.loc[~myDF.publisher.isna(),'publicador']=myDF[~myDF.publisher.isna()].publisher.apply(lambda x:x[0])
myDF.loc[:,'autores']=myDF.creator.apply(lambda x:x[0])
myDF.loc[:,'formato']=myDF.format.apply(lambda x:x[0])
myDF.loc[~myDF.language.isna(),'idioma']=myDF[~myDF.language.isna()].language.apply(lambda x:x[0])
myDF.loc[~myDF.identifier.isna(),'enlace']=myDF[~myDF.identifier.isna()].identifier.apply(lambda x:x[0])


finalDF = myDF.iloc[:,myDF.columns.get_loc("tipos"):].copy()
finalDF.sort_values(by='fecha_publicado',ascending=False,inplace=True)


# ### consolidar todos los topicos
# ### para debugging y opciones de filtrados
all_topics = ', '.join(finalDF.topicos)
array_topics0 = sorted(all_topics.split(", "))
array_topics = sorted(set(all_topics.split(", ")))
print(f"{len(array_topics)} unique subjects or topics identified!!!")

# ### consolidar todos los tipos
# ### para debugging y opciones de filtrados
all_types = ', '.join(finalDF.tipos)
array_types0 = sorted(all_types.split(", "))
array_types = sorted(set(all_types.split(", ")))
print(f"{len(array_types)} unique types identified!!!")
print(f"{len(finalDF)} records acquired!!!")
# # Google API
"""
BEFORE RUNNING:
---------------
1. If not already done, enable the Identity and Access Management (IAM) API
   and check the quota for your project at
   https://console.developers.google.com/apis/api/iam
2. This sample uses Application Default Credentials for authentication.
   If not already done, install the gcloud CLI from
   https://cloud.google.com/sdk and run
   `gcloud beta auth application-default login`.
   For more information, see
   https://developers.google.com/identity/protocols/application-default-credentials
3. Install the Python client library for Google APIs by running
   `pip install --upgrade google-api-python-client`
4. Install the OAuth 2.0 client for Google APIs by running
   `pip install --upgrade oauth2client`
"""

""""
from pprint import pprint

from googleapiclient import discovery
from oauth2client.client import GoogleCredentialscredentials = GoogleCredentials.get_application_default()service = discovery.build('iam', 'v1', credentials=credentials)

# The resource name of the service account in the following format:
# `projects/{PROJECT_ID}/serviceAccounts/{ACCOUNT}`.
# Using `-` as a wildcard for the `PROJECT_ID` will infer the project from
# the account. The `ACCOUNT` value can be the `email` address or the
# `unique_id` of the service account.
name = 'projects/exalted-well-306518/serviceAccounts/117408996984181101702'  # TODO: Update placeholder value.create_service_account_key_request_body = {
    # TODO: Add desired entries to the request body.
}request = service.projects().serviceAccounts().keys().create(name=name, body=create_service_account_key_request_body)response = request.execute()# TODO: Change code below to process the `response` dict:
pprint(response)
"""

print("Updating data in Google Sheet file")

# ## gspread
import gspread
try:
    from gspread_dataframe import set_with_dataframe
except ImportError:
    os.system('pip install gspread_dataframe')
finally:
    from gspread_dataframe import set_with_dataframe

#### Most regular way with basic secret
# ### Local JSON
# gc = gspread.service_account(filename="./myconfig/some.json")

# ### Secured JSON

import json
# import keyring

##### Regular method in local machine with keyring

# Retrieve the credentials from Keychain
# credentials = keyring.get_password("Google API Credentials Lanamme", "Google API Lanamme")

# Parse the JSON credentials
# credentials_dict = json.loads(credentials)


##### Method for Kaggle

from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("Google_API")
secret_value_1 = user_secrets.get_secret("SheetID")

credentials_dict = json.loads(secret_value_0)




gc = gspread.service_account_from_dict(credentials_dict)

sh = gc.open('LANAMME_2024')
sht1 = gc.open_by_key(secret_value_1)

from gspread_dataframe import set_with_dataframe
from datetime import datetime


datetime.today().strftime('%Y-%m-%d')
try:
    worksheet = sh.add_worksheet(title=f'{datetime.today().strftime("%Y-%m-%d")}',rows=1,cols=16)
except:
        worksheet = sh.worksheet(title=f'{datetime.today().strftime("%Y-%m-%d")}')
set_with_dataframe(worksheet, finalDF,include_index=True,resize=True)
worksheet = sht1.get_worksheet(0)  # Replace 0 with the index of your desired worksheet
set_with_dataframe(worksheet, finalDF,include_index=True,resize=True)
print("\nLANAMME data updated\n")

# Lanamme scripts

Repository with different python and ipython scripts for getting LanammeUCR documents metadata from DSpace repository and store it in spreadsheet as source for a visualization tool (like LookerStudio, Tableau, others).

## Script actions:

- Connect with repository
  - Retrieve records
  - Filter for records of specific matter
  - Build dataframe from selected records
- Connect with Google Sheet by API
  - Retrieve master table
- Get new records not in master table
- Obtain pdf direct url for records with no pdf url
- Request analysis to Gemini API
  - Send prompt to Gemini model by API
  - Get response and update dataframe
- Append new records to master table in Google Sheet
- Save new sheet with only new records

## Further development

- Future features to add
  - Update repository with sort of logging
  - Send email advising new records

import pandas as pd
import pyodbc

# Database connection parameters
server = 'LAPTOP-FMKJMC0O'
database = 'voka_db'
driver = 'ODBC Driver 17 for SQL Server'
trusted_connection = 'yes'

# Construct the connection string
connection_string = f'DRIVER={{{driver}}};SERVER={server};DATABASE={database};Trusted_Connection={trusted_connection}'

# Establish the connection
connection = pyodbc.connect(connection_string)

# Define the SQL query
query = """
SELECT
    fd.crm_account_id,
    fd.crm_financieledata_boekjaar,
    fd.crm_financieledata_aantal_maanden,
    fd.crm_financieledata_toegevoegde_waarde,
    fd.crm_financieledata_fte,
    fd.crm_financieledata_gewijzigd_op,
    l.crm_lidmaatschap_datum_opzeg,
    p.crm_persoon_marketingcommunicatie,
    a.crm_account_adres_provincie,
    a.crm_account_primaire_activiteit
FROM
    financiele_data fd
JOIN
    lidmaatschap l ON fd.crm_account_id = l.crm_account_id
JOIN
    account a ON fd.crm_account_id = a.crm_account_id
JOIN
    contact c ON a.crm_account_id = c.crm_contact_account
JOIN
    persoon p ON c.crm_persoon_id = p.crm_persoon_id;
"""
path=r'C:\Schoolµ\3de_jaar\sem_1\DEP\Github\Data-Engineering-Project-II\src\epic 8\epic8.csv'
# Read data from the database into a DataFrame
df = pd.read_sql(query, connection)
df.to_csv(path, index=False)


# Close the database connection
connection.close()
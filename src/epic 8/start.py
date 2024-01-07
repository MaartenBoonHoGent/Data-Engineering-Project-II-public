import pandas as pd
import pyodbc
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Database connection parameters
server = 'LAPTOP-BSJ8KIE2'
database = 'voka_db_6'
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

df = pd.read_sql(query, connection)

connection.close()

y = df['crm_lidmaatschap_datum_opzeg']
feature_columns = [col for col in df.columns if col != 'crm_lidmaatschap_datum_opzeg']
X = df[feature_columns]

# Encoding categorical features
categorical_columns = X.select_dtypes(include=['object', 'category']).columns
encoder = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
X_encoded = encoder.fit_transform(X[categorical_columns])

X = X.drop(columns=categorical_columns)
X = pd.concat([X.reset_index(drop=True), pd.DataFrame(X_encoded.toarray())], axis=1)

# Standardizing the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=0)

rf = RandomForestClassifier(random_state=0)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 4, 6]
}
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_rf = RandomForestClassifier(**best_params, random_state=0)
best_rf.fit(X_train, y_train)

predictions = best_rf.predict(X_test)
report = classification_report(y_test, predictions)

print(best_params)
print(report)
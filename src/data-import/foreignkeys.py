import json
import dotenv
import os

dotenv.load_dotenv()
# SQL statements provided
sql_statements = """
ALTER TABLE [dbo].[account_activiteitscode]  WITH NOCHECK ADD  CONSTRAINT [FK_account_activiteitscode.crm_account_id] FOREIGN KEY([crm_account_id])
REFERENCES [dbo].[account] ([crm_account_id])
GO
ALTER TABLE [dbo].[account_activiteitscode] NOCHECK CONSTRAINT [FK_account_activiteitscode.crm_account_id]
GO
ALTER TABLE [dbo].[account_activiteitscode]  WITH NOCHECK ADD  CONSTRAINT [FK_account_activiteitscode.crm_activiteitscode_id] FOREIGN KEY([crm_activiteitscode_id])
REFERENCES [dbo].[activiteitscode] ([crm_activiteitscode_id])
GO
ALTER TABLE [dbo].[account_activiteitscode] NOCHECK CONSTRAINT [FK_account_activiteitscode.crm_activiteitscode_id]
GO
ALTER TABLE [dbo].[activiteit_vereist_contact]  WITH NOCHECK ADD  CONSTRAINT [FK_activiteit_vereist_contact.crm_activiteitvereistcontact_reqattendee] FOREIGN KEY([crm_activiteitvereistcontact_reqattendee])
REFERENCES [dbo].[contact] ([crm_contact_id])
GO
ALTER TABLE [dbo].[activiteit_vereist_contact] NOCHECK CONSTRAINT [FK_activiteit_vereist_contact.crm_activiteitvereistcontact_reqattendee]
GO
ALTER TABLE [dbo].[activiteit_vereist_contact]  WITH NOCHECK ADD  CONSTRAINT [FK_activiteit_vereist_contact.crm_afspraak_id] FOREIGN KEY([crm_afspraak_id])
REFERENCES [dbo].[afspraak_alle] ([crm_afspraak_id])
GO
ALTER TABLE [dbo].[activiteit_vereist_contact] NOCHECK CONSTRAINT [FK_activiteit_vereist_contact.crm_afspraak_id]
GO
ALTER TABLE [dbo].[afspraak_account_gelinkt]  WITH NOCHECK ADD  CONSTRAINT [FK_afspraak_account_gelinkt.crm_account_id] FOREIGN KEY([crm_account_id])
REFERENCES [dbo].[account] ([crm_account_id])
GO
ALTER TABLE [dbo].[afspraak_account_gelinkt] NOCHECK CONSTRAINT [FK_afspraak_account_gelinkt.crm_account_id]
GO
ALTER TABLE [dbo].[afspraak_account_gelinkt]  WITH NOCHECK ADD  CONSTRAINT [FK_afspraak_account_gelinkt.crm_afspraak_id] FOREIGN KEY([crm_afspraak_id])
REFERENCES [dbo].[afspraak_alle] ([crm_afspraak_id])
GO
ALTER TABLE [dbo].[afspraak_account_gelinkt] NOCHECK CONSTRAINT [FK_afspraak_account_gelinkt.crm_afspraak_id]
GO
ALTER TABLE [dbo].[afspraak_betreft_account]  WITH NOCHECK ADD  CONSTRAINT [FK_afspraak_betreft_account.crm_account_id] FOREIGN KEY([crm_account_id])
REFERENCES [dbo].[account] ([crm_account_id])
GO
ALTER TABLE [dbo].[afspraak_betreft_account] NOCHECK CONSTRAINT [FK_afspraak_betreft_account.crm_account_id]
GO
ALTER TABLE [dbo].[afspraak_betreft_account]  WITH NOCHECK ADD  CONSTRAINT [FK_afspraak_betreft_account.crm_afspraak_id] FOREIGN KEY([crm_afspraak_id])
REFERENCES [dbo].[afspraak_alle] ([crm_afspraak_id])
GO
ALTER TABLE [dbo].[afspraak_betreft_account] NOCHECK CONSTRAINT [FK_afspraak_betreft_account.crm_afspraak_id]
GO
ALTER TABLE [dbo].[afspraak_betreft_contact]  WITH NOCHECK ADD  CONSTRAINT [FK_afspraak_betreft_contact.crm_afspraak_id] FOREIGN KEY([crm_afspraak_id])
REFERENCES [dbo].[afspraak_alle] ([crm_afspraak_id])
GO
ALTER TABLE [dbo].[afspraak_betreft_contact] NOCHECK CONSTRAINT [FK_afspraak_betreft_contact.crm_afspraak_id]
GO
ALTER TABLE [dbo].[afspraak_betreft_contact]  WITH NOCHECK ADD  CONSTRAINT [FK_afspraak_betreft_contact.crm_contact_id] FOREIGN KEY([crm_contact_id])
REFERENCES [dbo].[contact] ([crm_contact_id])
GO
ALTER TABLE [dbo].[afspraak_betreft_contact] NOCHECK CONSTRAINT [FK_afspraak_betreft_contact.crm_contact_id]
GO
ALTER TABLE [dbo].[cdi_pageview]  WITH NOCHECK ADD  CONSTRAINT [FK_cdi_pageview.campagne_id] FOREIGN KEY([campagne_id])
REFERENCES [dbo].[campagne] ([crm_campagne_id])
GO
ALTER TABLE [dbo].[cdi_pageview] NOCHECK CONSTRAINT [FK_cdi_pageview.campagne_id]
GO
ALTER TABLE [dbo].[cdi_pageview]  WITH NOCHECK ADD  CONSTRAINT [FK_cdi_pageview.visit_id] FOREIGN KEY([visit_id])
REFERENCES [dbo].[cdi_visit] ([crm_cdi_visit_id])
GO
ALTER TABLE [dbo].[cdi_pageview] NOCHECK CONSTRAINT [FK_cdi_pageview.visit_id]
GO
ALTER TABLE [dbo].[cdi_pageview]  WITH NOCHECK ADD  CONSTRAINT [FK_cdi_pageview.web_content] FOREIGN KEY([web_content])
REFERENCES [dbo].[cdi_web_content] ([crm_cdi_webcontent_web_content])
GO
ALTER TABLE [dbo].[cdi_pageview] NOCHECK CONSTRAINT [FK_cdi_pageview.web_content]
GO
ALTER TABLE [dbo].[cdi_sentemail_kliks]  WITH NOCHECK ADD  CONSTRAINT [FK_cdi_sentemail_kliks.crm_cdi_mailing_id] FOREIGN KEY([crm_cdi_mailing_id])
REFERENCES [dbo].[cdi_mailing] ([crm_cdi_mailing_id])
GO
ALTER TABLE [dbo].[cdi_sentemail_kliks] NOCHECK CONSTRAINT [FK_cdi_sentemail_kliks.crm_cdi_mailing_id]
GO
ALTER TABLE [dbo].[cdi_sentemail_kliks]  WITH NOCHECK ADD  CONSTRAINT [FK_cdi_sentemail_kliks.crm_persoon_id] FOREIGN KEY([crm_persoon_id])
REFERENCES [dbo].[persoon] ([crm_persoon_id])
GO
ALTER TABLE [dbo].[cdi_sentemail_kliks] NOCHECK CONSTRAINT [FK_cdi_sentemail_kliks.crm_persoon_id]
GO
ALTER TABLE [dbo].[cdi_visit]  WITH NOCHECK ADD  CONSTRAINT [FK_cdi_visit.crm_campagne_id] FOREIGN KEY([crm_campagne_id])
REFERENCES [dbo].[campagne] ([crm_campagne_id])
GO
ALTER TABLE [dbo].[cdi_visit] NOCHECK CONSTRAINT [FK_cdi_visit.crm_campagne_id]
GO
ALTER TABLE [dbo].[cdi_visit]  WITH NOCHECK ADD  CONSTRAINT [FK_cdi_visit.crm_persoon_id] FOREIGN KEY([crm_persoon_id])
REFERENCES [dbo].[persoon] ([crm_persoon_id])
GO
ALTER TABLE [dbo].[cdi_visit] NOCHECK CONSTRAINT [FK_cdi_visit.crm_persoon_id]
GO
ALTER TABLE [dbo].[contact]  WITH NOCHECK ADD  CONSTRAINT [FK_contact.crm_contact_account] FOREIGN KEY([crm_contact_account])
REFERENCES [dbo].[account] ([crm_account_id])
GO
ALTER TABLE [dbo].[contact] NOCHECK CONSTRAINT [FK_contact.crm_contact_account]
GO
ALTER TABLE [dbo].[contact]  WITH NOCHECK ADD  CONSTRAINT [FK_contact.crm_persoon_id] FOREIGN KEY([crm_persoon_id])
REFERENCES [dbo].[persoon] ([crm_persoon_id])
GO
ALTER TABLE [dbo].[contact] NOCHECK CONSTRAINT [FK_contact.crm_persoon_id]
GO
ALTER TABLE [dbo].[contactfunctie]  WITH NOCHECK ADD  CONSTRAINT [FK_contactfunctie.crm_contact_id] FOREIGN KEY([crm_contact_id])
REFERENCES [dbo].[contact] ([crm_contact_id])
GO
ALTER TABLE [dbo].[contactfunctie] NOCHECK CONSTRAINT [FK_contactfunctie.crm_contact_id]
GO
ALTER TABLE [dbo].[contactfunctie]  WITH NOCHECK ADD  CONSTRAINT [FK_contactfunctie.crm_functie_id] FOREIGN KEY([crm_functie_id])
REFERENCES [dbo].[functie] ([crm_functie_id])
GO
ALTER TABLE [dbo].[contactfunctie] NOCHECK CONSTRAINT [FK_contactfunctie.crm_functie_id]
GO
ALTER TABLE [dbo].[financiele_data]  WITH NOCHECK ADD  CONSTRAINT [FK_financiele_data.crm_account_id] FOREIGN KEY([crm_account_id])
REFERENCES [dbo].[account] ([crm_account_id])
GO
ALTER TABLE [dbo].[financiele_data] NOCHECK CONSTRAINT [FK_financiele_data.crm_account_id]
GO
ALTER TABLE [dbo].[info_en_klachten]  WITH NOCHECK ADD  CONSTRAINT [FK_info_en_klachten.crm_account_id] FOREIGN KEY([crm_account_id])
REFERENCES [dbo].[account] ([crm_account_id])
GO
ALTER TABLE [dbo].[info_en_klachten] NOCHECK CONSTRAINT [FK_info_en_klachten.crm_account_id]
GO
ALTER TABLE [dbo].[info_en_klachten]  WITH NOCHECK ADD  CONSTRAINT [FK_info_en_klachten.crm_gebruikers_id] FOREIGN KEY([crm_gebruikers_id])
REFERENCES [dbo].[gebruikers] ([crm_gebruikers_id])
GO
ALTER TABLE [dbo].[info_en_klachten] NOCHECK CONSTRAINT [FK_info_en_klachten.crm_gebruikers_id]
GO
ALTER TABLE [dbo].[inschrijving]  WITH NOCHECK ADD  CONSTRAINT [FK_inschrijving.crm_contact_id] FOREIGN KEY([crm_contact_id])
REFERENCES [dbo].[contact] ([crm_contact_id])
GO
ALTER TABLE [dbo].[inschrijving] NOCHECK CONSTRAINT [FK_inschrijving.crm_contact_id]
GO
ALTER TABLE [dbo].[lidmaatschap]  WITH NOCHECK ADD  CONSTRAINT [FK_lidmaatschap.crm_account_id] FOREIGN KEY([crm_account_id])
REFERENCES [dbo].[account] ([crm_account_id])
GO
ALTER TABLE [dbo].[lidmaatschap] NOCHECK CONSTRAINT [FK_lidmaatschap.crm_account_id]
GO
ALTER TABLE [dbo].[sessie]  WITH NOCHECK ADD  CONSTRAINT [FK_sessie.crm_campagne_id] FOREIGN KEY([crm_campagne_id])
REFERENCES [dbo].[campagne] ([crm_campagne_id])
GO
ALTER TABLE [dbo].[sessie] NOCHECK CONSTRAINT [FK_sessie.crm_campagne_id]
GO
ALTER TABLE [dbo].[sessie_inschrijving]  WITH NOCHECK ADD  CONSTRAINT [FK_sessie_inschrijving.crm_inschrijving_id] FOREIGN KEY([crm_inschrijving_id])
REFERENCES [dbo].[inschrijving] ([crm_inschrijving_id])
GO
ALTER TABLE [dbo].[sessie_inschrijving] NOCHECK CONSTRAINT [FK_sessie_inschrijving.crm_inschrijving_id]
GO
ALTER TABLE [dbo].[sessie_inschrijving]  WITH NOCHECK ADD  CONSTRAINT [FK_sessie_inschrijving.crm_sessie_id] FOREIGN KEY([crm_sessie_id])
REFERENCES [dbo].[sessie] ([crm_sessie_id])
GO
ALTER TABLE [dbo].[sessie_inschrijving] NOCHECK CONSTRAINT [FK_sessie_inschrijving.crm_sessie_id]
GO
"""

# Initialize an empty list to store the JSON data
json_data = []

# Regular expression pattern to extract table names and foreign key relationships

# Find all matches in the SQL statements
matches = sql_statements.split("GO")
matches = [match.strip() for match in matches]
matches = [match for match in matches if match != ""]

# Loop through all matches
for match in matches:
    if "REFERENCES" not in match:
        continue

    table_name_pk = ""  # In the first example: This would be "account"
    table_name_fk = ""  # In the first example: This would be "account_activiteitscode"
    foreign_key = ""  # In the first example: This would be "crm_account_id"
    primary_key = ""  # In the first example: This would be "crm_account_id"

    # Extract the table names
    table_name_fk = match.split("WITH NOCHECK ADD")[0].split("[dbo].[")[1].split("]")[0]
    table_name_pk = match.split("REFERENCES [dbo].[")[1].split("]")[0]

    # Extract the foreign key
    foreign_key = match.split("FOREIGN KEY([")[1].split("])")[0]

    # Extract the primary key
    primary_key = match.split(f"REFERENCES [dbo].[{table_name_pk}] ([")[1].split("])")[
        0
    ]

    print(primary_key, foreign_key, table_name_pk, table_name_fk)

    # add to json_data
    table_name_pks = [o["table"] for o in json_data]

    if table_name_pk not in table_name_pks:
        json_data.append(
            {
                "table": table_name_pk,
                "primary_key": primary_key,
                "foreign_keys": [{"foreign_key": foreign_key, "table": table_name_fk}],
            }
        )
    else:
        for table in json_data:
            if table["table"] == table_name_pk:
                table["foreign_keys"].append(
                    {"foreign_key": foreign_key, "table": table_name_fk}
                )

# Save the JSON data to a file
file = os.getenv("IMPORT_FOREIGN_KEYS_PATH")
with open(file, "w") as f:
    json.dump(json_data, f, indent=2)
print(f"JSON data has been saved to '{file}'")

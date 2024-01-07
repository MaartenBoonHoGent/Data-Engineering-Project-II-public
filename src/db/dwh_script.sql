CREATE DATABASE IF NOT EXISTS dwh;
USE dwh;
CREATE TABLE IF NOT EXISTS DimDate (
    date_key INT PRIMARY KEY,
    date DATE NOT NULL,
    day_of_week INT NOT NULL,
    day_of_month INT NOT NULL,
    day_of_year INT NOT NULL,
    year INT NOT NULL,
    isWeekDay BIT NOT NULL,
    dayName VARCHAR(10) NOT NULL,
    monthName VARCHAR(10) NOT NULL,
    numberOfQuarter INT NOT NULL,
    nameOfQuarter VARCHAR(10) NOT NULL,
    isWeekend BIT NOT NULL,
    isHoliday BIT NOT NULL
);
CREATE TABLE IF NOT EXISTS DimCampagne (
    crm_campagne_id INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
    crm_campagne_campagne_nr VARCHAR(100) NOT NULL,
    crm_campagne_einddatum DATE NOT NULL,
    crm_campagne_naam VARCHAR(100) NOT NULL,
    crm_campagne_naam_in_email VARCHAR(100) NOT NULL,
    crm_campagne_reden_van_status VARCHAR(100) NOT NULL,
    crm_campagne_startdatum DATE NOT NULL,
    crm_campagne_status VARCHAR(100) NOT NULL,
    crm_campagne_type_campagne VARCHAR(100) NOT NULL,
    crm_campagne_url_voka_be VARCHAR(100) NOT NULL,
    crm_campagne_soort_campagne VARCHAR(100) NOT NULL
);
CREATE TABLE IF NOT EXISTS DimVisits (
    crm_cdi_visit_id INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
    crm_cdi_visit_adobe_reader VARCHAR(50) NOT NULL,
    crm_cdi_visit_bounce VARCHAR(50) NOT NULL,
    crm_cdi_visit_browser VARCHAR(50) NOT NULL,
    crm_cdi_visit_campaigne_code VARCHAR(50) NOT NULL,
    crm_cdi_visit_ip_stad VARCHAR(100) NOT NULL,
    crm_cdi_visit_company VARCHAR(100) NOT NULL,
    crm_cdi_visit_contact_naam VARCHAR(100) NOT NULL,
    crm_cdi_visit_containsocialprofile VARCHAR(50) NOT NULL,
    crm_cdi_visit_ip_land VARCHAR(50) NOT NULL,
    crm_cdi_visit_duration double NOT NULL,
    crm_cdi_visit_ended_on DATE NOT NULL,
    crm_cdi_visit_entry_page VARCHAR(255),
    crm_cdi_visit_exit_page VARCHAR(255),
    crm_cdi_visit_first_visit VARCHAR(50),
    crm_cdi_visit_ip_address VARCHAR(15),
    crm_cdi_visit_ip_organization VARCHAR(100),
    crm_cdi_visit_keywords VARCHAR(255),
    crm_cdi_visit_ip_latitude double NOT NULL,
    crm_cdi_visit_ip_longitude double NOT NULL,
    crm_cdi_visit_operating_system VARCHAR(50),
    crm_cdi_visit_ip_postcode VARCHAR(20),
    crm_cdi_visit_referrer VARCHAR(255),
    crm_cdi_visit_referring_host VARCHAR(255),
    crm_cdi_visit_score double NOT NULL,
    crm_cdi_visit_started_on DATE NOT NULL,
    crm_cdi_visit_ip_status VARCHAR(20),
    crm_cdi_visit_time double NOT NULL,
    crm_cdi_visit_total_pages INT,
    crm_cdi_visit_gewijzigd_op DATE NOT NULL,
    crm_cdi_visit_aangemaakt_op DATE NOT NULL,
    FOREIGN KEY (crm_cdi_visit_id) REFERENCES DimVisits(crm_cdi_visit_id)
);
CREATE TABLE IF NOT EXISTS DimAfspraak (
    crm_afspraak_id INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
    crm_afspraak_betref_account_thema VARCHAR(255),
    crm_afspraak_betref_account_subthema VARCHAR(255),
    crm_afspraak_betref_account_onderwerp VARCHAR(255),
    crm_afspraak_betref_account_eindtijd DATE NOT NULL,
    crm_afspraak_betref_account_keyphrases VARCHAR(255),
    crm_afspraak_account_gelinkt_thema VARCHAR(255),
    crm_afspraak_account_gelinkt_subthema VARCHAR(255),
    crm_afspraak_account_gelinkt_onderwerp VARCHAR(255),
    crm_afspraak_account_gelinkt_eindtijd DATE NOT NULL,
    crm_afspraak_account_gelinkt_keyphrases VARCHAR(255),
    crm_afspraak_betref_contactfiche_thema VARCHAR(255),
    crm_afspraak_betref_contactfiche_subthema VARCHAR(255),
    crm_afspraak_betref_contactfiche_onderwerp VARCHAR(255),
    crm_afspraak_betref_contactfiche_eindtijd DATE NOT NULL,
    crm_afspraak_betref_contactfiche_keyphrases VARCHAR(255),
    crm_account_id INT NOT NULL,
);
CREATE TABLE IF NOT EXISTS DimPageview (
    pageview_id INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
    crm_cdi_visit_id INT NOT NULL,
    anonymous_visitor DOUBLE NULL,
    browser VARCHAR(25) NOT NULL,
    duration DOUBLE NULL,
    operating_system VARCHAR(100) NOT NULL,
    referrer_type VARCHAR(100) NOT NULL,
    time DATE NOT NULL,
    page_title VARCHAR(255) NOT NULL,
    type VARCHAR(100) NOT NULL,
    url VARCHAR(200) NOT NULL,
    viewed_on DATE NOT NULL,
    visitor_key VARCHAR(100) NOT NULL,
    web_content VARCHAR(100) NULL,
    created_on DATE NOT NULL,
    modified_by VARCHAR(150) NOT NULL,
    modified_on DATE NOT NULL,
    status VARCHAR(50) NOT NULL,
    status_reason VARCHAR(100) NOT NULL,
);
CREATE TABLE IF NOT EXISTS DimContact (
    crm_contact_id INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
    crm_activiteitvereistcontact_reqattendee VARCHAR(100) NOT NULL,
    crm_contact_account VARCHAR(100) NOT NULL,
    crm_contact_functietitel VARCHAR(100) NOT NULL,
    crm_contact_status VARCHAR(100) NOT NULL,
    crm_contact_voka_medewerker VARCHAR(100) NOT NULL
);
CREATE TABLE IF NOT EXISTS DimPersoon (
    crm_persoon_id INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
    crm_persoon_persoon VARCHAR(100) NOT NULL,
    crm_persoon_persoonsnr INT NOT NULL,
    crm_persoon_reden_van_status VARCHAR(100) NOT NULL,
    crm_persoon_web_login VARCHAR(100) NOT NULL,
    crm_persoon_mail_regio_antwerpen_waasland VARCHAR(100) NOT NULL,
    crm_persoon_mail_regio_brussel_hoofdstedelijk_gewest VARCHAR(100) NOT NULL,
    crm_persoon_mail_regio_limburg VARCHAR(100) NOT NULL,
    crm_persoon_mail_regio_mechelen_kempen VARCHAR(100) NOT NULL,
    crm_persoon_mail_regio_oost_vlaanderen VARCHAR(100) NOT NULL,
    crm_persoon_mail_regio_vlaams_brabant VARCHAR(100) NOT NULL,
    crm_persoon_mail_regio_voka_nationaal VARCHAR(100) NOT NULL,
    crm_persoon_mail_regio_west_vlaanderen VARCHAR(100) NOT NULL,
    crm_persoon_mail_thema_duurzaamheid VARCHAR(100) NOT NULL,
    crm_persoon_mail_thema_financieel_fiscaal VARCHAR(100) NOT NULL,
    crm_persoon_mail_thema_innovatie VARCHAR(100) NOT NULL,
    crm_persoon_mail_thema_INTernationaal_ondernemen VARCHAR(100) NOT NULL,
    crm_persoon_mail_thema_mobiliteit VARCHAR(100) NOT NULL,
    crm_persoon_mail_thema_omgeving VARCHAR(100) NOT NULL,
    crm_persoon_mail_thema_sales_marketing_communicatie VARCHAR(100) NOT NULL,
    crm_persoon_mail_thema_strategie_en_algemeen_management VARCHAR(100) NOT NULL,
    crm_persoon_mail_thema_talent VARCHAR(100) NOT NULL,
    crm_persoon_mail_thema_welzijn VARCHAR(100) NOT NULL,
    crm_persoon_mail_type_bevraging VARCHAR(100) NOT NULL,
    crm_persoon_mail_type_communities_en_projecten VARCHAR(100) NOT NULL,
    crm_persoon_mail_type_netwerkevenementen VARCHAR(100) NOT NULL,
    crm_persoon_mail_type_nieuwsbrieven VARCHAR(100) NOT NULL,
    crm_persoon_mail_type_opleidingen VARCHAR(100) NOT NULL,
    crm_persoon_mail_type_persberichten_belangrijke_meldingen VARCHAR(100) NOT NULL,
    crm_persoon_marketingcommunicatie VARCHAR(100) NOT NULL
);
CREATE TABLE IF NOT EXISTS DimMailing (
    crm_cdi_mailing_id INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
    crm_cdi_mailing_name VARCHAR(100) NOT NULL,
    crm_cdi_mailing_sent_on DATE NOT NULL,
    crm_cdi_mailing_onderwerp VARCHAR(100) NOT NULL
);
CREATE TABLE IF NOT EXISTS DimSenteMailKliks (
    crm_cdi_sentemail_kliks_sent_email INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
    crm_cdi_sentemailkliks INT NOT NULL,
    crm_cdi_mailing_id INT NOT NULL,
    FOREIGN KEY (crm_cdi_mailing_id) REFERENCES DimMailing(crm_cdi_mailing_id)
);
CREATE TABLE IF NOT EXISTS DimSessie (
    crm_sessie_id INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
    crm_sessie_activiteitstype VARCHAR(150) NOT NULL,
    crm_sessie_eind_datum_tijd DATE NOT NULL,
    crm_sessie_product VARCHAR(150) NOT NULL,
    crm_sessie_sessie_nr VARCHAR(100) NOT NULL,
    crm_sessie_start_datum_tijd DATE NOT NULL,
    crm_sessie_thema_naam VARCHAR(100) NOT NULL,
    crm_sessieinschrijving_sessieinschrijving VARCHAR(150) NOT NULL,
    crm_campagne_id INT NOT NULL,
    FOREIGN KEY (crm_campagne_id) REFERENCES DimCampagne(crm_campagne_id)
);
CREATE TABLE IF NOT EXISTS FactInschrijving (
    crm_inschrijving_id INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
    crm_inschrijving_facturatie_bedrag DOUBLE NOT NULL,
    crm_sessie_id INT NOT NULL,
    crm_persoon_id INT NOT NULL,
    crm_contact_id INT NOT NULL,
    crm_afspraak_id INT NOT NULL,
    FOREIGN KEY (crm_sessie_id) REFERENCES DimSessie(crm_sessie_id),
    FOREIGN KEY (crm_persoon_id) REFERENCES DimPersoon(crm_persoon_id),
    FOREIGN KEY (crm_contact_id) REFERENCES DimContact(crm_contact_id),
    FOREIGN KEY (crm_afspraak_id) REFERENCES DimAfspraak(crm_afspraak_id)
);

CREATE TABLE IF NOT EXISTS DimAccount (
    crm_account_id INT PRIMARY KEY,
    crm_account_adres_geografische_regio VARCHAR(255),
    crm_account_adres_geografische_subregio VARCHAR(255),
    crm_account_adres_plaats VARCHAR(255),
    crm_account_adres_postcode VARCHAR(10),
    crm_account_adres_provincie VARCHAR(255),
    crm_account_adres_land VARCHAR(255),
    crm_account_industriezone_naam VARCHAR(255),
    crm_account_is_voka_entiteit BOOLEAN,
    crm_account_ondernemingsvorm VARCHAR(255),
    crm_account_ondernemingstype VARCHAR(255),
    crm_account_oprichtingsdatum DATE,
    crm_account_primaire_activiteit VARCHAR(255),
    crm_account_reden_van_status VARCHAR(255),
    crm_account_status VARCHAR(50),
    crm_account_voka_nr VARCHAR(50),
    crm_account_hoofd_nace_code VARCHAR(10)
    FOREIGN KEY (crm_account_id) REFERENCES DimAfspraak(crm_account_id)
);
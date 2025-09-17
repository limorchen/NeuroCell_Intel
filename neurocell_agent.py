import os
import sqlite3
import smtplib
import logging
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

from Bio import Entrez
import requests
import pandas as pd

# ---------------------------------------------------
# Logging
# ---------------------------------------------------
logging.basicConfig(
    filename="neurocell_agent.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ---------------------------------------------------
# Environment variables
# ---------------------------------------------------
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
SENDER_EMAIL = os.getenv("SENDER_EMAIL")
RECIPIENT_EMAIL = os.getenv("RECIPIENT_EMAIL")
NCBI_EMAIL = os.getenv("NCBI_EMAIL")
PUBMED_TERM = os.getenv("PUBMED_TERM", "exosomes AND CNS")
MAX_RECORDS = int(os.getenv("MAX_RECORDS", 20))

DB_FILE = "neurocell_database.db"

Entrez.email = NCBI_EMAIL

# ---------------------------------------------------
# Database setup
# ---------------------------------------------------
def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS pubmed_articles (
            pmid TEXT PRIMARY KEY,
            title TEXT,
            abstract TEXT,
            journal TEXT,
            pub_date TEXT,
            link TEXT,
            retrieved_at TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS clinical_trials (
            nct_id TEXT PRIMARY KEY,
            title TEXT,
            status TEXT,
            conditions TEXT,
            interventions TEXT,
            last_update TEXT,
            link TEXT,
            retrieved_at TEXT
        )
    """)

    conn.commit()
    conn.close()

# ---------------------------------------------------
# PubMed fetch
# ---------------------------------------------------
def fetch_pubmed(term, days_back=7, max_records=50):
    logging.info("Fetching PubMed articles...")
    since_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y/%m/%d")

    handle = Entrez.esearch(db="pubmed", term=term, retmax=max_records, mindate=since_date, datetype="pdat")
    record = Entrez.read(handle)
    handle.close()

    ids = record.get("IdList", [])
    articles = []

    if not ids:
        logging.info("No PubMed results found.")
        return []

    handle = Entrez.efetch(db="pubmed", id=",".join(ids), rettype="medline", retmode="xml")
    records = Entrez.read(handle)
    handle.close()

    for r in records["PubmedArticle"]:
        pmid = r["MedlineCitation"]["PMID"]
        title = r["MedlineCitation"]["Article"]["ArticleTitle"]
        abstract = r["MedlineCitation"]["Article"].get("Abstract", {}).get("AbstractText", [""])[0]
        journal = r["MedlineCitation"]["Article"]["Journal"]["Title"]
        pub_date = r["MedlineCitation"]["Article"]["Journal"]["JournalIssue"]["PubDate"].get("Year", "")
        link = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

        articles.append((pmid, title, abstract, journal, pub_date, link, datetime.now().isoformat()))

    return articles

# ---------------------------------------------------
# ClinicalTrials.gov fetch
# ---------------------------------------------------
def fetch_clinical_trials(term, days_back=7, max_records=50):
    logging.info("Fetching ClinicalTrials.gov studies...")
    base_url = "https://clinicaltrials.gov/api/query/study_fields"
    since_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    params = {
        "expr": term,
        "fields": "NCTId,BriefTitle,OverallStatus,Condition,Intervention,LastUpdatePostDate",
        "min_rnk": 1,
        "max_rnk": max_records,
        "fmt": "json"
    }

    response = requests.get(base_url, params=params)
    if response.status_code != 200:
        logging.error(f"ClinicalTrials.gov API error: {response.text}")
        return []

    studies = response.json().get("StudyFieldsResponse", {}).get("StudyFields", [])
    results = []

    for s in studies:
        nct_id = s.get("NCTId", [""])[0]
        title = s.get("BriefTitle", [""])[0]
        status = s.get("OverallStatus", [""])[0]
        conditions = "; ".join(s.get("Condition", []))
        interventions = "; ".join(s.get("Intervention", []))
        last_update = s.get("LastUpdatePostDate", [""])[0]
        link = f"https://clinicaltrials.gov/study/{nct_id}"

        # filter by last update date
        try:
            if last_update and last_update >= since_date:
                results.append((nct_id, title, status, conditions, interventions, last_update, link, datetime.now().isoformat()))
        except Exception:
            pass

    return results

# ---------------------------------------------------
# Save to DB & detect new
# ---------------------------------------------------
def save_and_detect_new(data, table, key_field):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    new_records = []

    for row in data:
        key = row[0]
        cursor.execute(f"SELECT 1 FROM {table} WHERE {key_field}=?", (key,))
        if cursor.fetchone() is None:
            cursor.execute(f"INSERT INTO {table} VALUES ({','.join(['?']*len(row))})", row)
            new_records.append(row)

    conn.commit()
    conn.close()
    return new_records

# ---------------------------------------------------
# Send email
# ---------------------------------------------------
def send_email(new_pubmed, new_trials, stats):
    msg = MIMEMultipart()
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECIPIENT_EMAIL
    msg["Subject"] = "Weekly NeuroCell Agent Report"

    body = f"""
    Weekly NeuroCell Agent Report

    PubMed:
    - Total in DB: {stats['pubmed_total']}
    - New this week: {stats['pubmed_new']}

    ClinicalTrials.gov:
    - Total in DB: {stats['trials_total']}
    - New this week: {stats['trials_new']}

    See attached CSVs for details (only if new results).
    """
    msg.attach(MIMEText(body, "plain"))

    # attach CSVs if new
    if new_pubmed:
        df_pubmed = pd.DataFrame(new_pubmed, columns=["PMID", "Title", "Abstract", "Journal", "PubDate", "Link", "RetrievedAt"])
        df_pubmed.to_csv("new_pubmed_this_week.csv", index=False)
        attach_file(msg, "new_pubmed_this_week.csv")

    if new_trials:
        df_trials = pd.DataFrame(new_trials, columns=["NCTId", "Title", "Status", "Conditions", "Interventions", "LastUpdate", "Link", "RetrievedAt"])
        df_trials.to_csv("new_trials_this_week.csv", index=False)
        attach_file(msg, "new_trials_this_week.csv")

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(SENDER_EMAIL, EMAIL_PASSWORD)
        server.sendmail(SENDER_EMAIL, RECIPIENT_EMAIL.split(","), msg.as_string())

def attach_file(msg, filepath):
    with open(filepath, "rb") as f:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(f.read())
    encoders.encode_base64(part)
    part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(filepath)}")
    msg.attach(part)

# ---------------------------------------------------
# Main
# ---------------------------------------------------
def main():
    init_db()

    # PubMed
    pubmed_results = fetch_pubmed(PUBMED_TERM, days_back=7, max_records=MAX_RECORDS)
    new_pubmed = save_and_detect_new(pubmed_results, "pubmed_articles", "pmid")

    # ClinicalTrials
    trials_results = fetch_clinical_trials(PUBMED_TERM, days_back=7, max_records=MAX_RECORDS)
    new_trials = save_and_detect_new(trials_results, "clinical_trials", "nct_id")

    # Stats
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM pubmed_articles")
    pubmed_total = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM clinical_trials")
    trials_total = cursor.fetchone()[0]
    conn.close()

    stats = {
        "pubmed_total": pubmed_total,
        "pubmed_new": len(new_pubmed),
        "trials_total": trials_total,
        "trials_new": len(new_trials)
    }

    logging.info(f"Run completed. Stats: {stats}")
    send_email(new_pubmed, new_trials, stats)


if __name__ == "__main__":
    main()

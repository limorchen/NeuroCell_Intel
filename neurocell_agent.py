import os
import time
import hashlib
import logging
import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict
import pandas as pd
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from dotenv import load_dotenv
from Bio import Entrez

load_dotenv()

# ---------------------
# Logging configuration
# ---------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('neurocell_agent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ---------------------
# Configuration
# ---------------------
class Config:
    NCBI_EMAIL = os.getenv("NCBI_EMAIL", "chen.limor@gmail.com")
    SENDER_EMAIL = os.getenv("SENDER_EMAIL", "limor@nurexone.com")
    RECIPIENT_EMAIL = os.getenv("RECIPIENT_EMAIL", "limor@nurexone.com")
    SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    SMTP_PORT = int(os.getenv("SMTP_PORT", 465))
    EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
    PUBMED_TERM = os.getenv("PUBMED_TERM", "exosomes AND CNS")
    TRIALS_TERM = os.getenv("TRIALS_TERM", "exosomes CNS")
    MAX_RECORDS = int(os.getenv("MAX_RECORDS", 20))
    DB_FILE = "neurocell_database.db"

config = Config()

# ---------------------
# Utility
# ---------------------
def compute_content_hash(text: str) -> str:
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def current_timestamp() -> str:
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# ---------------------
# Database
# ---------------------
class NeuroCellDB:
    def __init__(self, db_path=config.DB_FILE):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self._create_tables()

    def _create_tables(self):
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS pubmed_articles (
            pmid TEXT PRIMARY KEY,
            title TEXT,
            abstract TEXT,
            authors TEXT,
            publication_date TEXT,
            journal TEXT,
            doi TEXT,
            url TEXT,
            is_new INTEGER,
            content_hash TEXT,
            first_seen TEXT
        );
        """)
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS clinical_trials (
            nct_id TEXT PRIMARY KEY,
            title TEXT,
            status TEXT,
            conditions TEXT,
            interventions TEXT,
            start_date TEXT,
            completion_date TEXT,
            url TEXT,
            is_new INTEGER,
            content_hash TEXT,
            first_seen TEXT
        );
        """)
        self.conn.commit()

    def mark_all_old(self):
        self.cursor.execute("UPDATE pubmed_articles SET is_new=0;")
        self.cursor.execute("UPDATE clinical_trials SET is_new=0;")
        self.conn.commit()

    def upsert_pubmed_article(self, article):
        hash_val = compute_content_hash(article['title'] + article['abstract'])
        self.cursor.execute("SELECT pmid, content_hash FROM pubmed_articles WHERE pmid=?", (article['pmid'],))
        row = self.cursor.fetchone()
        now = current_timestamp()
        if not row:
            self.cursor.execute("""
                INSERT INTO pubmed_articles
                (pmid, title, abstract, authors, publication_date, journal, doi, url, is_new, content_hash, first_seen)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?);
            """, (
                article['pmid'], article['title'], article['abstract'], article['authors'], article['publication_date'],
                article['journal'], article['doi'], article['url'], hash_val, now
            ))
            logger.info(f"Added NEW PubMed article: {article['pmid']}")
        else:
            if row[1] != hash_val:
                self.cursor.execute("""
                    UPDATE pubmed_articles
                    SET title=?, abstract=?, authors=?, publication_date=?, journal=?, doi=?, url=?, is_new=1, content_hash=?
                    WHERE pmid=?;
                """, (
                    article['title'], article['abstract'], article['authors'], article['publication_date'],
                    article['journal'], article['doi'], article['url'], hash_val, article['pmid']
                ))
                logger.info(f"Updated PubMed article: {article['pmid']}")
            else:
                self.cursor.execute("UPDATE pubmed_articles SET is_new=0 WHERE pmid=?;", (article['pmid'],))
        self.conn.commit()

    def upsert_clinical_trial(self, trial):
        hash_val = compute_content_hash(trial['title'] + trial['status'])
        self.cursor.execute("SELECT nct_id, content_hash FROM clinical_trials WHERE nct_id=?", (trial['nct_id'],))
        row = self.cursor.fetchone()
        now = current_timestamp()
        if not row:
            self.cursor.execute("""
                INSERT INTO clinical_trials
                (nct_id, title, status, conditions, interventions, start_date, completion_date, url, is_new, content_hash, first_seen)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?);
            """, (
                trial['nct_id'], trial['title'], trial['status'], trial['conditions'], trial['interventions'],
                trial['start_date'], trial['completion_date'], trial['url'], hash_val, now
            ))
            logger.info(f"Added NEW Clinical Trial: {trial['nct_id']}")
        else:
            if row[1] != hash_val:
                self.cursor.execute("""
                    UPDATE clinical_trials
                    SET title=?, status=?, conditions=?, interventions=?, start_date=?, completion_date=?, url=?, is_new=1, content_hash=?
                    WHERE nct_id=?;
                """, (
                    trial['title'], trial['status'], trial['conditions'], trial['interventions'],
                    trial['start_date'], trial['completion_date'], trial['url'], hash_val, trial['nct_id']
                ))
                logger.info(f"Updated Clinical Trial: {trial['nct_id']}")
            else:
                self.cursor.execute("UPDATE clinical_trials SET is_new=0 WHERE nct_id=?;", (trial['nct_id'],))
        self.conn.commit()

    def export_csv(self, table_name, filename, new_only=False):
        query = f"SELECT * FROM {table_name}"
        if new_only:
            query += " WHERE is_new=1"
        df = pd.read_sql_query(query, self.conn)
        df.to_csv(filename, index=False)
        logger.info(f"Exported data from {table_name} to {filename}")
        return filename

    def close(self):
        self.conn.close()

# ---------------------
# PubMed Fetcher
# ---------------------
class PubMedFetcher:
    def __init__(self, email: str):
        Entrez.email = email

    def fetch_articles(self, term: str, max_records: int = 20, days_back: int = 7) -> List[Dict]:
        date_filter = (datetime.now() - timedelta(days=days_back)).strftime("%Y/%m/%d")
        search_term = f'{term} AND ("{date_filter}"[Date - Publication] : "3000"[Date - Publication])'
        logger.info(f"Searching PubMed with term: {search_term}")
        try:
            handle = Entrez.esearch(db="pubmed", term=search_term, retmax=max_records)
            record = Entrez.read(handle)
            handle.close()
            ids = record.get("IdList", [])
            if not ids:
                return []
            handle = Entrez.efetch(db="pubmed", id=",".join(ids), rettype="xml")
            articles = Entrez.read(handle).get("PubmedArticle", [])
            handle.close()
            results = []
            for art in articles:
                pmid = art['MedlineCitation']['PMID']
                article_data = art['MedlineCitation']['Article']
                title = article_data.get("ArticleTitle", "")
                abstract_list = article_data.get("Abstract", {}).get("AbstractText", [])
                abstract = " ".join(abstract_list) if isinstance(abstract_list, list) else str(abstract_list)
                authors_list = article_data.get("AuthorList", [])
                authors = ", ".join([f"{a.get('LastName','')} {a.get('ForeName','')}".strip() for a in authors_list])
                journal = article_data.get("Journal", {}).get("Title", "")
                pub_date = article_data.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {}).get("Year", "")
                results.append({
                    "pmid": pmid,
                    "title": title,
                    "abstract": abstract,
                    "authors": authors,
                    "publication_date": pub_date,
                    "journal": journal,
                    "doi": "",
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                })
            return results
        except Exception as e:
            logger.error(f"Error fetching PubMed articles: {e}")
            return []

# ---------------------
# ClinicalTrials.gov Fetcher
# ---------------------
class ClinicalTrialsFetcher:
    def fetch_trials(self, term: str, max_records: int = 20) -> List[Dict]:
        base_url = "https://clinicaltrials.gov/api/query/study_fields"
        params = {
            "expr": term,
            "fields": "NCTId,BriefTitle,OverallStatus,Condition,InterventionName,StartDate,CompletionDate",
            "min_rnk": 1,
            "max_rnk": max_records,
            "fmt": "json"
        }
        try:
            response = requests.get(base_url, params=params, timeout=20)
            response.raise_for_status()
            data = response.json()
            studies = data["StudyFieldsResponse"]["StudyFields"]
            results = []
            for s in studies:
                results.append({
                    "nct_id": s["NCTId"][0] if s["NCTId"] else "",
                    "title": s["BriefTitle"][0] if s["BriefTitle"] else "",
                    "status": s["OverallStatus"][0] if s["OverallStatus"] else "",
                    "conditions": ", ".join(s.get("Condition", [])),
                    "interventions": ", ".join(s.get("InterventionName", [])),
                    "start_date": s["StartDate"][0] if s["StartDate"] else "",
                    "completion_date": s["CompletionDate"][0] if s["CompletionDate"] else "",
                    "url": f"https://clinicaltrials.gov/study/{s['NCTId'][0]}" if s["NCTId"] else ""
                })
            return results
        except Exception as e:
            logger.error(f"Error fetching ClinicalTrials.gov: {e}")
            return []

# ---------------------
# Email reporting
# ---------------------
def send_email(subject: str, body: str, attachments: List[str]):
    msg = MIMEMultipart()
    msg['From'] = config.SENDER_EMAIL
    msg['To'] = config.RECIPIENT_EMAIL
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    for file in attachments:
        part = MIMEBase('application', 'octet-stream')
        with open(file, 'rb') as f:
            part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', f"attachment; filename= {os.path.basename(file)}")
        msg.attach(part)

    try:
        server = smtplib.SMTP_SSL(config.SMTP_SERVER, config.SMTP_PORT)
        server.login(config.SENDER_EMAIL, config.EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()
        logger.info("Email sent successfully")
    except Exception as e:
        logger.error(f"Failed to send email: {e}")

# ---------------------
# Main
# ---------------------
def main():
    db = NeuroCellDB()
    db.mark_all_old()

    # PubMed
    pubmed_fetcher = PubMedFetcher(config.NCBI_EMAIL)
    pubmed_articles = pubmed_fetcher.fetch_articles(config.PUBMED_TERM, config.MAX_RECORDS)
    for article in pubmed_articles:
        db.upsert_pubmed_article(article)

    # ClinicalTrials.gov
    trials_fetcher = ClinicalTrialsFetcher()
    trials = trials_fetcher.fetch_trials(config.TRIALS_TERM, config.MAX_RECORDS)
    for trial in trials:
        db.upsert_clinical_trial(trial)

    # Export CSVs
    pubmed_csv = db.export_csv("pubmed_articles", "new_pubmed_this_week.csv", new_only=True)
    trials_csv = db.export_csv("clinical_trials", "new_trials_this_week.csv", new_only=True)

    # Send email
    send_email(
        subject="NeuroCell Weekly Report",
        body="Please find attached the latest PubMed articles and ClinicalTrials.gov studies.",
        attachments=[pubmed_csv, trials_csv]
    )

    db.close()

if __name__ == "__main__":
    main()

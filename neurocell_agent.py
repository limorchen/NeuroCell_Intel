import requests
from Bio import Entrez
import pandas as pd
from sentence_transformers import SentenceTransformer
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import os
import logging
import sqlite3
from datetime import datetime, timedelta
import hashlib
import time

# ---------------------
# Logging config
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
    CLINICALTRIALS_BASE = "https://clinicaltrials.gov/api/v2/studies"
    MAX_RECORDS = int(os.getenv("MAX_RECORDS", 20))
    RATE_LIMIT_DELAY = float(os.getenv("RATE_LIMIT_DELAY", 0.5))

    DB_FILE = "neurocell_database.db"


config = Config()

# ---------------------
# Utility functions
# ---------------------
def compute_content_hash(text: str) -> str:
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def current_timestamp():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# ---------------------
# SQLite DB Management
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
            volume TEXT,
            issue TEXT,
            pages TEXT,
            doi TEXT,
            keywords TEXT,
            url TEXT,
            is_new INTEGER,
            content_hash TEXT,
            first_seen TEXT
        )""")

        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS clinical_trials (
            nctid TEXT PRIMARY KEY,
            title TEXT,
            detailed_description TEXT,
            conditions TEXT,
            interventions TEXT,
            phases TEXT,
            study_type TEXT,
            status TEXT,
            start_date TEXT,
            completion_date TEXT,
            sponsor TEXT,
            enrollment TEXT,
            age_range TEXT,
            url TEXT,
            is_new INTEGER,
            content_hash TEXT,
            first_seen TEXT
        )""")
        self.conn.commit()

    def mark_all_old(self):
        self.cursor.execute("UPDATE pubmed_articles SET is_new=0")
        self.cursor.execute("UPDATE clinical_trials SET is_new=0")
        self.conn.commit()

    def upsert_pubmed_article(self, article):
        hash_val = compute_content_hash(article['title'] + article['abstract'])
        self.cursor.execute("SELECT pmid, content_hash FROM pubmed_articles WHERE pmid=?", (article['pmid'],))
        row = self.cursor.fetchone()

        now = current_timestamp()
        if not row:
            # Insert new article
            self.cursor.execute("""
                INSERT INTO pubmed_articles
                (pmid, title, abstract, authors, publication_date, journal, volume, issue, pages, doi, keywords, url, is_new, content_hash, first_seen)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
            """, (
                article['pmid'], article['title'], article['abstract'], article['authors'], article['publication_date'], article['journal'],
                article['volume'], article['issue'], article['pages'], article['doi'], ",".join(article['keywords']),
                article['url'], hash_val, now
            ))
            logger.info(f"Added NEW PubMed article: {article['pmid']}")
        else:
            # Update is_new flag if content hash changed
            if row[1] != hash_val:
                self.cursor.execute("""
                    UPDATE pubmed_articles SET title=?, abstract=?, authors=?, publication_date=?, journal=?, volume=?, issue=?, pages=?, doi=?, keywords=?, url=?, is_new=1, content_hash=?
                    WHERE pmid=?
                """, (
                    article['title'], article['abstract'], article['authors'], article['publication_date'], article['journal'],
                    article['volume'], article['issue'], article['pages'], article['doi'], ",".join(article['keywords']),
                    article['url'], hash_val, article['pmid']
                ))
                logger.info(f"Updated & marked NEW PubMed article: {article['pmid']}")
            else:
                # Just keep is_new as 0 for existing without change
                self.cursor.execute("UPDATE pubmed_articles SET is_new=0 WHERE pmid=?", (article['pmid'],))

        self.conn.commit()

    def upsert_clinical_trial(self, trial):
        hash_text = trial['title'] + (trial['detailed_description'] or "") + "".join(trial['conditions']) + "".join(trial['interventions'])
        hash_val = compute_content_hash(hash_text)

        self.cursor.execute("SELECT nctid, content_hash FROM clinical_trials WHERE nctid=?", (trial['nctid'],))
        row = self.cursor.fetchone()

        now = current_timestamp()
        if not row:
            self.cursor.execute("""
                INSERT INTO clinical_trials
                (nctid, title, detailed_description, conditions, interventions, phases, study_type, status, start_date, completion_date, sponsor, enrollment, age_range, url, is_new, content_hash, first_seen)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
            """, (
                trial['nctid'], trial['title'], trial['detailed_description'], ",".join(trial['conditions']), ",".join(trial['interventions']),
                ",".join(trial['phases']), trial['study_type'], trial['status'], trial['start_date'], trial['completion_date'], trial['sponsor'],
                str(trial['enrollment']), trial['age_range'], trial['url'], hash_val, now
            ))
            logger.info(f"Added NEW Clinical Trial: {trial['nctid']}")
        else:
            if row[1] != hash_val:
                self.cursor.execute("""
                    UPDATE clinical_trials SET
                    title=?, detailed_description=?, conditions=?, interventions=?, phases=?, study_type=?, status=?, start_date=?,
                    completion_date=?, sponsor=?, enrollment=?, age_range=?, url=?, is_new=1, content_hash=?
                    WHERE nctid=?
                """, (
                    trial['title'], trial['detailed_description'], ",".join(trial['conditions']), ",".join(trial['interventions']),
                    ",".join(trial['phases']), trial['study_type'], trial['status'], trial['start_date'],
                    trial['completion_date'], trial['sponsor'], str(trial['enrollment']), trial['age_range'], trial['url'], hash_val, trial['nctid']
                ))
                logger.info(f"Updated & marked NEW Clinical Trial: {trial['nctid']}")
            else:
                # Keep is_new=0 for unchanged existing
                self.cursor.execute("UPDATE clinical_trials SET is_new=0 WHERE nctid=?", (trial['nctid'],))

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
# PubMed Fetcher (as before)
# ---------------------
class PubMedFetcher:
    def __init__(self, email: str):
        Entrez.email = email
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

    # ... your existing methods remain unchanged ...


# ---------------------
# Clinical Trials Fetcher (as before)
# ---------------------
class ClinicalTrialsFetcher:
    def __init__(self):
        self.base_url = config.CLINICALTRIALS_BASE

    # ... your existing methods remain unchanged ...


# ---------------------
# Email Sender
# ---------------------
class EmailSender:
    def __init__(self, config_obj):
        self.config = config_obj

    def send_comprehensive_report(self, db: NeuroCellDB) -> bool:
        password = self.config.EMAIL_PASSWORD
        if not password:
            logger.error("Missing EMAIL_PASSWORD environment variable")
            return False

        try:
            # Export CSVs
            new_pubmed_csv = db.export_csv("pubmed_articles", "new_pubmed_this_week.csv", new_only=True)
            new_trials_csv = db.export_csv("clinical_trials", "new_trials_this_week.csv", new_only=True)
            all_pubmed_csv = db.export_csv("pubmed_articles", "all_pubmed_database.csv", new_only=False)
            all_trials_csv = db.export_csv("clinical_trials", "all_trials_database.csv", new_only=False)

            # Create email
            msg = MIMEMultipart()
            msg['From'] = self.config.SENDER_EMAIL
            msg['To'] = self.config.RECIPIENT_EMAIL
            msg['Subject'] = f"NeuroCell Intelligence Report - {datetime.now().strftime('%Y-%m-%d')}"

            # Construct summary text
            pubmed_new_count = pd.read_csv(new_pubmed_csv).shape[0]
            trials_new_count = pd.read_csv(new_trials_csv).shape[0]
            pubmed_total_count = pd.read_csv(all_pubmed_csv).shape[0]
            trials_total_count = pd.read_csv(all_trials_csv).shape[0]

            text = f"""
🧬 NeuroCell Intelligence Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Summary Statistics:
- New PubMed Articles this week: {pubmed_new_count}
- Total PubMed Articles in DB: {pubmed_total_count}

- New Clinical Trials this week: {trials_new_count}
- Total Clinical Trials in DB: {trials_total_count}

Please see the attached CSV files for details.
"""
            msg.attach(MIMEText(text, 'plain'))

            # Attach all CSVs
            for filepath in [new_pubmed_csv, new_trials_csv, all_pubmed_csv, all_trials_csv]:
                with open(filepath, "rb") as f:
                    part = MIMEBase("application", "octet-stream")
                    part.set_payload(f.read())
                encoders.encode_base64(part)
                part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(filepath)}")
                msg.attach(part)

            # Send email
            with smtplib.SMTP_SSL(self.config.SMTP_SERVER, self.config.SMTP_PORT) as server:
                server.login(self.config.SENDER_EMAIL, password)
                server.sendmail(self.config.SENDER_EMAIL, self.config.RECIPIENT_EMAIL, msg.as_string())

            logger.info("Email report sent successfully")
            return True

        except Exception as e:
            logger.error(f"Email sending failed: {e}")
            return False


# ---------------------
# Main Agent Orchestrator
# ---------------------
class NeuroCellAgent:
    def __init__(self):
        self.config = config
        self.db = NeuroCellDB()
        self.pubmed_fetcher = PubMedFetcher(self.config.NCBI_EMAIL)
        self.trials_fetcher = ClinicalTrialsFetcher()
        self.email_sender = EmailSender(self.config)

    def run_weekly(self, days_back=7):
        logger.info("Starting weekly NeuroCell Intelligence Agent run.")

        # Mark all existing entries as old
        self.db.mark_all_old()

        # Fetch PubMed articles
        articles = self.pubmed_fetcher.fetch_articles(
            self.config.PUBMED_TERM, self.config.MAX_RECORDS, days_back
        )

        # Insert/update database with new PubMed articles
        for art in articles:
            self.db.upsert_pubmed_article(art)

        # Fetch clinical trials
        trial_queries = [
            "exosomes AND neurological",
            "exosome AND CNS",
            "extracellular vesicles AND brain",
            "exosomes AND spinal cord"
        ]
        trials = self.trials_fetcher.search_trials(trial_queries, self.config.MAX_RECORDS)

        for tr in trials:
            self.db.upsert_clinical_trial(tr)

        # Send the email report with attached CSVs
        success = self.email_sender.send_comprehensive_report(self.db)

        self.db.close()

        if success:
            logger.info("Weekly run completed successfully.")
        else:
            logger.error("Weekly run encountered errors.")
        return success


if __name__ == "__main__":
    agent = NeuroCellAgent()
    success = agent.run_weekly()

    if success:
        print("✅ NeuroCell Intelligence Agent completed successfully")
    else:
        print("❌ NeuroCell Intelligence Agent encountered errors")

import os
import time
import hashlib
import logging
import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pandas as pd
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from dotenv import load_dotenv
from Bio import Entrez
from scholarly import scholarly

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
    SCHOLAR_TERM = os.getenv("SCHOLAR_TERM", "exosomes CNS")
    MAX_RECORDS = int(os.getenv("MAX_RECORDS", 20))
    RATE_LIMIT_DELAY = float(os.getenv("RATE_LIMIT_DELAY", 0.5))
    DB_FILE = "neurocell_database.db"

config = Config()

# ---------------------
# Utility functions
# ---------------------
def compute_content_hash(text: str) -> str:
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def current_timestamp() -> str:
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# ---------------------
# Database management
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
        );
        """)
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS scholar_articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            abstract TEXT,
            authors TEXT,
            publication_date TEXT,
            journal TEXT,
            url TEXT,
            is_new INTEGER,
            content_hash TEXT,
            first_seen TEXT
        );
        """)
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS consolidated_articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            abstract TEXT,
            authors TEXT,
            publication_date TEXT,
            journal TEXT,
            source TEXT,
            url TEXT,
            content_hash TEXT
        );
        """)
        self.conn.commit()

    def mark_all_old(self):
        self.cursor.execute("UPDATE pubmed_articles SET is_new=0;")
        self.cursor.execute("UPDATE scholar_articles SET is_new=0;")
        self.conn.commit()

    def upsert_pubmed_article(self, article):
        hash_val = compute_content_hash(article['title'] + article['abstract'])
        self.cursor.execute("SELECT pmid, content_hash FROM pubmed_articles WHERE pmid=?", (article['pmid'],))
        row = self.cursor.fetchone()
        now = current_timestamp()
        if not row:
            self.cursor.execute("""
                INSERT INTO pubmed_articles
                (pmid, title, abstract, authors, publication_date, journal, volume, issue, pages, doi, keywords, url, is_new, content_hash, first_seen)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?);
            """, (
                article['pmid'], article['title'], article['abstract'], article['authors'], article['publication_date'], article['journal'],
                article['volume'], article['issue'], article['pages'], article['doi'], ",".join(article.get('keywords', [])),
                article['url'], hash_val, now
            ))
            logger.info(f"Added NEW PubMed article: {article['pmid']}")
        else:
            if row[1] != hash_val:
                self.cursor.execute("""
                    UPDATE pubmed_articles SET title=?, abstract=?, authors=?, publication_date=?, journal=?, volume=?, issue=?, pages=?, doi=?, keywords=?, url=?, is_new=1, content_hash=? WHERE pmid=?;
                """, (
                    article['title'], article['abstract'], article['authors'], article['publication_date'], article['journal'],
                    article['volume'], article['issue'], article['pages'], article['doi'], ",".join(article.get('keywords', [])),
                    article['url'], hash_val, article['pmid']
                ))
                logger.info(f"Updated & marked NEW PubMed article: {article['pmid']}")
            else:
                self.cursor.execute("UPDATE pubmed_articles SET is_new=0 WHERE pmid=?;", (article['pmid'],))
        self.conn.commit()

    def upsert_scholar_article(self, article):
        hash_val = compute_content_hash(article['title'] + article['abstract'])
        self.cursor.execute("SELECT id, content_hash FROM scholar_articles WHERE title=?", (article['title'],))
        row = self.cursor.fetchone()
        now = current_timestamp()
        if not row:
            self.cursor.execute("""
                INSERT INTO scholar_articles
                (title, abstract, authors, publication_date, journal, url, is_new, content_hash, first_seen)
                VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?);
            """, (article['title'], article['abstract'], article['authors'], article['publication_date'],
                  article['journal'], article['url'], hash_val, now))
            logger.info(f"Added NEW Scholar article: {article['title']}")
        else:
            if row[1] != hash_val:
                self.cursor.execute("""
                    UPDATE scholar_articles SET abstract=?, authors=?, publication_date=?, journal=?, url=?, is_new=1, content_hash=? WHERE id=?;
                """, (article['abstract'], article['authors'], article['publication_date'], article['journal'], article['url'], hash_val, row[0]))
                logger.info(f"Updated & marked NEW Scholar article: {article['title']}")
            else:
                self.cursor.execute("UPDATE scholar_articles SET is_new=0 WHERE id=?;", (row[0],))
        self.conn.commit()

    def consolidate_articles(self):
        self.cursor.execute("DELETE FROM consolidated_articles;")
        self.cursor.execute("SELECT title, abstract, authors, publication_date, journal, url FROM pubmed_articles;")
        pubmed_articles = self.cursor.fetchall()
        self.cursor.execute("SELECT title, abstract, authors, publication_date, journal, url FROM scholar_articles;")
        scholar_articles = self.cursor.fetchall()
        combined = pubmed_articles + scholar_articles
        seen_hashes = set()
        for art in combined:
            text_hash = compute_content_hash(art[0] + (art[1] or ""))
            if text_hash in seen_hashes:
                continue
            seen_hashes.add(text_hash)
            source = "PubMed" if art in pubmed_articles else "Scholar"
            self.cursor.execute("""
                INSERT INTO consolidated_articles
                (title, abstract, authors, publication_date, journal, source, url, content_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """, (*art, source, text_hash))
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
# PubMed fetcher
# ---------------------
class PubMedFetcher:
    def __init__(self, email: str):
        Entrez.email = email

    def fetch_articles(self, term: str, max_records: int = 20, days_back: int = 30) -> List[Dict]:
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
                    "volume": "",
                    "issue": "",
                    "pages": "",
                    "doi": "",
                    "keywords": [],
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                })
            return results
        except Exception as e:
            logger.error(f"Error fetching PubMed articles: {e}")
            return []

# ---------------------
# Google Scholar fetcher
# ---------------------
class ScholarFetcher:
    def fetch_articles(self, term: str, max_records: int = 20) -> List[Dict]:
        results = []
        try:
            search_query = scholarly.search_pubs(term)
            for i, pub in enumerate(search_query):
                if i >= max_records:
                    break
                article = pub.fill()
                results.append({
                    "title": article.bib.get("title", ""),
                    "abstract": article.bib.get("abstract", ""),
                    "authors": ", ".join(article.bib.get("author", [])),
                    "publication_date": article.bib.get("year", ""),
                    "journal": article.bib.get("journal", ""),
                    "url": article.bib.get("url", "")
                })
            return results
        except Exception as e:
            logger.error(f"Error fetching Scholar articles: {e}")
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
# Main execution
# ---------------------
def main():
    db = NeuroCellDB()
    db.mark_all_old()

    # Fetch PubMed
    pubmed_fetcher = PubMedFetcher(config.NCBI_EMAIL)
    pubmed_articles = pubmed_fetcher.fetch_articles(config.PUBMED_TERM, config.MAX_RECORDS)
    for article in pubmed_articles:
        db.upsert_pubmed_article(article)

    # Fetch Google Scholar
    scholar_fetcher = ScholarFetcher()
    scholar_articles = scholar_fetcher.fetch_articles(config.SCHOLAR_TERM, config.MAX_RECORDS)
    for article in scholar_articles:
        db.upsert_scholar_article(article)

    # Consolidate and remove duplicates
    db.consolidate_articles()

    # Export CSVs
    pubmed_csv = db.export_csv("pubmed_articles", "pubmed_articles.csv", new_only=True)
    scholar_csv = db.export_csv("scholar_articles", "scholar_articles.csv", new_only=True)
    consolidated_csv = db.export_csv("consolidated_articles", "consolidated_articles.csv")

    # Send email
    send_email(
        subject="NeuroCell Weekly Article Report",
        body="Please find attached the latest PubMed and Scholar articles.",
        attachments=[pubmed_csv, scholar_csv, consolidated_csv]
    )

    db.close()

if __name__ == "__main__":
    main()

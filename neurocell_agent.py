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
from typing import List, Dict, Optional

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
# PubMed Fetcher complete implementation
# ---------------------
class PubMedFetcher:
    def __init__(self, email: str):
        Entrez.email = email
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

    def fetch_articles(self, term: str, max_records: int = 20, days_back: int = 30) -> List[Dict]:
        try:
            date_filter = (datetime.now() - timedelta(days=days_back)).strftime("%Y/%m/%d")
            search_term = f'{term} AND ("{date_filter}"[Date - Publication] : "3000"[Date - Publication])'
            logger.info(f"Searching PubMed with term: {search_term}")

            handle = Entrez.esearch(
                db="pubmed",
                term=search_term,
                retmax=max_records,
                sort="date",
                usehistory="y"
            )
            record = Entrez.read(handle)
            ids = record["IdList"]
            handle.close()

            if not ids:
                logger.warning("No PubMed articles found")
                return []

            time.sleep(config.RATE_LIMIT_DELAY)

            handle = Entrez.efetch(
                db="pubmed",
                id=",".join(ids),
                rettype="abstract",
                retmode="xml"
            )
            papers = Entrez.read(handle)
            handle.close()

            return self._parse_pubmed_articles(papers)
        except Exception as e:
            logger.error(f"Error fetching PubMed articles: {e}")
            return []

    def _parse_pubmed_articles(self, papers) -> List[Dict]:
        summaries = []

        for article in papers.get("PubmedArticle", []):
            try:
                parsed_article = self._parse_single_article(article)
                if parsed_article:
                    summaries.append(parsed_article)
            except Exception as e:
                logger.error(f"Error parsing individual PubMed article: {e}")
                continue
        logger.info(f"Successfully parsed {len(summaries)} PubMed articles")
        return summaries

    def _parse_single_article(self, article) -> Optional[Dict]:
        medline_citation = article.get("MedlineCitation", {})
        article_data = medline_citation.get("Article", {})

        pmid = medline_citation.get("PMID")
        title = article_data.get("ArticleTitle")
        if not (pmid and title):
            return None

        abstract_list = article_data.get("Abstract", {}).get("AbstractText", [])
        if isinstance(abstract_list, list):
            abstract = " ".join([str(abs_text) for abs_text in abstract_list])
        else:
            abstract = str(abstract_list) if abstract_list else ""

        authors = self._extract_authors(article_data.get("AuthorList", []))
        journal_info = article_data.get("Journal", {})
        pub_info = self._extract_publication_info(journal_info)
        keywords = self._extract_keywords(medline_citation)
        doi = self._extract_doi(article_data)

        return {
            "pmid": str(pmid),
            "title": str(title),
            "abstract": abstract,
            "authors": authors,
            "publication_date": pub_info["date"],
            "journal": pub_info["journal"],
            "volume": pub_info["volume"],
            "issue": pub_info["issue"],
            "pages": pub_info["pages"],
            "doi": doi,
            "keywords": keywords,
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        }

    def _extract_authors(self, author_list: List) -> str:
        authors = []
        for author in author_list[:10]:
            if "LastName" in author and "Initials" in author:
                authors.append(f"{author['LastName']} {author['Initials']}")
            elif "CollectiveName" in author:
                authors.append(author["CollectiveName"])
        if len(author_list) > 10:
            authors.append("et al.")
        return ", ".join(authors)

    def _extract_publication_info(self, journal_info: Dict) -> Dict:
        journal_issue = journal_info.get("JournalIssue", {})
        pub_date_data = journal_issue.get("PubDate", {})

        pub_date = "N/A"
        if "Year" in pub_date_data:
            year = pub_date_data["Year"]
            month = pub_date_data.get("Month")
            day = pub_date_data.get("Day")

            if all([year, month, day]):
                pub_date = f"{year}-{month:02d}-{day:02d}" if month.isdigit() else f"{year}-{month}-{day}"
            elif year and month:
                pub_date = f"{year}-{month}"
            elif year:
                pub_date = str(year)
        elif "MedlineDate" in pub_date_data:
            pub_date = pub_date_data["MedlineDate"]

        return {
            "date": pub_date,
            "journal": journal_info.get("Title", "N/A"),
            "volume": journal_issue.get("Volume", "N/A"),
            "issue": journal_issue.get("Issue", "N/A"),
            "pages": journal_info.get("Article", {}).get("Pagination", {}).get("StartPage", "N/A")
        }


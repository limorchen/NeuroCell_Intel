#!/usr/bin/env python3
"""
neurocell_agent.py

Weekly intelligence agent:
- fetches PubMed articles via Entrez
- fetches ClinicalTrials.gov studies via API
- fetches Google Scholar results via `scholarly` (best-effort)
- fetches EMA / EU Clinical Trials Register results by scraping (best-effort)
- stores results in SQLite and marks new items
- exports CSVs and emails a report (supports multiple recipients)
"""

import os
import logging
import time
import hashlib
import sqlite3
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional

# third-party
from dotenv import load_dotenv
from Bio import Entrez
import pandas as pd
from sentence_transformers import SentenceTransformer
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from bs4 import BeautifulSoup

# scholarly is an unofficial library for Google Scholar. It can be brittle.
# Install: pip install scholarly
try:
    from scholarly import scholarly
    SCHOLARLY_AVAILABLE = True
except Exception:
    SCHOLARLY_AVAILABLE = False

# Load local .env (harmless if missing)
load_dotenv()

# ---------------------
# Logging configuration
# ---------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("neurocell_agent.log"),
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
    RECIPIENT_EMAIL = os.getenv("RECIPIENT_EMAIL", "limor@nurexone.com")  # comma-separated allowed
    SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    SMTP_PORT = int(os.getenv("SMTP_PORT", 465))
    EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
    PUBMED_TERM = os.getenv("PUBMED_TERM", "exosomes AND CNS")
    CLINICALTRIALS_BASE = os.getenv("CLINICALTRIALS_BASE", "https://clinicaltrials.gov/api/v2/studies")
    MAX_RECORDS = int(os.getenv("MAX_RECORDS", 20))
    RATE_LIMIT_DELAY = float(os.getenv("RATE_LIMIT_DELAY", 0.5))
    DB_FILE = os.getenv("DB_FILE", "neurocell_database.db")
    # Google Scholar and EMA settings
    GS_MAX = int(os.getenv("GS_MAX", MAX_RECORDS))
    EMA_MAX = int(os.getenv("EMA_MAX", MAX_RECORDS))

config = Config()

# ---------------------
# Utility functions
# ---------------------
def compute_content_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def current_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def to_list_of_dicts(cursor) -> List[Dict]:
    """Utility: convert sqlite cursor results to list of dicts (not used widely)."""
    cols = [c[0] for c in cursor.description]
    return [dict(zip(cols, row)) for row in cursor.fetchall()]

# ---------------------
# Database management
# ---------------------
class NeuroCellDB:
    def __init__(self, db_path=config.DB_FILE):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self._create_tables()

    def _create_tables(self):
        # PubMed
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
        # ClinicalTrials.gov
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
        );
        """)
        # Google Scholar
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS google_scholar_articles (
            gs_id TEXT PRIMARY KEY,
            title TEXT,
            authors TEXT,
            year TEXT,
            abstract TEXT,
            url TEXT,
            is_new INTEGER,
            content_hash TEXT,
            first_seen TEXT
        );
        """)
        # EMA / EU Trials
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS ema_trials (
            trial_id TEXT PRIMARY KEY,
            title TEXT,
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
        self.cursor.execute("UPDATE google_scholar_articles SET is_new=0;")
        self.cursor.execute("UPDATE ema_trials SET is_new=0;")
        self.conn.commit()

    # PubMed upsert (keeps original behavior)
    def upsert_pubmed_article(self, article):
        hash_val = compute_content_hash((article.get("title","") or "") + (article.get("abstract","") or ""))
        self.cursor.execute("SELECT pmid, content_hash FROM pubmed_articles WHERE pmid=?", (article["pmid"],))
        row = self.cursor.fetchone()
        now = current_timestamp()
        if not row:
            self.cursor.execute("""
                INSERT INTO pubmed_articles
                (pmid, title, abstract, authors, publication_date, journal, volume, issue, pages, doi, keywords, url, is_new, content_hash, first_seen)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?);
            """, (
                article["pmid"], article["title"], article["abstract"], article["authors"], article["publication_date"], article["journal"],
                article["volume"], article["issue"], article["pages"], article["doi"], ",".join(article.get("keywords", [])),
                article["url"], hash_val, now
            ))
            logger.info(f"Added NEW PubMed article: {article['pmid']}")
        else:
            if row[1] != hash_val:
                self.cursor.execute("""
                    UPDATE pubmed_articles SET title=?, abstract=?, authors=?, publication_date=?, journal=?, volume=?, issue=?, pages=?, doi=?, keywords=?, url=?, is_new=1, content_hash=? WHERE pmid=?;
                """, (
                    article["title"], article["abstract"], article["authors"], article["publication_date"], article["journal"],
                    article["volume"], article["issue"], article["pages"], article["doi"], ",".join(article.get("keywords", [])),
                    article["url"], hash_val, article["pmid"]
                ))
                logger.info(f"Updated & marked NEW PubMed article: {article['pmid']}")
            else:
                self.cursor.execute("UPDATE pubmed_articles SET is_new=0 WHERE pmid=?;", (article["pmid"],))
        self.conn.commit()

    # ClinicalTrials.gov upsert
    def upsert_clinical_trial(self, trial):
        hash_text = trial["title"] + (trial.get("detailed_description") or "") + "".join(trial.get("conditions", [])) + "".join(trial.get("interventions", []))
        hash_val = compute_content_hash(hash_text)
        self.cursor.execute("SELECT nctid, content_hash FROM clinical_trials WHERE nctid=?", (trial["nctid"],))
        row = self.cursor.fetchone()
        now = current_timestamp()
        if not row:
            self.cursor.execute("""
                INSERT INTO clinical_trials
                (nctid, title, detailed_description, conditions, interventions, phases, study_type, status, start_date, completion_date, sponsor, enrollment, age_range, url, is_new, content_hash, first_seen)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?);
            """, (
                trial["nctid"], trial["title"], trial["detailed_description"], ",".join(trial.get("conditions", [])), ",".join(trial.get("interventions", [])),
                ",".join(trial.get("phases", [])), trial.get("study_type"), trial.get("status"), trial.get("start_date"), trial.get("completion_date"), trial.get("sponsor"),
                str(trial.get("enrollment")), trial.get("age_range"), trial.get("url"), hash_val, now
            ))
            logger.info(f"Added NEW Clinical Trial: {trial['nctid']}")
        else:
            if row[1] != hash_val:
                self.cursor.execute("""
                    UPDATE clinical_trials SET
                    title=?, detailed_description=?, conditions=?, interventions=?, phases=?, study_type=?, status=?, start_date=?,
                    completion_date=?, sponsor=?, enrollment=?, age_range=?, url=?, is_new=1, content_hash=? WHERE nctid=?;
                """, (
                    trial["title"], trial["detailed_description"], ",".join(trial.get("conditions", [])), ",".join(trial.get("interventions", [])),
                    ",".join(trial.get("phases", [])), trial.get("study_type"), trial.get("status"), trial.get("start_date"),
                    trial.get("completion_date"), trial.get("sponsor"), str(trial.get("enrollment")), trial.get("age_range"), trial.get("url"), hash_val, trial["nctid"]
                ))
                logger.info(f"Updated & marked NEW Clinical Trial: {trial['nctid']}")
            else:
                self.cursor.execute("UPDATE clinical_trials SET is_new=0 WHERE nctid=?;", (trial["nctid"],))
        self.conn.commit()

    # Google Scholar upsert
    def upsert_google_article(self, article):
        gs_id = article.get("gs_id") or compute_content_hash(article.get("title","") + article.get("authors",""))
        hash_val = compute_content_hash((article.get("title","") or "") + (article.get("abstract","") or ""))
        self.cursor.execute("SELECT gs_id, content_hash FROM google_scholar_articles WHERE gs_id=?", (gs_id,))
        row = self.cursor.fetchone()
        now = current_timestamp()
        if not row:
            self.cursor.execute("""
                INSERT INTO google_scholar_articles
                (gs_id, title, authors, year, abstract, url, is_new, content_hash, first_seen)
                VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?);
            """, (
                gs_id, article.get("title"), article.get("authors"), article.get("year"), article.get("abstract"),
                article.get("url"), hash_val, now
            ))
            logger.info(f"Added NEW Google Scholar article: {gs_id}")
        else:
            if row[1] != hash_val:
                self.cursor.execute("""
                    UPDATE google_scholar_articles SET title=?, authors=?, year=?, abstract=?, url=?, is_new=1, content_hash=? WHERE gs_id=?;
                """, (
                    article.get("title"), article.get("authors"), article.get("year"), article.get("abstract"),
                    article.get("url"), hash_val, gs_id
                ))
                logger.info(f"Updated & marked NEW Google Scholar article: {gs_id}")
            else:
                self.cursor.execute("UPDATE google_scholar_articles SET is_new=0 WHERE gs_id=?;", (gs_id,))
        self.conn.commit()

    # EMA upsert
    def upsert_ema_trial(self, trial):
        trial_id = trial.get("trial_id")
        hash_val = compute_content_hash((trial.get("title","") or "") + (trial.get("trial_id","") or ""))
        self.cursor.execute("SELECT trial_id, content_hash FROM ema_trials WHERE trial_id=?", (trial_id,))
        row = self.cursor.fetchone()
        now = current_timestamp()
        if not row:
            self.cursor.execute("""
                INSERT INTO ema_trials (trial_id, title, url, is_new, content_hash, first_seen)
                VALUES (?, ?, ?, 1, ?, ?)
            """, (trial_id, trial.get("title"), trial.get("url"), hash_val, now))
            logger.info(f"Added NEW EMA trial: {trial_id}")
        else:
            if row[1] != hash_val:
                self.cursor.execute("""
                    UPDATE ema_trials SET title=?, url=?, is_new=1, content_hash=? WHERE trial_id=?
                """, (trial.get("title"), trial.get("url"), hash_val, trial_id))
                logger.info(f"Updated & marked NEW EMA trial: {trial_id}")
            else:
                self.cursor.execute("UPDATE ema_trials SET is_new=0 WHERE trial_id=?;", (trial_id,))
        self.conn.commit()

    # CSV export
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
# PubMed fetcher (unchanged behavior)
# ---------------------
class PubMedFetcher:
    def __init__(self, email: str):
        Entrez.email = email
        # SentenceTransformer used elsewhere (kept for compatibility)
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

    def fetch_articles(self, term: str, max_records: int = 20, days_back: int = 7) -> List[Dict]:
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
            ids = record.get("IdList", [])
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

    def _extract_keywords(self, medline_citation: Dict) -> List[str]:
        keywords = []
        mesh_list = medline_citation.get("MeshHeadingList", [])
        for mesh in mesh_list[:5]:
            descriptor = mesh.get("DescriptorName", {})
            if hasattr(descriptor, "attributes") and "text" in descriptor.attributes:
                keywords.append(descriptor.attributes["text"])
        return keywords

    def _extract_doi(self, article_data: Dict) -> str:
        elocation_id = article_data.get("ELocationID", [])
        if isinstance(elocation_id, list):
            for loc in elocation_id:
                if hasattr(loc, "attributes") and loc.attributes.get("EIdType") == "doi":
                    return loc.attributes.get("text", "")
        return "N/A"


# ---------------------
# ClinicalTrials.gov fetcher (unchanged)
# ---------------------
class ClinicalTrialsFetcher:
    def __init__(self):
        self.base_url = config.CLINICALTRIALS_BASE

    def search_trials(self, queries: List[str], max_records: int = 20) -> List[Dict]:
        all_results = []
        for query in queries:
            logger.info(f"Searching ClinicalTrials.gov with query: {query}")
            results = self._fetch_trials(query, max_records)
            if results:
                logger.info(f"Found {len(results)} trials with query: {query}")
                for result in results:
                    if not any(r["nctid"] == result["nctid"] for r in all_results):
                        all_results.append(result)
                break  # Use first successful query
            time.sleep(config.RATE_LIMIT_DELAY)
        return all_results[:max_records]

    def _fetch_trials(self, query: str, max_records: int) -> List[Dict]:
        params = {
            "query.term": query,
            "pageSize": min(max_records, 100),
            "format": "json"
        }
        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            studies = data.get("studies", [])
            return [self._parse_study(study) for study in studies]
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return []
        except ValueError as e:
            logger.error(f"JSON parsing error: {e}")
            return []

    def _parse_study(self, study: Dict) -> Dict:
        protocol_section = study.get("protocolSection", {})
        identification = protocol_section.get("identificationModule", {})
        description = protocol_section.get("descriptionModule", {})
        design = protocol_section.get("designModule", {})
        status = protocol_section.get("statusModule", {})
        sponsor = protocol_section.get("sponsorCollaboratorsModule", {})
        arms = protocol_section.get("armsInterventionsModule", {})
        conditions = protocol_section.get("conditionsModule", {})
        eligibility = protocol_section.get("eligibilityModule", {})

        return {
            "nctid": identification.get("nctId", ""),
            "title": identification.get("briefTitle", ""),
            "detailed_description": description.get("detailedDescription", ""),
            "conditions": conditions.get("conditions", []),
            "interventions": [inv.get("name", "") for inv in arms.get("interventions", [])],
            "phases": design.get("phases", []),
            "study_type": design.get("studyType", "N/A"),
            "status": status.get("overallStatus", "N/A"),
            "start_date": status.get("startDateStruct", {}).get("date", "N/A"),
            "completion_date": status.get("completionDateStruct", {}).get("date", "N/A"),
            "sponsor": sponsor.get("leadSponsor", {}).get("name", "N/A"),
            "enrollment": design.get("enrollmentInfo", {}).get("count", "N/A"),
            "age_range": f"{eligibility.get('minimumAge', 'N/A')} - {eligibility.get('maximumAge', 'N/A')}",
            "url": f"https://clinicaltrials.gov/study/{identification.get('nctId', '')}"
        }


# ---------------------
# Google Scholar fetcher (best-effort)
# ---------------------
class GoogleScholarFetcher:
    def __init__(self, term: str, max_records: int = 20):
        self.term = term
        self.max_records = max_records

    def fetch(self) -> List[Dict]:
        if not SCHOLARLY_AVAILABLE:
            logger.warning("scholarly package not available — skipping Google Scholar fetch")
            return []

        results = []
        try:
            search_iter = scholarly.search_pubs(self.term)
            for i, pub in enumerate(search_iter):
                if i >= self.max_records:
                    break
                try:
                    bib = pub.get("bib", {}) if isinstance(pub, dict) else {}
                    title = bib.get("title", "")
                    authors = ", ".join(bib.get("author", [])) if bib.get("author") else ""
                    year = str(bib.get("pub_year", "")) if bib.get("pub_year") else ""
                    abstract = bib.get("abstract", "") if bib.get("abstract") else ""
                    url = pub.get("pub_url") if isinstance(pub, dict) else None
                    gs_id = pub.get("author_id", None) or compute_content_hash(title + authors + year)
                    results.append({
                        "gs_id": gs_id,
                        "title": title,
                        "authors": authors,
                        "year": year,
                        "abstract": abstract,
                        "url": url
                    })
                except Exception as e:
                    logger.debug(f"Error parsing a Google Scholar entry: {e}")
                    continue
                time.sleep(config.RATE_LIMIT_DELAY)
            logger.info(f"Fetched {len(results)} Google Scholar results (best-effort)")
        except Exception as e:
            logger.error(f"Error fetching Google Scholar: {e}")
        return results


# ---------------------
# EMA (EUCTR) fetcher (best-effort scraping)
# ---------------------
class EMAClinicalTrialsFetcher:
    def __init__(self, term: str, max_records: int = 20):
        self.term = term
        self.max_records = max_records
        # EUCTR search base (subject to site changes)
        self.base = "https://www.clinicaltrialsregister.eu/ctr-search/search"

    def fetch(self) -> List[Dict]:
        trials = []
        try:
            params = {"query": self.term}
            headers = {"User-Agent": "NeuroCellAgent/1.0 (+https://github.com/)"}
            resp = requests.get(self.base, params=params, headers=headers, timeout=30)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            # Attempt to find result rows — site layout may change
            table = soup.find("table", class_="result")
            if not table:
                # try alternate selectors
                rows = soup.select("tr.result")
            else:
                rows = table.select("tbody tr")

            for row in rows[: self.max_records]:
                cols = row.find_all("td")
                if len(cols) >= 2:
                    trial_id = cols[0].get_text(strip=True)
                    title = cols[1].get_text(strip=True)
                    trials.append({
                        "trial_id": trial_id,
                        "title": title,
                        "url": f"https://www.clinicaltrialsregister.eu/ctr-search/trial/{trial_id}/"
                    })
            logger.info(f"Fetched {len(trials)} EMA/EUCTR trials (best-effort)")
        except Exception as e:
            logger.error(f"Error fetching EMA trials: {e}")
        return trials


# ---------------------
# Email sender (supports multiple recipients)
# ---------------------
class EmailSender:
    def __init__(self, config_obj):
        self.config = config_obj

    def _get_receivers(self) -> List[str]:
        recipients = self.config.RECIPIENT_EMAIL or ""
        # Accept comma-separated list of addresses
        receivers = [r.strip() for r in recipients.split(",") if r.strip()]
        if not receivers:
            receivers = [self.config.SENDER_EMAIL]
        return receivers

    def send_comprehensive_report(self, db: NeuroCellDB) -> bool:
        password = self.config.EMAIL_PASSWORD
        if not password:
            logger.error("Missing EMAIL_PASSWORD environment variable")
            return False
        try:
            # Export CSVs
            new_pubmed_csv = db.export_csv("pubmed_articles", "new_pubmed_this_week.csv", new_only=True)
            new_trials_csv = db.export_csv("clinical_trials", "new_trials_this_week.csv", new_only=True)
            new_gs_csv = db.export_csv("google_scholar_articles", "new_google_scholar_this_week.csv", new_only=True)
            new_ema_csv = db.export_csv("ema_trials", "new_ema_this_week.csv", new_only=True)

            all_pubmed_csv = db.export_csv("pubmed_articles", "all_pubmed_database.csv", new_only=False)
            all_trials_csv = db.export_csv("clinical_trials", "all_trials_database.csv", new_only=False)
            all_gs_csv = db.export_csv("google_scholar_articles", "all_google_scholar_database.csv", new_only=False)
            all_ema_csv = db.export_csv("ema_trials", "all_ema_trials_database.csv", new_only=False)

            # Build email
            msg = MIMEMultipart()
            msg["From"] = self.config.SENDER_EMAIL
            receivers = self._get_receivers()
            msg["To"] = ", ".join(receivers)
            msg["Subject"] = f"NeuroCell Intelligence Report - {datetime.now().strftime('%Y-%m-%d')}"

            pubmed_new_count = pd.read_csv(new_pubmed_csv).shape[0] if os.path.exists(new_pubmed_csv) else 0
            trials_new_count = pd.read_csv(new_trials_csv).shape[0] if os.path.exists(new_trials_csv) else 0
            gs_new_count = pd.read_csv(new_gs_csv).shape[0] if os.path.exists(new_gs_csv) else 0
            ema_new_count = pd.read_csv(new_ema_csv).shape[0] if os.path.exists(new_ema_csv) else 0

            pubmed_total_count = pd.read_csv(all_pubmed_csv).shape[0] if os.path.exists(all_pubmed_csv) else 0
            trials_total_count = pd.read_csv(all_trials_csv).shape[0] if os.path.exists(all_trials_csv) else 0
            gs_total_count = pd.read_csv(all_gs_csv).shape[0] if os.path.exists(all_gs_csv) else 0
            ema_total_count = pd.read_csv(all_ema_csv).shape[0] if os.path.exists(all_ema_csv) else 0

            text = f"""
🧬 NeuroCell Intelligence Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Summary Statistics:
- New PubMed Articles this week: {pubmed_new_count}
- Total PubMed Articles in DB: {pubmed_total_count}

- New Clinical Trials (ClinicalTrials.gov) this week: {trials_new_count}
- Total ClinicalTrials.gov Trials in DB: {trials_total_count}

- New Google Scholar items this week: {gs_new_count}
- Total Google Scholar items in DB: {gs_total_count}

- New EMA/EUCTR trials this week: {ema_new_count}
- Total EMA trials in DB: {ema_total_count}

Please see the attached CSV files for details.
"""
            msg.attach(MIMEText(text, "plain"))

            # Attach CSV files if they exist
            for filepath in [new_pubmed_csv, new_trials_csv, new_gs_csv, new_ema_csv, all_pubmed_csv, all_trials_csv, all_gs_csv, all_ema_csv]:
                if os.path.exists(filepath):
                    with open(filepath, "rb") as f:
                        part = MIMEBase("application", "octet-stream")
                        part.set_payload(f.read())
                    encoders.encode_base64(part)
                    part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(filepath)}")
                    msg.attach(part)

            # Send email
            # Use SSL if specified port is 465, otherwise attempt STARTTLS on 587
            if self.config.SMTP_PORT == 465:
                with smtplib.SMTP_SSL(self.config.SMTP_SERVER, self.config.SMTP_PORT) as server:
                    server.login(self.config.SENDER_EMAIL, password)
                    server.sendmail(self.config.SENDER_EMAIL, receivers, msg.as_string())
            else:
                with smtplib.SMTP(self.config.SMTP_SERVER, self.config.SMTP_PORT) as server:
                    server.ehlo()
                    server.starttls()
                    server.login(self.config.SENDER_EMAIL, password)
                    server.sendmail(self.config.SENDER_EMAIL, receivers, msg.as_string())

            logger.info("Email report sent successfully")
            return True

        except Exception as e:
            logger.error(f"Email sending failed: {e}")
            return False


# ---------------------
# Main orchestrator
# ---------------------
class NeuroCellAgent:
    def __init__(self):
        self.config = config
        self.db = NeuroCellDB()
        self.pubmed_fetcher = PubMedFetcher(self.config.NCBI_EMAIL)
        self.trials_fetcher = ClinicalTrialsFetcher()
        self.gs_fetcher = GoogleScholarFetcher(term=self.config.PUBMED_TERM, max_records=config.GS_MAX)
        self.ema_fetcher = EMAClinicalTrialsFetcher(term=self.config.PUBMED_TERM, max_records=config.EMA_MAX)
        self.email_sender = EmailSender(self.config)

    def run_weekly(self, days_back=7):
        logger.info("Starting weekly NeuroCell Intelligence Agent run.")

        # Mark previous entries as old
        self.db.mark_all_old()

        # 1) PubMed
        articles = self.pubmed_fetcher.fetch_articles(self.config.PUBMED_TERM, self.config.MAX_RECORDS, days_back)
        for art in articles:
            self.db.upsert_pubmed_article(art)

        # 2) ClinicalTrials.gov
        trial_queries = [
            "exosomes AND neurological",
            "exosome AND CNS",
            "extracellular vesicles AND brain",
            "exosomes AND spinal cord"
        ]
        trials = self.trials_fetcher.search_trials(trial_queries, self.config.MAX_RECORDS)
        for tr in trials:
            self.db.upsert_clinical_trial(tr)

        # 3) Google Scholar (best-effort)
        try:
            gs_articles = self.gs_fetcher.fetch()
            for a in gs_articles:
                self.db.upsert_google_article(a)
        except Exception as e:
            logger.error(f"Google Scholar fetch failed: {e}")

        # 4) EMA / EU Trials (best-effort)
        try:
            ema_trials = self.ema_fetcher.fetch()
            for t in ema_trials:
                self.db.upsert_ema_trial(t)
        except Exception as e:
            logger.error(f"EMA fetch failed: {e}")

        # 5) Send email report
        success = self.email_sender.send_comprehensive_report(self.db)

        self.db.close()

        if success:
            logger.info("Weekly run completed successfully.")
        else:
            logger.error("Weekly run encountered errors.")

        return success


if __name__ == "__main__":
    agent = NeuroCellAgent()
    ok = agent.run_weekly()
    if ok:
        print("✅ NeuroCell Intelligence Agent completed successfully")
    else:
        print("❌ NeuroCell Intelligence Agent encountered errors")

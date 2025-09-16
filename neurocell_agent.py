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
from dotenv import load_dotenv
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
# Configuration class
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
                (pmid, title, abstract, authors, publication_date, journal, volume, issue, pages, doi, keywords, url, is_new, content_hash, first_seen)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?);
            """, (
                article['pmid'], article['title'], article['abstract'], article['authors'], article['publication_date'], article['journal'],
                article['volume'], article['issue'], article['pages'], article['doi'], ",".join(article['keywords']),
                article['url'], hash_val, now
            ))
            logger.info(f"Added NEW PubMed article: {article['pmid']}")
        else:
            if row[1] != hash_val:
                self.cursor.execute("""
                    UPDATE pubmed_articles SET title=?, abstract=?, authors=?, publication_date=?, journal=?, volume=?, issue=?, pages=?, doi=?, keywords=?, url=?, is_new=1, content_hash=? WHERE pmid=?;
                """, (
                    article['title'], article['abstract'], article['authors'], article['publication_date'], article['journal'],
                    article['volume'], article['issue'], article['pages'], article['doi'], ",".join(article['keywords']),
                    article['url'], hash_val, article['pmid']
                ))
                logger.info(f"Updated & marked NEW PubMed article: {article['pmid']}")
            else:
                self.cursor.execute("UPDATE pubmed_articles SET is_new=0 WHERE pmid=?;", (article['pmid'],))
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
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?);
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
                    completion_date=?, sponsor=?, enrollment=?, age_range=?, url=?, is_new=1, content_hash=? WHERE nctid=?;
                """, (
                    trial['title'], trial['detailed_description'], ",".join(trial['conditions']), ",".join(trial['interventions']),
                    ",".join(trial['phases']), trial['study_type'], trial['status'], trial['start_date'],
                    trial['completion_date'], trial['sponsor'], str(trial['enrollment']), trial['age_range'], trial['url'], hash_val, trial['nctid']
                ))
                logger.info(f"Updated & marked NEW Clinical Trial: {trial['nctid']}")
            else:
                self.cursor.execute("UPDATE clinical_trials SET is_new=0 WHERE nctid=?;", (trial['nctid'],))
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
# PubMed fetcher class
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

    def _extract_keywords(self, medline_citation: Dict) -> List[str]:
        keywords = []
        mesh_list = medline_citation.get("MeshHeadingList", [])
        for mesh in mesh_list[:5]:
            descriptor = mesh.get("DescriptorName", {})
            if hasattr(descriptor, 'attributes') and 'text' in descriptor.attributes:
                keywords.append(descriptor.attributes['text'])
        return keywords

    def _extract_doi(self, article_data: Dict) -> str:
        elocation_id = article_data.get("ELocationID", [])
        if isinstance(elocation_id, list):
            for loc in elocation_id:
                if hasattr(loc, 'attributes') and loc.attributes.get("EIdType") == "doi":
                    return loc.attributes.get('text', '')
        return "N/A"
# ---------------------
# Clinical Trials fetcher class
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
        except json.JSONDecodeError as e:
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
# Email sender class
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
            new_pubmed_csv = db.export_csv("pubmed_articles", "new_pubmed_this_week.csv", new_only=True)
            new_trials_csv = db.export_csv("clinical_trials", "new_trials_this_week.csv", new_only=True)
            all_pubmed_csv = db.export_csv("pubmed_articles", "all_pubmed_database.csv", new_only=False)
            all_trials_csv = db.export_csv("clinical_trials", "all_trials_database.csv", new_only=False)

            msg = MIMEMultipart()
            msg['From'] = self.config.SENDER_EMAIL
            msg['To'] = self.config.RECIPIENT_EMAIL
            msg['Subject'] = f"NeuroCell Intelligence Report - {datetime.now().strftime('%Y-%m-%d')}"

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

            # Attach CSV files
            for filepath in [new_pubmed_csv, new_trials_csv, all_pubmed_csv, all_trials_csv]:
                with open(filepath, "rb") as f:
                    part = MIMEBase("application", "octet-stream")
                    part.set_payload(f.read())
                encoders.encode_base64(part)
                part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(filepath)}")
                msg.attach(part)

            with smtplib.SMTP_SSL(self.config.SMTP_SERVER, self.config.SMTP_PORT) as server:
                server.login(self.config.SENDER_EMAIL, password)
                server.sendmail(self.config.SENDER_EMAIL, self.config.RECIPIENT_EMAIL, msg.as_string())

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
        self.email_sender = EmailSender(self.config)

    def run_weekly(self, days_back=7):
        logger.info("Starting weekly NeuroCell Intelligence Agent run.")

        # Mark all old entries as not new
        self.db.mark_all_old()

        # Fetch and upsert PubMed articles
        articles = self.pubmed_fetcher.fetch_articles(self.config.PUBMED_TERM, self.config.MAX_RECORDS, days_back)
        for art in articles:
            self.db.upsert_pubmed_article(art)

        # Fetch and upsert clinical trials
        trial_queries = [
            "exosomes AND neurological",
            "exosome AND CNS",
            "extracellular vesicles AND brain",
            "exosomes AND spinal cord"
        ]
        trials = self.trials_fetcher.search_trials(trial_queries, self.config.MAX_RECORDS)
        for tr in trials:
            self.db.upsert_clinical_trial(tr)

        # Send email report
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

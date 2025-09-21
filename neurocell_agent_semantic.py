#!/usr/bin/env python3
"""
neurocell_agent_semantic_fixed.py
Requirements:
- biopython
- requests
- python-dotenv
- sentence-transformers
- faiss-cpu (or faiss-gpu)
- torch
"""

import os
import time
import csv
import sqlite3
import logging
import requests
import smtplib
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from typing import List, Dict, Any
from Bio import Entrez
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer, util
import torch

# Load environment
load_dotenv(".env_semantic")

# -------------------------
# Configuration
# -------------------------
DB_FILE = os.getenv("DB_FILE", "neurocell_database_semantic.db")

SENDER_EMAIL = os.getenv("SENDER_EMAIL")
RECIPIENT_EMAIL = os.getenv("RECIPIENT_EMAIL")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 465))

NCBI_EMAIL = os.getenv("NCBI_EMAIL", "chen.limor@gmail.com")
PUBMED_TERM = os.getenv("PUBMED_TERM", "exosomes")
CLINICALTRIALS_INTERVENTION = os.getenv("CLINICALTRIALS_INTERVENTION",
                                        'exosomes OR "extracellular vesicles"')
CLINICALTRIALS_CONDITION = os.getenv("CLINICALTRIALS_CONDITION",
                                     'neurology OR "neurologic disorder"')
SEMANTIC_SEARCH_TERMS = [s.strip() for s in os.getenv(
    "SEMANTIC_SEARCH_TERMS",
    "exosomes and nervous system, extracellular vesicles in spinal cord"
).split(",") if s.strip()]

MAX_RECORDS = int(os.getenv("MAX_RECORDS", 50))
DAYS_BACK = int(os.getenv("DAYS_BACK", 30))
RATE_LIMIT_DELAY = float(os.getenv("RATE_LIMIT_DELAY", 0.34))
SEMANTIC_THRESHOLD = float(os.getenv("SEMANTIC_THRESHOLD", 0.30))

Entrez.email = NCBI_EMAIL

PUBMED_WEEKLY_CSV = "new_pubmed_semantic_this_week.csv"
TRIALS_WEEKLY_CSV = "new_trials_semantic_this_week.csv"
PUBMED_FULL_CSV = "all_pubmed_semantic_database.csv"
TRIALS_FULL_CSV = "all_trials_semantic_database.csv"

# -------------------------
# Logging
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("neurocell_agent_semantic.log"), logging.StreamHandler()]
)
logger = logging.getLogger("neurocell_agent")

# -------------------------
# Semantic Model
# -------------------------
logger.info("Loading Sentence-Transformer model 'all-MiniLM-L6-v2' ...")
try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("Sentence-Transformer loaded.")
except Exception as e:
    logger.error(f"Failed to load SentenceTransformer model: {e}")
    model = None

# -------------------------
# Utils
# -------------------------
def now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def contains_spinal(*texts: List[str]) -> bool:
    for t in texts:
        if t and "spinal" in t.lower():
            return True
    return False

def semantic_filter(docs: List[Dict[str, Any]], terms: List[str], threshold: float) -> List[Dict[str, Any]]:
    if not model:
        logger.warning("Semantic model not loaded — skipping semantic filtering.")
        for d in docs:
            d['semantic_score'] = 0.0
        return docs

    if not docs or not terms:
        for d in docs:
            d['semantic_score'] = 0.0
        return docs

    try:
        term_embeddings = model.encode(terms, convert_to_tensor=True)
    except Exception as e:
        logger.error(f"Error encoding semantic terms: {e}")
        term_embeddings = None

    filtered_docs = []
    for doc in docs:
        title = doc.get('title') or ''
        abstract = doc.get('abstract') or ''
        detailed = doc.get('detailed_description') or ''
        body = abstract.strip() if abstract.strip() else detailed.strip()
        doc_text = (title + " " + body).strip()
        if not doc_text:
            doc['semantic_score'] = 0.0
            continue
        try:
            if term_embeddings is not None:
                doc_embedding = model.encode(doc_text, convert_to_tensor=True)
                cosine_scores = util.cos_sim(doc_embedding, term_embeddings)[0]
                max_score = float(torch.max(cosine_scores).item())
            else:
                max_score = 0.0
        except Exception as e:
            logger.error(f"Error embedding doc '{title[:50]}': {e}")
            max_score = 0.0
        doc['semantic_score'] = round(max_score, 4)
        if max_score >= threshold:
            filtered_docs.append(doc)
    return filtered_docs

# -------------------------
# Database
# -------------------------
def init_db(path: str = DB_FILE):
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS pubmed_articles (
        pmid TEXT PRIMARY KEY,
        title TEXT,
        abstract TEXT,
        authors TEXT,
        publication_date TEXT,
        journal TEXT,
        doi TEXT,
        url TEXT,
        spinal_hit INTEGER,
        first_seen TEXT,
        semantic_score REAL
    );""")
    c.execute("""
    CREATE TABLE IF NOT EXISTS clinical_trials (
        nct_id TEXT PRIMARY KEY,
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
        spinal_hit INTEGER,
        first_seen TEXT,
        semantic_score REAL
    );""")
    conn.commit()
    conn.close()
    logger.info(f"Initialized DB at {path}")

# -------------------------
# PubMed Fetch
# -------------------------
def fetch_pubmed(term: str, max_records: int = MAX_RECORDS, days_back: int = DAYS_BACK) -> List[Dict[str, Any]]:
    try:
        handle = Entrez.esearch(db="pubmed", term=term, retmax=max_records, sort="date", reldate=days_back)
        record = Entrez.read(handle)
        handle.close()
        ids = record.get("IdList", [])
        if not ids: return []
        time.sleep(RATE_LIMIT_DELAY)
        handle = Entrez.efetch(db="pubmed", id=",".join(ids), rettype="abstract", retmode="xml")
        papers = Entrez.read(handle)
        handle.close()
        results = []
        for article in papers.get("PubmedArticle", []):
            med = article.get("MedlineCitation", {})
            pmid = str(med.get("PMID"))
            art = med.get("Article", {}) or {}
            title = art.get("ArticleTitle", "") or ""
            abstract_list = art.get("Abstract", {}).get("AbstractText", [])
            abstract = " ".join([str(a) for a in abstract_list]) if isinstance(abstract_list, list) else str(abstract_list or "")
            authors = []
            for a in art.get("AuthorList", [])[:20]:
                if "LastName" in a and "Initials" in a:
                    authors.append(f"{a['LastName']} {a['Initials']}")
                elif "CollectiveName" in a:
                    authors.append(a.get("CollectiveName"))
            authors_str = ", ".join(authors)
            journal = art.get("Journal", {}).get("Title", "")
            pubdate_struct = art.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {})
            pubdate = pubdate_struct.get("Year") or pubdate_struct.get("MedlineDate", "")
            doi = "N/A"
            elocs = art.get("ELocationID", [])
            if isinstance(elocs, list):
                for e in elocs:
                    if hasattr(e, "attributes") and e.attributes.get("EIdType") == "doi":
                        doi = e.attributes.get("text", "") or doi
            url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            spinal = 1 if contains_spinal(title, abstract) else 0
            results.append({
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "authors": authors_str,
                "publication_date": pubdate,
                "journal": journal,
                "doi": doi,
                "url": url,
                "spinal_hit": spinal,
                "semantic_score": None
            })
        return results
    except Exception as e:
        logger.exception("PubMed fetch error")
        return []

# -------------------------
# ClinicalTrials Fetch
# -------------------------
def fetch_clinical_trials_v1_fallback(expr: str, max_records: int = 100) -> List[Dict[str, Any]]:
    try:
        r = requests.get("https://clinicaltrials.gov/api/query/study_fields", params={
            'expr': expr,
            'fields': 'NCTId,BriefTitle,Condition,BriefSummary,OverallStatus,StudyType,StartDate,CompletionDate',
            'min_rnk': 1, 'max_rnk': max_records, 'fmt': 'json'
        }, timeout=15)
        r.raise_for_status()
        sf = r.json().get('StudyFieldsResponse', {}).get('StudyFields', [])
        results = []
        for s in sf:
            nct = s.get('NCTId',[''])[0]
            title = s.get('BriefTitle',[''])[0]
            summary = s.get('BriefSummary',[''])[0]
            status = s.get('OverallStatus',[''])[0]
            study_type = s.get('StudyType',[''])[0]
            start_date = s.get('StartDate',[''])[0]
            completion_date = s.get('CompletionDate',[''])[0]
            conditions_list = s.get('Condition', []) or []
            results.append({
                "nct_id": nct,
                "title": title,
                "detailed_description": summary,
                "conditions": conditions_list,
                "interventions": [],
                "phases": [],
                "study_type": study_type,
                "status": status,
                "start_date": start_date,
                "completion_date": completion_date,
                "sponsor": "",
                "enrollment": "",
                "age_range": "N/A",
                "url": f"https://clinicaltrials.gov/study/{nct}" if nct else "",
                "spinal_hit": 1 if contains_spinal(title, summary) else 0,
                "semantic_score": None
            })
        return results
    except Exception as e:
        logger.error(f"v1 fallback error: {e}")
        return []

def fetch_clinical_trials(intervention: str, condition: str, days_back: int = DAYS_BACK, max_records: int = MAX_RECORDS) -> List[Dict[str, Any]]:
    base_url = "https://clinicaltrials.gov/api/v2/studies"
    search_results = []
    page_token = None
    date_cutoff = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
    params = {'query.intr': intervention, 'query.cond': condition, 'filter.lastUpdatePostDate': f'{date_cutoff}..', 'pageSize': 100, 'format': 'json'}

    try:
        while len(search_results) < max_records * 2:
            if page_token: params['pageToken'] = page_token
            resp = requests.get(base_url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            studies = data.get('studies', [])
            if not studies: break
            for study in studies:
                proto = study.get('protocolSection',{}) or {}
                nct_id = proto.get('identificationModule',{}).get('nctId')
                title = proto.get('identificationModule',{}).get('briefTitle')
                summary = proto.get('descriptionModule',{}).get('briefSummary')
                status = proto.get('statusModule',{}).get('overallStatus')
                study_type = proto.get('designModule',{}).get('studyType')
                start_date = proto.get('statusModule',{}).get('startDateStruct',{}).get('date')
                completion_date = proto.get('statusModule',{}).get('completionDateStruct',{}).get('date')
                conditions_list = [c.get('name') for c in proto.get('conditionsModule',{}).get('conditions',[])] or []
                interventions_list = [i.get('name') for i in proto.get('armsInterventionsModule',{}).get('interventions',[])] or []
                phases_list = [p.get('phase') for p in proto.get('designModule',{}).get('phases',[])] or []
                sponsor_name = proto.get('sponsorCollaboratorsModule',{}).get('leadSponsor',{}).get('name')
                enrollment = proto.get('designModule',{}).get('enrollmentInfo',{}).get('count')
                age_min = proto.get('eligibilityModule',{}).get('minimumAge')
                age_max = proto.get('eligibilityModule',{}).get('maximumAge')
                age_range = f"{age_min} - {age_max}" if age_min or age_max else "N/A"
                url_study = f"https://clinicaltrials.gov/study/{nct_id}" if nct_id else ""
                spinal_hit = 1 if contains_spinal(title, summary) else 0
                search_results.append({
                    "nct_id": nct_id, "title": title, "detailed_description": summary, "conditions": conditions_list,
                    "interventions": interventions_list, "phases": phases_list, "study_type": study_type,
                    "status": status, "start_date": start_date, "completion_date": completion_date,
                    "sponsor": sponsor_name, "enrollment": enrollment, "age_range": age_range, "url": url_study,
                    "spinal_hit": spinal_hit, "semantic_score": None
                })
            page_token = data.get('nextPageToken')
            if not page_token or len(search_results) >= max_records*2: break
            time.sleep(RATE_LIMIT_DELAY)
    except requests.RequestException as e:
        logger.error(f"Error fetching data from ClinicalTrials.gov API v2: {e}")
    if not search_results:
        return fetch_clinical_trials_v1_fallback(f"{intervention} AND {condition}", max_records=max_records)
    return search_results

# -------------------------
# Database upsert
# -------------------------
def upsert_pubmed(db: str, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(db)
    cur = conn.cursor()
    new_items = []
    for a in articles:
        try:
            cur.execute("""INSERT INTO pubmed_articles (pmid,title,abstract,authors,publication_date,journal,doi,url,spinal_hit,first_seen,semantic_score)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)""", (a["pmid"],a["title"],a["abstract"],a["authors"],a["publication_date"],a["journal"],a["doi"],a["url"],a["spinal_hit"],now_ts(),a["semantic_score"]))
            new_items.append(a)
        except sqlite3.IntegrityError:
            continue
    conn.commit()
    conn.close()
    return new_items

def upsert_trials(db: str, trials: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(db)
    cur = conn.cursor()
    new_items = []
    for t in trials:
        try:
            cur.execute("""INSERT INTO clinical_trials
                (nct_id,title,detailed_description,conditions,interventions,phases,study_type,status,start_date,completion_date,sponsor,enrollment,age_range,url,spinal_hit,first_seen,semantic_score)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (t.get("nct_id"), t.get("title"), t.get("detailed_description"), "; ".join(t.get("conditions",[])),
                 "; ".join(t.get("interventions",[])), "; ".join(t.get("phases",[])),
                 t.get("study_type",""), t.get("status",""), t.get("start_date",""), t.get("completion_date",""),
                 t.get("sponsor",""), str(t.get("enrollment","")), t.get("age_range",""), t.get("url",""),
                 t.get("spinal_hit",0), now_ts(), t.get("semantic_score")))
            new_items.append(t)
        except sqlite3.IntegrityError:
            continue
    conn.commit()
    conn.close()
    return new_items

# -------------------------
# CSV Export
# -------------------------
def export_csv(filename: str, data: List[Dict[str, Any]], fieldnames: List[str]):
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow(row)
    logger.info(f"Exported {len(data)} rows to {filename}")

# -------------------------
# Email sending
# -------------------------
def send_email_with_attachments(subject: str, body: str, attachments: List[str]):
    try:
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = RECIPIENT_EMAIL
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        for file_path in attachments:
            part = MIMEBase('application', 'octet-stream')
            with open(file_path, 'rb') as f:
                part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', f"attachment; filename={os.path.basename(file_path)}")
            msg.attach(part)
        server = smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT)
        server.login(SENDER_EMAIL, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()
        logger.info(f"Email sent with attachments: {attachments}")
    except Exception as e:
        logger.exception("Failed to send email")

# -------------------------
# Main Execution
# -------------------------
def main():
    logger.info("Starting NeuroCell Agent Semantic Fetch...")

    # Initialize DB
    init_db(DB_FILE)

    # -------------------------
    # Fetch PubMed
    # -------------------------
    logger.info(f"Fetching PubMed articles for term: {PUBMED_TERM}")
    pubmed_articles = fetch_pubmed(PUBMED_TERM, max_records=MAX_RECORDS, days_back=DAYS_BACK)
    logger.info(f"Fetched {len(pubmed_articles)} PubMed articles")

    # Semantic filter
    pubmed_articles_filtered = semantic_filter(pubmed_articles, SEMANTIC_SEARCH_TERMS, SEMANTIC_THRESHOLD)
    logger.info(f"{len(pubmed_articles_filtered)} PubMed articles passed semantic filter")

    # Upsert into DB
    new_pubmed_articles = upsert_pubmed(DB_FILE, pubmed_articles_filtered)
    logger.info(f"{len(new_pubmed_articles)} new PubMed articles added to DB")

    # -------------------------
    # Fetch ClinicalTrials
    # -------------------------
    logger.info(f"Fetching ClinicalTrials for intervention: {CLINICALTRIALS_INTERVENTION}, condition: {CLINICALTRIALS_CONDITION}")
    trials = fetch_clinical_trials(CLINICALTRIALS_INTERVENTION, CLINICALTRIALS_CONDITION, days_back=DAYS_BACK, max_records=MAX_RECORDS)
    logger.info(f"Fetched {len(trials)} clinical trials")

    # Semantic filter
    trials_filtered = semantic_filter(trials, SEMANTIC_SEARCH_TERMS, SEMANTIC_THRESHOLD)
    logger.info(f"{len(trials_filtered)} clinical trials passed semantic filter")

    # Upsert into DB
    new_trials = upsert_trials(DB_FILE, trials_filtered)
    logger.info(f"{len(new_trials)} new clinical trials added to DB")

    # -------------------------
    # Export CSVs
    # -------------------------
    if new_pubmed_articles:
        export_csv(PUBMED_WEEKLY_CSV, new_pubmed_articles, list(new_pubmed_articles[0].keys()))
    if new_trials:
        export_csv(TRIALS_WEEKLY_CSV, new_trials, list(new_trials[0].keys()))

    # Export full DB CSV
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("SELECT * FROM pubmed_articles")
    rows = cur.fetchall()
    columns = [desc[0] for desc in cur.description]
    all_pubmed_data = [dict(zip(columns, r)) for r in rows]
    export_csv(PUBMED_FULL_CSV, all_pubmed_data, columns)

    cur.execute("SELECT * FROM clinical_trials")
    rows = cur.fetchall()
    columns = [desc[0] for desc in cur.description]
    all_trials_data = [dict(zip(columns, r)) for r in rows]
    export_csv(TRIALS_FULL_CSV, all_trials_data, columns)
    conn.close()

    # -------------------------
    # Send Email
    # -------------------------
    attachments = []
    if os.path.exists(PUBMED_WEEKLY_CSV):
        attachments.append(PUBMED_WEEKLY_CSV)
    if os.path.exists(TRIALS_WEEKLY_CSV):
        attachments.append(TRIALS_WEEKLY_CSV)

    if attachments:
        subject = f"NeuroCell Agent Weekly Update - {datetime.now().strftime('%Y-%m-%d')}"
        body = f"New PubMed articles: {len(new_pubmed_articles)}\nNew Clinical Trials: {len(new_trials)}"
        send_email_with_attachments(subject, body, attachments)
    else:
        logger.info("No new items to email this week.")

    logger.info("NeuroCell Agent Semantic Fetch completed.")


# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    main()

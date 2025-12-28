#!/usr/bin/env python3
"""
neurocell_agent_semantic_improved.py

Weekly CNS–exosome intelligence agent:
- Fetch PubMed + ClinicalTrials.gov (Cumulative Mode)
- Apply strict exosome filter + semantic filter + combined scoring
- Store in SQLite DB (Persistence check via Primary Key)
- Export ALL historical records to CSV with a new_this_run flag
- Send email report with only newly inserted items
"""

import os
import time
import csv
import sqlite3
import logging
import requests
import smtplib
import re
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

# Load environment variables
if os.path.exists(".env"):
    load_dotenv(".env")
elif os.path.exists(".env_semantic"):
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
NCBI_EMAIL = os.getenv("NCBI_EMAIL", "your-email@example.com")

PUBMED_TERM = os.getenv(
    "PUBMED_TERM",
    "exosomes AND (spinal cord OR neural OR CNS OR optic nerve)"
)
CLINICALTRIALS_INTERVENTION = os.getenv(
    "CLINICALTRIALS_INTERVENTION",
    "exosomes OR extracellular vesicles OR exosome therapy"
)
CLINICALTRIALS_CONDITION = os.getenv(
    "CLINICALTRIALS_CONDITION",
    "spinal cord injury OR optic nerve injury OR central nervous system"
)

MAX_RECORDS = int(os.getenv("MAX_RECORDS", 100))
# Increased to 90 to ensure the agent "re-scans" recent history to fill the DB
DAYS_BACK_PUBMED = int(os.getenv("DAYS_BACK_PUBMED", 90))
DAYS_BACK_TRIALS = int(os.getenv("DAYS_BACK_TRIALS", 90))
RATE_LIMIT_DELAY = float(os.getenv("RATE_LIMIT_DELAY", 0.5))

SEMANTIC_THRESHOLD_PUBMED = float(os.getenv("SEMANTIC_THRESHOLD_PUBMED", 0.45))
SEMANTIC_THRESHOLD_TRIALS = float(os.getenv("SEMANTIC_THRESHOLD_TRIALS", 0.45))

raw_terms = os.getenv(
    "SEMANTIC_SEARCH_TERMS",
    "exosome therapy spinal cord injury, extracellular vesicles neural regeneration"
).split(",")
SEMANTIC_SEARCH_TERMS = [s.strip() for s in raw_terms if s.strip()]

Entrez.email = NCBI_EMAIL

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

logger.info("Loading Sentence-Transformer model 'all-MiniLM-L6-v2' ...")
try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("Sentence-Transformer loaded successfully.")
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
        if not t: continue
        if "spinal" in t.lower() or "sci" in t.lower():
            return True
    return False

def semantic_filter(docs: List[Dict[str, Any]], terms: List[str], threshold: float) -> List[Dict[str, Any]]:
    if not model or not docs or not terms:
        for d in docs: d['semantic_score'] = 0.0
        return docs
    logger.info(f"Applying semantic filtering to {len(docs)} documents...")
    term_embeddings = model.encode(terms, convert_to_tensor=True)
    filtered_docs = []
    for doc in docs:
        title = doc.get('title') or ''
        abstract = doc.get('abstract') or doc.get('detailed_description') or ''
        doc_text = (title + " " + abstract).strip()
        if not doc_text:
            doc['semantic_score'] = 0.0
            continue
        doc_embedding = model.encode(doc_text, convert_to_tensor=True)
        cosine_scores = util.cos_sim(doc_embedding, term_embeddings)[0]
        max_score = float(torch.max(cosine_scores).item())
        doc['semantic_score'] = round(max_score, 4)
        if max_score >= threshold:
            filtered_docs.append(doc)
    return filtered_docs

def mandatory_exosome_filter(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    exosome_patterns = [r'\bexosome', r'\bextracellular vesicle', r'\bEVs\b', r'\bmicrovesicle', r'\bexosomal']
    filtered_docs = []
    for doc in docs:
        title = (doc.get('title') or '').lower()
        abstract = (doc.get('abstract') or doc.get('detailed_description') or '').lower()
        full_text = title + " " + abstract
        if any(re.search(pattern, full_text, re.IGNORECASE) for pattern in exosome_patterns):
            filtered_docs.append(doc)
    return filtered_docs

def calculate_relevance_score(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cns_terms = ['spinal cord', 'spinal', 'sci', 'central nervous system', 'cns', 'optic nerve', 'neural']
    for doc in docs:
        title = (doc.get('title') or '').lower()
        abstract = (doc.get('abstract') or doc.get('detailed_description') or '').lower()
        full_text = title + " " + abstract
        cns_hits = sum(1 for term in cns_terms if term in full_text)
        semantic_score = doc.get('semantic_score', 0)
        doc['combined_score'] = round(semantic_score + (0.1 * min(cns_hits, 3)), 4)
    return docs

# -------------------------
# DB init
# -------------------------
def init_db(path: str = DB_FILE):
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS pubmed_articles (
        pmid TEXT PRIMARY KEY, title TEXT, abstract TEXT, authors TEXT,
        publication_date TEXT, journal TEXT, doi TEXT, url TEXT,
        spinal_hit INTEGER, first_seen TEXT, semantic_score REAL, combined_score REAL
    );
    """)
    c.execute("""
    CREATE TABLE IF NOT EXISTS clinical_trials (
        nct_id TEXT PRIMARY KEY, title TEXT, detailed_description TEXT,
        conditions TEXT, interventions TEXT, phases TEXT, study_type TEXT,
        status TEXT, start_date TEXT, completion_date TEXT, sponsor TEXT,
        enrollment TEXT, age_range TEXT, url TEXT, spinal_hit INTEGER,
        first_seen TEXT, semantic_score REAL, combined_score REAL
    );
    """)
    conn.commit()
    conn.close()

# -------------------------
# PubMed fetcher
# -------------------------
def fetch_pubmed_fixed(term: str, max_records: int = MAX_RECORDS, days_back: int = DAYS_BACK_PUBMED) -> List[Dict[str, Any]]:
    logger.info(f"PubMed search (90-day window): '{term}'")
    try:
        search_handle = Entrez.esearch(db="pubmed", term=term, retmax=max_records, sort="date", reldate=days_back)
        search_record = Entrez.read(search_handle)
        ids = search_record.get("IdList", [])
        if not ids: return []

        all_results = []
        batch_size = 20
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i+batch_size]
            fetch_handle = Entrez.efetch(db="pubmed", id=",".join(batch_ids), rettype="abstract", retmode="xml")
            papers = Entrez.read(fetch_handle)
            for article in papers.get("PubmedArticle", []):
                try:
                    med = article.get("MedlineCitation", {})
                    pmid = str(med.get("PMID", ""))
                    art = med.get("Article", {})
                    title = str(art.get("ArticleTitle", ""))
                    
                    abstract_list = art.get("Abstract", {}).get("AbstractText", [])
                    abstract = " ".join([str(a) for a in abstract_list]) if isinstance(abstract_list, list) else str(abstract_list)
                    
                    # Author parsing
                    authors = []
                    for a in art.get("AuthorList", [])[:10]:
                        if "LastName" in a and "Initials" in a:
                            authors.append(f"{a['LastName']} {a['Initials']}")
                    
                    journal = str(art.get("Journal", {}).get("Title", ""))
                    pubdate = str(art.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {}).get("Year", "N/A"))

                    all_results.append({
                        "pmid": pmid, "title": title, "abstract": abstract,
                        "authors": ", ".join(authors), "publication_date": pubdate,
                        "journal": journal, "doi": "N/A", "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                        "spinal_hit": 1 if contains_spinal(title, abstract) else 0
                    })
                except: continue
            time.sleep(RATE_LIMIT_DELAY)
        return all_results
    except Exception as e:
        logger.error(f"PubMed Error: {e}"); return []

# -------------------------
# ClinicalTrials fetcher
# -------------------------
def fetch_clinical_trials_fixed(search_intervention: str, search_condition: str, days_back: int = DAYS_BACK_TRIALS, max_records: int = MAX_RECORDS) -> List[Dict[str, Any]]:
    base_url = "https://clinicaltrials.gov/api/v2/studies"
    date_cutoff = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
    params = {
        'query.intr': search_intervention, 'query.cond': search_condition,
        'filter.advanced': f'AREA[LastUpdatePostDate]RANGE[{date_cutoff},MAX]',
        'pageSize': max_records, 'format': 'json'
    }
    try:
        resp = requests.get(base_url, params=params, timeout=30)
        studies = resp.json().get('studies', [])
        results = []
        for study in studies:
            proto = study.get('protocolSection', {})
            ident = proto.get('identificationModule', {})
            desc = proto.get('descriptionModule', {})
            results.append({
                "nct_id": ident.get('nctId'), "title": ident.get('briefTitle'),
                "detailed_description": desc.get('briefSummary', ''),
                "conditions": proto.get('conditionsModule', {}).get('conditions', []),
                "interventions": [i.get('name') for i in proto.get('armsInterventionsModule', {}).get('interventions', [])],
                "phases": proto.get('designModule', {}).get('phases', []),
                "status": proto.get('statusModule', {}).get('overallStatus', ''),
                "url": f"https://clinicaltrials.gov/study/{ident.get('nctId')}",
                "spinal_hit": 1 if contains_spinal(ident.get('briefTitle'), desc.get('briefSummary', '')) else 0
            })
        return results
    except Exception as e:
        logger.error(f"Trials Error: {e}"); return []

# -------------------------
# DB upsert
# -------------------------
def upsert_pubmed(db: str, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(db); cur = conn.cursor(); new_items = []
    for a in articles:
        try:
            cur.execute("""INSERT INTO pubmed_articles VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (a["pmid"], a["title"], a["abstract"], a.get("authors"), a.get("publication_date"),
                 a.get("journal"), a.get("doi"), a.get("url"), a["spinal_hit"], now_ts(),
                 a.get("semantic_score"), a.get("combined_score")))
            new_items.append(a)
        except sqlite3.IntegrityError: continue
    conn.commit(); conn.close(); return new_items

def upsert_trials(db: str, trials: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(db); cur = conn.cursor(); new_items = []
    for t in trials:
        try:
            cur.execute("""INSERT INTO clinical_trials VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (t["nct_id"], t["title"], t["detailed_description"], "; ".join(t.get("conditions", [])),
                 "; ".join(t.get("interventions", [])), "; ".join(t.get("phases", [])), "",
                 t.get("status"), "", "", "", "", "", t["url"], t["spinal_hit"], now_ts(),
                 t.get("semantic_score"), t.get("combined_score")))
            new_items.append(t)
        except sqlite3.IntegrityError: continue
    conn.commit(); conn.close(); return new_items

# -------------------------
# CSV CUMULATIVE EXPORT
# -------------------------
def export_full_csvs(db: str = DB_FILE):
    conn = sqlite3.connect(db); cur = conn.cursor(); now = datetime.now()
    
    # PubMed
    cur.execute("SELECT * FROM pubmed_articles")
    rows = cur.fetchall()
    with open(PUBMED_FULL_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["pmid","title","abstract","authors","pub_date","journal","doi","url","spinal","first_seen","sem_score","comb_score","new_this_run"])
        for r in rows:
            is_new = "YES" if (now - datetime.strptime(r[9], "%Y-%m-%d %H:%M:%S")).total_seconds() < 86400 else "NO"
            writer.writerow(list(r) + [is_new])

    # Trials
    cur.execute("SELECT * FROM clinical_trials")
    rows = cur.fetchall()
    with open(TRIALS_FULL_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["nct_id","title","desc","cond","intv","phase","type","status","start","end","sponsor","enroll","age","url","spinal","first_seen","sem","comb","new_this_run"])
        for r in rows:
            is_new = "YES" if (now - datetime.strptime(r[15], "%Y-%m-%d %H:%M:%S")).total_seconds() < 86400 else "NO"
            writer.writerow(list(r) + [is_new])
    conn.close()

# -------------------------
# Email Notification
# -------------------------
def send_email(new_p, new_t):
    if not (SENDER_EMAIL and RECIPIENT_EMAIL and EMAIL_PASSWORD): return
    msg = MIMEMultipart()
    msg['Subject'] = f"NeuroCell Report: {len(new_p)} New Papers, {len(new_t)} New Trials"
    body = f"Total New Items Found Today: {len(new_p) + len(new_t)}\n\nCheck attached CSVs for the full cumulative database."
    msg.attach(MIMEText(body, 'plain'))
    
    for file in [PUBMED_FULL_CSV, TRIALS_FULL_CSV]:
        if os.path.exists(file):
            with open(file, "rb") as attachment:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(attachment.read())
                encoders.encode_base64(part)
                part.add_header("Content-Disposition", f"attachment; filename= {file}")
                msg.attach(part)
    
    with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
        server.login(SENDER_EMAIL, EMAIL_PASSWORD)
        server.sendmail(SENDER_EMAIL, RECIPIENT_EMAIL.split(","), msg.as_string())

# -------------------------
# Main Execution
# -------------------------
def main():
    init_db()
    # 1. Discovery
    p_raw = fetch_pubmed_fixed(PUBMED_TERM)
    t_raw = fetch_clinical_trials_fixed(CLINICALTRIALS_INTERVENTION, CLINICALTRIALS_CONDITION)
    
    # 2. Filtering & Scoring
    p_filt = calculate_relevance_score(semantic_filter(mandatory_exosome_filter(p_raw), SEMANTIC_SEARCH_TERMS, SEMANTIC_THRESHOLD_PUBMED))
    t_filt = calculate_relevance_score(semantic_filter(mandatory_exosome_filter(t_raw), SEMANTIC_SEARCH_TERMS, SEMANTIC_THRESHOLD_TRIALS))
    
    # 3. Save & Export
    new_p = upsert_pubmed(DB_FILE, p_filt)
    new_t = upsert_trials(DB_FILE, t_filt)
    export_full_csvs(DB_FILE)
    
    # 4. Notify
    if new_p or new_t:
        send_email(new_p, new_t)
        logger.info("New findings found and emailed.")
    else:
        logger.info("No new items to report today.")

if __name__ == "__main__":
    main()


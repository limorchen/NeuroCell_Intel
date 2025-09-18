#!/usr/bin/env python3
"""
neurocell_agent_semantic.py

Requirements (pip):
- biopython
- requests
- python-dotenv
- sentence-transformers
- faiss-cpu (or faiss-gpu)
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

# Semantic search libraries
from sentence_transformers import SentenceTransformer, util
import torch

# Load .env_semantic file for manual local runs
load_dotenv(".env_semantic")

# -------------------------
# Configuration
# -------------------------
DB_FILE = os.getenv("DB_FILE", "neurocell_database_semantic.db") # NEW DB FILE NAME

SENDER_EMAIL = os.getenv("SENDER_EMAIL")
RECIPIENT_EMAIL = os.getenv("RECIPIENT_EMAIL")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 465))

NCBI_EMAIL = os.getenv("NCBI_EMAIL", "chen.limor@gmail.com")
PUBMED_TERM = os.getenv("PUBMED_TERM", "exosomes AND CNS")
CLINICALTRIALS_INTERVENTION = os.getenv(
    "CLINICALTRIALS_INTERVENTION",
    'exosomes OR "extracellular vesicles"'
)
CLINICALTRIALS_CONDITION = os.getenv(
    "CLINICALTRIALS_CONDITION",
    'neurology OR "neurologic disorder"'
)
# New semantic search terms
SEMANTIC_SEARCH_TERMS = os.getenv(
    "SEMANTIC_SEARCH_TERMS",
    "exosomes and nervous system, extracellular vesicles for CNS regeneration"
).split(',')

MAX_RECORDS = int(os.getenv("MAX_RECORDS", 50))
DAYS_BACK = int(os.getenv("DAYS_BACK", 7))
RATE_LIMIT_DELAY = float(os.getenv("RATE_LIMIT_DELAY", 0.34))
SEMANTIC_THRESHOLD = float(os.getenv("SEMANTIC_THRESHOLD", 0.5))

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

# Initialize the semantic model globally
logger.info("Loading Sentence-Transformer model...")
try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
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
        if not t:
            continue
        if "spinal" in t.lower():
            return True
    return False

def semantic_filter(docs: List[Dict[str, Any]], terms: List[str], threshold: float) -> List[Dict[str, Any]]:
    """
    Filters documents based on semantic similarity to a list of terms.
    """
    if not model or not docs or not terms:
        return docs

    logger.info(f"Performing semantic filtering on {len(docs)} documents...")
    # Get embeddings for search terms
    term_embeddings = model.encode(terms, convert_to_tensor=True)

    filtered_docs = []
    for doc in docs:
        doc_text = doc.get('title', '') + " " + doc.get('abstract', '') or doc.get('detailed_description', '')
        if not doc_text:
            continue

        doc_embedding = model.encode(doc_text, convert_to_tensor=True)
        # Calculate cosine similarity
        cosine_scores = util.cos_sim(doc_embedding, term_embeddings)[0]
        max_score = torch.max(cosine_scores).item()

        if max_score >= threshold:
            doc['semantic_score'] = round(max_score, 4)
            filtered_docs.append(doc)
    
    logger.info(f"Semantic filtering kept {len(filtered_docs)} documents.")
    return filtered_docs

# -------------------------
# DB init
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
    );
    """)
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
    );
    """)
    conn.commit()
    conn.close()

# -------------------------
# PubMed fetcher
# -------------------------
def fetch_pubmed(term: str, max_records: int = MAX_RECORDS, days_back: int = DAYS_BACK) -> List[Dict[str, Any]]:
    logger.info(f"PubMed search term: {term} | days_back={days_back} | retmax={max_records}")
    try:
        handle = Entrez.esearch(
            db="pubmed",
            term=term,
            retmax=max_records,
            sort="date",
            reldate=days_back
        )
        record = Entrez.read(handle)
        handle.close()
        ids = record.get("IdList", [])
        logger.info(f"PubMed esearch returned {len(ids)} ids")
        if not ids:
            return []

        time.sleep(RATE_LIMIT_DELAY)
        handle = Entrez.efetch(db="pubmed", id=",".join(ids), rettype="abstract", retmode="xml")
        papers = Entrez.read(handle)
        handle.close()

        results = []
        for article in papers.get("PubmedArticle", []):
            med = article.get("MedlineCitation", {})
            pmid = str(med.get("PMID"))
            art = med.get("Article", {})
            title = art.get("ArticleTitle", "") or ""
            abstract_list = art.get("Abstract", {}).get("AbstractText", [])
            if isinstance(abstract_list, list):
                abstract = " ".join([str(a) for a in abstract_list])
            else:
                abstract = str(abstract_list) if abstract_list else ""
            authors = []
            for a in art.get("AuthorList", [])[:20]:
                if "LastName" in a and "Initials" in a:
                    authors.append(f"{a.get('LastName','')} {a.get('Initials','')}")
                elif "CollectiveName" in a:
                    authors.append(a.get("CollectiveName"))
            authors_str = ", ".join(authors)
            journal = art.get("Journal", {}).get("Title", "")
            pubdate = ""
            ji = art.get("Journal", {}).get("JournalIssue", {})
            if ji:
                pubdate_struct = ji.get("PubDate", {})
                if isinstance(pubdate_struct, dict):
                    pubdate = pubdate_struct.get("Year", "") or pubdate_struct.get("MedlineDate", "")
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
        logger.info(f"PubMed parsed {len(results)} articles")
        return results
    except Exception as e:
        logger.exception("PubMed fetch error")
        return []

# -------------------------
# ClinicalTrials fetcher - UPDATED FOR V2.0 API
# -------------------------
def fetch_clinical_trials(
    intervention: str,
    condition: str,
    days_back: int = DAYS_BACK,
    max_records: int = MAX_RECORDS
) -> List[Dict[str, Any]]:
    logger.info(f"ClinicalTrials.gov search: intervention='{intervention}', condition='{condition}', days_back={days_back}")
    
    base_url = "https://clinicaltrials.gov/api/v2/studies"
    search_results = []
    page_token = None
    
    date_cutoff = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
    
    params = {
        'query.intr': intervention,
        'query.cond': condition,
        'filter.lastUpdatePostDate': f'CURRENT | {date_cutoff}',
        'pageSize': 100,
        'format': 'json',
    }

    while len(search_results) < max_records * 2:
        if page_token:
            params['pageToken'] = page_token
            
        try:
            response = requests.get(base_url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            studies = data.get('studies', [])
            if not studies:
                break
                
            for study in studies:
                protocol_section = study.get('protocolSection', {})
                nct_id = protocol_section.get('identificationModule', {}).get('nctId')
                title = protocol_section.get('identificationModule', {}).get('briefTitle')
                summary = protocol_section.get('descriptionModule', {}).get('briefSummary')
                status = protocol_section.get('statusModule', {}).get('overallStatus')
                study_type = protocol_section.get('designModule', {}).get('studyType')
                start_date = protocol_section.get('statusModule', {}).get('startDateStruct', {}).get('date')
                completion_date = protocol_section.get('statusModule', {}).get('completionDateStruct', {}).get('date')
                
                conditions_list = [c.get('name') for c in protocol_section.get('conditionsModule', {}).get('conditions', [])]
                interventions_list = [i.get('name') for i in protocol_section.get('armsInterventionsModule', {}).get('interventions', [])]
                phases_list = [p.get('phase') for p in protocol_section.get('designModule', {}).get('phases', [])]
                sponsor_name = protocol_section.get('sponsorCollaboratorsModule', {}).get('leadSponsor', {}).get('name')
                enrollment = protocol_section.get('designModule', {}).get('enrollmentInfo', {}).get('count')
                age_min = protocol_section.get('eligibilityModule', {}).get('minimumAge')
                age_max = protocol_section.get('eligibilityModule', {}).get('maximumAge')
                age_range = f"{age_min} - {age_max}" if age_min or age_max else "N/A"
                
                url_study = f"https://clinicaltrials.gov/study/{nct_id}" if nct_id else ""
                spinal_hit = 1 if contains_spinal(title, summary) else 0

                search_results.append({
                    "nct_id": nct_id,
                    "title": title,
                    "detailed_description": summary,
                    "conditions": conditions_list,
                    "interventions": interventions_list,
                    "phases": phases_list,
                    "study_type": study_type,
                    "status": status,
                    "start_date": start_date,
                    "completion_date": completion_date,
                    "sponsor": sponsor_name,
                    "enrollment": enrollment,
                    "age_range": age_range,
                    "url": url_study,
                    "spinal_hit": spinal_hit,
                    "semantic_score": None
                })
            
            page_token = data.get('nextPageToken')
            if not page_token or len(search_results) >= max_records * 2:
                break
            
            time.sleep(RATE_LIMIT_DELAY)

        except requests.RequestException as e:
            logger.error(f"Error fetching data from ClinicalTrials.gov API: {e}")
            break
            
    logger.info(f"Found and parsed {len(search_results)} new clinical trials.")
    return search_results

# -------------------------
# DB upsert helpers
# -------------------------
def upsert_pubmed(db: str, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(db)
    cur = conn.cursor()
    new_items = []
    for a in articles:
        try:
            cur.execute("""
                INSERT INTO pubmed_articles
                (pmid, title, abstract, authors, publication_date, journal, doi, url, spinal_hit, first_seen, semantic_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (a["pmid"], a["title"], a["abstract"], a["authors"], a["publication_date"], a["journal"], a["doi"], a["url"], a["spinal_hit"], now_ts(), a["semantic_score"]))
            new_items.append(a)
        except sqlite3.IntegrityError:
            continue
    conn.commit()
    conn.close()
    logger.info(f"DB upsert_pubmed: inserted {len(new_items)} new")
    return new_items

def upsert_trials(db: str, trials: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(db)
    cur = conn.cursor()
    new_items = []
    for t in trials:
        try:
            cur.execute("""
                INSERT INTO clinical_trials
                (nct_id, title, detailed_description, conditions, interventions, phases, study_type, status, start_date, completion_date, sponsor, enrollment, age_range, url, spinal_hit, first_seen, semantic_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                t["nct_id"], t["title"], t["detailed_description"], "; ".join(t.get("conditions", [])),
                "; ".join(t.get("interventions", [])), "; ".join(t.get("phases", [])),
                t.get("study_type", ""), t.get("status", ""), t.get("start_date", ""), t.get("completion_date", ""),
                t.get("sponsor", ""), str(t.get("enrollment", "")), t.get("age_range", ""), t.get("url", ""), t.get("spinal_hit", 0), now_ts(), t["semantic_score"]
            ))
            new_items.append(t)
        except sqlite3.IntegrityError:
            continue
    conn.commit()
    conn.close()
    logger.info(f"DB upsert_trials: inserted {len(new_items)} new")
    return new_items

# -------------------------
# CSV helpers
# -------------------------
def append_pubmed_csv(rows: List[Dict[str, Any]], path: str = PUBMED_WEEKLY_CSV):
    if not rows: return
    file_exists = os.path.isfile(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        fieldnames = ["pmid","title","abstract","authors","publication_date","journal","doi","url","spinal_hit","first_seen","semantic_score"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for r in rows:
            writer.writerow({
                "pmid": r["pmid"],
                "title": r["title"],
                "abstract": r["abstract"],
                "authors": r["authors"],
                "publication_date": r["publication_date"],
                "journal": r["journal"],
                "doi": r["doi"],
                "url": r["url"],
                "spinal_hit": "YES" if r["spinal_hit"] else "NO",
                "first_seen": now_ts(),
                "semantic_score": r.get("semantic_score", "")
            })

def append_trials_csv(rows: List[Dict[str, Any]], path: str = TRIALS_WEEKLY_CSV):
    if not rows: return
    file_exists = os.path.isfile(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        fieldnames = ["nct_id","title","detailed_description","conditions","interventions","phases","study_type","status","start_date","completion_date","sponsor","enrollment","age_range","url","spinal_hit","first_seen","semantic_score"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for r in rows:
            writer.writerow({
                "nct_id": r["nct_id"],
                "title": r["title"],
                "detailed_description": r["detailed_description"],
                "conditions": "; ".join(r.get("conditions", [])),
                "interventions": "; ".join(r.get("interventions", [])),
                "phases": "; ".join(r.get("phases", [])),
                "study_type": r.get("study_type",""),
                "status": r.get("status",""),
                "start_date": r.get("start_date",""),
                "completion_date": r.get("completion_date",""),
                "sponsor": r.get("sponsor",""),
                "enrollment": r.get("enrollment",""),
                "age_range": r.get("age_range",""),
                "url": r.get("url",""),
                "spinal_hit": "YES" if r["spinal_hit"] else "NO",
                "first_seen": now_ts(),
                "semantic_score": r.get("semantic_score", "")
            })

# -------------------------
# Export full CSVs
# -------------------------
def export_full_csvs(db: str = DB_FILE):
    conn = sqlite3.connect(db)
    cur = conn.cursor()
    # PubMed
    cur.execute("SELECT pmid,title,abstract,authors,publication_date,journal,doi,url,spinal_hit,first_seen,semantic_score FROM pubmed_articles")
    rows = cur.fetchall()
    with open(PUBMED_FULL_CSV,"w",newline="",encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["pmid","title","abstract","authors","publication_date","journal","doi","url","spinal_hit","first_seen","semantic_score"])
        for r in rows:
            writer.writerow([r[0],r[1],r[2],r[3],r[4],r[5],r[6],r[7],"YES" if r[8] else "NO",r[9], r[10]])
    # Trials
    cur.execute("SELECT nct_id,title,detailed_description,conditions,interventions,phases,study_type,status,start_date,completion_date,sponsor,enrollment,age_range,url,spinal_hit,first_seen,semantic_score FROM clinical_trials")
    rows = cur.fetchall()
    with open(TRIALS_FULL_CSV,"w",newline="",encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["nct_id","title","detailed_description","conditions","interventions","phases","study_type","status","start_date","completion_date","sponsor","enrollment","age_range","url","spinal_hit","first_seen","semantic_score"])
        for r in rows:
            writer.writerow([r[0],r[1],r[2],r[3],r[4],r[5],r[6],r[7],r[8],r[9],r[10],r[11],r[12],r[13],"YES" if r[14] else "NO",r[15], r[16]])
    conn.close()
    logger.info("Exported full DB CSVs")

# -------------------------
# Email function
# -------------------------
def send_email(new_pubmed: List[Dict[str,Any]], new_trials: List[Dict[str,int]], stats: Dict[str,int], pubmed_term: str, trials_intervention: str, trials_condition: str) -> bool:
    if not (SENDER_EMAIL and RECIPIENT_EMAIL and EMAIL_PASSWORD):
        logger.error("Email credentials missing - skipping email send")
        return False
    recipients = [r.strip() for r in RECIPIENT_EMAIL.split(",")]
    msg = MIMEMultipart("mixed")
    msg["From"] = SENDER_EMAIL
    msg["To"] = ", ".join(recipients)
    msg["Subject"] = f"NeuroCell Intelligence Report - {datetime.now().strftime('%Y-%m-%d')}"
    html = f"<h2>NeuroCell Intelligence Report</h2>"
    html += f"<p><b>PubMed search term:</b> {pubmed_term}</p>"
    html += f"<p><b>ClinicalTrials search terms:</b> {trials_intervention} (Intervention), {trials_condition} (Condition)</p>"
    html += f"<p>New PubMed articles this week: {stats.get('new_pubmed',0)}</p>"
    html += f"<p>New Clinical Trials this week: {stats.get('new_trials',0)}</p>"
    if new_pubmed:
        html += "<h3>New PubMed Articles (with Semantic Scores)</h3><ul>"
        for a in sorted(new_pubmed, key=lambda x: x.get('semantic_score', 0), reverse=True):
            html += f"<li><a href='{a['url']}'>{a['title']}</a> (Score: {a.get('semantic_score', 'N/A')})</li>"
        html += "</ul>"
    if new_trials:
        html += "<h3>New Clinical Trials (with Semantic Scores)</h3><ul>"
        for t in sorted(new_trials, key=lambda x: x.get('semantic_score', 0), reverse=True):
            html += f"<li><a href='{t['url']}'>{t['title']}</a> ({t.get('status','')}) (Score: {t.get('semantic_score', 'N/A')})</li>"
        html += "</ul>"
    part1 = MIMEText(html, "html")
    msg.attach(part1)
    try:
        attachments = [PUBMED_WEEKLY_CSV, TRIALS_WEEKLY_CSV, PUBMED_FULL_CSV, TRIALS_FULL_CSV]
        for fname in attachments:
            if os.path.exists(fname):
                with open(fname, "rb") as f:
                    part = MIMEBase("application", "octet-stream")
                    part.set_payload(f.read())
                encoders.encode_base64(part)
                part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(fname)}")
                msg.attach(part)
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
            server.login(SENDER_EMAIL, EMAIL_PASSWORD)
            server.sendmail(SENDER_EMAIL, recipients, msg.as_string())
        logger.info("Email sent successfully")
        return True
    except Exception as e:
        logger.exception("Failed to send email")
        return False

# -------------------------
# Main weekly update
# -------------------------
def weekly_update():
    init_db()

    # Step 1: Fetch a broad set of data using keyword-based APIs
    pubmed_articles = fetch_pubmed(PUBMED_TERM, MAX_RECORDS*2, DAYS_BACK)
    trials = fetch_clinical_trials(CLINICALTRIALS_INTERVENTION, CLINICALTRIALS_CONDITION, DAYS_BACK, MAX_RECORDS*2)

    # Step 2: Filter and rank the data using semantic search
    relevant_pubmed = semantic_filter(pubmed_articles, SEMANTIC_SEARCH_TERMS, SEMANTIC_THRESHOLD)
    relevant_trials = semantic_filter(trials, SEMANTIC_SEARCH_TERMS, SEMANTIC_THRESHOLD)

    # Step 3: Take the top N results
    final_pubmed = sorted(relevant_pubmed, key=lambda x: x['semantic_score'], reverse=True)[:MAX_RECORDS]
    final_trials = sorted(relevant_trials, key=lambda x: x['semantic_score'], reverse=True)[:MAX_RECORDS]

    # Step 4: Upsert into DB and get new items
    new_pubmed = upsert_pubmed(DB_FILE, final_pubmed)
    new_trials = upsert_trials(DB_FILE, final_trials)

    # Step 5: Append weekly CSVs
    append_pubmed_csv(new_pubmed)
    append_trials_csv(new_trials)

    # Step 6: Export full CSVs
    export_full_csvs()

    # Step 7: Send summary email
    stats = {"new_pubmed": len(new_pubmed), "new_trials": len(new_trials)}
    send_email(new_pubmed, new_trials, stats, PUBMED_TERM, CLINICALTRIALS_INTERVENTION, CLINICALTRIALS_CONDITION)

# -------------------------
# Entry point for manual execution
# -------------------------
if __name__ == "__main__":
    weekly_update()
    weekly_update()

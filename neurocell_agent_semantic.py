#!/usr/bin/env python3
"""
neurocell_agent_semantic_fixed.py

This script automates the search for relevant scientific articles and clinical trials
using a combination of keyword and semantic search. It fetches data from PubMed and
ClinicalTrials.gov, filters it for relevance, stores it in an SQLite database,
exports it to CSV files, and sends an email report.
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

# Load environment variables from the .env_semantic file
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
# PubMed search using correct MeSH and field syntax for "exosomes AND nerve"
PUBMED_TERM = os.getenv("PUBMED_TERM", "exosomes[Title/Abstract] AND (nerve[Title/Abstract] OR nerves[Title/Abstract] OR neural[Title/Abstract] OR nervous[Title/Abstract])")
# ClinicalTrials.gov search for interventions only - no condition restriction
CLINICALTRIALS_INTERVENTION = os.getenv("CLINICALTRIALS_INTERVENTION", "exosomes")
CLINICALTRIALS_CONDITION = os.getenv("CLINICALTRIALS_CONDITION", "")
MAX_RECORDS = int(os.getenv("MAX_RECORDS", 50))
DAYS_BACK = int(os.getenv("DAYS_BACK", 30))
RATE_LIMIT_DELAY = float(os.getenv("RATE_LIMIT_DELAY", 0.5))
# Lowered threshold for better filtering
SEMANTIC_THRESHOLD = float(os.getenv("SEMANTIC_THRESHOLD", 0.25))
SEMANTIC_SEARCH_TERMS = [s.strip() for s in os.getenv(
    "SEMANTIC_SEARCH_TERMS",
    "exosomes nervous system, extracellular vesicles spinal cord, neural regeneration, neurological therapy, brain injury treatment, stem cell therapy nervous system"
).split(",") if s.strip()]

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
        if not t:
            continue
        if "spinal" in t.lower():
            return True
    return False

def semantic_filter(docs: List[Dict[str, Any]], terms: List[str], threshold: float) -> List[Dict[str, Any]]:
    if not model:
        logger.warning("Semantic model not loaded — skipping semantic filtering.")
        for d in docs:
            d['semantic_score'] = 0.0
        return docs

    if not docs:
        logger.info("No documents to filter")
        return []

    if not terms:
        logger.warning("No semantic search terms provided")
        for d in docs:
            d['semantic_score'] = 0.0
        return docs

    logger.info(f"Applying semantic filtering to {len(docs)} documents with {len(terms)} search terms")
    logger.info(f"Semantic threshold: {threshold}")

    try:
        term_embeddings = model.encode(terms, convert_to_tensor=True)
    except Exception as e:
        logger.error(f"Error encoding semantic terms: {e}")
        term_embeddings = None

    filtered_docs = []
    for i, doc in enumerate(docs):
        title = doc.get('title') or ''
        abstract = doc.get('abstract') or ''
        detailed = doc.get('detailed_description') or ''
        body = abstract.strip() if abstract.strip() else detailed.strip()
        doc_text = (title + " " + body).strip()
        
        if not doc_text:
            logger.debug(f"Document {i+1}: Empty text content")
            doc['semantic_score'] = 0.0
            continue

        try:
            if term_embeddings is not None:
                doc_embedding = model.encode(doc_text, convert_to_tensor=True)
                cosine_scores = util.cos_sim(doc_embedding, term_embeddings)[0]
                max_score = float(torch.max(cosine_scores).item())
                best_term_idx = int(torch.argmax(cosine_scores).item())
                best_term = terms[best_term_idx]
                logger.debug(f"Document {i+1}: '{title[:50]}...' scored {max_score:.4f} (best match: '{best_term}')")
            else:
                max_score = 0.0
                logger.debug(f"Document {i+1}: No embeddings available")
        except Exception as e:
            logger.error(f"Error embedding doc '{title[:50]}': {e}")
            max_score = 0.0

        doc['semantic_score'] = round(max_score, 4)
        if max_score >= threshold:
            filtered_docs.append(doc)
            logger.info(f"Document passed filter: '{title[:50]}...' (score: {max_score:.4f})")

    logger.info(f"Semantic filtering: {len(filtered_docs)}/{len(docs)} documents passed threshold {threshold}")
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
# PubMed fetcher (Fixed)
# -------------------------
def fetch_pubmed(term: str, max_records: int = MAX_RECORDS, days_back: int = DAYS_BACK) -> List[Dict[str, Any]]:
    logger.info(f"PubMed search term: {term}")
    logger.info(f"Parameters: days_back={days_back}, retmax={max_records}")
    
    try:
        # First, get the IDs
        search_handle = Entrez.esearch(
            db="pubmed", 
            term=term, 
            retmax=max_records, 
            sort="date",
            reldate=days_back
        )
        search_record = Entrez.read(search_handle)
        search_handle.close()
        
        ids = search_record.get("IdList", [])
        logger.info(f"PubMed esearch returned {len(ids)} article IDs")
        
        if not ids:
            logger.warning("No PubMed articles found for the given search criteria")
            return []

        time.sleep(RATE_LIMIT_DELAY)
        
        # Fetch the article details
        fetch_handle = Entrez.efetch(
            db="pubmed", 
            id=",".join(ids), 
            rettype="abstract", 
            retmode="xml"
        )
        papers = Entrez.read(fetch_handle)
        fetch_handle.close()

        results = []
        for i, article in enumerate(papers.get("PubmedArticle", [])):
            try:
                med = article.get("MedlineCitation", {})
                pmid = str(med.get("PMID", ""))
                art = med.get("Article", {}) or {}
                
                # Extract title
                title = str(art.get("ArticleTitle", "")) or ""
                
                # Extract abstract
                abstract_list = art.get("Abstract", {}).get("AbstractText", [])
                if isinstance(abstract_list, list):
                    abstract = " ".join([str(a) for a in abstract_list])
                else:
                    abstract = str(abstract_list) if abstract_list else ""
                
                # Extract authors
                authors = []
                author_list = art.get("AuthorList", [])
                if isinstance(author_list, list):
                    for a in author_list[:10]:  # Limit to first 10 authors
                        if isinstance(a, dict):
                            if "LastName" in a and "Initials" in a:
                                authors.append(f"{a.get('LastName','')} {a.get('Initials','')}")
                            elif "CollectiveName" in a:
                                authors.append(a.get("CollectiveName", ""))
                authors_str = ", ".join(authors)
                
                # Extract journal
                journal = str(art.get("Journal", {}).get("Title", ""))
                
                # Extract publication date
                pubdate = ""
                ji = art.get("Journal", {}).get("JournalIssue", {})
                if ji:
                    pubdate_struct = ji.get("PubDate", {})
                    if isinstance(pubdate_struct, dict):
                        pubdate = str(pubdate_struct.get("Year", "") or pubdate_struct.get("MedlineDate", ""))
                
                # Extract DOI
                doi = "N/A"
                elocs = art.get("ELocationID", [])
                if isinstance(elocs, list):
                    for e in elocs:
                        if hasattr(e, "attributes") and e.attributes.get("EIdType") == "doi":
                            doi = str(e) or doi
                            break
                
                url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                spinal = 1 if contains_spinal(title, abstract) else 0

                # Log what we found for debugging
                logger.debug(f"Article {i+1}: PMID={pmid}, Title='{title[:50]}...', Abstract_length={len(abstract)}")

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
            except Exception as e:
                logger.error(f"Error parsing article {i+1}: {e}")
                continue
        
        logger.info(f"Successfully parsed {len(results)} PubMed articles")
        return results
        
    except Exception as e:
        logger.exception(f"PubMed fetch error: {e}")
        return []

# -------------------------
# ClinicalTrials fetcher (Improved error handling)
# -------------------------
def fetch_clinical_trials(
    intervention: str,
    condition: str = "",
    days_back: int = DAYS_BACK,
    max_records: int = MAX_RECORDS
) -> List[Dict[str, Any]]:
    logger.info(f"ClinicalTrials.gov search: intervention='{intervention}'{', condition=' + repr(condition) if condition else ''}, days_back={days_back}")
    
    base_url = "https://clinicaltrials.gov/api/v2/studies"
    search_results = []
    page_token = None
    
    date_cutoff = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
    
    # Build parameters - only include condition if it's not empty
    params = {
        'query.intr': intervention,
        'filter.lastUpdatePostDate': f'{date_cutoff}..',
        'pageSize': min(100, max_records),
        'format': 'json',
    }
    
    # Only add condition filter if condition is provided and not empty
    if condition and condition.strip():
        params['query.cond'] = condition

    try:
        while len(search_results) < max_records:
            if page_token:
                params['pageToken'] = page_token
                
            logger.debug(f"Making request to ClinicalTrials.gov with params: {params}")
            response = requests.get(base_url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            studies = data.get('studies', [])
            if not studies:
                logger.info("No more clinical trials found")
                break
            
            logger.info(f"Retrieved {len(studies)} clinical trials from API")
                
            for study in studies:
                try:
                    protocol_section = study.get('protocolSection', {})
                    identification = protocol_section.get('identificationModule', {})
                    nct_id = identification.get('nctId', '')
                    title = identification.get('briefTitle', '')
                    
                    description = protocol_section.get('descriptionModule', {})
                    summary = description.get('briefSummary', '')
                    
                    status_module = protocol_section.get('statusModule', {})
                    status = status_module.get('overallStatus', '')
                    start_date = status_module.get('startDateStruct', {}).get('date', '')
                    completion_date = status_module.get('completionDateStruct', {}).get('date', '')
                    
                    design = protocol_section.get('designModule', {})
                    study_type = design.get('studyType', '')
                    enrollment = design.get('enrollmentInfo', {}).get('count', '')
                    
                    conditions_module = protocol_section.get('conditionsModule', {})
                    conditions_list = [c.get('name', '') for c in conditions_module.get('conditions', [])]
                    
                    interventions_module = protocol_section.get('armsInterventionsModule', {})
                    interventions_list = [i.get('name', '') for i in interventions_module.get('interventions', [])]
                    
                    phases = design.get('phases', [])
                    phases_list = [p for p in phases if p]
                    
                    sponsor_module = protocol_section.get('sponsorCollaboratorsModule', {})
                    sponsor_name = sponsor_module.get('leadSponsor', {}).get('name', '')
                    
                    eligibility = protocol_section.get('eligibilityModule', {})
                    age_min = eligibility.get('minimumAge', '')
                    age_max = eligibility.get('maximumAge', '')
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
                        "enrollment": str(enrollment),
                        "age_range": age_range,
                        "url": url_study,
                        "spinal_hit": spinal_hit,
                        "semantic_score": None
                    })
                except Exception as e:
                    logger.error(f"Error parsing clinical trial: {e}")
                    continue
            
            page_token = data.get('nextPageToken')
            if not page_token or len(search_results) >= max_records:
                break
            
            time.sleep(RATE_LIMIT_DELAY)

    except requests.RequestException as e:
        logger.error(f"Error fetching data from ClinicalTrials.gov API: {e}")
            
    logger.info(f"Found and parsed {len(search_results)} clinical trials.")
    return search_results[:max_records]

# -------------------------
# DB upsert helpers
# -------------------------
def upsert_pubmed(db: str, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not articles:
        return []
        
    conn = sqlite3.connect(db)
    cur = conn.cursor()
    new_items = []
    
    for a in articles:
        try:
            cur.execute("""
                INSERT INTO pubmed_articles
                (pmid, title, abstract, authors, publication_date, journal, doi, url, spinal_hit, first_seen, semantic_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                a["pmid"], a["title"], a["abstract"], a["authors"], 
                a["publication_date"], a["journal"], a["doi"], a["url"], 
                a["spinal_hit"], now_ts(), a.get("semantic_score")
            ))
            new_items.append(a)
        except sqlite3.IntegrityError:
            # Article already exists
            continue
        except Exception as e:
            logger.error(f"Error inserting article {a.get('pmid', 'unknown')}: {e}")
            continue
            
    conn.commit()
    conn.close()
    logger.info(f"DB upsert_pubmed: inserted {len(new_items)} new articles")
    return new_items

def upsert_trials(db: str, trials: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not trials:
        return []
        
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
                t["nct_id"], t["title"], t["detailed_description"], 
                "; ".join(t.get("conditions", [])),
                "; ".join(t.get("interventions", [])), 
                "; ".join(t.get("phases", [])),
                t.get("study_type", ""), t.get("status", ""), 
                t.get("start_date", ""), t.get("completion_date", ""),
                t.get("sponsor", ""), str(t.get("enrollment", "")), 
                t.get("age_range", ""), t.get("url", ""), 
                t.get("spinal_hit", 0), now_ts(), t.get("semantic_score")
            ))
            new_items.append(t)
        except sqlite3.IntegrityError:
            # Trial already exists
            continue
        except Exception as e:
            logger.error(f"Error inserting trial {t.get('nct_id', 'unknown')}: {e}")
            continue
            
    conn.commit()
    conn.close()
    logger.info(f"DB upsert_trials: inserted {len(new_items)} new trials")
    return new_items

# -------------------------
# CSV helpers
# -------------------------
def append_pubmed_csv(rows: List[Dict[str, Any]], path: str = PUBMED_WEEKLY_CSV):
    if not rows: 
        return
        
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
    if not rows:
        return
        
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
    logger.info("Exported full database to CSV files")

# -------------------------
# Email function
# -------------------------
def send_email(new_pubmed: List[Dict[str,Any]], new_trials: List[Dict[str,Any]], stats: Dict[str,int], pubmed_term: str, trials_intervention: str, trials_condition: str) -> bool:
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
    html += f"<p><b>ClinicalTrials search terms:</b> {trials_intervention} (Intervention){', ' + trials_condition + ' (Condition)' if trials_condition else ''}</p>"
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
    logger.info("=== Starting Weekly Update ===")
    init_db()

    # Step 1: Fetch data using improved searches
    logger.info("Step 1: Fetching PubMed articles...")
    pubmed_articles = fetch_pubmed(PUBMED_TERM, MAX_RECORDS * 2, DAYS_BACK)
    
    logger.info("Step 2: Fetching Clinical Trials...")
    trials = fetch_clinical_trials(CLINICALTRIALS_INTERVENTION, CLINICALTRIALS_CONDITION, DAYS_BACK, MAX_RECORDS * 2)

    # Step 2: Apply semantic filtering
    logger.info("Step 3: Applying semantic filtering to PubMed articles...")
    relevant_pubmed = semantic_filter(pubmed_articles, SEMANTIC_SEARCH_TERMS, SEMANTIC_THRESHOLD)
    
    logger.info("Step 4: Applying semantic filtering to Clinical Trials...")
    relevant_trials = semantic_filter(trials, SEMANTIC_SEARCH_TERMS, SEMANTIC_THRESHOLD)

    # Step 3: Take the top N results and sort by semantic score
    logger.info("Step 5: Sorting and limiting results...")
    final_pubmed = sorted(relevant_pubmed, key=lambda x: x.get('semantic_score', 0), reverse=True)[:MAX_RECORDS]
    final_trials = sorted(relevant_trials, key=lambda x: x.get('semantic_score', 0), reverse=True)[:MAX_RECORDS]
    
    logger.info(f"Final selection: {len(final_pubmed)} PubMed articles, {len(final_trials)} clinical trials")

    # Step 4: Upsert into DB and get new items
    logger.info("Step 6: Updating database...")
    new_pubmed = upsert_pubmed(DB_FILE, final_pubmed)
    new_trials = upsert_trials(DB_FILE, final_trials)

    # Step 5: Append weekly CSVs
    logger.info("Step 7: Updating CSV files...")
    append_pubmed_csv(new_pubmed)
    append_trials_csv(new_trials)

    # Step 6: Export full CSVs
    logger.info("Step 8: Exporting full database to CSV...")
    export_full_csvs()

    # Step 7: Send summary email
    logger.info("Step 9: Sending email report...")
    stats = {"new_pubmed": len(new_pubmed), "new_trials": len(new_trials)}
    email_sent = send_email(new_pubmed, new_trials, stats, PUBMED_TERM, CLINICALTRIALS_INTERVENTION, CLINICALTRIALS_CONDITION)
    
    # Summary
    logger.info("=== Weekly Update Complete ===")
    logger.info(f"Summary:")
    logger.info(f"  - PubMed articles fetched: {len(pubmed_articles)}")
    logger.info(f"  - PubMed articles after semantic filtering: {len(relevant_pubmed)}")
    logger.info(f"  - New PubMed articles added to database: {len(new_pubmed)}")
    logger.info(f"  - Clinical trials fetched: {len(trials)}")
    logger.info(f"  - Clinical trials after semantic filtering: {len(relevant_trials)}")
    logger.info(f"  - New clinical trials added to database: {len(new_trials)}")
    logger.info(f"  - Email sent: {'Yes' if email_sent else 'No'}")
    
    return {
        "pubmed_fetched": len(pubmed_articles),
        "pubmed_filtered": len(relevant_pubmed),
        "pubmed_new": len(new_pubmed),
        "trials_fetched": len(trials),
        "trials_filtered": len(relevant_trials),
        "trials_new": len(new_trials),
        "email_sent": email_sent
    }

# -------------------------
# Entry point for manual execution
# -------------------------
if __name__ == "__main__":
    try:
        results = weekly_update()
        logger.info("Script completed successfully")
        print("Weekly update completed!")
        print(f"Results: {results}")
        exit(0)  # Successful exit
    except Exception as e:
        logger.exception("Script failed with error")
        print(f"Script failed: {e}")
        exit(1)  # Exit with error code

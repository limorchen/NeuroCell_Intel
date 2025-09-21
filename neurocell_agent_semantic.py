#!/usr/bin/env python3
"""
neurocell_agent_semantic_fixed.py

Requirements (pip):
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

# Semantic search libraries
from sentence_transformers import SentenceTransformer, util
import torch

# Load .env_semantic file for manual local runs (optional)
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

CLINICALTRIALS_INTERVENTION = os.getenv("CLINICALTRIALS_INTERVENTION", "exosomes")
CLINICALTRIALS_CONDITION = ""  # ignore condition for filtering

# New semantic search terms (comma-separated in env)
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
        logger.info("No documents to semantic-filter.")
        return []

    if not terms:
        logger.info("No semantic terms provided — skipping semantic filtering.")
        for d in docs:
            d['semantic_score'] = 0.0
        return docs

    logger.info(f"Performing semantic filtering on {len(docs)} documents with threshold={threshold} ...")

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
            logger.debug(f"Skipping doc '{title[:50]}' because no text to embed")
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

        logger.debug(f"Doc '{title[:50]}' semantic_score={doc['semantic_score']}")

    logger.info(f"Semantic filtering kept {len(filtered_docs)} / {len(docs)} documents.")
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
    logger.info(f"Initialized DB at {path}")

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
            art = med.get("Article", {}) or {}
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
# ClinicalTrials fetcher
# -------------------------
def fetch_clinical_trials(intervention: str, condition: str = "", days_back: int = DAYS_BACK, max_records: int = MAX_RECORDS) -> List[Dict[str, Any]]:
    logger.info(f"ClinicalTrials.gov search: intervention='{intervention}'")
    base_url = "https://clinicaltrials.gov/api/v2/studies"
    search_results = []
    page_token = None
    date_cutoff = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
    params = {
        'query.intr': intervention,
        'format': 'json',
        'pageSize': 100,
    }
    if condition:
        params['query.cond'] = condition

    try:
        while len(search_results) < max_records * 2:
            if page_token:
                params['pageToken'] = page_token
            response = requests.get(base_url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()

            studies = data.get('studies', [])
            if not studies:
                logger.info("v2 API returned no studies (or structure changed). Breaking to fallback.")
                break

            for study in studies:
                protocol_section = study.get('protocolSection', {}) or {}
                nct_id = protocol_section.get('identificationModule', {}).get('nctId')
                title = protocol_section.get('identificationModule', {}).get('briefTitle')
                summary = protocol_section.get('descriptionModule', {}).get('briefSummary')
                status = protocol_section.get('statusModule', {}).get('overallStatus')
                study_type = protocol_section.get('designModule', {}).get('studyType')
                start_date = protocol_section.get('statusModule', {}).get('startDateStruct', {}).get('date')
                completion_date = protocol_section.get('statusModule', {}).get('completionDateStruct', {}).get('date')

                conditions_list = [c.get('name') for c in protocol_section.get('conditionsModule', {}).get('conditions', [])] or []
                interventions_list = [i.get('name') for i in protocol_section.get('armsInterventionsModule', {}).get('interventions', [])] or []
                phases_list = [p.get('phase') for p in protocol_section.get('designModule', {}).get('phases', [])] or []
                sponsor_name = protocol_section.get('sponsorCollaboratorsModule', {}).get('leadSponsor', {}).get('name')
                enrollment = protocol_section.get('designModule', {}).get('enrollmentInfo', {}).get('count')
                age_min = protocol_section.get('eligibilityModule', {}).get('minimumAge')
                age_max = protocol_section.get('eligibilityModule', {}).get('maximumAge')
                age_range = f"{age_min} - {age_max}" if age_min or age_max else "N/A"

                url_study = f"https://clinicaltrials.gov/study/{nct_id}"
                spinal = 1 if contains_spinal(title, summary) else 0

                search_results.append({
                    "nct_id": nct_id,
                    "title": title,
                    "detailed_description": summary,
                    "conditions": ", ".join(conditions_list),
                    "interventions": ", ".join(interventions_list),
                    "phases": ", ".join(phases_list),
                    "study_type": study_type,
                    "status": status,
                    "start_date": start_date,
                    "completion_date": completion_date,
                    "sponsor": sponsor_name,
                    "enrollment": enrollment,
                    "age_range": age_range,
                    "url": url_study,
                    "spinal_hit": spinal,
                    "semantic_score": None
                })

                if len(search_results) >= max_records:
                    break

            page_token = data.get("nextPageToken")
            if not page_token:
                break
        logger.info(f"ClinicalTrials.gov fetched {len(search_results)} trials")
        return search_results
    except Exception as e:
        logger.exception("ClinicalTrials.gov fetch error")
        return []

# -------------------------
# DB upsert helpers
# -------------------------
def upsert_pubmed_articles(articles: List[Dict[str, Any]]):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    for art in articles:
        c.execute("""
        INSERT INTO pubmed_articles
        (pmid, title, abstract, authors, publication_date, journal, doi, url, spinal_hit, first_seen, semantic_score)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(pmid) DO UPDATE SET
        semantic_score=excluded.semantic_score
        """, (
            art.get("pmid"),
            art.get("title"),
            art.get("abstract"),
            art.get("authors"),
            art.get("publication_date"),
            art.get("journal"),
            art.get("doi"),
            art.get("url"),
            art.get("spinal_hit"),
            now_ts(),
            art.get("semantic_score")
        ))
    conn.commit()
    conn.close()
    logger.info(f"Upserted {len(articles)} PubMed articles into DB.")

def upsert_clinical_trials(trials: List[Dict[str, Any]]):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    for t in trials:
        c.execute("""
        INSERT INTO clinical_trials
        (nct_id, title, detailed_description, conditions, interventions, phases, study_type,
         status, start_date, completion_date, sponsor, enrollment, age_range, url,
         spinal_hit, first_seen, semantic_score)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(nct_id) DO UPDATE SET
        semantic_score=excluded.semantic_score
        """, (
            t.get("nct_id"),
            t.get("title"),
            t.get("detailed_description"),
            t.get("conditions"),
            t.get("interventions"),
            t.get("phases"),
            t.get("study_type"),
            t.get("status"),
            t.get("start_date"),
            t.get("completion_date"),
            t.get("sponsor"),
            t.get("enrollment"),
            t.get("age_range"),
            t.get("url"),
            t.get("spinal_hit"),
            now_ts(),
            t.get("semantic_score")
        ))
    conn.commit()
    conn.close()
    logger.info(f"Upserted {len(trials)} clinical trials into DB.")

# -------------------------
# CSV export
# -------------------------
def save_csv(filename: str, records: List[Dict[str, Any]]):
    if not records:
        logger.info(f"No records to save to {filename}")
        return
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    logger.info(f"Saved {len(records)} records to {filename}")

# -------------------------
# Main
# -------------------------
def main():
    init_db()

    # 1. Fetch and semantic-filter PubMed
    new_pubmed = fetch_pubmed(PUBMED_TERM)
    new_pubmed = semantic_filter(new_pubmed, SEMANTIC_SEARCH_TERMS, SEMANTIC_THRESHOLD)
    upsert_pubmed_articles(new_pubmed)
    save_csv(PUBMED_WEEKLY_CSV, new_pubmed)

    # 2. Fetch and semantic-filter ClinicalTrials (intervention only)
    new_trials = fetch_clinical_trials(CLINICALTRIALS_INTERVENTION)
    new_trials = semantic_filter(new_trials, SEMANTIC_SEARCH_TERMS, SEMANTIC_THRESHOLD)
    upsert_clinical_trials(new_trials)
    save_csv(TRIALS_WEEKLY_CSV, new_trials)

    logger.info(f"New PubMed articles: {len(new_pubmed)}")
    logger.info(f"New Clinical Trials: {len(new_trials)}")

if __name__ == "__main__":
    main()

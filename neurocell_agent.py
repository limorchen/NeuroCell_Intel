#!/usr/bin/env python3
"""
neurocell_agent.py

Requirements (pip):
- biopython
- requests
- python-dotenv (if using .env)
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

# Load .env if present
load_dotenv()

# -------------------------
# Configuration (env / defaults)
# -------------------------
DB_FILE = os.getenv("DB_FILE", "neurocell_database.db")

SENDER_EMAIL = os.getenv("SENDER_EMAIL")
RECIPIENT_EMAIL = os.getenv("RECIPIENT_EMAIL")  # comma-separated allowed
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 465))

NCBI_EMAIL = os.getenv("NCBI_EMAIL", "chen.limor@gmail.com")
PUBMED_TERM = os.getenv("PUBMED_TERM", "exosomes AND CNS")
CLINICALTRIALS_TERM = os.getenv("CLINICALTRIALS_TERM",
                                '(AREA[InterventionName] "exosomes" OR AREA[InterventionName] "extracellular vesicles") AND (AREA[Condition] neurology OR AREA[Condition] "neurologic" OR AREA[Condition] "neurologic disorder")')
MAX_RECORDS = int(os.getenv("MAX_RECORDS", 20))
DAYS_BACK = int(os.getenv("DAYS_BACK", 7))
RATE_LIMIT_DELAY = float(os.getenv("RATE_LIMIT_DELAY", 0.34))

Entrez.email = NCBI_EMAIL

# CSV filenames
PUBMED_WEEKLY_CSV = "new_pubmed_this_week.csv"
TRIALS_WEEKLY_CSV = "new_trials_this_week.csv"
PUBMED_FULL_CSV = "all_pubmed_database.csv"
TRIALS_FULL_CSV = "all_trials_database.csv"

# -------------------------
# Logging
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("neurocell_agent.log"), logging.StreamHandler()]
)
logger = logging.getLogger("neurocell_agent")

# -------------------------
# Utils
# -------------------------
def now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def contains_spinal(*texts: List[str]) -> bool:
    """Case-insensitive check for the word 'spinal' in any provided text."""
    for t in texts:
        if not t:
            continue
        if "spinal" in t.lower():
            return True
    return False

# -------------------------
# Initialize DB (tables)
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
        first_seen TEXT
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
        first_seen TEXT
    );
    """)
    conn.commit()
    conn.close()

# -------------------------
# PubMed fetcher (Entrez)
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
                "spinal_hit": spinal
            })
        logger.info(f"PubMed parsed {len(results)} articles")
        return results
    except Exception as e:
        logger.exception("PubMed fetch error")
        return []

# -------------------------
# ClinicalTrials.gov fetcher (fielded expr -> study_fields)
# -------------------------
def fetch_clinical_trials_fielded(expr: str, max_records: int = MAX_RECORDS) -> List[Dict[str, Any]]:
    """
    Use ClinicalTrials.gov study_fields API with 'expr' supporting AREA[...] fielded queries.
    Example expr:
    (AREA[InterventionName] "exosomes" OR AREA[InterventionName] "extracellular vesicles")
    AND (AREA[Condition] neurology OR AREA[Condition] "neurologic" OR AREA[Condition] "neurologic disorder")
    """
    logger.info(f"ClinicalTrials.gov expr: {expr} | pageSize={max_records}")
    url = "https://clinicaltrials.gov/api/query/study_fields"
    params = {
        "expr": expr,
        "fields": "NCTId,BriefTitle,OverallStatus,Condition,InterventionName,StartDate,CompletionDate,BriefSummary,Phase",
        "min_rnk": 1,
        "max_rnk": max_records,
        "fmt": "json"
    }
    try:
        r = requests.get(url, params=params, timeout=30)
        if r.status_code != 200:
            logger.error(f"ClinicalTrials.gov API status {r.status_code}: {r.text[:200]}")
            return []
        data = r.json()
        studies = data.get("StudyFieldsResponse", {}).get("StudyFields", [])
        logger.info(f"ClinicalTrials.gov study_fields returned {len(studies)} studies")
        results = []
        for s in studies:
            nct = s.get("NCTId", [""])[0] if s.get("NCTId") else ""
            title = s.get("BriefTitle", [""])[0] if s.get("BriefTitle") else ""
            status = s.get("OverallStatus", [""])[0] if s.get("OverallStatus") else ""
            conditions = s.get("Condition", []) if s.get("Condition") else []
            interventions = s.get("InterventionName", []) if s.get("InterventionName") else []
            start_date = s.get("StartDate", [""])[0] if s.get("StartDate") else ""
            completion_date = s.get("CompletionDate", [""])[0] if s.get("CompletionDate") else ""
            brief_summary = s.get("BriefSummary", [""])[0] if s.get("BriefSummary") else ""
            phases = s.get("Phase", []) if s.get("Phase") else []
            url_study = f"https://clinicaltrials.gov/study/{nct}" if nct else ""
            spinal = 1 if contains_spinal(title, brief_summary) else 0

            results.append({
                "nct_id": nct,
                "title": title,
                "detailed_description": brief_summary,
                "conditions": conditions,
                "interventions": interventions,
                "phases": phases,
                "study_type": "",
                "status": status,
                "start_date": start_date,
                "completion_date": completion_date,
                "sponsor": "",
                "enrollment": "",
                "age_range": "",
                "url": url_study,
                "spinal_hit": spinal
            })
        return results
    except Exception as e:
        logger.exception("ClinicalTrials.gov fetch error")
        return []

# -------------------------
# DB upsert & CSV helpers
# -------------------------
def upsert_pubmed(db: str, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(db)
    cur = conn.cursor()
    new_items = []
    for a in articles:
        try:
            cur.execute("""
                INSERT INTO pubmed_articles
                (pmid, title, abstract, authors, publication_date, journal, doi, url, spinal_hit, first_seen)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (a["pmid"], a["title"], a["abstract"], a["authors"], a["publication_date"], a["journal"], a["doi"], a["url"], a["spinal_hit"], now_ts()))
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
                (nct_id, title, detailed_description, conditions, interventions, phases, study_type, status, start_date, completion_date, sponsor, enrollment, age_range, url, spinal_hit, first_seen)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                t["nct_id"], t["title"], t["detailed_description"], ",".join(t.get("conditions", [])),
                ",".join(t.get("interventions", [])), ",".join(t.get("phases", [])),
                t.get("study_type", ""), t.get("status", ""), t.get("start_date", ""), t.get("completion_date", ""),
                t.get("sponsor", ""), t.get("enrollment", ""), t.get("age_range", ""), t.get("url", ""), t.get("spinal_hit", 0), now_ts()
            ))
            new_items.append(t)
        except sqlite3.IntegrityError:
            continue
    conn.commit()
    conn.close()
    logger.info(f"DB upsert_trials: inserted {len(new_items)} new")
    return new_items

def append_pubmed_csv(rows: List[Dict[str, Any]], path: str = PUBMED_WEEKLY_CSV):
    if not rows:
        return
    file_exists = os.path.isfile(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        fieldnames = ["pmid", "title", "abstract", "authors", "publication_date", "journal", "doi", "url", "spinal_hit", "first_seen"]
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
                "first_seen": now_ts()
            })

def append_trials_csv(rows: List[Dict[str, Any]], path: str = TRIALS_WEEKLY_CSV):
    if not rows:
        return
    file_exists = os.path.isfile(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        fieldnames = ["nct_id", "title", "detailed_description", "conditions", "interventions", "phases", "study_type", "status", "start_date", "completion_date", "sponsor", "enrollment", "age_range", "url", "spinal_hit", "first_seen"]
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
                "study_type": r.get("study_type", ""),
                "status": r.get("status", ""),
                "start_date": r.get("start_date", ""),
                "completion_date": r.get("completion_date", ""),
                "sponsor": r.get("sponsor", ""),
                "enrollment": r.get("enrollment", ""),
                "age_range": r.get("age_range", ""),
                "url": r.get("url", ""),
                "spinal_hit": "YES" if r["spinal_hit"] else "NO",
                "first_seen": now_ts()
            })

# -------------------------
# Full DB exports
# -------------------------
def export_full_csvs(db: str = DB_FILE):
    conn = sqlite3.connect(db)
    cur = conn.cursor()
    # PubMed
    cur.execute("SELECT pmid, title, abstract, authors, publication_date, journal, doi, url, spinal_hit, first_seen FROM pubmed_articles")
    rows = cur.fetchall()
    with open(PUBMED_FULL_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["pmid", "title", "abstract", "authors", "publication_date", "journal", "doi", "url", "spinal_hit", "first_seen"])
        for r in rows:
            writer.writerow([r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], "YES" if r[8] else "NO", r[9]])
    # Trials
    cur.execute("SELECT nct_id, title, detailed_description, conditions, interventions, phases, study_type, status, start_date, completion_date, sponsor, enrollment, age_range, url, spinal_hit, first_seen FROM clinical_trials")
    rows = cur.fetchall()
    with open(TRIALS_FULL_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["nct_id", "title", "detailed_description", "conditions", "interventions", "phases", "study_type", "status", "start_date", "completion_date", "sponsor", "enrollment", "age_range", "url", "spinal_hit", "first_seen"])
        for r in rows:
            writer.writerow([r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9], r[10], r[11], r[12], r[13], "YES" if r[14] else "NO", r[15]])
    conn.close()
    logger.info("Exported full DB CSVs")

# -------------------------
# Email send (HTML + attachments)
# -------------------------
def send_email(new_pubmed: List[Dict[str,Any]], new_trials: List[Dict[str,Any]], stats: Dict[str,int], pubmed_term: str, trials_term: str) -> bool:
    if not (SENDER_EMAIL and RECIPIENT_EMAIL and EMAIL_PASSWORD):
        logger.error("Email credentials missing - skipping email send")
        return False

    recipients = [r.strip() for r in RECIPIENT_EMAIL.split(",")]

    msg = MIMEMultipart("mixed")
    msg["From"] = SENDER_EMAIL
    msg["To"] = ", ".join(recipients)
    msg["Subject"] = f"NeuroCell Intelligence Report - {datetime.now().strftime('%Y-%m-%d')}"

    # HTML body
    html = f"<h2>NeuroCell Intelligence Report</h2>"
    html += f"<p><b>PubMed search term:</b> {pubmed_term}<br><b>ClinicalTrials search expr:</b> {trials_term}</p>"
    html += "<h3>Summary</h3><ul>"
    html += f"<li>Total PubMed in DB: {stats.get('pubmed_total',0)}</li>"
    html += f"<li>New PubMed this run: {stats.get('pubmed_new',0)}</li>"
    html += f"<li>Total ClinicalTrials in DB: {stats.get('trials_total',0)}</li>"
    html += f"<li>New ClinicalTrials this run: {stats.get('trials_new',0)}</li>"
    html += "</ul>"

    if new_pubmed:
        html += "<h3>New PubMed articles</h3><ul>"
        for a in new_pubmed:
            marker = "<b>[SPINAL HIT]</b> " if a.get("spinal_hit") else ""
            html += f"<li>{marker}<a href='{a['url']}'>{a['title']}</a> ({a.get('publication_date','')})</li>"
        html += "</ul>"
    else:
        html += "<p>No new PubMed articles this run.</p>"

    if new_trials:
        html += "<h3>New ClinicalTrials.gov entries</h3><ul>"
        for t in new_trials:
            marker = "<b>[SPINAL HIT]</b> " if t.get("spinal_hit") else ""
            html += f"<li>{marker}<a href='{t['url']}'>{t['title']}</a> ({t.get('start_date','')})</li>"
        html += "</ul>"
    else:
        html += "<p>No new ClinicalTrials.gov entries this run.</p>"

    part = MIMEText(html, "html")
    msg.attach(part)

    # Attach CSVs (weekly + full)
    attachments = []
    for fname in [PUBMED_WEEKLY_CSV, TRIALS_WEEKLY_CSV, PUBMED_FULL_CSV, TRIALS_FULL_CSV]:
        if os.path.exists(fname):
            attachments.append(fname)
            with open(fname, "rb") as f:
                p = MIMEBase("application", "octet-stream")
                p.set_payload(f.read())
            encoders.encode_base64(p)
            p.add_header("Content-Disposition", f"attachment; filename={os.path.basename(fname)}")
            msg.attach(p)

    try:
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
            server.login(SENDER_EMAIL, EMAIL_PASSWORD)
            server.sendmail(SENDER_EMAIL, recipients, msg.as_string())
        logger.info(f"Sent email to {recipients} with attachments: {attachments}")
        return True
    except Exception as e:
        logger.exception("Failed to send email")
        return False

# -------------------------
# Stats helper
# -------------------------
def compute_stats(db: str = DB_FILE, new_pub_count:int=0, new_trials_count:int=0) -> Dict[str,int]:
    conn = sqlite3.connect(db)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM pubmed_articles")
    pub_total = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM clinical_trials")
    trials_total = cur.fetchone()[0]
    conn.close()
    return {"pubmed_total": pub_total, "trials_total": trials_total, "pubmed_new": new_pub_count, "trials_new": new_trials_count}

# -------------------------
# Main orchestrator
# -------------------------
def run_weekly():
    logger.info("==== NeuroCell weekly run start ====")
    init_db()

    # 1) Fetch PubMed
    pubmed_results = fetch_pubmed(PUBMED_TERM, max_records=MAX_RECORDS, days_back=DAYS_BACK)
    # upsert to DB and get new ones
    new_pubmed = upsert_pubmed(DB_FILE, pubmed_results)
    append_pubmed_csv(new_pubmed, PUBMED_WEEKLY_CSV)

    # 2) Fetch ClinicalTrials.gov with fielded expr
    # expr already in CLINICALTRIALS_TERM env var; ensure it's URL-safe via requests
    trials_results = fetch_clinical_trials_fielded(CLINICALTRIALS_TERM, max_records=MAX_RECORDS)
    new_trials = upsert_trials(DB_FILE, trials_results)
    append_trials_csv(new_trials, TRIALS_WEEKLY_CSV)

    # 3) Export full DB snapshots
    export_full_csvs(DB_FILE)

    # 4) Stats & email
    stats = compute_stats(DB_FILE, new_pub_count=len(new_pubmed), new_trials_count=len(new_trials))
    ok = send_email(new_pubmed, new_trials, stats, PUBMED_TERM, CLINICALTRIALS_TERM)
    if ok:
        logger.info("Run finished and email sent successfully.")
    else:
        logger.error("Run finished but email failed.")

    logger.info("==== NeuroCell weekly run complete ====")
    return ok

# -------------------------
# Run if executed directly
# -------------------------
if __name__ == "__main__":
    run_weekly()

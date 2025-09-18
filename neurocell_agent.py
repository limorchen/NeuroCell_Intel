#!/usr/bin/env python3
"""
neurocell_agent.py

Fetches PubMed articles and ClinicalTrials.gov studies using separate search terms,
stores everything in a persistent SQLite DB, exports new items into CSVs, and
sends a weekly email with stats and CSV attachments. Highlights items containing
the word "spinal" in title/abstract.

Environment variables (recommended):
- SENDER_EMAIL
- RECIPIENT_EMAIL
- EMAIL_PASSWORD
- SMTP_SERVER (default: smtp.gmail.com)
- SMTP_PORT (default: 465)
- NCBI_EMAIL
- PUBMED_TERM  (default: "exosomes AND CNS")
- CLINICALTRIALS_TERM (default: "exosomes extracellular vesicles AND neurology")
- MAX_RECORDS (default: 20)
- DAYS_BACK (default: 7)
"""

import os
import time
import sqlite3
import smtplib
import csv
import logging
import requests
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from typing import List, Dict, Any
from Bio import Entrez

# -------------------------
# Configuration & defaults
# -------------------------
DB_FILE = os.getenv("DB_FILE", "neurocell_database.db")

SENDER_EMAIL = os.getenv("SENDER_EMAIL")
RECIPIENT_EMAIL = os.getenv("RECIPIENT_EMAIL")  # comma-separated allowed
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 465))

NCBI_EMAIL = os.getenv("NCBI_EMAIL", "chen.limor@gmail.com")
PUBMED_TERM = os.getenv("PUBMED_TERM", "exosomes AND CNS")
CLINICALTRIALS_TERM = os.getenv("CLINICALTRIALS_TERM", "exosomes extracellular vesicles AND neurology")
MAX_RECORDS = int(os.getenv("MAX_RECORDS", 20))
DAYS_BACK = int(os.getenv("DAYS_BACK", 7))
RATE_LIMIT_DELAY = float(os.getenv("RATE_LIMIT_DELAY", 0.34))  # Entrez rate limiting

Entrez.email = NCBI_EMAIL

PUBMED_CSV = "new_pubmed_this_week.csv"
TRIALS_CSV = "new_trials_this_week.csv"
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
def current_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def contains_spinal(texts: List[str]) -> bool:
    """Return True if any of the provided texts contains the word 'spinal' (case-insensitive)."""
    for t in texts:
        if not t:
            continue
        if "spinal" in t.lower():
            return True
    return False

# -------------------------
# Database management
# -------------------------
def init_db(db_path: str = DB_FILE):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    # PubMed table
    cur.execute("""
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
    # Clinical trials table
    cur.execute("""
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
# PubMed fetcher
# -------------------------
def fetch_pubmed(term: str, max_records: int = MAX_RECORDS, days_back: int = DAYS_BACK) -> List[Dict[str, Any]]:
    """Fetch recent PubMed articles using Entrez and return parsed dicts."""
    logger.info(f"PubMed search term: {term} | days_back={days_back} | retmax={max_records}")
    try:
        date_filter = (datetime.now() - timedelta(days=days_back)).strftime("%Y/%m/%d")
        # Use Entrez.esearch with reldate to get recent items
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
        logger.info(f"PubMed search returned {len(ids)} ids")

        if not ids:
            return []

        time.sleep(RATE_LIMIT_DELAY)
        # Fetch full records (xml)
        handle = Entrez.efetch(db="pubmed", id=",".join(ids), rettype="abstract", retmode="xml")
        papers = Entrez.read(handle)
        handle.close()

        results = []
        for article in papers.get("PubmedArticle", []):
            med = article.get("MedlineCitation", {})
            pmid = str(med.get("PMID"))
            art = med.get("Article", {})
            title = art.get("ArticleTitle", "")
            abstract_list = art.get("Abstract", {}).get("AbstractText", [])
            if isinstance(abstract_list, list):
                abstract = " ".join([str(a) for a in abstract_list])
            else:
                abstract = str(abstract_list) if abstract_list else ""
            # authors
            authors = []
            for a in art.get("AuthorList", [])[:20]:
                if "LastName" in a and "Initials" in a:
                    authors.append(f"{a.get('LastName','')} {a.get('Initials','')}")
                elif "CollectiveName" in a:
                    authors.append(a.get("CollectiveName"))
            authors_str = ", ".join(authors)
            journal_info = art.get("Journal", {})
            journal = journal_info.get("Title", "")
            # pub date
            pubdate = ""
            ji = journal_info.get("JournalIssue", {})
            if ji:
                pd_struct = ji.get("PubDate", {})
                if isinstance(pd_struct, dict):
                    pubdate = pd_struct.get("Year", "") or pd_struct.get("MedlineDate", "")
            # doi extraction (ELocationID)
            doi = "N/A"
            elocs = art.get("ELocationID", [])
            if isinstance(elocs, list):
                for e in elocs:
                    if hasattr(e, "attributes") and e.attributes.get("EIdType") == "doi":
                        doi = e.attributes.get("text", "") or doi

            url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            spinal = 1 if contains_spinal([title, abstract]) else 0

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
        logger.exception(f"Error fetching PubMed: {e}")
        return []

# -------------------------
# ClinicalTrials.gov fetcher
# -------------------------
def fetch_clinical_trials(term: str, max_records: int = MAX_RECORDS, days_back: int = DAYS_BACK) -> List[Dict[str, Any]]:
    """
    Fetch studies from ClinicalTrials.gov v2 API.
    Use query.term for the search; return parsed list of dicts.
    """
    logger.info(f"ClinicalTrials.gov search term: {term} | pageSize={max_records}")
    try:
        base_url = "https://clinicaltrials.gov/api/v2/studies"
        params = {
            "query.term": term,
            "pageSize": max_records,
            "sort": "StudyFirstPostDate desc"
        }
        resp = requests.get(base_url, params=params, timeout=30)
        if resp.status_code != 200:
            logger.error(f"ClinicalTrials.gov API returned status {resp.status_code}: {resp.text}")
            return []

        data = resp.json()
        studies = data.get("studies", [])
        logger.info(f"ClinicalTrials.gov returned {len(studies)} studies")

        results = []
        for s in studies:
            proto = s.get("protocolSection", {})
            identification = proto.get("identificationModule", {})
            description_module = proto.get("descriptionModule", {})
            design_module = proto.get("designModule", {})
            status_module = proto.get("statusModule", {})
            sponsor_module = proto.get("sponsorCollaboratorsModule", {})
            arms = proto.get("armsInterventionsModule", {})
            conditions = proto.get("conditionsModule", {})
            eligibility = proto.get("eligibilityModule", {})

            nct = identification.get("nctId", "") or identification.get("nctId", "")
            title = identification.get("briefTitle", "") or identification.get("officialTitle", "")
            detailed_description = description_module.get("detailedDescription", "") or ""
            interventions = [inv.get("name", "") for inv in arms.get("interventions", [])] if arms else []
            conditions_list = conditions.get("conditions", []) if conditions else []
            phases = design_module.get("phases", []) if design_module else []
            study_type = design_module.get("studyType", "N/A")
            overall_status = status_module.get("overallStatus", "N/A")
            start_date = status_module.get("startDateStruct", {}).get("date", "")
            completion_date = status_module.get("completionDateStruct", {}).get("date", "")
            sponsor = sponsor_module.get("leadSponsor", {}).get("name", "N/A")
            enrollment = design_module.get("enrollmentInfo", {}).get("count", "N/A") if design_module else "N/A"
            age_range = f"{eligibility.get('minimumAge','N/A')} - {eligibility.get('maximumAge','N/A')}" if eligibility else "N/A"
            url = f"https://clinicaltrials.gov/study/{nct}" if nct else ""
            # spinal detection across title + description
            spinal = 1 if contains_spinal([title, detailed_description]) else 0

            results.append({
                "nct_id": nct,
                "title": title,
                "detailed_description": detailed_description,
                "conditions": conditions_list,
                "interventions": interventions,
                "phases": phases,
                "study_type": study_type,
                "status": overall_status,
                "start_date": start_date,
                "completion_date": completion_date,
                "sponsor": sponsor,
                "enrollment": enrollment,
                "age_range": age_range,
                "url": url,
                "spinal_hit": spinal
            })
        return results
    except Exception as e:
        logger.exception(f"Error fetching ClinicalTrials.gov: {e}")
        return []

# -------------------------
# DB upsert + CSV helpers
# -------------------------
def upsert_pubmed_articles(db_path: str, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Insert new pubmed articles to DB; return list of newly-inserted articles."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    new = []
    for a in articles:
        try:
            cur.execute("""
                INSERT INTO pubmed_articles (pmid, title, abstract, authors, publication_date, journal, doi, url, spinal_hit, first_seen)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                a["pmid"], a["title"], a["abstract"], a["authors"], a["publication_date"],
                a["journal"], a["doi"], a["url"], a["spinal_hit"], current_timestamp()
            ))
            new.append(a)
        except sqlite3.IntegrityError:
            # already present
            continue
    conn.commit()
    conn.close()
    logger.info(f"Inserted {len(new)} new PubMed articles into DB")
    return new

def upsert_clinical_trials(db_path: str, trials: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Insert new clinical trials to DB; return list of newly-inserted trials."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    new = []
    for t in trials:
        try:
            cur.execute("""
                INSERT INTO clinical_trials
                (nct_id, title, detailed_description, conditions, interventions, phases, study_type, status, start_date, completion_date, sponsor, enrollment, age_range, url, spinal_hit, first_seen)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                t["nct_id"], t["title"], t["detailed_description"], ",".join(t.get("conditions", [])),
                ",".join(t.get("interventions", [])), ",".join(t.get("phases", [])), t["study_type"], t["status"],
                t["start_date"], t["completion_date"], t["sponsor"], str(t["enrollment"]), t["age_range"], t["url"],
                t["spinal_hit"], current_timestamp()
            ))
            new.append(t)
        except sqlite3.IntegrityError:
            continue
    conn.commit()
    conn.close()
    logger.info(f"Inserted {len(new)} new ClinicalTrials into DB")
    return new

def append_to_csv_pubmed(rows: List[Dict[str, Any]], csv_path: str = PUBMED_CSV):
    if not rows:
        return
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "pmid", "title", "abstract", "authors", "publication_date", "journal", "doi", "url", "spinal_hit", "first_seen"
        ])
        if not file_exists:
            writer.writeheader()
        for r in rows:
            out = {
                "pmid": r["pmid"],
                "title": r["title"],
                "abstract": r["abstract"],
                "authors": r["authors"],
                "publication_date": r["publication_date"],
                "journal": r["journal"],
                "doi": r["doi"],
                "url": r["url"],
                "spinal_hit": "YES" if r["spinal_hit"] else "NO",
                "first_seen": current_timestamp()
            }
            writer.writerow(out)

def append_to_csv_trials(rows: List[Dict[str, Any]], csv_path: str = TRIALS_CSV):
    if not rows:
        return
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "nct_id", "title", "detailed_description", "conditions", "interventions", "phases", "study_type", "status",
            "start_date", "completion_date", "sponsor", "enrollment", "age_range", "url", "spinal_hit", "first_seen"
        ])
        if not file_exists:
            writer.writeheader()
        for r in rows:
            out = {
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
                "first_seen": current_timestamp()
            }
            writer.writerow(out)

# -------------------------
# CSV full exports (optional)
# -------------------------
def export_full_csvs(db_path: str = DB_FILE):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    # PubMed full export
    cur.execute("SELECT pmid, title, abstract, authors, publication_date, journal, doi, url, spinal_hit, first_seen FROM pubmed_articles")
    rows = cur.fetchall()
    with open(PUBMED_FULL_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["pmid", "title", "abstract", "authors", "publication_date", "journal", "doi", "url", "spinal_hit", "first_seen"])
        for r in rows:
            writer.writerow([r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], "YES" if r[8] else "NO", r[9]])
    # Trials full export
    cur.execute("SELECT nct_id, title, detailed_description, conditions, interventions, phases, study_type, status, start_date, completion_date, sponsor, enrollment, age_range, url, spinal_hit, first_seen FROM clinical_trials")
    rows = cur.fetchall()
    with open(TRIALS_FULL_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["nct_id", "title", "detailed_description", "conditions", "interventions", "phases", "study_type", "status", "start_date", "completion_date", "sponsor", "enrollment", "age_range", "url", "spinal_hit", "first_seen"])
        for r in rows:
            writer.writerow([r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9], r[10], r[11], r[12], r[13], "YES" if r[14] else "NO", r[15]])
    conn.close()
    logger.info("Exported full DB to CSVs")

# -------------------------
# Email sender (attach CSVs)
# -------------------------
def send_email_with_attachments(pubmed_new: List[Dict[str,Any]], trials_new: List[Dict[str,Any]], stats: Dict[str,int], pubmed_term: str, trials_term: str):
    if not SENDER_EMAIL or not RECIPIENT_EMAIL or not EMAIL_PASSWORD:
        logger.error("Missing email configuration. Not sending email.")
        return False

    recipients = [r.strip() for r in RECIPIENT_EMAIL.split(",")]

    msg = MIMEMultipart("mixed")
    msg["From"] = SENDER_EMAIL
    msg["To"] = ", ".join(recipients)
    msg["Subject"] = f"NeuroCell Intelligence Report - {datetime.now().strftime('%Y-%m-%d')}"

    # Body (HTML) includes search terms and highlights
    html = f"<h2>NeuroCell Intelligence Report</h2>"
    html += f"<p><b>PubMed search term:</b> {pubmed_term}<br>"
    html += f"<b>ClinicalTrials search term:</b> {trials_term}</p>"

    html += "<h3>Summary statistics</h3><ul>"
    html += f"<li>Total PubMed articles in DB: {stats.get('pubmed_total',0)}</li>"
    html += f"<li>New PubMed articles this run: {stats.get('pubmed_new',0)}</li>"
    html += f"<li>Total Clinical Trials in DB: {stats.get('trials_total',0)}</li>"
    html += f"<li>New Clinical Trials this run: {stats.get('trials_new',0)}</li>"
    html += "</ul>"

    if pubmed_new:
        html += "<h3>New PubMed articles (highlights marked)</h3><ul>"
        for a in pubmed_new:
            title = a["title"]
            marker = "<b>[SPINAL HIT]</b> " if a.get("spinal_hit") else ""
            html += f"<li>{marker}<a href='{a['url']}'>{title}</a> ({a.get('publication_date','')})</li>"
        html += "</ul>"
    else:
        html += "<p>No new PubMed articles in this run.</p>"

    if trials_new:
        html += "<h3>New Clinical Trials (highlights marked)</h3><ul>"
        for t in trials_new:
            title = t["title"]
            marker = "<b>[SPINAL HIT]</b> " if t.get("spinal_hit") else ""
            html += f"<li>{marker}<a href='{t['url']}'>{title}</a> ({t.get('start_date','')})</li>"
        html += "</ul>"
    else:
        html += "<p>No new ClinicalTrials.gov entries in this run.</p>"

    # Attach the HTML body
    part_body = MIMEText(html, "html")
    msg.attach(part_body)

    # Attach CSV files (only if they exist)
    attachments = []
    for file in [PUBMED_CSV, TRIALS_CSV, PUBMED_FULL_CSV, TRIALS_FULL_CSV]:
        if os.path.exists(file):
            attachments.append(file)
            with open(file, "rb") as f:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(file)}")
            msg.attach(part)

    # Send email
    try:
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
            server.login(SENDER_EMAIL, EMAIL_PASSWORD)
            server.sendmail(SENDER_EMAIL, recipients, msg.as_string())
        logger.info(f"Email sent to {recipients} with attachments: {attachments}")
        return True
    except Exception as e:
        logger.exception(f"Failed to send email: {e}")
        return False

# -------------------------
# Stats helper
# -------------------------
def compute_stats(db_path: str = DB_FILE) -> Dict[str,int]:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM pubmed_articles")
    pub_total = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM clinical_trials")
    trials_total = cur.fetchone()[0]
    conn.close()
    return {"pubmed_total": pub_total, "trials_total": trials_total}

# -------------------------
# Main orchestrator
# -------------------------
def run_weekly():
    logger.info("==== Starting NeuroCell weekly run ====")
    init_db()

    # Fetch sources
    pubmed_articles = fetch_pubmed(PUBMED_TERM, max_records=MAX_RECORDS, days_back=DAYS_BACK)
    trials = fetch_clinical_trials(CLINICALTRIALS_TERM, max_records=MAX_RECORDS, days_back=DAYS_BACK)

    # Upsert into DB and get newly inserted items
    new_pubmed = upsert_pubmed_articles(DB_FILE, pubmed_articles)
    new_trials = upsert_clinical_trials(DB_FILE, trials)

    # Append new items to weekly CSV logs
    append_to_csv_pubmed(new_pubmed, PUBMED_CSV)
    append_to_csv_trials(new_trials, TRIALS_CSV)

    # Export full DB snapshots (optional but attached)
    export_full_csvs(DB_FILE)

    stats = compute_stats(DB_FILE)
    stats.update({
        "pubmed_new": len(new_pubmed),
        "trials_new": len(new_trials)
    })

    # Send email with attachments and stats, show search terms
    ok = send_email_with_attachments(new_pubmed, new_trials, stats, PUBMED_TERM, CLINICALTRIALS_TERM)
    if ok:
        logger.info("Weekly run finished and email sent.")
    else:
        logger.error("Weekly run finished but email FAILED.")

    logger.info("==== Run complete ====")
    return ok

# -------------------------
# If run as script
# -------------------------
if __name__ == "__main__":
    run_weekly()

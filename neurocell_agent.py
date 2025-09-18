import os
import requests
import sqlite3
import smtplib
import time
import csv
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# ---------------------------
# Configuration
# ---------------------------
DB_FILE = "scientific_alerts.db"

SENDER_EMAIL = os.getenv("SENDER_EMAIL")
RECIPIENT_EMAIL = os.getenv("RECIPIENT_EMAIL")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 465))

PUBMED_TERM = os.getenv("PUBMED_TERM", "exosomes AND CNS")
CLINICALTRIALS_TERM = os.getenv("CLINICALTRIALS_TERM", "exosomes AND neurological")
NCBI_EMAIL = os.getenv("NCBI_EMAIL", "your_email@example.com")

MAX_RECORDS = int(os.getenv("MAX_RECORDS", 20))
DAYS_BACK = 7


# ---------------------------
# Database
# ---------------------------
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute(
        """CREATE TABLE IF NOT EXISTS pubmed_articles
           (pmid TEXT PRIMARY KEY, title TEXT, url TEXT, date TEXT)"""
    )
    c.execute(
        """CREATE TABLE IF NOT EXISTS clinical_trials
           (nct_id TEXT PRIMARY KEY, title TEXT, url TEXT, date TEXT)"""
    )
    conn.commit()
    conn.close()


# ---------------------------
# PubMed fetching
# ---------------------------
def fetch_pubmed(term, days_back=7, max_records=20):
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    today = datetime.today()
    start_date = (today - timedelta(days=days_back)).strftime("%Y/%m/%d")
    end_date = today.strftime("%Y/%m/%d")

    search_url = (
        f"{base_url}esearch.fcgi?db=pubmed&term={term}"
        f"&reldate={days_back}&datetype=pdat&retmax={max_records}&usehistory=y&email={NCBI_EMAIL}"
    )
    search_resp = requests.get(search_url)
    if search_resp.status_code != 200:
        return []

    from xml.etree import ElementTree as ET

    root = ET.fromstring(search_resp.content)
    id_list = [id_elem.text for id_elem in root.findall(".//Id")]

    articles = []
    for pmid in id_list:
        fetch_url = (
            f"{base_url}esummary.fcgi?db=pubmed&id={pmid}&retmode=json&email={NCBI_EMAIL}"
        )
        fetch_resp = requests.get(fetch_url)
        if fetch_resp.status_code == 200:
            data = fetch_resp.json()
            if "result" in data and pmid in data["result"]:
                item = data["result"][pmid]
                title = item.get("title", "")
                date = item.get("pubdate", "")
                url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                articles.append({"pmid": pmid, "title": title, "url": url, "date": date})
        time.sleep(0.34)  # rate limiting

    return articles


# ---------------------------
# ClinicalTrials.gov fetching
# ---------------------------
def fetch_clinical_trials(term, max_records=20):
    base_url = "https://clinicaltrials.gov/api/v2/studies"

    params = {
        "query.term": term,
        "pageSize": max_records,
        "sort": "StudyFirstPostDate desc",
    }

    resp = requests.get(base_url, params=params)
    if resp.status_code != 200:
        return []

    data = resp.json()
    trials = []
    for study in data.get("studies", []):
        nct_id = study.get("protocolSection", {}).get("identificationModule", {}).get("nctId")
        title = study.get("protocolSection", {}).get("identificationModule", {}).get("briefTitle")
        date = study.get("protocolSection", {}).get("statusModule", {}).get("studyFirstPostDateStruct", {}).get("date")
        url = f"https://clinicaltrials.gov/study/{nct_id}"
        if nct_id and title:
            trials.append({"nct_id": nct_id, "title": title, "url": url, "date": date})
    return trials


# ---------------------------
# DB Save and detection + CSV
# ---------------------------
def save_and_detect_new(entries, table, id_field, csv_file):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    new_entries = []
    for entry in entries:
        try:
            c.execute(
                f"INSERT INTO {table} ({id_field}, title, url, date) VALUES (?, ?, ?, ?)",
                (entry[id_field], entry["title"], entry["url"], entry["date"]),
            )
            new_entries.append(entry)
        except sqlite3.IntegrityError:
            pass  # already exists
    conn.commit()
    conn.close()

    if new_entries:
        write_to_csv(new_entries, csv_file, id_field)

    return new_entries


def write_to_csv(entries, csv_file, id_field):
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[id_field, "title", "url", "date"])
        if not file_exists:
            writer.writeheader()
        writer.writerows(entries)


# ---------------------------
# Email
# ---------------------------
def send_email(new_pubmed, new_trials, stats):
    msg = MIMEMultipart("alternative")
    msg["Subject"] = "Weekly Scientific Alerts"
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECIPIENT_EMAIL

    html = "<h2>Weekly Scientific Alerts</h2>"

    html += "<h3>Database Stats</h3><ul>"
    for k, v in stats.items():
        html += f"<li>{k}: {v}</li>"
    html += "</ul>"

    if new_pubmed:
        html += "<h3>New PubMed Articles</h3><ul>"
        for art in new_pubmed:
            html += f"<li><a href='{art['url']}'>{art['title']}</a> ({art['date']})</li>"
        html += "</ul>"

    if new_trials:
        html += "<h3>New Clinical Trials</h3><ul>"
        for trial in new_trials:
            html += f"<li><a href='{trial['url']}'>{trial['title']}</a> ({trial['date']})</li>"
        html += "</ul>"

    msg.attach(MIMEText(html, "html"))

    with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
        server.login(SENDER_EMAIL, EMAIL_PASSWORD)
        server.sendmail(SENDER_EMAIL, RECIPIENT_EMAIL, msg.as_string())


# ---------------------------
# Stats
# ---------------------------
def get_stats():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM pubmed_articles")
    pubmed_count = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM clinical_trials")
    trial_count = c.fetchone()[0]
    conn.close()
    return {"PubMed Articles": pubmed_count, "Clinical Trials": trial_count}


# ---------------------------
# Main
# ---------------------------
def main():
    init_db()

    pubmed_results = fetch_pubmed(PUBMED_TERM, days_back=DAYS_BACK, max_records=MAX_RECORDS)
    new_pubmed = save_and_detect_new(pubmed_results, "pubmed_articles", "pmid", "pubmed_articles.csv")

    trials_results = fetch_clinical_trials(CLINICALTRIALS_TERM, max_records=MAX_RECORDS)
    new_trials = save_and_detect_new(trials_results, "clinical_trials", "nct_id", "clinical_trials.csv")

    stats = get_stats()
    send_email(new_pubmed, new_trials, stats)


if __name__ == "__main__":
    main()

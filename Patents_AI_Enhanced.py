# ====================================================================
print("--- SCRIPT VERSION 1.1: DATE AND COLUMN FIXES DEPLOYED ---")
# ====================================================================

import os
import sys
import csv
import time
import smtplib
from datetime import datetime, timedelta
from pathlib import Path
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication

import pandas as pd
from lxml import etree
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from epo_ops import Client, models, middlewares
import epo_ops.exceptions as ops_exc

# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
CUMULATIVE_CSV = DATA_DIR / "patents_cumulative.csv"

# EPO API credentials
key = os.environ.get("EPO_OPS_KEY")
secret = os.environ.get("EPO_OPS_SECRET")

print("OPS key visible in Python?", bool(key))
print("OPS secret visible in Python?", bool(secret))

if not key or not secret:
    print("WARNING: Missing EPO OPS API credentials. Cannot run search.")
    # Allow script to load but OPS calls will be skipped
    client = None
else:
    # OPS client (Windows-safe: no Dogpile, Throttler only)
    middlewares_list = [
        middlewares.Throttler()  # Dogpile removed – it uses fcntl and breaks on Windows
    ]
    try:
        client = Client(
            key=key,
            secret=secret,
            middlewares=middlewares_list
        )
    except Exception as e:
        print(f"Error creating OPS client: {e}")
        client = None

# Research focus for relevance scoring
RESEARCH_FOCUS = """
Exosome-based drug delivery systems for central nervous system diseases and conditions,
including therapeutic applications for neurodegenerative conditions,
brain cancer, stroke, spinal cord injury and genetic brain disorders.
Focus on blood-brain barrier penetration and targeted CNS delivery.
"""

SEARCH_TERMS = '(ta=exosomes or ta="extracellular vesicles") and ta=CNS'

# Minimum relevance score (0-1 scale)
MIN_RELEVANCE_SCORE = 0.50  # Adjust this threshold as needed

# Initialize semantic search model (Used for Relevance Scoring)
try:
    print("Loading semantic search model...")
    semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
    research_focus_embedding = semantic_model.encode(RESEARCH_FOCUS)
    print("✓ Semantic model ready for local relevance scoring.")
except Exception as e:
    print(
        "FATAL ERROR: Could not load SentenceTransformer model. "
        "Please ensure 'sentence-transformers' is installed. "
        f"Error: {e}"
    )
    sys.exit(1)

# ---------------------------------------------------------------
# Helper Functions (Local AI/Summary is the key change here)
# ---------------------------------------------------------------

def generate_ai_summary(title, abstract, claims=""):
    """
    Generate concise AI summary using a simple local heuristic 
    (first 3 sentences of the abstract).
    """
    if not abstract:
        return "Insufficient data for summary"
    sentences = abstract.split(".")
    summary_sentences = sentences[:3]
    ai_summary = ".".join(summary_sentences).strip()
    if ai_summary and not ai_summary.endswith("."):
        ai_summary += "."
    if len(ai_summary) < 20 and title:
        return f"Summary: {title.strip()}"
    return ai_summary


def scan_patents_cql(cql_query, batch_size=25, max_records=None):
    """Iterate over OPS search results."""
    if not client:
        return
    start = 1
    total = None

    while True:
        end = start + batch_size - 1
        if max_records:
            end = min(end, max_records)
        if start > end:
            break

        try:
            resp = client.published_data_search(
                cql=cql_query,
                range_begin=start,
                range_end=end
            )
        except ops_exc.HTTPError as e:
            print(f"HTTP Error: {e.response.status_code}")
            print(f"Response text: {e.response.text[:500]}")
            break
        except Exception as e:
            print(f"An unexpected error occurred during search: {e}")
            break

        root = etree.fromstring(resp.content)
        ns = {
            "ops": "http://ops.epo.org",
            "ex": "http://www.epo.org/exchange"
        }

        if total is None:
            total_str = root.xpath(
                "string(//ops:biblio-search/@total-result-count)",
                namespaces=ns
            )
            if not total_str or total_str == "0":
                print("No results found.")
                break
            total = int(total_str)
            print(f"Total results: {total}")

        print(f"Fetching records {start}-{end}...")
        yield root

        if end >= total:
            break
        if max_records and end >= max_records:
            break

        start = end + 1
        time.sleep(0.3)


def get_biblio_data(country, number, kind):
    """Fetch full bibliographic data (use DOCDB, which is standard for biblio)."""
    if not client:
        return {}
    try:
        resp = client.published_data(
            reference_type="publication",
            input=models.Docdb(number, country, kind),
            endpoint="biblio"
        )
        root = etree.fromstring(resp.content)
        ns = {"ex": "http://www.epo.org/exchange"}

        # Title
        title = root.xpath(
            "string(//ex:invention-title[@lang='en'])",
            namespaces=ns
        )
        if not title:
            title = root.xpath("string(//ex:invention-title)", namespaces=ns)

        # Applicants
        applicants = root.xpath(
            "//ex:bibliographic-data/ex:parties/ex:applicants/"
            "ex:applicant/ex:applicant-name/ex:name/text()",
            namespaces=ns,
        )
        if not applicants:
            applicants = root.xpath(
                "//ex:bibliographic-data/ex:applicants/ex:applicant/"
                "ex:applicant-name/ex:name/text()",
                namespaces=ns,
            )
        if not applicants:
            applicants = root.xpath(
                "//*[local-name()='applicant-name']/*[local-name()='name']/text()"
            )
        applicants = [a.strip() for a in applicants if a and a.strip()]
        applicants_str = ", ".join(applicants[:3]) if applicants else "Not available"

        # Inventors
        inventors = root.xpath(
            "//ex:bibliographic-data/ex:parties/ex:inventors/"
            "ex:inventor/ex:inventor-name/ex:name/text()",
            namespaces=ns,
        )
        if not inventors:
            inventors = root.xpath(
                "//ex:bibliographic-data/ex:inventors/ex:inventor/"
                "ex:inventor-name/ex:name/text()",
                namespaces=ns,
            )
        if not inventors:
            inventors = root.xpath(
                "//*[local-name()='inventor-name']/*[local-name()='name']/text()"
            )
        inventors = [i.strip() for i in inventors if i and i.strip()]
        inventors_str = ", ".join(inventors[:3]) if inventors else "Not available"

        # Abstract
        abstract = root.xpath(
            "string(//ex:abstract[@lang='en']/ex:p)", namespaces=ns
        )
        if not abstract:
            abstract = root.xpath("string(//ex:abstract/ex:p)", namespaces=ns)

        # Dates
        pub_date = root.xpath(
            "string(//ex:publication-reference/ex:document-id"
            "[@document-id-type='docdb']/ex:date)",
            namespaces=ns,
        )
        priority_date = root.xpath(
            "string(//ex:priority-claims/ex:priority-claim[1]/ex:document-id/ex:date)",
            namespaces=ns,
        )

        return {
            "title": title,
            "applicants": applicants_str,
            "inventors": inventors_str,
            "abstract": abstract,
            "publication_date": pub_date,
            "priority_date": priority_date,
        }
    except ops_exc.HTTPError as e:
        if e.response.status_code == 404:
            print(f"  Biblio not available for {country}{number}{kind}")
        else:
            print(f"  HTTP error fetching biblio for {country}{number}{kind}: {e}")
        return {}
    except Exception as e:
        print(f"  Error fetching biblio for {country}{number}{kind}: {e}")
        return {}


def calculate_relevance_score(title, abstract):
    """Calculate semantic similarity to research focus (using global model)."""
    if not title and not abstract:
        return 0.0
    patent_text = f"{title} {abstract}"
    try:
        patent_embedding = semantic_model.encode(patent_text)
        similarity = cosine_similarity(
            research_focus_embedding.reshape(1, -1),
            patent_embedding.reshape(1, -1)
        )[0][0]
        return float(similarity)
    except Exception as e:
        print(f"Error calculating score: {e}")
        return 0.0


def parse_patent_refs(root):
    """Extract DOCDB numbers from search result XML."""
    ns = {
        "ops": "http://ops.epo.org",
        "ex": "http://www.epo.org/exchange"
    }
    results = []
    for pub_ref in root.xpath(
        "//ops:publication-reference/ex:document-id[@document-id-type='docdb']",
        namespaces=ns
    ):
        country = pub_ref.xpath("string(ex:country)", namespaces=ns)
        number = pub_ref.xpath("string(ex:doc-number)", namespaces=ns)
        kind = pub_ref.xpath("string(ex:kind)", namespaces=ns)
        if country and number and kind:
            results.append((country, number, kind))
    return results

# ---------------------------------------------------------------
# Search Logic
# ---------------------------------------------------------------

def get_date_range_sixty_days():
    """Return (start_date, end_date) covering the last 60 days in YYYYMMDD format."""
    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=60)

    start_str = start_date.strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")

    print(f"Searching a 60-day window: {start_str} to {end_str}")
    return start_str, end_str


def load_existing_patents():
    """Load existing patents from cumulative CSV and return a set of IDs."""
    if CUMULATIVE_CSV.exists():
        df = pd.read_csv(CUMULATIVE_CSV)
        if 'publication_number' in df.columns and 'kind' in df.columns:
            existing = set(
                df.apply(
                    lambda row: f"{row['country']}{row['publication_number']}{row['kind']}",
                    axis=1
                )
            )
            print(f"Loaded {len(existing)} existing patents from database")
            return existing
        else:
            print("Existing CSV file found but is missing required columns. Starting fresh.")
            return set()
    else:
        print("No existing database found - will create new one")
        return set()


def search_patents():
    """Run OPS search, fetch biblio, score relevance, and return new records as DataFrame."""
    start_date, end_date = get_date_range_sixty_days()
    existing_ids = load_existing_patents()

    cql = f'{SEARCH_TERMS} and pd within "{start_date} {end_date}"'
    print(f"Running CQL: {cql}")

    records = []
    count = 0
    new_count = 0
    skipped_count = 0
    filtered_count = 0
    current_run_date = datetime.now().strftime('%Y-%m-%d')

    for root in scan_patents_cql(cql, batch_size=25, max_records=500):
        refs = parse_patent_refs(root)
        print(f" Found {len(refs)} patents in this batch")

        for country, number, kind in refs:
            patent_id = f"{country}{number}{kind}"

            if patent_id in existing_ids:
                skipped_count += 1
                count += 1
                print(f" {count}. [SKIP] {patent_id} (already in database)")
                continue

            count += 1
            print(f" {count}. [NEW] {patent_id}...", end=" ")

            biblio = get_biblio_data(country, number, kind)
            if not biblio:
                print("✗ Failed to fetch")
                existing_ids.add(patent_id)
                continue

            relevance = calculate_relevance_score(
                biblio.get("title", ""),
                biblio.get("abstract", "")
            )
            print(f"[Relevance: {relevance:.2f}]", end=" ")

            if relevance < MIN_RELEVANCE_SCORE:
                filtered_count += 1
                print(f"✗ Filtered (below {MIN_RELEVANCE_SCORE})")
                existing_ids.add(patent_id)
                continue

            print("Summarizing...", end=" ")
            ai_summary = generate_ai_summary(
                biblio.get("title", ""),
                biblio.get("abstract", ""),
                ""
            )

            new_count += 1
            link = f"https://worldwide.espacenet.com/patent/search/family/{country}{number}{kind}"

            records.append({
                "country": country,
                "publication_number": number,
                "kind": kind,
                "title": biblio.get("title", ""),
                "applicants": biblio.get("applicants", ""),
                "inventors": biblio.get("inventors", ""),
                "abstract": biblio.get("abstract", "")[:500] if biblio.get("abstract") else "",
                "publication_date": biblio.get("publication_date", ""),
                "priority_date": biblio.get("priority_date", ""),
                "relevance_score": round(relevance, 2),
                "ai_summary": ai_summary,
                "link": link,
                "date_added": current_run_date,
                "is_new": "YES"
            })

            print(f"✓ {biblio.get('title', 'N/A')[:40]}")
            time.sleep(0.3)

    print(f"\n{'='*80}")
    print("Summary:")
    print(f" Total found: {count}")
    print(f" Already in DB: {skipped_count}")
    print(f" Below relevance threshold ({MIN_RELEVANCE_SCORE}): {filtered_count}")
    print(f" Added to database: {new_count}")
    print(f"{'='*80}\n")

    return pd.DataFrame(records)

# ---------------------------------------------------------------
# Backfill missing biblio
# ---------------------------------------------------------------

def backfill_biblio_for_existing():
    """
    For existing rows in patents_cumulative.csv that have missing or 'Not available'
    applicants/inventors, call get_biblio_data() once and update them.
    """
    if not CUMULATIVE_CSV.exists():
        print("No cumulative CSV found. Skipping biblio backfill.")
        return

    if not client:
        print("OPS client not initialized. Cannot backfill biblio.")
        return

    df = pd.read_csv(CUMULATIVE_CSV)
    if df.empty:
        print("Cumulative CSV is empty. Nothing to backfill.")
        return

    for col in ["applicants", "inventors"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)

    mask_missing = (
        (df["applicants"].str.strip().eq("")) |
        (df["applicants"].str.contains("Not available", case=False)) |
        (df["inventors"].str.strip().eq("")) |
        (df["inventors"].str.contains("Not available", case=False))
    )

    df_to_fix = df[mask_missing].copy()
    if df_to_fix.empty:
        print("No rows with missing applicants/inventors to backfill.")
        return

    print(
        f"🔧 Backfilling biblio for {len(df_to_fix)} existing patents "
        "with missing applicants/inventors..."
    )

    updated = 0
    for idx, row in df_to_fix.iterrows():
        country = str(row.get("country", "")).strip()
        number = str(row.get("publication_number", "")).strip()
        kind = str(row.get("kind", "")).strip()

        if not (country and number and kind):
            print(f"  Skipping row {idx}: incomplete ID {country}{number}{kind}")
            continue

        print(f"  Fetching biblio for {country}{number}{kind} ...", end=" ")
        biblio = get_biblio_data(country, number, kind)
        if not biblio:
            print("✗ failed")
            continue

        df.loc[idx, "title"] = biblio.get("title", df.loc[idx].get("title", ""))
        df.loc[idx, "applicants"] = biblio.get(
            "applicants", df.loc[idx].get("applicants", "")
        )
        df.loc[idx, "inventors"] = biblio.get(
            "inventors", df.loc[idx].get("inventors", "")
        )
        df.loc[idx, "abstract"] = biblio.get(
            "abstract", df.loc[idx].get("abstract", "")
        )
        df.loc[idx, "publication_date"] = biblio.get(
            "publication_date", df.loc[idx].get("publication_date", "")
        )
        df.loc[idx, "priority_date"] = biblio.get(
            "priority_date", df.loc[idx].get("priority_date", "")
        )

        updated += 1
        print("✓")
        time.sleep(0.3)

    print(f"✅ Backfill complete. Updated {updated} rows.")
    backup_path = CUMULATIVE_CSV.with_name(
        CUMULATIVE_CSV.stem + "_backup_before_biblio.csv"
    )
    df.to_csv(backup_path, index=False)
    print(f"Backup saved to {backup_path}")
    df.to_csv(CUMULATIVE_CSV, index=False)
    print("Patents CSV updated in-place with biblio data.")

# ---------------------------------------------------------------
# Processing Existing Data
# ---------------------------------------------------------------

def process_existing_records(df_old):
    """
    Ensures all existing records have relevance scores and AI summaries.
    """
    df = df_old.copy()

    if 'relevance_score' not in df.columns:
        df['relevance_score'] = 0.0
    if 'ai_summary' not in df.columns:
        df['ai_summary'] = ''

    df['relevance_score'] = pd.to_numeric(df['relevance_score'], errors='coerce').fillna(0.0)

    summary_col = df['ai_summary'].fillna('').astype(str).str.lower()
    missing_score = (df['relevance_score'] <= 0.001)
    missing_summary = (summary_col == '') | \
                      summary_col.str.contains('not available') | \
                      summary_col.str.contains('summary generation failed')

    missing_mask = missing_score | missing_summary
    df_to_update = df[missing_mask].copy()

    if df_to_update.empty:
        print("No existing records require scoring or summarization.")
        return df_old

    print(f"\n⚡ Processing {len(df_to_update)} existing records for scoring/summaries...")

    for index, row in df_to_update.iterrows():
        try:
            title = str(row['title']) if pd.notna(row['title']) else ""
            abstract = str(row['abstract']) if pd.notna(row['abstract']) else ""
            score = calculate_relevance_score(title, abstract)

            abstract_text = abstract.split('.')
            summary = '.'.join(abstract_text[:3]).strip()
            summary = summary if len(summary) > 10 else title

            df.loc[index, 'relevance_score'] = score
            df.loc[index, 'ai_summary'] = summary

            print(f" [UPDATED] {row['country']}{row['publication_number']} - Score: {score:.4f}")
        except Exception as e:
            patent_id = f"{row['country']}{row['publication_number']}"
            print(f" [ERROR] Could not process {patent_id}: {e}")
            df.loc[index, 'ai_summary'] = "Summary generation failed."

    df.sort_values(by='relevance_score', ascending=False, inplace=True)
    print("✅ Finished processing existing records.")
    return df

# ---------------------------------------------------------------
# CSV Merge
# ---------------------------------------------------------------

def update_cumulative_csv(df_new):
    """Merge new results with existing cumulative CSV and process old data."""
    FINAL_COLUMNS = [
        "country",
        "publication_number",
        "kind",
        "title",
        "applicants",
        "inventors",
        "abstract",
        "publication_date",
        "priority_date",
        "relevance_score",
        "ai_summary",
        "link",
        "date_added",
        "is_new"
    ]

    if CUMULATIVE_CSV.exists():
        df_old = pd.read_csv(CUMULATIVE_CSV)
        df_old = process_existing_records(df_old)
        df_old['is_new'] = 'NO'

        if 'date_added' not in df_old.columns:
            df_old['date_added'] = 'Unknown'
        if 'relevance_score' not in df_old.columns:
            df_old['relevance_score'] = 0.0
        if 'ai_summary' not in df_old.columns:
            df_old['ai_summary'] = 'Not available'

        df_all = pd.concat([df_old, df_new], ignore_index=True).drop_duplicates(
            subset=["country", "publication_number", "kind"],
            keep="first"
        )
        new_count = len(df_new)
        print(f"Added {new_count} new patents (marked as 'is_new=YES')")
    else:
        df_all = df_new
        print(f"Created new database with {len(df_all)} patents (all marked as 'is_new=YES')")

    df_all = df_all.reindex(columns=FINAL_COLUMNS, fill_value='')
    df_all = df_all.sort_values(['relevance_score', 'date_added'], ascending=[False, False])
    df_all.to_csv(CUMULATIVE_CSV, index=False)
    print(f"Saved cumulative CSV with {len(df_all)} total records.")
    return df_all

# ---------------------------------------------------------------
# Column Initialization Logic
# ---------------------------------------------------------------

def ensure_ai_columns_exist():
    """Reads existing data and adds/updates AI columns if they are missing."""
    if not CUMULATIVE_CSV.exists():
        return
    df = pd.read_csv(CUMULATIVE_CSV)
    if 'relevance_score' not in df.columns or 'ai_summary' not in df.columns:
        print("⚡ Processing existing records: Adding missing AI columns with default values.")
        if 'relevance_score' not in df.columns:
            df['relevance_score'] = 0.0
        if 'ai_summary' not in df.columns:
            df['ai_summary'] = 'Not available'
        df.to_csv(CUMULATIVE_CSV, index=False)
        print(f"✓ AI columns successfully initialized for {len(df)} existing records.")
    else:
        print("✓ AI columns already present in existing database.")

# ---------------------------------------------------------------
# Email Sending
# ---------------------------------------------------------------

def send_email_with_csv(df_all):
    """Send the updated CSV via email with summary."""
    sender = os.environ.get("SENDER_EMAIL")
    password = os.environ.get("EMAIL_PASSWORD")
    recipient = os.environ.get("RECIPIENT_EMAIL")

    if not sender or not recipient or not password:
        print("Warning: Email credentials (SENDER_EMAIL, EMAIL_PASSWORD, RECIPIENT_EMAIL) not found. Skipping email.")
        return

    new_patents = df_all[df_all['is_new'] == 'YES']

    email_body = f"""
Bimonthly Patent Update - {datetime.now().strftime('%Y-%m-%d')}
{'='*80}

NEW PATENTS: {len(new_patents)}
TOTAL DATABASE: {len(df_all)} patents

SEARCH TERMS:
{SEARCH_TERMS}

SEMANTIC SEARCH FOCUS:
{RESEARCH_FOCUS}
{'='*80}
"""

    if len(new_patents) > 0:
        email_body += "\n🔥 TOP 5 MOST RELEVANT NEW PATENTS:\n\n"
        top_patents = new_patents.nlargest(5, 'relevance_score')
        for idx, patent in enumerate(top_patents.itertuples(), 1):
            email_body += f"{idx}. [{patent.relevance_score:.2f}] {patent.title[:80]}\n"
            email_body += f"  Applicant: {patent.applicants[:60]}\n"
            email_body += f"  Summary: {patent.ai_summary[:200]}\n\n"

    email_body += f"\nSee attached CSV for full details.\n\n{'='*80}"

    msg = MIMEMultipart()
    msg["Subject"] = f"Patent Update - {len(new_patents)} New Patents - {datetime.now().strftime('%Y-%m-%d')}"
    msg["From"] = sender
    msg["To"] = recipient

    body = MIMEText(email_body, "plain")
    msg.attach(body)

    try:
        with open(CUMULATIVE_CSV, "rb") as f:
            attachment = MIMEApplication(f.read(), Name="patents_cumulative.csv")
        attachment["Content-Disposition"] = 'attachment; filename=\"patents_cumulative.csv\"'
        msg.attach(attachment)

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(sender, password)
            smtp.send_message(msg)
        print("✓ Email sent successfully.")
    except Exception as e:
        print(f"✗ Error sending email: {e}")

# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    print("="*80)
    print(f"Starting 1-Year AI-Enhanced Patent Search - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Relevance threshold: {MIN_RELEVANCE_SCORE}")
    print("AI summaries: Local Heuristic (No API Key Required)")
    print("="*80)

    ensure_ai_columns_exist()
    backfill_biblio_for_existing()
    df_new = search_patents()
    df_all = update_cumulative_csv(df_new)
    send_email_with_csv(df_all)

    print("\n" + "="*80)
    print("Process completed successfully.")
    print("="*80)


if __name__ == "__main__":
    main()



  

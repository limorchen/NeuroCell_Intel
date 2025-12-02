import os
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

from epo_ops import Client, models, middlewares
import epo_ops.exceptions as ops_exc

# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
CUMULATIVE_CSV = DATA_DIR / "patents_cumulative.csv"

key = os.environ.get("EPO_OPS_KEY")
secret = os.environ.get("EPO_OPS_SECRET")

if not key or not secret:
    raise ValueError("Missing EPO OPS API credentials.")

middlewares_list = [
    middlewares.Dogpile(),
    middlewares.Throttler()
]

client = Client(
    key=key,
    secret=secret,
    middlewares=middlewares_list
)


# ---------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------

def scan_patents_cql(cql_query, batch_size=25, max_records=None):
    """Iterate over OPS search results."""
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

        root = etree.fromstring(resp.content)
        ns = {"ops": "http://ops.epo.org"}

        if total is None:
            total_str = root.xpath("string(//ops:biblio-search/@total-result-count)", namespaces=ns)
            if not total_str or total_str == "0":
                print("No results found.")
                break
            total = int(total_str)
            print(f"Total results: {total}")

        print(f"Fetching records {start}–{end}...")
        yield root

        if end >= total:
            break
        if max_records and end >= max_records:
            break

        start = end + 1
        time.sleep(0.3)


def get_biblio_data(country, number, kind):
    """Fetch full bibliographic data."""
    try:
        resp = client.published_data(
            reference_type="publication",
            input=models.Docdb(number, country, kind),
            endpoint="biblio"
        )
        
        root = etree.fromstring(resp.content)
        ns = {"ex": "http://www.epo.org/exchange"}
        
        # Extract title
        title = root.xpath("string(//ex:invention-title[@lang='en'])", namespaces=ns)
        if not title:
            title = root.xpath("string(//ex:invention-title)", namespaces=ns)
        
        # Extract applicants
        applicants = root.xpath("//ex:applicants/ex:applicant/ex:applicant-name/ex:name/text()", namespaces=ns)
        applicants_str = ", ".join(applicants) if applicants else ""
        
        # Extract inventors
        inventors = root.xpath("//ex:inventors/ex:inventor/ex:inventor-name/ex:name/text()", namespaces=ns)
        inventors_str = ", ".join(inventors) if inventors else ""
        
        # Extract abstract
        abstract = root.xpath("string(//ex:abstract[@lang='en']/ex:p)", namespaces=ns)
        if not abstract:
            abstract = root.xpath("string(//ex:abstract/ex:p)", namespaces=ns)
        
        # Extract publication date
        pub_date = root.xpath("string(//ex:publication-reference/ex:document-id[@document-id-type='docdb']/ex:date)", namespaces=ns)
        
        # Extract priority date
        priority_date = root.xpath("string(//ex:priority-claims/ex:priority-claim[1]/ex:document-id/ex:date)", namespaces=ns)
        
        return {
            "title": title,
            "applicants": applicants_str,
            "inventors": inventors_str,
            "abstract": abstract[:500] if abstract else "",
            "publication_date": pub_date,
            "priority_date": priority_date
        }
    except ops_exc.HTTPError as e:
        if e.response.status_code == 404:
            print(f"  Biblio not available for {country}{number}")
        return {}
    except Exception as e:
        print(f"  Error fetching biblio for {country}{number}{kind}: {e}")
        return {}


def parse_patent_refs(root):
    """Extract DOCDB numbers from search result XML."""
    ns = {
        "ops": "http://ops.epo.org",
        "ex": "http://www.epo.org/exchange"
    }
    results = []

    # The correct XPath for publication references
    for pub_ref in root.xpath("//ops:publication-reference/ex:document-id[@document-id-type='docdb']", namespaces=ns):
        country = pub_ref.xpath("string(ex:country)", namespaces=ns)
        number = pub_ref.xpath("string(ex:doc-number)", namespaces=ns)
        kind = pub_ref.xpath("string(ex:kind)", namespaces=ns)

        if country and number and kind:
            results.append((country, number, kind))

    return results


# ---------------------------------------------------------------
# Search Logic
# ---------------------------------------------------------------

def get_date_range_two_months():
    """Return (start_date, end_date) covering the last 2 months in YYYYMMDD format."""
    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=60)
    
    # Format as YYYYMMDD (no dashes)
    start_str = start_date.strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")
    
    return start_str, end_str


def load_existing_patents():
    """Load existing patents from cumulative CSV and return a set of IDs."""
    if CUMULATIVE_CSV.exists():
        df = pd.read_csv(CUMULATIVE_CSV)
        # Create a set of unique patent identifiers
        existing = set(
            df.apply(lambda row: f"{row['country']}{row['publication_number']}{row['kind']}", axis=1)
        )
        print(f"Loaded {len(existing)} existing patents from database")
        return existing
    else:
        print("No existing database found - will create new one")
        return set()


def search_patents():
    start_date, end_date = get_date_range_two_months()
    
    # Load existing patents to avoid re-fetching
    existing_ids = load_existing_patents()

    # IMPORTANT: Add your search terms here to avoid fetching ALL patents
    # Example: searching for exosome + CNS patents
    search_terms = '(ta=exosomes or ta="extracellular vesicles") and ta=CNS'
    
    cql = f'{search_terms} and pd within "{start_date} {end_date}"'
    print(f"Running CQL: {cql}")

    records = []
    count = 0
    new_count = 0
    skipped_count = 0

    for root in scan_patents_cql(cql, batch_size=25, max_records=500):  # Limit to 500 for safety
        refs = parse_patent_refs(root)
        print(f"  Found {len(refs)} patents in this batch")
        
        for country, number, kind in refs:
            patent_id = f"{country}{number}{kind}"
            
            # Check if this patent is already in database
            if patent_id in existing_ids:
                skipped_count += 1
                count += 1
                print(f"  {count}. [SKIP] {patent_id} (already in database)")
                continue
            
            # This is a new patent - fetch full data
            new_count += 1
            count += 1
            print(f"  {count}. [NEW] {patent_id}...", end=" ")

            biblio = get_biblio_data(country, number, kind)
            
            records.append({
                "country": country,
                "publication_number": number,
                "kind": kind,
                "title": biblio.get("title", ""),
                "applicants": biblio.get("applicants", ""),
                "inventors": biblio.get("inventors", ""),
                "abstract": biblio.get("abstract", ""),
                "publication_date": biblio.get("publication_date", ""),
                "priority_date": biblio.get("priority_date", ""),
                "date_found": end_date
            })
            
            print(f"✓ {biblio.get('title', 'N/A')[:50]}")
            time.sleep(0.3)  # Be nice to the API
    
    print(f"\nSummary: Found {count} total, {new_count} new, {skipped_count} already in database")
    return pd.DataFrame(records)


# ---------------------------------------------------------------
# CSV Merge
# ---------------------------------------------------------------

def update_cumulative_csv(df_new):
    """Merge new results with existing cumulative CSV."""
    if df_new.empty:
        print("No new patents found.")
        if CUMULATIVE_CSV.exists():
            return pd.read_csv(CUMULATIVE_CSV)
        return df_new
    
    if CUMULATIVE_CSV.exists():
        df_old = pd.read_csv(CUMULATIVE_CSV)
        print(f"Loaded {len(df_old)} existing patents")
        
        # Merge and remove duplicates
        df_all = pd.concat([df_old, df_new], ignore_index=True).drop_duplicates(
            subset=["country", "publication_number", "kind"],
            keep="first"
        )
        
        new_count = len(df_all) - len(df_old)
        print(f"Added {new_count} new patents")
    else:
        df_all = df_new
        print(f"Created new database with {len(df_all)} patents")

    df_all.to_csv(CUMULATIVE_CSV, index=False)
    print(f"Saved cumulative CSV with {len(df_all)} total records.")
    return df_all


# ---------------------------------------------------------------
# Email Sending
# ---------------------------------------------------------------

def send_email_with_csv():
    """Send the updated CSV via email."""
    sender = os.environ.get("SENDER_EMAIL")
    password = os.environ.get("EMAIL_PASSWORD")
    recipient = os.environ.get("RECIPIENT_EMAIL")

    if not sender or not recipient or not password:
        print("Warning: Email credentials not found. Skipping email.")
        return

    msg = MIMEMultipart()
    msg["Subject"] = f"Bimonthly Patent Update - {datetime.now().strftime('%Y-%m-%d')}"
    msg["From"] = sender
    msg["To"] = recipient

    body = MIMEText("Attached is the updated patents CSV.", "plain")
    msg.attach(body)

    with open(CUMULATIVE_CSV, "rb") as f:
        attachment = MIMEApplication(f.read(), Name="patents_cumulative.csv")
    attachment["Content-Disposition"] = 'attachment; filename="patents_cumulative.csv"'
    msg.attach(attachment)

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(sender, password)
            smtp.send_message(msg)
        print("Email sent successfully.")
    except Exception as e:
        print(f"Error sending email: {e}")


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    print("="*80)
    print(f"Starting 2-month patent search - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    df_new = search_patents()
    df_all = update_cumulative_csv(df_new)
    
    # Optional: send email (only if credentials are set)
    send_email_with_csv()
    
    print("\n" + "="*80)
    print("Process completed successfully.")
    print("="*80)


if __name__ == "__main__":
    main()

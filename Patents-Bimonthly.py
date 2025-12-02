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
    except Exception as e:
        print(f"Error fetching biblio for {country}{number}{kind}: {e}")
        return {}

    try:
        root = etree.fromstring(resp.content)
        return {
            "title": root.xpath("string(//invention-title)", namespaces=root.nsmap),
            "applicants": ", ".join(root.xpath("//applicants//name/text()", namespaces=root.nsmap)),
            "inventors": ", ".join(root.xpath("//inventors//name/text()", namespaces=root.nsmap)),
            "abstract": root.xpath("string(//abstract)", namespaces=root.nsmap)
        }
    except Exception:
        return {}


def parse_patent_refs(root):
    """Extract DOCDB numbers from search result XML."""
    ns = {"ex": "http://www.epo.org/exchange"}
    results = []

    for pub in root.xpath("//ex:exchange-documents/ex:exchange-document", namespaces=ns):
        country = pub.get("country")
        number = pub.get("doc-number")
        kind = pub.get("kind")

        if not all([country, number, kind]):
            continue

        results.append((country, number, kind))

    return results


# ---------------------------------------------------------------
# Search Logic
# ---------------------------------------------------------------

def get_date_range_two_months():
    """Return (start_date, end_date) covering the last 2 months."""
    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=60)
    return start_date, end_date


def search_patents():
    start_date, end_date = get_date_range_two_months()

    cql = f'pd within "{start_date} {end_date}"'
    print(f"Running CQL: {cql}")

    records = []
    seen = set()

    for root in scan_patents_cql(cql, batch_size=25):
        refs = parse_patent_refs(root)
        for country, number, kind in refs:
            key_tuple = (country, number, kind)
            if key_tuple in seen:
                continue
            seen.add(key_tuple)

            biblio = get_biblio_data(country, number, kind)
            records.append({
                "country": country,
                "number": number,
                "kind": kind,
                "title": biblio.get("title", ""),
                "applicants": biblio.get("applicants", ""),
                "inventors": biblio.get("inventors", ""),
                "abstract": biblio.get("abstract", ""),
                "date_found": str(end_date)
            })

    return pd.DataFrame(records)


# ---------------------------------------------------------------
# CSV Merge
# ---------------------------------------------------------------

def update_cumulative_csv(df_new):
    if CUMULATIVE_CSV.exists():
        df_old = pd.read_csv(CUMULATIVE_CSV)
        df_all = pd.concat([df_old, df_new], ignore_index=True).drop_duplicates(
            subset=["country", "number", "kind"],
            keep="first"
        )
    else:
        df_all = df_new

    df_all.to_csv(CUMULATIVE_CSV, index=False)
    print(f"Saved cumulative CSV with {len(df_all)} records.")
    return df_all


# ---------------------------------------------------------------
# Email Sending
# ---------------------------------------------------------------

def send_email_with_csv():
    sender = os.environ.get("SENDER_EMAIL")
    password = os.environ.get("EMAIL_PASSWORD")
    recipient = os.environ.get("RECIPIENT_EMAIL")

    if not sender or not recipient or not password:
        raise ValueError("Missing email credentials in secrets.")

    msg = MIMEMultipart()
    msg["Subject"] = "Bimonthly Patent Update"
    msg["From"] = sender
    msg["To"] = recipient

    body = MIMEText("Attached is the updated patents CSV.", "plain")
    msg.attach(body)

    with open(CUMULATIVE_CSV, "rb") as f:
        attachment = MIMEApplication(f.read(), Name="patents_cumulative.csv")
    attachment["Content-Disposition"] = 'attachment; filename="patents_cumulative.csv"'
    msg.attach(attachment)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(sender, password)
        smtp.send_message(msg)

    print("Email sent successfully.")


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    print("Starting 2-month patent search...")
    df_new = search_patents()
    df_all = update_cumulative_csv(df_new)
    send_email_with_csv()
    print("Process completed.")


if __name__ == "__main__":
    main()

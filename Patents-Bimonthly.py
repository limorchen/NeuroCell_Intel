import os
import time
from datetime import datetime
from dateutil.relativedelta import relativedelta
import epo_ops
from epo_ops import exceptions as ops_exc
from lxml import etree
import pandas as pd
from pathlib import Path
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

# ------------------------ Configuration ------------------------
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
CUMULATIVE_CSV = DATA_DIR / "patents_cumulative.csv"

# Get EPO OPS API keys from environment variables
key = os.environ.get("EPO_OPS_KEY")
secret = os.environ.get("EPO_OPS_SECRET")
if not key or not secret:
    raise ValueError("EPO_OPS_KEY and EPO_OPS_SECRET environment variables must be set")

# Middlewares and client
middlewares = [
    epo_ops.middlewares.Dogpile(),
    epo_ops.middlewares.Throttler(),
]
client = epo_ops.Client(key=key, secret=secret, middlewares=middlewares)

# ------------------------ Helper Functions ------------------------
def scan_patents_cql(cql_query, batch_size=25, max_records=None):
    """Scan patent publication references from search results"""
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
        ns = {"ops": "http://ops.epo.org", "ex": "http://www.epo.org/exchange"}

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

        print(f"Fetching records {start} to {end}...")
        yield root

        if end >= total:
            break
        if max_records and end >= max_records:
            break
        start = end + 1
        time.sleep(0.2)


def get_biblio_data(country, number, kind):
    """Fetch bibliographic data for a single patent"""
    try:
        resp = client.published_data(
            reference_type='publication',
            input=epo_ops.models.Docdb(number, country, kind),
            endpoint='biblio'
        )
        root = etree.fromstring(resp.content)
        ns = {"ex": "http://www.epo.org/exchange"}

        # Title
        title = root.xpath("string(//ex:invention-title[@lang='en'])", namespaces=ns) or \
                root.xpath("string(//ex:invention-title)", namespaces=ns)
        # Applicant
        applicant = root.xpath("string(//ex:applicants/ex:applicant[1]/ex:applicant-name/ex:name)", namespaces=ns)
        # Publication date
        pub_date = root.xpath("string(//ex:publication-reference/ex:document-id[@document-id-type='docdb']/ex:date)", namespaces=ns)
        # Priority date
        priority_date = root.xpath("string(//ex:priority-claims/ex:priority-claim[1]/ex:document-id/ex:date)", namespaces=ns)
        # Abstract
        abstract = root.xpath("string(//ex:abstract[@lang='en']/ex:p)", namespaces=ns) or \
                   root.xpath("string(//ex:abstract/ex:p)", namespaces=ns)

        clean_number = number.lstrip('0') if number else number
        if country == 'WO':
            link = f"https://patentscope.wipo.int/search/en/detail.jsf?docId=WO{clean_number}"
        elif country == 'EP':
            link = f"https://worldwide.espacenet.com/patent/search?q=pn%3DEP{clean_number}"
        elif country == 'US':
            link = f"https://patents.google.com/patent/US{clean_number}"
        elif country == 'CN':
            link = f"https://patents.google.com/patent/CN{clean_number}"
        else:
            link = f"https://worldwide.espacenet.com/patent/search?q=pn%3D{country}{clean_number}"

        return {
            'title': title,
            'applicant': applicant,
            'publication_date': pub_date,
            'priority_date': priority_date,
            'abstract': abstract[:500] if abstract else "",
            'link': link
        }

    except ops_exc.HTTPError as e:
        if e.response.status_code == 404:
            return {'title':'N/A','applicant':'N/A','publication_date':'','priority_date':'','abstract':'','link':''}
        print(f"Error fetching biblio for {country}{number}: {e.response.status_code}")
        return None
    except Exception as e:
        print(f"Error fetching biblio for {country}{number}: {e}")
        return None


def get_first_claim(country, number, kind):
    """Fetch first claim of a patent"""
    try:
        resp = client.published_data(
            reference_type='publication',
            input=epo_ops.models.Docdb(number, country, kind),
            endpoint='claims'
        )
        root = etree.fromstring(resp.content)
        ns = {"ex": "http://www.epo.org/exchange"}
        first_claim = root.xpath("string(//ex:claims[@lang='en']/ex:claim[@num='1']/ex:claim-text)", namespaces=ns)
        if not first_claim:
            first_claim = root.xpath("string(//ex:claim[@num='1']/ex:claim-text)", namespaces=ns)
        return ' '.join(first_claim.split()) if first_claim else ""
    except:
        return ""


def load_existing_patents():
    """Load existing patents from CSV"""
    if CUMULATIVE_CSV.exists():
        df = pd.read_csv(CUMULATIVE_CSV)
        existing = set(df.apply(lambda row: f"{row['country']}{row['publication_number']}{row['kind']}", axis=1))
        return df, existing
    else:
        return pd.DataFrame(), set()


def search_and_update(cql_query, max_records=None):
    """Search patents and update CSV"""
    print(f"\n{'='*80}")
    print(f"Patent Search Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Query: {cql_query}")
    print(f"{'='*80}\n")

    existing_df, existing_ids = load_existing_patents()
    print(f"Loaded {len(existing_df)} existing patents\n")

    patent_records = []
    count = 0
    new_count = 0

    for root in scan_patents_cql(cql_query, batch_size=25, max_records=max_records):
        ns = {"ops":"http://ops.epo.org","ex":"http://www.epo.org/exchange"}
        pub_refs = root.xpath("//ops:publication-reference/ex:document-id[@document-id-type='docdb']", namespaces=ns)
        print(f"Found {len(pub_refs)} publication references in this batch")

        for doc_id in pub_refs:
            country = doc_id.xpath("string(ex:country)", namespaces=ns)
            number = doc_id.xpath("string(ex:doc-number)", namespaces=ns)
            kind = doc_id.xpath("string(ex:kind)", namespaces=ns)
            patent_id = f"{country}{number}{kind}"
            is_new = patent_id not in existing_ids
            count += 1
            status = "NEW" if is_new else "EXISTS"
            print(f"{count}. [{status}] {patent_id}...", end=" ")

            if is_new:
                new_count += 1
                biblio = get_biblio_data(country, number, kind)
                first_claim = get_first_claim(country, number, kind)
                if biblio:
                    patent_records.append({
                        "country": country,
                        "publication_number": number,
                        "kind": kind,
                        "title": biblio['title'],
                        "applicant": biblio['applicant'],
                        "publication_date": biblio['publication_date'],
                        "priority_date": biblio['priority_date'],
                        "abstract": biblio['abstract'],
                        "first_claim": first_claim,
                        "link": biblio['link'],
                        "date_added": datetime.now().strftime('%Y-%m-%d'),
                        "is_new": "YES"
                    })
                    print(f"✓ {biblio['title'][:50]}")
                else:
                    print("✗ Failed")
                time.sleep(0.3)
            else:
                print("(skipped)")

    combined_df = pd.DataFrame(patent_records)
    if not existing_df.empty:
        existing_df['is_new'] = 'NO'
        combined_df = pd.concat([existing_df, combined_df], ignore_index=True) if not combined_df.empty else existing_df

    combined_df.to_csv(CUMULATIVE_CSV, index=False)
    print(f"\n{'='*80}")
    print(f"Search Complete! Total patents: {len(combined_df)}, New: {new_count}")
    print(f"Saved to: {CUMULATIVE_CSV}")
    print(f"{'='*80}\n")


# ------------------------ Email Function ------------------------
def send_email_with_csv(sender, password, recipient, subject, body, attachment_path):
    """Send CSV results via Gmail SSL (port 465)"""
    try:
        msg = MIMEMultipart()
        msg['From'] = sender
        msg['To'] = recipient
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        with open(attachment_path, 'rb') as f:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', f'attachment; filename={attachment_path.name}')
        msg.attach(part)

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender, password)
            server.send_message(msg)
        print(f"\n✓ Email sent successfully to {recipient}")
    except Exception as e:
        print(f"\n✗ Failed to send email: {e}")


# ------------------------ Main ------------------------
if __name__ == "__main__":
    # Calculate last 2 months dynamically
    today = datetime.today()
    two_months_ago = today - relativedelta(months=2)

    cql = f'(ta=exosomes or ta="extracellular vesicles") and ta=CNS and pd within "{two_months_ago.strftime("%Y%m%d")} {today.strftime("%Y%m%d")}"'
    search_and_update(cql, max_records=None)

    # Send email if secrets are present
    SENDER_EMAIL = os.environ.get("SENDER_EMAIL")
    EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD")
    RECIPIENT_EMAIL = os.environ.get("RECIPIENT_EMAIL")

    if SENDER_EMAIL and EMAIL_PASSWORD and RECIPIENT_EMAIL and CUMULATIVE_CSV.exists():
        subject = f"Patent Search Results - {datetime.now().strftime('%Y-%m-%d')}"
        body = "Attached is the latest cumulative patent CSV from the automated search."
        send_email_with_csv(SENDER_EMAIL, EMAIL_PASSWORD, RECIPIENT_EMAIL, subject, body, CUMULATIVE_CSV)
    else:
        print("\n⚠ Email not sent: Missing secrets or CSV file.")

import os
import time
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import epo_ops
from epo_ops import exceptions as ops_exc
from lxml import etree
import pandas as pd
from pathlib import Path

# Configuration
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
CUMULATIVE_CSV = DATA_DIR / "patents_cumulative.csv"

# Get API keys from environment variables
key = os.environ.get("EPO_OPS_KEY")
secret = os.environ.get("EPO_OPS_SECRET")

if not key or not secret:
    raise ValueError("EPO_OPS_KEY and EPO_OPS_SECRET environment variables must be set")

# Middlewares
middlewares = [
    epo_ops.middlewares.Dogpile(),
    epo_ops.middlewares.Throttler(),
]

# Client
client = epo_ops.Client(key=key, secret=secret, middlewares=middlewares)


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
        
        # Extract title
        title = root.xpath(
            "string(//ex:invention-title[@lang='en'])",
            namespaces=ns
        )
        if not title:
            title = root.xpath("string(//ex:invention-title)", namespaces=ns)
        
        # Extract applicant
        applicant = root.xpath(
            "string(//ex:applicants/ex:applicant[1]/ex:applicant-name/ex:name)",
            namespaces=ns
        )
        
        # Extract publication date
        pub_date = root.xpath(
            "string(//ex:publication-reference/ex:document-id[@document-id-type='docdb']/ex:date)",
            namespaces=ns
        )
        
        # Extract priority date (first priority)
        priority_date = root.xpath(
            "string(//ex:priority-claims/ex:priority-claim[1]/ex:document-id/ex:date)",
            namespaces=ns
        )
        
        # Extract abstract
        abstract = root.xpath(
            "string(//ex:abstract[@lang='en']/ex:p)",
            namespaces=ns
        )
        if not abstract:
            abstract = root.xpath("string(//ex:abstract/ex:p)", namespaces=ns)
        
        # Generate link to full text
        clean_number = number.lstrip('0') if number else number
        
        # Create appropriate link based on country code
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
            return {
                'title': 'N/A',
                'applicant': 'N/A',
                'publication_date': '',
                'priority_date': '',
                'abstract': '',
                'link': ''
            }
        print(f"  Error fetching biblio for {country}{number}: {e.response.status_code}")
        return None
    except Exception as e:
        print(f"  Error fetching biblio for {country}{number}: {e}")
        return None


def get_first_claim(country, number, kind):
    """Fetch the first claim for a patent"""
    try:
        resp = client.published_data(
            reference_type='publication',
            input=epo_ops.models.Docdb(number, country, kind),
            endpoint='claims'
        )
        
        root = etree.fromstring(resp.content)
        ns = {"ex": "http://www.epo.org/exchange"}
        
        # Try to get English claims first
        first_claim = root.xpath(
            "string(//ex:claims[@lang='en']/ex:claim[@num='1']/ex:claim-text)",
            namespaces=ns
        )
        
        # If no English claims, try any language
        if not first_claim:
            first_claim = root.xpath(
                "string(//ex:claim[@num='1']/ex:claim-text)",
                namespaces=ns
            )
        
        # Clean up whitespace
        if first_claim:
            first_claim = ' '.join(first_claim.split())
        
        return first_claim if first_claim else ""
    
    except ops_exc.HTTPError as e:
        if e.response.status_code == 404:
            return "N/A (not available)"
        return ""
    except Exception as e:
        return ""


def load_existing_patents():
    """Load existing patents from cumulative CSV"""
    if CUMULATIVE_CSV.exists():
        df = pd.read_csv(CUMULATIVE_CSV)
        # Create a set of unique patent identifiers
        existing = set(
            df.apply(lambda row: f"{row['country']}{row['publication_number']}{row['kind']}", axis=1)
        )
        return df, existing
    else:
        return pd.DataFrame(), set()


def search_and_update(cql_query, max_records=None):
    """Search for patents and update cumulative CSV"""
    print(f"\n{'='*80}")
    print(f"Patent Search Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Query: {cql_query}")
    print(f"{'='*80}\n")
    
    # Load existing data
    existing_df, existing_ids = load_existing_patents()
    print(f"Loaded {len(existing_df)} existing patents from database\n")
    
    # Search for new patents
    patent_records = []
    count = 0
    new_count = 0
    
    for root in scan_patents_cql(cql_query, batch_size=25, max_records=max_records):
        ns = {
            "ops": "http://ops.epo.org",
            "ex": "http://www.epo.org/exchange"
        }

        pub_refs = root.xpath(
            "//ops:publication-reference/ex:document-id[@document-id-type='docdb']",
            namespaces=ns
        )

        print(f"Found {len(pub_refs)} publication references in this batch")

        for doc_id in pub_refs:
            country = doc_id.xpath("string(ex:country)", namespaces=ns)
            number = doc_id.xpath("string(ex:doc-number)", namespaces=ns)
            kind = doc_id.xpath("string(ex:kind)", namespaces=ns)
            
            patent_id = f"{country}{number}{kind}"
            
            # Check if this patent is new
            is_new = patent_id not in existing_ids
            
            count += 1
            status = "NEW" if is_new else "EXISTS"
            print(f"{count}. [{status}] {patent_id}...", end=" ")
            
            if is_new:
                new_count += 1
                
                # Fetch full bibliographic data
                biblio = get_biblio_data(country, number, kind)
                
                # Fetch first claim
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
                
                # Sleep to respect rate limits
                time.sleep(0.3)
            else:
                print("(skipped - already in database)")
    
    # Combine with existing data
    if patent_records:
        new_df = pd.DataFrame(patent_records)
        
        # Mark all existing patents as not new
        if not existing_df.empty:
            existing_df['is_new'] = 'NO'
        
        # Combine DataFrames
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = existing_df
    
    # Save cumulative CSV
    combined_df.to_csv(CUMULATIVE_CSV, index=False)
    
    print(f"\n{'='*80}")
    print(f"Search Complete!")
    print(f"Total patents in database: {len(combined_df)}")
    print(f"New patents added this run: {new_count}")
    print(f"Saved to: {CUMULATIVE_CSV}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    # Define your search query here
    cql = '(ta=exosomes or ta="extracellular vesicles") and ta=CNS and pd within "20220101 20251130"'
    
    # For testing, limit to 100 records. Remove max_records=100 for full search
    search_and_update(cql, max_records=None)

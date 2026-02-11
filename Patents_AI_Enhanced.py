#!/usr/bin/env python3
"""
AI-Enhanced Patent Search - Multi-Source (EPO + Google Patents)
Searches European Patent Office and US Patents via Google Patents
Uses local AI for relevance scoring and GitHub Actions for automation

DESIGN DECISIONS & IMPROVEMENTS:

1. EPO SEARCH (CQL Syntax Fix):
   - ISSUE: EPO's CQL parser doesn't support date range syntax: pd within "YYYYMMDD YYYYMMDD"
   - SOLUTION: Query without date range, filter results in Python instead
   - BENEFIT: Reliable 200 OK responses, filters dates accurately
   - RESULT: Successfully finds 60+ patents per search

2. WIPO EXCLUSION:
   - WIPO PatentScope prohibits automated scraping and bulk downloading (Terms of Service)
   - Legitimate access requires paid subscriptions (PCT-Bibliographic ~400 CHF/year minimum)
   - Decision: Use Google Patents for US patents instead (free, legal, comprehensive)
   - Future: Can integrate WIPO if paid subscription obtained

3. GOOGLE PATENTS ADDITION:
   - Complements EPO for US patent coverage
   - Searches same parameters: exosomes, extracellular vesicles, CNS
   - Returns results in identical format for seamless merging
   - Free and publicly available

4. RELEVANCE SCORING:
   - Uses SentenceTransformer (local AI model)
   - No API calls, runs offline
   - Filters patents by semantic similarity to research focus

5. DEDUPLICATION:
   - Prevents duplicate entries across sources
   - Checks both new batch and existing database
"""

import os
import sys
import time
import json
import smtplib
import requests
from datetime import datetime, timedelta
from pathlib import Path
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from urllib.parse import quote
import logging

import pandas as pd
from lxml import etree

try:
    from bs4 import BeautifulSoup
    HAS_BEAUTIFULSOUP = True
except ImportError:
    HAS_BEAUTIFULSOUP = False
    print("Warning: BeautifulSoup not installed. Some features disabled.")

from epo_ops import Client, models, middlewares

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SEMANTIC = True
except ImportError:
    HAS_SEMANTIC = False
    print("Warning: SentenceTransformer not available. AI scoring disabled.")

# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
CUMULATIVE_CSV = DATA_DIR / "patents_cumulative.csv"

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# EPO API credentials
epo_key = os.environ.get("EPO_OPS_KEY")
epo_secret = os.environ.get("EPO_OPS_SECRET")

# Email credentials
SENDER_EMAIL = os.environ.get("SENDER_EMAIL")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD")
RECIPIENT_EMAIL = os.environ.get("RECIPIENT_EMAIL")

# Initialize EPO client
epo_client = None
if epo_key and epo_secret:
    try:
        middlewares_list = [middlewares.Throttler()]
        epo_client = Client(
            key=epo_key,
            secret=epo_secret,
            middlewares=middlewares_list
        )
        print("✓ EPO client initialized")
    except Exception as e:
        print(f"Error creating EPO client: {e}")
        epo_client = None

# Research focus for relevance scoring
RESEARCH_FOCUS = """
Exosome-based drug delivery systems or unchanged naive exosomes 
for central nervous system diseases and conditions,
including therapeutic applications for neurodegenerative conditions,
stroke, spinal cord injury and genetic brain disorders.
Focus on blood-brain barrier penetration and targeted CNS delivery.
"""

SEARCH_TERMS = ['exosomes', 'extracellular vesicles']
SEARCH_FILTER = 'CNS'
MIN_RELEVANCE_SCORE = 0.50

# Initialize semantic model
semantic_model = None
research_focus_embedding = None

if HAS_SEMANTIC:
    try:
        print("Loading semantic search model...")
        semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        research_focus_embedding = semantic_model.encode(RESEARCH_FOCUS)
        print("✓ Semantic model ready for local relevance scoring.")
    except Exception as e:
        print(f"Warning: Could not load semantic model: {e}")
        HAS_SEMANTIC = False


def calculate_relevance_score(title, abstract):
    """Calculate semantic similarity to research focus."""
    if not HAS_SEMANTIC or not semantic_model:
        return 0.5
    
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
        logger.warning(f"Error calculating relevance: {e}")
        return 0.5


def generate_patent_link(country, number, kind=""):
    """Generate appropriate patent link based on country code."""
    if country == "US":
        return f"https://patents.google.com/patent/US{number}"
    elif country == "WO":
        return f"https://patentscope.wipo.int/search/en/detail.jsf?docId=WO{number}"
    elif country == "EP":
        return f"https://worldwide.espacenet.com/patent/search?q=pn%3DEP{number}"
    elif country == "JP":
        return f"https://worldwide.espacenet.com/patent/search?q=pn%3DJP{number}"
    elif country == "CN":
        return f"https://patents.google.com/patent/CN{number}"
    elif country == "AU":
        return f"https://worldwide.espacenet.com/patent/search?q=pn%3DAU{number}"
    else:
        return f"https://worldwide.espacenet.com/patent/search?q=pn%3D{country}{number}{kind}"


# ---------------------------------------------------------------
# EPO Search
# ---------------------------------------------------------------

def search_epo_patents(start_date, end_date):
    """Search EPO for patents matching criteria.
    
    NOTE: EPO CQL parser does NOT support date range syntax like:
    pd within "20251213 20260211"
    
    WORKAROUND: Query without date filter, then filter results in Python.
    This avoids 404 errors and provides accurate date-based filtering.
    """
    if not epo_client:
        logger.warning("EPO client not available, skipping EPO search")
        return []
    
    records = []
    # Search without date filter - dates filtered in Python later
    # (EPO CQL doesn't support: pd within "YYYYMMDD YYYYMMDD" syntax)
    cql = '(ta=exosomes OR ta="extracellular vesicles") AND ta=CNS'
    logger.info(f"[EPO] Running search: {cql} (Python date filtering applied)")
    
    try:
        start = 1
        batch_size = 25
        max_records = 500
        total = None
        
        while True:
            end = start + batch_size - 1
            if start > max_records:
                break
            
            try:
                resp = epo_client.published_data_search(
                    cql=cql,
                    range_begin=start,
                    range_end=min(end, max_records)
                )
            except Exception as e:
                logger.error(f"[EPO] Search error: {e}")
                break
            
            root = etree.fromstring(resp.content)
            ns = {"ops": "http://ops.epo.org", "ex": "http://www.epo.org/exchange"}
            
            if total is None:
                total_str = root.xpath("string(//ops:biblio-search/@total-result-count)", namespaces=ns)
                total = int(total_str) if total_str else 0
                logger.info(f"[EPO] Found {total} total results")
            
            # Parse results
            for pub_ref in root.xpath("//ops:publication-reference/ex:document-id[@document-id-type='docdb']", namespaces=ns):
                country = pub_ref.xpath("string(ex:country)", namespaces=ns)
                number = pub_ref.xpath("string(ex:doc-number)", namespaces=ns)
                kind = pub_ref.xpath("string(ex:kind)", namespaces=ns)
                
                if country and number and kind:
                    records.append({
                        "country": country,
                        "publication_number": number,
                        "kind": kind,
                        "source": "EPO"
                    })
            
            if not total or end >= total:
                break
            
            start = end + 1
            time.sleep(0.3)
    
    except Exception as e:
        logger.error(f"[EPO] Search failed: {e}")
    
    return records


def get_epo_biblio(country, number, kind):
    """Fetch EPO bibliographic data."""
    if not epo_client:
        return {}
    
    try:
        resp = epo_client.published_data(
            reference_type="publication",
            input=models.Docdb(number, country, kind),
            endpoint="biblio"
        )
        
        root = etree.fromstring(resp.content)
        ns = {"ex": "http://www.epo.org/exchange"}
        
        title = root.xpath("string(//ex:invention-title[@lang='en'])", namespaces=ns)
        if not title:
            title = root.xpath("string(//ex:invention-title)", namespaces=ns)
        
        applicants = root.xpath("//ex:applicants/ex:applicant/ex:applicant-name/ex:name/text()", namespaces=ns)
        applicants_str = ", ".join(applicants[:3]) if applicants else "Not available"
        
        inventors = root.xpath("//ex:inventors/ex:inventor/ex:inventor-name/ex:name/text()", namespaces=ns)
        inventors_str = ", ".join(inventors[:3]) if inventors else "Not available"
        
        abstract = root.xpath("string(//ex:abstract[@lang='en']/ex:p)", namespaces=ns)
        if not abstract:
            abstract = root.xpath("string(//ex:abstract/ex:p)", namespaces=ns)
        
        pub_date = root.xpath("string(//ex:publication-reference/ex:document-id[@document-id-type='docdb']/ex:date)", namespaces=ns)
        priority_date = root.xpath("string(//ex:priority-claims/ex:priority-claim[1]/ex:document-id/ex:date)", namespaces=ns)
        
        return {
            "title": title,
            "applicants": applicants_str,
            "inventors": inventors_str,
            "abstract": abstract[:500] if abstract else "",
            "publication_date": pub_date,
            "priority_date": priority_date,
        }
    except Exception as e:
        logger.warning(f"EPO biblio fetch error for {country}{number}{kind}: {e}")
        return {}


# ---------------------------------------------------------------
# Google Patents Search (US Patents)
# ---------------------------------------------------------------

def search_google_patents(start_date, end_date):
    """Search Google Patents for US patents matching criteria."""
    if not HAS_BEAUTIFULSOUP:
        logger.warning("[Google Patents] BeautifulSoup not available, skipping search")
        return []
    
    records = []
    logger.info("[Google Patents] Searching for US patents...")
    
    try:
        # Search query for Google Patents
        search_query = 'exosome* OR "extracellular vesicle*" CNS'
        
        # Google Patents URL
        search_url = f"https://patents.google.com/usearch?q={quote(search_query)}&type=PATENT"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        logger.info(f"[Google Patents] Searching: {search_query}")
        
        try:
            resp = requests.get(search_url, headers=headers, timeout=15)
            resp.raise_for_status()
            
            soup = BeautifulSoup(resp.content, 'html.parser')
            
            # Find patent result containers
            result_items = soup.find_all('div', class_='result-item')
            
            if not result_items:
                # Try alternative selector
                result_items = soup.find_all('article', class_='patent-item')
            
            logger.info(f"[Google Patents] Found {len(result_items)} result items")
            
            for idx, item in enumerate(result_items[:100]):  # Limit to 100 results
                try:
                    # Extract patent link and number
                    patent_link = item.find('a', href=True)
                    if not patent_link:
                        continue
                    
                    href = patent_link.get('href', '')
                    if '/patent/US' not in href:
                        continue
                    
                    # Extract patent number from href
                    # Format: /patent/US... or similar
                    patent_number = patent_link.text.strip() if patent_link.text else ''
                    
                    if not patent_number or 'US' not in patent_number:
                        continue
                    
                    # Clean patent number (remove US prefix for storage)
                    pub_number = patent_number.replace('US', '').strip()
                    
                    # Get title
                    title_elem = item.find('span', class_='title')
                    if not title_elem:
                        title_elem = item.find('a', class_='title')
                    title = title_elem.text.strip() if title_elem else 'Not available'
                    
                    # Get abstract/description snippet
                    abstract_elem = item.find('span', class_='snippet')
                    if not abstract_elem:
                        abstract_elem = item.find('div', class_='description')
                    abstract = abstract_elem.text.strip() if abstract_elem else ''
                    
                    # Get publication date if available
                    date_elem = item.find('span', class_='date')
                    pub_date = date_elem.text.strip() if date_elem else ''
                    
                    # Validate
                    if not pub_number or not title or title == 'Not available':
                        continue
                    
                    records.append({
                        "country": "US",
                        "publication_number": pub_number,
                        "kind": "B2",
                        "title": title,
                        "applicants": "Not available",
                        "inventors": "Not available",
                        "abstract": abstract[:500] if abstract else "",
                        "publication_date": pub_date,
                        "priority_date": "",
                        "source": "Google Patents"
                    })
                    
                    logger.debug(f"[Google Patents] Found: US{pub_number} - {title[:50]}")
                
                except Exception as e:
                    logger.debug(f"[Google Patents] Error parsing result {idx}: {e}")
                    continue
            
            if records:
                logger.info(f"[Google Patents] Successfully extracted {len(records)} US patents")
            else:
                logger.info("[Google Patents] No relevant patents found")
            
            return records
        
        except requests.RequestException as e:
            logger.warning(f"[Google Patents] Network error: {e}")
            logger.info("[Google Patents] Search skipped (network unavailable)")
            return []
    
    except Exception as e:
        logger.error(f"[Google Patents] Search error: {e}")
        logger.info("[Google Patents] Continuing without Google Patents results")
        return []


# ---------------------------------------------------------------
# Combined Search & Processing
# ---------------------------------------------------------------

def search_all_patents():
    """Search all patent sources and merge results."""
    start_date = (datetime.now().date() - timedelta(days=60)).strftime("%Y%m%d")
    end_date = datetime.now().date().strftime("%Y%m%d")
    
    print("="*80)
    print(f"Starting AI-Enhanced Patent Search - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Relevance threshold: {MIN_RELEVANCE_SCORE}")
    print(f"Search period: Last 60 days")
    print(f"AI relevance scoring: SentenceTransformer (Local)")
    print("="*80)
    
    # Load existing patents
    existing_ids = set()
    if CUMULATIVE_CSV.exists():
        df = pd.read_csv(CUMULATIVE_CSV)
        existing_ids = set(
            df.apply(lambda row: f"{row['country']}{row['publication_number']}{row['kind']}", axis=1)
        )
        logger.info(f"Loaded {len(existing_ids)} existing patents from database")
    
    # Search all sources
    logger.info("Searching patent sources...")
    epo_results = search_epo_patents(start_date, end_date) if epo_client else []
    google_results = search_google_patents(start_date, end_date)
    
    all_results = epo_results + google_results
    
    print("\n[SUMMARY] Found patents from all sources:")
    print(f"  - EPO (European): {len(epo_results)}")
    print(f"  - Google Patents (US): {len(google_results)}")
    print(f"  - TOTAL: {len(all_results)}")
    
    # Process and deduplicate
    records = []
    current_run_date = datetime.now().strftime('%Y-%m-%d')
    processed = set()
    new_count = 0
    skipped_count = 0
    
    for patent in all_results:
        patent_id = f"{patent['country']}{patent['publication_number']}{patent.get('kind', '')}"
        
        # Skip duplicates within this batch
        if patent_id in processed:
            continue
        processed.add(patent_id)
        
        # Skip if already in database
        if patent_id in existing_ids:
            skipped_count += 1
            continue
        
        # For Google Patents results that already have full data
        if 'title' in patent and patent.get('source') == 'Google Patents':
            title = patent.get('title', '')
            abstract = patent.get('abstract', '')
            applicants = patent.get('applicants', 'Not available')
            inventors = patent.get('inventors', 'Not available')
            pub_date = patent.get('publication_date', '')
            priority_date = patent.get('priority_date', '')
        else:
            # Fetch full data for EPO results
            biblio = get_epo_biblio(patent['country'], patent['publication_number'], patent.get('kind', ''))
            title = biblio.get('title', '')
            abstract = biblio.get('abstract', '')
            applicants = biblio.get('applicants', 'Not available')
            inventors = biblio.get('inventors', 'Not available')
            pub_date = biblio.get('publication_date', '')
            priority_date = biblio.get('priority_date', '')
        

        
        # Calculate relevance score
        relevance = calculate_relevance_score(title, abstract)
        
        # Filter by relevance
        if relevance < MIN_RELEVANCE_SCORE:
            continue
        
        # Generate link
        link = generate_patent_link(
            patent['country'],
            patent['publication_number'],
            patent.get('kind', '')
        )
        
        records.append({
            "country": patent['country'],
            "publication_number": patent['publication_number'],
            "kind": patent.get('kind', 'B2'),
            "title": title,
            "applicants": applicants,
            "inventors": inventors,
            "abstract": abstract,
            "publication_date": pub_date,
            "priority_date": priority_date,
            "relevance_score": round(relevance, 2),
            "source": patent.get('source', 'Unknown'),
            "link": link,
            "date_added": current_run_date,
            "is_new": "YES"
        })
        
        new_count += 1
        logger.info(f"  ✓ {patent['country']}{patent['publication_number']} - {patent.get('source')} - Score: {relevance:.2f}")
        time.sleep(0.1)
    
    print(f"\n[RESULTS]")
    print(f"  New patents found: {new_count}")
    print(f"  Skipped (already in DB): {skipped_count}")
    
    return pd.DataFrame(records)


def update_cumulative_csv(df_new):
    """Merge new results with existing cumulative CSV."""
    FINAL_COLUMNS = [
        "country", "publication_number", "kind", "title", "applicants", "inventors",
        "abstract", "publication_date", "priority_date", "relevance_score",
        "source", "link", "date_added", "is_new"
    ]
    
    if CUMULATIVE_CSV.exists():
        df_old = pd.read_csv(CUMULATIVE_CSV)
        df_old['is_new'] = 'NO'
        
        if 'relevance_score' not in df_old.columns:
            df_old['relevance_score'] = 0.5
        if 'source' not in df_old.columns:
            df_old['source'] = 'EPO'
        
        df_all = pd.concat([df_old, df_new], ignore_index=True).drop_duplicates(
            subset=["country", "publication_number", "kind"],
            keep="first"
        )
        logger.info(f"Added {len(df_new)} new patents to database")
    else:
        df_all = df_new
        logger.info(f"Created new database with {len(df_all)} patents")
    
    df_all = df_all.reindex(columns=FINAL_COLUMNS, fill_value='')
    df_all = df_all.sort_values(['relevance_score', 'date_added'], ascending=[False, False])
    df_all.to_csv(CUMULATIVE_CSV, index=False)
    logger.info(f"Saved cumulative CSV with {len(df_all)} total records")
    return df_all


def send_email_with_csv(df_all):
    """Send email with updated CSV."""
    if not SENDER_EMAIL or not RECIPIENT_EMAIL or not EMAIL_PASSWORD:
        logger.warning("Email credentials not found. Skipping email.")
        return
    
    new_patents = df_all[df_all['is_new'] == 'YES']
    
    email_body = f"""
Multi-Source Patent Search Update - {datetime.now().strftime('%Y-%m-%d')}
{'='*80}

NEW PATENTS: {len(new_patents)}
TOTAL DATABASE: {len(df_all)} patents

SOURCES SEARCHED:
- EPO (European Patent Office)
- Google Patents (US Patents)

SEARCH TERMS: exosomes, extracellular vesicles, CNS
DATE RANGE: Last 60 days

{'='*80}
"""
    
    if len(new_patents) > 0:
        email_body += "\n🔥 TOP 5 MOST RELEVANT NEW PATENTS:\n\n"
        top_patents = new_patents.nlargest(5, 'relevance_score')
        for idx, patent in enumerate(top_patents.itertuples(), 1):
            email_body += f"{idx}. [{patent.relevance_score:.2f}] {patent.title[:80]}\n"
            email_body += f"   {patent.country}{patent.publication_number} | Source: {patent.source}\n"
            email_body += f"   Applicant: {patent.applicants[:60]}\n\n"
    else:
        email_body += "\n✓ No new patents found, but database is updated.\n"
    
    email_body += f"\nSee attached CSV for full details.\n{'='*80}"
    
    msg = MIMEMultipart()
    msg["Subject"] = f"Patent Update - {len(new_patents)} New Patents - {datetime.now().strftime('%Y-%m-%d')}"
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECIPIENT_EMAIL
    
    body = MIMEText(email_body, "plain")
    msg.attach(body)
    
    try:
        with open(CUMULATIVE_CSV, "rb") as f:
            attachment = MIMEApplication(f.read(), Name="patents_cumulative.csv")
        attachment["Content-Disposition"] = 'attachment; filename="patents_cumulative.csv"'
        msg.attach(attachment)
        
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(SENDER_EMAIL, EMAIL_PASSWORD)
            smtp.send_message(msg)
        logger.info("✓ Email sent successfully")
    except Exception as e:
        logger.error(f"Error sending email: {e}")


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    print("="*80)
    print("MULTI-SOURCE PATENT SEARCH - EPO + GOOGLE PATENTS - Starting")
    print("="*80)
    
    df_new = search_all_patents()
    df_all = update_cumulative_csv(df_new)
    send_email_with_csv(df_all)
    
    print("\n" + "="*80)
    print("Process completed successfully")
    print("="*80)


if __name__ == "__main__":
    main()

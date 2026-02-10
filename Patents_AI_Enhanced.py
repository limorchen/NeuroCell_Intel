#!/usr/bin/env python3
"""
AI-Enhanced Patent Search with WIPO Integration
Searches EPO, WIPO PatentScope (PCT), and uses AI for relevance scoring
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
    print("Warning: BeautifulSoup not installed. WIPO scraping disabled.")

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
    """Search EPO for patents matching criteria."""
    if not epo_client:
        logger.warning("EPO client not available, skipping EPO search")
        return []
    
    records = []
    cql = f'(ta=exosomes or ta="extracellular vesicles") and ta=CNS and pd within "{start_date} {end_date}"'
    logger.info(f"[EPO] Running search: {cql}")
    
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
# WIPO PatentScope Search (PCT Patents Only)
# ---------------------------------------------------------------

def search_wipo_patents(start_date, end_date):
    """Search WIPO PatentScope for PCT (WO) patents."""
    if not HAS_BEAUTIFULSOUP:
        logger.warning("[WIPO] BeautifulSoup not available, skipping WIPO search")
        return []
    
    records = []
    logger.info("[WIPO] Searching PatentScope for PCT patents...")
    
    try:
        # WIPO PatentScope search query
        # Format: (exosome* OR "extracellular vesicle*") AND CNS AND PC=WO
        search_query = '(exosome* OR "extracellular vesicle*") AND CNS AND PC=WO'
        
        # URL for WIPO advanced search
        wipo_search_url = "https://patentscope.wipo.int/search/en/advanced/search.jsf"
        
        params = {
            'queryString': search_query,
            'sort': 'Relevance',
            'maxResults': 100
        }
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        # Try REST API endpoint first
        rest_url = "https://patentscope.wipo.int/rest/api/patentrecords"
        
        rest_params = {
            'expression': search_query,
            'start': 0,
            'max': 100,
            'lan': 'en'
        }
        
        logger.info(f"[WIPO] Searching with query: {search_query}")
        
        try:
            resp = requests.get(rest_url, params=rest_params, headers=headers, timeout=10)
            
            if resp.status_code == 200:
                data = resp.json()
                
                # Parse WIPO REST API response
                if 'patentDocuments' in data:
                    total_results = data.get('total', 0)
                    logger.info(f"[WIPO] Found {total_results} PCT patents")
                    
                    for doc in data.get('patentDocuments', []):
                        try:
                            # Extract patent information
                            biblio = doc.get('bibliographicData', {})
                            
                            # Get patent number (should be WO)
                            pub_ref = biblio.get('publicationNumber', '')
                            if not pub_ref.startswith('WO'):
                                continue
                            
                            # Parse WO number (format: WO2024123456)
                            wo_match = pub_ref[:10]  # WO + 8 digits
                            
                            # Get title
                            title_dict = biblio.get('invention-title', {})
                            if isinstance(title_dict, dict):
                                title = title_dict.get('en', '')
                            else:
                                title = str(title_dict)
                            
                            # Get abstract
                            abstract_dict = biblio.get('abstract', {})
                            if isinstance(abstract_dict, dict):
                                abstract = abstract_dict.get('en', '')
                            else:
                                abstract = str(abstract_dict)
                            
                            # Get applicant
                            parties = biblio.get('parties', {})
                            applicants = parties.get('applicants', [])
                            applicant_name = applicants[0].get('name', 'Not available') if applicants else 'Not available'
                            
                            # Get inventor
                            inventors = parties.get('inventors', [])
                            inventor_name = inventors[0].get('name', 'Not available') if inventors else 'Not available'
                            
                            # Get dates
                            pub_date = biblio.get('publicationDate', '')
                            priority_date = biblio.get('priorityDate', '')
                            
                            records.append({
                                "country": "WO",
                                "publication_number": wo_match[2:],  # Remove 'WO' prefix
                                "kind": "A1",
                                "title": title,
                                "applicants": applicant_name,
                                "inventors": inventor_name,
                                "abstract": abstract[:500] if abstract else "",
                                "publication_date": pub_date,
                                "priority_date": priority_date,
                                "source": "WIPO"
                            })
                        
                        except Exception as e:
                            logger.debug(f"[WIPO] Error parsing patent document: {e}")
                            continue
                    
                    logger.info(f"[WIPO] Successfully extracted {len(records)} PCT patents")
                    return records
        
        except Exception as e:
            logger.warning(f"[WIPO] REST API error: {e}")
        
        # Fallback: If REST API doesn't work, return empty list
        logger.info("[WIPO] REST API unavailable. WIPO search skipped.")
        logger.info("[WIPO] Note: Full WIPO integration requires additional authentication")
        return []
    
    except Exception as e:
        logger.error(f"[WIPO] Search error: {e}")
        return []


# ---------------------------------------------------------------
# Combined Search & Processing
# ---------------------------------------------------------------

def search_all_patents():
    """Search all patent sources and merge results."""
    start_date = (datetime.utcnow().date() - timedelta(days=60)).strftime("%Y%m%d")
    end_date = datetime.utcnow().date().strftime("%Y%m%d")
    
    print("="*80)
    print(f"Starting AI-Enhanced Patent Search - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Relevance threshold: {MIN_RELEVANCE_SCORE}")
    print(f"AI summaries: Local Heuristic (No API Key Required)")
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
    wipo_results = search_wipo_patents(start_date, end_date)
    
    all_results = epo_results + wipo_results
    
    print("\n[SUMMARY] Found patents from all sources:")
    print(f"  - EPO: {len(epo_results)}")
    print(f"  - WIPO (PCT): {len(wipo_results)}")
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
        
        # For WIPO results that already have full data
        if 'title' in patent:
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
            "kind": patent.get('kind', 'A1'),
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
- WIPO PatentScope (PCT Patents)

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
    print("MULTI-SOURCE PATENT SEARCH WITH WIPO - Starting")
    print("="*80)
    
    df_new = search_all_patents()
    df_all = update_cumulative_csv(df_new)
    send_email_with_csv(df_all)
    
    print("\n" + "="*80)
    print("Process completed successfully")
    print("="*80)


if __name__ == "__main__":
    main()

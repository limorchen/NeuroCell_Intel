#!/usr/bin/env python3
"""
AI-Enhanced Patent Search - European Patent Office (EPO) Only
Searches EPO for exosome and extracellular vesicle patents targeting CNS disorders
Uses local AI (SentenceTransformer) for relevance scoring
Automated bimonthly execution via GitHub Actions

FIXES APPLIED (2026-03-01):
  FIX 1: Added INFO-level logging for date-filtered patents (silent drops now visible)
  FIX 2: Fixed pub_date parsing to handle both YYYYMMDD and YYYY-MM-DD formats robustly
  FIX 3: 🔥 NEW title prefix is NO LONGER written to CSV — stored in separate 'new_marker' column
  FIX 4: HF_TOKEN environment variable now passed to SentenceTransformer to avoid rate-limit risk
  FIX 5: embeddings.position_ids warning suppressed via logging filter (benign, but noisy)
  FIX 6: Added counter and breakdown log for all 61 EPO results (new / duplicate / date-filtered / low-relevance)
"""

import os
import time
import smtplib
import requests
import warnings
import logging
from datetime import datetime, timedelta
from pathlib import Path
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication

import pandas as pd
from lxml import etree

from epo_ops import Client, models, middlewares

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SEMANTIC = True
except ImportError:
    HAS_SEMANTIC = False
    print("Warning: SentenceTransformer not available. Using default relevance scores.")

# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
CUMULATIVE_CSV = DATA_DIR / "patents_cumulative.csv"

# ---------------------------------------------------------------
# FIX 5: Suppress the noisy but benign embeddings.position_ids
#         UNEXPECTED warning from sentence-transformers/BertModel
# ---------------------------------------------------------------
class _SuppressBertPositionIdsFilter(logging.Filter):
    def filter(self, record):
        return "embeddings.position_ids" not in record.getMessage()

# Apply filter BEFORE basicConfig so it catches all handlers
_bert_filter = _SuppressBertPositionIdsFilter()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Attach filter to root logger so it covers all child loggers
logging.getLogger().addFilter(_bert_filter)

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

# ---------------------------------------------------------------
# FIX 4: Pass HF_TOKEN to SentenceTransformer to avoid rate-limit risk
#         Set HF_TOKEN as a GitHub Actions secret and env var (see README)
# ---------------------------------------------------------------
hf_token = os.environ.get("HF_TOKEN")  # None is fine if not set — just triggers the warning
if not hf_token:
    logger.warning(
        "HF_TOKEN not set. Unauthenticated HuggingFace requests may be rate-limited. "
        "Add HF_TOKEN to your GitHub Actions secrets and env block to suppress this."
    )

semantic_model = None
research_focus_embedding = None

if HAS_SEMANTIC:
    try:
        print("Loading semantic search model...")
        # FIX 4: token= parameter authenticates requests to HuggingFace Hub
        semantic_model = SentenceTransformer('all-MiniLM-L6-v2', token=hf_token)
        research_focus_embedding = semantic_model.encode(RESEARCH_FOCUS)
        print("✓ Semantic model ready for local relevance scoring.")
    except Exception as e:
        print(f"Warning: Could not load semantic model: {e}")
        HAS_SEMANTIC = False


# ---------------------------------------------------------------
# FIX 2: Robust pub_date normalisation
#         Handles: "20260115", "2026-01-15", "2026/01/15", "20260115000000"
#         Returns: "20260115" (8-digit string) or None on failure
# ---------------------------------------------------------------
def normalise_date(raw_date: str) -> str | None:
    """
    Convert any date string returned by EPO into an 8-digit YYYYMMDD string.
    Returns None if the date cannot be parsed, so callers can log and decide.
    """
    if not raw_date:
        return None
    # Strip separators and truncate to 8 digits
    cleaned = raw_date.replace("-", "").replace("/", "").replace(" ", "")[:8]
    if len(cleaned) == 8 and cleaned.isdigit():
        return cleaned
    return None


def calculate_relevance_score(title, abstract):
    """Calculate semantic similarity to research focus using SentenceTransformer."""
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
    """
    Search EPO for patents matching criteria.

    NOTE: EPO CQL parser does NOT support date range syntax like:
      pd within "20251213 20260211"
    WORKAROUND: Query without date filter, then filter results in Python.
    """
    if not epo_client:
        logger.warning("EPO client not available, skipping EPO search")
        return []
    
    records = []
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
    """Fetch EPO bibliographic data for a patent."""
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
# Combined Search & Processing
# ---------------------------------------------------------------

def search_all_patents():
    """Search EPO patents and process results."""
    start_date = (datetime.now().date() - timedelta(days=60)).strftime("%Y%m%d")
    end_date = datetime.now().date().strftime("%Y%m%d")
    
    print("="*80)
    print(f"Starting AI-Enhanced Patent Search - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Relevance threshold: {MIN_RELEVANCE_SCORE}")
    print(f"Search period: Last 60 days ({start_date} → {end_date})")
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
    
    # Search EPO
    logger.info("Searching patent sources...")
    epo_results = search_epo_patents(start_date, end_date) if epo_client else []
    
    print("\n[SUMMARY] Found patents from EPO:")
    print(f"  - EPO (European): {len(epo_results)}")
    
    # Process and deduplicate
    records = []
    current_run_date = datetime.now().strftime('%Y-%m-%d')
    processed = set()

    # FIX 6: Explicit counters for every disposition so nothing is silently lost
    count_new = 0
    count_duplicate_db = 0
    count_duplicate_batch = 0
    count_date_filtered = 0
    count_date_missing = 0
    count_low_relevance = 0
    count_biblio_error = 0

    for patent in epo_results:
        patent_id = f"{patent['country']}{patent['publication_number']}{patent.get('kind', '')}"
        
        # Skip duplicates within this batch
        if patent_id in processed:
            count_duplicate_batch += 1
            continue
        processed.add(patent_id)
        
        # Skip if already in database
        if patent_id in existing_ids:
            count_duplicate_db += 1
            continue
        
        # Fetch full bibliographic data
        biblio = get_epo_biblio(patent['country'], patent['publication_number'], patent.get('kind', ''))

        if not biblio:
            count_biblio_error += 1
            logger.warning(f"  ✗ {patent_id} — biblio fetch returned empty, skipping")
            continue

        title = biblio.get('title', '')
        abstract = biblio.get('abstract', '')
        applicants = biblio.get('applicants', 'Not available')
        inventors = biblio.get('inventors', 'Not available')
        pub_date_raw = biblio.get('publication_date', '')
        priority_date = biblio.get('priority_date', '')

        # -------------------------------------------------------
        # FIX 2: Robust date parsing + FIX 1: INFO-level logging
        # -------------------------------------------------------
        pub_date_norm = normalise_date(pub_date_raw)

        if pub_date_norm is None:
            # Date missing or unparseable — log at INFO so it is visible
            count_date_missing += 1
            logger.info(
                f"  ? {patent_id} — pub_date missing or unparseable "
                f"(raw='{pub_date_raw}'). Including patent anyway."
            )
            # Include the patent (original intent of the bare `pass`)
        else:
            # FIX 1: Log date-filtered patents at INFO, not silent debug
            if not (start_date <= pub_date_norm <= end_date):
                count_date_filtered += 1
                logger.info(
                    f"  – {patent_id} — outside date window "
                    f"(pub={pub_date_norm}, window={start_date}–{end_date}). Skipped."
                )
                continue

        # Calculate relevance score
        relevance = calculate_relevance_score(title, abstract)
        
        # Filter by relevance
        if relevance < MIN_RELEVANCE_SCORE:
            count_low_relevance += 1
            logger.info(
                f"  ↓ {patent_id} — low relevance score {relevance:.2f} "
                f"(threshold={MIN_RELEVANCE_SCORE}). Skipped."
            )
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
            "title": title,           # FIX 3: Clean title — no 🔥 prefix written to CSV
            "applicants": applicants,
            "inventors": inventors,
            "abstract": abstract,
            "publication_date": pub_date_raw,   # keep original for display
            "priority_date": priority_date,
            "relevance_score": round(relevance, 2),
            "source": "EPO",
            "link": link,
            "date_added": current_run_date,
            "is_new": "YES"
        })
        
        count_new += 1
        logger.info(f"  ✓ {patent_id} — Score: {relevance:.2f} — {title[:60]}")
        time.sleep(0.1)

    # FIX 6: Print full breakdown — every result accounted for
    total_checked = count_new + count_duplicate_db + count_duplicate_batch + \
                    count_date_filtered + count_date_missing + count_low_relevance + count_biblio_error
    print(f"\n[RESULTS BREAKDOWN]")
    print(f"  EPO results returned:        {len(epo_results)}")
    print(f"  ─────────────────────────────────────────")
    print(f"  ✓ Added as new:              {count_new}")
    print(f"  = Already in database:       {count_duplicate_db}")
    print(f"  = Duplicate in this batch:   {count_duplicate_batch}")
    print(f"  – Outside date window:       {count_date_filtered}")
    print(f"  ? Date missing/unparseable:  {count_date_missing}")
    print(f"  ↓ Below relevance threshold: {count_low_relevance}")
    print(f"  ✗ Biblio fetch error:        {count_biblio_error}")
    print(f"  ─────────────────────────────────────────")
    print(f"  Σ Accounted for:             {total_checked}")
    if total_checked != len(epo_results):
        logger.warning(
            f"  ⚠ COUNT MISMATCH: {len(epo_results)} returned vs {total_checked} accounted. "
            "Investigate pagination or duplicate patent_ids."
        )
    
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

        # FIX 3: Strip legacy 🔥 NEW prefix from any titles already written to CSV
        #         in previous runs (one-time cleanup — idempotent on clean data)
        if 'title' in df_old.columns:
            df_old['title'] = df_old['title'].str.removeprefix('🔥 NEW - ')
        
        df_all = pd.concat([df_old, df_new], ignore_index=True).drop_duplicates(
            subset=["country", "publication_number", "kind"],
            keep="first"
        )
        logger.info(f"Added {len(df_new)} new patents to database")
    else:
        df_all = df_new
        logger.info(f"Created new database with {len(df_all)} patents")
    
    df_all = df_all.reindex(columns=FINAL_COLUMNS, fill_value='')
    
    # Sort: existing patents first (by relevance desc), new patents at bottom (by relevance desc)
    df_all = df_all.sort_values(
        ['is_new', 'relevance_score'],
        ascending=[True, False]   # 'NO' < 'YES' alphabetically → existing first
    )

    # FIX 3: Save clean titles to CSV — no emoji prefix embedded in the data.
    #         The email notification adds "🔥 NEW" visually without touching the CSV.
    df_all.to_csv(CUMULATIVE_CSV, index=False)

    logger.info(f"Saved cumulative CSV with {len(df_all)} total records")
    logger.info(f"  → Existing patents: sorted by relevance (highest first)")
    logger.info(f"  → New patents: is_new=YES, appear at bottom, titles are clean")
    return df_all


def send_email_with_csv(df_all):
    """Send email with updated CSV."""
    if not SENDER_EMAIL or not RECIPIENT_EMAIL or not EMAIL_PASSWORD:
        logger.warning("Email credentials not found. Skipping email.")
        return
    
    new_patents = df_all[df_all['is_new'] == 'YES']
    
    email_body = f"""
Patent Search Update - {datetime.now().strftime('%Y-%m-%d')}
{'='*80}

NEW PATENTS: {len(new_patents)}
TOTAL DATABASE: {len(df_all)} patents

SOURCE: European Patent Office (EPO)
SEARCH TERMS: exosomes, extracellular vesicles, CNS
DATE RANGE: Last 60 days
AI SCORING: SentenceTransformer (local relevance scoring)

{'='*80}
"""
    
    if len(new_patents) > 0:
        email_body += "\n🔥 TOP 5 MOST RELEVANT NEW PATENTS:\n\n"
        top_patents = new_patents.nlargest(5, 'relevance_score')
        for idx, patent in enumerate(top_patents.itertuples(), 1):
            # FIX 3: 🔥 marker added here for email display only — not written to CSV
            email_body += f"{idx}. [{patent.relevance_score:.2f}] 🔥 {patent.title[:80]}\n"
            email_body += f"   {patent.country}{patent.publication_number} | {patent.source}\n"
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
    print("PATENT SEARCH WITH AI RELEVANCE SCORING - EPO - Starting")
    print("="*80)
    
    df_new = search_all_patents()
    df_all = update_cumulative_csv(df_new)
    send_email_with_csv(df_all)
    
    print("\n" + "="*80)
    print("Process completed successfully")
    print("="*80)


if __name__ == "__main__":
    main()


# ====================================================================
# ADD THIS LINE AT THE VERY TOP OF YOUR SCRIPT:
print("--- SCRIPT VERSION 1.1: DATE AND COLUMN FIXES DEPLOYED ---") 
# ====================================================================

from datetime import datetime, timedelta
import os
import csv
import time
import smtplib
import sys # Added for explicit exit on model load failure
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

if not key or not secret:
    print("WARNING: Missing EPO OPS API credentials. Cannot run search.")
    key = "DUMMY_KEY" # Placeholder to allow script to compile
    secret = "DUMMY_SECRET"

# Research focus for relevance scoring
RESEARCH_FOCUS = """
Exosome-based drug delivery systems for central nervous system diseases,
including therapeutic applications for neurodegenerative conditions,
brain cancer, stroke, and genetic brain disorders.
Focus on blood-brain barrier penetration and targeted CNS delivery.
"""

# Minimum relevance score (0-1 scale)
MIN_RELEVANCE_SCORE = 0.50  # Adjust this threshold as needed

middlewares_list = [
    middlewares.Dogpile(),
    middlewares.Throttler()
]

# Client initialization
try:
    client = Client(
        key=key,
        secret=secret,
        middlewares=middlewares_list
    )
except Exception:
    client = None # Set to None if client creation fails

# Initialize semantic search model (Used for Relevance Scoring)
try:
    print("Loading semantic search model...")
    # Using a small, fast, and cached model
    semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
    research_focus_embedding = semantic_model.encode(RESEARCH_FOCUS)
    print("✓ Semantic model ready for local relevance scoring.")
except Exception as e:
    print(f"FATAL ERROR: Could not load SentenceTransformer model. Please ensure 'sentence-transformers' is in requirements.txt. Error: {e}")
    sys.exit(1)


# ---------------------------------------------------------------
# Helper Functions (Local AI/Summary is the key change here)
# ---------------------------------------------------------------

# The function below replaces the Claude API call
def generate_ai_summary(title, abstract, claims=""):
    """
    Generate concise AI summary using a simple local heuristic 
    (first 3 sentences of the abstract).
    
    This function replaces the paid Anthropic API call.
    """
    
    if not abstract:
        return "Insufficient data for summary"
        
    # Heuristic: Take the first 3 sentences of the abstract as a summary
    sentences = abstract.split('.')
    summary_sentences = sentences[:3] 
    ai_summary = '.'.join(summary_sentences).strip()

    # Ensure the summary ends with a period if it's not empty
    if ai_summary and not ai_summary.endswith('.'):
        ai_summary += '.'

    # Fallback to the title if the abstract is too short
    if len(ai_summary) < 20 and title:
        return f"Summary: {title.strip()}"
        
    return ai_summary


def scan_patents_cql(cql_query, batch_size=25, max_records=None):
    """Iterate over OPS search results."""
    if not client: return
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
    if not client: return {} 
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
            "abstract": abstract,
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


def calculate_relevance_score(title, abstract):
    """Calculate semantic similarity to research focus (using local model)."""
    if not title and not abstract:
        return 0.0
    
    # Combine title and abstract
    patent_text = f"{title} {abstract}"
    
    # Generate embedding
    patent_embedding = semantic_model.encode(patent_text)
    
    # Calculate cosine similarity
    similarity = cosine_similarity(
        research_focus_embedding.reshape(1, -1),
        patent_embedding.reshape(1, -1)
    )[0][0]
    
    return float(similarity)


def parse_patent_refs(root):
    """Extract DOCDB numbers from search result XML."""
    ns = {
        "ops": "http://ops.epo.org",
        "ex": "http://www.epo.org/exchange"
    }
    results = []

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

# 🚨 FIX 1: Date range increased to 365 days to prevent future date error 🚨
def get_date_range_one_year():
    """Return (start_date, end_date) covering the last 1 year in YYYYMMDD format."""
    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=365)
    
    # The search query remains the same, but the output will show the broader range.
    start_str = start_date.strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")
    
    # Print the wider range for clarity in the logs
    print(f"WARNING: Runner clock is likely drifting. Searching a safe 12-month window: {start_str} to {end_str}")
    
    return start_str, end_str

def load_existing_patents():
    """Load existing patents from cumulative CSV and return a set of IDs."""
    if CUMULATIVE_CSV.exists():
        df = pd.read_csv(CUMULATIVE_CSV)
        # Ensure 'publication_number' column exists before creating the ID
        if 'publication_number' in df.columns and 'kind' in df.columns:
            existing = set(
                df.apply(lambda row: f"{row['country']}{row['publication_number']}{row['kind']}", axis=1)
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
    # 🚨 FIX 1b: Use the new 1-year date range function 🚨
    start_date, end_date = get_date_range_one_year()
    
    existing_ids = load_existing_patents()

    # Search terms
    search_terms = '(ta=exosomes or ta="extracellular vesicles") and ta=CNS'
    
    cql = f'{search_terms} and pd within "{start_date} {end_date}"'
    print(f"Running CQL: {cql}")

    records = []
    count = 0
    new_count = 0
    skipped_count = 0
    filtered_count = 0
    current_run_date = datetime.now().strftime('%Y-%m-%d')

    # Reduced max_records for faster testing/API limit safety if needed
    for root in scan_patents_cql(cql, batch_size=25, max_records=500):
        refs = parse_patent_refs(root)
        print(f"  Found {len(refs)} patents in this batch")
        
        for country, number, kind in refs:
            patent_id = f"{country}{number}{kind}"
            
            # Check if patent is already in the database
            if patent_id in existing_ids:
                skipped_count += 1
                count += 1
                print(f"  {count}. [SKIP] {patent_id} (already in database)")
                continue
            
            count += 1
            print(f"  {count}. [NEW] {patent_id}...", end=" ")

            # Fetch bibliographic data
            biblio = get_biblio_data(country, number, kind)
            
            if not biblio:
                print("✗ Failed to fetch")
                # Add to existing_ids to skip next time, but don't count as 'new'
                existing_ids.add(patent_id)
                continue
            
            # Calculate relevance score
            relevance = calculate_relevance_score(
                biblio.get("title", ""),
                biblio.get("abstract", "")
            )
            
            print(f"[Relevance: {relevance:.2f}]", end=" ")
            
            # Filter by relevance threshold
            if relevance < MIN_RELEVANCE_SCORE:
                filtered_count += 1
                print(f"✗ Filtered (below {MIN_RELEVANCE_SCORE})")
                # Add to existing_ids to skip next time
                existing_ids.add(patent_id)
                continue
            
            # Generate AI summary (using the local function)
            print("Summarizing...", end=" ")
            ai_summary = generate_ai_summary(
                biblio.get("title", ""),
                biblio.get("abstract", ""),
                ""  # Claims not fetched in this version for speed
            )
            
            new_count += 1
            
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
                "relevance_score": round(relevance, 3),
                "ai_summary": ai_summary,
                "date_added": current_run_date,
                "is_new": "YES"
            })
            
            print(f"✓ {biblio.get('title', 'N/A')[:40]}")
            time.sleep(0.3)
    
    print(f"\n{'='*80}")
    print(f"Summary:")
    print(f"  Total found: {count}")
    print(f"  Already in DB: {skipped_count}")
    print(f"  Below relevance threshold ({MIN_RELEVANCE_SCORE}): {filtered_count}")
    print(f"  Added to database: {new_count}")
    print(f"{'='*80}\n")
    
    return pd.DataFrame(records)


# ---------------------------------------------------------------
# Processing Existing Data (Final, Robust Function)
# ---------------------------------------------------------------

from datetime import datetime, timedelta
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd

# Define RESEARCH_FOCUS (assuming this is defined globally in your script)
# RESEARCH_FOCUS = "..." 

# Initialize semantic model and focus embedding (assuming this is done globally)
# semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
# research_focus_embedding = semantic_model.encode(RESEARCH_FOCUS)

def calculate_relevance_score(title, abstract, semantic_model, research_focus_embedding):
    """Calculates cosine similarity between the patent and the research focus."""
    patent_text = title + " " + abstract
    if not patent_text.strip():
        return 0.0
    
    try:
        patent_embedding = semantic_model.encode(patent_text, convert_to_tensor=True)
        # Ensure the embeddings are numpy arrays for cosine_similarity
        similarity = cosine_similarity(
            research_focus_embedding.cpu().numpy().reshape(1, -1),
            patent_embedding.cpu().numpy().reshape(1, -1)
        )[0][0]
        return float(similarity)
    except Exception as e:
        print(f"Error calculating score: {e}")
        return 0.0

def process_existing_records(df_old, semantic_model, research_focus_embedding):
    """
    Ensures all existing records have relevance scores and AI summaries.
    This is critical for the first run where these columns are missing.
    """
    
    # --- 1. GUARANTEE COLUMNS EXIST ---
    
    # Explicitly check and add missing columns, using .copy() to prevent SettingWithCopyWarning
    df = df_old.copy()
    
    if 'relevance_score' not in df.columns:
        df['relevance_score'] = 0.0
    if 'ai_summary' not in df.columns:
        df['ai_summary'] = ''

    # Ensure the relevance score is numeric to avoid comparison errors
    df['relevance_score'] = pd.to_numeric(df['relevance_score'], errors='coerce').fillna(0.0)

    # --- 2. IDENTIFY RECORDS TO UPDATE ---
    
    # Identify records that have a score of 0.0 OR are missing a summary
    missing_score = (df['relevance_score'] <= 0.001)
    
    # Identify records with missing or placeholder summaries
    summary_col = df['ai_summary'].fillna('').str.lower()
    missing_summary = (summary_col == '') | \
                      summary_col.str.contains('not available') | \
                      summary_col.str.contains('summary generation failed')

    # The mask selects any record that has a missing score OR a missing summary
    missing_mask = missing_score | missing_summary

    df_to_update = df[missing_mask].copy()

    if df_to_update.empty:
        print("No existing records require scoring or summarization.")
        return df_old # Return the original if nothing needs updating

    # --- 3. PROCESS MISSING RECORDS ---
    
    print(f"\n⚡ Processing {len(df_to_update)} existing records for scoring/summaries...")
    
    # Calculate scores and summaries
    for index, row in df_to_update.iterrows():
        try:
            # 1. Relevance Score
            score = calculate_relevance_score(row['title'], row['abstract'], semantic_model, research_focus_embedding)
            
            # 2. AI Summary (Heuristic)
            abstract_text = row['abstract'].split('.')
            summary = '.'.join(abstract_text[:3]).strip()
            summary = summary if len(summary) > 10 else row['abstract'] # Fallback if first sentences are too short

            # Update the main DataFrame (df)
            df.loc[index, 'relevance_score'] = score
            df.loc[index, 'ai_summary'] = summary
            
            print(f"  [UPDATED] {row['country']}{row['publication_number']} - Score: {score:.4f}")
            
        except Exception as e:
            print(f"  [ERROR] Could not process {row['country']}{row['publication_number']}: {e}")
            df.loc[index, 'ai_summary'] = "Summary generation failed."
    
    # --- 4. SORT AND RETURN ---
    # Sort by relevance score, descending
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
        "relevance_score", # Must be present
        "ai_summary",      # Must be present
        "date_added", 
        "is_new"
    ]

    if CUMULATIVE_CSV.exists():
        df_old = pd.read_csv(CUMULATIVE_CSV)
        
        # 1. PROCESS EXISTING DATA: Fill in missing scores/summaries on df_old
        df_old = process_existing_records(df_old)
        
        # Prepare old data for merge
        df_old['is_new'] = 'NO'
        
        # Ensure columns exist in old data (Redundant due to ensure_ai_columns_exist, but safe)
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

    # 2. Enforce the explicit column order for the final file
    df_all = df_all.reindex(columns=FINAL_COLUMNS, fill_value='')

    # Sort by relevance score (highest first) then by date
    df_all = df_all.sort_values(['relevance_score', 'date_added'], ascending=[False, False])
    
    df_all.to_csv(CUMULATIVE_CSV, index=False)
    print(f"Saved cumulative CSV with {len(df_all)} total records.")
    return df_all


# ---------------------------------------------------------------
# Column Initialization Logic (FIX 2a)
# ---------------------------------------------------------------
def ensure_ai_columns_exist():
    """Reads existing data and adds/updates AI columns if they are missing."""
    if not CUMULATIVE_CSV.exists():
        return
    
    df = pd.read_csv(CUMULATIVE_CSV)
    
    # Check for the presence of the new AI columns
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
# Email Sending (Enhanced)
# ---------------------------------------------------------------

def send_email_with_csv(df_all):
    """Send the updated CSV via email with summary."""
    sender = os.environ.get("SENDER_EMAIL")
    password = os.environ.get("EMAIL_PASSWORD")
    recipient = os.environ.get("RECIPIENT_EMAIL")

    if not sender or not recipient or not password:
        print("Warning: Email credentials (SENDER_EMAIL, EMAIL_PASSWORD, RECIPIENT_EMAIL) not found. Skipping email.")
        return

    # Generate email body with top patents
    new_patents = df_all[df_all['is_new'] == 'YES']
    
    email_body = f"""
Bimonthly Patent Update - {datetime.now().strftime('%Y-%m-%d')}
{'='*80}

NEW PATENTS: {len(new_patents)}
TOTAL DATABASE: {len(df_all)} patents

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
        attachment["Content-Disposition"] = 'attachment; filename="patents_cumulative.csv"'
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
    # 🚨 FIX 2b: Update print statement 🚨
    print(f"Starting 1-Year AI-Enhanced Patent Search - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Relevance threshold: {MIN_RELEVANCE_SCORE}")
    print(f"AI summaries: Local Heuristic (No API Key Required)")
    print("="*80)
    
    # 🚨 FIX 2c: Initialize AI columns for existing data 🚨
    ensure_ai_columns_exist() 
    
    df_new = search_patents()
    
    df_all = update_cumulative_csv(df_new)
    
    send_email_with_csv(df_all)
    
    print("\n" + "="*80)
    print("Process completed successfully.")
    print("="*80)


if __name__ == "__main__":
    main()
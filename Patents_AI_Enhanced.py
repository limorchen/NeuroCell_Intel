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
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from epo_ops import Client, models, middlewares
import epo_ops.exceptions as ops_exc

# Import Claude API (optional - will work without it)
try:
    import anthropic
    CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False
    print("Warning: anthropic package not installed. AI summaries will be disabled.")

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
    # Changed to a print/exit for a complete script execution example, 
    # but the original raise ValueError is more robust for a library
    print("WARNING: Missing EPO OPS API credentials. Cannot run search.")
    # raise ValueError("Missing EPO OPS API credentials.")
    key = "DUMMY_KEY" # Placeholder to allow script to compile
    secret = "DUMMY_SECRET"

# Claude API key (optional)
claude_api_key = os.environ.get("ANTHROPIC_API_KEY")

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

# Initialize semantic search model
print("Loading semantic search model...")
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
research_focus_embedding = semantic_model.encode(RESEARCH_FOCUS)
print("✓ Semantic model ready")

# Initialize Claude client if available
if CLAUDE_AVAILABLE and claude_api_key:
    claude_client = anthropic.Anthropic(api_key=claude_api_key)
    print("✓ Claude API ready")
else:
    claude_client = None
    print("✗ Claude API not available (summaries will be disabled)")


# ---------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------

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
    """Fetch full bibliographic data. (Simplified for existing records - only abstract/title needed)"""
    # NOTE: This function is preserved from the original script and fetches full data.
    # For processing existing records, we assume abstract/title are already in the CSV.
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
    """Calculate semantic similarity to research focus."""
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


def generate_ai_summary(title, abstract, claims=""):
    """Generate concise AI summary using Claude."""
    if not claude_client:
        return "AI summary not available (Claude API not configured)"
    
    if not title and not abstract:
        return "Insufficient data for summary"
    
    try:
        prompt = f"""Summarize this patent in 2-3 concise sentences. Focus on:
1. The main innovation or technology
2. The target disease or medical condition
3. Practical clinical application

Title: {title}
Abstract: {abstract[:1000]}
{f'First Claim: {claims[:500]}' if claims else ''}

Provide a clear, jargon-free summary suitable for researchers."""

        message = claude_client.messages.create(
            model="claude-3-haiku-20240307",  # Fast and economical
            max_tokens=200,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        return message.content[0].text.strip()
    
    except Exception as e:
        print(f"    Error generating summary: {e}")
        return "Summary generation failed"


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

def get_date_range_two_months():
    """Return (start_date, end_date) covering the last 2 months in YYYYMMDD format."""
    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=60)
    
    start_str = start_date.strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")
    
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
    if not client:
        print("Patent search skipped due to missing API credentials.")
        return pd.DataFrame()

    start_date, end_date = get_date_range_two_months()
    
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
            
            # Generate AI summary
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

def process_existing_records(df_old):
    """
    Checks existing records for missing relevance scores and AI summaries 
    and calculates them where necessary. This version is highly robust to errors.
    """
    
    # 1. Force necessary columns to exist, filling missing with empty string/zero
    for col in ['title', 'abstract', 'relevance_score', 'ai_summary', 'country', 'publication_number', 'kind']:
        if col not in df_old.columns:
            # Add missing columns with placeholder data
            df_old[col] = ''
            if col == 'relevance_score':
                df_old[col] = 0.0

    # Ensure 'relevance_score' is numeric for comparison
    df_old['relevance_score'] = pd.to_numeric(df_old['relevance_score'], errors='coerce').fillna(0.0)

    # 2. Identify records that are missing either score or summary
    # Missing if score is 0.0 OR if summary is empty/failed message
    missing_mask = (df_old['relevance_score'] <= 0.0) | \
                   (df_old['ai_summary'].fillna('').apply(lambda x: not x or 'Summary generation failed' in x or 'not available' in x))
                   
    df_to_update = df_old[missing_mask].copy()
    
    if df_to_update.empty:
        print("No existing records require scoring or summarization.")
        return df_old
        
    print(f"\n⚡ Processing {len(df_to_update)} existing records for scoring/summaries...")
    updated_count = 0

    for index, row in df_to_update.iterrows():
        try:
            # Locate the original index to update df_old in place
            original_index = row.name # Using the index from the .copy() is usually safe here
            
            title = row['title'] if pd.notna(row['title']) else ""
            abstract = row['abstract'] if pd.notna(row['abstract']) else ""
            patent_id = f"{row['country']}{row['publication_number']}"
            
            if not title and not abstract:
                # Still set the relevance score to 0.0 explicitly if no text
                df_old.loc[original_index, 'relevance_score'] = 0.0
                df_old.loc[original_index, 'ai_summary'] = "No text available for analysis."
                # print(f"  [SKIP] {patent_id} - No title or abstract.")
                continue
                
            # 1. Calculate Relevance Score
            relevance = calculate_relevance_score(title, abstract)
            
            # 2. Generate AI Summary (only if relevance meets minimum)
            if relevance >= MIN_RELEVANCE_SCORE:
                ai_summary = generate_ai_summary(title, abstract, "")
            else:
                ai_summary = "N/A - Below Relevance Threshold"
            
            # 3. Update the original DataFrame
            df_old.loc[original_index, 'relevance_score'] = round(relevance, 3)
            df_old.loc[original_index, 'ai_summary'] = ai_summary
            updated_count += 1
            
            print(f"  [UPDATED] {patent_id} | Score: {relevance:.2f} | Title: {title[:40]}...")
            time.sleep(0.3)
            
        except Exception as e:
            # Handle single record error without crashing the loop
            error_id = f"{row.get('country')}{row.get('publication_number')}"
            print(f"  [ERROR] Failed to process record {error_id}. Error: {e}")
            df_old.loc[original_index, 'ai_summary'] = f"Processing failed due to error: {e}"
            continue

    print(f"✓ Finished processing. Updated {updated_count} existing records.")
    return df_old

# ---------------------------------------------------------------
# CSV Merge (Corrected)
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
        
        # Combine new and old data. New data is prioritized ('keep="first"').
        df_all = pd.concat([df_new, df_old], ignore_index=True).drop_duplicates(
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
        
        # Only use top patents if 'relevance_score' is not entirely missing
        if 'relevance_score' in new_patents.columns:
            top_patents = new_patents.nlargest(5, 'relevance_score')
            for idx, patent in enumerate(top_patents.itertuples(), 1):
                email_body += f"{idx}. [{patent.relevance_score:.2f}] {patent.title[:80]}\n"
                email_body += f"    Applicant: {patent.applicants[:60]}\n"
                email_body += f"    Summary: {patent.ai_summary[:200]}\n\n"
    
    email_body += f"\nSee attached CSV for full details.\n\n{'='*80}"

    msg = MIMEMultipart()
    msg["Subject"] = f"Patent Update - {len(new_patents)} New Patents - {datetime.now().strftime('%Y-%m-%d')}"
    msg["From"] = sender
    msg["To"] = recipient

    body = MIMEText(email_body, "plain")
    msg.attach(body)

    with open(CUMULATIVE_CSV, "rb") as f:
        attachment = MIMEApplication(f.read(), Name="patents_cumulative.csv")
    attachment["Content-Disposition"] = 'attachment; filename="patents_cumulative.csv"'
    msg.attach(attachment)

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(sender, password)
            smtp.send_message(msg)
        print("✓ Email sent successfully.")
    except Exception as e:
        print(f"✗ Error sending email: {e}")

# ---------------------------------------------------------------
# CSV Correction Utility (Run once)
# ---------------------------------------------------------------

def manual_csv_fixer():
    """Forces the correct column structure into the cumulative CSV."""
    if not CUMULATIVE_CSV.exists():
        print("CSV file not found, skipping manual fix.")
        return
        
    FINAL_COLUMNS = [
        "country", "publication_number", "kind", "title", "applicants", "inventors", 
        "abstract", "publication_date", "priority_date", 
        "relevance_score", "ai_summary", "date_added", "is_new"
    ]
    
    try:
        df = pd.read_csv(CUMULATIVE_CSV)
        
        # Check if the columns are already present
        if 'relevance_score' in df.columns and 'ai_summary' in df.columns:
            print("CSV already has the required analysis columns. Skipping manual fix.")
            return

        print("🚨 Fixing CSV structure: Adding missing analysis columns...")
        
        # Explicitly reindex to add missing columns, filling with defaults
        fill_values = {'relevance_score': 0.0, 'ai_summary': '', 'date_added': ''}
        df = df.reindex(columns=FINAL_COLUMNS).fillna(fill_values)

        # Save the corrected DataFrame back to the file
        df.to_csv(CUMULATIVE_CSV, index=False)
        print(f"✓ CSV file structure corrected. Total records: {len(df)}")
        
    except Exception as e:
        print(f"ERROR: Failed to run manual CSV fix: {e}")

# ---------------------------------------------------------------
# Main function (Ensure only this version exists in your file)
# ---------------------------------------------------------------
def main():
    print("="*80)
    print(f"Starting AI-Enhanced Patent Search - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Relevance threshold: {MIN_RELEVANCE_SCORE}")
    print(f"AI summaries: {'Enabled' if claude_client else 'Disabled'}")
    print("="*80)
    
    # 1. FIX THE CSV HEADERS
    manual_csv_fixer() 
    
    # 2. RUN SEARCH AND GET NEW PATENTS (df_new is empty in your case)
    df_new = search_patents()
    
    # 3. UPDATE/PROCESS EXISTING PATENTS
    # This calls process_existing_records to score/summarize the 39 existing patents.
    df_all = update_cumulative_csv(df_new)
    
    # 4. SEND EMAIL
    send_email_with_csv(df_all)
    
    print("\n" + "="*80)
    print("Process completed successfully.")
    print("="*80)


if __name__ == "__main__":
    main()


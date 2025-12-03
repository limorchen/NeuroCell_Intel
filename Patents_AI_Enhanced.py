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
    raise ValueError("Missing EPO OPS API credentials.")

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

client = Client(
    key=key,
    secret=secret,
    middlewares=middlewares_list
)

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
            "abstract": abstract,  # Keep full abstract for AI processing
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

    for root in scan_patents_cql(cql, batch_size=25, max_records=500):
        refs = parse_patent_refs(root)
        print(f"  Found {len(refs)} patents in this batch")
        
        for country, number, kind in refs:
            patent_id = f"{country}{number}{kind}"
            
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
# CSV Merge
# ---------------------------------------------------------------

def update_cumulative_csv(df_new):
    """Merge new results with existing cumulative CSV."""
    if df_new.empty:
        print("No new patents found this run.")
        if CUMULATIVE_CSV.exists():
            df_old = pd.read_csv(CUMULATIVE_CSV)
            df_old['is_new'] = 'NO'
            df_old.to_csv(CUMULATIVE_CSV, index=False)
            return df_old
        return df_new
    
    if CUMULATIVE_CSV.exists():
        df_old = pd.read_csv(CUMULATIVE_CSV)
        print(f"Loaded {len(df_old)} existing patents")
        
        df_old['is_new'] = 'NO'
        
        # Ensure columns exist in old data
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
        print("Warning: Email credentials not found. Skipping email.")
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
            email_body += f"   Applicant: {patent.applicants[:60]}\n"
            email_body += f"   Summary: {patent.ai_summary[:200]}\n\n"
    
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
# Main
# ---------------------------------------------------------------

def main():
    print("="*80)
    print(f"Starting AI-Enhanced Patent Search - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Relevance threshold: {MIN_RELEVANCE_SCORE}")
    print(f"AI summaries: {'Enabled' if claude_client else 'Disabled'}")
    print("="*80)
    
    df_new = search_patents()
    df_all = update_cumulative_csv(df_new)
    
    send_email_with_csv(df_all)
    
    print("\n" + "="*80)
    print("Process completed successfully.")
    print("="*80)


if __name__ == "__main__":
    main()

import os
import re
import datetime as dt
from dateutil import parser as dateparser
import requests
import feedparser
from bs4 import BeautifulSoup
import pandas as pd
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import smtplib
from email.message import EmailMessage
import trafilatura
import time

# Load environment variables
load_dotenv()

# Set HuggingFace token if available
if not os.getenv("HF_TOKEN"):
    print("Warning: HF_TOKEN not set. Consider adding it to .env for faster model downloads.")

# Configuration
OUTPUT_DIR = "./industry_deals"
SINCE_DAYS = 35
TOP_N_TO_EMAIL = 10
CUMULATIVE_FILENAME = "exosome_deals_DATABASE.xlsx"

# FILTERING THRESHOLDS (tunable)
MIN_RELEVANCE_SCORE = 0.50
MIN_EXOSOME_TERM_MATCH = True
MIN_EVENT_TYPE_CONFIDENCE = 0.3
MIN_TITLE_LENGTH = 20
MIN_SUMMARY_LENGTH = 50

RSS_FEEDS = [
    # Core Biotech/Pharma
    "https://www.fiercebiotech.com/rss.xml",
    "https://endpts.com/feed/",
    "https://www.labiotech.eu/feed/",
    "https://www.biocentury.com/rss",
    "https://www.evaluate.com/vantage/rss",
    
    # Public Wire Feeds
    "https://www.prnewswire.com/rss/health-care-latest-news/health-care-latest-news-list.rss",
    "https://www.globenewswire.com/RssFeed/subjectcode/46-Healthcare%20Business",
    "https://finance.yahoo.com/rss/headline?s=MDXH",
    
    # Google News - Exosome Deals
    "https://news.google.com/rss/search?q=exosome+(acquisition+OR+funding+OR+partnership)&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=%22extracellular+vesicles%22+(deal+OR+funding+OR+partnership)&hl=en-US",
    "https://news.google.com/rss/search?q=exosome+company+(raised+OR+secures+OR+closes)&hl=en-US",
    "https://news.google.com/rss/search?q=exosome+OR+%22extracellular+vesicle%22+AND+(neuro+OR+neurology+OR+regenerat)&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=%22exosome+therapy%22+OR+%22exosome+therapeutics%22&hl=en-US",
    "https://news.google.com/rss/search?q=Evox+OR+Capricor+OR+Codiak+OR+%22Direct+Biologics%22&hl=en-US",
    "https://news.google.com/rss/search?q=%22extracellular+vesicle%22+clinical+trial&hl=en-US",
]

# Spam filtering
SPAM_TERMS = [
    "register for this webinar",
    "join this webinar",
    "register here now",
    "save the date",
    "rsvp to attend",
    "subscribe to unlock",
    "download our report",
    "get the report",
    "market forecast",
    "top 5",
    "top 10",
    "year in review",
    "what to expect in 20",
]

EXOSOME_TERMS = [
    "exosome", "exosomes",
    "extracellular vesicle", "extracellular vesicles",
    "exosomal", "ev therapy", " evs ",
]

INDICATION_KEYWORDS = [
    "neurology", "neuro", "stroke", "als", "amyotrophic", "parkinson", "spinal cord", "neurodegeneration",
    "regenerat", "regeneration", "repair", "rejuvenat", "therapeutic",
    "cancer", "oncology", "tumor", "carcinoma",
    "cardiovascular", "cardiac", "heart", "myocardial",
    "inflammatory", "autoimmune", "immune",
    "kidney", "renal", "liver", "hepatic",
    "lung", "pulmonary", "respiratory",
    "diagnostic", "biomarker", "detection", "screening",
    "liquid biopsy", "early detection",
    "drug delivery", "therapeutic delivery", "targeted therapy"
]

# Known exosome companies
EXOSOME_COMPANIES = [
    "evox", "capricor", "codiak", "direct biologics", "abeona", "aduro",
    "intrinsic", "argenx", "cerlytics", "synthetic biologics", "evo"
]

# Initialize NLP models
nlp = spacy.load("en_core_web_sm")
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# =====================================================
# CORE FUNCTIONS
# =====================================================

def fetch_rss_feeds():
    """Fetch and parse RSS feeds"""
    entries = []
    for feed_url in RSS_FEEDS:
        try:
            print(f"Fetching RSS: {feed_url}")
            feed = feedparser.parse(feed_url)
            entries.extend(feed.entries)
        except Exception as e:
            print(f"Error fetching {feed_url}: {e}")
            continue
    return entries

def clean_text(text):
    """Clean HTML entities and extra whitespace"""
    if not text:
        return ""
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&quot;", '"').replace("&#39;", "'").replace("&nbsp;", " ")
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_full_text(url):
    """Extract article text from URL"""
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(downloaded)
            return text[:5000] if text else ""
    except:
        pass
    return ""

def classify_event_type(title, summary, text):
    """Classify the type of deal/event"""
    content = f"{title} {summary} {text}".lower()
    
    if any(word in content for word in ["fund", "raise", "seed", "series", "investment"]):
        return "funding"
    elif any(word in content for word in ["acqui", "merger", "merge", "buy"]):
        return "acquisition"
    elif any(word in content for word in ["partner", "collaborat", "joint", "agree"]):
        return "partnership"
    elif any(word in content for word in ["license", "licensed"]):
        return "licensing"
    elif any(word in content for word in ["deal"]):
        return "deal"
    else:
        return "news"

def extract_companies(text):
    """Extract company names using NER"""
    if not text:
        return ""
    
    doc = nlp(text[:2000])
    companies = set()
    
    for ent in doc.ents:
        if ent.label_ in ["ORG", "GPE"]:
            companies.add(ent.text)
    
    return "; ".join(sorted(list(companies)))[:200]

def extract_amounts(text):
    """Extract deal amounts"""
    if not text:
        return ""
    
    patterns = [
        r'\$?\s*(\d+(?:\.\d+)?)\s*(?:million|m|bn|billion|b)',
    ]
    
    amounts = []
    for pattern in patterns:
        matches = re.findall(pattern, text.lower())
        if matches:
            for match in matches[:3]:
                try:
                    val = float(match)
                    if val > 0.1:
                        if val >= 1000:
                            amounts.append(f"${val:,.0f}M")
                        else:
                            amounts.append(f"${val:,.1f}M")
                except:
                    pass
    
    return "; ".join(amounts)[:100] if amounts else ""

def extract_indications(text):
    """Extract therapeutic indications"""
    if not text:
        return ""
    
    text_lower = text.lower()
    found = set()
    
    indication_map = {
        "neurological": ["neurology", "neuro", "stroke", "als", "parkinson", "alzheimer", "ms", "spinal cord"],
        "regenerative": ["regenerat", "repair", "rejuvenat", "tissue"],
        "oncology": ["cancer", "oncology", "tumor", "carcinoma", "leukemia"],
        "cardiovascular": ["cardiovascular", "cardiac", "heart", "myocardial"],
        "immunology": ["immune", "autoimmune", "inflammatory"],
        "other": ["kidney", "liver", "lung", "renal", "hepatic", "pulmonary"]
    }
    
    for category, keywords in indication_map.items():
        for keyword in keywords:
            if keyword in text_lower:
                found.add(category)
                break
    
    return "; ".join(sorted(found))[:150] if found else ""

def compute_relevance_score(title, summary, text, event_type):
    """Compute relevance score (0-1)"""
    score = 0.0
    content = f"{title} {summary} {text}".lower()
    
    if any(term in content for term in EXOSOME_TERMS):
        score += 0.6
    
    if event_type in ["funding", "acquisition", "partnership"]:
        score += 0.2
    elif event_type in ["licensing", "deal"]:
        score += 0.1
    
    if any(company in content for company in EXOSOME_COMPANIES):
        score += 0.1
    
    if any(indication in content for indication in INDICATION_KEYWORDS):
        score += 0.1
    
    return min(score, 1.0)

def is_spam(title, summary):
    """Check if content is spam"""
    text = f"{title} {summary}".lower()
    return any(spam in text for spam in SPAM_TERMS)

def is_exosome_relevant(text):
    """Check if content is exosome-related"""
    text_lower = text.lower()
    return any(term in text_lower for term in EXOSOME_TERMS)

def load_existing_cumulative(path):
    """Load existing cumulative database and remove duplicates"""
    if os.path.exists(path):
        try:
            df = pd.read_excel(path)
            
            print(f"Loaded existing DB: {len(df)} records")
            
            # STEP 1: Remove rows with null Title or URL
            df = df.dropna(subset=['Title', 'URL'], how='any')
            print(f"After removing nulls: {len(df)} records")
            
            # STEP 2: Remove duplicate URLs (keep highest RelevanceScore)
            if 'RelevanceScore' in df.columns:
                df = df.sort_values('RelevanceScore', ascending=False)
                df = df.drop_duplicates(subset=['URL'], keep='first')
                print(f"After removing URL duplicates: {len(df)} records")
            
            # STEP 3: Remove duplicate Titles (keep highest RelevanceScore)
            if 'RelevanceScore' in df.columns:
                df = df.sort_values('RelevanceScore', ascending=False)
                df = df.drop_duplicates(subset=['Title'], keep='first')
                print(f"After removing Title duplicates: {len(df)} records")
            
            df = df.reset_index(drop=True)
            return df
        except Exception as e:
            print(f"Error loading DB: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

def generate_month_narrative(df):
    """Generate a narrative summary of the month's deals"""
    if df.empty:
        return ""
    
    # Convert date for grouping
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    # Get last month
    last_month = (dt.datetime.now() - dt.timedelta(days=30)).strftime("%B %Y")
    
    narrative = []
    narrative.append(f"LAST MONTH'S ACTIVITY SUMMARY ({last_month.upper()})")
    narrative.append("=" * 80)
    narrative.append("")
    
    # Get top deals by score
    top_deals = df.nlargest(5, 'RelevanceScore')
    
    # Generate narrative for each top deal
    if len(top_deals) > 0:
        narrative.append("MAJOR DEVELOPMENTS:")
        narrative.append("")
        
        for idx, (_, row) in enumerate(top_deals.iterrows(), 1):
            title = row['Title']
            event_type = row['EventType']
            score = row['RelevanceScore']
            companies = row['Companies'] if pd.notna(row['Companies']) else ""
            amount = row['Amounts'] if pd.notna(row['Amounts']) and str(row['Amounts']) != "" else ""
            indications = row['Indications'] if pd.notna(row['Indications']) else ""
            
            # Build narrative line
            narrative_line = f"* "
            
            # Add title
            narrative_line += title[:70]
            if len(str(row['Title'])) > 70:
                narrative_line += "..."
            
            # Add event type and score indicator
            if score >= 0.8:
                narrative_line += f" [CRITICAL]"
            elif score >= 0.7:
                narrative_line += f" [KEY]"
            
            narrative.append(narrative_line)
            
            # Add details on next line
            details = []
            if event_type:
                details.append(f"{event_type.title()}")
            if amount:
                details.append(f"Amount: {amount}")
            if indications:
                indications_short = "; ".join(indications.split(";")[:2])
                details.append(f"Focus: {indications_short}")
            
            if details:
                narrative.append("  " + " | ".join(details))
            
            narrative.append("")
    
    # Summary by event type
    event_counts = df['EventType'].value_counts()
    if len(event_counts) > 0:
        narrative.append("ACTIVITY SNAPSHOT:")
        narrative.append("")
        
        summary_parts = []
        if 'funding' in event_counts.index:
            summary_parts.append(f"{event_counts['funding']} funding rounds")
        if 'acquisition' in event_counts.index:
            summary_parts.append(f"{event_counts['acquisition']} acquisitions")
        if 'partnership' in event_counts.index:
            summary_parts.append(f"{event_counts['partnership']} partnerships")
        if 'licensing' in event_counts.index:
            summary_parts.append(f"{event_counts['licensing']} licensing deals")
        
        if summary_parts:
            narrative.append("* " + " | ".join(summary_parts))
            narrative.append("")
        
        # Therapeutic focus
        all_indications = " ".join(df['Indications'].dropna().astype(str))
        if all_indications.strip():
            unique_indications = list(set([i.strip() for i in all_indications.split(";") if i.strip()]))
            unique_indications = unique_indications[:6]  # Top 6
            narrative.append(f"* Therapeutic focus: {', '.join(unique_indications)}")
            narrative.append("")
    
    # Funding highlights
    with_amounts = df[df['Amounts'].notna() & (df['Amounts'] != "")]
    if len(with_amounts) > 0:
        narrative.append("FUNDING HIGHLIGHTS:")
        narrative.append("")
        
        # Extract and sum amounts
        top_funding = with_amounts.nlargest(3, 'RelevanceScore')
        for _, row in top_funding.iterrows():
            companies_str = row['Companies'][:40] if pd.notna(row['Companies']) else "Undisclosed"
            narrative.append(f"* {companies_str} - {row['Amounts']}")
        
        narrative.append("")
    
    narrative.append("=" * 80)
    narrative.append("")
    
    return "\n".join(narrative)

def save_cumulative(df, path):
    """Save cumulative database with formatting and new record marking"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Ensure correct column order
    columns = ['Date', 'Title', 'EventType', 'Companies', 'Amounts', 
               'Indications', 'RelevanceScore', 'IsExosome', 'Quality',
               'DateAddedToDB', 'IsNewThisMonth', 'Summary', 'URL', 'RawText']
    
    df = df[[col for col in columns if col in df.columns]]
    
    # SORT: Old records first, new records last (by DateAddedToDB ascending)
    # This ensures new entries from today's search appear at the bottom
    if 'DateAddedToDB' in df.columns:
        df = df.sort_values('DateAddedToDB', ascending=True, na_position='last')
        df = df.reset_index(drop=True)
    
    try:
        with pd.ExcelWriter(path, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Deals')
            
            # Format worksheet
            worksheet = writer.sheets['Deals']
            
            # Set column widths
            worksheet.column_dimensions['A'].width = 12
            worksheet.column_dimensions['B'].width = 50
            worksheet.column_dimensions['C'].width = 12
            worksheet.column_dimensions['D'].width = 30
            worksheet.column_dimensions['E'].width = 15
            worksheet.column_dimensions['F'].width = 20
            worksheet.column_dimensions['G'].width = 8
            worksheet.column_dimensions['H'].width = 8
            worksheet.column_dimensions['I'].width = 8
            worksheet.column_dimensions['J'].width = 12
            worksheet.column_dimensions['K'].width = 12
            
            # Freeze header row
            worksheet.freeze_panes = 'A2'
        
        print(f"Saved cumulative DB to: {path}")
    except Exception as e:
        print(f"Error saving DB: {e}")

def send_email_with_top_deals(df, df_all, top_n=10):
    """Send email with narrative summary and top deals"""
    if df.empty:
        print("No deals to email")
        return
    
    # Sort by relevance score, get top N
    df_top = df.nlargest(top_n, 'RelevanceScore')
    
    # Generate narrative summary from all database
    month_narrative = generate_month_narrative(df_all)
    
    # Prepare email body
    body = []
    body.append("MONTHLY EXOSOME INTELLIGENCE REPORT")
    body.append("=" * 80)
    body.append(f"Report Date: {dt.datetime.now().strftime('%B %d, %Y')}")
    body.append(f"New Items This Run: {len(df)}")
    body.append("")
    body.append("")
    
    # Add narrative summary
    body.append(month_narrative)
    
    # Add today's top deals
    body.append("TODAY'S TOP DISCOVERIES")
    body.append("=" * 80)
    body.append(f"Showing top {len(df_top)} of {len(df)} new items")
    body.append("")
    
    for idx, (_, row) in enumerate(df_top.iterrows(), 1):
        body.append(f"{idx}. {row['Title']}")
        body.append(f"   Type: {row['EventType']} | Relevance: {row['RelevanceScore']:.2f}")
        
        companies = row['Companies'] if pd.notna(row['Companies']) else "Not specified"
        body.append(f"   Companies: {companies[:60]}")
        
        amount = row['Amounts'] if pd.notna(row['Amounts']) and str(row['Amounts']) != "" else "Undisclosed"
        body.append(f"   Amount: {amount}")
        
        indications = row['Indications'] if pd.notna(row['Indications']) else "Various"
        body.append(f"   Focus: {indications[:60]}")
        
        body.append(f"   Source: {row['URL'][:70]}...")
        body.append("")
    
    body.append("=" * 80)
    body.append("Attached: Complete Exosome Deals Database (Excel)")
    body.append("")
    
    # Send email
    try:
        host = os.getenv("SMTP_HOST_465")
        user = os.getenv("SMTP_USER_465")
        pwd = os.getenv("SMTP_PASS_465")
        to_addr = os.getenv("EMAIL_TO_465")
        
        if not all([host, user, pwd, to_addr]):
            print("Email not sent: SMTP or EMAIL_TO_465 env vars missing.")
            return
        
        msg = EmailMessage()
        msg['Subject'] = f"Exosome Intelligence Report - {dt.datetime.now().strftime('%B %d, %Y')}"
        msg['From'] = user
        msg['To'] = to_addr
        msg.set_content("\n".join(body))
        
        # Attach Excel file
        excel_path = os.path.join(OUTPUT_DIR, CUMULATIVE_FILENAME)
        if os.path.exists(excel_path):
            with open(excel_path, 'rb') as attachment:
                msg.add_attachment(attachment.read(), maintype='application', 
                                  subtype='vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                                  filename='exosome_deals_DATABASE.xlsx')
        
        with smtplib.SMTP_SSL(host, 465) as smtp:
            smtp.login(user, pwd)
            smtp.send_message(msg)
        
        print("Email summary sent with attachment.")
    except Exception as e:
        print(f"Email not sent: {e}")

# =====================================================
# MAIN FUNCTION
# =====================================================

def main():
    print("Starting monthly run for NeuroCell Intelligence")
    
    today = dt.datetime.now().strftime("%Y-%m-%d")
    
    # 1) Fetch RSS entries
    entries = fetch_rss_feeds()
    print(f"Collected {len(entries)} recent entries.")
    
    # 2) Normalize entries
    unique_entries = {}
    for entry in entries:
        url = entry.get('link', '')
        if url not in unique_entries:
            unique_entries[url] = entry
    
    print(f"Normalized to {len(unique_entries)} unique items.")
    
    # 3) Load existing database to track new records
    cumulative_path = os.path.join(OUTPUT_DIR, CUMULATIVE_FILENAME)
    df_existing = load_existing_cumulative(cumulative_path)
    
    # Get list of existing URLs to compare against
    existing_urls = set(df_existing['URL'].values) if not df_existing.empty else set()
    
    # 4) Process entries
    processed = []
    for url, entry in unique_entries.items():
        try:
            title = clean_text(entry.get('title', ''))
            
            # Skip if too short
            if len(title) < MIN_TITLE_LENGTH:
                continue
            
            # Skip spam
            summary = clean_text(entry.get('summary', ''))
            if is_spam(title, summary):
                continue
            
            # Extract full text
            full_text = extract_full_text(url)
            
            # Get publish date
            pub = entry.get('published_parsed')
            if pub:
                pub = dt.datetime(*pub[:6])
            else:
                pub = dt.datetime.now()
            
            # Extract information
            event_type = classify_event_type(title, summary, full_text)
            companies = extract_companies(f"{title} {summary} {full_text}")
            amounts = extract_amounts(f"{summary} {full_text}")
            indications = extract_indications(f"{title} {summary} {full_text}")
            
            # Compute relevance
            relevance_score = compute_relevance_score(title, summary, full_text, event_type)
            
            # Check minimum relevance
            if relevance_score < MIN_RELEVANCE_SCORE:
                continue
            
            # Check minimum exosome relevance
            if MIN_EXOSOME_TERM_MATCH and not is_exosome_relevant(f"{title} {summary} {full_text}"):
                continue
            
            # Quality scoring
            quality = "HIGH" if relevance_score >= 0.7 else "MEDIUM" if relevance_score >= 0.5 else "LOW"
            is_exosome = is_exosome_relevant(f"{title} {summary}")
            
            # Mark if this is a NEW record
            is_new = url not in existing_urls
            
            processed.append({
                "Date": pub.strftime("%Y-%m-%d") if isinstance(pub, dt.datetime) else "",
                "Title": title,
                "EventType": event_type,
                "Companies": companies,
                "Amounts": amounts,
                "Indications": indications,
                "RelevanceScore": round(relevance_score, 2),
                "IsExosome": is_exosome,
                "Quality": quality,
                "DateAddedToDB": today,
                "IsNewThisMonth": is_new,
                "Summary": summary,
                "URL": url,
                "RawText": full_text[:2000]
            })
        
        except Exception as e:
            print(f"Error processing entry: {e}")
            continue
    
    print(f"Processed and kept {len(processed)} items after relevance filtering.")
    
    # 5) Create DataFrame
    df_new = pd.DataFrame(processed)
    
    # 6) Merge with cumulative
    if not df_existing.empty:
        df_merged = pd.concat([df_existing, df_new], ignore_index=True)
        # Remove duplicates by URL (keep last/highest)
        df_merged = df_merged.drop_duplicates(subset=['URL'], keep='last')
        # Remove duplicates by Title (keep last/highest)
        df_merged = df_merged.drop_duplicates(subset=['Title'], keep='last')
    else:
        df_merged = df_new.copy()
        df_merged['IsNewThisMonth'] = True
    
    save_cumulative(df_merged, cumulative_path)
    
    # 7) Save dated snapshot (only NEW records from this run)
    today_str = dt.datetime.utcnow().strftime("%Y%m%d")
    snapshot_path = os.path.join(OUTPUT_DIR, f"exosome_deals_run_{today_str}.xlsx")
    try:
        df_new.to_excel(snapshot_path, index=False)
        print(f"Saved run snapshot to: {snapshot_path}")
    except Exception as e:
        print(f"Failed to save run snapshot: {e}")
    
    # 8) Send email with narrative summary AND top deals from this run
    if not df_new.empty:
        send_email_with_top_deals(df_new, df_merged, 10)


if __name__ == "__main__":
    main()

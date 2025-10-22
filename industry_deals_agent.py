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


# ---------------------------------------
# 🔐 Load environment variables
# ---------------------------------------
load_dotenv()
# NEWSAPI_KEY is not used in this script
# NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "") 

# ---------------------------------------
# 📁 Configuration
# ---------------------------------------
OUTPUT_DIR = "./industry_deals"
SINCE_DAYS = 360
TOP_N_TO_EMAIL = 10

# NOTE: This list is from the previous working version. If you encounter 
# 403 or 404 errors again, you MUST use the cleaned list from the prior step.
RSS_FEEDS = [
    # Core Biotech/Pharma Feeds (Working and Stable)
    "https://www.fiercebiotech.com/rss.xml",
    "https://endpts.com/feed/",
    "https://www.labiotech.eu/feed/",
    "https://www.biocentury.com/rss",
    "https://www.bioworld.com/rss",
    "https://www.evaluate.com/vantage/rss",
    
    # Stable Public Wire Feeds
    "https://www.prnewswire.com/rss/health-care-latest-news/health-care-latest-news-list.rss",
    "https://www.globenewswire.com/RssFeed/subjectcode/46-Healthcare%20Business", 
    "https://finance.yahoo.com/rss/headline?s=MDXH", 
    
    # Google News search focused on exosomes deals (Crucial for overall coverage)
    "https://news.google.com/rss/search?q=exosome+(acquisition+OR+funding+OR+partnership)&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=%22extracellular+vesicles%22+(deal+OR+funding+OR+partnership)&hl=en-US",
    "https://news.google.com/rss/search?q=exosome+company+(raised+OR+secures+OR+closes)&hl=en-US",
]
PR_PAGES = []

# Expanded indication keywords
INDICATION_KEYWORDS = [
    "neurology","neuro","stroke","als","amyotrophic","parkinson","spinal cord","neurodegeneration",
    "regenerat","regeneration","repair","rejuvenat","therapeutic",
    "cancer","oncology","tumor","carcinoma",
    "cardiovascular","cardiac","heart","myocardial",
    "inflammatory","autoimmune","immune",
    "kidney","renal","liver","hepatic",
    "lung","pulmonary","respiratory",
    "diagnostic","biomarker","detection","screening",
    "liquid biopsy","early detection",
    "drug delivery","therapeutic delivery","targeted therapy"
]

EVENT_KEYWORDS = {
    "acquisition": ["acquir","acquisition","acquired","merger","merged","buyout","takeover"],
    "partnership": ["partner","partnership","collaborat","alliance","strategic relationship"],
    "licensing": ["license","licensing","licensed","in-license","out-license"],
    "funding": ["funding","raised","series a","series b","grant","investment","seed","financ","venture"],
    "deal": ["deal", "agreement","term sheet","option agreement","commercialization"]
}

EXOSOME_COMPANIES = [
    "codiak", "evox", "anjarium", "capricor", "cartherics", "evelo",
    "exosome diagnostics", "paige.ai", "direct biologics", "kimera labs",
    "aegle therapeutics", "avalon globocare", "aruna bio", "evotec",
    "vesigen", "ciloa", "exosomics", "exopharm", "ilias biologics",
    "exosome therapeutics", "clara biotech", "lonza", "tavec", 
    "roosterbio", "exocobio", "versatope therapeutics",
    "nanosomix", "paracrine therapeutics", "exocelbio", 
    "regeneveda", "mdxhealth", "bio-techne", "nurexone biologic", "biorestorative therapeutics",
    "reneuron", "pl bioscience", "everzom", "exo biologics", "ilbs",
    "corestemchemon", "cellgenic", "abio materials", "resilielle cosmetics", "skinseqnc", "zeo sceinetifix",
    "bpartnership", "clinic utoquai", "swiss derma clinic", "laclinique", "exogems", "ags therapeutics"
]

# ---------------------------------------
# 🧠 Load NLP models
# ---------------------------------------
nlp = spacy.load("en_core_web_sm")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------------------------------
# 🛠 Helper functions
# ---------------------------------------
def ensure_outdir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def fetch_rss_entries(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    for attempt in range(2):
        try:
            # Use requests for better control and pass content to feedparser
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                f = feedparser.parse(response.content) 
                return f.entries
            else:
                print(f"RSS HTTP error (Attempt {attempt+1}/2) {url}: Status {response.status_code}")
                time.sleep(3)
        except Exception as e:
            print(f"RSS general error (Attempt {attempt+1}/2) {url}: {e}")
            time.sleep(3)
    return []

def fetch_article_text(url, timeout=10):
    """Fetch article text using trafilatura - more reliable than newspaper3k"""
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
            if text and len(text) > 100:
                return text[:10000]  # Limit to 10k chars
        return ""
    except Exception as e:
        print(f"Article fetch failed for {url[:50]}...: {str(e)[:30]}")
        return ""

# -----------------------------------------------------
# 💰 ENHANCED MONEY EXTRACTION AND VALIDATION FUNCTIONS 
# -----------------------------------------------------

def normalize_amount(text):
    """
    Enhanced normalization: Converts money string to a single integer amount.
    """
    if not text or not isinstance(text, str):
        return None
    
    t = text.lower().strip()
    
    num_match = re.search(r'(\d+(?:[,\s]\d{3})*(?:\.\d+)?)', t)
    if not num_match:
        return None
    
    num_str = num_match.group(1).replace(',', '').replace(' ', '')
    
    try:
        num = float(num_str)
    except Exception:
        return None
    
    # Apply multipliers
    if re.search(r'\b(trillion|tn|t)\b', t):
        num *= 1_000_000_000_000
    elif re.search(r'\b(billion|bn|b)\b', t):
        num *= 1_000_000_000
    elif re.search(r'\b(million|mn|m)\b', t):
        num *= 1_000_000
    elif re.search(r'\b(thousand|k)\b', t):
        num *= 1_000
    
    try:
        return int(round(num))
    except Exception:
        return None

def extract_amounts(text):
    """
    Enhanced money extraction with better pattern matching for strings.
    """
    if not text:
        return []

    amount_patterns = [
        # $15 million, $5M, $250,000.00, $1.5B
        r'\$\s?\d{1,3}(?:[,\s]\d{3})*(?:\.\d+)?\s?(?:million|billion|thousand|trillion|m|b|k|bn|tn)?',
        
        # 15 million dollars, 3M USD, 250 thousand EUR
        r'\d{1,3}(?:[,\s]\d{3})*(?:\.\d+)?\s?(?:million|billion|thousand|trillion|m|b|k|bn|tn)\s?(?:USD|usd|dollars?|EUR|eur|€|\$)?',
        
        # USD 15 million, EUR 3M
        r'(?:USD|usd|EUR|eur|€)\s?\d{1,3}(?:[,\s]\d{3})*(?:\.\d+)?\s?(?:million|billion|thousand|trillion|m|b|k|bn|tn)?',
        
        # Edge cases: "a $15M", "approximately $3 million"
        r'(?:approximately|about|around|nearly|up\s+to|over)?\s?\$?\s?\d{1,3}(?:[,\s]\d{3})*(?:\.\d+)?\s?(?:million|billion|thousand|m|b|k|bn)',
    ]

    matches = []
    for pat in amount_patterns:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            amt = m.group(0).strip()
            amt = re.sub(r'\s+', ' ', amt)
            amt = re.sub(r'^(?:approximately|about|around|nearly|up to|over)\s+', '', amt, flags=re.I)
            matches.append(amt)

    seen = set()
    unique = []
    for m in matches:
        key = re.sub(r'[^\d.]', '', m.lower())
        if key and key not in seen and len(key) >= 1:
            seen.add(key)
            unique.append(m)

    return unique[:5] # Return top 5 amounts

def extract_deal_context(text, amount_str):
    """Extract surrounding context around a dollar amount."""
    if not text or not amount_str:
        return ""
    
    pattern = re.escape(amount_str)
    match = re.search(pattern, text, re.IGNORECASE)
    
    if match:
        start = max(0, match.start() - 100)
        end = min(len(text), match.end() + 100)
        context = text[start:end]
        return context
    
    return ""

def validate_deal_amount(amount_str, context, event_type):
    """
    Validate if an extracted amount is likely a real deal amount using context.
    Returns confidence score 0-1.
    """
    if not amount_str:
        return 0.0
    
    score = 0.5 
    context_lower = context.lower()
    
    positive_keywords = [
        'raised', 'raised in', 'closed', 'secured', 'acquired for',
        'purchased for', 'valued at', 'worth', 'funding round',
        'investment', 'series', 'financing', 'deal worth',
        'transaction', 'acquisition price', 'upfront payment'
    ]
    
    for keyword in positive_keywords:
        if keyword in context_lower:
            score += 0.2
            break
    
    negative_keywords = [
        'market size', 'revenue', 'annual', 'quarterly',
        'sales', 'profit', 'loss', 'stock price',
        'market cap', 'valuation of company', 'worth of market'
    ]
    
    for keyword in negative_keywords:
        if keyword in context_lower:
            score -= 0.3
            break
    
    if event_type in ['acquisition', 'funding', 'licensing']:
        score += 0.1
    
    # Amount range validation
    normalized = normalize_amount(amount_str)
    if normalized:
        if 1_000_000 <= normalized <= 10_000_000_000:
            score += 0.2
        elif normalized < 100_000 or normalized > 100_000_000_000:
            score -= 0.3
    
    return max(0.0, min(1.0, score))

def extract_amounts_with_validation(title, text, summary, event_type):
    """
    Main extraction function that combines extraction + validation
    Returns list of validated amounts with confidence scores
    """
    full_text = f"{title} {text} {summary}"
    all_amounts = extract_amounts(full_text)
    
    validated_amounts = []
    for amount in all_amounts:
        context = extract_deal_context(full_text, amount)
        confidence = validate_deal_amount(amount, context, event_type)
        
        if confidence >= 0.4: # Threshold for inclusion
            validated_amounts.append({
                'amount': amount,
                'confidence': confidence,
                'context': context[:200]
            })
    
    validated_amounts.sort(key=lambda x: x['confidence'], reverse=True)
    return validated_amounts

def search_for_deal_amount(title, companies, event_type):
    """
    Secondary web scraper (DuckDuckGo HTML) for deal amount fallback.
    """
    if event_type not in ["acquisition", "funding"]:
        return []
    
    time.sleep(1)

    try:
        company_str = " ".join(companies[:2]) if companies else ""
        search_query = f"{company_str} {event_type} amount million"
        search_url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(search_query)}"
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        
        response = requests.get(search_url, headers=headers, timeout=5)
        if response.status_code != 200:
            return []
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        snippets = []
        for result in soup.find_all('a', class_='result__snippet'):
            snippets.append(result.get_text())
        for result in soup.find_all('a', class_='result__a'):
            snippets.append(result.get_text())
        
        combined_text = " ".join(snippets[:5])
        amounts = extract_amounts(combined_text)
        
        if amounts:
            print(f"🔍 Found amount via search: {amounts[0]} for {title[:50]}...")
            return amounts[:2]
        
        return []
        
    except Exception as e:
        print(f"Search failed: {str(e)[:50]}")
        return []

# -----------------------------------------------------
# 📚 REMAINING HELPER FUNCTIONS 
# -----------------------------------------------------

def extract_companies(text):
    """Extract company names with aggressive filtering"""
    doc = nlp(text)
    orgs = []
    
    IGNORE_ORGS = [
        "msn", "manila times", "reuters", "bloomberg", "fiercebiotech",
        "endpoints", "yahoo", "google", "facebook", "twitter", "linkedin",
        "ap", "associated press", "wall street journal", "new york times",
        "cnn", "bbc", "fox news", "nbc", "cbs", "abc news", "tipranks",
        "globe newswire", "business wire", "pr newswire", "marketwatch",
        "seeking alpha", "motley fool", "benzinga", "zacks", "biospace",
        "genengnews", "labiotech", "fiercepharma"
    ]
    
    REMOVE_SUFFIXES = [
        " - tipranks", " tipranks", "the manila times", " - msn",
        " acquisition", " diagnostics acquisition"
    ]
    
    for ent in doc.ents:
        if ent.label_ == "ORG":
            t = ent.text.strip()
            for suffix in REMOVE_SUFFIXES:
                if t.lower().endswith(suffix):
                    t = t[:-len(suffix)].strip()
            if len(t) < 2 or len(t.split()) > 6:
                continue
            if t.lower() in IGNORE_ORGS:
                continue
            if any(ignore in t.lower() for ignore in IGNORE_ORGS):
                continue
            if t.lower() in ["acquisition", "diagnostics", "acquisition from", "bio", "techne"]:
                continue
            orgs.append(t)
    
    seen = set()
    unique_orgs = []
    for org in orgs:
        if org.lower() not in seen:
            seen.add(org.lower())
            unique_orgs.append(org)
    
    return unique_orgs[:5]

def extract_acquisition_details(title, text):
    """Manually extract acquisition details from text"""
    combined = title + " " + text
    patterns = [
        r'([A-Z][A-Za-z0-9\s&\.]+?)\s+(?:completes?|closes?|announces?)\s+(?:the\s+)?(?:acquisition of|purchase of)\s+([A-Z][A-Za-z0-9\s&\.]+?)(?:\s+(?:for|from|$))',
        r'([A-Z][A-Za-z0-9\s&\.]+?)\s+(?:acquires?|buys?|acquired|purchased)\s+([A-Z][A-Za-z0-9\s&\.]+?)(?:\s+(?:for|from|$))',
        r'([A-Z][A-Za-z0-9\s&\.]+?)\s+acquisition\s+(?:of|by|from)\s+([A-Z][A-Za-z0-9\s&\.]+?)(?:\s|$)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, combined, re.I)
        if match:
            acquirer = match.group(1).strip()
            target = match.group(2).strip() if len(match.groups()) > 1 else ""
            acquirer = re.sub(r'\s+(announces|completes|closes|acquisition).*', '', acquirer, flags=re.I)
            target = re.sub(r'\s+(from|for|acquisition).*', '', target, flags=re.I)
            if acquirer and target and len(acquirer) > 2 and len(target) > 2:
                return [acquirer, target]
    
    return []

def classify_event(text):
    tl = text.lower()
    scores = {}
    for ev, kws in EVENT_KEYWORDS.items():
        scores[ev] = sum(1 for k in kws if k in tl)
    best = max(scores.items(), key=lambda x: x[1])
    return best[0] if best[1] > 0 else "news"

def detect_indications(text):
    tl = text.lower()
    hits = [kw for kw in INDICATION_KEYWORDS if kw in tl]
    return sorted(set(hits))

def summarize_short(text, max_sent=2):
    sents = re.split(r'(?<=[.!?])\s+', text)
    return " ".join(sents[:max_sent]).strip()

def normalize_title(title):
    title = re.split(r'\s*[-–—]\s*', title)[0]
    for word in ['announces', 'completes', 'closes', 'closing', 'announces closing']:
        title = re.sub(r'\b' + word + r'\b', '', title, flags=re.I)
    title = re.sub(r'[^\w\s]', '', title.lower()).strip()
    title = re.sub(r'\s+', ' ', title)
    return title

def is_exosome_relevant(text, title):
    combined = (title + " " + text).lower()
    
    SPAM_TERMS = [
        "webinar", "sponsored", "whitepaper", "advertise",
        "sign up to read", "subscribe", "newsletter",
        "market research", "market size", "market report", "market insights",
        "pipeline insights", "download", "forecast", "market analysis"
    ]
    
    exosome_terms = [
        "exosome", "exosomes",
        "extracellular vesicle", "extracellular vesicles",
        "exosomal", "ev therapy", " evs ",
    ]
    
    company_match = any(comp.lower() in combined for comp in EXOSOME_COMPANIES)
    exosome_hits = sum(term in combined for term in exosome_terms)
    
    if not ((company_match and exosome_hits > 0) or (exosome_hits > 1)):
        return False
    
    if any(term in combined for term in SPAM_TERMS):
        return False
    
    return True

def send_email_with_attachment(subject, body, attachment_path):
    SMTP_HOST = os.getenv("SMTP_HOST", "smtp.example.com")
    SMTP_PORT = int(os.getenv("SMTP_PORT", 465))
    SMTP_USER = os.getenv("SMTP_USER", "")
    SMTP_PASS = os.getenv("SMTP_PASS", "")
    EMAIL_FROM = os.getenv("EMAIL_FROM", SMTP_USER)
    EMAIL_TO = os.getenv("EMAIL_TO", "").split(",")

    if not SMTP_USER or not SMTP_PASS or not EMAIL_TO:
        print("SMTP credentials or recipients not set. Skipping email.")
        return

    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = EMAIL_FROM
    msg['To'] = ", ".join(EMAIL_TO)
    msg.set_content(body)

    if attachment_path and os.path.isfile(attachment_path):
        with open(attachment_path, 'rb') as f:
            file_data = f.read()
            file_name = os.path.basename(attachment_path)
        msg.add_attachment(file_data, maintype='application', subtype='octet-stream', filename=file_name)

    try:
        with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT) as server:
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)
        print("Email sent to", EMAIL_TO)
    except Exception as e:
        print("Failed to send email:", e)

# ---------------------------------------
# 🧭 Main pipeline (MODIFIED)
# ---------------------------------------
def run_agent():
    ensure_outdir()
    since = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=SINCE_DAYS)
    collected = []

    # 1) RSS - early filter by deal keywords
    DEAL_KEYWORDS_LOWER = [
        "acquire","acquisition","acquired","merger","merge","buyout","takeover",
        "partner","partnership","collaborate","alliance","strategic relationship",
        "license","licensing","funding","raised","series a","series b","seed","investment","venture",
        "deal","agreement","term sheet","commercialization"
    ]

    for rss in RSS_FEEDS:
        print(f"Fetching: {rss[:60]}...")
        entries = fetch_rss_entries(rss)
        
        # Add a short delay between feeds
        time.sleep(0.5)

        for e in entries:
            title = e.get("title","") or ""
            raw_summary = e.get("summary","") or e.get("description","") or ""
            clean_summary = BeautifulSoup(raw_summary, "html.parser").get_text(strip=True)
            
            # Early filter: skip if no deal-related keywords
            combined_text = (title + " " + clean_summary).lower()
            if not any(k in combined_text for k in DEAL_KEYWORDS_LOWER):
                continue

            pub = e.get("published") or e.get("updated") or e.get("pubDate")
            try:
                pub_dt = dateparser.parse(pub) if pub else None
            except Exception:
                pub_dt = None
            if pub_dt:
                if pub_dt.tzinfo is None:
                    pub_dt = pub_dt.replace(tzinfo=dt.timezone.utc)
                if pub_dt < since:
                    continue
            
            collected.append({
                "title": title,
                "link": e.get("link",""),
                "published": pub_dt.isoformat() if pub_dt else None,
                "summary": clean_summary
            })

    print(f"Initial collection: {len(collected)} items")

    # Dedupe by URL
    uniq = {}
    for c in collected:
        key = (c.get("link") or c.get("title")).strip()
        if key and key not in uniq:
            uniq[key] = c
    collected = list(uniq.values())
    print(f"After URL deduplication: {len(collected)} items")

    # Dedupe by normalized title + date
    date_title_map = {}
    for item in collected:
        norm_title = normalize_title(item.get("title", ""))
        pub_date = item.get("published", "")[:10] if item.get("published") else "unknown"
        key = f"{norm_title}_{pub_date}"
        if key not in date_title_map and norm_title and len(norm_title) > 10:
            date_title_map[key] = item
    collected = list(date_title_map.values())
    print(f"After date+title deduplication: {len(collected)} items")

    # -----------------
    # 3) Process
    # -----------------
    processed = []
    for item in collected:
        title = item.get("title","")
        url = item.get("link","")
        pub = item.get("published")
        summary = item.get("summary","") or ""
        
        # Try to fetch article text
        text = fetch_article_text(url) if url else ""
        
        # Use the most comprehensive available text for filtering and processing
        if not text or len(text) < 200:
            full_text = summary if len(summary) > 50 else title
        else:
            full_text = text

        if not is_exosome_relevant(full_text, title):
            continue

        short = summarize_short(full_text, max_sent=2)
        event = classify_event(full_text + " " + title)

        # Extract companies
        # --- ENHANCED COMPANY EXTRACTION LOGIC ---
        
        # 1. Always prioritize the acquisition regex for ACQUISITION events
        if event == "acquisition":
            companies = extract_acquisition_details(title, full_text)
            
            # If the acquisition regex worked, use those two key companies only.
            if len(companies) >= 2:
                # Add a filter to remove common junk from these key names
                companies = [re.sub(r'SA|SA\s*\(NASDAQ:[^\)]+\)|LLC|Inc\.?|Corp\.?|Corporation|Limited', '', c).strip() 
                             for c in companies]
                # Fall through to general NLP *only* if key extraction failed to find 2 names
            else: 
                # Fallback to general NLP entity extraction
                companies_from_text = extract_companies(full_text)
                companies_from_title = extract_companies(title + " " + summary)
                companies = list(dict.fromkeys(companies_from_text + companies_from_title))[:5]

        # 2. General NLP for all other events (Funding, Partnership, News)
        else:
            companies_from_text = extract_companies(full_text)
            companies_from_title = extract_companies(title + " " + summary)
            companies = list(dict.fromkeys(companies_from_text + companies_from_title))[:5]

        # --- END ENHANCED COMPANY EXTRACTION LOGIC ---

        # 💰 ENHANCED MONEY EXTRACTION 💰
        validated_money = extract_amounts_with_validation(title, full_text, summary, event)
        
        # Get just the amount strings for output
        money = [vm['amount'] for vm in validated_money[:3]]

        # Normalize numeric
        amounts_numeric = []
        for m in money:
            n = normalize_amount(m)
            if n is not None:
                amounts_numeric.append(n)
        amounts_numeric = list(dict.fromkeys(amounts_numeric))

        # 🔍 Fallback: Secondary web search scraper (Only if primary extraction failed)
        if not money and event in ["acquisition", "funding"]:
            print(f"🔍 Searching web for amount: {title[:50]}...")
            search_amounts = search_for_deal_amount(title, companies, event)
            if search_amounts:
                money = search_amounts
                for m in search_amounts:
                    n = normalize_amount(m)
                    if n is not None:
                        amounts_numeric.append(n)
        
        # END MONEY EXTRACTION 
   
        indications = detect_indications(full_text + " " + title + " " + summary)

        # Scoring
        score = (1.5 if event in ["acquisition","partnership","licensing","funding"] else 0.2)
        score += 0.5 * len(indications)
        score += 0.8 if money else 0.0
        score += 0.3 * len(companies)
        exosome_count = full_text.lower().count("exosome") + full_text.lower().count("extracellular vesicle")
        score += min(exosome_count * 0.3, 2.0)

        processed.append({
            "title": title,
            "url": url,
            "published": pub,
            "date": pub,
            "companies": companies,
            "event_type": event,
            "amounts": money,
            "amounts_numeric": amounts_numeric,
            "indications": indications,
            "short_summary": short,
            "full_text": full_text,
            "score": score
        })

    print(f"After relevance filtering: {len(processed)} items")
    if not processed:
        print("No processed items found.")
        return None

    # -----------------
    # 4) Dedupe via embeddings
    # -----------------
    texts = [p["title"] + " " + p["short_summary"] for p in processed]
    emb = embedder.encode(texts)
    sim = cosine_similarity(emb)
    n = len(processed)
    drop = set()
    for i in range(n):
        for j in range(i+1, n):
            if sim[i,j] > 0.75:
                if processed[i]["score"] >= processed[j]["score"]:
                    drop.add(j)
                else:
                    drop.add(i)
    filtered = [p for idx,p in enumerate(processed) if idx not in drop]
    print(f"Filtered {len(processed)-len(filtered)} embedding duplicates; {len(filtered)} items remain")

    # -----------------
    # 5) DataFrame & export
    # -----------------
    df = pd.DataFrame(filtered)
    def parse_dt(x):
        if not x: return pd.NaT
        try:
            return pd.to_datetime(x)
        except Exception:
            try:
                return pd.to_datetime(dateparser.parse(x))
            except Exception:
                return pd.NaT
    df["published_dt"] = df["published"].apply(parse_dt)
    df = df.sort_values(["published_dt","score"], ascending=[False,False])

    ensure_outdir()
    outfn = os.path.join(OUTPUT_DIR, f"exosome_deals_{dt.datetime.utcnow().strftime('%Y_%m_%d')}.xlsx")
    df_export = df.copy()

    # Convert lists to clean strings
    list_columns = ["companies", "amounts", "amounts_numeric", "indications"]
    for col in list_columns:
        df_export[col] = df_export[col].apply(lambda x: "; ".join(map(str, x)) if isinstance(x, (list, tuple)) else x)

    # Short summary should just be text
    df_export["short_summary"] = df_export["short_summary"].apply(lambda x: str(x) if x else "")

    # Full text truncated to avoid huge cells
    df_export["full_text"] = df_export["full_text"].apply(lambda x: x[:1000] if x else "")

    # Ensure published_dt is clean (no timezone info)
    df_export["published_dt"] = pd.to_datetime(df_export["published_dt"]).dt.tz_localize(None)

    # Save Excel
    df_export.to_excel(outfn, index=False)
    print("Exported to", outfn)
    return df_export

# -----------------
# Run the agent
# -----------------
if __name__ == "__main__":
    print("----- Starting monthly run for NeuroCell Intelligence (SMTP 465) -----")
    df_export = run_agent()
    
    if df_export is not None and not df_export.empty:
        # Sort by score for email and select TOP_N_TO_EMAIL
        df_email = df_export.sort_values("score", ascending=False).head(TOP_N_TO_EMAIL)

        subject = f"Exosome Deals — Summary (last {SINCE_DAYS} days)"
        body_lines = [
            f"Exosome Deals — Summary (last {SINCE_DAYS} days)",
            f"Generated: {dt.datetime.utcnow().isoformat()}",
            ""
        ]
        
        for _, row in df_email.iterrows():
            event_label = row['event_type'].upper() if row['event_type'] else "NEWS"
            date_str = pd.to_datetime(row['published_dt']).strftime('%Y-%m-%d') if pd.notnull(row['published_dt']) else "Unknown"
            
            body_lines.append(f"- [{event_label}] {row['title']}")
            body_lines.append(f"  Date: {date_str}")
            body_lines.append(f"  Companies: {row['companies']}")
            body_lines.append(f"  Amounts: {row['amounts']}")
            body_lines.append(f"  Indications: {row['indications']}")
            body_lines.append(f"  Summary: {row['short_summary']}")
            body_lines.append(f"  Link: {row['url']}")
            body_lines.append("")
        
        body = "\n".join(body_lines)

        outfn = os.path.join(OUTPUT_DIR, f"exosome_deals_{dt.datetime.utcnow().strftime('%Y_%m_%d')}.xlsx")
        send_email_with_attachment(subject, body, outfn)
    else:
        print("No deals found — skipping email.")
    
    print("----- Run completed -----")

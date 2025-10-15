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

# ---------------------------------------
# 🔐 Load environment variables
# ---------------------------------------
load_dotenv()
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")

# ---------------------------------------
# 📁 Configuration
# ---------------------------------------
OUTPUT_DIR = "./industry_deals"
SINCE_DAYS = 40
TOP_N_TO_EMAIL = 10

RSS_FEEDS = [
    # Biotech/pharma specific feeds
    "https://www.fiercebiotech.com/rss.xml",
    "https://endpts.com/feed/",
    "https://www.biospace.com/rss",
    "https://www.genengnews.com/feed/",
    "https://www.labiotech.eu/feed/",
    
    # Business wire feeds
    "https://www.businesswire.com/portal/site/home/news/subject/landing/biotechnology",
    "https://www.prnewswire.com/rss/health-care-latest-news/health-care-latest-news-list.rss",
    
    # Google News search focused on exosomes deals
    "https://news.google.com/rss/search?q=exosome+(acquisition+OR+funding+OR+partnership)&hl=en-US&gl=US&ceid=US:en",
]

PR_PAGES = []

# Expanded indication keywords
INDICATION_KEYWORDS = [
    # Neurological
    "neurology","neuro","stroke","als","amyotrophic","parkinson","spinal cord","neurodegeneration",
    
    # General therapeutic areas
    "regenerat","regeneration","repair","rejuvenat","therapeutic",
    "cancer","oncology","tumor","carcinoma",
    "cardiovascular","cardiac","heart","myocardial",
    "inflammatory","autoimmune","immune",
    "kidney","renal","liver","hepatic",
    "lung","pulmonary","respiratory",
    
    # Diagnostic applications
    "diagnostic","biomarker","detection","screening",
    "liquid biopsy","early detection",
    
    # Drug delivery
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
    try:
        f = feedparser.parse(url)
        return f.entries
    except Exception as e:
        print("RSS error", url, e)
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

def normalize_amount(text):
    if not text or not isinstance(text, str):
        return None
    t = text.lower().strip()
    is_usd = '$' in t or 'usd' in t
    num_match = re.search(r'(\d+(?:,\d{3})*(?:\.\d+)?)', t)
    if not num_match:
        return None
    num_str = num_match.group(1).replace(',', '')
    try:
        num = float(num_str)
    except Exception:
        return None
    if re.search(r'\b(billion|bn|b)\b', t):
        num *= 1_000_000_000
    elif re.search(r'\b(million|m)\b', t):
        num *= 1_000_000
    elif re.search(r'\b(thousand|k)\b', t):
        num *= 1_000
    try:
        return int(round(num))
    except Exception:
        return None

def extract_amounts(text):
    if not text:
        return []

    patterns = [
        r'(\$\s?\d{1,3}(?:[,\d{3}]*)?(?:\.\d+)?\s?(?:million|billion|thousand|M|B|k|bn)?)',
        r'((?:USD|usd)\s?\d+(?:,\d{3})*(?:\.\d+)?\s?(?:million|billion|M|B|bn|k)?)',
        r'((?:€|EUR|eur)\s?\d+(?:,\d{3})*(?:\.\d+)?\s?(?:million|billion|M|B|bn|k)?)',
        r'(\d+(?:,\d{3})*(?:\.\d+)?\s?(?:million|billion|thousand|M|B|bn|k)\s?(?:usd|dollars?)?)',
    ]

    matches = []
    for pat in patterns:
        for m in re.finditer(pat, text, flags=re.I):
            amt = m.group(0).strip()
            amt = re.sub(r'\s+', ' ', amt)
            if not amt.startswith(('$', '€')) and re.search(r'\b(usd|dollars?)\b', amt, flags=re.I):
                amt = '$' + amt
            matches.append(amt)

    seen = set()
    unique = []
    for m in matches:
        key = re.sub(r'[^\d.]', '', m)
        if key not in seen and key:
            seen.add(key)
            unique.append(m)

    return unique[:5]

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
# 🧭 Main pipeline
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
        entries = fetch_rss_entries(rss)
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

    # 2) PR pages
    for name, pr_url in PR_PAGES:
        try:
            r = requests.get(pr_url, timeout=8)
            soup = BeautifulSoup(r.text, "html.parser")
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if any(x in href.lower() for x in ["press","news","releases","press-release"]):
                    link = href if href.startswith("http") else requests.compat.urljoin(pr_url, href)
                    collected.append({"title": a.get_text(strip=True),"link": link,"published": None,"summary": ""})
        except Exception as e:
            print("PR page error", pr_url, e)

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
        if not text or len(text) < 200:
            full_text = summary if len(summary) > 50 else title
        else:
            full_text = text

        if not is_exosome_relevant(full_text, title):
            continue

        short = summarize_short(full_text, max_sent=2)
        event = classify_event(full_text + " " + title)

        # Extract companies
        if "acqui" in title.lower() or event == "acquisition":
            companies = extract_acquisition_details(title, full_text)
            if not companies or len(companies) < 2:
                companies_from_text = extract_companies(full_text)
                companies_from_title = extract_companies(title + " " + summary)
                companies = list(dict.fromkeys(companies_from_text + companies_from_title))[:5]
        else:
            companies_from_text = extract_companies(full_text)
            companies_from_title = extract_companies(title + " " + summary)
            companies = list(dict.fromkeys(companies_from_text + companies_from_title))[:5]

        # Extract amounts
        all_money = extract_amounts(full_text) + extract_amounts(title) + extract_amounts(summary)
        money = list(dict.fromkeys(all_money))[:3]

        # Normalize numeric
        amounts_numeric = []
        for m in money:
            n = normalize_amount(m)
            if n is not None:
                amounts_numeric.append(n)
        amounts_numeric = list(dict.fromkeys(amounts_numeric))

        # Fallback for missing amounts
        if not money and event in ["acquisition", "funding"]:
            fallback_pattern = r'\b\d+\.?\d*\s?(?:million|billion|M|B|bn|k)\b'
            fallback_matches = re.findall(fallback_pattern, title + " " + summary + " " + full_text, re.I)
            if fallback_matches:
                f_matches = ['$' + m for m in fallback_matches[:2]]
                money = f_matches
                for m in f_matches:
                    n = normalize_amount(m)
                    if n is not None:
                        amounts_numeric.append(n)

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
    df_export = run_agent()
    if df_export is not None and not df_export.empty:
        # Compose email in "old style" detailed format
        subject = f"Exosome Deals — Summary (last {SINCE_DAYS} days)"
        body_lines = [
            f"Exosome Deals — Summary (last {SINCE_DAYS} days)",
            f"Generated: {dt.datetime.utcnow().isoformat()}",
            ""
        ]
        
        for _, row in df_export.iterrows():
            event_label = row['event_type'].upper() if row['event_type'] else "NEWS"
            date_str = row['published_dt'].strftime('%Y-%m-%d') if pd.notnull(row['published_dt']) else "Unknown"
            companies = row['companies'] if isinstance(row['companies'], str) else "; ".join(row['companies'])
            amounts = row['amounts'] if isinstance(row['amounts'], str) else "; ".join(map(str, row['amounts']))
            indications = row['indications'] if isinstance(row['indications'], str) else "; ".join(row['indications'])
            summary = row['short_summary'] if row['short_summary'] else ""
            link = row['url'] if row['url'] else ""
            
            body_lines.append(f"- [{event_label}] {row['title']}")
            body_lines.append(f"  Date: {date_str}")
            body_lines.append(f"  Companies: {companies}")
            body_lines.append(f"  Amounts: {amounts}")
            body_lines.append(f"  Indications: {indications}")
            body_lines.append(f"  Summary: {summary}")
            body_lines.append(f"  Link: {link}")
            body_lines.append("")  # blank line between entries
        
        body = "\n".join(body_lines)

        # Send the email with the Excel attachment
        outfn = os.path.join(OUTPUT_DIR, f"exosome_deals_{dt.datetime.utcnow().strftime('%Y_%m_%d')}.xlsx")
        send_email_with_attachment(subject, body, outfn)
    else:
        print("No deals found — skipping email.")

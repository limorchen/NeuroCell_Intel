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
    "bpartnership", "clinic utoquai", "swiss derma clinic", "laclinique", "exogems"
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

# ---------- NEW and IMPROVED amount extraction + normalization ----------
def normalize_amount(text):
    """
    Normalize amount strings to integer number of USD (approx):
    e.g. "$15 million" -> 15000000
          "$5M" -> 5000000
          "€10M" -> None (we do not convert currencies automatically, return None or keep as-is)
    Returns int or None when cannot normalize.
    """
    if not text or not isinstance(text, str):
        return None
    t = text.lower().strip()
    # remove currency symbols for numeric extraction but keep currency marker
    is_usd = '$' in t or 'usd' in t
    # find numeric part
    num_match = re.search(r'(\d+(?:,\d{3})*(?:\.\d+)?)', t)
    if not num_match:
        return None
    num_str = num_match.group(1).replace(',', '')
    try:
        num = float(num_str)
    except Exception:
        return None
    # multipliers
    if re.search(r'\b(billion|bn|b)\b', t):
        num *= 1_000_000_000
    elif re.search(r'\b(million|m)\b', t):
        num *= 1_000_000
    elif re.search(r'\b(thousand|k)\b', t):
        num *= 1_000
    # Only return numeric if currency is USD or there's a $ sign OR user prefers to store number anyway
    # We'll return numeric regardless of symbol — but user should be aware numbers are as-written (not currency-converted)
    try:
        return int(round(num))
    except Exception:
        return None

def extract_money(text):
    """Extract monetary amounts with better pattern matching and context.
       Returns list of amount strings (as found/normalized) — e.g. ['$15 million', '$5 million']"""
    if not text:
        return []
    # Patterns to capture:
    patterns = [
        # $15 million, $15M, $15.5M, $1,200,000.00
        r'(\$\s?\d{1,3}(?:[,\d{3}]*)(?:\.\d+)?\s?(?:million|billion|thousand|M|B|k|bn)?)',
        # USD 15 million
        r'((?:USD|usd)\s?\d+(?:,\d{3})*(?:\.\d+)?\s?(?:million|billion|M|B|bn|k)?)',
        # €10M or EUR 10 million
        r'((?:€|EUR|eur)\s?\d+(?:,\d{3})*(?:\.\d+)?\s?(?:million|billion|M|B|bn|k)?)',
        # 15 million USD / 15 million dollars
        r'(\d+(?:,\d{3})*(?:\.\d+)?\s?(?:million|billion|thousand|M|B|bn|k)\s?(?:usd|dollars?)?)',
    ]
    matches = []
    lowered = text  # preserve original for matched substrings
    for pat in patterns:
        for m in re.finditer(pat, lowered, flags=re.I):
            amt = m.group(0).strip()
            # normalize whitespace
            amt = re.sub(r'\s+', ' ', amt)
            matches.append(amt)
    # deduplicate preserving order
    seen = set()
    unique = []
    for m in matches:
        key = re.sub(r'[^0-9]', '', m)  # rough key by digits
        if key not in seen:
            seen.add(key)
            unique.append(m)
    return unique[:5]

def extract_amount_from_title(title):
    """Extract amount specifically from title with context and return normalized strings."""
    if not title:
        return []
    patterns = [
        r'for\s+\$?\d+\.?\d*\s?(?:million|billion|M|B|bn|k)\b',
        r'\$\d+\.?\d*\s?(?:million|billion|M|B|bn|k)\b',
        r'\d+\.?\d*\s?(?:million|billion|M|B|bn|k)\s+(?:deal|funding|investment|acquisition|raise|raised)',
        r'€\d+\.?\d*\s?(?:million|billion|M|B|bn|k)\b',
        r'USD\s?\d+\.?\d*\s?(?:million|billion|M|B|bn|k)?\b',
    ]
    found = []
    for pat in patterns:
        m = re.search(pat, title, flags=re.I)
        if m:
            amt = m.group(0)
            # remove leading "for "
            amt = re.sub(r'^\s*for\s+', '', amt, flags=re.I)
            amt = amt.strip()
            if not amt.startswith(('$', '€')) and not re.search(r'\b(usd|eur)\b', amt, flags=re.I):
                # if missing symbol, prepend $ for consistency — but keep in mind this is heuristic
                amt = '$' + amt
            found.append(amt)
    return found

# -----------------------------------------------------------------------

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
    """Aggressive normalization for deduplication"""
    title = re.split(r'\s*[-–—]\s*', title)[0]
    for word in ['announces', 'completes', 'closes', 'closing', 'announces closing']:
        title = re.sub(r'\b' + word + r'\b', '', title, flags=re.I)
    title = re.sub(r'[^\w\s]', '', title.lower()).strip()
    title = re.sub(r'\s+', ' ', title)
    return title

def is_exosome_relevant(text, title):
    """Check if content is about exosomes/EVs"""
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

    # 3) Process
    processed = []
    for item in collected:
        title = item.get("title","")
        url = item.get("link","")
        pub = item.get("published")
        summary = item.get("summary","") or ""
        
        # Try to fetch article text with trafilatura
        text = fetch_article_text(url) if url else ""
        
        # Use summary if article fetch failed or text too short
        if not text or len(text) < 200:
            full_text = summary if len(summary) > 50 else title
            print(f"📰 Using summary for: {title[:50]}...")
        else:
            full_text = text
            print(f"✅ Fetched article ({len(text)} chars): {title[:50]}...")

        # Filter non-exosome content
        if not is_exosome_relevant(full_text, title):
            continue

        short = summarize_short(full_text, max_sent=2)
        
        # Classify event first
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
        
        # Extract money aggressively from all sources
        money_from_text = extract_money(full_text)
        money_from_title = extract_amount_from_title(title)
        money_from_summary = extract_money(summary)
        all_money = money_from_text + money_from_title + money_from_summary
        money = list(dict.fromkeys(all_money))[:3]

        # Normalize amounts to numeric values where possible
        amounts_numeric = []
        for m in money:
            n = normalize_amount(m)
            if n is not None:
                amounts_numeric.append(n)
        # Deduplicate numbers
        amounts_numeric = list(dict.fromkeys(amounts_numeric))

        # Fallback for acquisitions/funding with no amount
        if not money and event in ["acquisition", "funding"]:
            fallback_pattern = r'\b\d+\.?\d*\s?(?:million|billion|M|B|bn|k)\b'
            fallback_matches = re.findall(fallback_pattern, title + " " + summary + " " + full_text, re.I)
            if fallback_matches:
                f_matches = ['$' + m for m in fallback_matches[:2]]
                money = f_matches
                # normalize fallback
                for m in f_matches:
                    n = normalize_amount(m)
                    if n is not None:
                        amounts_numeric.append(n)

        indications = detect_indications(full_text + " " + title + " " + summary)
        
        # Debug logging
        if event in ["acquisition", "funding"]:
            if money:
                print(f"💰 Found amount: {money} (numeric: {amounts_numeric}) for {title[:80]}...")
            else:
                print(f"⚠️ No amount found for: {title[:60]}...")
                # show snippet for debugging
                snippet = (full_text[:500] + "...") if len(full_text) > 500 else full_text
                print("🔸 Text snippet for inspection:", snippet)

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

    # 4) Dedupe via embeddings
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

    # 5) DataFrame
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

    # 6) Export
    ensure_outdir()
    outfn = os.path.join(OUTPUT_DIR, f"exosome_deals_{dt.datetime.utcnow().strftime('%Y_%m_%d')}.xlsx")
    df_export = df.copy()
    # remove tz to make Excel happy
    df_export["published_dt"] = df_export["published_dt"].dt.tz_localize(None)
    df_export["companies"] = df_export["companies"].apply(lambda x: "; ".join(x) if isinstance(x,list) else x)
    df_export["amounts"] = df_export["amounts"].apply(lambda x: "; ".join(map(str,x)) if isinstance(x,list) else x)
    # amounts_numeric column as semicolon-joined numbers (for Excel) and also a separate column for first numeric value
    df_export["amounts_numeric"] = df_export["amounts_numeric"].apply(lambda x: "; ".join(map(str,x)) if isinstance(x,list) else x)
    df_export["first_amount_numeric"] = df_export["amounts_numeric"].apply(lambda s: int(s.split(";")[0]) if isinstance(s,str) and s.strip() and s.split(";")[0].isdigit() else None)
    df_export["indications"] = df_export["indications"].apply(lambda x: "; ".join(x) if isinstance(x,list) else x)
    df_export[["published_dt","title","url","event_type","companies","amounts","amounts_numeric","first_amount_numeric","indications","short_summary","score"]].to_excel(outfn, index=False)
    print("Wrote", outfn)

    # 7) Email
    top = df_export.head(TOP_N_TO_EMAIL)
    lines = [f"Exosome Deals — Summary (last {SINCE_DAYS} days)\nGenerated: {dt.datetime.utcnow().isoformat()}\n"]
    for _, r in top.iterrows():
        date = r["published_dt"].strftime("%Y-%m-%d") if pd.notnull(r["published_dt"]) else "N/A"
        lines.append(f"- [{r['event_type'].upper()}] {r['title']}")
        lines.append(f"  Date: {date}")
        lines.append(f"  Companies: {r['companies']}")
        lines.append(f"  Amounts: {r['amounts']}")
        lines.append(f"  Amounts (numeric): {r.get('amounts_numeric', '')}")
        lines.append(f"  First amount (numeric): {r.get('first_amount_numeric', '')}")
        lines.append(f"  Indications: {r['indications']}")
        lines.append(f"  Summary: {r['short_summary']}")
        lines.append(f"  Link: {r['url']}\n")

    body = "\n".join(lines)
    send_email_with_attachment(
        subject=f"Exosome Deals — {dt.datetime.utcnow().strftime('%B %Y')}",
        body=body,
        attachment_path=outfn
    )

    return outfn

# ---------------------------------------
# 🚀 Run
# ---------------------------------------
if __name__ == "__main__":
    run_agent()

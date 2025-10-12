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
from html import unescape

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
    "https://news.google.com/rss/search?q=(exosome+OR+exosomes+OR+%22extracellular+vesicles%22+OR+%22EV+therapy%22)+(acquisition+OR+partnership+OR+licensing+OR+funding+OR+deal+OR+raised)&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=exosome+(company+OR+biotech+OR+therapeutics)+(funding+OR+investment+OR+series)&hl=en-US&gl=US&ceid=US:en",
    "https://www.fiercebiotech.com/rss.xml",
    "https://endpts.com/feed/",
]

PR_PAGES = []

INDICATION_KEYWORDS = [
    "neurology","neuro","stroke","als","amyotrophic","parkinson","spinal cord","neurodegeneration",
    "regenerat","regeneration","repair","rejuvenat","therapeutic"
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
    "regeneveda", "mdxhealth", "bio-techne"
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

def fetch_article_text(url, timeout=8):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, timeout=timeout, headers=headers)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        article = soup.find('article')
        if article:
            text = article.get_text(" ", strip=True)
        else:
            paras = soup.find_all('p')
            text = " ".join(p.get_text(" ", strip=True) for p in paras)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    except Exception:
        return ""

def extract_companies(text):
    doc = nlp(text)
    orgs = []
    for ent in doc.ents:
        if ent.label_ == "ORG":
            t = ent.text.strip()
            if len(t) < 2:
                continue
            if len(t.split()) > 6:
                continue
            if any(word in t.lower() for word in ["biotech", "cell", "gene", "therapy", "venture", "capital", "diagnostics"]):
                continue
            orgs.append(t)
    return list(dict.fromkeys(orgs))

def extract_money(text):
    patterns = [
        r"\$\s?[0-9\.,]+\s?(million|billion|bn|m|k)?",
        r"[0-9\.,]+\s?(million|billion|bn|m|k)\s+(usd|dollars|eur|€|\$)?",
        r"USD\s?[0-9\.,]+\s?(million|billion|m|bn)?"
    ]
    matches = []
    for p in patterns:
        for m in re.finditer(p, text, flags=re.I):
            matches.append(m.group(0))
    return list(dict.fromkeys(matches))

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
    """Normalize title for deduplication"""
    # Remove source attribution (e.g., "- MSN", "- The Manila Times")
    title = re.split(r'\s*[-–—]\s*[A-Z][a-z]+', title)[0]
    # Convert to lowercase and remove punctuation
    title = re.sub(r'[^\w\s]', '', title.lower()).strip()
    return title

def is_exosome_relevant(text, title):
    combined = (title + " " + text).lower()
    
    # Filter out spam/promotional content
    SPAM_TERMS = [
        "webinar", "sponsored", "whitepaper", "advertise", "iqvia", "syngene",
        "sign up to read", "subscribe", "newsletter",
        "market research", "market size", "market report", "market insights",
        "pipeline insights", "precedence research", "openpr.com",
        "download", "reportlinker", "press release distribution", "forecast"
    ]
    if any(term in combined for term in SPAM_TERMS):
        return False

    exosome_terms = [
        "exosome", "exosomes",
        "extracellular vesicle", "extracellular vesicles",
        "exosomal", "ev therapy", " evs"
    ]

    title_hits = sum(term in title.lower() for term in exosome_terms)
    company_match = any(comp.lower() in combined for comp in EXOSOME_COMPANIES)

    # Must have exosome terms in title OR be a known exosome company
    if title_hits == 0 and not company_match:
        return False

    # RELAXED: Accept if we have title match, even with short text
    if title_hits > 0:
        return True
    
    # For company matches, require some content
    if company_match and len(text) >= 100:  # Reduced from 300
        return True

    return False

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

    # 1) RSS - with HTML cleaning
    for rss in RSS_FEEDS:
        entries = fetch_rss_entries(rss)
        for e in entries:
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
            
            # Clean HTML from summary
            raw_summary = e.get("summary","") or e.get("description","")
            clean_summary = BeautifulSoup(raw_summary, "html.parser").get_text(strip=True)
            
            collected.append({
                "title": e.get("title",""),
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

    # Dedupe by URL first
    uniq = {}
    for c in collected:
        key = (c.get("link") or c.get("title")).strip()
        if key and key not in uniq:
            uniq[key] = c
    collected = list(uniq.values())
    print(f"After URL deduplication: {len(collected)} items")

    # Dedupe by normalized title
    title_map = {}
    for item in collected:
        norm_title = normalize_title(item.get("title", ""))
        if norm_title and len(norm_title) > 10 and norm_title not in title_map:
            title_map[norm_title] = item
    collected = list(title_map.values())
    print(f"After title deduplication: {len(collected)} items")

    # 3) Process
    processed = []
    for item in collected:
        title = item.get("title","")
        url = item.get("link","")
        pub = item.get("published")
        summary = item.get("summary","") or ""
        text = fetch_article_text(url) if url else ""
        full_text = text if text else summary if summary else title

        # Filter non-exosome content
        if not is_exosome_relevant(full_text, title):
            continue

        short = summarize_short(full_text, max_sent=2)
        
        # Extract companies from both full text and title/summary
        companies_from_text = extract_companies(full_text)
        companies_from_title = extract_companies(title + " " + summary)
        companies = list(dict.fromkeys(companies_from_text + companies_from_title))[:5]  # Max 5
        
        money = extract_money(full_text)
        event = classify_event(full_text + " " + title)
        indications = detect_indications(full_text + " " + title)
        
        # Scoring with exosome boost
        score = (1.5 if event in ["acquisition","partnership","licensing","funding"] else 0.2)
        score += 0.5 * len(indications)
        score += 0.8 if money else 0.0
        score += 0.2 * len(companies)
        
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
            "indications": indications,
            "short_summary": short,
            "full_text": full_text,
            "score": score
        })

    print(f"After relevance filtering: {len(processed)} items")

    if not processed:
        print("No processed items found.")
        return None

    # 4) Dedupe via embeddings (more aggressive)
    texts = [p["title"] + " " + p["short_summary"] for p in processed]
    emb = embedder.encode(texts)
    sim = cosine_similarity(emb)
    n = len(processed)
    drop = set()
    for i in range(n):
        for j in range(i+1, n):
            if sim[i,j] > 0.80:  # More aggressive threshold
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
    df_export["published_dt"] = df_export["published_dt"].dt.tz_localize(None)
    df_export["companies"] = df_export["companies"].apply(lambda x: "; ".join(x) if isinstance(x,list) else x)
    df_export["amounts"] = df_export["amounts"].apply(lambda x: "; ".join(x) if isinstance(x,list) else x)
    df_export["indications"] = df_export["indications"].apply(lambda x: "; ".join(x) if isinstance(x,list) else x)
    df_export[["published_dt","title","url","event_type","companies","amounts","indications","short_summary","score"]].to_excel(outfn, index=False)
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



from dotenv import load_dotenv
import os
load_dotenv()

# read config from environment (fallbacks are shown)
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.example.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")
EMAIL_FROM = os.getenv("EMAIL_FROM", SMTP_USER)
EMAIL_TO = os.getenv("EMAIL_TO", "alerts@example.com").split(",")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")


import os, re, json, datetime as dt
from dateutil import parser as dateparser
import requests, feedparser
from bs4 import BeautifulSoup
import pandas as pd
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import smtplib
from email.message import EmailMessage

# -------------- CONFIG --------------
OUTPUT_DIR = "./industry_deals"
SINCE_DAYS = 40  # window for scraping (monthly ~30)
TOP_N_TO_EMAIL = 10

RSS_FEEDS = [
    # Focused search via Google News RSS (strong starter)
    "https://news.google.com/rss/search?q=exosome+OR+exosomes+(acquir*+OR+partner*+OR+licens*+OR+funding+OR+collaborat*+OR+deal)+AND+(neurology+OR+neuro+OR+stroke+OR+ALS+OR+Parkinson+OR+regenerat*)&hl=en-US&gl=US&ceid=US:en",
    "https://www.fiercebiotech.com/rss.xml",
    "https://endpts.com/feed/",
    "https://www.prnewswire.com/rss/all-news.xml",
    "https://www.businesswire.com/rss/home/all-news"
]

# add specific company PR pages (tuple: name, url)
PR_PAGES = [
    # ("CompanyName", "https://company.com/press-releases"),
]

# indication keywords (lowercase)
INDICATION_KEYWORDS = [
    "neurology","neuro","stroke","als","amyotrophic","parkinson","spinal cord","neurodegeneration",
    "regenerat","regeneration","repair","rejuvenat","therapeutic"
]

# deal-event keywords mapped to event type
EVENT_KEYWORDS = {
    "acquisition": ["acquir","acquisition","acquired","merger","merged","buyout","takeover"],
    "partnership": ["partner","partnership","collaborat","alliance","strategic relationship"],
    "licensing": ["license","licensing","licensed","in-license","out-license"],
    "funding": ["funding","raised","series a","series b","grant","investment","seed","financ","venture"],
    "deal": ["deal", "agreement","term sheet","option agreement","commercialization"]
}

# email config (read from env in production/GHA)
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.example.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
SMTP_USER = os.getenv("SMTP_USER", "user@example.com")
SMTP_PASS = os.getenv("SMTP_PASS", "password")
EMAIL_FROM = os.getenv("EMAIL_FROM", SMTP_USER)
EMAIL_TO = os.getenv("EMAIL_TO", "recipient@example.com").split(",")

# -------------- models --------------
nlp = spacy.load("en_core_web_sm")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# -------------- helpers --------------
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
        # prefer article tag, fallback to paragraphs
        article = soup.find('article')
        if article:
            text = article.get_text(" ", strip=True)
        else:
            paras = soup.find_all('p')
            text = " ".join(p.get_text(" ", strip=True) for p in paras)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    except Exception as e:
        # print("fetch article", e)
        return ""

def extract_companies(text):
    doc = nlp(text)
    orgs = [ent.text.strip() for ent in doc.ents if ent.label_ == "ORG"]
    # simple cleanup + dedupe
    seen = set(); out=[]
    for o in orgs:
        key = o.lower()
        if len(o) > 1 and key not in seen:
            seen.add(key); out.append(o)
    return out

def extract_money(text):
    # patterns: $1.2B, $120 million, USD 50m, €30m, 10 million dollars
    patterns = [
        r"\$\s?[0-9\.,]+\s?(million|billion|bn|m|k)?",
        r"[0-9\.,]+\s?(million|billion|bn|m|k)\s+(usd|dollars|eur|€|\$)?",
        r"USD\s?[0-9\.,]+\s?(million|billion|m|bn)?"
    ]
    matches=[]
    for p in patterns:
        for m in re.finditer(p, text, flags=re.I):
            matches.append(m.group(0))
    # return unique matches
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
    # naive: first N sentences
    sents = re.split(r'(?<=[.!?])\s+', text)
    return " ".join(sents[:max_sent]).strip()

# -------------- pipeline --------------
def run_agent():
    ensure_outdir()
    since = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=SINCE_DAYS)
    collected = []

    # 1) RSS
    for rss in RSS_FEEDS:
        entries = fetch_rss_entries(rss)
        for e in entries:
            pub = e.get("published") or e.get("updated") or e.get("pubDate")
            try:
                pub_dt = dateparser.parse(pub) if pub else None
            except Exception:
                pub_dt = None
            if pub_dt:
                # Make pub_dt timezone-aware if it isn't
                if pub_dt.tzinfo is None:
                    pub_dt = pub_dt.replace(tzinfo=dt.timezone.utc)
                if pub_dt < since:
                   continue
            collected.append({
                "title": e.get("title",""),
                "link": e.get("link",""),
                "published": pub_dt.isoformat() if pub_dt else None,
                "summary": e.get("summary","") or e.get("description","")
            })

    # 2) PR pages (simple link gather)
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

    # dedupe by link/title
    uniq = {}
    for c in collected:
        key = (c.get("link") or c.get("title")).strip()
        if not key: continue
        if key not in uniq:
            uniq[key] = c
    collected = list(uniq.values())
    print(f"Collected {len(collected)} candidate items")

    # 3) Process each: fetch text, extract orgs, money, event, indications
    processed=[]
    for item in collected:
        title = item.get("title","")
        url = item.get("link","")
        pub = item.get("published")
        summary = item.get("summary","") or ""
        text = fetch_article_text(url) if url else ""
        full_text = text if text else summary if summary else title
        short = summarize_short(full_text, max_sent=2)
        companies = extract_companies(full_text)
        money = extract_money(full_text)
        event = classify_event(full_text + " " + title)
        indications = detect_indications(full_text + " " + title)
        score = 0
        # scoring: event presence + indication match + money + number of companies
        score += (1.5 if event in ["acquisition","partnership","licensing","funding"] else 0.2)
        score += 1.0 * len(indications)
        score += 0.8 if money else 0.0
        score += 0.2 * len(companies)
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

    if not processed:
        print("No processed items found.")
        return None

    # 4) dedupe/group similar items via embeddings (keep highest score)
    texts = [p["title"] + " " + p["short_summary"] for p in processed]
    emb = embedder.encode(texts)
    sim = cosine_similarity(emb)
    n = len(processed)
    drop=set()
    for i in range(n):
        for j in range(i+1, n):
            if sim[i,j] > 0.90:
                # keep higher score
                if processed[i]["score"] >= processed[j]["score"]:
                    drop.add(j)
                else:
                    drop.add(i)
    filtered = [p for idx,p in enumerate(processed) if idx not in drop]
    print(f"Filtered {len(processed)-len(filtered)} duplicates; {len(filtered)} items remain")

    # 5) DataFrame, sort by date desc then score
    df = pd.DataFrame(filtered)
    # parse dates
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

    # 6) Write chronologically (most recent first)
    ensure_outdir()
    outfn = os.path.join(OUTPUT_DIR, f"exosome_deals_{dt.datetime.utcnow().strftime('%Y_%m_%d')}.xlsx")
    # Expand lists to semicolon strings for Excel
    df_export = df.copy()
    df_export["companies"] = df_export["companies"].apply(lambda x: "; ".join(x) if isinstance(x,list) else x)
    df_export["amounts"] = df_export["amounts"].apply(lambda x: "; ".join(x) if isinstance(x,list) else x)
    df_export["indications"] = df_export["indications"].apply(lambda x: "; ".join(x) if isinstance(x,list) else x)
    df_export[["published_dt","title","url","event_type","companies","amounts","indications","short_summary","score"]].to_excel(outfn, index=False)
    print("Wrote", outfn)

    # 7) Compose email summary with top N items
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
    send_email_with_attachment(f"Exosome Deals — {dt.datetime.utcnow().strftime('%B %Y')}", body, outfn)
    return outfn

# -------------- email helper --------------
def send_email_with_attachment(subject, body, attachment_path):
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = EMAIL_FROM
    msg["To"] = ", ".join(EMAIL_TO)
    msg.set_content(body)
    # attach file
    with open(attachment_path, "rb") as f:
        data = f.read()
    msg.add_attachment(data, maintype="application", subtype="vnd.openxmlformats-officedocument.spreadsheetml.sheet", filename=os.path.basename(attachment_path))
    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
            s.starttls()
            s.login(SMTP_USER, SMTP_PASS)
            s.send_message(msg)
        print("Email sent to", EMAIL_TO)
    except Exception as e:
        print("Email send failed:", e)

# -------------- run --------------
if __name__ == "__main__":
    run_agent()

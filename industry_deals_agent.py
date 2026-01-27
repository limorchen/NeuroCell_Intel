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

# ---------------------------------------
# 📁 Configuration
# ---------------------------------------
OUTPUT_DIR = "./industry_deals"
SINCE_DAYS = 35
TOP_N_TO_EMAIL = 10
CUMULATIVE_FILENAME = "exosome_deals_DATABASE.xlsx"

RSS_FEEDS = [
    # Core Biotech/Pharma Feeds
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

    # Google News Searches - Exosome Deals
    "https://news.google.com/rss/search?q=exosome+(acquisition+OR+funding+OR+partnership)&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=%22extracellular+vesicles%22+(deal+OR+funding+OR+partnership)&hl=en-US",
    "https://news.google.com/rss/search?q=exosome+company+(raised+OR+secures+OR+closes)&hl=en-US",
    "https://news.google.com/rss/search?q=exosome+OR+%22extracellular+vesicle%22+AND+(neuro+OR+neurology+OR+regenerat)&hl=en-US&gl=US&ceid=US:en",

    # Specific Company & Therapy Searches
    "https://news.google.com/rss/search?q=%22exosome+therapy%22+OR+%22exosome+therapeutics%22&hl=en-US",
    "https://news.google.com/rss/search?q=Evox+OR+Capricor+OR+Codiak+OR+%22Direct+Biologics%22&hl=en-US",
    "https://news.google.com/rss/search?q=%22extracellular+vesicle%22+clinical+trial&hl=en-US",
]
PR_PAGES = []

# Refined SPAM Terms - More targeted
SPAM_TERMS = [
    # Events/Webinars
    "register for this webinar", "join this webinar", "register here",
    "join this session", "save the date", "rsvp",

    # Promotional content
    "sign up to read", "subscribe to unlock", "premium content access",
    "/premium/webinar", "fiercebiotech premium",

    # Reports/Analysis
    "download our report", "get the report", "annual report",
    "market forecast", "industry forecast",

    # Listicles/Roundups
    "top 5", "top 10", "biggest deals of", "year in review",
    "monthly recap", "weekly roundup", "what to expect in 20",
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

CORE_INTEREST_TERMS = ["neuro", "neurology", "regenerat", "repair", "therapeutic"]

EVENT_KEYWORDS = {
    "acquisition": ["acquir", "acquisition", "acquired", "merger", "merged", "buyout", "takeover"],
    "partnership": ["partner", "partnership", "collaborat", "alliance", "strategic relationship"],
    "licensing": ["license", "licensing", "licensed", "in-license", "out-license"],
    "funding": ["funding", "raised", "series a", "series b", "grant", "investment", "seed", "financ", "venture"],
    "deal": ["deal", "agreement", "term sheet", "option agreement", "commercialization"]
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
    "bpartnership", "clinic utoquai", "swiss derma clinic", "laclinique", "exogems", "ags therapeutics", "phoenestra gmbh",
    "exosla therapeutics", "tiny cargo company", "rion inc.", "exosomica", "exogenus therapeutics", "ev therapeutics",
    "nano24", "pandorum", "nucelion", "nippon shinyaku",
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

def resolve_google_news_url(google_url):
    """Extract the actual article URL from Google News redirect."""
    try:
        if 'news.google.com' in google_url and '/articles/' in google_url:
            response = requests.get(
                google_url,
                allow_redirects=True,
                timeout=10,
                headers={"User-Agent": "Mozilla/5.0"}
            )
            return response.url
        return google_url
    except Exception as e:
        print(f"Failed to resolve Google News URL: {str(e)[:50]}")
        return google_url

def fetch_article_text(url, timeout=15):
    """Fetch article text using requests + trafilatura."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; NeuroCellBot/1.0; +https://neurocellintel.ai)"}
        response = requests.get(url, headers=headers, timeout=timeout)
        if response.status_code != 200 or not response.text:
            return ""

        soup = BeautifulSoup(response.text, "html.parser")
        for script in soup(["script", "style", "noscript"]):
            script.decompose()
        cleaned_html = str(soup)

        text = trafilatura.extract(
            cleaned_html,
            include_comments=False,
            include_tables=False,
            favor_recall=True
        )

        if text and len(text) > 100:
            return text[:10000]
        else:
            return ""

    except Exception as e:
        print(f"Article fetch failed for {url[:50]}...: {str(e)[:50]}")
        return ""

# -----------------------------------------------------
# 💰 MONEY EXTRACTION FUNCTIONS
# -----------------------------------------------------
def normalize_amount(text):
    """Converts money string to a single integer amount."""
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
    """Extract all currency amounts from text."""
    if not text:
        return []

    amount_patterns = [
        r'[\$£€¥]\s?\d{1,3}(?:[,.\s]\d{3})*(?:\.\d+)?\s?(?:trillion|billion|million|thousand|m|b|k|bn|tn)?',
        r'(?:USD|EUR|GBP|CAD|AUD|usd|eur|gbp|cad|aud)\s?\d{1,3}(?:[,.\s]\d{3})*(?:\.\d+)?\s?(?:trillion|billion|million|thousand|m|b|k|bn|tn)?',
        r'\d{1,3}(?:[,.\s]\d{3})*(?:\.\d+)?\s?(?:trillion|billion|million|thousand|m|b|k|bn|tn)\s?(?:USD|usd|dollars?|EUR|eur|€|£|GBP|gbp)?',
        r'(?:approximately|about|around|nearly|up\s+to|over|valued\s+at|worth)\s+[\$£€¥]?\s?\d{1,3}(?:[,.\s]\d{3})*(?:\.\d+)?\s?(?:trillion|billion|million|thousand|m|b|k|bn|tn)?',
    ]

    matches = []
    for pat in amount_patterns:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            amt = m.group(0).strip()
            amt = re.sub(r'\s+', ' ', amt)
            amt = re.sub(
                r'^(?:approximately|about|around|nearly|up to|over|valued at|worth)\s+',
                '',
                amt,
                flags=re.I
            )
            matches.append(amt)

    # Deduplication by normalized values
    seen_values = {}
    unique = []
    for m in matches:
        normalized_val = normalize_amount(m)
        if normalized_val and normalized_val not in seen_values:
            seen_values[normalized_val] = m
            unique.append(m)
        elif not normalized_val:
            key = re.sub(r'[^\d.]', '', m.lower())
            if key and key not in [re.sub(r'[^\d.]', '', u.lower()) for u in unique]:
                unique.append(m)

    return unique

def extract_extended_deal_context(text, amount_str, window_size=300):
    """Extract surrounding context around a dollar amount."""
    if not text or not amount_str:
        return "", -1

    pattern = re.escape(amount_str)
    match = re.search(pattern, text, re.IGNORECASE)

    if match:
        start = max(0, match.start() - window_size)
        end = min(len(text), match.end() + window_size)
        context = text[start:end]
        relative_pos = match.start() - start
        return context, relative_pos

    return "", -1

def extract_deal_structure(text, amounts):
    """Extract detailed deal structure information."""
    if not amounts or not text:
        return ""

    text_lower = text.lower()
    structure_parts = []

    structure_patterns = {
        'total_value': [
            r'total (?:acquisition |deal |transaction )?(?:value|price|consideration)(?:\s+is|\s+of)?\s+(?:approximately\s+)?([£$€¥]?[\d.,]+\s*(?:million|billion|m|b|k|thousand)?)',
            r'(?:valued|priced|worth)\s+at\s+(?:approximately\s+)?([£$€¥]?[\d.,]+\s*(?:million|billion|m|b|k|thousand)?)',
            r'for\s+a\s+total\s+(?:of\s+)?([£$€¥]?[\d.,]+\s*(?:million|billion|m|b|k|thousand)?)',
            r'acquisition\s+(?:price\s+)?of\s+([£$€¥]?[\d.,]+\s*(?:million|billion|m|b|k|thousand)?)',
        ],
        'upfront': [
            r'([£$€¥][\d.,]+\s*(?:million|billion|m|b)?)\s+(?:paid\s+)?(?:at\s+)?(?:closing|upfront|immediately)',
            r'(?:upfront|initial)\s+payment\s+of\s+([£$€¥]?[\d.,]+\s*(?:million|billion|m|b|thousand)?)',
            r'([£$€¥][\d.,]+\s*(?:million|billion|m|b)?)\s+in\s+(?:cash|stock)(?:\s+(?:paid\s+)?at\s+closing)',
            r'(?:cash|stock)\s+payment\s+of\s+([£$€¥]?[\d.,]+\s*(?:million|billion|m|b)?)',
        ],
        'milestone': [
            r'up\s+to\s+([£$€¥]?[\d.,]+\s*(?:million|billion|m|b)?)\s+in\s+(?:milestone|contingent|earnout)',
            r'([£$€¥]?[\d.,]+\s*(?:million|billion|m|b)?)\s+in\s+(?:development|regulatory|commercial|sales)\s+milestones?',
            r'additional\s+([£$€¥]?[\d.,]+\s*(?:million|billion|m|b)?)\s+(?:in\s+)?(?:based\s+on|contingent|milestone)',
            r'milestone\s+payments?\s+(?:of\s+)?(?:up\s+to\s+)?([£$€¥]?[\d.,]+\s*(?:million|billion|m|b)?)',
        ],
        'equity': [
            r'([£$€¥]?[\d.,]+\s*(?:million|billion|m|b)?)\s+in\s+(?:stock|equity|shares)',
            r'([£$€¥]?[\d.,]+\s*(?:million|billion|m|b)?)\s+(?:of\s+)?(?:common\s+)?(?:stock|equity)',
            r'(?:stock|equity)\s+(?:valued\s+at|worth)\s+([£$€¥]?[\d.,]+\s*(?:million|billion|m|b)?)',
        ],
        'periodic': [
            r'([£$€¥]?[\d.,]+\s*(?:million|billion|m|b)?)\s+(?:paid\s+)?(?:annually|yearly|per\s+year)',
            r'([£$€¥]?[\d.,]+\s*(?:million|billion|m|b)?)\s+(?:over|in|during)\s+(?:the\s+)?(?:course\s+of\s+)?(\d+)\s+years?',
            r'([£$€¥]?[\d.,]+\s*(?:million|billion|m|b)?)\s+in\s+(\d+)\s+(?:annual\s+)?(?:installments|payments)',
            r'(\d+)\s+annual\s+payments?\s+of\s+([£$€¥]?[\d.,]+\s*(?:million|billion|m|b)?)',
        ],
        'royalty': [
            r'royalt(?:y|ies)\s+(?:of\s+)?(?:up\s+to\s+)?(\d+(?:\.\d+)?)\s*%',
            r'(\d+(?:\.\d+)?)\s*%\s+royalty',
            r'(?:single|mid|low|high)[\s-]digit\s+royalt(?:y|ies)',
        ]
    }

    found_components = {}

    for component_type, patterns in structure_patterns.items():
        for pattern in patterns:
            matches = list(re.finditer(pattern, text_lower, re.IGNORECASE))
            if matches:
                matched_text = matches[0].group(0)
                if component_type not in found_components:
                    found_components[component_type] = matched_text
                    break

    if 'total_value' in found_components:
        structure_parts.append(f"Total value: {found_components['total_value']}")
    if 'upfront' in found_components:
        structure_parts.append(f"Upfront: {found_components['upfront']}")
    if 'equity' in found_components:
        structure_parts.append(f"Equity: {found_components['equity']}")
    if 'milestone' in found_components:
        structure_parts.append(f"Milestones: {found_components['milestone']}")
    if 'periodic' in found_components:
        structure_parts.append(f"Periodic: {found_components['periodic']}")
    if 'royalty' in found_components:
        structure_parts.append(f"Royalty: {found_components['royalty']}")

    if structure_parts:
        return "; ".join(structure_parts)

    # Fallback: Extract sentences with amounts and deal keywords
    deal_keywords = [
        'acquisition', 'purchase', 'valued', 'worth', 'paid', 'payment',
        'closing', 'upfront', 'milestone', 'stock', 'cash', 'consideration',
        'transaction', 'financing', 'raised', 'secured', 'investment'
    ]

    sentences_with_amounts = []
    for sent in re.split(r'[.!?]+', text):
        sent_lower = sent.lower()
        has_amount = any(amt.lower() in sent_lower for amt in amounts[:3])
        has_deal_keyword = any(kw in sent_lower for kw in deal_keywords)

        if has_amount and has_deal_keyword:
            clean_sent = sent.strip()
            if 20 < len(clean_sent) < 200:
                sentences_with_amounts.append(clean_sent)

    if sentences_with_amounts:
        return " | ".join(sentences_with_amounts[:2])

    return ""

def validate_deal_amount(amount_str, context, event_type):
    """Validate if an extracted amount is likely a real deal amount."""
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

    normalized = normalize_amount(amount_str)
    if normalized:
        if 1_000_000 <= normalized <= 10_000_000_000:
            score += 0.2
        elif normalized < 100_000 or normalized > 100_000_000_000:
            score -= 0.3

    return max(0.0, min(1.0, score))

def extract_amounts_with_validation(title, text, summary, event_type):
    """Main extraction function combining extraction + validation + structure."""
    full_text = f"{title} {text} {summary}"
    all_amounts = extract_amounts(full_text)

    validated_amounts = []
    for amount in all_amounts:
        context, _ = extract_extended_deal_context(full_text, amount, window_size=300)
        confidence = validate_deal_amount(amount, context, event_type)

        if confidence >= 0.4:
            validated_amounts.append({
                'amount': amount,
                'confidence': confidence,
                'context': context[:300]
            })

    validated_amounts.sort(key=lambda x: x['confidence'], reverse=True)

    deal_structure = ""
    if validated_amounts:
        amounts_list = [va['amount'] for va in validated_amounts]
        deal_structure = extract_deal_structure(full_text, amounts_list)

    return validated_amounts, deal_structure

def search_for_deal_amount(title, companies, event_type):
    """Secondary web scraper for deal amount fallback."""
    if event_type not in ["acquisition", "funding"]:
        return []

    time.sleep(1)

    try:
        company_str = " ".join(companies[:2]) if companies else ""
        search_query = f"{company_str} {event_type} amount million"
        search_url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(search_query)}"

        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

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
# 📚 COMPANY EXTRACTION FUNCTIONS
# -----------------------------------------------------
def extract_companies_from_title(title):
    """Extract company names from title using pattern matching."""
    companies = []

    # Pattern: "Company1 Partners With Company2"
    partner_pattern = r'([A-Z][A-Za-z0-9]+)\s+(?:Partners|Collaborates|Teams)\s+(?:With|Up With)\s+([A-Z][A-Za-z0-9]+)'
    match = re.search(partner_pattern, title)
    if match:
        companies.extend([match.group(1), match.group(2)])

    # Pattern: "Company Announces/Reports/Completes"
    company_pattern = r'^([A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+)?)\s+(?:Announces|Reports|Completes|Partners|Signs)'
    match = re.search(company_pattern, title)
    if match:
        companies.append(match.group(1))

    # Pattern: ticker in parentheses like "(CAPR)"
    ticker_pattern = r'\(([A-Z]{2,5})\)'
    match = re.search(ticker_pattern, title)
    if match:
        pre_ticker = title.split('(')[0].strip()
        words = pre_ticker.split()
        if len(words) >= 2:
            companies.append(' '.join(words[-2:]))

    return [c for c in companies if len(c) > 2]

def extract_companies(text):
    """Extract company names using spaCy NER with aggressive filtering."""
    doc = nlp(text)
    orgs = []

    REMOVE_SUFFIXES = [
        " - tipranks", " tipranks", "the manila times", " - msn",
        " acquisition", " diagnostics acquisition"
    ]
    IGNORE_ORGS = [
        "msn", "manila times", "reuters", "bloomberg", "fiercebiotech",
        "endpoints", "yahoo", "google", "facebook", "twitter", "linkedin",
        "ap", "associated press", "wall street journal", "new york times",
        "cnn", "bbc", "fox news", "nbc", "cbs", "abc news", "tipranks",
        "globe newswire", "business wire", "pr newswire", "marketwatch",
        "seeking alpha", "motley fool", "benzinga", "zacks", "biospace",
        "genengnews", "labiotech", "fiercepharma"
    ]

    STOP_WORDS = {"the", "a", "an", "group", "company", "inc", "corp", "llc"}

    for ent in doc.ents:
        if ent.label_ == "ORG":
            t = ent.text.strip()
            for suffix in REMOVE_SUFFIXES:
                if t.lower().endswith(suffix):
                    t = t[:-len(suffix)].strip()
            if len(t) < 3:
                continue
            if len(t.split()) > 6:
                continue
            if t.lower() in IGNORE_ORGS or t.lower() in STOP_WORDS:
                continue
            if any(ignore in t.lower() for ignore in IGNORE_ORGS):
                continue
            if t.lower() in ["acquisition", "diagnostics", "acquisition from", "bio", "techne"]:
                continue
            orgs.append(t)

    seen = set()
    unique_orgs = []
    for org in orgs:
        org_lower = org.lower()
        if org_lower not in seen:
            seen.add(org_lower)
            unique_orgs.append(org)

    return unique_orgs[:5]

def extract_acquisition_details(title, text):
    """Manually extract acquisition details from text."""
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

# -----------------------------------------------------
# 📚 OTHER HELPER FUNCTIONS
# -----------------------------------------------------
def classify_event(text):
    """Classify the event type based on keywords."""
    tl = text.lower()
    scores = {}
    total_hits = 0
    for ev, kws in EVENT_KEYWORDS.items():
        hits = sum(1 for k in kws if k in tl)
        scores[ev] = hits
        total_hits += hits

    if total_hits == 0:
        return "news"

    best = max(scores.items(), key=lambda x: x[1])
    return best[0]

def detect_indications(text):
    """Detect therapeutic indications from text."""
    tl = text.lower()
    hits = []

    for kw in INDICATION_KEYWORDS:
        if kw in tl:
            hits.append(kw)

    # Simple normalization/grouping
    unique_hits = sorted(set(hits))
    return ", ".join(unique_hits) if unique_hits else ""

def is_spam_article(text):
    """Filter out irrelevant or obviously promotional content."""
    tl = text.lower()
    for term in SPAM_TERMS:
        if term in tl:
            return True
    return False

def contains_exosome_terms(text):
    """Check if text contains exosome-related terms."""
    tl = text.lower()
    return any(term in tl for term in EXOSOME_TERMS)

def contains_core_interest_terms(text):
    """Check for core neuro/regen/therapeutic interest terms."""
    tl = text.lower()
    return any(term in tl for term in CORE_INTEREST_TERMS)

def is_exosome_relevant(full_text, title):
    """Relaxed relevance filter: exosome terms OR companies OR strong neuro/regen context."""
    combined = f"{title} {full_text}".lower()

    if is_spam_article(combined):
        return False

    if contains_exosome_terms(combined):
        return True

    # Company name heuristic
    for cname in EXOSOME_COMPANIES:
        if cname.lower() in combined:
            return True

    # Core therapeutic interest without exosome terms: keep but lower priority later
    if contains_core_interest_terms(combined):
        return True

    return False

def compute_relevance_score(title, full_text):
    """Semantic/scoring model for ranking."""
    base_score = 0.0
    tl = (title + " " + full_text).lower()

    if contains_exosome_terms(tl):
        base_score += 0.6
    if contains_core_interest_terms(tl):
        base_score += 0.2
    for cname in EXOSOME_COMPANIES:
        if cname.lower() in tl:
            base_score += 0.2
            break

    # Clip
    base_score = min(1.0, base_score)

    return base_score

def parse_pubdate(entry):
    """Parse publication date from RSS entry."""
    pub = entry.get("published") or entry.get("updated") or entry.get("pubDate")
    if not pub:
        return None
    try:
        return dateparser.parse(pub)
    except Exception:
        return None

def within_days(dt_obj, days=SINCE_DAYS):
    if not dt_obj:
        return False
    cutoff = dt.datetime.utcnow() - dt.timedelta(days=days)
    return dt_obj > cutoff

def load_existing_cumulative(path):
    if os.path.exists(path):
        try:
            return pd.read_excel(path)
        except Exception as e:
            print(f"Failed to load existing cumulative file: {e}")
    return pd.DataFrame()

def save_cumulative(df, path):
    try:
        df.to_excel(path, index=False)
        print(f"✅ Saved cumulative DB to: {path}")
    except Exception as e:
        print(f"Failed to save cumulative DB: {e}")

def send_email_with_top_deals(df, top_n=TOP_N_TO_EMAIL):
    """Send summary email of top N deals."""
    smtp_host = os.getenv("SMTP_HOST")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER")
    smtp_pass = os.getenv("SMTP_PASS")
    to_email = os.getenv("TO_EMAIL")

    if not (smtp_host and smtp_user and smtp_pass and to_email):
        print("⚠️ Email not sent: SMTP or TO_EMAIL env vars missing.")
        return

    df_sorted = df.sort_values("RelevanceScore", ascending=False).head(top_n)
    lines = []
    for _, row in df_sorted.iterrows():
        line = (
            f"- {row.get('Date','')} | {row.get('EventType','')} | "
            f"{row.get('Title','')} | {row.get('Companies','')} | "
            f"{row.get('Amounts','')} | {row.get('URL','')}"
        )
        lines.append(line)

    body = "Top Exosome/EV Deals & Events\n\n" + "\n".join(lines)

    msg = EmailMessage()
    msg["Subject"] = "Exosome/EV Deals & Funding Digest"
    msg["From"] = smtp_user
    msg["To"] = to_email
    msg.set_content(body)

    try:
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)
        print("📧 Email summary sent.")
    except Exception as e:
        print(f"Failed to send email: {e}")

# -----------------------------------------------------
# 🚀 MAIN PIPELINE
# -----------------------------------------------------
def main():
    ensure_outdir()

    all_entries = []

    # 1) Collect RSS entries
    for url in RSS_FEEDS:
        print(f"🔎 Fetching RSS: {url}")
        entries = fetch_rss_entries(url)
        for e in entries:
            pub_dt = parse_pubdate(e)
            if not within_days(pub_dt, SINCE_DAYS):
                continue
            all_entries.append(e)

    print(f"Collected {len(all_entries)} recent entries.")

    # 2) Normalize collected items
    collected = []
    seen_urls = set()

    for e in all_entries:
        title = e.get("title", "")
        link = e.get("link") or e.get("id") or ""
        summary = e.get("summary", "") or ""
        pub = parse_pubdate(e)

        if link and link in seen_urls:
            continue
        seen_urls.add(link)

        collected.append({
            "title": title,
            "link": link,
            "summary": BeautifulSoup(summary, "html.parser").get_text(" ", strip=True),
            "published": pub
        })

    print(f"Normalized to {len(collected)} unique items.")

    # -----------------
    # 3) Process
    # -----------------
    processed = []
    for item in collected:
        title = item.get("title", "")
        url = item.get("link", "")
        pub = item.get("published")
        summary = item.get("summary", "") or ""

        # 🆕 Resolve Google News URLs before fetching
        if url:
            if 'news.google.com' in url:
                resolved_url = resolve_google_news_url(url)
                print(f"🔗 Resolved Google News URL: {resolved_url[:60]}...")
                url = resolved_url

            text = fetch_article_text(url)
        else:
            text = ""

        # Use the most comprehensive available text for filtering and processing
        if not text or len(text) < 200:
            full_text = summary if len(summary) > 50 else title
        else:
            full_text = text

        # CRITICAL FILTER: is_exosome_relevant
        if not is_exosome_relevant(full_text, title):
            # Optional: add debug here if desired
            continue

        event_type = classify_event(full_text)
        indications = detect_indications(full_text)

        # Company extraction
        companies = extract_companies_from_title(title)
        if len(companies) < 1:
            more_companies = extract_companies(full_text)
            for c in more_companies:
                if c not in companies:
                    companies.append(c)
        companies_str = ", ".join(companies) if companies else ""

        # Deal amounts and structure
        validated_amounts, deal_structure = extract_amounts_with_validation(
            title, full_text, summary, event_type
        )
        amounts_list = [va['amount'] for va in validated_amounts]
        amounts_str = "; ".join(amounts_list) if amounts_list else ""

        # Fallback amount search for key event types
        if not amounts_list and event_type in ["acquisition", "funding"]:
            fallback_amounts = search_for_deal_amount(title, companies, event_type)
            if fallback_amounts:
                amounts_str = "; ".join(fallback_amounts)

        relevance_score = compute_relevance_score(title, full_text)

        processed.append({
            "Date": pub.date().isoformat() if isinstance(pub, dt.datetime) else "",
            "Title": title,
            "URL": url,
            "Summary": summary,
            "EventType": event_type,
            "Indications": indications,
            "Companies": companies_str,
            "Amounts": amounts_str,
            "DealStructure": deal_structure,
            "RelevanceScore": relevance_score,
            "RawText": full_text[:5000]
        })

    df_new = pd.DataFrame(processed)
    print(f"Processed and kept {len(df_new)} items after relevance filtering.")

    # 4) Merge with cumulative DB
    cumulative_path = os.path.join(OUTPUT_DIR, CUMULATIVE_FILENAME)
    df_existing = load_existing_cumulative(cumulative_path)

    if not df_existing.empty:
        df_merged = pd.concat([df_existing, df_new], ignore_index=True)
        df_merged.drop_duplicates(subset=["Title", "URL"], keep="last", inplace=True)
    else:
        df_merged = df_new

    save_cumulative(df_merged, cumulative_path)

    # 5) Also save a dated snapshot of this run
    today_str = dt.datetime.utcnow().strftime("%Y%m%d")
    snapshot_path = os.path.join(OUTPUT_DIR, f"exosome_deals_run_{today_str}.xlsx")
    try:
        df_new.to_excel(snapshot_path, index=False)
        print(f"📁 Saved run snapshot to: {snapshot_path}")
    except Exception as e:
        print(f"Failed to save run snapshot: {e}")

    # 6) Optional: send email with top deals
    if not df_new.empty:
        send_email_with_top_deals(df_new, TOP_N_TO_EMAIL)

if __name__ == "__main__":
    main()

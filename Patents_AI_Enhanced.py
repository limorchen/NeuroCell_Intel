#!/usr/bin/env python3
"""
AI-Enhanced Patent Search - European Patent Office (EPO) Only
Searches EPO for exosome and extracellular vesicle patents targeting CNS disorders
Uses local AI (SentenceTransformer) for relevance scoring
Automated bimonthly execution via GitHub Actions

FIXES APPLIED (2026-03-01):
  FIX 1: Added INFO-level logging for date-filtered patents (silent drops now visible)
  FIX 2: Fixed pub_date parsing to handle both YYYYMMDD and YYYY-MM-DD formats robustly
  FIX 3: NEW title prefix is NO LONGER written to CSV
  FIX 4: HF_TOKEN environment variable now passed to SentenceTransformer
  FIX 5: embeddings.position_ids warning suppressed via logging filter
  FIX 6: Added counter and breakdown log for all EPO results
  FIX 7: backfill_first_claims() pandas TypeError fixed (object dtype)
  FIX 8: EPO fair use 403 — exponential backoff + 2s inter-request sleep
  FIX 9: Claim text no longer truncated at 1000 chars — full claim stored
  FIX 11: Google Patents response forced to UTF-8 — fixes mojibake on CN/JP patents
  FIX 12: English translation extracted robustly — handles both numbered (1.) and
           lettered (a)/(b) JP/CN claim structures; returns empty if no English found
  FIX 13: Backfill now re-fetches garbled (mojibake) claims automatically
  FIX 14: All CSV saves use QUOTE_ALL — prevents phantom columns from long claim text
  FIX 15: Backfill also re-fetches claims truncated at legacy 1000-char limit
  FIX 16: Interleaved native-language text stripped from translated patent claims
  FIX 12: English translation extracted from CN/JP claims when available
  FIX 10: EPO date filter now applied server-side via CQL pd>= operator —
           eliminates ~5 min wasted processing 275 out-of-window results

CHANGES (2026-03-04):
  CHANGE 1: Abstract field removed from all data collection, storage, and email output
  CHANGE 2: First claim extraction — two-tier strategy:
              Tier 1 (EP patents): EPO OPS claims endpoint (direct HTTP, cached OAuth token).
                epo_ops library does NOT correctly route this endpoint — replaced with
                direct requests call. Exponential backoff on 403 fair use throttling.
              Tier 2 (all other jurisdictions + EP fallback): Google Patents scraper.
                Covers US, WO, EP, CN, JP, KR, AU, CA, MX and most jurisdictions.
                Polite 3s inter-request sleep + browser-like User-Agent.
              Claim source stored in new 'claim_source' CSV column.

CHANGES (2026-03-05):
  CHANGE 3: Added parallel GRANTED PATENT search (kind=B1/B2) with NO date restriction.
              - Runs a separate CQL query targeting kind=B1 and kind=B2 EP grants.
              - Extended CQL terms: neuroprotection, spinal cord, TBI, neuroinflammation,
                nerve injury, regeneration — matches updated RESEARCH_FOCUS vocabulary.
              - Deduplicates against existing_ids so already-captured grants are skipped.
              - First run performs a full historical sweep; subsequent runs only capture
                newly issued grants (since they will already be in the CSV after run 1).
              - Results breakdown separates application hits from grant hits.
              - Email subject line and body now report grants and applications separately.
  CHANGE 4: RESEARCH_FOCUS and search terms updated to cover TBI, SCI, peripheral nerve,
              facial nerve, neuroprotection, anti-inflammatory and regenerative mechanisms.
  CHANGE 5: search_epo_patents() refactored — shared _run_epo_cql_query() helper handles
              pagination for both queries. _process_patent_list() handles scoring/filtering
              for both lists. _print_breakdown() formats per-query summary logs.

FIXES (2026-03-05):
  FIX 17: EPO OPS 413 (Request Entity Too Large) — BASE_CNS_CQL was too long for the
           EPO OPS URL limit (~2000 chars). Fix: split BASE_CNS_CQL into two halves
           (CNS_CQL_A and CNS_CQL_B). _run_epo_cql_query() now runs both sub-queries
           and deduplicates the combined result list before returning. Each individual
           CQL stays well within EPO OPS limits.
  FIX 18: Google Patents 404 for US published applications — EPO docdb stores US
           publication numbers without leading zeros in the application serial portion,
           e.g. "2023285291" (10 digits). Google Patents requires the full zero-padded
           11-digit format, e.g. "20230285291". normalise_us_number() detects US
           publications (A1/A2 kind codes) and zero-pads the serial portion when the
           stored number is 10 digits so the Google Patents URL resolves correctly.

DEPENDENCIES (add to requirements.txt):
  beautifulsoup4>=4.12
  lxml>=4.9
  (all others were already required)
"""

import os
import re
import time
import smtplib
import requests
import logging
from datetime import datetime, timedelta
from pathlib import Path
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication

import pandas as pd
from lxml import etree

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False
    print("Warning: beautifulsoup4 not installed. Google Patents fallback disabled.")
    print("         Install with: pip install beautifulsoup4")

from epo_ops import Client, models, middlewares

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SEMANTIC = True
except ImportError:
    HAS_SEMANTIC = False
    print("Warning: SentenceTransformer not available. Using default relevance scores.")

# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
CUMULATIVE_CSV = DATA_DIR / "patents_cumulative.csv"

# ---------------------------------------------------------------
# FIX 5: Suppress noisy but benign embeddings.position_ids warning
# ---------------------------------------------------------------
class _SuppressBertPositionIdsFilter(logging.Filter):
    def filter(self, record):
        return "embeddings.position_ids" not in record.getMessage()

_bert_filter = _SuppressBertPositionIdsFilter()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logging.getLogger().addFilter(_bert_filter)

# EPO API credentials
epo_key    = os.environ.get("EPO_OPS_KEY")
epo_secret = os.environ.get("EPO_OPS_SECRET")

# Email credentials
SENDER_EMAIL    = os.environ.get("SENDER_EMAIL")
EMAIL_PASSWORD  = os.environ.get("EMAIL_PASSWORD")
RECIPIENT_EMAIL = os.environ.get("RECIPIENT_EMAIL")

# Initialize EPO client
epo_client = None
if epo_key and epo_secret:
    try:
        epo_client = Client(
            key=epo_key,
            secret=epo_secret,
            middlewares=[middlewares.Throttler()]
        )
        print("✓ EPO client initialized")
    except Exception as e:
        print(f"Error creating EPO client: {e}")

# ---------------------------------------------------------------
# EPO OAuth token — fetched once per run, cached at module level.
# Using direct requests avoids all epo_ops middleware quirks.
# ---------------------------------------------------------------
_epo_token_cache: dict = {"token": None}

def get_epo_access_token() -> str | None:
    """
    Obtain (or return the cached) EPO OPS OAuth2 access token.
    Fetches directly via requests so we never touch epo_ops internals.
    Returns None if credentials are missing or the request fails.
    """
    if _epo_token_cache["token"]:
        return _epo_token_cache["token"]
    if not epo_key or not epo_secret:
        return None
    try:
        resp = requests.post(
            "https://ops.epo.org/3.2/auth/accesstoken",
            data={"grant_type": "client_credentials"},
            auth=(epo_key, epo_secret),
            timeout=15,
        )
        resp.raise_for_status()
        token = resp.json().get("access_token")
        _epo_token_cache["token"] = token
        logger.info("✓ EPO OAuth token obtained")
        return token
    except Exception as e:
        logger.error(f"Failed to obtain EPO OAuth token: {e}")
        return None


# ---------------------------------------------------------------
# CHANGE 4: Updated research focus and search configuration
# ---------------------------------------------------------------

RESEARCH_FOCUS = """
Exosomes and extracellular vesicles (EVs) — either naive/unmodified or engineered/loaded 
with therapeutic payloads (proteins, miRNA, siRNA, drugs, growth factors) — for treating 
neural injuries and conditions. 

Primary focus areas:
- Traumatic brain injury (TBI): neuroprotection, neuroinflammation reduction, axonal repair
- Spinal cord injury (SCI): regeneration, remyelination, motor function recovery
- Peripheral nerve injuries: facial nerve palsy, cranial nerve damage, nerve crush/transection
- Stroke and hypoxic-ischemic injury: penumbra salvage, angiogenesis, functional recovery

Key mechanisms of interest:
- Neuroprotective effects (reducing apoptosis, oxidative stress, excitotoxicity)
- Anti-inflammatory modulation (M1/M2 macrophage polarization, microglial regulation, cytokine suppression)
- Regenerative and pro-repair signaling (axonogenesis, synaptogenesis, myelin repair)
- Blood-brain barrier (BBB) and blood-nerve barrier (BNB) integrity and penetration
- Stem cell-derived EVs (MSC, neural stem cell) and their paracrine regenerative effects
- Surface engineering for targeted CNS/PNS delivery (RVG peptide, targeting ligands)

Exclude: cancer drug delivery, non-neural applications, purely diagnostic exosome studies.
"""

SEARCH_TERMS = [
    'exosomes',
    'extracellular vesicles',
    'EVs',
    'nanovesicles'
]

SEARCH_FILTER = 'TBI OR spinal cord OR nerve injury OR neuroprotection OR neuroinflammation'
MIN_RELEVANCE_SCORE = 0.55

# ---------------------------------------------------------------
# FIX 17 (revised): CQL sub-queries — kept deliberately minimal
#
# The EPO OPS CQL limit is ~175 raw characters per query string.
# URL-encoding multiplies length further. The original working query
# (from the script's first run) was ~175 chars and succeeded.
# Any query exceeding that threshold returns 413.
#
# Strategy: use 3 short sub-queries per search type, each well under
# 160 chars. The semantic model (MIN_RELEVANCE_SCORE) is the primary
# relevance filter — the CQL just needs to cast a broad enough net.
#
# ta=EVs and ta=nanovesicles are removed from the EV filter:
#   - "EVs" is a generic abbreviation that matches unrelated patents
#   - "nanovesicles" rarely appears in patent titles/abstracts
#   Both added length without adding useful recall.
#
# _EV      — core exosome/EV title+abstract filter (~44 chars)
# _CNS_A   — core CNS/neurological terms         (~73 chars)
# _CNS_B   — injury/TBI/SCI/stroke terms         (~84 chars)
# _CNS_C   — BBB/neurodegenerative/nerve repair  (~76 chars)
#
# Longest combined sub-query: _EV AND _CNS_B AND pd>=YYYYMMDD ≈ 153 chars
# ---------------------------------------------------------------

_EV = '(ta=exosomes OR ta="extracellular vesicles")'

_CNS_A = '(ta=CNS OR ta=neurological OR ta=neuroprotection OR ta=neuroinflammation)'

_CNS_B = '(ta="spinal cord" OR ta=TBI OR ta="nerve injury" OR ta=stroke OR ta="brain injury")'

_CNS_C = '(ta="blood-brain barrier" OR ta=neurodegenerative OR ta="nerve regeneration")'


# FIX 4: Pass HF_TOKEN to SentenceTransformer to avoid rate-limit risk
hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    logger.warning(
        "HF_TOKEN not set. Unauthenticated HuggingFace requests may be rate-limited. "
        "Add HF_TOKEN to your GitHub Actions secrets and env block to suppress this."
    )

semantic_model = None
research_focus_embedding = None

if HAS_SEMANTIC:
    try:
        print("Loading semantic search model...")
        semantic_model = SentenceTransformer('all-MiniLM-L6-v2', token=hf_token)
        research_focus_embedding = semantic_model.encode(RESEARCH_FOCUS)
        print("✓ Semantic model ready for local relevance scoring.")
    except Exception as e:
        print(f"Warning: Could not load semantic model: {e}")
        HAS_SEMANTIC = False


# ---------------------------------------------------------------
# FIX 2: Robust pub_date normalisation
# ---------------------------------------------------------------
def normalise_date(raw_date: str) -> str | None:
    """
    Convert any date string returned by EPO into an 8-digit YYYYMMDD string.
    Handles: "20260115", "2026-01-15", "2026/01/15", "20260115000000"
    Returns None if unparseable.
    """
    if not raw_date:
        return None
    cleaned = raw_date.replace("-", "").replace("/", "").replace(" ", "")[:8]
    if len(cleaned) == 8 and cleaned.isdigit():
        return cleaned
    return None


def calculate_relevance_score(title, first_claim):
    """
    Calculate semantic similarity to research focus using SentenceTransformer.
    Scores against title + first claim text.
    """
    if not HAS_SEMANTIC or not semantic_model:
        return 0.5
    if not title and not first_claim:
        return 0.0
    patent_text = f"{title} {first_claim}"
    try:
        patent_embedding = semantic_model.encode(patent_text)
        similarity = cosine_similarity(
            research_focus_embedding.reshape(1, -1),
            patent_embedding.reshape(1, -1)
        )[0][0]
        return float(similarity)
    except Exception as e:
        logger.warning(f"Error calculating relevance: {e}")
        return 0.5


def generate_patent_link(country, number, kind=""):
    """Generate appropriate patent link based on country code."""
    # FIX 18: normalise US application numbers for correct Google Patents URLs
    if country == "US":
        number = normalise_us_number(number, kind)
        return f"https://patents.google.com/patent/US{number}{kind}"
    elif country == "WO":
        return f"https://patentscope.wipo.int/search/en/detail.jsf?docId=WO{number}"
    elif country == "EP":
        return f"https://worldwide.espacenet.com/patent/search?q=pn%3DEP{number}"
    elif country == "JP":
        return f"https://worldwide.espacenet.com/patent/search?q=pn%3DJP{number}"
    elif country == "CN":
        return f"https://patents.google.com/patent/CN{number}{kind}"
    elif country == "AU":
        return f"https://worldwide.espacenet.com/patent/search?q=pn%3DAU{number}"
    else:
        return f"https://worldwide.espacenet.com/patent/search?q=pn%3D{country}{number}{kind}"


# ---------------------------------------------------------------
# Tier 1: EPO OPS claims endpoint (EP patents only)
# ---------------------------------------------------------------

def get_epo_first_claim(country, number, kind) -> str:
    """
    Fetch the first claim directly from EPO OPS REST API.

    WHY DIRECT HTTP: The epo_ops library omits the document reference from the
    URL when routing the 'claims' endpoint (.../docdb/claims instead of
    .../docdb/EP4132584.A1/claims), returning 404. We build the URL manually.

    EPO FAIR USE: Full-text endpoints enforce rate limits. Burst requests trigger
    403 "Fair Use policy violation". Retries use exponential backoff (20s->40s->80s).
    Keep inter-request sleep >= 2s in calling code.

    Returns claim text or '' if unavailable.
    """
    token = get_epo_access_token()
    if not token:
        return ""

    doc_id = f"{country}{number}.{kind}"
    url = (
        f"https://ops.epo.org/3.2/rest-services/"
        f"published-data/publication/docdb/{doc_id}/claims"
    )

    max_retries = 3
    backoff = 20  # seconds

    for attempt in range(max_retries):
        try:
            resp = requests.get(
                url,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/xml"
                },
                timeout=15,
            )

            # Token expired mid-run — refresh once and retry
            if resp.status_code == 401:
                logger.info("EPO token expired mid-run, refreshing...")
                _epo_token_cache["token"] = None
                token = get_epo_access_token()
                if not token:
                    return ""
                continue

            # Fair use throttling — exponential backoff
            if resp.status_code == 403:
                wait = backoff * (2 ** attempt)
                logger.warning(
                    f"EPO fair use 403 for {country}{number}{kind} "
                    f"(attempt {attempt + 1}/{max_retries}). "
                    f"Backing off {wait}s..."
                )
                time.sleep(wait)
                continue

            if resp.status_code == 404:
                logger.info(f"  – EPO OPS: {country}{number}{kind} claims not indexed (404)")
                return ""

            resp.raise_for_status()

            root = etree.fromstring(resp.content)
            ns = {"ex": "http://www.epo.org/exchange"}

            # Primary: first claim in English
            first_claim = root.xpath(
                "string(//ex:claims[@lang='en']/ex:claim[1]/ex:claim-text)",
                namespaces=ns
            )
            # Fallback: first claim in any available language
            if not first_claim:
                first_claim = root.xpath(
                    "string(//ex:claims/ex:claim[1]/ex:claim-text)",
                    namespaces=ns
                )

            return first_claim.strip() if first_claim else ""

        except requests.exceptions.HTTPError as e:
            logger.warning(f"EPO claims HTTP error for {country}{number}{kind}: {e}")
            return ""
        except Exception as e:
            logger.warning(f"EPO claims error for {country}{number}{kind}: {e}")
            return ""

    logger.warning(
        f"EPO claims: gave up after {max_retries} attempts for {country}{number}{kind}"
    )
    return ""


# ---------------------------------------------------------------
# Tier 2: Google Patents scraper (universal fallback)
# ---------------------------------------------------------------

# Browser-like headers to avoid Google bot detection
_GOOGLE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def _clean_claim_text(text: str) -> str:
    """
    Normalise whitespace and strip leading claim number artefacts (e.g. '1. ').
    When Google Patents shows both native-language text and an English translation
    (common for CN and JP patents), extract only the English portion.
    """
    if not text:
        return ""

    lang_match = re.search(
        r'Translated from (?:Chinese|Japanese|Korean|German|French|Spanish)\b',
        text, flags=re.IGNORECASE
    )
    if lang_match:
        remainder = text[lang_match.end():]
        english_start = None
        eng_match = re.search(
            r'(?:^|\s)(?:1[.\s]|\(a\)|[A-Z][a-z]{2,})',
            remainder
        )
        if eng_match:
            candidate = remainder[eng_match.start():].strip()
            non_ascii = sum(1 for c in candidate[:200] if ord(c) > 127)
            if non_ascii / max(len(candidate[:200]), 1) < 0.1:
                english_start = eng_match.start()
        if english_start is not None:
            text = remainder[english_start:].strip()
        else:
            return ""

    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'^(claim\s*)?1[\.\s]\s*', '', text, flags=re.IGNORECASE).strip()
    return text


def _extract_first_claim_from_text(text: str) -> str:
    """
    Heuristic: split a full claims blob at the boundary where claim 2 starts.
    Returns just the first claim text.
    """
    split = re.split(r'\b2[\.\s]\s+', text, maxsplit=1)
    return split[0].strip() if split else text.strip()


def normalise_us_number(number: str, kind: str) -> str:
    """
    FIX 18: Normalise US publication numbers for Google Patents URLs.

    EPO's docdb format stores US published application numbers with the
    application serial portion un-padded, e.g. "2023285291" (10 digits:
    4-digit year + 6-digit serial). Google Patents requires the full
    zero-padded 11-digit format: "20230285291" (4-digit year + 7-digit serial).

    This only applies to US published applications (kind A1/A2). US granted
    patents (kind B1/B2) use a plain grant number that needs no padding.

    Examples:
        "2023285291", "A1"  →  "20230285291"   (6-digit serial → 7-digit)
        "20230285291", "A1" →  "20230285291"   (already correct, unchanged)
        "11234567", "B2"    →  "11234567"      (grant number, unchanged)
    """
    if kind in ("A1", "A2") and len(number) == 10 and number.isdigit():
        year   = number[:4]
        serial = number[4:]          # 6 digits from EPO
        return f"{year}0{serial}"    # pad to 7 digits → 11 total
    return number


def get_google_patents_first_claim(country, number, kind) -> str:
    """
    Scrape the first claim from Google Patents.

    Covers: US, WO, EP, CN, JP, KR, AU, CA, MX and most other jurisdictions.
    Google Patents holds full-text claims for the vast majority of patent families.
    """
    if not HAS_BS4:
        return ""

    # FIX 18: normalise US publication numbers to the zero-padded format
    # that Google Patents expects (EPO docdb omits the leading zero in the serial)
    if country == "US":
        number = normalise_us_number(number, kind)

    patent_id = f"{country}{number}{kind}"
    url = f"https://patents.google.com/patent/{patent_id}/en"

    try:
        resp = requests.get(url, headers=_GOOGLE_HEADERS, timeout=20)

        if resp.status_code == 404:
            logger.info(f"  – Google Patents: {patent_id} not found (404)")
            return ""

        if resp.status_code == 429:
            logger.warning(
                f"  – Google Patents: rate limited (429) for {patent_id}. Waiting 60s..."
            )
            time.sleep(60)
            resp = requests.get(url, headers=_GOOGLE_HEADERS, timeout=20)

        if resp.status_code != 200:
            logger.warning(
                f"  – Google Patents: HTTP {resp.status_code} for {patent_id}"
            )
            return ""

        resp.encoding = 'utf-8'
        soup = BeautifulSoup(resp.text, "lxml")

        # ── Selector 1: standard claims div ────────────────────────────────────
        claims_div = soup.find("div", class_="claims")
        if claims_div:
            first_claim_div = claims_div.find("div", class_="claim")
            if first_claim_div:
                claim_text_div = first_claim_div.find("div", class_="claim-text")
                text = (
                    claim_text_div.get_text(separator=" ", strip=True)
                    if claim_text_div
                    else first_claim_div.get_text(separator=" ", strip=True)
                )
                if text:
                    return _clean_claim_text(text)

        # ── Selector 2: itemprop="claims" section ──────────────────────────────
        claims_section = soup.find(attrs={"itemprop": "claims"})
        if claims_section:
            claim_span = claims_section.find(attrs={"itemprop": "claimText"})
            if claim_span:
                text = claim_span.get_text(separator=" ", strip=True)
                if text:
                    return _clean_claim_text(text)
            full_text = claims_section.get_text(separator=" ", strip=True)
            if full_text:
                return _clean_claim_text(_extract_first_claim_from_text(full_text))

        # ── Selector 3: patent-text custom element ─────────────────────────────
        patent_text_el = soup.find(
            "patent-text",
            attrs={"heading": re.compile(r"claim", re.I)}
        )
        if patent_text_el:
            text = patent_text_el.get_text(separator=" ", strip=True)
            if text:
                return _clean_claim_text(_extract_first_claim_from_text(text))

        logger.info(f"  – Google Patents: no claims section found for {patent_id}")
        return ""

    except requests.exceptions.RequestException as e:
        logger.warning(f"Google Patents request error for {patent_id}: {e}")
        return ""
    except Exception as e:
        logger.warning(f"Google Patents parse error for {patent_id}: {e}")
        return ""


# ---------------------------------------------------------------
# Combined claim fetcher — EPO OPS first (EP), Google Patents fallback
# ---------------------------------------------------------------

def get_first_claim(country, number, kind) -> tuple[str, str]:
    """
    Two-tier first claim fetch with source attribution.

    Tier 1 (EP patents only): EPO OPS claims endpoint.
    Tier 2 (all others, or if EPO returns nothing): Google Patents scraper.

    Returns: (claim_text, source_label)
      source_label one of: 'EPO OPS' | 'Google Patents' | 'unavailable'
    """
    if country == "EP":
        claim = get_epo_first_claim(country, number, kind)
        if claim:
            logger.info(f"    ✓ Claim via EPO OPS ({len(claim)} chars)")
            return claim, "EPO OPS"
        logger.info("    – EPO OPS returned nothing, trying Google Patents...")

    time.sleep(3)
    claim = get_google_patents_first_claim(country, number, kind)
    if claim:
        logger.info(f"    ✓ Claim via Google Patents ({len(claim)} chars)")
        return claim, "Google Patents"

    return "", "unavailable"


# ---------------------------------------------------------------
# EPO Search — internal helpers
# ---------------------------------------------------------------

def _run_epo_cql_query(cql_list: list[str], label: str, max_records: int = 500) -> list[dict]:
    """
    Execute one or more CQL queries against EPO OPS and return deduplicated results.

    FIX 17: EPO OPS rejects queries whose URL exceeds ~2000 chars with 413.
    Accepting a list of CQL strings allows callers to split large queries into
    safe sub-queries. Results from all sub-queries are merged and deduplicated
    on (country, publication_number, kind) before being returned.

    Args:
        cql_list:    One or more CQL query strings.
        label:       Human-readable label for log messages.
        max_records: Per-query hard cap (EPO OPS index max is 2000).

    Returns:
        Deduplicated list of dicts with keys: country, publication_number, kind, source.
    """
    all_records: list[dict] = []
    seen_ids:    set[str]   = set()

    for sub_idx, cql in enumerate(cql_list, 1):
        sub_label = f"{label}/Q{sub_idx}" if len(cql_list) > 1 else label
        logger.info(f"[EPO/{sub_label}] Running CQL: {cql}")

        try:
            start      = 1
            batch_size = 25
            total      = None

            while True:
                end = start + batch_size - 1
                if start > max_records:
                    break

                try:
                    resp = epo_client.published_data_search(
                        cql=cql,
                        range_begin=start,
                        range_end=min(end, max_records)
                    )
                except Exception as e:
                    err_str = str(e)
                    # EPO OPS returns 404 when a valid query matches zero results.
                    # Treat this as empty — not a real error.
                    if "404" in err_str:
                        logger.info(f"[EPO/{sub_label}] No results (404 — zero matches)")
                    else:
                        logger.error(f"[EPO/{sub_label}] Search error at offset {start}: {e}")
                    break

                root = etree.fromstring(resp.content)
                ns   = {"ops": "http://ops.epo.org", "ex": "http://www.epo.org/exchange"}

                if total is None:
                    total_str = root.xpath(
                        "string(//ops:biblio-search/@total-result-count)",
                        namespaces=ns
                    )
                    total = int(total_str) if total_str else 0
                    logger.info(f"[EPO/{sub_label}] Server reports {total} total results")

                for pub_ref in root.xpath(
                    "//ops:publication-reference"
                    "/ex:document-id[@document-id-type='docdb']",
                    namespaces=ns
                ):
                    country = pub_ref.xpath("string(ex:country)", namespaces=ns)
                    number  = pub_ref.xpath("string(ex:doc-number)", namespaces=ns)
                    kind    = pub_ref.xpath("string(ex:kind)", namespaces=ns)

                    if country and number and kind:
                        uid = f"{country}{number}{kind}"
                        if uid not in seen_ids:
                            seen_ids.add(uid)
                            all_records.append({
                                "country":            country,
                                "publication_number": number,
                                "kind":               kind,
                                "source":             "EPO"
                            })

                if not total or end >= min(total, max_records):
                    break

                start = end + 1
                time.sleep(0.3)

        except Exception as e:
            logger.error(f"[EPO/{sub_label}] Query failed: {e}")

    logger.info(f"[EPO/{label}] Total unique references after dedup: {len(all_records)}")
    return all_records


def _process_patent_list(
    patent_list:       list[dict],
    existing_ids:      set,
    processed:         set,
    start_date:        str,
    end_date:          str,
    current_run_date:  str,
    apply_date_filter: bool,
    kind_filter:       set | None = None,
) -> tuple[list[dict], dict]:
    """
    Process a list of raw EPO patent references into scored, filtered records.

    Args:
        patent_list:       Raw references from _run_epo_cql_query().
        existing_ids:      Patent IDs already in the cumulative CSV.
        processed:         Shared set of IDs already handled this run.
                           Passed by reference — updated in place.
                           Prevents a patent from being processed by both the
                           applications and grants queries in the same run.
        start_date:        Window start YYYYMMDD.
        end_date:          Window end YYYYMMDD.
        current_run_date:  ISO date string written to date_added field.
        apply_date_filter: True  → enforce 60-day Python date filter (applications).
                           False → no date restriction (grants).
        kind_filter:       Optional set of allowed kind codes, e.g. {'B1', 'B2'}.
                           Patents whose kind code is not in the set are skipped.
                           None = accept all kind codes (default).
                           Used for grants because EPO CQL has no kind-code field.

    Returns:
        (records, counters)
    """
    records  = []
    counters = {
        "new":             0,
        "duplicate_db":    0,
        "duplicate_batch": 0,
        "date_filtered":   0,
        "date_missing":    0,
        "kind_filtered":   0,
        "low_relevance":   0,
        "biblio_error":    0,
    }

    for patent in patent_list:
        patent_id = (
            f"{patent['country']}"
            f"{patent['publication_number']}"
            f"{patent.get('kind', '')}"
        )

        if patent_id in processed:
            counters["duplicate_batch"] += 1
            continue
        processed.add(patent_id)

        if patent_id in existing_ids:
            counters["duplicate_db"] += 1
            continue

        # Kind-code filter (grants only): skip applications (A1/A2) that
        # slip through the subject-only grants CQL query.
        if kind_filter and patent.get('kind', '') not in kind_filter:
            counters["kind_filtered"] += 1
            logger.debug(
                f"  – {patent_id} — kind '{patent.get('kind', '')}' "
                f"not in {kind_filter}. Skipped."
            )
            continue

        biblio = get_epo_biblio(
            patent['country'], patent['publication_number'], patent.get('kind', '')
        )
        if not biblio:
            counters["biblio_error"] += 1
            logger.warning(f"  ✗ {patent_id} — biblio fetch returned empty, skipping")
            continue

        title         = biblio.get('title', '')
        applicants    = biblio.get('applicants', 'Not available')
        inventors     = biblio.get('inventors', 'Not available')
        pub_date_raw  = biblio.get('publication_date', '')
        priority_date = biblio.get('priority_date', '')

        pub_date_norm = normalise_date(pub_date_raw)

        # FIX 2 + FIX 1: date filter only applied for applications
        if apply_date_filter:
            if pub_date_norm is None:
                counters["date_missing"] += 1
                logger.info(
                    f"  ? {patent_id} — pub_date missing/unparseable "
                    f"(raw='{pub_date_raw}'). Including anyway."
                )
            else:
                if not (start_date <= pub_date_norm <= end_date):
                    counters["date_filtered"] += 1
                    logger.info(
                        f"  – {patent_id} — outside date window "
                        f"(pub={pub_date_norm}, window={start_date}–{end_date}). Skipped."
                    )
                    continue

        # Two-tier claim fetch: EPO OPS → Google Patents
        logger.info(f"  Fetching claim for {patent_id}...")
        first_claim, claim_source = get_first_claim(
            patent['country'], patent['publication_number'], patent.get('kind', '')
        )
        time.sleep(2)  # EPO fair use: >= 2s between full-text requests

        relevance = calculate_relevance_score(title, first_claim)

        if relevance < MIN_RELEVANCE_SCORE:
            counters["low_relevance"] += 1
            logger.info(
                f"  ↓ {patent_id} — low relevance {relevance:.2f} "
                f"(threshold={MIN_RELEVANCE_SCORE}). Skipped."
            )
            continue

        link = generate_patent_link(
            patent['country'], patent['publication_number'], patent.get('kind', '')
        )

        records.append({
            "country":            patent['country'],
            "publication_number": patent['publication_number'],
            "kind":               patent.get('kind', 'A1'),
            "title":              title,
            "applicants":         applicants,
            "inventors":          inventors,
            "first_claim":        first_claim if first_claim else "",
            "claim_source":       claim_source,
            "publication_date":   pub_date_raw,
            "priority_date":      priority_date,
            "relevance_score":    round(relevance, 2),
            "source":             "EPO",
            "link":               link,
            "date_added":         current_run_date,
            "is_new":             "YES"
        })

        counters["new"] += 1
        logger.info(
            f"  ✓ {patent_id} — Score: {relevance:.2f} "
            f"— Claim: {claim_source} — {title[:60]}"
        )

    return records, counters


def _print_breakdown(label: str, total_returned: int, counters: dict):
    """Print a formatted results breakdown for one query type."""
    total_checked = sum(counters.values())
    print(f"\n[{label} BREAKDOWN]")
    print(f"  EPO results returned:        {total_returned}")
    print(f"  ─────────────────────────────────────────")
    print(f"  ✓ Added as new:              {counters['new']}")
    print(f"  = Already in database:       {counters['duplicate_db']}")
    print(f"  = Duplicate in this batch:   {counters['duplicate_batch']}")
    if label == "APPLICATIONS":
        print(f"  – Outside date window:       {counters['date_filtered']}")
        print(f"  ? Date missing/unparseable:  {counters['date_missing']}")
    if label == "GRANTS":
        print(f"  – Wrong kind code (A*/WO):   {counters['kind_filtered']}")
    print(f"  ↓ Below relevance threshold: {counters['low_relevance']}")
    print(f"  ✗ Biblio fetch error:        {counters['biblio_error']}")
    print(f"  ─────────────────────────────────────────")
    print(f"  Σ Accounted for:             {total_checked}")
    if total_checked != total_returned:
        logger.warning(
            f"  ⚠ COUNT MISMATCH: {total_returned} returned vs {total_checked} accounted."
        )


# ---------------------------------------------------------------
# EPO Search — public entry point
# ---------------------------------------------------------------

def search_epo_patents(start_date: str, end_date: str) -> tuple[list[dict], list[dict]]:
    """
    Run two parallel EPO OPS searches and return results separately.

    Query A — APPLICATIONS (date-windowed, last 60 days):
        Targets recently published patent applications (A1/A2).
        The pd>= operator limits EPO server-side results to the 60-day window.
        Python date filtering is applied as a safety net for edge-case results.

    Query B — GRANTS (NO date restriction, ki=B1 OR ki=B2):
        Targets examination-complete granted patents with finalised claims.
        Granted patents are filed 3–8 years before issue and would never appear
        in a 60-day window. No date filter is applied here.
        Extended BASE_CNS_CQL vocabulary matches the specific language used in
        granted patent claims (TBI, SCI, nerve injury, facial nerve, etc.).
        Deduplication against existing_ids in the caller ensures already-captured
        grants are not reprocessed on subsequent runs. The first run performs
        a full historical sweep; later runs only pick up newly issued grants.

    Returns:
        (app_results, grant_results) — two lists of patent reference dicts.
    """
    if not epo_client:
        logger.warning("EPO client not available, skipping EPO search")
        return [], []

    # ── Query A: Published applications — 60-day window ───────────────────────
    # Three sub-queries, each under ~160 raw chars to stay within EPO OPS limit.
    # Semantic scoring handles fine-grained relevance filtering.
    cql_apps = [
        f"{_EV} AND {_CNS_A} AND pd>={start_date}",   # ~137 chars
        f"{_EV} AND {_CNS_B} AND pd>={start_date}",   # ~153 chars
        f"{_EV} AND {_CNS_C} AND pd>={start_date}",   # ~140 chars
    ]

    # ── Query B: Granted patents — full historical sweep ──────────────────────
    # EPO OPS published-data/search CQL does not support a standalone kind-code
    # field (ki=, kind=, etc. all return 400 Bad Request). Kind filtering is
    # handled in Python post-processing via the kind_filter parameter passed to
    # _process_patent_list, which checks the kind code returned in biblio data.
    # Queries are identical subject terms to apps but WITHOUT a date restriction,
    # so the full historical corpus is searched. Deduplication against existing_ids
    # and the shared processed set prevents reprocessing patents already captured.
    cql_grants = [
        f"{_EV} AND {_CNS_A}",   # ~114 chars
        f"{_EV} AND {_CNS_B}",   # ~130 chars
        f"{_EV} AND {_CNS_C}",   # ~117 chars
    ]

    app_results   = _run_epo_cql_query(cql_apps,   label="APPLICATIONS", max_records=500)
    grant_results = _run_epo_cql_query(cql_grants, label="GRANTS",        max_records=2000)

    return app_results, grant_results


def get_epo_biblio(country, number, kind):
    """
    Fetch EPO bibliographic data for a patent.
    Abstract intentionally excluded — first claim used for content scoring.
    """
    if not epo_client:
        return {}

    try:
        resp = epo_client.published_data(
            reference_type="publication",
            input=models.Docdb(number, country, kind),
            endpoint="biblio"
        )

        root = etree.fromstring(resp.content)
        ns   = {"ex": "http://www.epo.org/exchange"}

        title = root.xpath("string(//ex:invention-title[@lang='en'])", namespaces=ns)
        if not title:
            title = root.xpath("string(//ex:invention-title)", namespaces=ns)

        applicants = root.xpath(
            "//ex:applicants/ex:applicant/ex:applicant-name/ex:name/text()",
            namespaces=ns
        )
        applicants_str = ", ".join(applicants[:3]) if applicants else "Not available"

        inventors = root.xpath(
            "//ex:inventors/ex:inventor/ex:inventor-name/ex:name/text()",
            namespaces=ns
        )
        inventors_str = ", ".join(inventors[:3]) if inventors else "Not available"

        pub_date = root.xpath(
            "string(//ex:publication-reference"
            "/ex:document-id[@document-id-type='docdb']/ex:date)",
            namespaces=ns
        )
        priority_date = root.xpath(
            "string(//ex:priority-claims/ex:priority-claim[1]/ex:document-id/ex:date)",
            namespaces=ns
        )

        return {
            "title":            title,
            "applicants":       applicants_str,
            "inventors":        inventors_str,
            "publication_date": pub_date,
            "priority_date":    priority_date,
        }
    except Exception as e:
        logger.warning(f"EPO biblio fetch error for {country}{number}{kind}: {e}")
        return {}


# ---------------------------------------------------------------
# Combined Search & Processing
# ---------------------------------------------------------------

def search_all_patents():
    """
    Search EPO for both published applications (60-day window) and granted patents
    (full historical sweep). Process, score, and return a combined DataFrame.
    """
    start_date = (datetime.now().date() - timedelta(days=60)).strftime("%Y%m%d")
    end_date   = datetime.now().date().strftime("%Y%m%d")

    print("=" * 80)
    print(f"Starting AI-Enhanced Patent Search — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Relevance threshold:  {MIN_RELEVANCE_SCORE}")
    print(f"Applications window:  Last 60 days ({start_date} → {end_date})")
    print(f"Grants window:        Full historical (no date restriction)")
    print(f"AI relevance scoring: SentenceTransformer (title + first claim)")
    print(f"Claim sources:        EPO OPS (EP) → Google Patents (all others)")
    print("=" * 80)

    # Load existing patents
    existing_ids = set()
    if CUMULATIVE_CSV.exists():
        df = pd.read_csv(CUMULATIVE_CSV, dtype={'first_claim': object})
        existing_ids = set(
            df.apply(
                lambda row: f"{row['country']}{row['publication_number']}{row['kind']}",
                axis=1
            )
        )
        logger.info(f"Loaded {len(existing_ids)} existing patents from database")

    logger.info("Searching patent sources...")
    app_results, grant_results = search_epo_patents(start_date, end_date)

    print(f"\n[SUMMARY] Applications found: {len(app_results)}")
    print(f"[SUMMARY] Grants found:        {len(grant_results)}")

    current_run_date = datetime.now().strftime('%Y-%m-%d')

    # Shared processed set — prevents the same patent ID from being processed
    # by both the applications and grants loops in the same run.
    processed = set()

    # ── Process applications ────────────────────────────────────────────────
    print(f"\n{'─' * 40}")
    print("Processing APPLICATIONS...")
    print(f"{'─' * 40}")
    app_records, app_counters = _process_patent_list(
        patent_list       = app_results,
        existing_ids      = existing_ids,
        processed         = processed,
        start_date        = start_date,
        end_date          = end_date,
        current_run_date  = current_run_date,
        apply_date_filter = True,
        kind_filter       = None,   # accept all kind codes for applications
    )
    _print_breakdown("APPLICATIONS", len(app_results), app_counters)

    # ── Process grants ──────────────────────────────────────────────────────
    print(f"\n{'─' * 40}")
    print("Processing GRANTS (historical sweep)...")
    print(f"{'─' * 40}")
    grant_records, grant_counters = _process_patent_list(
        patent_list       = grant_results,
        existing_ids      = existing_ids,
        processed         = processed,
        start_date        = start_date,
        end_date          = end_date,
        current_run_date  = current_run_date,
        apply_date_filter = False,
        kind_filter       = {'B1', 'B2'},   # grants only — filter out A1/A2/WO etc.
    )
    _print_breakdown("GRANTS", len(grant_results), grant_counters)

    # ── Combined summary ────────────────────────────────────────────────────
    total_new = app_counters["new"] + grant_counters["new"]
    print(f"\n{'=' * 40}")
    print(f"TOTAL NEW PATENTS ADDED: {total_new}")
    print(f"  From applications: {app_counters['new']}")
    print(f"  From grants:       {grant_counters['new']}")
    print(f"{'=' * 40}")

    all_records = app_records + grant_records
    return pd.DataFrame(all_records)


def update_cumulative_csv(df_new):
    """Merge new results with existing cumulative CSV."""
    FINAL_COLUMNS = [
        "country", "publication_number", "kind", "title", "applicants", "inventors",
        "first_claim", "claim_source", "publication_date", "priority_date",
        "relevance_score", "source", "link", "date_added", "is_new"
    ]

    if CUMULATIVE_CSV.exists():
        df_old = pd.read_csv(CUMULATIVE_CSV, dtype={'first_claim': object})
        df_old['is_new'] = 'NO'

        if 'relevance_score' not in df_old.columns:
            df_old['relevance_score'] = 0.5

        # FIX 3: Strip legacy emoji prefix — one-time cleanup, idempotent
        if 'title' in df_old.columns:
            df_old['title'] = df_old['title'].str.removeprefix('🔥 NEW - ')

        # Migration: drop legacy abstract column
        if 'abstract' in df_old.columns:
            df_old = df_old.drop(columns=['abstract'])
            logger.info("Migrated: removed legacy 'abstract' column")

        # Migration: add first_claim column (object dtype — FIX 7)
        if 'first_claim' not in df_old.columns:
            df_old['first_claim'] = pd.array([''] * len(df_old), dtype=object)
            logger.info("Migrated: added 'first_claim' column")

        # Migration: add claim_source column
        if 'claim_source' not in df_old.columns:
            df_old['claim_source'] = ''
            logger.info("Migrated: added 'claim_source' column")

        # Unconditional NaN → '' (handles previous failed runs — FIX 7)
        df_old['first_claim'] = df_old['first_claim'].fillna('')

        df_all = pd.concat([df_old, df_new], ignore_index=True).drop_duplicates(
            subset=["country", "publication_number", "kind"],
            keep="first"
        )
        logger.info(f"Added {len(df_new)} new patents to database")
    else:
        df_all = df_new
        logger.info(f"Created new database with {len(df_all)} patents")

    df_all = df_all.reindex(columns=FINAL_COLUMNS, fill_value='')

    # Sort: existing patents first (relevance desc), new patents at bottom (relevance desc)
    df_all = df_all.sort_values(
        ['is_new', 'relevance_score'],
        ascending=[True, False]
    )

    df_all.to_csv(CUMULATIVE_CSV, index=False, quoting=1)

    logger.info(f"Saved cumulative CSV with {len(df_all)} total records")
    logger.info("  → Existing patents: sorted by relevance (highest first)")
    logger.info("  → New patents: is_new=YES, appear at bottom, titles are clean")
    return df_all


def send_email_with_csv(df_all):
    """
    Send email with updated CSV.
    New applications and new grants are reported in separate sections.
    """
    if not SENDER_EMAIL or not RECIPIENT_EMAIL or not EMAIL_PASSWORD:
        logger.warning("Email credentials not found. Skipping email.")
        return

    new_patents = df_all[df_all['is_new'] == 'YES']

    # Separate applications (A1/A2) from grants (B1/B2) in new results
    new_apps   = new_patents[new_patents['kind'].isin(['A1', 'A2'])]
    new_grants = new_patents[new_patents['kind'].isin(['B1', 'B2'])]

    email_body = f"""
Patent Search Update — {datetime.now().strftime('%Y-%m-%d')}
{'='*80}

NEW GRANTS:       {len(new_grants)}
NEW APPLICATIONS: {len(new_apps)}
TOTAL NEW:        {len(new_patents)}
TOTAL DATABASE:   {len(df_all)} patents

SOURCE:       European Patent Office (EPO)
SEARCH TERMS: exosomes, extracellular vesicles, EVs, nanovesicles
              TBI, spinal cord injury, nerve injury, neuroprotection, neuroinflammation
DATE RANGE:   Applications: Last 60 days | Grants: Full historical sweep
AI SCORING:   SentenceTransformer (title + first claim)
CLAIM DATA:   EPO OPS (EP patents) → Google Patents fallback (all others)

{'='*80}
"""

    def _format_top5(patents_df, section_label):
        if len(patents_df) == 0:
            return f"\n✓ No new {section_label} found.\n"
        count = min(5, len(patents_df))
        out = f"\n🔥 TOP {count} MOST RELEVANT NEW {section_label.upper()}:\n\n"
        top = patents_df.nlargest(count, 'relevance_score')
        for idx, patent in enumerate(top.itertuples(), 1):
            out += f"{idx}. [{patent.relevance_score:.2f}] 🔥 {patent.title[:80]}\n"
            out += (
                f"   {patent.country}{patent.publication_number}{patent.kind}"
                f" | {patent.source}\n"
            )
            out += f"   Applicant:  {patent.applicants[:60]}\n"
            out += f"   Priority:   {patent.priority_date}\n"
            if patent.first_claim:
                src = getattr(patent, 'claim_source', '')
                out += f"   Claim 1 [{src}]: {patent.first_claim[:200]}...\n"
            out += "\n"
        return out

    email_body += _format_top5(new_grants, "Grants")
    email_body += _format_top5(new_apps, "Applications")
    email_body += f"\nSee attached CSV for full details.\n{'='*80}"

    msg = MIMEMultipart()
    msg["Subject"] = (
        f"Patent Update — {len(new_grants)} Grants | {len(new_apps)} Applications — "
        f"{datetime.now().strftime('%Y-%m-%d')}"
    )
    msg["From"] = SENDER_EMAIL
    msg["To"]   = RECIPIENT_EMAIL
    msg.attach(MIMEText(email_body, "plain"))

    try:
        with open(CUMULATIVE_CSV, "rb") as f:
            attachment = MIMEApplication(f.read(), Name="patents_cumulative.csv")
        attachment["Content-Disposition"] = 'attachment; filename="patents_cumulative.csv"'
        msg.attach(attachment)

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(SENDER_EMAIL, EMAIL_PASSWORD)
            smtp.send_message(msg)
        logger.info("✓ Email sent successfully")
    except Exception as e:
        logger.error(f"Error sending email: {e}")


# ---------------------------------------------------------------
# Backfill first_claim for existing CSV records
# ---------------------------------------------------------------

def backfill_first_claims():
    """
    Fetch and populate first_claim for any existing CSV records where it is empty,
    garbled (mojibake), or truncated at the legacy 1000-char limit.

    Safe to re-run — already-populated rows are skipped.
    Progress saved to CSV after every record (crash-safe).
    Inter-request sleep: 2s base + 3s inside get_first_claim before Google Patents.
    """
    if not CUMULATIVE_CSV.exists():
        logger.info("No existing CSV found — nothing to backfill.")
        return

    df = pd.read_csv(CUMULATIVE_CSV, dtype={'first_claim': object})

    if 'first_claim' not in df.columns:
        df['first_claim'] = pd.array([''] * len(df), dtype=object)

    if 'claim_source' not in df.columns:
        df['claim_source'] = ''

    df['first_claim']  = df['first_claim'].fillna('')
    df['claim_source'] = df['claim_source'].fillna('')

    def _is_garbled(text) -> bool:
        """Detect mojibake — non-ASCII ratio above 15% means encoding failure."""
        if not text or str(text).strip() == '':
            return False
        text = str(text)
        non_ascii = sum(1 for c in text if ord(c) > 127)
        return (non_ascii / len(text)) > 0.15

    truncated_mask = df['first_claim'].fillna('').str.len() == 1000
    mask = (
        (df['first_claim'].astype(str).str.strip() == '')
        | df['first_claim'].apply(_is_garbled)
        | truncated_mask
    )
    to_backfill = df[mask]

    total = len(to_backfill)
    if total == 0:
        logger.info("Backfill: all existing records already have first_claim populated.")
        return

    print("=" * 80)
    print(f"BACKFILL: Fetching first claims for {total} existing records...")
    print("          Strategy: EPO OPS (EP) → Google Patents (all others)")
    print("=" * 80)

    count_success = 0
    count_empty   = 0

    for i, (idx, row) in enumerate(to_backfill.iterrows(), 1):
        country   = str(row.get('country', ''))
        number    = str(row.get('publication_number', ''))
        kind      = str(row.get('kind', ''))
        patent_id = f"{country}{number}{kind}"

        logger.info(f"  [{i}/{total}] {patent_id}...")

        claim, source = get_first_claim(country, number, kind)
        if claim:
            claim_clean = re.sub(r'[^\x00-\x7F]+', ' ', claim)
            claim_clean = re.sub(r'\s{2,}', ' ', claim_clean).strip()
            claim = claim_clean if len(claim_clean) > 50 else ''

        df.at[idx, 'first_claim']  = claim if claim else ''
        df.at[idx, 'claim_source'] = source

        if claim:
            count_success += 1
        else:
            count_empty += 1

        # Save after every record so a crash loses nothing
        df.to_csv(CUMULATIVE_CSV, index=False, quoting=1)
        time.sleep(2)

    print(f"\n[BACKFILL COMPLETE]")
    print(f"  ✓ Claims fetched:    {count_success}")
    print(f"  – Not available:     {count_empty}")
    print(f"  Σ Processed:         {total}")
    print(f"  CSV saved: {CUMULATIVE_CSV}")


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    print("=" * 80)
    print("PATENT SEARCH WITH AI RELEVANCE SCORING — EPO — Starting")
    print("=" * 80)

    # Step 1: Backfill first_claim for any existing records that lack it
    backfill_first_claims()

    # Step 2: Run applications search (60-day window) + grants sweep (historical)
    df_new = search_all_patents()
    df_all = update_cumulative_csv(df_new)
    send_email_with_csv(df_all)

    print("\n" + "=" * 80)
    print("Process completed successfully")
    print("=" * 80)


if __name__ == "__main__":
    main()













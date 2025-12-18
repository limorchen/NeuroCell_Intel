#!/usr/bin/env python3
"""
neurocell_agent_semantic_improved.py

Weekly CNS–exosome intelligence agent:
- Fetch PubMed + ClinicalTrials.gov
- Apply strict exosome filter + semantic filter + combined scoring
- Store in SQLite DB
- Export ONE cumulative CSV per source with a new_this_run flag
- Send email report with only newly inserted items
"""

import os
import time
import csv
import sqlite3
import logging
import requests
import smtplib
import re
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from typing import List, Dict, Any
from Bio import Entrez
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
import torch

# Load environment variables
if os.path.exists(".env"):
    load_dotenv(".env")
elif os.path.exists(".env_semantic"):
    load_dotenv(".env_semantic")

# -------------------------
# Configuration
# -------------------------
DB_FILE = os.getenv("DB_FILE", "neurocell_database_semantic.db")
SENDER_EMAIL = os.getenv("SENDER_EMAIL")
RECIPIENT_EMAIL = os.getenv("RECIPIENT_EMAIL")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 465))
NCBI_EMAIL = os.getenv("NCBI_EMAIL", "chen.limor@gmail.com")

# Search terms with better defaults
PUBMED_TERM = os.getenv(
    "PUBMED_TERM",
    "exosomes AND (spinal cord OR neural OR CNS OR optic nerve)"
)
CLINICALTRIALS_INTERVENTION = os.getenv(
    "CLINICALTRIALS_INTERVENTION",
    "exosomes OR extracellular vesicles OR exosome therapy"
)
CLINICALTRIALS_CONDITION = os.getenv(
    "CLINICALTRIALS_CONDITION",
    "spinal cord injury OR optic nerve injury OR central nervous system"
)

# Construct clinical trials search expression (kept for completeness; not used by v2 API call)
search_parts = []
if CLINICALTRIALS_INTERVENTION:
    search_parts.append(f"intr:({CLINICALTRIALS_INTERVENTION})")
if CLINICALTRIALS_CONDITION:
    search_parts.append(f"cond:({CLINICALTRIALS_CONDITION})")

CLINICALTRIALS_SEARCH_EXPRESSION = " AND ".join(search_parts) if search_parts else "nervous system regeneration"

MAX_RECORDS = int(os.getenv("MAX_RECORDS", 50))
DAYS_BACK_PUBMED = int(os.getenv("DAYS_BACK_PUBMED", 7))
DAYS_BACK_TRIALS = int(os.getenv("DAYS_BACK_TRIALS", 7))
RATE_LIMIT_DELAY = float(os.getenv("RATE_LIMIT_DELAY", 0.5))

# Thresholds
SEMANTIC_THRESHOLD_PUBMED = float(os.getenv("SEMANTIC_THRESHOLD_PUBMED", 0.45))
SEMANTIC_THRESHOLD_TRIALS = float(os.getenv("SEMANTIC_THRESHOLD_TRIALS", 0.45))

# Semantic search terms
raw_terms = os.getenv(
    "SEMANTIC_SEARCH_TERMS",
    "exosome therapy spinal cord injury, "
    "extracellular vesicles neural regeneration, "
    "exosomal treatment CNS damage, "
    "mesenchymal stem cell exosomes spinal injury, "
    "exosome-mediated nerve repair, "
    "therapeutic exosomes optic nerve, "
    "exosome delivery central nervous system, "
    "neuroprotective exosomes brain injury, "
    "exosome spinal cord repair, "
    "extracellular vesicle neurological therapy"
).split(",")
SEMANTIC_SEARCH_TERMS = [s.strip() for s in raw_terms if s.strip()]

Entrez.email = NCBI_EMAIL

PUBMED_FULL_CSV = "all_pubmed_semantic_database.csv"
TRIALS_FULL_CSV = "all_trials_semantic_database.csv"

# -------------------------
# Logging
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("neurocell_agent_semantic.log"), logging.StreamHandler()]
)
logger = logging.getLogger("neurocell_agent")

logger.info("Loading Sentence-Transformer model 'all-MiniLM-L6-v2' ...")
try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("Sentence-Transformer loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load SentenceTransformer model: {e}")
    model = None

# -------------------------
# Utils
# -------------------------
def now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def contains_spinal(*texts: List[str]) -> bool:
    for t in texts:
        if not t:
            continue
        if "spinal" in t.lower() or "sci" in t.lower():
            return True
    return False

def semantic_filter(docs: List[Dict[str, Any]], terms: List[str], threshold: float) -> List[Dict[str, Any]]:
    if not model:
        logger.warning("Semantic model not loaded — skipping semantic filtering.")
        for d in docs:
            d['semantic_score'] = 0.0
        return docs

    if not docs:
        logger.info("No documents to filter")
        return []

    if not terms:
        logger.warning("No semantic search terms provided. Scores will be 0.0.")
        for d in docs:
            d['semantic_score'] = 0.0
        return docs

    logger.info(f"Applying semantic filtering to {len(docs)} documents with {len(terms)} search terms")
    logger.info(f"Semantic threshold: {threshold}")

    try:
        term_embeddings = model.encode(terms, convert_to_tensor=True)
    except Exception as e:
        logger.error(f"Error encoding semantic terms: {e}")
        term_embeddings = None

    filtered_docs = []
    for i, doc in enumerate(docs):
        title = doc.get('title') or ''
        abstract = doc.get('abstract') or doc.get('detailed_description') or ''
        doc_text = (title + " " + abstract).strip()

        if not doc_text:
            logger.debug(f"Document {i+1}: Empty text content")
            doc['semantic_score'] = 0.0
            continue

        try:
            if term_embeddings is not None:
                doc_embedding = model.encode(doc_text, convert_to_tensor=True)
                cosine_scores = util.cos_sim(doc_embedding, term_embeddings)[0]
                max_score = float(torch.max(cosine_scores).item())
                best_term_idx = int(torch.argmax(cosine_scores).item())
                best_term = terms[best_term_idx]
                logger.debug(
                    f"Document {i+1}: '{title[:50]}...' scored {max_score:.4f} "
                    f"(best match: '{best_term}')"
                )
            else:
                max_score = 0.0
                logger.debug(f"Document {i+1}: No embeddings available")
        except Exception as e:
            logger.error(f"Error embedding doc '{title[:50]}': {e}")
            max_score = 0.0

        doc['semantic_score'] = round(max_score, 4)
        if max_score >= threshold:
            filtered_docs.append(doc)
            logger.info(f"Document passed filter: '{title[:50]}...' (score: {max_score:.4f})")

    logger.info(f"Semantic filtering: {len(filtered_docs)}/{len(docs)} documents passed threshold {threshold}")
    return filtered_docs

# Mandatory exosome/EV filter
def mandatory_exosome_filter(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Hard filter to ensure documents contain exosome/EV terminology with word boundaries
    """
    exosome_patterns = [
        r'\bexosome\b',
        r'\bexosomes\b',
        r'\bexosomal\b',
        r'\bextracellular vesicle\b',
        r'\bextracellular vesicles\b',
        r'\bmicrovesicle\b',
        r'\bmicrovesicles\b',
        r'\bEVs\b',
        r'\bexosome-derived\b',
        r'\bexosome therapy\b',
        r'\bexosome-mediated\b',
        r'\btherapeutic exosome\b'
    ]

    filtered_docs = []
    for doc in docs:
        title = (doc.get('title') or '').lower()
        abstract = (doc.get('abstract') or doc.get('detailed_description') or '').lower()
        full_text = title + " " + abstract

        if any(re.search(pattern, full_text, re.IGNORECASE) for pattern in exosome_patterns):
            filtered_docs.append(doc)
        else:
            logger.debug(f"Filtered out (no exosome terms): {doc.get('title', '')[:50]}...")
    logger.info(f"Mandatory exosome filter: {len(filtered_docs)}/{len(docs)} documents contain exosome/EV terms")
    return filtered_docs

# Combined relevance scoring
def calculate_relevance_score(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cns_terms = [
        'spinal cord', 'spinal', 'sci', 'central nervous system', 'cns',
        'optic nerve', 'brain injury', 'neural', 'neurological', 'nerve injury'
    ]
    exo_terms = ['exosome', 'extracellular vesicle', 'exosomal', 'microvesicle']

    for doc in docs:
        title = (doc.get('title') or '').lower()
        abstract = (doc.get('abstract') or doc.get('detailed_description') or '').lower()
        full_text = title + " " + abstract

        cns_hits = sum(1 for term in cns_terms if term in full_text)
        exo_hits = sum(1 for term in exo_terms if term in full_text)

        semantic_score = doc.get('semantic_score', 0)
        relevance_boost = 0.1 * min(cns_hits, 3) + 0.1 * min(exo_hits, 2)

        doc['combined_score'] = round(semantic_score + relevance_boost, 4)
        doc['cns_hits'] = cns_hits
        doc['exo_hits'] = exo_hits

        logger.debug(
            f"Doc: '{doc.get('title', '')[:50]}' - Semantic: {semantic_score:.4f}, "
            f"CNS hits: {cns_hits}, Exo hits: {exo_hits}, Combined: {doc['combined_score']:.4f}"
        )

    return docs

# -------------------------
# DB init
# -------------------------
def init_db(path: str = DB_FILE):
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS pubmed_articles (
        pmid TEXT PRIMARY KEY,
        title TEXT,
        abstract TEXT,
        authors TEXT,
        publication_date TEXT,
        journal TEXT,
        doi TEXT,
        url TEXT,
        spinal_hit INTEGER,
        first_seen TEXT,
        semantic_score REAL,
        combined_score REAL
    );
    """)
    c.execute("""
    CREATE TABLE IF NOT EXISTS clinical_trials (
        nct_id TEXT PRIMARY KEY,
        title TEXT,
        detailed_description TEXT,
        conditions TEXT,
        interventions TEXT,
        phases TEXT,
        study_type TEXT,
        status TEXT,
        start_date TEXT,
        completion_date TEXT,
        sponsor TEXT,
        enrollment TEXT,
        age_range TEXT,
        url TEXT,
        spinal_hit INTEGER,
        first_seen TEXT,
        semantic_score REAL,
        combined_score REAL
    );
    """)
    conn.commit()
    conn.close()

# -------------------------
# PubMed fetcher
# -------------------------
def fetch_pubmed_fixed(term: str, max_records: int = MAX_RECORDS, days_back: int = DAYS_BACK_PUBMED) -> List[Dict[str, Any]]:
    logger.info(f"PubMed search term (RAW): '{term}'")
    logger.info(f"Parameters: days_back={days_back}, retmax={max_records}")

    try:
        logger.info("Attempting PubMed esearch...")

        search_handle = Entrez.esearch(
            db="pubmed",
            term=term,
            retmax=max_records,
            sort="date",
            reldate=days_back,
            usehistory="y"
        )
        search_record = Entrez.read(search_handle)
        search_handle.close()

        ids = search_record.get("IdList", [])
        logger.info(f"PubMed esearch returned {len(ids)} article IDs")

        if ids:
            logger.info(f"First 5 PMIDs: {ids[:5]}")
        else:
            logger.warning("No PubMed articles found - check your search term syntax")
            simple_term = "exosomes AND (nerve OR neural OR CNS)"
            logger.info(f"Trying simplified search: {simple_term}")

            search_handle = Entrez.esearch(
                db="pubmed",
                term=simple_term,
                retmax=max_records,
                sort="date",
                reldate=days_back
            )
            search_record = Entrez.read(search_handle)
            search_handle.close()
            ids = search_record.get("IdList", [])
            logger.info(f"Simplified search returned {len(ids)} article IDs")

        if not ids:
            return []

        time.sleep(RATE_LIMIT_DELAY)

        batch_size = 20
        all_results = []

        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i+batch_size]
            logger.info(f"Fetching batch {i//batch_size + 1}: PMIDs {batch_ids[0]} to {batch_ids[-1]}")

            fetch_handle = Entrez.efetch(
                db="pubmed",
                id=",".join(batch_ids),
                rettype="abstract",
                retmode="xml"
            )
            papers = Entrez.read(fetch_handle)
            fetch_handle.close()

            batch_results = []
            for j, article in enumerate(papers.get("PubmedArticle", [])):
                try:
                    med = article.get("MedlineCitation", {})
                    pmid = str(med.get("PMID", ""))
                    art = med.get("Article", {}) or {}

                    title = str(art.get("ArticleTitle", "")) or ""

                    abstract_list = art.get("Abstract", {}).get("AbstractText", [])
                    if isinstance(abstract_list, list):
                        abstract = " ".join([str(a) for a in abstract_list])
                    else:
                        abstract = str(abstract_list) if abstract_list else ""

                    logger.debug(f"Article {i+j+1}: PMID={pmid}")
                    logger.debug(f"  Title: {title[:100]}...")
                    logger.debug(f"  Abstract length: {len(abstract)}")

                    title_lower = title.lower()
                    abstract_lower = abstract.lower()

                    has_exosome = any(
                        term in title_lower or term in abstract_lower
                        for term in ['exosome', 'extracellular vesicle', 'ev']
                    )
                    has_neuro = any(
                        term in title_lower or term in abstract_lower
                        for term in ['nerve', 'neural', 'neuro', 'cns', 'spinal', 'brain']
                    )

                    relevance_score = 0
                    if has_exosome:
                        relevance_score += 1
                    if has_neuro:
                        relevance_score += 1

                    logger.debug(
                        f"  Relevance check: exosome={has_exosome}, neuro={has_neuro}, "
                        f"score={relevance_score}"
                    )

                    authors = []
                    author_list = art.get("AuthorList", [])
                    if isinstance(author_list, list):
                        for a in author_list[:10]:
                            if isinstance(a, dict):
                                if "LastName" in a and "Initials" in a:
                                    authors.append(f"{a.get('LastName','')} {a.get('Initials','')}")
                                elif "CollectiveName" in a:
                                    authors.append(a.get("CollectiveName", ""))
                    authors_str = ", ".join(authors)

                    journal = str(art.get("Journal", {}).get("Title", ""))

                    pubdate = ""
                    ji = art.get("Journal", {}).get("JournalIssue", {})
                    if ji:
                        pubdate_struct = ji.get("PubDate", {})
                        if isinstance(pubdate_struct, dict):
                            pubdate = str(
                                pubdate_struct.get("Year", "") or
                                pubdate_struct.get("MedlineDate", "")
                            )

                    doi = "N/A"
                    elocs = art.get("ELocationID", [])
                    if isinstance(elocs, list):
                        for e in elocs:
                            if hasattr(e, "attributes") and e.attributes.get("EIdType") == "doi":
                                doi = str(e) or doi
                                break

                    url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                    spinal = 1 if contains_spinal(title, abstract) else 0

                    result = {
                        "pmid": pmid,
                        "title": title,
                        "abstract": abstract,
                        "authors": authors_str,
                        "publication_date": pubdate,
                        "journal": journal,
                        "doi": doi,
                        "url": url,
                        "spinal_hit": spinal,
                        "semantic_score": None,
                        "relevance_score": relevance_score
                    }

                    batch_results.append(result)

                except Exception as e:
                    logger.error(
                        f"Error parsing article in batch {i//batch_size + 1}, "
                        f"position {j}: {e}"
                    )
                    continue

            all_results.extend(batch_results)
            time.sleep(RATE_LIMIT_DELAY)

        logger.info(f"Successfully parsed {len(all_results)} PubMed articles")

        relevance_dist = {}
        for result in all_results:
            score = result.get('relevance_score', 0)
            relevance_dist[score] = relevance_dist.get(score, 0) + 1

        logger.info(f"Relevance score distribution: {relevance_dist}")

        return all_results

    except Exception as e:
        logger.exception(f"PubMed fetch error: {e}")
        return []

# -------------------------
# ClinicalTrials fetcher
# -------------------------
def fetch_clinical_trials_fixed(
    search_intervention: str = "exosomes",
    search_condition: str = "CNS",
    days_back: int = DAYS_BACK_TRIALS,
    max_records: int = MAX_RECORDS
) -> List[Dict[str, Any]]:
    logger.info(
        f"ClinicalTrials.gov search - intervention: '{search_intervention}', "
        f"condition: '{search_condition}'"
    )
    logger.info(f"Days back: {days_back}, Max records: {max_records}")

    base_url = "https://clinicaltrials.gov/api/v2/studies"
    search_results = []

    date_cutoff = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
    logger.info(f"Date cutoff: {date_cutoff}")

    search_strategies = []

    if search_intervention and search_condition:
        search_strategies.append({
            'query.intr': search_intervention,
            'query.cond': search_condition,
            'name': 'Original terms'
        })

    if search_intervention:
        search_strategies.append({
            'query.intr': search_intervention,
            'name': 'Intervention only'
        })

    if search_condition:
        search_strategies.append({
            'query.cond': search_condition,
            'name': 'Condition only'
        })

    if search_intervention or search_condition:
        combined_term = []
        if search_intervention:
            combined_term.append(search_intervention)
        if search_condition:
            combined_term.append(search_condition)

        search_strategies.append({
            'query.term': ' OR '.join(combined_term),
            'name': 'Combined term search'
        })

    search_strategies.append({
        'query.term': 'exosomes OR "extracellular vesicles"',
        'name': 'Fallback broad search'
    })

    for strategy_idx, strategy_params in enumerate(search_strategies):
        logger.info(f"Trying strategy {strategy_idx + 1}: {strategy_params['name']}")

        params = {
            'pageSize': min(100, max_records),
            'format': 'json',
        }

        for key, value in strategy_params.items():
            if key != 'name':
                params[key] = value

        params_with_date = params.copy()
        params_with_date['filter.advanced'] = f'AREA[LastUpdatePostDate]RANGE[{date_cutoff},MAX]'

        logger.info("  Trying with date filter")
        logger.debug(f"  Request params: {params_with_date}")

        try:
            response = requests.get(base_url, params=params_with_date, timeout=30)
            response.raise_for_status()
            data = response.json()

            studies = data.get('studies', [])
            total_count = data.get('totalCount', 0)

            logger.info(
                f"  API Response: {len(studies)} studies returned, totalCount: {total_count}"
            )

            if studies:
                logger.info(
                    f"SUCCESS: Found {len(studies)} studies with {strategy_params['name']} "
                    f"(with date filter)"
                )

                for study in studies:
                    try:
                        protocol_section = study.get('protocolSection', {})
                        identification = protocol_section.get('identificationModule', {})
                        nct_id = identification.get('nctId', '')
                        title = identification.get('briefTitle', '')

                        description = protocol_section.get('descriptionModule', {})
                        summary = description.get('briefSummary', '')
                        detailed_description = description.get('detailedDescription', '') or summary

                        all_text = (title + " " + summary + " " + detailed_description).lower()
                        has_exosome = any(
                            term in all_text
                            for term in ['exosome', 'extracellular vesicle', 'ev', 'microvesicle']
                        )
                        has_neuro = any(
                            term in all_text
                            for term in ['neuro', 'neural', 'nerve', 'cns', 'spinal', 'brain']
                        )

                        logger.debug(
                            f"  Study {nct_id}: exosome={has_exosome}, neuro={has_neuro}"
                        )
                        logger.debug(f"    Title: {title[:100]}...")

                        status_module = protocol_section.get('statusModule', {})
                        status = status_module.get('overallStatus', '')
                        start_date = status_module.get('startDateStruct', {}).get('date', '')
                        completion_date = status_module.get('completionDateStruct', {}).get('date', '')

                        design = protocol_section.get('designModule', {})
                        study_type = design.get('studyType', '')
                        enrollment = design.get('enrollmentInfo', {}).get('count', '')

                        conditions_module = protocol_section.get('conditionsModule', {})
                        conditions_list = conditions_module.get('conditions', [])

                        interventions_module = protocol_section.get('armsInterventionsModule', {})
                        interventions_list = [
                            i.get('name', '') for i in interventions_module.get('interventions', [])
                        ]

                        phases = design.get('phases', [])
                        phases_list = [p for p in phases if p]

                        sponsor_module = protocol_section.get('sponsorCollaboratorsModule', {})
                        sponsor_name = sponsor_module.get('leadSponsor', {}).get('name', '')

                        eligibility = protocol_section.get('eligibilityModule', {})
                        age_min = eligibility.get('minimumAge', '')
                        age_max = eligibility.get('maximumAge', '')
                        age_range = f"{age_min} - {age_max}" if age_min or age_max else "N/A"

                        url_study = f"https://clinicaltrials.gov/study/{nct_id}" if nct_id else ""
                        spinal_hit = 1 if contains_spinal(title, summary, detailed_description) else 0

                        search_results.append({
                            "nct_id": nct_id,
                            "title": title,
                            "detailed_description": detailed_description,
                            "conditions": conditions_list,
                            "interventions": interventions_list,
                            "phases": phases_list,
                            "study_type": study_type,
                            "status": status,
                            "start_date": start_date,
                            "completion_date": completion_date,
                            "sponsor": sponsor_name,
                            "enrollment": str(enrollment),
                            "age_range": age_range,
                            "url": url_study,
                            "spinal_hit": spinal_hit,
                            "semantic_score": None,
                            "search_strategy": strategy_params['name']
                        })
                    except Exception as e:
                        logger.error(f"Error parsing clinical trial: {e}")
                        continue

                if search_results:
                    break

            else:
                logger.info(
                    f"  No studies found with {strategy_params['name']} (with date filter)"
                )

        except requests.RequestException as e:
            logger.error(
                f"  API request failed for {strategy_params['name']} (with date filter): {e}"
            )
            continue

        if search_results:
            break

    logger.info(f"Final result: Found {len(search_results)} clinical trials")

    if search_results:
        strategies_used = set(result.get('search_strategy', 'Unknown') for result in search_results)
        logger.info(f"Successful strategies: {strategies_used}")

    return search_results[:max_records]

# -------------------------
# DB upsert helpers
# -------------------------
def upsert_pubmed(db: str, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not articles:
        return []

    conn = sqlite3.connect(db)
    cur = conn.cursor()
    new_items = []

    for a in articles:
        try:
            cur.execute("""
                INSERT INTO pubmed_articles
                (pmid, title, abstract, authors, publication_date, journal, doi, url,
                 spinal_hit, first_seen, semantic_score, combined_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                a["pmid"], a["title"], a["abstract"], a["authors"],
                a["publication_date"], a["journal"], a["doi"], a["url"],
                a["spinal_hit"], now_ts(), a.get("semantic_score"), a.get("combined_score")
            ))
            new_items.append(a)
        except sqlite3.IntegrityError:
            continue
        except Exception as e:
            logger.error(f"Error inserting article {a.get('pmid', 'unknown')}: {e}")
            continue

    conn.commit()
    conn.close()
    logger.info(f"DB upsert_pubmed: inserted {len(new_items)} new articles")
    return new_items

def upsert_trials(db: str, trials: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not trials:
        return []

    conn = sqlite3.connect(db)
    cur = conn.cursor()
    new_items = []

    for t in trials:
        try:
            cur.execute("""
                INSERT INTO clinical_trials
                (nct_id, title, detailed_description, conditions, interventions, phases,
                 study_type, status, start_date, completion_date, sponsor, enrollment,
                 age_range, url, spinal_hit, first_seen, semantic_score, combined_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                t["nct_id"], t["title"], t["detailed_description"],
                "; ".join(t.get("conditions", [])),
                "; ".join(t.get("interventions", [])),
                "; ".join(t.get("phases", [])),
                t.get("study_type", ""), t.get("status", ""),
                t.get("start_date", ""), t.get("completion_date", ""),
                t.get("sponsor", ""), str(t.get("enrollment", "")),
                t.get("age_range", ""), t.get("url", ""),
                t.get("spinal_hit", 0), now_ts(),
                t.get("semantic_score"), t.get("combined_score")
            ))
            new_items.append(t)
        except sqlite3.IntegrityError:
            continue
        except Exception as e:
            logger.error(f"Error inserting trial {t.get('nct_id', 'unknown')}: {e}")
            continue

    conn.commit()
    conn.close()
    logger.info(f"DB upsert_trials: inserted {len(new_items)} new trials")
    return new_items

# -------------------------
# CSV full exports with new_this_run
# -------------------------
def export_full_csvs(db: str = DB_FILE):
    conn = sqlite3.connect(db)
    cur = conn.cursor()
    now = datetime.now()

    # PubMed
    cur.execute("""
        SELECT pmid,title,abstract,authors,publication_date,journal,doi,url,
               spinal_hit,first_seen,semantic_score,combined_score
        FROM pubmed_articles
    """)
    rows = cur.fetchall()
    with open(PUBMED_FULL_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "pmid","title","abstract","authors","publication_date","journal",
            "doi","url","spinal_hit","first_seen","semantic_score",
            "combined_score","new_this_run"
        ])
        for r in rows:
            first_seen_str = r[9] or ""
            is_new = "NO"
            if first_seen_str:
                try:
                    first_seen_dt = datetime.strptime(first_seen_str, "%Y-%m-%d %H:%M:%S")
                    if (now - first_seen_dt).days < DAYS_BACK_PUBMED:
                        is_new = "YES"
                except Exception:
                    pass
            writer.writerow([
                r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7],
                "YES" if r[8] else "NO", r[9], r[10], r[11], is_new
            ])

    # Trials
    cur.execute("""
        SELECT nct_id,title,detailed_description,conditions,interventions,phases,
               study_type,status,start_date,completion_date,sponsor,enrollment,
               age_range,url,spinal_hit,first_seen,semantic_score,combined_score
        FROM clinical_trials
    """)
    rows = cur.fetchall()
    with open(TRIALS_FULL_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "nct_id","title","detailed_description","conditions","interventions",
            "phases","study_type","status","start_date","completion_date","sponsor",
            "enrollment","age_range","url","spinal_hit","first_seen",
            "semantic_score","combined_score","new_this_run"
        ])
        for r in rows:
            first_seen_str = r[15] or ""
            is_new = "NO"
            if first_seen_str:
                try:
                    first_seen_dt = datetime.strptime(first_seen_str, "%Y-%m-%d %H:%M:%S")
                    if (now - first_seen_dt).days < DAYS_BACK_TRIALS:
                        is_new = "YES"
                except Exception:
                    pass
            writer.writerow([
                r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7],
                r[8], r[9], r[10], r[11], r[12], r[13],
                "YES" if r[14] else "NO", r[15], r[16], r[17], is_new
            ])

    conn.close()
    logger.info("Exported full database to CSV files with new_this_run flag")

# -------------------------
# Email function
# -------------------------
def send_email(
    new_pubmed: List[Dict[str,Any]],
    new_trials: List[Dict[str,Any]],
    stats: Dict[str,int],
    pubmed_term: str,
    trials_intervention: str,
    trials_condition: str
) -> bool:
    if not (SENDER_EMAIL and RECIPIENT_EMAIL and EMAIL_PASSWORD):
        logger.error("Email credentials missing - skipping email send")
        return False

    recipients = [r.strip() for r in RECIPIENT_EMAIL.split(",")]
    msg = MIMEMultipart("mixed")
    msg["From"] = SENDER_EMAIL
    msg["To"] = ", ".join(recipients)
    msg["Subject"] = f"CNS-Exosomes Weekly Intelligence Report - {datetime.now().strftime('%Y-%m-%d')}"

    trials_terms_display = f"{trials_intervention} (Intervention)"
    if trials_condition:
        trials_terms_display += f" AND {trials_condition} (Condition)"

    html = "<h2>CNS-Exosomes Weekly Intelligence Report</h2>"
    html += f"<p><b>PubMed search term:</b> {pubmed_term}</p>"
    html += f"<p><b>ClinicalTrials search terms:</b> {trials_terms_display}</p>"
    html += f"<p>New PubMed articles this week: {stats.get('new_pubmed',0)}</p>"
    html += f"<p>New Clinical Trials this week: {stats.get('new_trials',0)}</p>"

    if new_pubmed:
        html += "<h3>New PubMed Articles (with Combined Scores)</h3><ul>"
        for a in sorted(new_pubmed, key=lambda x: x.get('combined_score', 0), reverse=True):
            sem_score = a.get('semantic_score', 'N/A')
            comb_score = a.get('combined_score', 'N/A')
            html += f"<li><a href='{a['url']}'>{a['title']}</a><br>"
            html += f"<small>Semantic: {sem_score} | Combined: {comb_score}</small></li>"
        html += "</ul>"

    if new_trials:
        html += "<h3>New Clinical Trials (with Combined Scores)</h3><ul>"
        for t in sorted(new_trials, key=lambda x: x.get('combined_score', 0), reverse=True):
            sem_score = t.get('semantic_score', 'N/A')
            comb_score = t.get('combined_score', 'N/A')
            html += f"<li><a href='{t['url']}'>{t['title']}</a> ({t.get('status','')})<br>"
            html += f"<small>Semantic: {sem_score} | Combined: {comb_score}</small></li>"
        html += "</ul>"

    part1 = MIMEText(html, "html")
    msg.attach(part1)

    try:
        # Attach only the cumulative CSVs
        attachments = [PUBMED_FULL_CSV, TRIALS_FULL_CSV]
        for fname in attachments:
            if os.path.exists(fname):
                with open(fname, "rb") as f:
                    part = MIMEBase("application", "octet-stream")
                    part.set_payload(f.read())
                encoders.encode_base64(part)
                part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(fname)}")
                msg.attach(part)

        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
            server.login(SENDER_EMAIL, EMAIL_PASSWORD)
            server.sendmail(SENDER_EMAIL, recipients, msg.as_string())

        logger.info("Email sent successfully")
        return True

    except Exception as e:
        logger.exception("Failed to send email")
        return False

# -------------------------
# Main weekly update
# -------------------------
def weekly_update():
    logger.info("=== Starting Weekly Update ===")
    init_db()

    # Step 1: Fetch data
    logger.info("Step 1: Fetching PubMed articles...")
    pubmed_articles = fetch_pubmed_fixed(PUBMED_TERM, MAX_RECORDS * 2, DAYS_BACK_PUBMED)

    logger.info("Step 2: Fetching Clinical Trials...")
    trials = fetch_clinical_trials_fixed(
       search_intervention=CLINICALTRIALS_INTERVENTION,
       search_condition=CLINICALTRIALS_CONDITION,
       days_back=DAYS_BACK_TRIALS,
       max_records=MAX_RECORDS * 2
    )

    # Mandatory exosome filter
    logger.info("Step 3: Applying mandatory exosome/EV filter to PubMed articles...")
    exosome_pubmed = mandatory_exosome_filter(pubmed_articles)

    logger.info("Step 4: Applying mandatory exosome/EV filter to Clinical Trials...")
    exosome_trials = mandatory_exosome_filter(trials)

    # Semantic filtering
    logger.info("Step 5: Applying semantic filtering to PubMed articles...")
    relevant_pubmed = semantic_filter(exosome_pubmed, SEMANTIC_SEARCH_TERMS, SEMANTIC_THRESHOLD_PUBMED)

    logger.info("Step 6: Applying semantic filtering to Clinical Trials...")
    relevant_trials = semantic_filter(exosome_trials, SEMANTIC_SEARCH_TERMS, SEMANTIC_THRESHOLD_TRIALS)

    # Combined relevance scores
    logger.info("Step 7: Calculating combined relevance scores for PubMed...")
    relevant_pubmed = calculate_relevance_score(relevant_pubmed)

    logger.info("Step 8: Calculating combined relevance scores for Clinical Trials...")
    relevant_trials = calculate_relevance_score(relevant_trials)

    # Sort and limit
    logger.info("Step 9: Sorting and limiting results by combined score...")
    final_pubmed = sorted(relevant_pubmed, key=lambda x: x.get('combined_score', 0), reverse=True)[:MAX_RECORDS]
    final_trials = sorted(relevant_trials, key=lambda x: x.get('combined_score', 0), reverse=True)[:MAX_RECORDS]

    logger.info(f"Final selection: {len(final_pubmed)} PubMed articles, {len(final_trials)} clinical trials")

    # Upsert into DB and determine new items
    logger.info("Step 10: Updating database...")
    new_pubmed = upsert_pubmed(DB_FILE, final_pubmed)
    new_trials = upsert_trials(DB_FILE, final_trials)

    stats = {
        "new_pubmed": len(new_pubmed),
        "new_trials": len(new_trials)
    }

    # Export single cumulative CSVs with new_this_run flag
    logger.info("Step 11: Exporting cumulative CSVs with new_this_run flag...")
    export_full_csvs(DB_FILE)

    # Send email with new items; attach cumulative CSVs
    logger.info("Step 12: Sending email report...")
    send_email(new_pubmed, new_trials, stats, PUBMED_TERM, CLINICALTRIALS_INTERVENTION, CLINICALTRIALS_CONDITION)

    logger.info("=== Weekly Update Complete ===")

if __name__ == "__main__":
    weekly_update()

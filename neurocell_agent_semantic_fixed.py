#!/usr/bin/env python3
"""
neurocell_agent_semantic_fixed.py

This script automates the search for relevant scientific articles and clinical trials
using a combination of keyword and semantic search. It fetches data from PubMed and
ClinicalTrials.gov, filters it for relevance, stores it in an SQLite database,
exports it to CSV files, and sends an email report.
"""

import os
import time
import csv
import sqlite3
import logging
import requests
import smtplib
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from typing import List, Dict, Any
from Bio import Entrez
from dotenv import load_dotenv # Used to load secrets/config locally or in CI/CD
from sentence_transformers import SentenceTransformer, util
import torch

# Load environment variables.
# FIX: The GitHub Action creates a file named '.env'. This command loads either '.env' (for CI/CD)
# or '.env_semantic' (for local testing), prioritizing the standard '.env' if present.
if os.path.exists(".env"):
    load_dotenv(".env")
elif os.path.exists(".env_semantic"):
    load_dotenv(".env_semantic")
else:
    # If neither is found, variables will default to hardcoded values or fail.
    pass

# -------------------------
# Configuration (CRITICAL FIXES HERE)
# -------------------------
DB_FILE = os.getenv("DB_FILE", "neurocell_database_semantic.db")
SENDER_EMAIL = os.getenv("SENDER_EMAIL")
RECIPIENT_EMAIL = os.getenv("RECIPIENT_EMAIL")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 465))
NCBI_EMAIL = os.getenv("NCBI_EMAIL", "chen.limor@gmail.com")

# Retrieve search terms from environment variables. Use robust defaults.
PUBMED_TERM = os.getenv("PUBMED_TERM", "exosomes AND Spinal")
CLINICALTRIALS_INTERVENTION = os.getenv("CLINICALTRIALS_INTERVENTION", "exosomes OR extracellular vesicles")
CLINICALTRIALS_CONDITION = os.getenv("CLINICALTRIALS_CONDITION", "spinal cord injury OR optic nerve")

# FIX: Construct the combined search expression dynamically, ensuring terms are not empty.
# This prevents malformed queries like 'intr:() AND cond:()' when a secret is empty.
search_parts = []
if CLINICALTRIALS_INTERVENTION:
    search_parts.append(f"intr:({CLINICALTRIALS_INTERVENTION})")
if CLINICALTRIALS_CONDITION:
    search_parts.append(f"cond:({CLINICALTRIALS_CONDITION})")

CLINICALTRIALS_SEARCH_EXPRESSION = " AND ".join(search_parts) if search_parts else "nervous system regeneration"
if not CLINICALTRIALS_SEARCH_EXPRESSION:
    # Fallback to a very general term if all secrets are empty
    CLINICALTRIALS_SEARCH_EXPRESSION = "nervous system regeneration"

MAX_RECORDS = int(os.getenv("MAX_RECORDS", 50))
DAYS_BACK_PUBMED = int(os.getenv("DAYS_BACK_PUBMED", 7))
DAYS_BACK_TRIALS = int(os.getenv("DAYS_BACK_TRIALS", 7))
RATE_LIMIT_DELAY = float(os.getenv("RATE_LIMIT_DELAY", 0.5))
SEMANTIC_THRESHOLD_PUBMED = float(os.getenv("SEMANTIC_THRESHOLD", 0.28))
SEMANTIC_THRESHOLD_TRIALS = float(os.getenv("SEMANTIC_THRESHOLD_TRIALS", 0.26))
# FIX: Ensure SEMANTIC_SEARCH_TERMS is an array of non-empty strings.
raw_terms = os.getenv(
    "SEMANTIC_SEARCH_TERMS",
    "exosomes spinal cord injury, extracellular vesicles neural repair, targeted neurological therapy, brain injury exosome"
).split(",")
SEMANTIC_SEARCH_TERMS = [s.strip() for s in raw_terms if s.strip()]

Entrez.email = NCBI_EMAIL

PUBMED_WEEKLY_CSV = "new_pubmed_semantic_this_week.csv"
TRIALS_WEEKLY_CSV = "new_trials_semantic_this_week.csv"
PUBMED_FULL_CSV = "all_pubmed_semantic_database.csv"
TRIALS_FULL_CSV = "all_trials_semantic_database.csv"

# -------------------------
# Logging (No change)
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
# Utils (No change)
# -------------------------
def now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def contains_spinal(*texts: List[str]) -> bool:
    # A simple keyword hit is retained for easy database filtering.
    for t in texts:
        if not t:
            continue
        if "spinal" in t.lower() or "sci" in t.lower():
            return True
    return False

def semantic_filter(docs: List[Dict[str, Any]], terms: List[str], threshold: float) -> List[Dict[str, Any]]:
    if not model:
        logger.warning("Semantic model not loaded — skipping semantic ing.")
        for d in docs:
            d['semantic_score'] = 0.0
        return docs

    if not docs:
        logger.info("No documents to filter")
        return []

    # FIX: Check if terms are empty after loading from environment (critical for 0.0 scores)
    if not terms:
        logger.warning("No semantic search terms provided from environment/secrets. Scores will be 0.0.")
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
        
        # Combine title and abstract/description for a rich text body
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
                logger.debug(f"Document {i+1}: '{title[:50]}...' scored {max_score:.4f} (best match: '{best_term}')")
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

# Mandatory exosomes filter 

def mandatory_exosome_filter(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Hard filter to ensure documents contain exosome/EV terminology
    """
    exosome_terms = ['exosome', 'exosomes', 'extracellular vesicle', 'extracellular vesicles', 
                     'ev', 'evs', 'microvesicle', 'exosomal']
    
    filtered_docs = []
    for doc in docs:
        title = (doc.get('title') or '').lower()
        abstract = (doc.get('abstract') or doc.get('detailed_description') or '').lower()
        full_text = title + " " + abstract
        
        if any(term in full_text for term in exosome_terms):
            filtered_docs.append(doc)
        else:
            logger.debug(f"Filtered out (no exosome terms): {doc.get('title', '')[:50]}...")
    
    logger.info(f"Mandatory exosome filter: {len(filtered_docs)}/{len(docs)} documents contain exosome/EV terms")
    return filtered_docs


# -------------------------
# DB init (No change)
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
        semantic_score REAL
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
        semantic_score REAL
    );
    """)
    conn.commit()
    conn.close()

# -------------------------
# PubMed fetcher (No change)
# -------------------------
def fetch_pubmed_fixed(term: str, max_records: int = MAX_RECORDS, days_back: int = DAYS_BACK_PUBMED) -> List[Dict[str, Any]]:
    """
    Fixed PubMed search with better query construction and debugging
    """
    # ISSUE 1: Your search might be too broad or malformed
    # Let's add debugging and fix the query structure
    
    logger.info(f"PubMed search term (RAW): '{term}'")
    logger.info(f"Parameters: days_back={days_back}, retmax={max_records}")
    
    try:
        # DEBUGGING: Let's see what the actual query looks like
        logger.info("Attempting PubMed esearch...")
        
        search_handle = Entrez.esearch(
            db="pubmed", 
            term=term, 
            retmax=max_records, 
            sort="date",
            reldate=days_back,
            usehistory="y"  # Add this for better search handling
        )
        search_record = Entrez.read(search_handle)
        search_handle.close()
        
        ids = search_record.get("IdList", [])
        logger.info(f"PubMed esearch returned {len(ids)} article IDs")
        
        # DEBUGGING: Log the first few IDs to verify
        if ids:
            logger.info(f"First 5 PMIDs: {ids[:5]}")
        else:
            logger.warning("No PubMed articles found - check your search term syntax")
            # Let's try a simpler version of your search term
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
        
        # Fetch details in smaller batches to avoid timeouts
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
                    
                    # Better abstract extraction
                    abstract_list = art.get("Abstract", {}).get("AbstractText", [])
                    if isinstance(abstract_list, list):
                        abstract = " ".join([str(a) for a in abstract_list])
                    else:
                        abstract = str(abstract_list) if abstract_list else ""
                    
                    # DEBUGGING: Log what we found
                    logger.debug(f"Article {i+j+1}: PMID={pmid}")
                    logger.debug(f"  Title: {title[:100]}...")
                    logger.debug(f"  Abstract length: {len(abstract)}")
                    
                    # Check if this article actually matches our terms
                    title_lower = title.lower()
                    abstract_lower = abstract.lower()
                    
                    # Look for our key terms
                    has_exosome = any(term in title_lower or term in abstract_lower 
                                    for term in ['exosome', 'extracellular vesicle', 'ev'])
                    has_neuro = any(term in title_lower or term in abstract_lower 
                                  for term in ['nerve', 'neural', 'neuro', 'cns', 'spinal', 'brain'])
                    
                    relevance_score = 0
                    if has_exosome: relevance_score += 1
                    if has_neuro: relevance_score += 1
                    
                    logger.debug(f"  Relevance check: exosome={has_exosome}, neuro={has_neuro}, score={relevance_score}")
                    
                    # Extract other fields...
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
                            pubdate = str(pubdate_struct.get("Year", "") or pubdate_struct.get("MedlineDate", ""))
                    
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
                        "relevance_score": relevance_score  # Add this for debugging
                    }
                    
                    batch_results.append(result)
                    
                except Exception as e:
                    logger.error(f"Error parsing article in batch {i//batch_size + 1}, position {j}: {e}")
                    continue
            
            all_results.extend(batch_results)
            time.sleep(RATE_LIMIT_DELAY)  # Rate limit between batches
        
        logger.info(f"Successfully parsed {len(all_results)} PubMed articles")
        
        # DEBUGGING: Show relevance distribution
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
# ClinicalTrials fetcher (No change needed since API expression is corrected above)
# -------------------------
def fetch_clinical_trials_fixed(
    search_intervention: str = "exosomes",
    search_condition: str = "CNS",
    days_back: int = DAYS_BACK_TRIALS,
    max_records: int = MAX_RECORDS
) -> List[Dict[str, Any]]:
    """
    Fixed Clinical Trials search with better debugging and fallback strategies
    """
    logger.info(f"ClinicalTrials.gov search - intervention: '{search_intervention}', condition: '{search_condition}'")
    logger.info(f"Days back: {days_back}, Max records: {max_records}")

    base_url = "https://clinicaltrials.gov/api/v2/studies"
    search_results = []
    page_token = None

    date_cutoff = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
    logger.info(f"Date cutoff: {date_cutoff}")

    # Try different search strategies
    search_strategies = []
    
    # Strategy 1: Your original terms
    if search_intervention and search_condition:
        search_strategies.append({
            'query.intr': search_intervention,
            'query.cond': search_condition,
            'name': 'Original terms'
        })
    
    # Strategy 2: Just intervention
    if search_intervention:
        search_strategies.append({
            'query.intr': search_intervention,
            'name': 'Intervention only'
        })
    
    # Strategy 3: Just condition
    if search_condition:
        search_strategies.append({
            'query.cond': search_condition,
            'name': 'Condition only'
        })
    
    # Strategy 4: Combined search
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
    
    # Strategy 5: Fallback broad search
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
        
        # Add the strategy-specific parameters
        for key, value in strategy_params.items():
            if key != 'name':
                params[key] = value
        
        # Add date filter (try with and without)
        params_with_date = params.copy()
        params_with_date['filter.advanced'] = f'AREA[LastUpdatePostDate]RANGE[{date_cutoff},MAX]'
        
        for date_filter_name, current_params in [('with date filter', params_with_date), ('without date filter', params)]:
            logger.info(f"  Trying {date_filter_name}")
            logger.debug(f"  Request params: {current_params}")
            
            try:
                response = requests.get(base_url, params=current_params, timeout=30)
                response.raise_for_status()
                data = response.json()

                studies = data.get('studies', [])
                total_count = data.get('totalCount', 0)
                
                logger.info(f"  API Response: {len(studies)} studies returned, totalCount: {total_count}")
                
                if studies:
                    logger.info(f"SUCCESS: Found {len(studies)} studies with {strategy_params['name']} ({date_filter_name})")
                    
                    # Process the studies
                    for study in studies:
                        try:
                            protocol_section = study.get('protocolSection', {})
                            identification = protocol_section.get('identificationModule', {})
                            nct_id = identification.get('nctId', '')
                            title = identification.get('briefTitle', '')

                            # DEBUGGING: Check relevance
                            description = protocol_section.get('descriptionModule', {})
                            summary = description.get('briefSummary', '')
                            detailed_description = description.get('detailedDescription', '') or summary
                            
                            # Check for exosome/EV terms
                            all_text = (title + " " + summary + " " + detailed_description).lower()
                            has_exosome = any(term in all_text for term in ['exosome', 'extracellular vesicle', 'ev', 'microvesicle'])
                            has_neuro = any(term in all_text for term in ['neuro', 'neural', 'nerve', 'cns', 'spinal', 'brain'])
                            
                            logger.debug(f"  Study {nct_id}: exosome={has_exosome}, neuro={has_neuro}")
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
                            interventions_list = [i.get('name', '') for i in interventions_module.get('interventions', [])]

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
                                "search_strategy": strategy_params['name']  # Add for debugging
                            })
                        except Exception as e:
                            logger.error(f"Error parsing clinical trial: {e}")
                            continue

                    # If we found results, break out of the strategy loop
                    if search_results:
                        break
                        
                else:
                    logger.info(f"  No studies found with {strategy_params['name']} ({date_filter_name})")
                    
            except requests.RequestException as e:
                logger.error(f"  API request failed for {strategy_params['name']} ({date_filter_name}): {e}")
                continue
        
        # If we found results, break out of the strategy loop
        if search_results:
            break

    logger.info(f"Final result: Found {len(search_results)} clinical trials")
    
    if search_results:
        # Show which strategy worked
        strategies_used = set(result.get('search_strategy', 'Unknown') for result in search_results)
        logger.info(f"Successful strategies: {strategies_used}")
    
    return search_results[:max_records]



# -------------------------
# DB upsert helpers (No change)
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
                (pmid, title, abstract, authors, publication_date, journal, doi, url, spinal_hit, first_seen, semantic_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                a["pmid"], a["title"], a["abstract"], a["authors"], 
                a["publication_date"], a["journal"], a["doi"], a["url"], 
                a["spinal_hit"], now_ts(), a.get("semantic_score")
            ))
            new_items.append(a)
        except sqlite3.IntegrityError:
            # Article already exists
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
                (nct_id, title, detailed_description, conditions, interventions, phases, study_type, status, start_date, completion_date, sponsor, enrollment, age_range, url, spinal_hit, first_seen, semantic_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                t["nct_id"], t["title"], t["detailed_description"], 
                "; ".join(t.get("conditions", [])),
                "; ".join(t.get("interventions", [])), 
                "; ".join(t.get("phases", [])),
                t.get("study_type", ""), t.get("status", ""), 
                t.get("start_date", ""), t.get("completion_date", ""),
                t.get("sponsor", ""), str(t.get("enrollment", "")), 
                t.get("age_range", ""), t.get("url", ""), 
                t.get("spinal_hit", 0), now_ts(), t.get("semantic_score")
            ))
            new_items.append(t)
        except sqlite3.IntegrityError:
            # Trial already exists
            continue
        except Exception as e:
            logger.error(f"Error inserting trial {t.get('nct_id', 'unknown')}: {e}")
            continue
            
    conn.commit()
    conn.close()
    logger.info(f"DB upsert_trials: inserted {len(new_items)} new trials")
    return new_items

# -------------------------
# CSV helpers (No change)
# -------------------------
def append_pubmed_csv(rows: List[Dict[str, Any]], path: str = PUBMED_WEEKLY_CSV):
    if not rows: 
        return
        
    file_exists = os.path.isfile(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        fieldnames = ["pmid","title","abstract","authors","publication_date","journal","doi","url","spinal_hit","first_seen","semantic_score"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for r in rows:
            writer.writerow({
                "pmid": r["pmid"],
                "title": r["title"],
                "abstract": r["abstract"],
                "authors": r["authors"],
                "publication_date": r["publication_date"],
                "journal": r["journal"],
                "doi": r["doi"],
                "url": r["url"],
                "spinal_hit": "YES" if r["spinal_hit"] else "NO",
                "first_seen": now_ts(),
                "semantic_score": r.get("semantic_score", "")
            })

def append_trials_csv(rows: List[Dict[str, Any]], path: str = TRIALS_WEEKLY_CSV):
    if not rows:
        return
        
    file_exists = os.path.isfile(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        fieldnames = ["nct_id","title","detailed_description","conditions","interventions","phases","study_type","status","start_date","completion_date","sponsor","enrollment","age_range","url","spinal_hit","first_seen","semantic_score"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for r in rows:
            writer.writerow({
                "nct_id": r["nct_id"],
                "title": r["title"],
                "detailed_description": r["detailed_description"],
                "conditions": "; ".join(r.get("conditions", [])),
                "interventions": "; ".join(r.get("interventions", [])),
                "phases": "; ".join(r.get("phases", [])),
                "study_type": r.get("study_type",""),
                "status": r.get("status",""),
                "start_date": r.get("start_date",""),
                "completion_date": r.get("completion_date",""),
                "sponsor": r.get("sponsor",""),
                "enrollment": r.get("enrollment",""),
                "age_range": r.get("age_range",""),
                "url": r.get("url",""),
                "spinal_hit": "YES" if r["spinal_hit"] else "NO",
                "first_seen": now_ts(),
                "semantic_score": r.get("semantic_score", "")
            })

# -------------------------
# Export full CSVs (No change)
# -------------------------
def export_full_csvs(db: str = DB_FILE):
    conn = sqlite3.connect(db)
    cur = conn.cursor()
    
    # PubMed
    cur.execute("SELECT pmid,title,abstract,authors,publication_date,journal,doi,url,spinal_hit,first_seen,semantic_score FROM pubmed_articles")
    rows = cur.fetchall()
    with open(PUBMED_FULL_CSV,"w",newline="",encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["pmid","title","abstract","authors","publication_date","journal","doi","url","spinal_hit","first_seen","semantic_score"])
        for r in rows:
            writer.writerow([r[0],r[1],r[2],r[3],r[4],r[5],r[6],r[7],"YES" if r[8] else "NO",r[9], r[10]])
    
    # Trials
    cur.execute("SELECT nct_id,title,detailed_description,conditions,interventions,phases,study_type,status,start_date,completion_date,sponsor,enrollment,age_range,url,spinal_hit,first_seen,semantic_score FROM clinical_trials")
    rows = cur.fetchall()
    with open(TRIALS_FULL_CSV,"w",newline="",encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["nct_id","title","detailed_description","conditions","interventions","phases","study_type","status","start_date","completion_date","sponsor","enrollment","age_range","url","spinal_hit","first_seen","semantic_score"])
        for r in rows:
            writer.writerow([r[0],r[1],r[2],r[3],r[4],r[5],r[6],r[7],r[8],r[9],r[10],r[11],r[12],r[13],"YES" if r[14] else "NO",r[15], r[16]])
    
    conn.close()
    logger.info("Exported full database to CSV files")

# -------------------------
# Email function (FIXED)
# -------------------------
def send_email(new_pubmed: List[Dict[str,Any]], new_trials: List[Dict[str,Any]], stats: Dict[str,int], pubmed_term: str, trials_intervention: str, trials_condition: str) -> bool:
    if not (SENDER_EMAIL and RECIPIENT_EMAIL and EMAIL_PASSWORD):
        logger.error("Email credentials missing - skipping email send")
        return False
    
    recipients = [r.strip() for r in RECIPIENT_EMAIL.split(",")]
    msg = MIMEMultipart("mixed")
    msg["From"] = SENDER_EMAIL
    msg["To"] = ", ".join(recipients)
    msg["Subject"] = f"CNS-Exosomes Weekly Intelligence Report - {datetime.now().strftime('%Y-%m-%d')}"
    
    # Display ClinicalTrials search terms clearly
    trials_terms_display = f"{trials_intervention} (Intervention)"
    if trials_condition:
        trials_terms_display += f" AND {trials_condition} (Condition)"

    html = f"<h2>CNS-Exosomes Weekly Intelligence Report</h2>"
    html += f"<p><b>PubMed search term:</b> {pubmed_term}</p>"
    html += f"<p><b>ClinicalTrials search terms:</b> {trials_terms_display}</p>"
    html += f"<p>New PubMed articles this week: {stats.get('new_pubmed',0)}</p>"
    html += f"<p>New Clinical Trials this week: {stats.get('new_trials',0)}</p>"
    
    if new_pubmed:
        html += "<h3>New PubMed Articles (with Semantic Scores)</h3><ul>"
        for a in sorted(new_pubmed, key=lambda x: x.get('semantic_score', 0), reverse=True):
            html += f"<li><a href='{a['url']}'>{a['title']}</a> (Score: {a.get('semantic_score', 'N/A')})</li>"
        html += "</ul>"
        
    if new_trials:
        html += "<h3>New Clinical Trials (with Semantic Scores)</h3><ul>"
        for t in sorted(new_trials, key=lambda x: x.get('semantic_score', 0), reverse=True):
            html += f"<li><a href='{t['url']}'>{t['title']}</a> ({t.get('status','')}) (Score: {t.get('semantic_score', 'N/A')})</li>"
        html += "</ul>"
    
    part1 = MIMEText(html, "html")
    msg.attach(part1)
    
    try:
        attachments = [PUBMED_WEEKLY_CSV, TRIALS_WEEKLY_CSV, PUBMED_FULL_CSV, TRIALS_FULL_CSV]
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
# Main weekly update (No change)
# -------------------------
def weekly_update():
    logger.info("=== Starting Weekly Update ===")
    init_db()

    # Step 1: Fetch data using improved searches
    logger.info("Step 1: Fetching PubMed articles...")
    # NOTE: fetch_pubmed will now use the revised PUBMED_TERM from config
    pubmed_articles = fetch_pubmed_fixed(PUBMED_TERM, MAX_RECORDS * 2, DAYS_BACK_PUBMED)
    trials = fetch_clinical_trials_fixed(
       search_intervention=CLINICALTRIALS_INTERVENTION,
       search_condition=CLINICALTRIALS_CONDITION,
       days_back=DAYS_BACK_TRIALS,
       max_records=MAX_RECORDS * 2
    )

    # Step 2: Apply mandatory exosome filter first
    logger.info("Step 3: Applying mandatory exosome/EV filter to PubMed articles...")
    exosome_pubmed = mandatory_exosome_filter(pubmed_articles)

    logger.info("Step 4: Applying mandatory exosome/EV filter to Clinical Trials...")
    exosome_trials = mandatory_exosome_filter(trials)

    # Step 3: Apply semantic filtering
    logger.info("Step 5: Applying semantic filtering to PubMed articles...")
    relevant_pubmed = semantic_filter(exosome_pubmed, SEMANTIC_SEARCH_TERMS, SEMANTIC_THRESHOLD_PUBMED)

    logger.info("Step 6: Applying semantic filtering to Clinical Trials...")
    relevant_trials = semantic_filter(exosome_trials, SEMANTIC_SEARCH_TERMS, SEMANTIC_THRESHOLD_TRIALS)
    
    # Step 3: Take the top N results and sort by semantic score
    logger.info("Step 5: Sorting and limiting results...")
    final_pubmed = sorted(relevant_pubmed, key=lambda x: x.get('semantic_score', 0), reverse=True)[:MAX_RECORDS]
    final_trials = sorted(relevant_trials, key=lambda x: x.get('semantic_score', 0), reverse=True)[:MAX_RECORDS]
    
    logger.info(f"Final selection: {len(final_pubmed)} PubMed articles, {len(final_trials)} clinical trials")

    # Step 4: Upsert into DB and get new items
    logger.info("Step 6: Updating database...")
    new_pubmed = upsert_pubmed(DB_FILE, final_pubmed)
    new_trials = upsert_trials(DB_FILE, final_trials)

    # Step 5: Append weekly CSVs
    logger.info("Step 7: Updating CSV files...")
    append_pubmed_csv(new_pubmed)
    append_trials_csv(new_trials)

    # Step 6: Export full CSVs
    logger.info("Step 8: Exporting full database to CSV...")
    export_full_csvs()

    # Step 7: Send summary email
    logger.info("Step 9: Sending email report...")
    stats = {"new_pubmed": len(new_pubmed), "new_trials": len(new_trials)}
    email_sent = send_email(new_pubmed, new_trials, stats, PUBMED_TERM, CLINICALTRIALS_INTERVENTION, CLINICALTRIALS_CONDITION)
    
    # Summary
    logger.info("=== Weekly Update Complete ===")
    logger.info(f"Summary:")
    logger.info(f"  - PubMed articles fetched: {len(pubmed_articles)}")
    logger.info(f"  - PubMed articles after semantic filtering: {len(relevant_pubmed)}")
    logger.info(f"  - New PubMed articles added to database: {len(new_pubmed)}")
    logger.info(f"  - Clinical trials fetched: {len(trials)}")
    logger.info(f"  - Clinical trials after semantic filtering: {len(relevant_trials)}")
    logger.info(f"  - New clinical trials added to database: {len(new_trials)}")
    logger.info(f"  - Email sent: {'Yes' if email_sent else 'No'}")
    
    return {
        "pubmed_fetched": len(pubmed_articles),
        "pubmed_filtered": len(relevant_pubmed),
        "pubmed_new": len(new_pubmed),
        "trials_fetched": len(trials),
        "trials_filtered": len(relevant_trials),
        "trials_new": len(new_trials),
        "email_sent": email_sent
    }

# -------------------------
# Entry point for manual execution
# -------------------------
if __name__ == "__main__":
    try:
        results = weekly_update()
        logger.info("Script completed successfully")
        print("Weekly update completed!")
        print(f"Results: {results}")
        exit(0) # Successful exit
    except Exception as e:
        logger.exception("Script failed unexpectedly")
        # Ensure a non-zero exit code to alert the GitHub Action runner
        exit(1)

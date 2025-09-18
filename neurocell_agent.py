#!/usr/bin/env python3
"""
Updated ClinicalTrials.gov API v2 implementation
This replaces the old deprecated API calls
"""

import requests
import json
import logging
from typing import List, Dict, Any
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def contains_spinal(*texts: List[str]) -> bool:
    """Case-insensitive check for the word 'spinal' in any provided text."""
    for t in texts:
        if not t:
            continue
        if "spinal" in t.lower():
            return True
    return False

def fetch_clinical_trials_v2(query: str, max_records: int = 20) -> List[Dict[str, Any]]:
    """
    Use the NEW ClinicalTrials.gov API v2
    
    The new API endpoint is: https://clinicaltrials.gov/api/v2/studies
    """
    logger.info(f"ClinicalTrials.gov v2 API search: {query} | max_records={max_records}")
    
    # New API v2 endpoint
    url = "https://clinicaltrials.gov/api/v2/studies"
    
    # New API v2 parameters
    params = {
        "query.term": query,
        "pageSize": min(max_records, 1000),  # API limit is 1000
        "format": "json"
    }
    
    try:
        logger.info(f"Making request to: {url}")
        logger.info(f"With params: {params}")
        
        response = requests.get(url, params=params, timeout=30)
        logger.info(f"Response status: {response.status_code}")
        logger.info(f"Full URL: {response.url}")
        
        if response.status_code != 200:
            logger.error(f"API returned status {response.status_code}: {response.text[:500]}")
            return []
        
        data = response.json()
        logger.info(f"Response keys: {list(data.keys())}")
        
        # Parse the new API v2 response structure
        total_count = data.get("totalCount", 0)
        studies = data.get("studies", [])
        
        logger.info(f"Total studies found: {total_count}")
        logger.info(f"Studies in response: {len(studies)}")
        
        if not studies:
            logger.warning("No studies found in response")
            return []
        
        results = []
        for i, study in enumerate(studies):
            try:
                # Extract data from the new v2 structure
                protocol_section = study.get("protocolSection", {})
                identification = protocol_section.get("identificationModule", {})
                description = protocol_section.get("descriptionModule", {})
                status = protocol_section.get("statusModule", {})
                design = protocol_section.get("designModule", {})
                conditions = protocol_section.get("conditionsModule", {})
                interventions = protocol_section.get("armsInterventionsModule", {})
                eligibility = protocol_section.get("eligibilityModule", {})
                contacts = protocol_section.get("contactsLocationsModule", {})
                
                # Basic info
                nct_id = identification.get("nctId", "")
                title = identification.get("briefTitle", "")
                detailed_description = description.get("detailedDescription", "")
                brief_summary = description.get("briefSummary", "")
                
                # Status info
                overall_status = status.get("overallStatus", "")
                start_date = status.get("startDateStruct", {}).get("date", "")
                completion_date = status.get("primaryCompletionDateStruct", {}).get("date", "")
                
                # Study details
                phases = design.get("phases", [])
                study_type = design.get("studyType", "")
                
                # Conditions and interventions
                condition_list = conditions.get("conditions", [])
                intervention_list = []
                if interventions.get("interventions"):
                    intervention_list = [i.get("name", "") for i in interventions.get("interventions", [])]
                
                # Sponsor and enrollment
                sponsor_info = protocol_section.get("sponsorCollaboratorsModule", {})
                lead_sponsor = sponsor_info.get("leadSponsor", {}).get("name", "")
                
                enrollment_info = design.get("enrollmentInfo", {})
                enrollment_count = enrollment_info.get("count", "")
                
                # Eligibility
                min_age = eligibility.get("minimumAge", "")
                max_age = eligibility.get("maximumAge", "")
                age_range = f"{min_age} to {max_age}" if min_age or max_age else ""
                
                # URL
                study_url = f"https://clinicaltrials.gov/study/{nct_id}" if nct_id else ""
                
                # Check for spinal hits
                spinal_hit = 1 if contains_spinal(title, detailed_description, brief_summary) else 0
                
                # Log first few for debugging
                if i < 3:
                    logger.info(f"Study {i+1}: {nct_id} - {title[:50]}...")
                
                result = {
                    "nct_id": nct_id,
                    "title": title,
                    "detailed_description": detailed_description or brief_summary,
                    "conditions": condition_list,
                    "interventions": intervention_list,
                    "phases": phases,
                    "study_type": study_type,
                    "status": overall_status,
                    "start_date": start_date,
                    "completion_date": completion_date,
                    "sponsor": lead_sponsor,
                    "enrollment": str(enrollment_count),
                    "age_range": age_range,
                    "url": study_url,
                    "spinal_hit": spinal_hit
                }
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing study {i}: {e}")
                continue
        
        logger.info(f"Successfully processed {len(results)} studies")
        return results
        
    except requests.exceptions.Timeout:
        logger.error("Request timed out")
        return []
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error: {e}")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        return []
    except Exception as e:
        logger.exception("Unexpected error in ClinicalTrials.gov v2 fetch")
        return []

def test_new_api():
    """Test the new API with various search terms"""
    test_queries = [
        "exosomes",
        "exosomes neurology",
        "extracellular vesicles",
        "exosomes spinal",
        "exosomes central nervous system"
    ]
    
    print("=== Testing New ClinicalTrials.gov API v2 ===\n")
    
    for query in test_queries:
        print(f"Testing query: '{query}'")
        results = fetch_clinical_trials_v2(query, max_records=5)
        print(f"Found {len(results)} studies")
        
        if results:
            for i, study in enumerate(results[:2]):  # Show first 2
                print(f"  {i+1}. {study['nct_id']}: {study['title'][:60]}...")
                print(f"     Status: {study['status']}, Spinal Hit: {'YES' if study['spinal_hit'] else 'NO'}")
        
        print("-" * 60)
        print()

if __name__ == "__main__":
    test_new_api()

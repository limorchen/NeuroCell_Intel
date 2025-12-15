import pandas as pd
from pathlib import Path
from lxml import etree
from epo_ops import Client, models, middlewares
import epo_ops.exceptions as ops_exc
import time
import os

# Setup
DATA_DIR = Path("data")
CSV_FILE = DATA_DIR / "patents_cumulative.csv"

# Get API credentials
key = os.environ.get("EPO_OPS_KEY")
secret = os.environ.get("EPO_OPS_SECRET")

if not key or not secret:
    print("ERROR: EPO_OPS_KEY and EPO_OPS_SECRET environment variables must be set!")
    print("\nSet them with:")
    print('  $env:EPO_OPS_KEY="your_key"')
    print('  $env:EPO_OPS_SECRET="your_secret"')
    exit(1)

# Initialize client
client = Client(
    key=key,
    secret=secret,
    middlewares=[middlewares.Dogpile(), middlewares.Throttler()]
)

print("="*80)
print("BACKFILL SCRIPT: Links + Applicants + Inventors")
print("="*80)

# Load CSV
print("\nLoading patents database...")
df = pd.read_csv(CSV_FILE)
print(f"✓ Loaded {len(df)} patents")

def generate_link(country, number):
    """Generate office-specific patent link"""
    clean_number = str(number).lstrip('0')
    
    if country == 'WO':
        return f"https://patentscope.wipo.int/search/en/detail.jsf?docId=WO{clean_number}"
    elif country == 'EP':
        return f"https://worldwide.espacenet.com/patent/search?q=pn%3DEP{clean_number}"
    elif country == 'US':
        return f"https://patents.google.com/patent/US{clean_number}"
    elif country == 'CN':
        return f"https://patents.google.com/patent/CN{clean_number}"
    else:
        return f"https://worldwide.espacenet.com/patent/search?q=pn%3D{country}{clean_number}"

def get_applicants_inventors(country, number, kind):
    """Fetch applicants and inventors with robust XPath"""
    try:
        # Clean the number - remove leading zeros and handle special characters
        clean_number = str(number).strip()
        
        resp = client.published_data(
            reference_type="publication",
            input=models.Docdb(clean_number, country, kind),
            endpoint="biblio"
        )
        
        root = etree.fromstring(resp.content)
        ns = {"ex": "http://www.epo.org/exchange"}
        
        # Extract applicants - try multiple paths
        applicants = root.xpath("//ex:applicants/ex:applicant/ex:applicant-name/ex:name/text()", namespaces=ns)
        if not applicants:
            applicants = root.xpath("//*[local-name()='applicant-name']/*[local-name()='name']/text()")
        if not applicants:
            applicants = root.xpath("//ex:parties/ex:applicants/ex:applicant/ex:applicant-name/ex:name/text()", namespaces=ns)
        
        applicants = [a.strip() for a in applicants if a.strip()]
        applicants_str = ", ".join(applicants[:3]) if applicants else "Not available"
        
        # Extract inventors - try multiple paths
        inventors = root.xpath("//ex:inventors/ex:inventor/ex:inventor-name/ex:name/text()", namespaces=ns)
        if not inventors:
            inventors = root.xpath("//*[local-name()='inventor-name']/*[local-name()='name']/text()")
        if not inventors:
            inventors = root.xpath("//ex:parties/ex:inventors/ex:inventor/ex:inventor-name/ex:name/text()", namespaces=ns)
        
        inventors = [i.strip() for i in inventors if i.strip()]
        inventors_str = ", ".join(inventors[:3]) if inventors else "Not available"
        
        return applicants_str, inventors_str
        
    except Exception as e:
        # Return "Not available" instead of "Error" to avoid dtype warning
        error_msg = str(e)
        if "quote_from_bytes" in error_msg:
            return "Not available (format error)", "Not available (format error)"
        elif "404" in error_msg:
            return "Not available (404)", "Not available (404)"
        else:
            return "Not available (error)", "Not available (error)"
# Ensure columns exist with correct dtype
if 'link' not in df.columns:
    df['link'] = ''
if 'applicants' not in df.columns:
    df['applicants'] = ''
    df['applicants'] = df['applicants'].astype(str)  # Ensure string dtype
if 'inventors' not in df.columns:
    df['inventors'] = ''
    df['inventors'] = df['inventors'].astype(str)  # Ensure string dtype

# Count what needs updating
empty_links = df['link'].fillna('').eq('').sum()
empty_applicants = ((df['applicants'].fillna('') == '') | 
                    (df['applicants'] == 'Not available')).sum()
empty_inventors = ((df['inventors'].fillna('') == '') | 
                   (df['inventors'] == 'Not available')).sum()

print(f"\nFound:")
print(f"  - {empty_links} patents without links")
print(f"  - {empty_applicants} patents without applicants")
print(f"  - {empty_inventors} patents without inventors")

# Process each patent
print(f"\nProcessing {len(df)} patents...\n")

for idx, row in df.iterrows():
    needs_update = False
    patent_id = f"{row['country']}{row['publication_number']}{row['kind']}"
    
    # Check what needs updating
    needs_link = pd.isna(row.get('link')) or row.get('link') == ''
    
    needs_applicants = (pd.isna(row.get('applicants')) or 
                   row.get('applicants') == '' or 
                   row.get('applicants') == 'Not available' or
                   row.get('applicants') == 'Error')  # ADD THIS
    needs_inventors = (pd.isna(row.get('inventors')) or 
                  row.get('inventors') == '' or 
                  row.get('inventors') == 'Not available' or
                  row.get('inventors') == 'Error')  # ADD THIS
    
    if not (needs_link or needs_applicants or needs_inventors):
        print(f"{idx+1:2d}. [SKIP] {patent_id} - Already complete")
        continue
    
    print(f"{idx+1:2d}. [UPDATE] {patent_id}...", end=" ")
    
    # Generate link if needed
    if needs_link:
        link = generate_link(row['country'], row['publication_number'])
        df.at[idx, 'link'] = link
        print(f"Link ✓", end=" ")
    
    # Fetch applicants/inventors if needed
    if needs_applicants or needs_inventors:
        applicants, inventors = get_applicants_inventors(
            row['country'],
            row['publication_number'],
            row['kind']
        )
        
        if needs_applicants:
            df.at[idx, 'applicants'] = applicants
            print(f"App ✓", end=" ")
        
        if needs_inventors:
            df.at[idx, 'inventors'] = inventors
            print(f"Inv ✓", end=" ")
        
        # Rate limiting
        time.sleep(0.5)
    
    print(f"→ {applicants[:40] if needs_applicants or needs_inventors else 'Done'}")

# Save updated CSV
df.to_csv(CSV_FILE, index=False)

print("\n" + "="*80)
print(f"✓ Successfully updated {len(df)} patents!")
print(f"✓ Saved to {CSV_FILE}")
print("="*80)

# Summary
final_empty_links = df['link'].fillna('').eq('').sum()
final_empty_applicants = df['applicants'].fillna('').eq('').sum()
final_empty_inventors = df['inventors'].fillna('').eq('').sum()

print(f"\nFinal status:")
print(f"  - Patents with links: {len(df) - final_empty_links}/{len(df)}")
print(f"  - Patents with applicants: {len(df) - final_empty_applicants}/{len(df)}")
print(f"  - Patents with inventors: {len(df) - final_empty_inventors}/{len(df)}")
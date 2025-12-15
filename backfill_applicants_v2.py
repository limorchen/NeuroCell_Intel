import pandas as pd
from pathlib import Path
from lxml import etree
from epo_ops import Client, models, middlewares
import time
import os

# ---------------------------
# CONFIG
# ---------------------------
DATA_DIR = Path("data")
CSV_FILE = DATA_DIR / "patents_cumulative.csv"

SLEEP_SECONDS = 0.6

# ---------------------------
# OPS AUTH
# ---------------------------
key = os.environ.get("EPO_OPS_KEY")
secret = os.environ.get("EPO_OPS_SECRET")

if not key or not secret:
    raise RuntimeError("Set EPO_OPS_KEY and EPO_OPS_SECRET")

client = Client(
    key=key,
    secret=secret,
    middlewares=[middlewares.Dogpile(), middlewares.Throttler()]
)

# ---------------------------
# CORE FUNCTION
# ---------------------------
def get_applicants_inventors(country, number, kind):
    """
    OPS-maximal retrieval of applicants and inventors.
    Returns:
        applicants, inventors, party_source, party_status
    """

    def extract_parties(root):
        apps = root.xpath(
            "//*[local-name()='applicant-name']/*[local-name()='name']/text()"
        )
        invs = root.xpath(
            "//*[local-name()='inventor-name']/*[local-name()='name']/text()"
        )
        return (
            [a.strip() for a in apps if a.strip()],
            [i.strip() for i in invs if i.strip()],
        )

    def format_parties(apps, invs):
        return (
            ", ".join(apps[:3]) if apps else "Not available",
            ", ".join(invs[:3]) if invs else "Not available",
        )

    pub_epodoc = f"{country}{number}"

    # ------------------------------------------------
    # 1️⃣ EP published-data
    # ------------------------------------------------
    if country == "EP":
        try:
            resp = client.published_data(
                reference_type="publication",
                input=models.Epodoc(pub_epodoc),
                endpoint="biblio",
            )
            root = etree.fromstring(resp.content)
            apps, invs = extract_parties(root)

            if apps or invs:
                return (*format_parties(apps, invs),
                        "EP-published",
                        "success")
        except Exception:
            pass

    # ------------------------------------------------
    # 2️⃣ Publication → application
    # ------------------------------------------------
    try:
        resp = client.published_data(
            reference_type="publication",
            input=models.Epodoc(pub_epodoc),
            endpoint="biblio",
        )
        root = etree.fromstring(resp.content)

        app_ref = root.xpath(
            "//*[local-name()='application-reference']"
            "/*[local-name()='document-id']"
            "[*[local-name()='country'] and *[local-name()='doc-number']]"
        )

        if not app_ref:
            return "Not available", "Not available", "none", "no_application_ref"

        app_country = app_ref[0].xpath("*[local-name()='country']/text()")[0]
        app_number = app_ref[0].xpath("*[local-name()='doc-number']/text()")[0]
        app_epodoc = f"{app_country}{app_number}"

    except Exception:
        return "Not available", "Not available", "none", "publication_lookup_failed"

    # ------------------------------------------------
    # 3️⃣ Register (direct)
    # ------------------------------------------------
    try:
        reg = client.register(
            reference_type="application",
            input=models.Epodoc(app_epodoc),
            endpoint="biblio",
        )
        reg_root = etree.fromstring(reg.content)
        apps, invs = extract_parties(reg_root)

        if apps or invs:
            return (*format_parties(apps, invs),
                    "register-direct",
                    "success")
    except Exception:
        pass

    # ------------------------------------------------
    # 4️⃣ INPADOC family → EP member
    # ------------------------------------------------
    try:
        fam = client.published_data(
            reference_type="publication",
            input=models.Epodoc(pub_epodoc),
            endpoint="family",
        )
        fam_root = etree.fromstring(fam.content)

        ep_members = fam_root.xpath(
            "//*[local-name()='publication-reference']"
            "[*[local-name()='country']='EP']"
            "/*[local-name()='document-id']"
            "[*[local-name()='doc-number']]"
        )

        if not ep_members:
            return "Not available", "Not available", "family", "no_ep_family"

        ep_number = ep_members[0].xpath(
            "*[local-name()='doc-number']/text()"
        )[0]

        reg = client.register(
            reference_type="application",
            input=models.Epodoc(f"EP{ep_number}"),
            endpoint="biblio",
        )
        reg_root = etree.fromstring(reg.content)
        apps, invs = extract_parties(reg_root)

        if apps or invs:
            return (*format_parties(apps, invs),
                    "EP-family-register",
                    "success")

        return "Not available", "Not available", "EP-family-register", "early_stage"

    except Exception:
        return "Not available", "Not available", "family", "family_lookup_failed"


# ---------------------------
# PIPELINE
# ---------------------------
df = pd.read_csv(CSV_FILE)

for col in ["party_source", "party_status"]:
    if col not in df.columns:
        df[col] = ""

needs_update = df[
    (df["applicants"].str.contains("Not available", na=True)) |
    (df["inventors"].str.contains("Not available", na=True))
]

print(f"Loaded {len(df)} patents")
print(f"Updating {len(needs_update)} patents\n")

for i, (idx, row) in enumerate(needs_update.iterrows(), start=1):
    patent_id = f"{row['country']}{row['publication_number']}{row['kind']}"
    print(f"{i}. {patent_id} ...", end=" ")

    apps, invs, source, status = get_applicants_inventors(
        row["country"],
        row["publication_number"],
        row["kind"],
    )

    df.at[idx, "applicants"] = apps
    df.at[idx, "inventors"] = invs
    df.at[idx, "party_source"] = source
    df.at[idx, "party_status"] = status

    print(f"{status} ({source})")
    time.sleep(SLEEP_SECONDS)

df.to_csv(CSV_FILE, index=False)
print("\n✓ Done")

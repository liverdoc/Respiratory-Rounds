import os
import sqlite3
import xml.etree.ElementTree as ET
import time
import random
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

import requests
import streamlit as st
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# ==========================================================
# CONFIG
# ==========================================================

st.set_page_config(
    page_title="Respiratory Intelligence Engine",
    page_icon="🫁",
    layout="wide"
)

DB_PATH = "respiratory_intelligence.db"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# ==========================================================
# SECRETS
# ==========================================================

def get_secret(key: str) -> str:
    if key in st.secrets:
        return st.secrets[key]
    return os.getenv(key, "")

GEMINI_API_KEY = get_secret("GEMINI_API_KEY")
NCBI_API_KEY = get_secret("NCBI_API_KEY")

# ==========================================================
# MODELS
# ==========================================================

class RawPaper(BaseModel):
    pmid: str
    doi: Optional[str] = None
    title: str
    abstract: str
    journal: str
    publication_date: str

class PaperAnalysis(BaseModel):
    title: str
    journal: str
    publication_date: str

    relevance_score: int
    study_quality_score: int
    clinical_impact_score: int

    specialty_tags: List[str]

    key_findings: List[str]
    limitations: List[str]

    change_on_monday: bool
    bottom_line: str

    confidence_score: int

# ==========================================================
# DATABASE
# ==========================================================

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():

    with get_db_connection() as conn:

        conn.execute("""
        CREATE TABLE IF NOT EXISTS papers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pmid TEXT UNIQUE,
            doi TEXT,
            title TEXT,
            journal TEXT,
            publication_date TEXT,
            analysis_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)

        conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_pmid
        ON papers(pmid)
        """)

        conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_created
        ON papers(created_at)
        """)

        conn.commit()

def paper_exists(pmid: str) -> bool:

    with get_db_connection() as conn:

        result = conn.execute(
            "SELECT 1 FROM papers WHERE pmid=?",
            (pmid,)
        ).fetchone()

    return result is not None

def save_paper(
    pmid,
    doi,
    title,
    journal,
    pub_date,
    analysis
):

    try:

        with get_db_connection() as conn:

            conn.execute(
                """
                INSERT INTO papers
                (
                    pmid,
                    doi,
                    title,
                    journal,
                    publication_date,
                    analysis_json
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    pmid,
                    doi or "",
                    title,
                    journal,
                    pub_date,
                    analysis.model_dump_json()
                )
            )

            conn.commit()

        return True

    except sqlite3.IntegrityError:
        return False

def get_recent_papers(days=7):

    with get_db_connection() as conn:

        rows = conn.execute(
            """
            SELECT *
            FROM papers
            WHERE created_at >= datetime('now', ?)
            ORDER BY created_at DESC
            """,
            (f"-{days} days",)
        ).fetchall()

    return [dict(x) for x in rows]

# ==========================================================
# PUBMED
# ==========================================================

class PubMedClient:

    def __init__(self):

        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        self.session = requests.Session()
        self.api_key = NCBI_API_KEY

    def _build_url(self, endpoint, params):

        if self.api_key:
            params["api_key"] = self.api_key

        query = "&".join(
            f"{k}={v}" for k, v in params.items()
        )

        return f"{self.base_url}{endpoint}?{query}"

    def search_recent_papers(self, days_back=3):

        journals = [
            "The Lancet Respiratory Medicine",
            "American Journal of Respiratory and Critical Care Medicine",
            "European Respiratory Journal",
            "Thorax",
            "CHEST",
            "Journal of Thoracic Oncology",
            "Respirology",
            "Respiratory Medicine",
            "Lung Cancer",
            "ERJ Open Research",
            "New England Journal of Medicine",
            "JAMA",
            "The Lancet"
        ]

        topics = [
            "Pulmonary Hypertension",
            "ILD",
            "Interstitial Lung Disease",
            "Asthma",
            "COPD",
            "Lung Cancer",
            "Bronchoscopy",
            "EBUS",
            "Endobronchial Ultrasound"
        ]

        journal_query = " OR ".join(
            [f'"{j}"[Journal]' for j in journals]
        )

        topic_query = " OR ".join(
            [f'"{t}"[Title/Abstract]' for t in topics]
        )

        query = f"({journal_query}) AND ({topic_query})"

        date_threshold = (
            datetime.now()
            - timedelta(days=days_back)
        ).strftime("%Y/%m/%d")

        params = {
            "db": "pubmed",
            "term": query,
            "mindate": date_threshold,
            "retmax": "30",
            "retmode": "json"
        }

        try:

            res = self.session.get(
                self._build_url(
                    "esearch.fcgi",
                    params
                ),
                timeout=15
            )

            res.raise_for_status()

            return res.json()["esearchresult"]["idlist"]

        except Exception as e:

            st.error(f"PubMed search error: {e}")
            return []

    def fetch_paper_details(self, pmids):

        if not pmids:
            return []

        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml"
        }

        try:

            res = self.session.get(
                self._build_url(
                    "efetch.fcgi",
                    params
                ),
                timeout=20
            )

            res.raise_for_status()

            return self._parse_xml(res.content)

        except Exception as e:

            st.error(f"PubMed fetch error: {e}")
            return []

    def fetch_paper_by_pmid(self, pmid):

        papers = self.fetch_paper_details([pmid])

        if papers:
            return papers[0]

        return None

    def fetch_paper_by_doi(self, doi):

        params = {
            "db": "pubmed",
            "term": f"{doi}[LID]",
            "retmode": "json"
        }

        try:

            res = self.session.get(
                self._build_url(
                    "esearch.fcgi",
                    params
                ),
                timeout=10
            )

            res.raise_for_status()

            ids = res.json()["esearchresult"]["idlist"]

            if ids:
                return self.fetch_paper_by_pmid(ids[0])

            return None

        except:
            return None

    def _parse_xml(self, xml_content):

        papers = []

        try:

            root = ET.fromstring(xml_content)

            for article in root.findall(".//PubmedArticle"):

                pmid = article.find(".//PMID").text

                title_el = article.find(".//ArticleTitle")
                title = (
                    "".join(title_el.itertext()).strip()
                    if title_el is not None
                    else "No title"
                )

                abstract_elements = article.findall(".//AbstractText")

                abstract = " ".join(
                    "".join(x.itertext()).strip()
                    for x in abstract_elements
                )

                journal_el = article.find(".//Journal/Title")

                journal = (
                    journal_el.text.strip()
                    if journal_el is not None
                    else "Unknown"
                )

                doi = None

                for el in article.findall(".//ArticleId"):

                    if el.get("IdType") == "doi":
                        doi = el.text
                        break

                pub_date = str(datetime.now().year)

                year_el = article.find(".//PubDate/Year")

                if year_el is not None:
                    pub_date = year_el.text

                papers.append(
                    RawPaper(
                        pmid=pmid,
                        doi=doi,
                        title=title,
                        abstract=abstract,
                        journal=journal,
                        publication_date=pub_date
                    )
                )

        except Exception:
            pass

        return papers

# ==========================================================
# GEMINI
# ==========================================================

class IntelligenceEngine:

    def __init__(self):

        if not GEMINI_API_KEY:

            st.error("Missing GEMINI_API_KEY")
            st.stop()

        self.client = genai.Client(
            api_key=GEMINI_API_KEY
        )

        self.last_call = 0

    def _rate_limit(self):

        minimum_interval = 6

        elapsed = time.time() - self.last_call

        if elapsed < minimum_interval:
            time.sleep(minimum_interval - elapsed)

        self.last_call = time.time()

    def analyze_paper(self, paper):

        self._rate_limit()

        system_prompt = """
You are a consultant respiratory physician.

Analyse the paper.

Be evidence-focused.
Be critical.
Score quality honestly.

Return structured JSON only.
"""

        content = f"""
Title: {paper.title}

Journal: {paper.journal}

Abstract:
{paper.abstract}
"""

        for attempt in range(5):

            try:

                response = self.client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=content,
                    config=types.GenerateContentConfig(
                        temperature=0.1,
                        system_instruction=system_prompt,
                        response_schema=PaperAnalysis,
                        response_mime_type="application/json"
                    )
                )

                return PaperAnalysis.model_validate_json(
                    response.text
                )

            except Exception as e:

                error_text = str(e).lower()

                if (
                    "429" in error_text
                    or "503" in error_text
                ):

                    wait = min(
                        (2 ** attempt) * 10
                        + random.uniform(1, 5),
                        60
                    )

                    st.warning(
                        f"API limit reached. Waiting {wait:.0f}s..."
                    )

                    time.sleep(wait)

                else:

                    st.error(str(e))
                    return None

        return None

    def generate_weekly_algorithm(self, analyses):

        if not analyses:
            return "No papers available."

        summaries = []

        for a in analyses:

            summaries.append(
                f"""
Title: {a.title}
Bottom Line: {a.bottom_line}
Tags: {",".join(a.specialty_tags)}
"""
            )

        prompt = "\n".join(summaries)

        try:

            self._rate_limit()

            response = self.client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    system_instruction="""
Create a consultant-level respiratory
clinical algorithm.

Use markdown.

Include:
1. Practice changes
2. Diagnostic pathway
3. Therapy pathway
4. Caveats
5. Stepwise algorithm
"""
                )
            )

            return response.text

        except Exception as e:

            return str(e)

# ==========================================================
# CACHE
# ==========================================================

@st.cache_data(ttl=3600)
def cached_search(days_back):
    return PubMedClient().search_recent_papers(days_back)

@st.cache_data(ttl=3600)
def cached_fetch(pmids):
    return PubMedClient().fetch_paper_details(list(pmids))

# ==========================================================
# UI
# ==========================================================

def render_paper_card(analysis, pmid):

    with st.container(border=True):

        st.subheader(analysis.title)

        st.caption(
            f"{analysis.journal} | PMID {pmid}"
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Relevance",
            analysis.relevance_score
        )

        c2.metric(
            "Quality",
            analysis.study_quality_score
        )

        c3.metric(
            "Impact",
            analysis.clinical_impact_score
        )

        if analysis.change_on_monday:
            st.error("CHANGE ON MONDAY")
        else:
            st.info("No immediate practice change")

        st.write(
            f"**Bottom line:** {analysis.bottom_line}"
        )

        with st.expander("Details"):

            st.write("### Findings")

            for item in analysis.key_findings:
                st.write(f"- {item}")

            st.write("### Limitations")

            for item in analysis.limitations:
                st.write(f"- {item}")

            st.write(
                f"Tags: {', '.join(analysis.specialty_tags)}"
            )

# ==========================================================
# PAGES
# ==========================================================

def page_manual_review():

    st.title("📄 Manual Paper Review")

    pmid = st.text_input("PMID")

    doi = st.text_input("DOI")

    if st.button("Analyse Paper"):

        pubmed = PubMedClient()

        paper = None

        if pmid:
            paper = pubmed.fetch_paper_by_pmid(pmid)

        elif doi:
            paper = pubmed.fetch_paper_by_doi(doi)

        if not paper:

            st.error("Paper not found.")
            return

        engine = IntelligenceEngine()

        with st.spinner("Analysing..."):

            analysis = engine.analyze_paper(
                paper
            )

        if analysis:

            save_paper(
                paper.pmid,
                paper.doi,
                paper.title,
                paper.journal,
                paper.publication_date,
                analysis
            )

            render_paper_card(
                analysis,
                paper.pmid
            )

def page_daily_digest():

    st.title("📅 Daily Digest")

    if st.button(
        "Query Recent Respiratory Literature",
        use_container_width=True
    ):

        pmids = cached_search(3)

        if not pmids:

            st.warning("No papers found.")
            return

        papers = cached_fetch(tuple(pmids[:10]))

        engine = IntelligenceEngine()

        processed = 0

        for paper in papers:

            if processed >= 2:
                break

            if paper_exists(paper.pmid):
                continue

            analysis = engine.analyze_paper(
                paper
            )

            if not analysis:
                continue

            save_paper(
                paper.pmid,
                paper.doi,
                paper.title,
                paper.journal,
                paper.publication_date,
                analysis
            )

            processed += 1

        st.success(
            f"{processed} papers analysed."
        )

    rows = get_recent_papers(7)

    for row in rows:

        analysis = (
            PaperAnalysis.model_validate_json(
                row["analysis_json"]
            )
        )

        render_paper_card(
            analysis,
            row["pmid"]
        )

def page_weekly_algorithm():

    st.title("🧠 Weekly Algorithm")

    rows = get_recent_papers(7)

    analyses = [
        PaperAnalysis.model_validate_json(
            x["analysis_json"]
        )
        for x in rows
    ]

    if not analyses:
        st.info("No papers.")
        return

    if st.button(
        "Generate Algorithm",
        use_container_width=True
    ):

        engine = IntelligenceEngine()

        with st.spinner("Generating..."):

            report = (
                engine.generate_weekly_algorithm(
                    analyses
                )
            )

        st.markdown(report)

def page_stats():

    st.title("📊 Database")

    rows = get_recent_papers(3650)

    st.metric(
        "Total Papers",
        len(rows)
    )

# ==========================================================
# MAIN
# ==========================================================

def main():

    init_db()

    st.sidebar.title(
        "🫁 Respiratory Intelligence Engine"
    )

    page = st.sidebar.radio(
        "Navigation",
        [
            "Daily Digest",
            "Manual Paper Review",
            "Weekly Algorithm",
            "Database"
        ]
    )

    if page == "Daily Digest":
        page_daily_digest()

    elif page == "Manual Paper Review":
        page_manual_review()

    elif page == "Weekly Algorithm":
        page_weekly_algorithm()

    elif page == "Database":
        page_stats()

if __name__ == "__main__":
    main()
import os
import sqlite3
import xml.etree.ElementTree as ET
import time
from datetime import datetime, timedelta
from typing import List, Optional

import requests
import streamlit as st
from pydantic import BaseModel
from google import genai
from google.genai import types

# ==========================================================
# CONFIG & SECRETS
# ==========================================================

st.set_page_config(
    page_title="Respiratory Intelligence Engine",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

DB_PATH = "respiratory_intelligence.db"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

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
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
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
        conn.execute("CREATE INDEX IF NOT EXISTS idx_pmid ON papers(pmid)")
        conn.commit()

def paper_exists(pmid: str) -> bool:
    with get_db_connection() as conn:
        return conn.execute("SELECT 1 FROM papers WHERE pmid=?", (pmid,)).fetchone() is not None

def save_paper(paper: RawPaper, analysis: PaperAnalysis) -> bool:
    try:
        with get_db_connection() as conn:
            conn.execute(
                """
                INSERT INTO papers (pmid, doi, title, journal, publication_date, analysis_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (paper.pmid, paper.doi or "", paper.title, paper.journal, paper.publication_date, analysis.model_dump_json())
            )
            conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False

def get_recent_papers(days=7):
    with get_db_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM papers WHERE created_at >= datetime('now', ?) ORDER BY created_at DESC",
            (f"-{days} days",)
        ).fetchall()
    return [dict(x) for x in rows]

def delete_paper(pmid: str):
    with get_db_connection() as conn:
        conn.execute("DELETE FROM papers WHERE pmid=?", (pmid,))
        conn.commit()

# ==========================================================
# PUBMED CLIENT
# ==========================================================

class PubMedClient:
    def __init__(self):
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        self.session = requests.Session()
        self.api_key = NCBI_API_KEY

    def _build_url(self, endpoint, params):
        if self.api_key:
            params["api_key"] = self.api_key
        query = "&".join(f"{k}={v}" for k, v in params.items())
        return f"{self.base_url}{endpoint}?{query}"

    def search_recent_papers(self, days_back=3):
        journals = [
            "The Lancet Respiratory Medicine", "American Journal of Respiratory and Critical Care Medicine",
            "European Respiratory Journal", "Thorax", "CHEST", "Journal of Thoracic Oncology",
            "Respirology", "Respiratory Medicine", "Lung Cancer", "New England Journal of Medicine", "JAMA", "The Lancet"
        ]
        topics = [
            "Pulmonary Hypertension", "ILD", "Interstitial Lung Disease", "Asthma", "COPD",
            "Lung Cancer", "Bronchoscopy", "EBUS", "Pleural"
        ]
        
        journal_query = " OR ".join([f'"{j}"[Journal]' for j in journals])
        topic_query = " OR ".join([f'"{t}"[Title/Abstract]' for t in topics])
        query = f"({journal_query}) AND ({topic_query})"
        
        date_threshold = (datetime.now() - timedelta(days=days_back)).strftime("%Y/%m/%d")
        
        params = {
            "db": "pubmed", "term": query, "mindate": date_threshold, 
            "datetype": "edat", "retmax": "50", "retmode": "json"
        }
        
        try:
            res = self.session.get(self._build_url("esearch.fcgi", params), timeout=10)
            res.raise_for_status()
            return res.json().get("esearchresult", {}).get("idlist", [])
        except Exception as e:
            st.error(f"PubMed Search Error: {e}")
            return []

    def fetch_paper_details(self, pmids):
        if not pmids: return []
        params = {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"}
        
        try:
            res = self.session.get(self._build_url("efetch.fcgi", params), timeout=15)
            res.raise_for_status()
            return self._parse_xml(res.content)
        except Exception as e:
            st.error(f"PubMed Fetch Error: {e}")
            return []

    def _parse_xml(self, xml_content):
        papers = []
        try:
            root = ET.fromstring(xml_content)
            for article in root.findall(".//PubmedArticle"):
                pmid_el = article.find(".//PMID")
                pmid = pmid_el.text if pmid_el is not None else "Unknown"
                
                title_el = article.find(".//ArticleTitle")
                title = "".join(title_el.itertext()).strip() if title_el is not None else "No title"
                
                abstract_elements = article.findall(".//AbstractText")
                abstract = " ".join("".join(x.itertext()).strip() for x in abstract_elements) if abstract_elements else ""
                
                if len(abstract) < 50:
                    continue
                
                journal_el = article.find(".//Journal/Title")
                journal = journal_el.text if journal_el is not None else "Unknown"
                
                doi = next((el.text for el in article.findall(".//ArticleId") if el.get("IdType") == "doi"), None)
                
                year_el = article.find(".//PubDate/Year")
                pub_date = year_el.text if year_el is not None else str(datetime.now().year)
                
                papers.append(RawPaper(pmid=pmid, doi=doi, title=title, abstract=abstract, journal=journal, publication_date=pub_date))
        except Exception as e:
            st.error(f"XML Parsing Error: {e}")
        return papers

# ==========================================================
# GEMINI ENGINE
# ==========================================================

SYSTEM_PROMPT = """
You are Respiratory Intelligence Engine (RIE), an elite consultant-level respiratory medicine evidence analyst.
CRITICAL APPRAISAL RULES:
- Always actively look for flaws. Never assume positive results are true.
- Score Relevance, Quality, Impact, and Confidence strictly from 0-10.
- CHANGE ON MONDAY TEST: True ONLY IF meaningful patient benefit, credible methodology, and clinically actionable.
- BOTTOM LINE: Exactly ONE sentence (Population + Intervention + Key Finding + Clinical Meaning).
- KEY FINDINGS & LIMITATIONS: Exactly 3 concise bullets each. Include hard numbers (HR, OR, CI) if available.
- Be concise, sceptical, and avoid hype. Output valid JSON only.
"""

class IntelligenceEngine:
    def __init__(self):
        if not GEMINI_API_KEY:
            st.error("Missing GEMINI_API_KEY in secrets/env.")
            st.stop()
        self.client = genai.Client(api_key=GEMINI_API_KEY)

    def analyze_paper(self, paper: RawPaper, status_container) -> Optional[PaperAnalysis]:
        content = f"Title: {paper.title}\nJournal: {paper.journal}\nAbstract:\n{paper.abstract}"
        
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                response = self.client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=content,
                    config=types.GenerateContentConfig(
                        temperature=0.1,
                        system_instruction=SYSTEM_PROMPT,
                        response_schema=PaperAnalysis,
                        response_mime_type="application/json",
                        # CRITICAL FIX: Disable safety filters so medical terms (mortality, cancer, death) don't trigger a ClientError
                        safety_settings=[
                            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                        ]
                    )
                )
                return PaperAnalysis.model_validate_json(response.text)
            
            except Exception as e:
                error_msg = str(e).lower()
                
                if "429" in error_msg or "quota" in error_msg:
                    wait_time = 15 * (attempt + 1)
                    status_container.warning(f"API Rate Limit Hit. Pausing for {wait_time}s...")
                    time.sleep(wait_time)
                elif "400" in error_msg or "clienterror" in error_msg:
                    status_container.error(f"Gemini rejected the prompt (Likely a schema issue). Skipping paper. Error: {e}")
                    return None # Don't retry 400 errors, they will just fail again
                else:
                    status_container.error(f"Unexpected Error: {e}")
                    time.sleep(5)
                    
        return None

# ==========================================================
# UI COMPONENTS
# ==========================================================

def render_paper_card(analysis: PaperAnalysis, pmid: str, show_delete: bool = False):
    with st.container(border=True):
        st.markdown(f"### {analysis.title}")
        st.caption(f"**{analysis.journal}** | Published: {analysis.publication_date} | PMID: [{pmid}](https://pubmed.ncbi.nlm.nih.gov/{pmid}/)")
        
        if analysis.change_on_monday:
            st.error("🚨 **CHANGE ON MONDAY** - Immediate Practice Impact")
        
        st.info(f"**Bottom Line:** {analysis.bottom_line}")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Relevance", f"{analysis.relevance_score}/10")
        c2.metric("Quality", f"{analysis.study_quality_score}/10")
        c3.metric("Impact", f"{analysis.clinical_impact_score}/10")
        c4.metric("Confidence", f"{analysis.confidence_score}/10")

        with st.expander("View Detailed Appraisal"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 📈 Key Findings")
                for item in analysis.key_findings:
                    st.markdown(f"- {item}")
            with col2:
                st.markdown("#### ⚠️ Limitations")
                for item in analysis.limitations:
                    st.markdown(f"- {item}")
            
            st.markdown(f"**Tags:** `{', '.join(analysis.specialty_tags)}`")
            
            if show_delete:
                if st.button("Delete Record", key=f"del_{pmid}", type="primary"):
                    delete_paper(pmid)
                    st.rerun()

# ==========================================================
# PAGES
# ==========================================================

def page_daily_digest():
    st.title("📅 Daily Respiratory Digest")
    
    col1, col2 = st.columns([3, 1])
    with col2:
        max_papers = st.slider("Max papers to analyze", 1, 10, 3, help="Keep low to avoid API limits")
        days_back = st.number_input("Days to look back", 1, 14, 3)

    if st.button("🔍 Run Intelligence Engine", type="primary", use_container_width=True):
        pubmed = PubMedClient()
        engine = IntelligenceEngine()
        
        status_ui = st.empty()
        
        status_ui.info("Querying PubMed for recent literature...")
        pmids = pubmed.search_recent_papers(days_back)
        
        if not pmids:
            status_ui.warning("No new papers found in the specified timeframe.")
            return
            
        new_pmids = [p for p in pmids if not paper_exists(p)][:max_papers]
        
        if not new_pmids:
            status_ui.success(f"Found {len(pmids)} papers, but all are already in your database!")
            return
            
        status_ui.info(f"Fetching abstracts for {len(new_pmids)} new papers...")
        papers = pubmed.fetch_paper_details(new_pmids)
        
        if not papers:
            status_ui.error("Failed to extract abstracts from PubMed XML. Papers might be title-only.")
            return

        progress_bar = st.progress(0)
        processed = 0
        
        for i, paper in enumerate(papers):
            status_ui.info(f"🧠 Appraising ({i+1}/{len(papers)}): *{paper.title[:60]}...*")
            
            analysis = engine.analyze_paper(paper, status_ui)
            
            if analysis:
                save_paper(paper, analysis)
                processed += 1
                
            progress_bar.progress((i + 1) / len(papers))
            
            # Small baseline pause to respect free tier limits (15 RPM)
            if i < len(papers) - 1:
                time.sleep(5)
                
        status_ui.success(f"✅ Successfully appraised {processed} new papers!")
        time.sleep(2)
        st.rerun()

    st.divider()
    st.subheader("Recent Appraisals")
    rows = get_recent_papers(7)
    if not rows:
        st.info("No papers appraised in the last 7 days.")
    for row in rows:
        analysis = PaperAnalysis.model_validate_json(row["analysis_json"])
        render_paper_card(analysis, row["pmid"])

def page_manual_review():
    st.title("📄 Manual Paper Review")
    pmid = st.text_input("Enter PubMed ID (PMID)", placeholder="e.g., 38123456")
    
    if st.button("Appraise Paper", type="primary"):
        if not pmid:
            st.warning("Please enter a PMID.")
            return
        if paper_exists(pmid):
            st.info("This paper is already in the database.")
            return

        pubmed = PubMedClient()
        engine = IntelligenceEngine()
        status_ui = st.empty()
        
        status_ui.info("Fetching paper details...")
        papers = pubmed.fetch_paper_details([pmid])
        
        if not papers:
            status_ui.error("Paper not found or no abstract available.")
            return
            
        paper = papers[0]
        status_ui.info("Appraising literature...")
        
        analysis = engine.analyze_paper(paper, status_ui)
        if analysis:
            save_paper(paper, analysis)
            status_ui.success("Appraisal complete!")
            render_paper_card(analysis, paper.pmid)

def page_database():
    st.title("🗄️ Intelligence Database")
    rows = get_recent_papers(3650)
    
    st.metric("Total Appraised Papers", len(rows))
    search = st.text_input("Search titles or tags...")
    
    for row in rows:
        analysis = PaperAnalysis.model_validate_json(row["analysis_json"])
        if search.lower() in analysis.title.lower() or search.lower() in " ".join(analysis.specialty_tags).lower():
            render_paper_card(analysis, row["pmid"], show_delete=True)

# ==========================================================
# MAIN
# ==========================================================

def main():
    init_db()
    
    st.sidebar.title("🫁 R.I.E.")
    st.sidebar.caption("Respiratory Intelligence Engine")
    st.sidebar.divider()
    
    page = st.sidebar.radio(
        "Navigation",
        ["Daily Digest", "Manual Review", "Database"],
        label_visibility="collapsed"
    )
    
    if page == "Daily Digest":
        page_daily_digest()
    elif page == "Manual Review":
        page_manual_review()
    elif page == "Database":
        page_database()

if __name__ == "__main__":
    main()
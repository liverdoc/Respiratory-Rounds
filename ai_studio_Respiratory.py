import os
import sqlite3
import json
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

# ==========================================
# CONFIGURATION & SECRETS
# ==========================================

st.set_page_config(
    page_title="Respiratory Medicine Intelligence Engine",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

def get_secret(key: str) -> str:
    """Safely retrieve secrets from Streamlit secrets or environment variables."""
    if key in st.secrets:
        return st.secrets[key]
    return os.environ.get(key, "")

GEMINI_API_KEY = get_secret("GEMINI_API_KEY")
NCBI_API_KEY = get_secret("NCBI_API_KEY")

# ==========================================
# PYDANTIC MODELS (STRUCTURED OUTPUTS)
# ==========================================

class RawPaper(BaseModel):
    """Represents the raw data extracted from PubMed XML."""
    pmid: str
    doi: Optional[str] = None
    title: str
    abstract: str
    journal: str
    publication_date: str

class PaperAnalysis(BaseModel):
    """Strict structured output model for Gemini respiratory analysis."""
    title: str
    journal: str
    publication_date: str
    relevance_score: int = Field(description="Score 0-10 on relevance to specific respiratory target fields (PH, ILD, Asthma, COPD, Lung Cancer, Bronchoscopy, EBUS).")
    study_quality_score: int = Field(description="Score 0-10 evaluating methodological rigor and trial design validation.")
    clinical_impact_score: int = Field(description="Score 0-10 evaluating potential to change respiratory practice, guidelines, or diagnostics.")
    specialty_tags: List[str] = Field(description="List of relevant target tags (e.g., 'Pulmonary Hypertension', 'ILD', 'Asthma', 'COPD', 'Lung Cancer', 'Bronchoscopy', 'EBUS').")
    key_findings: List[str] = Field(description="2-3 bullet points detailing critical findings, survival data, diagnostic accuracy, or hazard ratios.")
    limitations: List[str] = Field(description="2-3 bullet points highlighting surrogate endpoints (e.g., FEV1 vs exacerbations), micro-samples, missing control arms, or unblinded biases.")
    change_on_monday: bool = Field(description="True if this study offers an actionable reason to alter prescribing, interventional diagnostic approach, or staging pipelines immediately.")
    bottom_line: str = Field(description="One concise sentence summarizing the core takeaway for a Consultant Pulmonologist.")
    confidence_score: int = Field(description="Score 0-10 on how confident the AI is in this assessment based on abstract detail.")

# ==========================================
# DATABASE LAYER (SQLite)
# ==========================================

DB_PATH = "respiratory_intelligence.db"

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
        conn.commit()

def save_paper(pmid: str, doi: str, title: str, journal: str, pub_date: str, analysis: PaperAnalysis) -> bool:
    try:
        with get_db_connection() as conn:
            conn.execute("""
                INSERT INTO papers (pmid, doi, title, journal, publication_date, analysis_json)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (pmid, doi or "", title, journal, pub_date, analysis.model_dump_json()))
            conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False 

def get_recent_papers(days: int = 7) -> List[Dict[str, Any]]:
    with get_db_connection() as conn:
        cursor = conn.execute("""
            SELECT * FROM papers 
            WHERE created_at >= datetime('now', ?)
            ORDER BY created_at DESC
        """, (f"-{days} days",))
        return [dict(row) for row in cursor.fetchall()]

def get_db_stats() -> Dict[str, Any]:
    with get_db_connection() as conn:
        total = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
        this_week = conn.execute("SELECT COUNT(*) FROM papers WHERE created_at >= datetime('now', '-7 days')").fetchone()[0]
        
        cursor = conn.execute("SELECT analysis_json FROM papers")
        analyses = [PaperAnalysis.model_validate_json(row[0]) for row in cursor.fetchall()]
        
    if not analyses:
        return {"total": total, "this_week": this_week, "avg_rel": 0, "avg_imp": 0, "journals": [], "topics": []}

    avg_rel = sum(a.relevance_score for a in analyses) / len(analyses)
    avg_imp = sum(a.clinical_impact_score for a in analyses) / len(analyses)
    
    journals = {}
    topics = {}
    for a in analyses:
        journals[a.journal] = journals.get(a.journal, 0) + 1
        for tag in a.specialty_tags:
            topics[tag] = topics.get(tag, 0) + 1
            
    top_journals = sorted(journals.items(), key=lambda x: x[1], reverse=True)[:5]
    top_topics = sorted(topics.items(), key=lambda x: x[1], reverse=True)[:5]

    return {
        "total": total,
        "this_week": this_week,
        "avg_rel": round(avg_rel, 1),
        "avg_imp": round(avg_imp, 1),
        "journals": top_journals,
        "topics": top_topics
    }

# ==========================================
# PUBMED CLIENT LAYER (TOP 20 JOURNALS)
# ==========================================

class PubMedClient:
    def __init__(self):
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        self.session = requests.Session()
        self.api_key = NCBI_API_KEY
        
    def _build_url(self, endpoint: str, params: Dict[str, str]) -> str:
        if self.api_key:
            params['api_key'] = self.api_key
        query_string = "&".join(f"{k}={v}" for k, v in params.items())
        return f"{self.base_url}{endpoint}?{query_string}"

    def search_recent_papers(self, days_back: int = 2) -> List[str]:
        """Searches PubMed using the top 20 structural journals + specific clinical criteria."""
        # Top 20 High Impact Respiratory, General Medical, and Thoracic Oncology Journals
        journals = [
            "The Lancet Respiratory Medicine", "American Journal of Respiratory and Critical Care Medicine",
            "European Respiratory Journal", "Journal of Thoracic Oncology", "Thorax", "CHEST",
            "European Respiratory Review", "Archivos de Bronconeumologia", "Pulmonology", "Respirology",
            "Journal of Heart and Lung Transplantation", "American Journal of Respiratory Cell and Molecular Biology",
            "npj Primary Care Respiratory Medicine", "Clinics in Chest Medicine", "Lung Cancer",
            "ERJ Open Research", "Respiration", "Translational Lung Cancer Research", "Respiratory Medicine",
            "New England Journal of Medicine", "JAMA", "The Lancet"
        ]
        
        # Expressed focus interests from prompt
        topics = [
            "Pulmonary Hypertension", "ILD", "Interstitial Lung Disease", 
            "Asthma", "COPD", "Lung Cancer", "Bronchoscopy", "EBUS", 
            "Endobronchial Ultrasound"
        ]
        
        journal_query = " OR ".join([f'"{j}"[Journal]' for j in journals])
        topic_query = " OR ".join([f'"{t}"[Title/Abstract]' for t in topics])
        query = f"({journal_query}) AND ({topic_query})"
        
        date_threshold = (datetime.now() - timedelta(days=days_back)).strftime("%Y/%m/%d")
        
        params = {
            "db": "pubmed",
            "term": query,
            "mindate": date_threshold,
            "retmode": "json",
            "retmax": "50"
        }
        
        try:
            res = self.session.get(self._build_url("esearch.fcgi", params), timeout=10)
            res.raise_for_status()
            return res.json().get("esearchresult", {}).get("idlist", [])
        except Exception as e:
            st.error(f"PubMed Search Error: {e}")
            return []

    def fetch_paper_details(self, pmids: List[str]) -> List[RawPaper]:
        if not pmids:
            return []
            
        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml"
        }
        
        try:
            res = self.session.get(self._build_url("efetch.fcgi", params), timeout=15)
            res.raise_for_status()
            return self._parse_xml(res.content)
        except Exception as e:
            st.error(f"PubMed Fetch Error: {e}")
            return []

    def fetch_paper_by_pmid(self, pmid: str) -> Optional[RawPaper]:
        papers = self.fetch_paper_details([pmid])
        return papers[0] if papers else None

    def fetch_paper_by_doi(self, doi: str) -> Optional[RawPaper]:
        params = {
            "db": "pubmed",
            "term": f"{doi}[LID]",
            "retmode": "json"
        }
        try:
            res = self.session.get(self._build_url("esearch.fcgi", params), timeout=10)
            res.raise_for_status()
            idlist = res.json().get("esearchresult", {}).get("idlist", [])
            if idlist:
                return self.fetch_paper_by_pmid(idlist[0])
            return None
        except Exception as e:
            st.error(f"PubMed DOI Search Error: {e}")
            return None

    def _parse_xml(self, xml_content: bytes) -> List[RawPaper]:
        papers = []
        try:
            root = ET.fromstring(xml_content)
            for article in root.findall(".//PubmedArticle"):
                pmid = article.find(".//PMID").text
                
                title_el = article.find(".//ArticleTitle")
                title = "".join(title_el.itertext()).strip() if title_el is not None else "No Title"
                
                abstract_elements = article.findall(".//AbstractText")
                abstract = " ".join(["".join(el.itertext()).strip() for el in abstract_elements])
                
                journal_el = article.find(".//Journal/Title")
                journal = journal_el.text.strip() if journal_el is not None else "Unknown Journal"
                
                doi = None
                for elid in article.findall(".//ArticleId"):
                    if elid.get("IdType") == "doi":
                        doi = elid.text
                        break
                        
                pub_date = datetime.now().strftime("%Y-%m-%d")
                date_el = article.find(".//PubDate/Year")
                if date_el is not None:
                    pub_date = date_el.text
                    
                papers.append(RawPaper(
                    pmid=pmid,
                    doi=doi,
                    title=title,
                    abstract=abstract,
                    journal=journal,
                    publication_date=pub_date
                ))
        except ET.ParseError as e:
            st.error(f"XML Parsing Error: {e}")
        return papers

# ==========================================
# GEMINI INTELLIGENCE LAYER (RESPIRATORY LENS)
# ==========================================

class IntelligenceEngine:
    def __init__(self):
        if not GEMINI_API_KEY:
            st.error("GEMINI_API_KEY is missing. Please configure your environment variables or secrets.")
            st.stop()
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        
    def analyze_paper(self, paper: RawPaper) -> Optional[PaperAnalysis]:
        """Appraises a respiratory paper under a strict clinical pulmonology lens."""
        if len(paper.abstract.strip()) < 50:
            st.warning(f"Skipped PMID {paper.pmid}: Abstract too short (<50 chars).")
            return None

        system_prompt = """You are an elite Consultant Respiratory Physician (Pulmonologist) and clinical trialist grading articles for an academic specialist panel.
        
CRITICAL CLINICAL LENS DIRECTIONS:
1. Prioritize these specific fields with extreme relevance weights: Pulmonary Hypertension (PAH/CTEPH pathways), Interstitial Lung Disease (ILD/IPF antifibrotics), Asthma (biologics, airway remodeling), COPD (triple therapies, exacerbation metrics), Lung Cancer (screening, staging, mutations), Bronchoscopy, and EBUS (lymph node staging metrics, diagnostic sensitivity).
2. Heavily scrutinize trials. Is look-back or abstract spin trying to cover up missed primary endpoints? Be highly skeptical if a study celebrates minor FEV1 changes but ignores raw clinical exacerbation metrics or overall survival numbers.
3. Critically analyze interventional diagnostics (Bronchoscopy/EBUS). Ensure you report sample size issues, rapid on-site evaluation (ROSE) biases, or operator dependency flaws.
"""
        user_content = f"Title: {paper.title}\nJournal: {paper.journal}\nDate: {paper.publication_date}\nAbstract: {paper.abstract}"

        max_retries = 5
        base_delay = 4.0  
        time.sleep(4.5)  # Enforce spacing to honor 15 RPM Free Tier ceiling

        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=user_content,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=0.0,
                        response_mime_type="application/json",
                        response_schema=PaperAnalysis,
                    ),
                )
                return PaperAnalysis.model_validate_json(response.text)

            except Exception as e:
                err_msg = str(e)
                if "429" in err_msg or "503" in err_msg or "RESOURCE_EXHAUSTED" in err_msg or "UNAVAILABLE" in err_msg:
                    if attempt == max_retries - 1:
                        st.error(f"Failed PMID {paper.pmid} after {max_retries} attempts due to API limits.")
                        return None
                    
                    sleep_time = (base_delay * (2 ** attempt)) + random.uniform(0, 1)
                    st.warning(f"⚠️ API limit hit (429/503). Retrying in {round(sleep_time, 1)}s...")
                    time.sleep(sleep_time)
                else:
                    st.error(f"Gemini API Error: {e}")
                    return None
        return None

    def generate_weekly_algorithm(self, analyses: List[PaperAnalysis]) -> str:
        """Generates a cohesive management or diagnostic protocol template from clinical papers."""
        if not analyses:
            return "No papers available to generate an algorithm."
            
    def generate_weekly_algorithm(self, analyses: List[PaperAnalysis]) -> str:
        """Generates a cohesive management or diagnostic protocol template from clinical papers."""
        if not analyses:
            return "No papers available to generate an algorithm."
            
        system_prompt = """You are a Chief Consultant Pulmonologist converting recent study evidence into a departmental protocol.
Focus structural planning on: Pulmonary Hypertension, ILD, Asthma, COPD, Lung Cancer staging, Bronchoscopy, and EBUS diagnostics.
Output in clean Markdown.
Include these exact structured headings:
1. Key Practice Modifications
2. Diagnostic & Interventional Pathways (Include Bronchoscopy/EBUS adjustments if present)
3. Prescribing/Therapy Pipeline Enhancements
4. Areas of Caution and Structural Trial Faults
5. Step-by-Step Clinical Decision Algorithm"""

        summaries = [f"- {a.title} ({a.journal}): {a.bottom_line}. Specialty Areas: {', '.join(a.specialty_tags)}. Change on Monday: {a.change_on_monday}" for a in analyses]
        user_content = "Recent High-Impact Respiratory Papers:\n" + "\n".join(summaries)

        for attempt in range(3):
            try:
                response = self.client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=user_content,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=0.2
                    )
                )
                return response.text
            except Exception as e:
                if "429" in str(e) or "503" in str(e):
                    time.sleep(6.0 * (attempt + 1))
                else:
                    st.error(f"Gemini API Error during synthesis: {e}")
                    return "Error generating protocol."
        return "Error generating algorithm due to API limits."
    # ==========================================
# UI COMPONENTS & PAGES
# ==========================================

def render_paper_card(analysis: PaperAnalysis, pmid: str):
    """Renders a mobile-friendly card for a processed paper."""
    with st.container(border=True):
        st.subheader(analysis.title)
        st.caption(f"**{analysis.journal}** | {analysis.publication_date} | PMID: {pmid}")
        
        # Metrics Row
        col1, col2, col3 = st.columns(3)
        col1.metric("Relevance", f"{analysis.relevance_score}/10")
        col2.metric("Study Quality", f"{analysis.study_quality_score}/10")
        col3.metric("Clinical Impact", f"{analysis.clinical_impact_score}/10")
        
        # Status Indicator for Practice Change
        if analysis.change_on_monday:
            st.error("🚨 **CHANGE ON MONDAY: YES**")
        else:
            st.info("ℹ️ **Change on Monday: No**")
            
        st.write(f"**Bottom Line:** {analysis.bottom_line}")
        
        # Expandable Details
        with st.expander("View Detailed Analysis"):
            st.write("**Key Findings:**")
            for kf in analysis.key_findings:
                st.write(f"- {kf}")
                
            st.write("**Limitations:**")
            for lim in analysis.limitations:
                st.write(f"- {lim}")
                
            st.write(f"**Specialty Tags:** {', '.join(analysis.specialty_tags)}")
            st.write(f"**AI Confidence Score:** {analysis.confidence_score}/10")

def page_daily_digest():
    st.title("📅 Respiratory Medicine Daily Digest")
    st.write("Screening 10 recent papers to find and analyze the top 2 most clinically relevant studies.")
    
    if st.button("Query Top 20 Journals", type="primary", use_container_width=True):
        with st.spinner("Executing optimized search strategy across top-tier respiratory platforms..."):
            pubmed = PubMedClient()
            # Pull a pool of up to 10 recent papers
            pmids = pubmed.search_recent_papers(days_back=3)
            
        if not pmids:
            st.warning("No new relevant respiratory papers matching your targeted topics were identified in the last 72 hours.")
            return
            
        # Ensure we look through a maximum of 10 papers to find our nuggets
        pmids_pool = pmids[:10]
        
        with st.spinner(f"Screening pool of {len(pmids_pool)} papers to find 2 highly relevant matches..."):
            raw_papers = pubmed.fetch_paper_details(pmids_pool)
            engine = IntelligenceEngine()
            
            processed_count = 0
            skipped_low_relevance = 0
            
            # Target terms to pre-screen keywords before burning Gemini API credits
            high_yield_keywords = ["hypertension", "pah", "cteph", "ild", "ipf", "fibrosis", "asthma", 
                                   "copd", "exacerbation", "cancer", "carcinoma", "nodule", "nsclc", 
                                   "bronchoscopy", "ebus", "ultrasound", "biopsy", "lymph"]

            for raw in raw_papers:
                # CRITICAL CAP: Stop the moment we successfully analyze 2 high-yield papers
                if processed_count >= 2:
                    break
                    
                with get_db_connection() as conn:
                    exists = conn.execute("SELECT 1 FROM papers WHERE pmid = ?", (raw.pmid,)).fetchone()
                if exists:
                    continue
                
                # PRE-SCREENING FILTER: Check if keywords exist in title or abstract
                combined_text = (raw.title + " " + raw.abstract).lower()
                has_keyword = any(keyword in combined_text for keyword in high_yield_keywords)
                
                if not has_keyword:
                    skipped_low_relevance += 1
                    continue # Skip this paper entirely without calling Gemini
                    
                # If it passes the filter, run the full Gemini clinical grading
                analysis = engine.analyze_paper(raw)
                if analysis:
                    # Double check if Gemini agreed it was relevant (Score 5 or higher)
                    if analysis.relevance_score >= 5:
                        save_paper(raw.pmid, raw.doi, raw.title, raw.journal, raw.publication_date, analysis)
                        processed_count += 1
                    else:
                        skipped_low_relevance += 1
                        
        if processed_count > 0:
            st.success(f"Execution successful! Screened the pool, filtered out {skipped_low_relevance} lower-yield targets, and analyzed the top {processed_count} papers.")
        else:
            st.info("Screened recent papers, but no brand-new high-yield matches found. The latest curated papers are already in your portfolio below!")
        
    st.divider()
    st.subheader("Clinical Evidence Portfolio")
    
    sort_option = st.selectbox("Rank dynamic list by:", ["Relevance Score", "Trial/Method Quality", "Clinical Impact"])
    recent_db_papers = get_recent_papers(days=3)
    
    if not recent_db_papers:
        st.info("No documents have been mapped to the database environment within the last 3 days.")
        return
        
    display_list = []
    for row in recent_db_papers:
        analysis = PaperAnalysis.model_validate_json(row["analysis_json"])
        display_list.append((row["pmid"], analysis))
        
    if sort_option == "Relevance Score":
        display_list.sort(key=lambda x: x[1].relevance_score, reverse=True)
    elif sort_option == "Trial/Method Quality":
        display_list.sort(key=lambda x: x[1].study_quality_score, reverse=True)
    else:
        display_list.sort(key=lambda x: x[1].clinical_impact_score, reverse=True)
        
    for pmid, analysis in display_list:
        render_paper_card(analysis, pmid)

def page_weekly_algorithm():
    st.title("🧠 Weekly Algorithm")
    st.write("Synthesize the last 7 days of literature into a clinical protocol.")
    
    recent_papers = get_recent_papers(days=7)
    analyses = [PaperAnalysis.model_validate_json(row["analysis_json"]) for row in recent_papers]
    
    if not analyses:
        st.warning("No papers processed in the last 7 days.")
        return
        
    # Display Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Papers Analyzed", len(analyses))
    col2.metric("Avg Relevance", round(sum(a.relevance_score for a in analyses) / len(analyses), 1))
    col3.metric("Avg Quality", round(sum(a.study_quality_score for a in analyses) / len(analyses), 1))
    col4.metric("Avg Impact", round(sum(a.clinical_impact_score for a in analyses) / len(analyses), 1))
    
    if st.button("Generate Weekly Clinical Algorithm", type="primary", use_container_width=True):
        with st.spinner("Synthesizing consensus algorithm..."):
            engine = IntelligenceEngine()
            markdown_report = engine.generate_weekly_algorithm(analyses)
            
        st.divider()
        st.markdown(markdown_report)

def page_db_stats():
    st.title("📊 Database Stats")
    
    stats = get_db_stats()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Papers Stored", stats["total"])
        st.metric("Average Relevance", stats["avg_rel"])
    with col2:
        st.metric("Papers This Week", stats["this_week"])
        st.metric("Average Impact", stats["avg_imp"])
        
    st.divider()
    
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Top Journals")
        for journal, count in stats["journals"]:
            st.write(f"- **{journal}**: {count}")
            
    with col4:
        st.subheader("Top Topics")
        for topic, count in stats["topics"]:
            st.write(f"- **{topic}**: {count}")

# ==========================================
# MAIN APPLICATION ROUTING
# ==========================================

def main():
    init_db()
    
    st.sidebar.title("🫀 Anaesthesia Engine")
    page = st.sidebar.radio("Navigation", [
        "Daily Digest", 
        "Add a Paper", 
        "Weekly Algorithm", 
        "Database Stats"
    ])
    
    if page == "Daily Digest":
        page_daily_digest()
    elif page == "Add a Paper":
        page_manual_review()
    elif page == "Weekly Algorithm":
        page_weekly_algorithm()
    elif page == "Database Stats":
        page_db_stats()

if __name__ == "__main__":
    main()
import json
import streamlit as st
import requests
import numpy as np
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq
import os
from pathlib import Path
from typing import List, Dict, Any, Union
from dotenv import load_dotenv
import pandas as pd
import httpx
import uuid
import re

# Load environment variables first
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Constants
MIN_RESULTS = 1
MAX_RESULTS = 10
MODEL_NAME = "all-MiniLM-L6-v2"
DB_PATH = "./chroma_db"
JSON_PATH = "./dummy_dataset.json"

# Set up sentence transformer
sentence_transformer = SentenceTransformer(MODEL_NAME)

# Set up embedding function for ChromaDB
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=MODEL_NAME)

# Initialize Groq client with httpx workaround
custom_client = httpx.Client()  # default config, no 'proxies'
groq_client = Groq(api_key=GROQ_API_KEY, http_client=custom_client)

# LLM model name
LLAMA3_MODEL = "llama-3.3-70b-versatile"

# Add skills dictionary at the top of the file after imports
SKILLS_DICTIONARY = {
    # Programming Languages
    'python': ['python', 'django', 'flask', 'numpy', 'pandas', 'scikit-learn', 'tensorflow', 'pytorch', 'fastapi'],
    'javascript': ['javascript', 'js', 'node.js', 'react', 'angular', 'vue', 'typescript', 'express', 'next.js'],
    'java': ['java', 'spring', 'hibernate', 'jsp', 'servlet', 'maven', 'gradle'],
    'sql': ['sql', 'mysql', 'postgresql', 'oracle', 'sql server', 'nosql', 'mongodb', 'redis'],
    'html': ['html', 'html5', 'css', 'css3', 'bootstrap', 'tailwind', 'sass', 'less'],
    
    # Data Science & ML
    'data_science': ['data science', 'machine learning', 'deep learning', 'ai', 'artificial intelligence', 'nlp', 'natural language processing'],
    'analytics': ['data analytics', 'business intelligence', 'bi', 'tableau', 'power bi', 'looker', 'data visualization'],
    
    # Cloud & DevOps
    'cloud': ['aws', 'azure', 'gcp', 'cloud', 'docker', 'kubernetes', 'k8s', 'terraform', 'jenkins', 'ci/cd'],
    'devops': ['devops', 'git', 'github', 'gitlab', 'bitbucket', 'jira', 'confluence', 'ansible', 'puppet', 'chef'],
    
    # Testing & QA
    'testing': ['testing', 'qa', 'quality assurance', 'selenium', 'junit', 'testng', 'cypress', 'jest', 'mocha'],
    
    # Soft Skills
    'soft_skills': ['communication', 'leadership', 'teamwork', 'problem solving', 'critical thinking', 'collaboration', 'time management'],
    
    # Business Skills
    'business': ['project management', 'agile', 'scrum', 'kanban', 'business analysis', 'product management', 'stakeholder management']
}

def convert_metadata_values(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Convert list values in metadata to comma-separated strings"""
    converted = {}
    for key, value in metadata.items():
        if isinstance(value, list):
            converted[key] = ", ".join(str(v) for v in value)
        else:
            converted[key] = value
    return converted

# Create ChromaDB collection
def setup_chroma_db():
    """Initialize and return the ChromaDB client and collection"""
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(
        name="shl_assessments",
        embedding_function=embedding_function
    )
    return client, collection

def load_assessments() -> List[Dict]:
    """Load assessments from JSON file"""
    try:
        with open(JSON_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        st.error(f"Assessment data file not found: {JSON_PATH}")
        return []
    except json.JSONDecodeError:
        st.error(f"Error decoding JSON from {JSON_PATH}")
        return []

def extract_text_from_url(url: str) -> str:
    """Extract text content from a URL"""
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
            
        # Get text
        text = soup.get_text()
        
        # Break into lines and remove leading and trailing space
        lines = (line.strip() for line in text.splitlines())
        
        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        
        # Remove blank lines
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        # Extract key information using regex patterns
        patterns = [
            r'requirements:.*?(?=\n\n|\Z)',
            r'qualifications:.*?(?=\n\n|\Z)',
            r'skills:.*?(?=\n\n|\Z)',
            r'responsibilities:.*?(?=\n\n|\Z)',
            r'job description:.*?(?=\n\n|\Z)',
            r'about the role:.*?(?=\n\n|\Z)',
            r"what you'll do:.*?(?=\n\n|\Z)",
            r"what you need:.*?(?=\n\n|\Z)"
        ]
        
        extracted_text = []
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                extracted_text.append(match.group())
        
        # If no patterns found, use the full text
        if not extracted_text:
            extracted_text = [text]
        
        return ' '.join(extracted_text)
    except Exception as e:
        st.error(f"Error extracting text from URL: {str(e)}")
        return ""

def preprocess_query(query: str) -> str:
    """Preprocess the query to improve matching"""
    # Convert to lowercase
    query = query.lower()
    
    # Extract skills from the query
    extracted_skills = []
    for category, skills in SKILLS_DICTIONARY.items():
        for skill in skills:
            if skill.lower() in query:
                extracted_skills.extend(skills)
    
    # Add extracted skills to the query
    if extracted_skills:
        query = f"{query} {' '.join(extracted_skills)}"
    
    return query

def query_vector_db(query: str, collection, n_results: int = MAX_RESULTS) -> List[Dict]:
    """Query the vector database for similar assessments"""
    # Preprocess the query
    processed_query = preprocess_query(query)
    
    results = collection.query(
        query_texts=[processed_query],
        n_results=n_results,
        include=['metadatas', 'distances']
    )
    
    # Combine metadata with similarity scores
    combined_results = []
    for metadata, distance in zip(results['metadatas'][0], results['distances'][0]):
        # Convert distance to similarity score (0-100%)
        # Using a sigmoid function to better distribute scores
        similarity_score = 100 * (1 / (1 + np.exp(distance * 2)))
        metadata['similarity_score'] = round(similarity_score, 2)
        combined_results.append(metadata)
    
    # Sort by similarity score in descending order
    combined_results.sort(key=lambda x: x['similarity_score'], reverse=True)
    return combined_results

def rerank_with_llm(query: str, results: List[Dict], top_k: int = MAX_RESULTS) -> List[Dict]:
    """Rerank results using LLM and add reasoning"""
    if not results:
        return []
    
    # Format the prompt
    prompt = f"""You are an expert in assessment and job matching. Analyze the following job description/query and assessment recommendations.
    
Job Description/Query: {query}

Recommended Assessments:
"""
    
    for i, result in enumerate(results):
        prompt += f"""
Assessment {i+1}: {result['name']}
Description: {result.get('Description', 'N/A')}
Job Levels: {result.get('Job levels', 'N/A')}
Category: {result.get('category', 'N/A')}
"""
    
    prompt += """
For each assessment, explain why it's a good match based on the job requirements and assessment details. Focus on key matching points and potential benefits.
"""
    
    # Get LLM analysis
    try:
        completion = groq_client.chat.completions.create(
            model=LLAMA3_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert in assessment and job matching."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=2000
        )
        
        analysis = completion.choices[0].message.content
        
        # Add analysis to each result
        for result in results:
            result['llm_analysis'] = analysis
            
        return results
    except Exception as e:
        st.error(f"Error getting LLM analysis: {str(e)}")
        # Return results without analysis
        for result in results:
            result['llm_analysis'] = "Analysis not available"
        return results

def format_results_for_display(results: List[Dict]) -> pd.DataFrame:
    """Format results for display in Streamlit"""
    if not results:
        return pd.DataFrame()
    
    # Create a list to store formatted results
    formatted_results = []
    
    for result in results:
        formatted_result = {
            'Assessment Name': f'<a href="{result["url"]}" target="_blank">{result["name"]}</a>',
            'Relevance Score': f"{result['similarity_score']}%",
            'Description': result.get('Description', 'N/A'),
            'Remote Testing': result.get('remote_testing', 'No'),
            'Adaptive/IRT': result.get('adaptive_irt', 'No'),
            'Duration': result.get('Assessment length', 'N/A'),
            'Test Type': result.get('Test Type', 'N/A'),
            'Languages': result.get('Languages', 'N/A'),
            'Job Levels': result.get('Job levels', 'N/A')
        }
        formatted_results.append(formatted_result)
    
    # Convert to DataFrame
    df = pd.DataFrame(formatted_results)
    return df

def create_rich_text_representation(item: Dict) -> str:
    """Create a rich text representation for better matching with weighted fields"""
    # Define base weights and multipliers for different field types
    base_weights = {
        'primary': 8,    # For most important fields (name, description)
        'secondary': 5,  # For important fields (skills, test type)
        'tertiary': 3,   # For supporting fields (languages, duration)
        'auxiliary': 2   # For additional fields (support features)
    }
    
    # Define field categories
    field_categories = {
        'primary': ['name', 'Description'],
        'secondary': ['Skills', 'Test Type', 'Job levels'],
        'tertiary': ['Languages', 'Assessment length', 'category'],
        'auxiliary': ['Remote Testing Support', 'Adaptive/IRT Support']
    }
    
    # Create weighted text representation
    text_parts = []
    
    # Process each category of fields
    for category, weight in base_weights.items():
        for field in field_categories[category]:
            if field in item:
                value = item[field]
                
                # Handle different types of values
                if isinstance(value, list):
                    # For list values, join them and add weight
                    text = ' '.join(str(v) for v in value)
                    # Add individual items with higher weight for better matching
                    text_parts.extend([text] * weight)
                    
                    # Add related skills from dictionary
                    for v in value:
                        v_lower = str(v).lower()
                        # Add the original value
                        text_parts.extend([str(v)] * (weight // 2))
                        # Add related skills
                        for skill_category, related_skills in SKILLS_DICTIONARY.items():
                            if any(skill in v_lower for skill in related_skills):
                                text_parts.extend(related_skills * (weight // 4))
                
                elif isinstance(value, bool):
                    # For boolean values, add descriptive text
                    if value:
                        text_parts.extend([f"{field} supported"] * weight)
                
                elif isinstance(value, str):
                    # For string values, handle special cases
                    if field == 'Assessment length' and '=' in value:
                        # Extract duration number
                        duration = value.split('=')[-1].strip()
                        text_parts.extend([f"duration {duration} minutes"] * weight)
                    else:
                        # Add the full text
                        text_parts.extend([value] * weight)
                        # Add individual words for better matching
                        words = value.split()
                        text_parts.extend(words * (weight // 2))
                        
                        # Add related skills from dictionary
                        value_lower = value.lower()
                        for skill_category, related_skills in SKILLS_DICTIONARY.items():
                            if any(skill in value_lower for skill in related_skills):
                                text_parts.extend(related_skills * (weight // 4))
    
    return ' '.join(text_parts)

def format_test_type(test_type: str) -> str:
    """Convert test type codes to full names"""
    test_type_map = {
        'A': 'Ability & Aptitude',
        'B': 'Biodata & Situational Judgement',
        'C': 'Competencies',
        'D': 'Development & 360',
        'E': 'Assessment Exercises',
        'K': 'Knowledge & Skills',
        'P': 'Personality & Behavior',
        'S': 'Simulations'
    }
    if isinstance(test_type, list):
        return ', '.join(test_type_map.get(t, t) for t in test_type)
    return test_type_map.get(test_type, test_type)

def main():
    st.set_page_config(
        page_title="SHL Assessment Recommender",
        page_icon="🎯",
        layout="wide"
    )
    
    st.title("🎯 SHL Assessment Recommender")
    st.markdown("""
    This system helps you find the most relevant SHL assessments based on your requirements.
    You can either:
    - Enter a natural language description of the role and requirements
    - Provide a URL to a job description
    
    **Tips for better results:**
    - Include specific skills you're looking for (e.g., Python, Java, SQL)
    - Mention job levels (e.g., entry-level, mid-level, senior)
    - Specify time constraints (e.g., "within 30 minutes")
    - Include test types if you have preferences (e.g., cognitive, personality)
    
    **Example queries:**
    - "I am hiring for Java developers who can also collaborate effectively with my business teams. Looking for an assessment(s) that can be completed in 40 minutes."
    - "Looking to hire mid-level professionals who are proficient in Python, SQL and JavaScript. Need an assessment package that can test all skills with max duration of 60 minutes."
    - "I am hiring for an analyst and wants applications to screen using Cognitive and personality tests, what options are available within 45 mins."
    """)
    
    # Initialize components
    @st.cache_resource
    def initialize_components():
        # Load assessments
        assessments = load_assessments()
        
        # Setup ChromaDB
        client, collection = setup_chroma_db()
        
        # Index assessments if collection is empty
        if collection.count() == 0 and assessments:
            st.info("Indexing assessments in the vector database...")
            
            # Prepare documents for indexing
            documents = []
            metadatas = []
            ids = []
            
            # Create a mapping to track original test_ids
            test_id_to_chroma_id = {}
            
            for item in assessments:
                # Create a rich text representation for embedding
                text = create_rich_text_representation(item)
                documents.append(text)
                
                # Convert metadata values to ensure they are ChromaDB compatible
                converted_metadata = convert_metadata_values(item)
                metadatas.append(converted_metadata)
                
                # Generate a unique ID for ChromaDB
                test_id = str(item['test_id'])
                if test_id in test_id_to_chroma_id:
                    # If this test_id already exists, create a new unique ID
                    chroma_id = f"{test_id}_{uuid.uuid4().hex[:8]}"
                else:
                    # First occurrence of this test_id
                    chroma_id = test_id
                    test_id_to_chroma_id[test_id] = chroma_id
                
                ids.append(chroma_id)
            
            # Add to ChromaDB
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            st.success(f"Indexed {len(assessments)} assessments")
        
        return assessments, collection
    
    # Initialize components
    assessments, collection = initialize_components()
    
    if not assessments:
        st.error("No assessment data available. Please check the JSON file.")
        return
    
    # Input section
    st.header("Input")
    input_type = st.radio(
        "Choose input type:",
        ["Text Query", "Job Description URL"]
    )
    
    query = None
    if input_type == "Text Query":
        query = st.text_area(
            "Enter your requirements:",
            height=150,
            placeholder="Describe the job role and requirements..."
        )
    else:
        url = st.text_input(
            "Enter job description URL:",
            placeholder="https://..."
        )
        if url:
            try:
                with st.spinner("Extracting text from URL..."):
                    query = extract_text_from_url(url)
                st.success("✅ Successfully extracted text from URL")
                with st.expander("View extracted text"):
                    st.text(query)
            except Exception as e:
                st.error(f"❌ Error extracting text from URL: {str(e)}")
    
    # Number of results slider
    n_results = st.slider(
        "Number of recommendations:",
        min_value=1,
        max_value=10,
        value=5
    )
    
    # Process query
    if st.button("Get Recommendations") and query:
        try:
            with st.spinner("🔍 Searching for relevant assessments..."):
                # Get initial recommendations with similarity scores
                recommendations = query_vector_db(query, collection, n_results)
                
                # Display results
                if recommendations:
                    st.header("Recommendations")
                    
                    # Display each recommendation
                    for i, result in enumerate(recommendations, 1):
                        # Create a more descriptive title without the relevance score
                        title = f"{i}. {result['name']}"
                        if result.get('Test Type'):
                            title += f" ({format_test_type(result['Test Type'])})"
                        
                        with st.expander(title):
                            # Create two columns for better layout
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                st.markdown("### Description")
                                st.write(result.get('Description', 'N/A'))
                                
                                # Extract duration from Assessment length
                                duration = result.get('Assessment length', 'N/A')
                                if isinstance(duration, str) and '=' in duration:
                                    duration = duration.split('=')[-1].strip()
                                
                                # Create a table with key information
                                key_info = pd.DataFrame({
                                    "Assessment Name": [result['name']],
                                    "Remote Testing": [result.get('remote_testing', 'No')],
                                    "Adaptive/IRT": [result.get('adaptive_irt', 'No')],
                                    "Duration": [f"{duration} minutes"],
                                    "Test Type": [format_test_type(result.get('Test Type', 'N/A'))]
                                })
                                
                                # Display the table without index
                                st.table(key_info.style.hide(axis='index'))
                                
                                # Add link to assessment if URL exists
                                if 'url' in result and result['url']:
                                    st.markdown(f"[View Assessment Details]({result['url']})")
                            
                            with col2:
                                st.markdown("### Additional Details")
                                st.markdown(f"**Test ID:** {result['test_id']}")
                                st.markdown(f"**Description:** {result['Description']}")
                                #st.markdown("**Support Features:**")
                                #st.markdown(f"- Remote Testing: {result['remote_testing']}")
                                #st.markdown(f"- Adaptive/IRT Support: {result['adaptive_irt']}")
                                st.markdown("**Additional Information:**")
                                st.markdown(f"- Job Levels: {result['Job levels']}")
                                st.markdown(f"- Languages: {result['Languages']}")
                                #st.markdown(f"- Category: {result['category']}")
                                
                                # Display test type meanings
                                #st.markdown("### Test Type Meanings")
                                #test_type_meanings = {
                                #    "A": "Ability & Aptitude",
                                #    "B": "Biodata & Situational Judgement",
                                    #"C": "Competencies",
                                   # "D": "Development & 360",
                                   # "E": "Assessment Exercises",
                                   # "K": "Knowledge & Skills",
                                    #"P": "Personality & Behavior",
                                    #"S": "Simulations"
                                #}
                                
                                # Get test types from the result
                                test_types = result.get('Test Type', '')
                                if isinstance(test_types, str):
                                    test_types = [t.strip() for t in test_types.split(',')]
                                elif isinstance(test_types, list):
                                    test_types = [str(t).strip() for t in test_types]
                                
                                # Display meanings for each test type
                                #for test_type in test_types:
                                #    if test_type in test_type_meanings:
                                #        st.markdown(f"**{test_type}**: {test_type_meanings[test_type]}")
                                #    else:
                                #        st.markdown(f"**{test_type}**: Unknown test type")
                else:
                    st.warning("No recommendations found.")
        except Exception as e:
            st.error(f"❌ Error processing request: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("Built with Streamlit, ChromaDB, and Groq LLM")

if __name__ == "__main__":
    main()

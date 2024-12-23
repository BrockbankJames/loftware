import streamlit as st
import json
import numpy as np
import pandas as pd
import io
import sqlite3
import pickle
from bs4 import BeautifulSoup
import requests
import re
import os
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# We'll use scipy for cosine similarity
from scipy.spatial.distance import cosine

# Import Sentence-BERT (sentence-transformers)
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

###############################################################################
# 1. Initialize SBERT Model
###############################################################################
@st.cache_resource
def load_sbert_model():
    """
    Load the Sentence-BERT model once and cache it.
    """
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model

sbert_model = load_sbert_model()

###############################################################################
# 2. Cosine Similarity
###############################################################################
def cosine_similarity(a, b):
    """
    Returns cosine similarity in [0, 1].
    1 - cosine distance under the hood.
    """
    return 1.0 - cosine(a, b)

def calculate_seo_weighted_embedding(embeddings_array, sections, keyword_text):
    """
    Calculate SEO-optimized weighted embedding
    """
    num_sections = len(sections)
    weights = np.ones(num_sections)
    
    # 1. Keyword Analysis
    keyword_terms = set(keyword_text.lower().split())
    
    for i, section in enumerate(sections):
        content = section['content'].lower()
        tag_type = section['tag_type']
        
        # Calculate keyword density
        words = content.split()
        word_count = len(words)
        if word_count == 0:
            continue
            
        # Count keyword occurrences
        keyword_count = sum(1 for term in keyword_terms if term in content)
        keyword_density = keyword_count / word_count
        
        # Boost weight based on keyword presence
        weights[i] *= (1 + keyword_density * 2)
        
        # 2. Position/Structure Importance
        if tag_type in ['H1', 'H2', 'H3']:
            weights[i] *= 2.0  # Higher weight for headings
        elif i < 3:  # First three sections
            weights[i] *= 1.5
            
        # 3. Length Factor
        length_factor = min(word_count / 100, 2.0)
        weights[i] *= length_factor

    # 4. TF-IDF Component
    tfidf = TfidfVectorizer(
        stop_words='english',
        max_features=1000,
        ngram_range=(1, 2)
    )
    
    texts = [section['content'] for section in sections]
    try:
        tfidf_matrix = tfidf.fit_transform(texts)
        tfidf_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
        weights *= (tfidf_scores / tfidf_scores.max())
    except:
        pass
    
    # Normalize final weights
    weights = weights / weights.sum()
    
    # Calculate weighted average embedding
    weighted_embedding = np.average(embeddings_array, axis=0, weights=weights)
    
    return weighted_embedding, weights

###############################################################################
# 3. Scrape & Split Content: Paragraphs, Headings, and DIV-only text
###############################################################################
def scrape_and_split_content(url):
    """
    Scrape the given URL and return a list of content sections with hierarchical structure.
    Each section is a dict with:
        {
            "title": <human-readable label>,
            "content": <the extracted text>,
            "tag_type": <HTML tag or type>,
            "html": <the original HTML>,
            "level": <hierarchical level>,
            "parent_header": <parent header text>
        }
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
        }
        
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer', 'form', 'button']):
            element.decompose()
        
        sections = []
        seen_content = set()
        current_headers = {'h1': None, 'h2': None, 'h3': None, 'h4': None, 'h5': None, 'h6': None}
        
        # Identify main content area
        main_content = (
            soup.find(['main', 'article', '[role="main"]']) 
            or soup.find('body')
        )
        
        if not main_content:
            return sections
        
        # Process all content elements in order
        for element in main_content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'div']):
            tag = element.name
            text = ' '.join(element.stripped_strings)
            
            # Skip empty or very short content
            if not text or len(text) <= 5 or text in seen_content:
                continue
                
            # Handle headers
            if tag.startswith('h'):
                level = int(tag[1])
                # Update current header at this level
                current_headers[tag] = text
                # Clear all lower-level headers
                for i in range(level + 1, 7):
                    current_headers[f'h{i}'] = None
                
                sections.append({
                    "title": f"{tag.upper()}: {text}",
                    "content": text,
                    "tag_type": tag.upper(),
                    "html": str(element),
                    "level": level,
                    "parent_header": current_headers.get(f'h{level-1}') if level > 1 else None
                })
                seen_content.add(text)
            
            # Handle paragraphs
            elif tag == 'p':
                # Find the closest parent header
                parent_header = None
                parent_level = None
                for i in range(1, 7):
                    if current_headers[f'h{i}']:
                        parent_header = current_headers[f'h{i}']
                        parent_level = i
                        break
                
                sections.append({
                    "title": f"P: {text[:50]}...",
                    "content": text,
                    "tag_type": "P",
                    "html": str(element),
                    "level": parent_level + 1 if parent_level else 7,  # If no parent header, put at bottom
                    "parent_header": parent_header
                })
                seen_content.add(text)
            
            # Handle divs (only those without nested p or h tags)
            elif tag == 'div':
                # Skip if this <div> has nested <p> or <h>
                if element.find(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                    continue
                
                div_text_list = []
                for content in element.contents:
                    if content.name not in ['p','h1','h2','h3','h4','h5','h6'] and content.string:
                        clean_text = content.string.strip()
                        if clean_text:
                            div_text_list.append(clean_text)
                    elif isinstance(content, str) and content.strip():
                        div_text_list.append(content.strip())
                
                final_text = ' '.join(div_text_list)
                if final_text and len(final_text) > 5 and final_text not in seen_content:
                    # Find closest parent header
                    parent_header = None
                    parent_level = None
                    for i in range(1, 7):
                        if current_headers[f'h{i}']:
                            parent_header = current_headers[f'h{i}']
                            parent_level = i
                            break
                    
                    sections.append({
                        "title": f"DIV: {final_text[:50]}...",
                        "content": final_text,
                        "tag_type": "DIV",
                        "html": str(element),
                        "level": parent_level + 1 if parent_level else 7,
                        "parent_header": parent_header
                    })
                    seen_content.add(final_text)
        
        # Sort sections by their hierarchical level
        sections.sort(key=lambda x: (x.get('level', 7), sections.index(x)))
        
        return sections
    
    except Exception as e:
        raise Exception(f"Error scraping URL: {str(e)}")

###############################################################################
# 4. Database Setup & Utility Functions
###############################################################################
if 'current_embedding' not in st.session_state:
    st.session_state.current_embedding = None

def setup_database():
    """Create the database and table if they don't exist."""
    conn = sqlite3.connect('embeddings.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS embeddings
                 (keyword TEXT PRIMARY KEY, 
                  embedding BLOB,
                  content TEXT,
                  embedding_type TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    return conn

def setup_similarity_table():
    """Create the similarity table if it doesn't exist."""
    conn = sqlite3.connect('embeddings.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS similarities
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  keyword_text TEXT,
                  url TEXT,
                  similarity_score REAL,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  UNIQUE(keyword_text, url))''')
    conn.commit()
    return conn

def save_similarity(keyword_text, url, similarity_score):
    """Save the similarity score as a float in the database."""
    try:
        conn = setup_similarity_table()
        cursor = conn.cursor()

        # Ensure we're storing a float
        if not isinstance(similarity_score, float):
            similarity_score = float(similarity_score)

        cursor.execute('''
            INSERT OR REPLACE INTO similarities 
            (keyword_text, url, similarity_score)
            VALUES (?, ?, ?)
        ''', (keyword_text, url, similarity_score))
        conn.commit()
        st.success(f"Saved similarity score for '{keyword_text}' and '{url}'")
    except Exception as e:
        st.error(f"Error saving similarity: {e}")
    finally:
        conn.close()

def get_all_similarities():
    """Retrieve all similarity scores from the database as floats."""
    conn = setup_similarity_table()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT keyword_text, url, similarity_score, created_at
        FROM similarities
        ORDER BY created_at DESC
    ''')
    results = cursor.fetchall()
    conn.close()
    
    # Each row is (keyword_text: str, url: str, similarity_score: float, created_at: str)
    # If the column type is REAL, SQLite should return a float automatically.
    return results

def save_to_sqlite(keyword, embedding, content="", embedding_type="keyword"):
    """Save the embedding and content to SQLite database."""
    try:
        conn = setup_database()
        cursor = conn.cursor()
        embedding_bytes = pickle.dumps(embedding)
        cursor.execute('''INSERT OR REPLACE INTO embeddings 
                         (keyword, embedding, content, embedding_type)
                         VALUES (?, ?, ?, ?)''', 
                      (keyword, embedding_bytes, content, embedding_type))
        conn.commit()
        st.success(f"Saved {embedding_type} embedding for '{keyword}' to database")
    except Exception as e:
        st.error(f"Error saving to database: {e}")
    finally:
        conn.close()

def delete_from_sqlite(keyword):
    """Delete an embedding from the database."""
    try:
        conn = setup_database()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM embeddings WHERE keyword = ?', (keyword,))
        conn.commit()
        st.success(f"Deleted embedding for '{keyword}' from database")
    except Exception as e:
        st.error(f"Error deleting from database: {e}")
    finally:
        conn.close()

def delete_all_embeddings():
    """Delete all embeddings from the database."""
    try:
        conn = setup_database()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM embeddings')
        conn.commit()
        st.success("Deleted all embeddings from database")
    except Exception as e:
        st.error(f"Error deleting embeddings: {e}")
    finally:
        conn.close()

def get_all_embeddings():
    """Retrieve all embeddings from the database."""
    conn = setup_database()
    cursor = conn.cursor()
    cursor.execute('SELECT keyword, embedding, embedding_type, created_at FROM embeddings')
    results = cursor.fetchall()
    conn.close()
    return [(row[0], pickle.loads(row[1]), row[2], row[3]) for row in results]

def get_embeddings_by_type(embedding_type):
    """Retrieve embeddings of a specific type from the database."""
    conn = setup_database()
    cursor = conn.cursor()
    cursor.execute('SELECT keyword, embedding, embedding_type, created_at FROM embeddings WHERE embedding_type = ?', (embedding_type,))
    results = cursor.fetchall()
    conn.close()
    return [(row[0], pickle.loads(row[1]), row[2], row[3]) for row in results]

def calculate_hierarchical_embedding(sections):
    """
    Calculate embeddings using hierarchical structure
    
    Structure:
    - H1 (weight: 1.5)
        - H2 (weight: 1.3)
            - H3 (weight: 1.2)
                - Paragraphs (weight: 1.0)
    """
    
    def get_tag_weight(tag_type):
        weights = {
            'H1': 1.5,
            'H2': 1.3,
            'H3': 1.2,
            'P': 1.0
        }
        return weights.get(tag_type.upper(), 1.0)

    # Group sections by their hierarchical level
    hierarchy = {}
    current_h1 = None
    current_h2 = None
    current_h3 = None

    for section in sections:
        tag_type = section['tag_type'].upper()
        embedding = section['embedding']
        
        if tag_type == 'H1':
            current_h1 = embedding
            current_h2 = None
            current_h3 = None
            if 'H1' not in hierarchy:
                hierarchy['H1'] = []
            hierarchy['H1'].append(embedding)
            
        elif tag_type == 'H2':
            current_h2 = embedding
            current_h3 = None
            if 'H2' not in hierarchy:
                hierarchy['H2'] = []
            hierarchy['H2'].append(embedding)
            
        elif tag_type == 'H3':
            current_h3 = embedding
            if 'H3' not in hierarchy:
                hierarchy['H3'] = []
            hierarchy['H3'].append(embedding)
            
        elif tag_type == 'P':
            # Associate paragraph with its nearest header
            if 'P' not in hierarchy:
                hierarchy['P'] = []
            hierarchy['P'].append({
                'embedding': embedding,
                'h1': current_h1,
                'h2': current_h2,
                'h3': current_h3
            })

    # Calculate weighted averages at each level
    final_embedding = np.zeros_like(sections[0]['embedding'])
    total_weight = 0

    # Process headers
    for tag_type in ['H1', 'H2', 'H3']:
        if tag_type in hierarchy and hierarchy[tag_type]:
            weight = get_tag_weight(tag_type)
            level_embedding = np.mean([emb for emb in hierarchy[tag_type]], axis=0)
            final_embedding += level_embedding * weight
            total_weight += weight

    # Process paragraphs with their hierarchical context
    if 'P' in hierarchy and hierarchy['P']:
        p_weight = get_tag_weight('P')
        p_embeddings = []
        
        for p in hierarchy['P']:
            p_emb = p['embedding']
            context_emb = p_emb
            
            # Add contextual influence from headers
            if p['h3'] is not None:
                context_emb = (context_emb + p['h3'] * 0.2)
            if p['h2'] is not None:
                context_emb = (context_emb + p['h2'] * 0.3)
            if p['h1'] is not None:
                context_emb = (context_emb + p['h1'] * 0.5)
                
            p_embeddings.append(context_emb)
        
        if p_embeddings:
            p_final = np.mean(p_embeddings, axis=0)
            final_embedding += p_final * p_weight
            total_weight += p_weight

    # Normalize
    if total_weight > 0:
        final_embedding /= total_weight

    return final_embedding

def advanced_similarity_analysis(keyword, sections, keyword_embedding, settings):
    """
    Comprehensive similarity analysis using hierarchical structure
    """
    try:
        # Convert sections to proper format
        formatted_sections = [
            {
                'tag_type': section['tag_type'],
                'embedding': section['embedding'],
                'content': section['content']
            }
            for section in sections
        ]
        
        # Calculate hierarchical embedding
        hierarchical_embedding = calculate_hierarchical_embedding(formatted_sections)
        
        # Calculate base similarity with hierarchical embedding
        base_sim = cosine_similarity(keyword_embedding, hierarchical_embedding)
        
        # Rest of your similarity calculations...
        # [Previous code for context and structural similarity]
        
        return base_sim, {
            'base_similarity': base_sim,
            'hierarchical_embedding': hierarchical_embedding
        }
        
    except Exception as e:
        print(f"Advanced analysis error: {e}")
        return 0.0, {'base_similarity': 0.0}

###############################################################################
# Advanced Similarity Analysis Functions
###############################################################################

def calculate_context_weighted_similarity(keyword_embedding, section_embedding, section_content, keyword):
    """Enhanced similarity with context awareness"""
    base_similarity = cosine_similarity(keyword_embedding, section_embedding)
    multipliers = 1.0
    
    # Title/heading importance
    if any(tag in section_content.lower() for tag in ['<h1>', '<h2>', '<h3>']):
        multipliers *= 1.2
    
    # Keyword proximity
    keyword_terms = set(keyword.lower().split())
    try:
        sentences = sent_tokenize(section_content.lower())
        
        for sentence in sentences:
            words = word_tokenize(sentence)
            for i, word in enumerate(words):
                if word in keyword_terms:
                    context_window = words[max(0, i-5):min(len(words), i+6)]
                    other_terms = keyword_terms - {word}
                    if any(term in context_window for term in other_terms):
                        multipliers *= 1.1
                        break
    except Exception as e:
        print(f"Error in tokenization: {e}")
    
    return base_similarity * multipliers

def structural_similarity(keyword_embedding, section_embeddings, section_positions):
    """Calculate similarity with position-based weighting"""
    num_sections = len(section_embeddings)
    if num_sections == 0:
        return 0.0
        
    # Position weights decay exponentially
    position_weights = np.exp(-0.5 * np.array(section_positions) / num_sections)
    
    # Calculate weighted similarities
    similarities = []
    for emb, weight in zip(section_embeddings, position_weights):
        sim = cosine_similarity(keyword_embedding, emb) * weight
        similarities.append(sim)
    
    return np.mean(similarities)

def keyword_density_analysis(content, keyword):
    """Analyze keyword density and placement"""
    try:
        words = word_tokenize(content.lower())
        keyword_terms = set(keyword.lower().split())
        
        # Count keyword occurrences
        total_words = len(words)
        keyword_count = sum(1 for word in words if word in keyword_terms)
        
        if total_words == 0:
            return 0.0
            
        density = keyword_count / total_words
        return min(density * 5, 1.0)  # Cap at 1.0
    except:
        return 0.0

def advanced_similarity_analysis(keyword, sections, keyword_embedding, settings):
    """
    Comprehensive similarity analysis combining multiple approaches
    """
    try:
        # Prepare section data
        section_embeddings = [section['embedding'] for section in sections]
        section_contents = [section['content'] for section in sections]
        
        # 1. Base cosine similarity
        base_similarities = [
            cosine_similarity(keyword_embedding, emb) 
            for emb in section_embeddings
        ]
        base_sim = np.mean(base_similarities)
        
        # Get weights from settings or use defaults
        weights = settings.get('weights', {
            'base': 0.4,
            'context': 0.3,
            'structural': 0.15,
            'keyword_density': 0.15
        })
        
        final_score = 0.0
        
        # 2. Context-aware similarity
        if settings.get('use_context_weighting', True):
            context_similarities = [
                calculate_context_weighted_similarity(
                    keyword_embedding, emb, content, keyword
                )
                for emb, content in zip(section_embeddings, section_contents)
            ]
            final_score += weights['context'] * np.mean(context_similarities)
        else:
            weights['base'] += weights['context']
        
        # 3. Structural similarity
        struct_sim = structural_similarity(
            keyword_embedding,
            section_embeddings,
            range(len(section_embeddings))
        )
        final_score += weights['structural'] * struct_sim
        
        # 4. Keyword density
        density_scores = [
            keyword_density_analysis(content, keyword)
            for content in section_contents
        ]
        density_score = np.mean(density_scores)
        final_score += weights['keyword_density'] * density_score
        
        # Add weighted base similarity
        final_score += weights['base'] * base_sim
        
        return final_score, {
            'base_similarity': base_sim,
            'context_similarity': np.mean(context_similarities) if settings.get('use_context_weighting') else None,
            'structural_similarity': struct_sim,
            'keyword_density': density_score
        }
        
    except Exception as e:
        print(f"Advanced analysis error: {e}")
        return base_sim, {'base_similarity': base_sim}

###############################################################################
# 5. Detailed Section Analysis
###############################################################################
def show_full_analysis(keyword_text, url, keyword_embedding, url_embs, overall_similarity):
    """
    Show detailed analysis of similarity between keyword and URL content.
    """
    st.write("### Detailed Section Analysis")
    try:
        # Create a DataFrame to store section similarities
        section_data = []
        
        for section_info in url_embs:
            # Unpack the tuple with all 4 values
            keyword_, embedding_, content_, timestamp_ = section_info
            
            # Calculate similarity for this section
            section_similarity = cosine_similarity(keyword_embedding, embedding_)
            
            # Get section title (remove URL part)
            section_title = keyword_.split(" (URL:")[0]
            
            section_data.append({
                'Section': section_title,
                'Similarity': section_similarity,
                'Content': content_,
                'Timestamp': timestamp_
            })
        
        # Convert to DataFrame and sort by similarity
        df = pd.DataFrame(section_data)
        df = df.sort_values('Similarity', ascending=False)
        
        # Display each section's analysis
        for _, row in df.iterrows():
            with st.expander(f"📑 {row['Section']} (Score: {row['Similarity']:.4f})"):
                st.write("**Content:**")
                st.write(row['Content'])
                st.write(f"**Analyzed:** {row['Timestamp']}")
                
                # Create a color gradient based on similarity
                color = 'green' if row['Similarity'] >= 0.8 else 'orange' if row['Similarity'] >= 0.6 else 'red'
                st.markdown(f"**Match Level:** :{color}[{row['Similarity']:.4f}]")
        
        # Show overall statistics
        st.write("### Summary Statistics")
        st.write(f"**Overall Similarity:** {overall_similarity:.4f}")
        st.write(f"**Highest Section Match:** {df['Similarity'].max():.4f}")
        st.write(f"**Average Section Match:** {df['Similarity'].mean():.4f}")
        st.write(f"**Number of Sections:** {len(df)}")
        
        # CSV download
        df['Similarity'] = df['Similarity'].apply(lambda x: f"{x:.4f}")
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Analysis (CSV)",
            data=csv,
            file_name=f"similarity_analysis_{keyword_text}_{url}.csv",
            mime="text/csv"
        )
        
    except Exception as e:
        st.error(f"An error occurred in analysis: {str(e)}")

###############################################################################
# 6. Streamlit Tabs
###############################################################################
tab1, tab2, tab3, tab4 = st.tabs(["Keywords", "URLs", "Compare", "Settings"])

# ---------------- Tab 1: Keywords ----------------
with tab1:
    st.write("Enter a keyword to generate a Sentence-BERT embedding.")
    keyword = st.text_input("Enter keyword:")
    
    if st.button("Generate and Save Embedding"):
        if keyword:
            try:
                # Use Sentence-BERT to create an embedding
                embedding = sbert_model.encode(keyword)
                
                # Save to database
                save_to_sqlite(keyword, embedding, embedding_type="keyword")
                
                st.success(f"Generated and saved SBERT embedding for: {keyword}")
                st.write("Preview of embedding vector:")
                st.code(str(embedding[:10]) + "...", language="python")
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("Please enter a keyword.")
    
    # Display saved keyword embeddings
    st.subheader("Saved Keyword Embeddings")
    keyword_embeddings = get_embeddings_by_type("keyword")
    
    if keyword_embeddings:
        for i, (kwd, emb, emb_type, _) in enumerate(keyword_embeddings):
            with st.expander(f"📝 {kwd}"):
                st.write(f"Embedding type: {emb_type}")
                st.write(f"Embedding length: {len(emb)}")
                if st.button("Delete", key=f"delete_keyword_{i}"):
                    delete_from_sqlite(kwd)
                    st.rerun()
    else:
        st.write("No keyword embeddings saved yet.")

# ---------------- Tab 2: URLs ----------------
with tab2:
    # Initialize session state variables
    if 'section_contents' not in st.session_state:
        st.session_state.section_contents = {}
    if 'current_sections' not in st.session_state:
        st.session_state.current_sections = None
    if 'current_url' not in st.session_state:
        st.session_state.current_url = None
    if 'current_embeddings' not in st.session_state:
        st.session_state.current_embeddings = None

    st.write("Enter a URL to generate embeddings for its content sections.")
    url = st.text_input("Enter URL:")
    
    # Add the custom exclusion settings
    with st.expander("Advanced Scraping Settings"):
        # Default exclusions
        default_exclusions = ['script', 'style', 'nav', 'footer', 'form', 'button']
        st.write("Default excluded elements:", ", ".join(default_exclusions))
        
        # Custom class exclusions
        custom_classes = st.text_input(
            "Exclude CSS classes (separate by commas):",
            help="Example: sidebar, menu-wrapper, advertisement"
        )
        
        # Custom ID exclusions
        custom_ids = st.text_input(
            "Exclude element IDs (separate by commas):",
            help="Example: sidebar, comments-section, related-posts"
        )
        
        # Preview button
        if st.button("Preview Exclusions"):
            if url:
                try:
                    response = requests.get(url)
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Find elements that would be excluded
                    excluded_elements = []
                    
                    # Check classes
                    if custom_classes:
                        for class_name in custom_classes.split(','):
                            class_name = class_name.strip()
                            elements = soup.find_all(class_=class_name)
                            for elem in elements:
                                excluded_elements.append({
                                    'type': 'class',
                                    'name': class_name,
                                    'content': elem.get_text()[:100] + '...' if len(elem.get_text()) > 100 else elem.get_text()
                                })
                    
                    # Check IDs
                    if custom_ids:
                        for id_name in custom_ids.split(','):
                            id_name = id_name.strip()
                            element = soup.find(id=id_name)
                            if element:
                                excluded_elements.append({
                                    'type': 'id',
                                    'name': id_name,
                                    'content': element.get_text()[:100] + '...' if len(element.get_text()) > 100 else element.get_text()
                                })
                    
                    if excluded_elements:
                        st.write("### Elements that will be excluded:")
                        for elem in excluded_elements:
                            st.write(f"**{elem['type'].upper()}:** {elem['name']}")
                            st.write(f"Content preview: {elem['content']}")
                    else:
                        st.write("No elements found matching the exclusion criteria.")
                        
                except Exception as e:
                    st.error(f"Error previewing exclusions: {str(e)}")
            else:
                st.warning("Please enter a URL first.")
    
    if st.button("Scrape and Generate Embeddings"):
        if url:
            try:
                # Get custom exclusions from inputs
                classes_to_exclude = custom_classes if custom_classes else ""
                ids_to_exclude = custom_ids if custom_ids else ""
                
                # Show what's being excluded
                if classes_to_exclude or ids_to_exclude:
                    st.write("### Excluding from scrape:")
                    if classes_to_exclude:
                        st.write("**Classes:**", classes_to_exclude)
                    if ids_to_exclude:
                        st.write("**IDs:**", ids_to_exclude)
                
                # Call scraping function with exclusions
                sections = scrape_and_split_content(
                    url,
                    custom_classes=classes_to_exclude,
                    custom_ids=ids_to_exclude
                )
                
                if sections:
                    st.success(f"Found {len(sections)} content sections!")
                    
                    # Store in session state
                    st.session_state.current_sections = sections
                    st.session_state.current_url = url
                    
                    # Store section contents and generate embeddings
                    st.session_state.section_contents = {}
                    all_embeddings = []
                    
                    # Group sections by their hierarchical level
                    hierarchy = {'H1': [], 'H2': [], 'H3': [], 'H4': [], 'H5': [], 'H6': [], 'P': [], 'DIV': []}
                    
                    # Calculate hierarchical embedding
                    sections_with_embeddings = []
                    for section in sections:
                        # Generate embedding for this section
                        emb = sbert_model.encode(section['content'])
                        
                        sections_with_embeddings.append({
                            'tag_type': section['tag_type'],
                            'embedding': emb,
                            'content': section['content'],
                            'level': section.get('level', 7),
                            'parent_header': section.get('parent_header')
                        })
                        
                        # Store in appropriate hierarchy level
                        tag_type = section['tag_type']
                        hierarchy[tag_type].append({
                            'content': section['content'],
                            'embedding': emb,
                            'parent_header': section.get('parent_header'),
                            'level': section.get('level', 7)
                        })
                        
                        # Store for display and later use
                        url_keyword = f"{tag_type}: {section['title']} (URL: {url})"
                        st.session_state.section_contents[url_keyword] = {
                            'content': section['content'],
                            'tag_type': tag_type,
                            'parent_header': section.get('parent_header'),
                            'level': section.get('level', 7)
                        }
                        all_embeddings.append(emb)
                    
                    # Calculate final hierarchical embedding
                    hierarchical_emb = calculate_hierarchical_embedding(sections_with_embeddings)
                    st.session_state.current_embeddings = all_embeddings
                    
                    # Display sections in hierarchical order
                    st.write("### Content Structure")
                    
                    # Display H1 headers and their children
                    for h1_section in hierarchy['H1']:
                        with st.expander(f"📑 H1: {h1_section['content'][:100]}..."):
                            st.write(h1_section['content'])
                            st.write("Embedding generated!")
                            
                            # Find and display children
                            for tag in ['H2', 'H3', 'H4', 'H5', 'H6', 'P', 'DIV']:
                                children = [
                                    s for s in hierarchy[tag] 
                                    if s['parent_header'] == h1_section['content']
                                ]
                                for child in children:
                                    with st.expander(f"{tag}: {child['content'][:100]}..."):
                                        st.write(child['content'])
                                        st.write("Embedding generated!")
                    
                    # Display orphaned sections (no H1 parent)
                    orphaned = []
                    for tag in ['H2', 'H3', 'H4', 'H5', 'H6', 'P', 'DIV']:
                        orphaned.extend([
                            s for s in hierarchy[tag]
                            if not s['parent_header']
                        ])
                    
                    if orphaned:
                        st.write("### Other Content")
                        for section in orphaned:
                            with st.expander(f"📄 {section['content'][:100]}..."):
                                st.write(section['content'])
                                st.write("Embedding generated!")
                    
                    # Save to database button
                    if st.button("Save All Sections to Database"):
                        try:
                            # Save hierarchical embedding first
                            save_to_sqlite(
                                f"HIERARCHICAL_EMBEDDING (URL: {url})",
                                hierarchical_emb,
                                content="Hierarchical embedding of all content",
                                embedding_type="url"
                            )
                            
                            # Save individual sections
                            for section in sections:
                                url_keyword = f"{section['tag_type']}: {section['title']} (URL: {url})"
                                emb = sbert_model.encode(section['content'])
                                save_to_sqlite(
                                    url_keyword,
                                    emb,
                                    content=section['content'],
                                    embedding_type="url",
                                    metadata=json.dumps({
                                        'tag_type': section['tag_type'],
                                        'parent_header': section.get('parent_header'),
                                        'level': section.get('level', 7)
                                    })
                                )
                            st.success("All sections saved successfully!")
                            st.rerun()
                        except Exception as save_error:
                            st.error(f"Error saving sections: {save_error}")
                else:
                    st.warning("No content sections found on the page.")
            except Exception as e:
                st.error(f"Error processing URL: {str(e)}")
        else:
            st.warning("Please enter a URL.")

    # Display saved embeddings grouped by URL
    st.subheader("Saved URL Embeddings")
    try:
        url_embeddings = get_embeddings_by_type("url")
        
        # Group by URL
        url_groups = {}
        for keyword_, embedding_, content_, timestamp_ in url_embeddings:
            url_part = keyword_.split("URL: ")[-1].strip(")")
            if url_part not in url_groups:
                url_groups[url_part] = []
            url_groups[url_part].append((keyword_, embedding_, content_, timestamp_))
        
        if url_groups:
            for url_, sections_ in url_groups.items():
                with st.expander(f"🌐 {url_}"):
                    st.write(f"Number of sections: {len(sections_)}")
                    
                    # Show hierarchical view
                    if st.checkbox("Show sections", key=f"show_sections_{url_}"):
                        # Group sections by tag type
                        sections_by_type = {'H1': [], 'H2': [], 'H3': [], 'H4': [], 'H5': [], 'H6': [], 'P': [], 'DIV': []}
                        for section_info in sections_:
                            tag_type = section_info[0].split(":")[0].strip()
                            sections_by_type[tag_type].append(section_info)
                        
                        # Display in hierarchical order
                        for tag in ['H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'P', 'DIV']:
                            if sections_by_type[tag]:
                                for section in sections_by_type[tag]:
                                    section_title = section[0].split(" (URL:")[0]
                                    st.write(f"- {section_title}")
                    
                    # Delete URL button
                    if st.button("Delete URL", key=f"delete_url_{url_}"):
                        for section_info in sections_:
                            delete_from_sqlite(section_info[0])
                        st.rerun()
        else:
            st.write("No URL embeddings saved yet.")
    except Exception as load_error:
        st.error(f"Error loading saved embeddings: {load_error}")

# ---------------- Tab 3: Compare ----------------
with tab3:
    st.subheader("Compare Embeddings")
    
    try:
        # 1. Fetch keyword embeddings
        keyword_embeddings = get_embeddings_by_type("keyword")
        keyword_options = [emb[0] for emb in keyword_embeddings]
        
        # 2. Fetch URL embeddings
        url_embeddings = get_embeddings_by_type("url")
        url_groups = {}
        for keyword_, embedding_, content_, timestamp_ in url_embeddings:
            url_part = keyword_.split("URL: ")[-1].strip(")")
            if url_part not in url_groups:
                url_groups[url_part] = []
            url_groups[url_part].append((keyword_, embedding_, content_, timestamp_))
        
        # 3. If we have at least one keyword embedding and one URL set
        if keyword_options and url_groups:
            col1, col2 = st.columns(2)
            
            with col1:
                selected_keyword = st.selectbox("Select Keyword Embedding", keyword_options)
                
            with col2:
                selected_url = st.selectbox("Select URL", list(url_groups.keys()))
            
            # Add analysis options
            analysis_type = st.radio(
                "Analysis Type",
                ["Simple", "SEO-Optimized"],
                help="""
                Simple: Basic cosine similarity
                SEO-Optimized: Weighted analysis considering keyword placement, density, and content structure
                """
            )
            
            if st.button("Analyze Content-Keyword Match"):
                # Retrieve the keyword embedding
                keyword_embedding = next(e[1] for e in keyword_embeddings if e[0] == selected_keyword)
                
                # Gather all chunk embeddings for the selected URL
                url_embs = url_groups[selected_url]
                embeddings_array = np.vstack([emb[1] for emb in url_embs])
                
                # Get sections content
                sections = [
                    {
                        "content": emb[2] if emb[2] else "",  # Use stored content from database
                        "tag_type": emb[0].split(":")[0].strip(),
                    }
                    for emb in url_embs
                ]
                
                if analysis_type == "SEO-Optimized":
                    average_embedding, section_weights = calculate_seo_weighted_embedding(
                        embeddings_array,
                        sections,
                        selected_keyword
                    )
                    
                    # Show detailed SEO analysis
                    with st.expander("SEO Analysis Details"):
                        seo_data = []
                        for i, (section, weight) in enumerate(zip(sections, section_weights)):
                            content_preview = section['content'][:100] + "..."
                            seo_data.append({
                                'Section': f"{section['tag_type']}: {content_preview}",
                                'SEO Weight': f"{weight:.4f}",
                                'Position': i + 1
                            })
                        
                        st.dataframe(
                            pd.DataFrame(seo_data),
                            use_container_width=True
                        )
                else:
                    # Simple averaging for comparison
                    average_embedding = np.mean(embeddings_array, axis=0)
                
                # Compute similarity
                overall_similarity = cosine_similarity(keyword_embedding, average_embedding)
                
                # Display results
                st.write("### Content-Keyword Match Analysis")
                st.write(f"**Overall Match Score:** {overall_similarity:.4f}")
                
                # Interpret the score
                if overall_similarity >= 0.8:
                    status = "Strong"
                    color = "green"
                elif overall_similarity >= 0.6:
                    status = "Moderate"
                    color = "orange"
                else:
                    status = "Weak"
                    color = "red"
                    
                st.markdown(f"**Match Status:** :{color}[{status}]")
                
                # SEO Recommendations
                with st.expander("SEO Recommendations"):
                    st.write("### Content Optimization Suggestions")
                    
                    if overall_similarity < 0.8:
                        st.write("Consider these improvements:")
                        suggestions = []
                        
                        # Analyze keyword placement
                        if sections:  # Check if there are any sections
                            first_section = sections[0]['content'].lower()
                            if not any(term in first_section for term in selected_keyword.lower().split()):
                                suggestions.append("- Include the target keyword in the first paragraph")
                        
                        # Check heading usage
                        if not any(s['tag_type'].startswith('H') for s in sections):
                            suggestions.append("- Add proper heading structure (H1, H2, etc.)")
                        
                        # Check content length
                        total_words = sum(len(s['content'].split()) for s in sections)
                        if total_words < 300:
                            suggestions.append("- Expand content length (aim for 300+ words)")
                        
                        if suggestions:
                            for suggestion in suggestions:
                                st.markdown(suggestion)
                    else:
                        st.success("Content is well-optimized for the target keyword!")
                
                # Save analysis results
                save_similarity(selected_keyword, selected_url, overall_similarity)
                
                # Show full analysis
                show_full_analysis(selected_keyword, selected_url, keyword_embedding, url_embs, overall_similarity)
        else:
            st.warning("Please add both keyword and URL embeddings before comparing.")

        # 5. Display saved similarity scores
        st.subheader("Saved Similarity Scores")
        similarities = get_all_similarities()
        
        if similarities:
            for keyword_text, url_, score, timestamp_ in similarities:
                with st.expander(f"📊 {keyword_text} vs {url_}"):
                    st.write(f"**Score:** {score:.4f}")
                    st.write(f"**Calculated:** {timestamp_}")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("Delete", key=f"delete_sim_{keyword_text}_{url_}"):
                            conn = setup_similarity_table()
                            cursor = conn.cursor()
                            cursor.execute(
                                'DELETE FROM similarities WHERE keyword_text = ? AND url = ?',
                                (keyword_text, url_)
                            )
                            conn.commit()
                            conn.close()
                            st.rerun()
                    
                    with col2:
                        if st.button("See Full Analysis", key=f"analysis_{keyword_text}_{url_}"):
                            kwd_emb = next(e[1] for e in keyword_embeddings if e[0] == keyword_text)
                            url_embs = url_groups[url_]
                            
                            # Pass the float 'score' directly to analysis
                            show_full_analysis(keyword_text, url_, kwd_emb, url_embs, score)
        else:
            st.write("No similarity scores saved yet.")
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# ---------------- Tab 4: Settings ----------------
with tab4:
    tab4_1, tab4_2 = st.tabs(["Manage Embeddings", "Analysis Settings"])
    
    # Tab 4.1: Manage Embeddings
    with tab4_1:
        st.subheader("Manage Saved Embeddings")
        
        embedding_type_filter = st.selectbox("Show embeddings of type:", ["All", "Keyword", "URL"])
        if embedding_type_filter == "All":
            saved_embeddings = get_all_embeddings()
        else:
            saved_embeddings = get_embeddings_by_type(embedding_type_filter.lower())

        if saved_embeddings:
            sort_by = st.selectbox("Sort embeddings by:", 
                                   ["Keyword (A-Z)", "Keyword (Z-A)", "Date (Newest)", "Date (Oldest)"])
            
            if sort_by == "Keyword (A-Z)":
                saved_embeddings.sort(key=lambda x: x[0].lower())
            elif sort_by == "Keyword (Z-A)":
                saved_embeddings.sort(key=lambda x: x[0].lower(), reverse=True)
            elif sort_by == "Date (Newest)":
                saved_embeddings.sort(key=lambda x: x[3], reverse=True)
            else:  # Date (Oldest)
                saved_embeddings.sort(key=lambda x: x[3])
            
            search = st.text_input("Search keywords:", "")
            filtered_embeddings = [emb for emb in saved_embeddings if search.lower() in emb[0].lower()]
            
            st.write(f"Showing {len(filtered_embeddings)} embeddings")
            
            if st.button("Delete All Embeddings"):
                if st.session_state.get('confirm_delete_all', False):
                    delete_all_embeddings()
                    st.session_state.confirm_delete_all = False
                    st.rerun()
                else:
                    st.session_state.confirm_delete_all = True
                    st.warning("Click again to confirm deleting all embeddings")
            
            for keyword_, embedding_, emb_type_, timestamp_ in filtered_embeddings:
                with st.expander(f"📌 {keyword_} ({emb_type_})"):
                    st.write(f"**Created:** {timestamp_}")
                    st.code(str(embedding_), language='python')
                    
                    col1, col2, col3 = st.columns([2, 2, 1])
                    with col1:
                        df = pd.DataFrame([embedding_])
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download CSV",
                            data=csv,
                            file_name=f"{keyword_}_embedding.csv",
                            mime="text/csv",
                            key=f"csv_{keyword_}"
                        )
                    with col2:
                        # Safely convert embedding_ to a JSON-serializable type
                        if hasattr(embedding_, "tolist"):
                            embedding_data = embedding_.tolist()
                        else:
                            embedding_data = embedding_
                        
                        json_data = json.dumps({
                            "keyword": keyword_,
                            "embedding": embedding_data
                        })
                        
                        st.download_button(
                            label="Download JSON",
                            data=json_data,
                            file_name=f"{keyword_}_embedding.json",
                            mime="application/json",
                            key=f"json_{keyword_}"
                        )
                    with col3:
                        if st.button("Delete", key=f"delete_{keyword_}"):
                            delete_from_sqlite(keyword_)
                            st.rerun()
        else:
            st.write("No embeddings saved yet.")
    
    # Tab 4.2: Analysis Settings
    with tab4_2:
        st.subheader("Analysis Settings")
        
        # Analysis settings
        st.write("### Similarity Analysis Settings")
        
        analysis_settings = {
            "use_context_weighting": st.checkbox("Use Context Weighting", value=True,
                help="Consider keyword proximity and HTML structure"),
            
            "use_topic_analysis": st.checkbox("Use Topic Analysis", value=True,
                help="Analyze topic distribution in content"),
            
            "num_topics": st.slider("Number of Topics", 3, 10, 5,
                help="Number of topics for topic modeling"),
            
            "position_weight": st.slider("Position Importance", 0.0, 1.0, 0.3,
                help="Weight of section position in similarity calculation")
        }
        
        # Weight adjustments
        st.write("### Similarity Weight Adjustments")
        weights = {
            "base": st.slider("Base Similarity Weight", 0.0, 1.0, 0.4, 0.05),
            "context": st.slider("Context Weight", 0.0, 1.0, 0.3, 0.05),
            "topic": st.slider("Topic Weight", 0.0, 1.0, 0.15, 0.05),
            "structural": st.slider("Structural Weight", 0.0, 1.0, 0.15, 0.05)
        }
        
        # Normalize weights to sum to 1
        total = sum(weights.values())
        if total > 0:
            weights = {k: v/total for k, v in weights.items()}
            analysis_settings['weights'] = weights
        
        # Save settings
        if st.button("Save Analysis Settings"):
            st.session_state.analysis_settings = analysis_settings
            st.success("Settings saved successfully!")
            
            # Show current weights
            st.write("### Current Normalized Weights")
            for weight_type, value in weights.items():
                st.write(f"**{weight_type.title()}:** {value:.3f}")

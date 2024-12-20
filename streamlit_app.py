import streamlit as st
from openai import OpenAI
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

# Initialize the OpenAI client with your API key from Streamlit secrets
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def setup_database():
    """Create the database and table if they don't exist"""
    conn = sqlite3.connect('embeddings.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS embeddings
                 (keyword TEXT PRIMARY KEY, 
                  embedding BLOB,
                  embedding_type TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    return conn

def setup_similarity_table():
    """Create the similarity table if it doesn't exist"""
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
    """Save the similarity score to database"""
    try:
        conn = setup_similarity_table()
        cursor = conn.cursor()
        cursor.execute('''INSERT OR REPLACE INTO similarities 
                         (keyword_text, url, similarity_score)
                         VALUES (?, ?, ?)''', 
                         (keyword_text, url, similarity_score))
        conn.commit()
        st.success(f"Saved similarity score for '{keyword_text}' and '{url}'")
    except Exception as e:
        st.error(f"Error saving similarity: {e}")
    finally:
        conn.close()

def get_all_similarities():
    """Retrieve all similarity scores from the database"""
    conn = setup_similarity_table()
    cursor = conn.cursor()
    cursor.execute('SELECT keyword_text, url, similarity_score, created_at FROM similarities ORDER BY created_at DESC')
    results = cursor.fetchall()
    conn.close()
    return results

def save_to_sqlite(keyword, embedding, embedding_type="keyword"):
    """Save the embedding to SQLite database"""
    try:
        conn = setup_database()
        cursor = conn.cursor()
        embedding_bytes = pickle.dumps(embedding)
        cursor.execute('''INSERT OR REPLACE INTO embeddings (keyword, embedding, embedding_type)
                         VALUES (?, ?, ?)''', (keyword, embedding_bytes, embedding_type))
        conn.commit()
        st.success(f"Saved {embedding_type} embedding for '{keyword}' to database")
    except Exception as e:
        st.error(f"Error saving to database: {e}")
    finally:
        conn.close()

def delete_from_sqlite(keyword):
    """Delete an embedding from the database"""
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
    """Delete all embeddings from the database"""
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
    """Retrieve all embeddings from the database"""
    conn = setup_database()
    cursor = conn.cursor()
    cursor.execute('SELECT keyword, embedding, embedding_type, created_at FROM embeddings')
    results = cursor.fetchall()
    conn.close()
    return [(row[0], pickle.loads(row[1]), row[2], row[3]) for row in results]

def get_embeddings_by_type(embedding_type):
    """Retrieve embeddings of a specific type from the database"""
    conn = setup_database()
    cursor = conn.cursor()
    cursor.execute('SELECT keyword, embedding, embedding_type, created_at FROM embeddings WHERE embedding_type = ?', (embedding_type,))
    results = cursor.fetchall()
    conn.close()
    return [(row[0], pickle.loads(row[1]), row[2], row[3]) for row in results]

def scrape_and_split_content(url):
    """Scrape webpage and split content by individual tags"""
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
        for element in soup(['script', 'style', 'header', 'nav', 'footer', 'form', 'button']):
            element.decompose()
            
        sections = []
        seen_content = set()
        
        # Find main content area
        main_content = soup.find(['main', 'article', '[role="main"]']) or soup.find('body')
        
        if main_content:
            # Process headings first
            for tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                for element in main_content.find_all(tag, recursive=True):
                    text = ' '.join(element.stripped_strings)
                    if text and len(text) > 5 and text not in seen_content:
                        sections.append({
                            "title": f"{tag.upper()}: {text}",
                            "content": text,
                            "tag_type": tag.upper()
                        })
                        seen_content.add(text)
            
            # Process paragraphs
            for p in main_content.find_all('p', recursive=True):
                # Skip if parent is a div we'll process later
                if p.parent.name != 'div':
                    text = ' '.join(p.stripped_strings)
                    if text and len(text) > 5 and text not in seen_content:
                        sections.append({
                            "title": f"P: {text[:50]}...",
                            "content": text,
                            "tag_type": "P"
                        })
                        seen_content.add(text)
            
            # Process divs last, only if they contain direct text
            for div in main_content.find_all('div', recursive=False):  # Only top-level divs
                # Get only direct text content
                direct_text = ' '.join(t for t in div.strings 
                                     if t.parent == div and t.strip())
                
                if direct_text and len(direct_text) > 5 and direct_text not in seen_content:
                    sections.append({
                        "title": f"DIV: {direct_text[:50]}...",
                        "content": direct_text,
                        "tag_type": "DIV"
                    })
                    seen_content.add(direct_text)
        
        return sections
    except Exception as e:
        raise Exception(f"Error scraping URL: {str(e)}")

def show_full_analysis(keyword_text, url, keyword_embedding, url_embs, overall_similarity):
    """Show detailed analysis of similarities"""
    st.write("---")
    st.subheader("Detailed Section Analysis")
    
    section_similarities = []
    
    for keyword, embedding, timestamp in url_embs:
        section_norm = np.linalg.norm(embedding)
        if section_norm > 0:
            normalized_section_embedding = embedding / section_norm
        else:
            normalized_section_embedding = embedding
        
        keyword_norm = np.linalg.norm(keyword_embedding)
        if keyword_norm > 0:
            normalized_keyword_embedding = keyword_embedding / keyword_norm
        else:
            normalized_keyword_embedding = keyword_embedding
        
        similarity = np.dot(normalized_keyword_embedding, normalized_section_embedding)
        section_similarities.append({
            'Section': keyword.split('(URL:')[0],
            'Similarity': similarity
        })
    
    # Convert to DataFrame and sort by similarity
    df = pd.DataFrame(section_similarities)
    df = df.sort_values('Similarity', ascending=False)
    
    # Format similarity scores
    df['Similarity'] = df['Similarity'].apply(lambda x: f"{x:.4f}")
    
    # Display the table with full width
    st.write("Section-by-Section Similarity Scores:")
    st.dataframe(
        df,
        use_container_width=True,  # This makes it full width
        hide_index=True  # This hides the index column
    )
    
    # Rest of the analysis code...
    st.write("---")
    st.write("### Overall Analysis")
    st.write(f"**Keyword:** {keyword_text}")
    st.write(f"**URL:** {url}")
    st.write(f"**Overall Similarity Score:** {overall_similarity:.4f}")
    
    # Create downloadable analysis
    analysis_json = {
        "keyword": keyword_text,
        "url": url,
        "overall_similarity": overall_similarity,
        "sections": [
            {
                "section": row['Section'],
                "similarity": row['Similarity']
            }
            for _, row in df.iterrows()
        ]
    }
    
    json_str = json.dumps(analysis_json, indent=2)
    st.download_button(
        label="Download Analysis (JSON)",
        data=json_str,
        file_name=f"similarity_analysis_{keyword_text}_{url}.json",
        mime="application/json"
    )

def check_and_migrate_database():
    """Check if database needs migration and perform if necessary"""
    try:
        conn = sqlite3.connect('embeddings.db')
        cursor = conn.cursor()
        
        cursor.execute("PRAGMA table_info(embeddings)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'embedding_type' not in columns:
            st.warning("Performing database migration...")
            
            cursor.execute('''CREATE TABLE IF NOT EXISTS embeddings_new
                            (keyword TEXT PRIMARY KEY, 
                             embedding BLOB,
                             embedding_type TEXT,
                             created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
            
            cursor.execute('''INSERT INTO embeddings_new (keyword, embedding, embedding_type, created_at)
                            SELECT keyword, embedding, 
                            CASE 
                                WHEN keyword LIKE '%URL:%' THEN 'url'
                                ELSE 'keyword'
                            END as embedding_type,
                            created_at
                            FROM embeddings''')
            
            cursor.execute('DROP TABLE embeddings')
            cursor.execute('ALTER TABLE embeddings_new RENAME TO embeddings')
            
            conn.commit()
            st.success("Database migration completed!")
    except Exception as e:
        st.error(f"Error during database migration: {e}")
    finally:
        conn.close()

# Run migration check at startup
check_and_migrate_database()

# Streamlit app
st.title("Embedding Generator")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["Generate Embedding", "URL Content Embeddings", "Manage Database", "Compare Embeddings"])

with tab1:
    st.write("Enter a keyword to generate its vector embedding using OpenAI's API.")
    
    keyword = st.text_input("Enter a keyword:")

    if 'current_embedding' not in st.session_state:
        st.session_state.current_embedding = None

    if st.button("Generate Embedding"):
        if keyword:
            try:
                response = client.embeddings.create(
                    model="text-embedding-ada-002",
                    input=keyword
                )
                embedding = response.data[0].embedding
                st.session_state.current_embedding = embedding
                st.success("Embedding generated successfully!")
                st.write("Embedding vector:")
                st.code(str(embedding), language='python')

                df = pd.DataFrame([embedding])
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download as CSV",
                    data=csv,
                    file_name=f"{keyword}_embedding.csv",
                    mime="text/csv"
                )

                json_data = json.dumps({"keyword": keyword, "embedding": embedding})
                st.download_button(
                    label="Download as JSON",
                    data=json_data,
                    file_name=f"{keyword}_embedding.json",
                    mime="application/json"
                )

            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter a keyword.")

    if st.session_state.current_embedding is not None:
        if st.button("Save to Database"):
            save_to_sqlite(keyword, st.session_state.current_embedding)
            st.session_state.current_embedding = None

with tab2:
    st.write("Enter a URL to generate embeddings for its content sections.")
    url = st.text_input("Enter URL:")
    
    if st.button("Scrape and Generate Embeddings"):
        if url:
            try:
                sections = scrape_and_split_content(url)
                if sections:
                    st.success(f"Found {len(sections)} content sections!")
                    
                    # Store sections and URL in session state
                    st.session_state['current_sections'] = sections
                    st.session_state['current_url'] = url
                    
                    # Generate all embeddings first
                    all_embeddings = []
                    for section in sections:
                        response = client.embeddings.create(
                            model="text-embedding-ada-002",
                            input=section['content']
                        )
                        all_embeddings.append(response.data[0].embedding)
                    
                    # Store embeddings in session state
                    st.session_state['current_embeddings'] = all_embeddings
                    
                    # Display individual sections
                    for i, (section, embedding) in enumerate(zip(sections, all_embeddings)):
                        with st.expander(f"Section {i+1}: {section['title']}"):
                            st.write("Content:")
                            st.write(section['content'])
                            st.write("Embedding generated successfully!")
                            
                            url_keyword = f"{section['title']} (URL: {url})"
                            
                            if st.button("Save to Database", key=f"save_{i}"):
                                try:
                                    save_to_sqlite(url_keyword, embedding, embedding_type="url")
                                    st.success(f"Successfully saved section {i+1} to database!")
                                except Exception as save_error:
                                    st.error(f"Error saving to database: {save_error}")
                            
                            st.write(f"Embedding length: {len(embedding)}")
                else:
                    st.warning("No content sections found on the page.")
            except Exception as e:
                st.error(f"Error processing URL: {e}")
        else:
            st.warning("Please enter a URL.")

    # Add Save All button outside the if statement
    if 'current_sections' in st.session_state and 'current_embeddings' in st.session_state:
        if st.button("Save All Sections to Database", key="save_all"):
            try:
                for section, embedding in zip(st.session_state['current_sections'], 
                                           st.session_state['current_embeddings']):
                    url_keyword = f"{section['title']} (URL: {st.session_state['current_url']})"
                    save_to_sqlite(url_keyword, embedding, embedding_type="url")
                st.success("All sections saved successfully!")
                st.rerun()
            except Exception as save_all_error:
                st.error(f"Error saving all sections: {save_all_error}")

        # Display saved embeddings
    st.subheader("Saved URL Embeddings")
    try:
        url_embeddings = get_embeddings_by_type("url")
        st.write(f"Found {len(url_embeddings)} saved embeddings")
        
        if url_embeddings:
            for i, (keyword, embedding, embedding_type, _) in enumerate(url_embeddings):
                with st.expander(f"📄 {keyword}"):
                    st.write(f"Embedding type: {embedding_type}")
                    st.write(f"Embedding length: {len(embedding)}")
                    if st.button("Delete", key=f"delete_saved_{i}"):
                        delete_from_sqlite(keyword)
                        st.rerun()
        else:
            st.write("No URL embeddings saved yet.")
    except Exception as load_error:
        st.error(f"Error loading saved embeddings: {load_error}")

with tab3:
    st.subheader("Manage Saved Embeddings")
    
    embedding_type_filter = st.selectbox(
        "Show embeddings of type:",
        ["All", "Keyword", "URL"]
    )

    if embedding_type_filter == "All":
        saved_embeddings = get_all_embeddings()
    else:
        saved_embeddings = get_embeddings_by_type(embedding_type_filter.lower())

    if saved_embeddings:
        sort_by = st.selectbox(
            "Sort embeddings by:",
            ["Keyword (A-Z)", "Keyword (Z-A)", "Date (Newest)", "Date (Oldest)"]
        )
        
        if sort_by == "Keyword (A-Z)":
            saved_embeddings.sort(key=lambda x: x[0].lower())
        elif sort_by == "Keyword (Z-A)":
            saved_embeddings.sort(key=lambda x: x[0].lower(), reverse=True)
        elif sort_by == "Date (Newest)":
            saved_embeddings.sort(key=lambda x: x[3], reverse=True)
        else:  # Date (Oldest)
            saved_embeddings.sort(key=lambda x: x[3])

        search = st.text_input("Search keywords:", "")
        filtered_embeddings = [
            emb for emb in saved_embeddings 
            if search.lower() in emb[0].lower()
        ]
        
        st.write(f"Showing {len(filtered_embeddings)} embeddings")
        
        if st.button("Delete All Embeddings"):
            if st.session_state.get('confirm_delete_all', False):
                delete_all_embeddings()
                st.session_state.confirm_delete_all = False
                st.rerun()
            else:
                st.session_state.confirm_delete_all = True
                st.warning("Click again to confirm deleting all embeddings")
        
        for keyword, embedding, embedding_type, timestamp in filtered_embeddings:
            with st.expander(f"📌 {keyword} ({embedding_type})"):
                st.write(f"**Created:** {timestamp}")
                st.code(str(embedding), language='python')
                
                col1, col2, col3 = st.columns([2, 2, 1])
                with col1:
                    df = pd.DataFrame([embedding])
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"{keyword}_embedding.csv",
                        mime="text/csv",
                        key=f"csv_{keyword}"
                    )
                with col2:
                    json_data = json.dumps({"keyword": keyword, "embedding": embedding})
                    st.download_button(
                        label="Download JSON",
                        data=json_data,
                        file_name=f"{keyword}_embedding.json",
                        mime="application/json",
                        key=f"json_{keyword}"
                    )
                with col3:
                    if st.button("Delete", key=f"delete_{keyword}"):
                        delete_from_sqlite(keyword)
                        st.rerun()
    else:
        st.write("No embeddings saved yet.")

with tab4:
    st.subheader("Compare Embeddings")
    
    keyword_embeddings = get_embeddings_by_type("keyword")
    keyword_options = [emb[0] for emb in keyword_embeddings]
    
    url_embeddings = get_embeddings_by_type("url")
    url_groups = {}
    for keyword, embedding, embedding_type, timestamp in url_embeddings:
        url_part = keyword.split("URL: ")[-1].strip(")")
        if url_part not in url_groups:
            url_groups[url_part] = []
        url_groups[url_part].append((keyword, embedding, timestamp))
    
    if keyword_options and url_groups:
        col1, col2 = st.columns(2)
        
        with col1:
            selected_keyword = st.selectbox(
                "Select Keyword Embedding",
                keyword_options
            )
            
        with col2:
            selected_url = st.selectbox(
                "Select URL",
                list(url_groups.keys())
            )
        
        if st.button("Calculate Similarity"):
            keyword_embedding = next(emb[1] for emb in keyword_embeddings if emb[0] == selected_keyword)
            
            url_embs = url_groups[selected_url]
            embeddings_array = np.vstack([emb[1] for emb in url_embs])
            average_embedding = np.mean(embeddings_array, axis=0)
            
            norm = np.linalg.norm(average_embedding)
            if norm > 0:
                normalized_url_embedding = average_embedding / norm
            else:
                normalized_url_embedding = average_embedding
            
            keyword_norm = np.linalg.norm(keyword_embedding)
            if keyword_norm > 0:
                normalized_keyword_embedding = keyword_embedding / keyword_norm
            else:
                normalized_keyword_embedding = keyword_embedding
            
            overall_similarity = np.dot(normalized_keyword_embedding, normalized_url_embedding)
            
            st.write("**Overall Cosine Similarity Score:**")
            st.write(f"{overall_similarity:.4f}")
            
            save_similarity(selected_keyword, selected_url, overall_similarity)
            
            show_full_analysis(selected_keyword, selected_url, keyword_embedding, url_embs, overall_similarity)

    st.subheader("Saved Similarity Scores")
    similarities = get_all_similarities()
    
    if similarities:
        for keyword_text, url, score, timestamp in similarities:
            with st.expander(f"📊 {keyword_text} vs {url}"):
                st.write(f"**Score:** {score:.4f}")
                st.write(f"**Calculated:** {timestamp}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("Delete", key=f"delete_sim_{keyword_text}_{url}"):
                        conn = setup_similarity_table()
                        cursor = conn.cursor()
                        cursor.execute('DELETE FROM similarities WHERE keyword_text = ? AND url = ?', 
                                     (keyword_text, url))
                        conn.commit()
                        conn.close()
                        st.rerun()
                
                with col2:
                    if st.button("See Full Analysis", key=f"analysis_{keyword_text}_{url}"):
                        keyword_embedding = next(emb[1] for emb in keyword_embeddings if emb[0] == keyword_text)
                        url_embs = url_groups[url]
                        show_full_analysis(keyword_text, url, keyword_embedding, url_embs, score)
    else:
        st.write("No similarity scores saved yet.")

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

import requests
from bs4 import BeautifulSoup

def scrape_and_split_content(url):
    """Scrape webpage and split content by headers"""
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
        
        # Remove <script>, <style>, and <nav> elements
        for element in soup.find_all(['script', 'style', 'nav']):
            element.decompose()
        
        # Remove the element with id 'headercontent'
        header_content = soup.find(id='headercontent')
        if header_content:
            header_content.decompose()
            
        sections = []
        current_section = {"title": "", "content": []}
        
        main_content = soup.find('body')
        if main_content:
            elements = main_content.find_all(['h1', 'h2', 'h3', 'p'])
            
            for element in elements:
                if element.name in ['h1', 'h2', 'h3']:
                    if current_section["content"]:
                        sections.append({
                            "title": current_section["title"],
                            "content": " ".join(current_section["content"])
                        })
                    current_section = {
                        "title": element.get_text(strip=True),
                        "content": []
                    }
                elif element.name == 'p':
                    text = element.get_text(strip=True)
                    if text:
                        current_section["content"].append(text)
            
            if current_section["content"]:
                sections.append({
                    "title": current_section["title"],
                    "content": " ".join(current_section["content"])
                })
                
        return sections
    except Exception as e:
        raise Exception(f"Error scraping URL: {str(e)}")


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
st.title("VectorRank")

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
            save_to_sqlite(keyword, st.session_state.current_embedding, "keyword")

    st.subheader("Saved Keyword Embeddings")
    keyword_embeddings = get_embeddings_by_type("keyword")
    
    if keyword_embeddings:
        st.write(f"Found {len(keyword_embeddings)} saved keyword embeddings")
        
        keyword_embeddings.sort(key=lambda x: x[3], reverse=True)
        
        for keyword, embedding, embedding_type, timestamp in keyword_embeddings:
            with st.expander(f"📌 {keyword}"):
                st.write(f"Created: {timestamp}")
                st.code(str(embedding), language='python')
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    df = pd.DataFrame([embedding])
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"{keyword}_embedding.csv",
                        mime="text/csv",
                        key=f"csv_keyword_{keyword}"
                    )
                with col2:
                    json_data = json.dumps({
                        "keyword": keyword,
                        "embedding": embedding
                    })
                    st.download_button(
                        label="Download JSON",
                        data=json_data,
                        file_name=f"{keyword}_embedding.json",
                        mime="application/json",
                        key=f"json_keyword_{keyword}"
                    )
                with col3:
                    if st.button("Delete", key=f"delete_keyword_{keyword}"):
                        delete_from_sqlite(keyword)
                        st.rerun()
    else:
        st.write("No keyword embeddings saved yet.")

with tab2:
    st.write("Generate embeddings from webpage content")
    url = st.text_input("Enter URL:")
    
    if st.button("Scrape and Generate Embeddings"):
        if url:
            try:
                with st.spinner("Scraping webpage..."):
                    sections = scrape_and_split_content(url)
                
                st.success(f"Found {len(sections)} sections")
                
                st.session_state.sections = sections
                st.session_state.url = url
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
        else:
            st.warning("Please enter a URL")
    
    if hasattr(st.session_state, 'sections'):
        if st.button("Generate All Embeddings"):
            try:
                with st.spinner("Generating embeddings for all sections..."):
                    for i, section in enumerate(st.session_state.sections):
                        response = client.embeddings.create(
                            model="text-embedding-ada-002",
                            input=section['content']
                        )
                        embedding = response.data[0].embedding
                        st.session_state[f'embedding_{i}'] = embedding
                    st.success("Generated embeddings for all sections!")
            except Exception as e:
                st.error(f"Error generating embeddings: {str(e)}")

        if st.button("Save All Embeddings"):
            try:
                saved_count = 0
                for i, section in enumerate(st.session_state.sections):
                    if f'embedding_{i}' in st.session_state:
                        save_to_sqlite(
                            f"{section['title']} (URL: {st.session_state.url})",
                            st.session_state[f'embedding_{i}'],
                            "url"
                        )
                        saved_count += 1
                if saved_count > 0:
                    st.success(f"Saved {saved_count} embeddings to database!")
                else:
                    st.warning("No embeddings to save. Generate embeddings first.")
            except Exception as e:
                st.error(f"Error saving embeddings: {str(e)}")

        for i, section in enumerate(st.session_state.sections):
            with st.expander(f"Section {i+1}: {section['title']}"):
                st.write("Content preview:")
                st.write(section['content'][:200] + "...")
                
                generate_key = f"generate_{i}"
                if st.button("Generate Embedding", key=generate_key):
                    try:
                        response = client.embeddings.create(
                            model="text-embedding-ada-002",
                            input=section['content']
                        )
                        embedding = response.data[0].embedding
                        st.session_state[f'embedding_{i}'] = embedding
                        
                        st.write("Embedding vector:")
                        st.code(str(embedding), language='python')
                        
                        col1, col2, col3 = st.columns([2, 2, 2])
                        
                        with col1:
                            if st.button("Save to Database", key=f"save_{i}"):
                                save_to_sqlite(
                                    f"{section['title']} (URL: {st.session_state.url})", 
                                    embedding,
                                    "url"
                                )
                        
                        with col2:
                            df = pd.DataFrame([embedding])
                            csv = df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="Download CSV",
                                data=csv,
                                file_name=f"section_{i+1}_embedding.csv",
                                mime="text/csv",
                                key=f"csv_section_{i}"
                            )
                        
                        with col3:
                            json_data = json.dumps({
                                "title": section['title'],
                                "url": st.session_state.url,
                                "embedding": embedding
                            })
                            st.download_button(
                                label="Download JSON",
                                data=json_data,
                                file_name=f"section_{i+1}_embedding.json",
                                mime="application/json",
                                key=f"json_section_{i}"
                            )
                    
                    except Exception as e:
                        st.error(f"Error generating embedding: {str(e)}")
                
                if f'embedding_{i}' in st.session_state:
                    st.write("Existing embedding:")
                    st.code(str(st.session_state[f'embedding_{i}']), language='python')

    st.subheader("Saved URL Embeddings")
    url_embeddings = get_embeddings_by_type("url")
    
    if url_embeddings:
        st.write(f"Found {len(url_embeddings)} saved URL embeddings")
        
        url_groups = {}
        for keyword, embedding, embedding_type, timestamp in url_embeddings:
            url_part = keyword.split("URL: ")[-1].strip(")")
            if url_part not in url_groups:
                url_groups[url_part] = []
            url_groups[url_part].append((keyword, embedding, timestamp))
        
        for url_key, embeddings in url_groups.items():
            with st.expander(f"📌 {url_key}"):
                st.write(f"**{len(embeddings)} sections**")
                
                embeddings_array = np.vstack([emb[1] for emb in embeddings])
                average_embedding = np.mean(embeddings_array, axis=0)
                
                norm = np.linalg.norm(average_embedding)
                if norm > 0:
                    normalized_average_embedding = average_embedding / norm
                else:
                    normalized_average_embedding = average_embedding
                
                st.write("**Average Embedding (Normalized):**")
                st.code(str(normalized_average_embedding), language='python')
                
                col1, col2 = st.columns(2)
                with col1:
                    df = pd.DataFrame([normalized_average_embedding])
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Average (CSV)",
                        data=csv,
                        file_name=f"{url_key}_average_embedding.csv",
                        mime="text/csv",
                        key=f"csv_avg_{url_key}"
                    )
                with col2:
                    json_data = json.dumps({
                        "url": url_key,
                        "average_embedding": normalized_average_embedding.tolist()
                    })
                    st.download_button(
                        label="Download Average (JSON)",
                        data=json_data,
                        file_name=f"{url_key}_average_embedding.json",
                        mime="application/json",
                        key=f"json_avg_{url_key}"
                    )
                
                st.write("---")
                st.write("**Individual Section Embeddings:**")
                
                for i, (keyword, embedding, timestamp) in enumerate(embeddings):
                    st.write("---")
                    st.write(f"**Section {i+1}:** {keyword.split('(URL:')[0]}")
                    st.write(f"Created: {timestamp}")
                    st.code(str(embedding), language='python')
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        df = pd.DataFrame([embedding])
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download CSV",
                            data=csv,
                            file_name=f"section_{i+1}_embedding.csv",
                            mime="text/csv",
                            key=f"csv_url_{url_key}_{i}"
                        )
                    with col2:
                        json_data = json.dumps({
                            "title": keyword,
                            "url": url_key,
                            "embedding": embedding
                        })
                        st.download_button(
                            label="Download JSON",
                            data=json_data,
                            file_name=f"section_{i+1}_embedding.json",
                            mime="application/json",
                            key=f"json_url_{url_key}_{i}"
                        )
                    with col3:
                        if st.button("Delete", key=f"delete_url_{url_key}_{i}"):
                            delete_from_sqlite(keyword)
                            st.rerun()
    else:
        st.write("No URL embeddings saved yet.")

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
            
            similarity = np.dot(normalized_keyword_embedding, normalized_url_embedding)
            
            st.write("**Cosine Similarity Score:**")
            st.write(f"{similarity:.4f}")
            
            save_similarity(selected_keyword, selected_url, similarity)
        
        st.subheader("Saved Similarity Scores")
        similarities = get_all_similarities()
        
        if similarities:
            for keyword_text, url, score, timestamp in similarities:
                with st.expander(f"📊 {keyword_text} vs {url}"):
                    st.write(f"**Score:** {score:.4f}")
                    st.write(f"**Calculated:** {timestamp}")
                    
                    if st.button("Delete", key=f"delete_sim_{keyword_text}_{url}"):
                        conn = setup_similarity_table()
                        cursor = conn.cursor()
                        cursor.execute('DELETE FROM similarities WHERE keyword_text = ? AND url = ?', 
                                     (keyword_text, url))
                        conn.commit()
                        conn.close()
                        st.rerun()
        else:
            st.write("No similarity scores saved yet.")
    else:
        st.warning("Please generate both keyword and URL embeddings first.")

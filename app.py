import os
import json
import argparse
import pandas as pd
import logging
import glob
import re
import requests
from flask import Flask, render_template, request, jsonify, redirect, url_for, send_from_directory, flash, session

app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['JSON_SORT_KEYS'] = False
app.secret_key = 'mysecretkey123'  # Needed for flash messages and session

# Global variables to store data
query_data = {}
submission_data = {}
images_to_article = {}
article_to_url = {}
article_to_images = {}
stage1_retrieval = {}
stage1_details = {}  # New: store stage1 detailed entities
article_to_content = {}  # New: store article content
base_dir = None
top_k = 10
csv_dir = "csv_app"
results_dir = "app_results"  # Directory for result sets
queries_json = None
display_mode = "image"  # Default mode: "image" or "stage1"
es_host = "http://localhost:9200"  # Elasticsearch host

def load_queries_json(file_path):
    """Load queries and their entities from JSON file (optional)"""
    global queries_json
    try:
        if not os.path.exists(file_path):
            logging.info(f"Queries JSON file {file_path} not found, skipping entity loading")
            queries_json = {}
            return
            
        with open(file_path, 'r', encoding='utf-8') as f:
            queries_json = json.load(f)
        logging.info(f"Loaded {len(queries_json)} queries from JSON")
    except Exception as e:
        logging.warning(f"Could not load queries JSON: {str(e)}, continuing without entity data")
        queries_json = {}

def get_query_entities(query_id):
    """Get entities for a query from queries.json (optional)"""
    if not queries_json:
        return []
    
    try:
        # Handle if queries_json is a list of queries
        if isinstance(queries_json, list):
            for query in queries_json:
                if isinstance(query, dict) and query.get('query_id') == query_id:
                    return query.get('entities', [])
        # Handle if queries_json is a dict with query_id as keys
        elif isinstance(queries_json, dict):
            if query_id in queries_json:
                query_data = queries_json[query_id]
                if isinstance(query_data, dict):
                    return query_data.get('entities', [])
                elif isinstance(query_data, list):
                    return query_data
    except Exception as e:
        logging.debug(f"Error getting entities for query {query_id}: {str(e)}")
    
    return []

def get_available_csv_files():
    """Get all CSV files from the csv_app directory (legacy support)"""
    csv_files = {}
    
    # Ensure directory exists
    os.makedirs(csv_dir, exist_ok=True)
    
    # Get all CSV files
    all_csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
    all_csv_files = [os.path.basename(f) for f in all_csv_files]
    
    # Let user choose any CSV file for any purpose
    csv_files['submission'] = all_csv_files.copy()
    csv_files['stage1'] = all_csv_files.copy()
    csv_files['query'] = all_csv_files.copy()
    
    return csv_files

def load_data_from_result_set(result_set_name):
    """Load data from a complete result set directory"""
    global query_data, submission_data, images_to_article, article_to_url, article_to_images, stage1_retrieval, stage1_details, base_dir
    
    result_path = os.path.join(results_dir, result_set_name)
    if not os.path.exists(result_path):
        raise Exception(f"Result set '{result_set_name}' not found")
    
    files = analyze_result_set(result_path)
    
    # Load query data from CSV (check session if available, default to public)
    try:
        dataset_type = session.get('dataset_type', 'public')
    except RuntimeError:
        # Outside request context, use default
        dataset_type = 'public'
    load_query_data(dataset_type)
    
    # Load submission data (stage1, required)
    if files['submission']:
        submission_file_path = os.path.join(result_path, files['submission'])
        if os.path.exists(submission_file_path):
            submission_df = pd.read_csv(submission_file_path)
            stage1_retrieval = {}
            for _, row in submission_df.iterrows():
                query_id = row['query_id']
                articles = []
                for i in range(1, top_k + 1):
                    col = f'article_id_{i}'
                    if col in row and not pd.isna(row[col]) and row[col] != "#":
                        articles.append(row[col])
                stage1_retrieval[query_id] = articles
            logging.info(f"Loaded stage1 data for {len(stage1_retrieval)} queries")
    
    # Load track2 submission data (images, optional)
    if files['track2_submission']:
        track2_file_path = os.path.join(result_path, files['track2_submission'])
        if os.path.exists(track2_file_path):
            track2_df = pd.read_csv(track2_file_path)
            submission_data = {}
            for _, row in track2_df.iterrows():
                query_id = row['query_id']
                images = []
                for i in range(1, 11):
                    col = f'image_id_{i}'
                    if col in row and not pd.isna(row[col]) and row[col] != "#":
                        images.append(row[col])
                submission_data[query_id] = images
            logging.info(f"Loaded track2 data for {len(submission_data)} queries")
    
    # Load stage1 details (entity matching, optional)
    if files['stage1_details']:
        details_file_path = os.path.join(result_path, files['stage1_details'])
        if os.path.exists(details_file_path):
            try:
                with open(details_file_path, 'r', encoding='utf-8') as f:
                    stage1_details = json.load(f)
                logging.info(f"Loaded stage1 details for {len(stage1_details)} queries")
            except Exception as e:
                logging.warning(f"Could not load stage1 details: {str(e)}, continuing without entity matching data")
                stage1_details = {}
        else:
            logging.info(f"Stage1 details file not found, continuing without entity matching data")
            stage1_details = {}
    else:
        logging.info(f"No stage1 details file specified, continuing without entity matching data")
        stage1_details = {}
    
    # Load supporting JSON files (fallback to root directory)
    load_supporting_files()

def load_query_data(dataset_type="public"):
    """Load query data from CSV file based on dataset type"""
    global query_data
    
    # Determine query file based on dataset type
    if dataset_type == "private":
        query_files = [
            "csv_app/query_private.csv",
            "query_private.csv"
        ]
    else:  # public
        query_files = [
            "csv_app/query_public.csv", 
            "query_public.csv",
            "queries.csv"
        ]
    
    for query_file_path in query_files:
        if os.path.exists(query_file_path):
            query_df = pd.read_csv(query_file_path)
            query_data = {}
            for _, row in query_df.iterrows():
                query_id = row['query_id']
                query_text = row['query_text']
                
                # Check for summary and concise fields
                summary = row['summary'] if 'summary' in row else ""
                concise = row['concise'] if 'concise' in row else ""
                
                query_data[query_id] = {
                    'query_text': query_text,
                    'summary': summary,
                    'concise': concise
                }
            logging.info(f"Loaded {len(query_data)} queries from {query_file_path} ({dataset_type} dataset)")
            return
    
    logging.warning(f"No {dataset_type} query CSV file found, query data will be empty")

def load_supporting_files():
    """Load supporting JSON files from root directory"""
    global images_to_article, article_to_url, article_to_images, base_dir
    
    # Load images to article mapping
    images_to_article_file = "database_images_to_article_v.0.1.json"
    if os.path.exists(images_to_article_file):
        with open(images_to_article_file, 'r', encoding='utf-8') as f:
            images_to_article = json.load(f)
        logging.info(f"Loaded images to article mapping")
    else:
        logging.warning(f"Images to article mapping file not found: {images_to_article_file}")
        images_to_article = {}
    
    # Load article to URL mapping
    article_to_url_file = "database_article_to_url.json"
    if os.path.exists(article_to_url_file):
        with open(article_to_url_file, 'r', encoding='utf-8') as f:
            article_to_url = json.load(f)
        logging.info(f"Loaded article to URL mapping")
    else:
        logging.warning(f"Article to URL mapping file not found: {article_to_url_file}")
        article_to_url = {}
    
    # Load article to images mapping
    article_to_images_file = "database_article_to_images_v.0.1.json"
    if os.path.exists(article_to_images_file):
        with open(article_to_images_file, 'r', encoding='utf-8') as f:
            article_to_images = json.load(f)
        logging.info(f"Loaded article to images mapping")
    else:
        logging.warning(f"Article to images mapping file not found: {article_to_images_file}")
        article_to_images = {}
    
    # Set base directory for images (update this path to your actual images directory)
    possible_image_dirs = [
        "D:/database_compressed_images/database_images_compressed90",
        "D:\\database_compressed_images\\database_images_compressed90",
        "./images",
        "./static/images"
    ]
    
    base_dir = None
    for dir_path in possible_image_dirs:
        if os.path.exists(dir_path):
            base_dir = dir_path
            logging.info(f"Using image directory: {base_dir}")
            break
    
    if not base_dir:
        base_dir = possible_image_dirs[0]  # Use first as fallback
        logging.warning(f"Image directory not found, using fallback: {base_dir}")

def load_data(submission_file=None, query_file=None, stage1_file=None, images_to_article_file=None, 
              article_to_url_file=None, article_to_images_file=None, database_dir=None):
    """Load data from specified files or use default paths (legacy support)"""
    global query_data, submission_data, images_to_article, article_to_url, article_to_images, stage1_retrieval, base_dir, top_k
    
    # Set base directory for images
    if database_dir:
        base_dir = database_dir
    
    # Load query data
    if query_file:
        query_file_path = os.path.join(csv_dir, query_file)
        if os.path.exists(query_file_path):
            query_df = pd.read_csv(query_file_path)
            query_data = {}
            for _, row in query_df.iterrows():
                query_id = row['query_id']
                query_text = row['query_text']
                
                # Check for summary and concise fields
                summary = row['summary'] if 'summary' in row else ""
                concise = row['concise'] if 'concise' in row else ""
                
                query_data[query_id] = {
                    'query_text': query_text,
                    'summary': summary,
                    'concise': concise
                }
            logging.info(f"Loaded {len(query_data)} queries from {query_file}")
    
    # Load submission data
    if submission_file:
        submission_file_path = os.path.join(csv_dir, submission_file)
        if os.path.exists(submission_file_path):
            submission_df = pd.read_csv(submission_file_path)
            submission_data = {}
            for _, row in submission_df.iterrows():
                query_id = row['query_id']
                images = []
                for i in range(1, 11):  # Assuming image_id_1 to image_id_10
                    col = f'image_id_{i}'
                    if col in row and not pd.isna(row[col]) and row[col] != "#":
                        images.append(row[col])
                submission_data[query_id] = images
            logging.info(f"Loaded submission data for {len(submission_data)} queries")
    
    # Load stage1 retrieval if provided
    if stage1_file:
        stage1_file_path = os.path.join(csv_dir, stage1_file)
        if os.path.exists(stage1_file_path):
            stage1_df = pd.read_csv(stage1_file_path)
            stage1_retrieval = {}
            for _, row in stage1_df.iterrows():
                query_id = row['query_id']
                articles = []
                for i in range(1, top_k + 1):
                    col = f'article_id_{i}'
                    if col in row and not pd.isna(row[col]) and row[col] != "#":
                        articles.append(row[col])
                stage1_retrieval[query_id] = articles
            logging.info(f"Loaded stage1 retrieval data for {len(stage1_retrieval)} queries")
    
    # Load JSON mappings
    if images_to_article_file and os.path.exists(images_to_article_file):
        with open(images_to_article_file, 'r', encoding='utf-8') as f:
            images_to_article = json.load(f)
        logging.info(f"Loaded images to article mapping")
    
    if article_to_url_file and os.path.exists(article_to_url_file):
        with open(article_to_url_file, 'r', encoding='utf-8') as f:
            article_to_url = json.load(f)
        logging.info(f"Loaded article to URL mapping")
    
    if article_to_images_file and os.path.exists(article_to_images_file):
        with open(article_to_images_file, 'r', encoding='utf-8') as f:
            article_to_images = json.load(f)
        logging.info(f"Loaded article to images mapping")

def get_adjacent_queries(current_query_id):
    """Get the previous and next query IDs based on the current query ID."""
    if not query_data:
        return None, None
        
    sorted_query_ids = sorted(list(query_data.keys()))
    if current_query_id not in sorted_query_ids:
        return None, None
        
    current_idx = sorted_query_ids.index(current_query_id)
    prev_query_id = sorted_query_ids[current_idx - 1] if current_idx > 0 else None
    next_query_id = sorted_query_ids[current_idx + 1] if current_idx < len(sorted_query_ids) - 1 else None
    
    return prev_query_id, next_query_id

@app.route('/switch_dataset', methods=['POST'])
def switch_dataset():
    """Switch between public and private datasets"""
    dataset_type = request.form.get('dataset_type', 'public')
    
    # Save dataset choice to session
    session['dataset_type'] = dataset_type
    
    # Reload query data with new dataset
    load_query_data(dataset_type)
    
    flash(f"Switched to {dataset_type.title()} dataset ({len(query_data)} queries)", "success")
    return redirect('/')

@app.route('/')
def index():
    # Get available result sets and CSV files (for backward compatibility)
    result_sets = get_available_result_sets()
    csv_files = get_available_csv_files()
    
    # Get current selections from session
    current_result_set = session.get('result_set', '')
    current_submission = session.get('submission_file', '')
    current_query = session.get('query_file', '')
    current_stage1 = session.get('stage1_file', '')
    current_mode = session.get('display_mode', 'stage1')  # Default to stage1 now
    current_dataset = session.get('dataset_type', 'public')  # Default to public
    
    if query_data:
        query_ids = sorted(list(query_data.keys()))
    else:
        query_ids = []
        
    return render_template('index.html', 
                          query_ids=query_ids,
                          result_sets=result_sets,
                          csv_files=csv_files,
                          current_result_set=current_result_set,
                          current_submission=current_submission,
                          current_query=current_query,
                          current_stage1=current_stage1,
                          current_mode=current_mode,
                          current_dataset=current_dataset)

@app.route('/set_result_set', methods=['POST'])
def set_result_set():
    """Set the result set to use and load data"""
    global display_mode
    
    result_set = request.form.get('result_set', '')
    display_mode = request.form.get('display_mode', 'stage1')
    
    # Save selections to session
    session['result_set'] = result_set
    session['display_mode'] = display_mode
    
    # Load queries JSON (optional)
    queries_json_file = "queries.json"
    load_queries_json(queries_json_file)
    
    # Load database articles first
    load_database_articles()
    
    # Load data from result set
    try:
        if result_set:
            load_data_from_result_set(result_set)
            flash(f"Result set '{result_set}' loaded successfully!", "success")
        else:
            flash("Please select a result set", "warning")
    except Exception as e:
        flash(f"Error loading result set: {str(e)}", "error")
    
    return redirect(url_for('index'))

@app.route('/set_files', methods=['POST'])
def set_files():
    """Set the files to use and load data (legacy support)"""
    global display_mode
    
    submission_file = request.form.get('submission_file', '')
    query_file = request.form.get('query_file', '')
    stage1_file = request.form.get('stage1_file', '')
    display_mode = request.form.get('display_mode', 'image')
    
    # Save selections to session
    session['submission_file'] = submission_file
    session['query_file'] = query_file
    session['stage1_file'] = stage1_file
    session['display_mode'] = display_mode
    
    # Set file paths
    images_to_article_file = "database_images_to_article_v.0.1.json"
    article_to_url_file = "database_article_to_url.json"
    article_to_images_file = "database_article_to_images_v.0.1.json"
    database_dir = "D:\database_compressed_images\database_images_compressed90"
    
    # Load queries JSON (optional)
    queries_json_file = "queries.json"
    load_queries_json(queries_json_file)
    
    # Load database articles first
    load_database_articles()
    
    # Load data
    try:
        load_data(
            submission_file=submission_file, 
            query_file=query_file,
            stage1_file=stage1_file,
            images_to_article_file=images_to_article_file,
            article_to_url_file=article_to_url_file,
            article_to_images_file=article_to_images_file,
            database_dir=database_dir
        )
        flash("Data loaded successfully!", "success")
    except Exception as e:
        flash(f"Error loading data: {str(e)}", "error")
    
    return redirect(url_for('index'))

@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory(base_dir, filename)

@app.route('/query/<query_id>')
def show_query(query_id):
    if query_id not in query_data:
        return "Query ID not found", 404
    
    query_info = query_data[query_id]
    query_text = query_info['query_text']
    summary = query_info.get('summary', '')
    concise = query_info.get('concise', '')
    
    # Get query entities
    query_entities = get_query_entities(query_id)
    
    # Get adjacent query IDs for navigation
    prev_query_id, next_query_id = get_adjacent_queries(query_id)
    
    # Check display mode
    if display_mode == 'stage1':
        return redirect(url_for('stage1_view', query_id=query_id))
    
    # For image mode, continue with normal view
    image_ids = submission_data.get(query_id, [])
    
    # Get article info for each image
    image_info = []
    for rank, img_id in enumerate(image_ids, 1):
        article_id = images_to_article.get(img_id, "Unknown")
        article_url = article_to_url.get(article_id, "#")
        img_filename = img_id + ".jpg"
        img_url = url_for('serve_image', filename=img_filename)
        
        # Get article content info
        content_info = article_to_content.get(article_id, {})
        
        image_info.append({
            "rank": rank,
            "image_id": img_id,
            "article_id": article_id,
            "article_url": article_url,
            "image_path": img_url,
            "article_title": content_info.get('title', ''),
            "article_preview": content_info.get('preview', ''),
            "article_date": content_info.get('date', '')
        })
    
    return render_template(
        'query.html', 
        query_id=query_id, 
        query_text=query_text, 
        summary=summary,
        concise=concise,
        query_entities=query_entities,
        image_info=image_info,
        prev_query_id=prev_query_id,
        next_query_id=next_query_id,
        display_mode=display_mode
    )

@app.route('/article_view/<query_id>')
def article_view(query_id):
    if query_id not in query_data:
        return "Query ID not found", 404
    
    query_info = query_data[query_id]
    query_text = query_info['query_text']
    summary = query_info.get('summary', '')
    concise = query_info.get('concise', '')
    
    # Get query entities
    query_entities = get_query_entities(query_id)
    
    # Get adjacent query IDs for navigation
    prev_query_id, next_query_id = get_adjacent_queries(query_id)
    
    # Check if we have submission data for this query
    if query_id in submission_data:
        top_images = submission_data[query_id]
    else:
        top_images = []
    
    # Get all relevant articles (either from stage1 if available or from top image results)
    if stage1_retrieval and query_id in stage1_retrieval:
        relevant_articles = stage1_retrieval[query_id]
    else:
        # Extract articles from top images
        relevant_articles = []
        for img_id in top_images:
            article = images_to_article.get(img_id)
            if article and article not in relevant_articles:
                relevant_articles.append(article)
    
    # Build article data structure
    articles_data = []
    for article_id in relevant_articles:
        if article_id not in article_to_images:
            continue
            
        article_images = article_to_images[article_id]
        images_info = []
        
        for img_id in article_images:
            # Check if this image is in top results
            is_top = img_id in top_images
            rank = top_images.index(img_id) + 1 if is_top else None
            img_filename = img_id + ".jpg"
            img_url = url_for('serve_image', filename=img_filename)
            
            images_info.append({
                "image_id": img_id,
                "is_top": is_top,
                "rank": rank,
                "image_path": img_url
            })
        
        articles_data.append({
            "article_id": article_id,
            "url": article_to_url.get(article_id, "#"),
            "images": images_info
        })
    
    return render_template(
        'article_view.html',
        query_id=query_id,
        query_text=query_text,
        summary=summary,
        concise=concise,
        query_entities=query_entities,
        articles_data=articles_data,
        prev_query_id=prev_query_id,
        next_query_id=next_query_id
    )

@app.route('/stage1_view/<query_id>')
def stage1_view(query_id):
    """View showing only stage1 retrieval results with entities"""
    if query_id not in query_data:
        return "Query ID not found", 404
    
    if not stage1_retrieval or query_id not in stage1_retrieval:
        flash("No stage1 retrieval data available for this query", "error")
        return redirect(url_for('show_query', query_id=query_id))
    
    query_info = query_data[query_id]
    query_text = query_info['query_text']
    summary = query_info.get('summary', '')
    concise = query_info.get('concise', '')
    
    # Get query entities
    query_entities = get_query_entities(query_id)
    
    # Get adjacent query IDs for navigation
    prev_query_id, next_query_id = get_adjacent_queries(query_id)
    
    # Get articles from stage1 retrieval
    articles = stage1_retrieval[query_id]
    
    # Get matched entities from stage1_details if available
    stage1_query_data = stage1_details.get(query_id, {}) if stage1_details else {}
    stage1_query_entities = stage1_query_data.get('query_entities', [])
    stage1_articles_data = stage1_query_data.get('articles', [])
    
    # Build article data structure
    articles_data = []
    for rank, article_id in enumerate(articles, 1):
        article_url = article_to_url.get(article_id, "#")
        
        # Get article content info
        content_info = article_to_content.get(article_id, {})
        
        # Find matching article data from stage1_details
        matched_entities = []
        score = 0
        if stage1_articles_data:
            for stage1_article in stage1_articles_data:
                if stage1_article.get('article_id') == article_id:
                    matched_entities = stage1_article.get('entities', [])
                    break
            score = stage1_articles_data[rank-1].get('score', 0) if rank-1 < len(stage1_articles_data) else 0
        
        articles_data.append({
            "rank": rank,
            "article_id": article_id,
            "url": article_url,
            "entities": matched_entities,
            "article_title": content_info.get('title', ''),
            "article_preview": content_info.get('preview', ''),
            "article_date": content_info.get('date', ''),
            "score": score
        })
    
    return render_template(
        'stage1_view.html',
        query_id=query_id,
        query_text=query_text,
        summary=summary,
        concise=concise,
        query_entities=stage1_query_entities if stage1_query_entities else query_entities,
        articles_data=articles_data,
        prev_query_id=prev_query_id,
        next_query_id=next_query_id,
        match_type=stage1_query_data.get('match_type', '')
    )

def get_article_entities_from_es(article_ids, es_host="http://localhost:9200"):
    """Get entities for specific articles from Elasticsearch"""
    if not article_ids:
        return {}
    
    # Prepare search query for multiple articles
    search_query = {
        "query": {
            "terms": {
                "article_id": article_ids
            }
        },
        "_source": ["article_id", "entities"],
        "size": len(article_ids)
    }
    
    try:
        import requests
        response = requests.post(
            f"{es_host}/articles/_search",
            headers={"Content-Type": "application/json"},
            json=search_query,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            hits = result["hits"]["hits"]
            
            # Build dict mapping article_id to entities
            article_entities = {}
            for hit in hits:
                article_data = hit["_source"]
                article_id = article_data["article_id"]
                entities = article_data.get("entities", [])
                article_entities[article_id] = entities
            
            logging.info(f"Retrieved entities for {len(article_entities)} articles from Elasticsearch")
            return article_entities
        else:
            logging.error(f"Elasticsearch error: {response.status_code} - {response.text}")
            return {}
    except Exception as e:
        logging.error(f"Error connecting to Elasticsearch: {str(e)}")
        return {}

def get_article_entities_from_json(article_ids, json_file="articles.json"):
    """Fallback method to get entities from JSON file"""
    if not article_ids or not os.path.exists(json_file):
        return {}
    
    try:
        logging.info(f"Loading entities from {json_file} (fallback method)...")
        article_entities = {}
        
        with open(json_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                if line_num % 10000 == 0:
                    logging.info(f"Processed {line_num} lines...")
                
                try:
                    article_data = json.loads(line.strip())
                    article_id = article_data.get("article_id")
                    
                    if article_id in article_ids:
                        entities = article_data.get("entities", [])
                        article_entities[article_id] = entities
                        
                        # Early exit if we found all articles
                        if len(article_entities) == len(article_ids):
                            break
                            
                except json.JSONDecodeError:
                    continue
        
        logging.info(f"Retrieved entities for {len(article_entities)} articles from JSON file")
        return article_entities
        
    except Exception as e:
        logging.error(f"Error reading JSON file: {str(e)}")
        return {}

def load_database_articles(file_path="database_article_to_alls_optimized.json"):
    """Load article content, metadata from database_article_to_alls.json"""
    global article_to_content, article_to_url, article_to_images
    
    if not os.path.exists(file_path):
        logging.warning(f"Database file {file_path} not found, skipping...")
        return
    
    try:
        logging.info(f"Loading database articles from {file_path}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Update global mappings
        for article_id, article_data in data.items():
            article_to_url[article_id] = article_data.get('url', '#')
            article_to_images[article_id] = article_data.get('images', [])
            
            # Store content (first 500 characters for display)
            content = article_data.get('content', '')
            article_to_content[article_id] = {
                'full': content,
                'preview': content[:500] + '...' if len(content) > 500 else content,
                'title': article_data.get('title', ''),
                'date': article_data.get('date', ''),
                'url': article_data.get('url', '#')
            }
        
        logging.info(f"Loaded {len(data)} articles from database")
    except Exception as e:
        logging.error(f"Error loading database articles: {str(e)}")

def get_available_result_sets():
    """Get all available result sets from results directory"""
    result_sets = {}
    
    # Ensure directory exists
    os.makedirs(results_dir, exist_ok=True)
    
    # Get all subdirectories
    for item in os.listdir(results_dir):
        item_path = os.path.join(results_dir, item)
        if os.path.isdir(item_path):
            result_sets[item] = analyze_result_set(item_path)
    
    return result_sets

def analyze_result_set(result_path):
    """Analyze a result set directory to identify available files"""
    files = {
        'submission': None,      # submission_*.csv (stage1, required)
        'track2_submission': None,  # track2_*.csv (images, optional)
        'stage1_details': None   # stage_1_*.json (entity details, optional)
    }
    
    # Search for files in directory and subdirectories
    for root, dirs, filenames in os.walk(result_path):
        for filename in filenames:
            file_path = os.path.join(root, filename)
            rel_path = os.path.relpath(file_path, result_path)
            
            if filename.startswith('submission_') and filename.endswith('.csv'):
                files['submission'] = rel_path
            elif (filename.startswith('track2_') or 'track2' in filename.lower()) and filename.endswith('.csv'):
                files['track2_submission'] = rel_path
            elif filename.startswith('stage_1_') and filename.endswith('.json'):
                files['stage1_details'] = rel_path
    
    return files

def main():
    global es_host
    parser = argparse.ArgumentParser(description='Image Search Visualizer')
    parser.add_argument('--database_dir', default="D:/database_compressed_images/database_images_compressed90", help='Path to image database directory')
    parser.add_argument('--top_k', type=int, default=10, help='Number of top results to consider')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the app on')
    parser.add_argument('--es_host', default="http://localhost:9200", help='Elasticsearch host URL')
    
    args = parser.parse_args()
    es_host = args.es_host
    
    # Ensure CSV directory exists
    os.makedirs(csv_dir, exist_ok=True)
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    try:
        # Load supporting files and query data on startup
        load_supporting_files()

        # Load query data from CSV (default to public dataset)
        load_query_data("public")

        # Initial data loading will be handled by the web interface
        app.run(debug=True, port=args.port)
    except Exception as e:
        logging.error(f"Error starting app: {str(e)}")
        raise

if __name__ == '__main__':
    main() 
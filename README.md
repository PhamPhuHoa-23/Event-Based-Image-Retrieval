# Professional Search Pipeline

A comprehensive search system combining enhanced entity search with multi-model image retrieval for academic and research purposes.

## System Overview

This pipeline consists of three main components:
1. **Enhanced Entity Search System** - Weighted entity scoring for text retrieval
2. **Multi-Model Image Search** - Vector-based image similarity search
3. **Web Interface** - Interactive result visualization and analysis

## Directory Structure

```
ElasticSearchSystem/
├── CORE SEARCH PIPELINE
│   ├── search_pipeline.py              # Main professional pipeline
│   ├── entity_search_system.py         # Enhanced entity search with weighted scoring
│   └── rrf_rerank.py                   # Ranking and reranking system
│
├── WEB APPLICATION  
│   ├── app.py                          # Main web interface
│   ├── debug_app.py                    # Debugging interface
│   ├── templates/                      # Web templates
│   └── static/                         # Web assets (CSS, JS, images)
│
├── UPLOAD & MONITORING
│   ├── upload_and_clean_articles_streaming.py
│   ├── upload_private_queries.py
│   └── monitor_upload.py
│
├── ELASTICSEARCH NOTEBOOKS
│   ├── elasticsearch_search_notebook.ipynb    # Full-text search with Elasticsearch
│   └── elasticsearch_upload_notebook.ipynb    # Upload data to Elasticsearch
│
└── DATA & CONFIG
    ├── config.json                     # Pipeline configuration
    ├── query_expansion.json            # Query expansion data (1.1MB)
    ├── articles_by_year.json           # Date filtering data (4.8MB)
    ├── database_article_to_images_v.0.1.json  # Article mapping (15MB)
    └── celeb_blacklist.csv             # Celebrity blacklist
```

## Prerequisites

### Required Services
- **Elasticsearch** (version 7.x or 8.x) running on `http://localhost:9200`
- **Qdrant** vector database running on `localhost:6333`

### Python Dependencies
```bash
pip install requests pandas elasticsearch qdrant-client flask jupyter
```

### Data Requirements
- Articles index in Elasticsearch
- Query index in Elasticsearch (public or private)
- Vector embeddings in Qdrant collections
- Article-to-image mapping file

## Getting Started

### 1. Upload Data to Elasticsearch

Use the Jupyter notebooks to upload your data:

**Upload Articles and Queries:**
```bash
jupyter notebook elasticsearch_upload_notebook.ipynb
```

**Test Full-text Search:**
```bash
jupyter notebook elasticsearch_search_notebook.ipynb
```

### 2. Entity Search Pipeline

**Generate submission file using entity search:**
```bash
python entity_search_system.py --mode search_all --output submission_entity.csv --top-k 30
```

Options:
- `--mode`: Choose between "search_all" or "sample"
- `--output`: Output CSV file name
- `--top-k`: Number of articles to retrieve per query
- `--max-queries`: Limit number of queries for testing
- `--use-private`: Use private queries instead of public
- `--use-clean-articles`: Use clean articles index

### 3. Full-text Elasticsearch Search

Run the full-text search notebook to generate another submission file:
- Open `elasticsearch_search_notebook.ipynb`
- Configure your query index and articles index
- Run all cells to generate `submission_fulltext.csv`

### 4. Complete Search Pipeline

**Text Search Only (Entity-based):**
```bash
python search_pipeline.py --text-search-only --json-config config.json
```

**Image Search with Multiple CSV Files:**
```bash
python search_pipeline.py --image-search-only --csv-files submission_entity.csv submission_fulltext.csv --use-voting
```

**Complete End-to-End Pipeline:**
```bash
python search_pipeline.py --json-config config.json --use-voting --debug
```

### 5. Web Interface

**Start the web application:**
```bash
python app.py --host 0.0.0.0 --port 5000 --debug
```

**Access the interface:**
- Open browser to `http://localhost:5000`
- Select result set from `app_results/` directory
- Browse queries and view results

## Configuration

### JSON Config Format
```json
{
  "Database_Initialized_Large": {
    "weight": 1.0,
    "query_collections": [
      {"Query_Initialized_Large": 1.0},
      {"Summary_Initialized_Large": 0.8},
      {"Concise_Initialized_Large": 1.2}
    ]
  }
}
```

### Entity Weight Configuration
The system uses optimized entity weights:
- PERSON: 4.3 (players, fans, crowds)
- CARDINAL: 3.5 (scores, jersey numbers)
- ORG: 3.8 (teams, organizations)
- GPE: 3.1 (countries, cities, venues)
- EVENT: 2.9 (tournaments, matches)

## Workflow Examples

### Complete Research Workflow

1. **Data Preparation:**
   ```bash
   # Upload articles and queries using Jupyter notebooks
   jupyter notebook elasticsearch_upload_notebook.ipynb
   ```

2. **Generate Entity-based Results:**
   ```bash
   python entity_search_system.py --mode search_all --output submission_entity.csv --top-k 30
   ```

3. **Generate Full-text Results:**
   ```bash
   # Run elasticsearch_search_notebook.ipynb to create submission_fulltext.csv
   jupyter notebook elasticsearch_search_notebook.ipynb
   ```

4. **Combine with Image Search:**
   ```bash
   python search_pipeline.py --image-search-only \
     --csv-files submission_entity.csv submission_fulltext.csv \
     --use-voting \
     --config-name "combined_search"
   ```

5. **View Results:**
   ```bash
   python app.py
   # Open http://localhost:5000
   # Select result set from app_results/combined_search_TIMESTAMP/
   ```

### Quick Testing Workflow

1. **Test Entity Search:**
   ```bash
   python entity_search_system.py --mode sample --sample-size 5
   ```

2. **Test Pipeline:**
   ```bash
   python search_pipeline.py --text-search-only --max-queries 10 --debug
   ```

3. **Test Web Interface:**
   ```bash
   python app.py --debug
   ```

## Output Structure

Results are saved in timestamped directories under `app_results/`:

```
app_results/
└── search_YYYYMMDD_HHMMSS/
    ├── submission_entity.csv          # Stage 1 text search results
    ├── stage_1_entity.json           # Detailed search metadata
    ├── track2_images.csv             # Stage 2 image search results
    └── config.txt                    # Pipeline configuration
```

## File Formats

### Submission CSV Format
```csv
query_id,article_id_1,article_id_2,...,article_id_N
Q001,A123,A456,A789,...
Q002,A234,A567,A890,...
```

### Track2 CSV Format
```csv
query_id,image_id_1,image_id_2,...,image_id_10
Q001,IMG001,IMG002,IMG003,...
Q002,IMG011,IMG012,IMG013,...
```

## Advanced Usage

### Custom Entity Weights
Modify `entity_search_system.py` to adjust entity weights:
```python
self.entity_weights = {
    "PERSON": 4.3,
    "CARDINAL": 3.5,
    "ORG": 3.8,
    # Add your custom weights
}
```

### Custom Search Pipeline
```python
from search_pipeline import ProfessionalSearchPipeline

pipeline = ProfessionalSearchPipeline(
    es_host="http://localhost:9200",
    qdrant_host="localhost",
    qdrant_port=6333,
    private_test_mode=False,
    json_config_file="custom_config.json"
)

results = pipeline.run_integrated_cascade_pipeline(
    config_name="custom_search",
    text_top_k=20,
    max_queries=100
)
```

### Web Application Customization
- Modify templates in `templates/` directory
- Update CSS/JS in `static/` directory
- Configure result directories in `app.py`

## Troubleshooting

### Common Issues

**Elasticsearch Connection Error:**
- Verify Elasticsearch is running: `curl http://localhost:9200`
- Check index exists: `curl http://localhost:9200/_cat/indices`

**Qdrant Connection Error:**
- Verify Qdrant is running: `curl http://localhost:6333/collections`
- Check collections exist and have data

**Missing Data Files:**
- Ensure all required JSON files are present
- Check file paths in configuration

**Web Interface Issues:**
- Verify result directories exist in `app_results/`
- Check CSV file formats match expected structure

### Performance Optimization

**Entity Search:**
- Reduce `top_k` for faster results
- Use `max_queries` for testing subsets
- Enable `use_clean_articles` for cleaner data

**Image Search:**
- Adjust `final_top_k` to reduce processing time
- Use `--disable-multi-model` for single model search
- Configure `similarity_threshold` for better precision

## Support

For issues and questions:
1. Check the troubleshooting section
2. Verify all prerequisites are installed
3. Ensure data is properly uploaded to Elasticsearch and Qdrant
4. Check log outputs for specific error messages

## License

This system is designed for academic and research purposes. Please ensure compliance with your institution's data usage policies. 
# SHL Assessment Recommender System

A comprehensive system for recommending SHL assessments based on job descriptions, skills, and requirements. The system includes both a web interface and an API endpoint for easy integration.

## Features

- **Web Interface**: User-friendly interface built with Streamlit
- **API Endpoint**: RESTful API built with FastAPI
- **Semantic Search**: Advanced search using sentence transformers
- **Vector Database**: Efficient storage and retrieval using ChromaDB
- **Data Scraping**: Tools for collecting assessment data from SHL website

## Project Structure

```
shl-final/
├── app.py                 # Streamlit web interface
├── api.py                 # FastAPI endpoint
├── inner_page.py          # Scraper for assessment details
├── outer_page.py          # Scraper for assessment list
├── combined_scraper.py    # Combined scraping functionality
├── database.json         #  Assessment data
├── chroma_db/            # Vector database storage
├── .env                   # Environment variables
├── .venv/                # Virtual environment
├── requirements.txt       # Project dependencies
└── README.md             # Project documentation
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd shl-final
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file with the following variables:
```
GROQ_API_KEY=your_groq_api_key_here
```

## Usage

### Web Interface

Run the Streamlit app:
```bash
streamlit run app.py
```

The web interface provides:
- Text input for job descriptions or requirements
- Example queries for quick testing
- Detailed assessment information
- Relevance scores and explanations

### API Endpoint

Start the FastAPI server:
```bash
uvicorn api:app --reload
```

Make API requests:
```python
import requests

response = requests.post(
    "http://localhost:8000/recommend",
    json={
        "text": "Looking for a Python developer assessment",
        "num_results": 5
    }
)

recommendations = response.json()
```

### Data Scraping

To scrape assessment data:
```bash
# Scrape a single record
python one_record_scraper.py

# Scrape multiple records
python combined_scraper.py
```

## API Documentation

### Endpoints

1. `GET /`
   - Returns API status message

2. `POST /recommend`
   - Request body:
     ```json
     {
       "text": "string",
       "num_results": integer (optional, default: 5)
     }
     ```
   - Response: List of assessment recommendations

### Response Format

```json
[
  {
    "test_id": "string",
    "name": "string",
    "url": "string",
    "description": "string",
    "remote_testing": "Yes/No",
    "adaptive_irt": "Yes/No",
    "assessment_length": "string",
    "test_type": ["string"],
    "languages": "string",
    "job_levels": "string",
    "category": "string",
    "similarity_score": float
  }
]
```

## Features

### Web Interface
- Interactive search interface
- Detailed assessment information display
- Support for job descriptions and requirements
- Relevance scoring
- Test type explanations
- Example queries

### API
- RESTful endpoint
- JSON response format
- Configurable number of results
- Error handling
- Swagger documentation

### Data Collection
- Automated scraping tools
- Data validation
- Error handling
- Rate limiting
- Data persistence

## Dependencies

- FastAPI
- Streamlit
- ChromaDB
- Sentence Transformers
- Beautiful Soup
- Requests
- Pydantic
- NumPy
- Python-dotenv
- Groq

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
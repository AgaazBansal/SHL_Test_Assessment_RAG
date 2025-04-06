from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import json
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
import numpy as np
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="SHL Assessment Recommender API")

# Constants
MODEL_NAME = "all-MiniLM-L6-v2"
DB_PATH = "./chroma_db"
JSON_PATH = "./dummy_dataset.json"

# Set up sentence transformer
sentence_transformer = SentenceTransformer(MODEL_NAME)

# Set up embedding function for ChromaDB
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=MODEL_NAME)

# Initialize ChromaDB
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection(
    name="shl_assessments",
    embedding_function=embedding_function
)

class Query(BaseModel):
    text: str
    num_results: Optional[int] = 5

class AssessmentResponse(BaseModel):
    test_id: str
    name: str
    url: str
    description: str
    remote_testing: str
    adaptive_irt: str
    assessment_length: str
    test_type: List[str]
    languages: str
    job_levels: str
    category: str
    similarity_score: float

@app.get("/")
async def root():
    return {"message": "SHL Assessment Recommender API"}

@app.post("/recommend", response_model=List[AssessmentResponse])
async def get_recommendations(query: Query):
    try:
        # Query the vector database
        results = collection.query(
            query_texts=[query.text],
            n_results=query.num_results,
            include=['metadatas', 'distances']
        )
        
        # Format results
        recommendations = []
        for metadata, distance in zip(results['metadatas'][0], results['distances'][0]):
            # Convert distance to similarity score (0-100%)
            similarity_score = 100 * (1 / (1 + np.exp(distance * 2)))
            
            recommendation = AssessmentResponse(
                test_id=metadata['test_id'],
                name=metadata['name'],
                url=metadata['url'],
                description=metadata.get('Description', 'N/A'),
                remote_testing=metadata.get('remote_testing', 'No'),
                adaptive_irt=metadata.get('adaptive_irt', 'No'),
                assessment_length=metadata.get('Assessment length', 'N/A'),
                test_type=metadata.get('Test Type', []),
                languages=metadata.get('Languages', 'N/A'),
                job_levels=metadata.get('Job levels', 'N/A'),
                category=metadata.get('category', 'N/A'),
                similarity_score=round(similarity_score, 2)
            )
            recommendations.append(recommendation)
        
        return recommendations
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 
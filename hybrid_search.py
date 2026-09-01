"""
Qdrant Hybrid Search Example
Demonstrates combining dense embeddings (semantic), sparse embeddings (BM25 keyword),
and late interaction embeddings (ColBERT reranking) for advanced search.
"""

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, models
from fastembed import TextEmbedding, SparseTextEmbedding, LateInteractionTextEmbedding
from example_document import DOCUMENTS, EXAMPLE_QUERIES

# Load environment variables
load_dotenv()

COLLECTION_NAME = "demo-hybrid-search"


def initialize_client():
    """Initialize Qdrant client against the local Docker instance."""
    client = QdrantClient(url="http://localhost:6333", timeout=300)
    print("✓ Connected to Qdrant at http://localhost:6333")
    return client


def initialize_models():
    """Initialize the three embedding models for hybrid search."""
    print("\nLoading embedding models...")
    
    dense_model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
    print("✓ Dense embedding model loaded (all-MiniLM-L6-v2)")
    
    sparse_model = SparseTextEmbedding("Qdrant/bm25")
    print("✓ Sparse embedding model loaded (BM25)")
    
    late_interaction_model = LateInteractionTextEmbedding("colbert-ir/colbertv2.0")
    print("✓ Late interaction model loaded (ColBERTv2.0)")
    
    return dense_model, sparse_model, late_interaction_model


def create_collection(client, dense_model, sparse_model, late_interaction_model):
    """Create collection with multi-vector configuration for hybrid search."""
    
    # Delete collection if it exists
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"\n✓ Deleted existing collection '{COLLECTION_NAME}'")
    except Exception:
        pass
    
    # Generate sample embeddings to get dimensions
    sample_text = ["sample"]
    dense_sample = list(dense_model.embed(sample_text))[0]
    late_sample = list(late_interaction_model.embed(sample_text))[0]
    
    # Create collection with proper configuration
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "all-MiniLM-L6-v2": models.VectorParams(
                size=len(dense_sample),
                distance=models.Distance.COSINE,
            ),
            "colbertv2.0": models.VectorParams(
                size=len(late_sample[0]),
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM,
                ),
                hnsw_config=models.HnswConfigDiff(m=0)  # Disable HNSW for reranking
            ),
        },
        sparse_vectors_config={
            "bm25": models.SparseVectorParams(
                modifier=models.Modifier.IDF
            )
        }
    )
    print(f"✓ Created collection '{COLLECTION_NAME}' with hybrid search configuration")


def index_documents(client, documents, dense_model, sparse_model, late_interaction_model):
    """Generate embeddings and index documents into Qdrant."""
    print(f"\nIndexing {len(documents)} documents...")
    
    # Generate all three types of embeddings
    dense_embeddings = list(dense_model.embed(documents))
    sparse_embeddings = list(sparse_model.embed(documents))
    late_interaction_embeddings = list(late_interaction_model.embed(documents))
    
    # Create points with all embedding types and upload in smaller batches
    batch_size = 3
    total_indexed = 0
    
    for i in range(0, len(documents), batch_size):
        batch_points = []
        for idx in range(i, min(i + batch_size, len(documents))):
            point = PointStruct(
                id=idx,
                vector={
                    "all-MiniLM-L6-v2": dense_embeddings[idx],
                    "bm25": sparse_embeddings[idx].as_object(),
                    "colbertv2.0": late_interaction_embeddings[idx],
                },
                payload={"document": documents[idx]}
            )
            batch_points.append(point)
        
        # Upload batch to Qdrant
        client.upsert(collection_name=COLLECTION_NAME, points=batch_points)
        total_indexed += len(batch_points)
        print(f"  Uploaded batch {i//batch_size + 1}: {total_indexed}/{len(documents)} documents")
    
    print(f"✓ Indexed {total_indexed} documents with dense, sparse, and late interaction embeddings")


def hybrid_query(client, query, dense_model, sparse_model, late_interaction_model, limit=10):
    """
    Execute hybrid search with reranking.
    1. Prefetch results using dense (semantic) and sparse (keyword) search
    2. Rerank combined results using ColBERT late interaction
    """
    # Generate query embeddings
    dense_query_vector = list(dense_model.query_embed(query))[0]
    sparse_query_vector = list(sparse_model.query_embed(query))[0]
    late_interaction_query_vector = list(late_interaction_model.query_embed(query))[0]
    
    # Set up prefetch for hybrid search
    prefetch = [
        models.Prefetch(
            query=dense_query_vector,
            using="all-MiniLM-L6-v2",
            limit=20,
        ),
        models.Prefetch(
            query=models.SparseVector(**sparse_query_vector.as_object()),
            using="bm25",
            limit=20,
        ),
    ]
    
    # Execute search with ColBERT reranking
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=prefetch,
        query=late_interaction_query_vector,
        using="colbertv2.0",
        with_payload=True,
        limit=limit,
    )
    
    return results


def display_results(query, results):
    """Display search results in a formatted way."""
    print(f"\n{'='*80}")
    print(f"Query: {query}")
    print(f"{'='*80}")
    
    if not results.points:
        print("No results found.")
        return
    
    for rank, point in enumerate(results.points, 1):
        doc = point.payload["document"]
        score = point.score
        
        # Truncate long documents for display
        doc_preview = doc if len(doc) <= 150 else doc[:147] + "..."
        
        print(f"\nRank {rank} (Score: {score:.4f})")
        print(f"  {doc_preview}")


def main():
    """Main execution flow."""
    try:
        print("="*80)
        print("Qdrant Hybrid Search Demo")
        print("="*80)
        
        # Initialize
        client = initialize_client()
        dense_model, sparse_model, late_interaction_model = initialize_models()
        
        # Create collection
        create_collection(client, dense_model, sparse_model, late_interaction_model)
        
        # Index documents
        index_documents(client, DOCUMENTS, dense_model, sparse_model, late_interaction_model)
        
        # Run example queries
        print("\n" + "="*80)
        print("Running Example Queries")
        print("="*80)
        
        for query in EXAMPLE_QUERIES:
            results = hybrid_query(
                client, query, dense_model, sparse_model, late_interaction_model, limit=3
            )
            display_results(query, results)
        
        print("\n" + "="*80)
        print("Demo completed successfully!")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
    
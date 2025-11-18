"""
🔥 VECTOR SEARCH + RAG SYSTEM
Pinecone vector database with OpenAI embeddings
Semantic search across credit reports and insights
Author: Rick Jefferson Solutions
"""

import os
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import hashlib

try:
    from pinecone import Pinecone, ServerlessSpec
    import openai
    VECTOR_DB_AVAILABLE = True
except ImportError:
    VECTOR_DB_AVAILABLE = False
    print("⚠️  Pinecone/OpenAI not available, using mock vector search")


class VectorSearchService:
    """
    Elite semantic search system using Pinecone + OpenAI embeddings
    Enables RAG (Retrieval Augmented Generation) for credit intelligence
    """
    
    def __init__(self):
        self.pinecone_api_key = os.getenv('PINECONE_API_KEY')
        self.pinecone_environment = os.getenv('PINECONE_ENVIRONMENT', 'us-west1-gcp')
        self.index_name = os.getenv('PINECONE_INDEX_NAME', 'credit-intelligence')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        
        self.pc = None
        self.index = None
        self.initialized = False
        
        if VECTOR_DB_AVAILABLE and self.pinecone_api_key and self.openai_api_key:
            self._initialize()
        else:
            print("⚠️  Vector search using mock mode (no Pinecone/OpenAI keys)")
    
    def _initialize(self):
        """Initialize Pinecone connection"""
        try:
            # Initialize Pinecone
            self.pc = Pinecone(api_key=self.pinecone_api_key)
            
            # Create index if it doesn't exist
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            
            if self.index_name not in existing_indexes:
                self.pc.create_index(
                    name=self.index_name,
                    dimension=3072,  # text-embedding-3-large dimension
                    metric='cosine',
                    spec=ServerlessSpec(
                        cloud='aws',
                        region='us-west-2'
                    )
                )
                print(f"✅ Created Pinecone index: {self.index_name}")
            
            # Connect to index
            self.index = self.pc.Index(self.index_name)
            
            # Set OpenAI API key
            openai.api_key = self.openai_api_key
            
            self.initialized = True
            print(f"✅ Vector search initialized (Pinecone + OpenAI)")
            
        except Exception as e:
            print(f"❌ Vector search initialization failed: {e}")
            self.initialized = False
    
    def generate_embedding(self, text: str, model: str = "text-embedding-3-large") -> List[float]:
        """Generate OpenAI embedding for text"""
        if not self.initialized:
            # Return mock embedding
            return [0.0] * 3072
        
        try:
            response = openai.embeddings.create(
                model=model,
                input=text,
                encoding_format="float"
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"❌ Embedding generation failed: {e}")
            return [0.0] * 3072
    
    def chunk_credit_report(self, credit_report: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk credit report into semantic sections
        Each chunk is a searchable unit with metadata
        """
        chunks = []
        report_id = credit_report.get('report_id', 'unknown')
        user_id = credit_report.get('user_id', 'unknown')
        
        # Chunk 1: Summary
        if 'summary' in credit_report:
            chunks.append({
                'text': f"Credit Report Summary: {json.dumps(credit_report['summary'])}",
                'type': 'summary',
                'report_id': report_id,
                'user_id': user_id,
                'metadata': {
                    'score': credit_report['summary'].get('credit_score'),
                    'utilization': credit_report['summary'].get('credit_utilization'),
                    'accounts': credit_report['summary'].get('total_accounts')
                }
            })
        
        # Chunk 2: Payment History
        if 'payment_history' in credit_report:
            chunks.append({
                'text': f"Payment History: {json.dumps(credit_report['payment_history'])}",
                'type': 'payment_history',
                'report_id': report_id,
                'user_id': user_id,
                'metadata': {
                    'late_payments': credit_report['payment_history'].get('late_payments', 0),
                    'on_time_rate': credit_report['payment_history'].get('on_time_rate', 1.0)
                }
            })
        
        # Chunk 3: Accounts (split by account type)
        if 'accounts' in credit_report:
            for account in credit_report['accounts']:
                chunks.append({
                    'text': f"Credit Account: {json.dumps(account)}",
                    'type': 'account',
                    'report_id': report_id,
                    'user_id': user_id,
                    'metadata': {
                        'account_type': account.get('type'),
                        'creditor': account.get('creditor_name'),
                        'balance': account.get('balance'),
                        'status': account.get('status')
                    }
                })
        
        # Chunk 4: Inquiries
        if 'inquiries' in credit_report:
            chunks.append({
                'text': f"Hard Inquiries: {json.dumps(credit_report['inquiries'])}",
                'type': 'inquiries',
                'report_id': report_id,
                'user_id': user_id,
                'metadata': {
                    'inquiry_count': len(credit_report['inquiries']),
                    'recent_inquiries': credit_report['inquiries'][:5]
                }
            })
        
        # Chunk 5: Negative Items
        if 'negative_items' in credit_report:
            chunks.append({
                'text': f"Negative Items: {json.dumps(credit_report['negative_items'])}",
                'type': 'negative_items',
                'report_id': report_id,
                'user_id': user_id,
                'metadata': {
                    'collections_count': credit_report['negative_items'].get('collections', 0),
                    'derogatory_count': credit_report['negative_items'].get('derogatory_marks', 0)
                }
            })
        
        return chunks
    
    def index_credit_report(
        self,
        credit_report: Dict[str, Any],
        namespace: str = "credit_reports"
    ) -> Dict[str, Any]:
        """
        Index credit report into Pinecone for semantic search
        Returns indexing stats
        """
        if not self.initialized:
            return {
                'success': False,
                'message': 'Vector DB not initialized',
                'chunks_indexed': 0
            }
        
        try:
            # Chunk the report
            chunks = self.chunk_credit_report(credit_report)
            
            # Generate embeddings and index
            vectors = []
            for i, chunk in enumerate(chunks):
                # Generate unique ID
                chunk_id = f"{chunk['report_id']}-{chunk['type']}-{i}"
                
                # Generate embedding
                embedding = self.generate_embedding(chunk['text'])
                
                # Prepare vector
                vectors.append({
                    'id': chunk_id,
                    'values': embedding,
                    'metadata': {
                        'text': chunk['text'][:1000],  # Truncate for storage
                        'type': chunk['type'],
                        'report_id': chunk['report_id'],
                        'user_id': chunk['user_id'],
                        'indexed_at': datetime.utcnow().isoformat(),
                        **chunk.get('metadata', {})
                    }
                })
            
            # Batch upsert to Pinecone
            self.index.upsert(vectors=vectors, namespace=namespace)
            
            return {
                'success': True,
                'message': f'Indexed {len(chunks)} chunks',
                'chunks_indexed': len(chunks),
                'report_id': credit_report.get('report_id')
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'Indexing failed: {e}',
                'chunks_indexed': 0
            }
    
    def semantic_search(
        self,
        query: str,
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        namespace: str = "credit_reports"
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search across credit reports
        Returns most relevant chunks
        """
        if not self.initialized:
            return [{'error': 'Vector DB not initialized'}]
        
        try:
            # Generate query embedding
            query_embedding = self.generate_embedding(query)
            
            # Search Pinecone
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=filter,
                namespace=namespace
            )
            
            # Format results
            matches = []
            for match in results.matches:
                matches.append({
                    'id': match.id,
                    'score': match.score,
                    'text': match.metadata.get('text', ''),
                    'type': match.metadata.get('type', ''),
                    'report_id': match.metadata.get('report_id', ''),
                    'user_id': match.metadata.get('user_id', ''),
                    'metadata': {
                        k: v for k, v in match.metadata.items()
                        if k not in ['text', 'type', 'report_id', 'user_id']
                    }
                })
            
            return matches
            
        except Exception as e:
            print(f"❌ Semantic search failed: {e}")
            return []
    
    def rag_query(
        self,
        question: str,
        user_id: str,
        top_k: int = 5
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        RAG (Retrieval Augmented Generation) query
        Retrieves relevant context and generates answer using OpenRouter
        """
        if not self.initialized:
            return "Vector DB not initialized", []
        
        try:
            # Step 1: Semantic search for relevant context
            filter = {'user_id': user_id} if user_id else None
            context_chunks = self.semantic_search(
                query=question,
                top_k=top_k,
                filter=filter
            )
            
            if not context_chunks:
                return "No relevant information found", []
            
            # Step 2: Build context
            context = "\n\n".join([
                f"[{chunk['type']}]: {chunk['text']}"
                for chunk in context_chunks
            ])
            
            # Step 3: Generate answer using OpenRouter (imported from other service)
            from services.openrouter_service import OpenRouterService
            openrouter = OpenRouterService()
            
            prompt = f"""Answer this question based ONLY on the provided credit report context:

QUESTION: {question}

CONTEXT:
{context}

Provide a clear, specific answer with actionable insights. If the context doesn't contain enough information, say so."""
            
            answer = asyncio.run(openrouter.chat_completion(
                messages=[
                    {"role": "system", "content": "You are a credit expert. Answer based only on provided context."},
                    {"role": "user", "content": prompt}
                ],
                tier='free',
                max_tokens=500
            ))
            
            answer_text = answer['choices'][0]['message']['content']
            
            return answer_text, context_chunks
            
        except Exception as e:
            print(f"❌ RAG query failed: {e}")
            return f"Query failed: {e}", []
    
    def get_similar_reports(
        self,
        report_id: str,
        top_k: int = 5,
        namespace: str = "credit_reports"
    ) -> List[Dict[str, Any]]:
        """Find similar credit reports for benchmarking"""
        if not self.initialized:
            return []
        
        try:
            # Get the report's summary embedding
            results = self.index.query(
                id=f"{report_id}-summary-0",
                top_k=top_k + 1,  # +1 to exclude self
                include_metadata=True,
                namespace=namespace
            )
            
            # Filter out the source report
            similar = [
                {
                    'report_id': match.metadata.get('report_id'),
                    'similarity_score': match.score,
                    'score': match.metadata.get('score'),
                    'utilization': match.metadata.get('utilization')
                }
                for match in results.matches
                if match.metadata.get('report_id') != report_id
            ]
            
            return similar[:top_k]
            
        except Exception as e:
            print(f"❌ Similar reports search failed: {e}")
            return []
    
    def delete_user_data(self, user_id: str, namespace: str = "credit_reports"):
        """Delete all indexed data for a user (GDPR compliance)"""
        if not self.initialized:
            return {'success': False, 'message': 'Vector DB not initialized'}
        
        try:
            # Delete by filter
            self.index.delete(
                filter={'user_id': user_id},
                namespace=namespace
            )
            
            return {
                'success': True,
                'message': f'Deleted all data for user {user_id}'
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'Deletion failed: {e}'
            }
    
    def get_index_stats(self, namespace: str = "credit_reports") -> Dict[str, Any]:
        """Get Pinecone index statistics"""
        if not self.initialized:
            return {'error': 'Vector DB not initialized'}
        
        try:
            stats = self.index.describe_index_stats()
            return {
                'total_vectors': stats.total_vector_count,
                'dimension': stats.dimension,
                'namespaces': {ns: info.vector_count for ns, info in stats.namespaces.items()},
                'index_fullness': stats.index_fullness
            }
        except Exception as e:
            return {'error': f'Stats retrieval failed: {e}'}


# Singleton instance
_vector_service: Optional[VectorSearchService] = None


def get_vector_service() -> VectorSearchService:
    """Get singleton vector service instance"""
    global _vector_service
    if _vector_service is None:
        _vector_service = VectorSearchService()
    return _vector_service


# Demo usage
if __name__ == "__main__":
    print("🔥 VECTOR SEARCH + RAG - DEMO")
    print("=" * 60)
    
    # Initialize service
    vector_service = VectorSearchService()
    
    # Test credit report
    test_report = {
        'report_id': 'report-demo-001',
        'user_id': 'demo_user',
        'summary': {
            'credit_score': 704,
            'credit_utilization': 0.35,
            'total_accounts': 12
        },
        'payment_history': {
            'late_payments': 2,
            'on_time_rate': 0.85
        },
        'accounts': [
            {
                'type': 'credit_card',
                'creditor_name': 'Chase Bank',
                'balance': 5000,
                'status': 'open'
            }
        ],
        'inquiries': [
            {'creditor': 'Bank of America', 'date': '2024-01-15'}
        ],
        'negative_items': {
            'collections': 26,
            'derogatory_marks': 3
        }
    }
    
    print("\n📊 TEST: Indexing Credit Report")
    result = vector_service.index_credit_report(test_report)
    print(f"Result: {result}")
    
    print("\n📊 TEST: Semantic Search")
    search_results = vector_service.semantic_search(
        query="What is the credit utilization?",
        top_k=3
    )
    print(f"Found {len(search_results)} results")
    for i, result in enumerate(search_results, 1):
        print(f"  {i}. Score: {result['score']:.3f} - Type: {result['type']}")
    
    print("\n📊 TEST: Index Stats")
    stats = vector_service.get_index_stats()
    print(f"Stats: {stats}")
    
    print("\n✅ Vector search demo complete")

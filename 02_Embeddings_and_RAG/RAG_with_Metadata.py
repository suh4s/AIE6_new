#!/usr/bin/env python3
"""
RAG Example with Metadata Support
=================================
This script demonstrates how to use the enhanced VectorDatabase
with metadata support for a Retrieval Augmented Generation (RAG) application.
"""

# Import necessary modules
from aimakerspace.vectordatabase import VectorDatabase, cosine_similarity
from aimakerspace.text_utils import TextFileLoader, CharacterTextSplitter
from aimakerspace.openai_utils.embedding import EmbeddingModel
from aimakerspace.prompts import SystemRolePrompt, UserRolePrompt
from aimakerspace.openai_utils.chat import ChatOpenAI
import numpy as np
import asyncio

def main():
    # Load and split the documents
    print("Loading and splitting documents...")
    text_loader = TextFileLoader("data/PMarcaBlogs.txt")
    documents = text_loader.load_documents()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_documents = text_splitter.split_texts(documents)
    print(f"Created {len(split_documents)} document chunks")

    # Create a VectorDatabase instance
    vector_db = VectorDatabase()

    # Create metadata for each chunk
    async def build_db_with_metadata():
        print("Generating embeddings and adding metadata...")
        # Get embeddings for all documents
        embeddings = await vector_db.embedding_model.async_get_embeddings(split_documents)
        
        # Add documents to the vector database with metadata
        for i, (text, embedding) in enumerate(zip(split_documents, embeddings)):
            # Create metadata for this chunk
            metadata = {
                "source": "PMarcaBlogs.txt",
                "author": "Marc Andreessen",
                "date": "2007-2009",
                "chunk_id": i,
                "chunk_size": len(text)
            }
            # Insert the document with its embedding and metadata
            vector_db.insert(text, np.array(embedding), metadata=metadata)
        
        print("Database built successfully")
        return vector_db

    # Run the async function to build the database
    vector_db = asyncio.run(build_db_with_metadata())

    # Define the RAG prompts
    RAG_PROMPT_TEMPLATE = """ \
Use the provided context to answer the user's query.
Include relevant metadata (like source and publication date) in your response when appropriate.

You may not answer the user's query unless there is specific context in the following text.

If you do not know the answer, or cannot answer, please respond with "I don't know".
"""

    rag_prompt = SystemRolePrompt(RAG_PROMPT_TEMPLATE)

    USER_PROMPT_TEMPLATE = """ \
Context:
{context}

User Query:
{user_query}
"""

    user_prompt = UserRolePrompt(USER_PROMPT_TEMPLATE)

    # Enhanced RetrievalAugmentedQAPipeline class that includes metadata
    class RetrievalAugmentedQAPipeline:
        def __init__(self, llm: ChatOpenAI(), vector_db_retriever: VectorDatabase) -> None:
            self.llm = llm
            self.vector_db_retriever = vector_db_retriever

        def run_pipeline(self, user_query: str) -> dict:
            print(f"Processing query: {user_query}")
            # Get relevant documents using cosine similarity
            context_list = self.vector_db_retriever.search_by_text(
                user_query, 
                k=4, 
                distance_measure=cosine_similarity
            )

            # Build context prompt with text and metadata
            context_prompt = ""
            metadata_info = []
            
            for context in context_list:
                text = context[0]
                similarity_score = context[1]
                
                # Get metadata for this text
                metadata = self.vector_db_retriever.get_metadata(text)
                
                # Add text to context
                context_prompt += f"{text}\n\n"
                
                # Store metadata for later use
                metadata_info.append({
                    "text": text[:100] + "...",  # Show first 100 chars only
                    "similarity": similarity_score,
                    "metadata": metadata
                })

            # Create prompts
            formatted_system_prompt = rag_prompt.create_message()
            formatted_user_prompt = user_prompt.create_message(
                user_query=user_query, 
                context=context_prompt
            )

            # Get response from LLM
            print("Generating response from LLM...")
            response = self.llm.run([formatted_system_prompt, formatted_user_prompt])
            
            # Return response with metadata
            return {
                "response": response,
                "context": context_list,
                "metadata": metadata_info
            }

    # Create and use the pipeline
    print("Creating RAG pipeline...")
    chat_openai = ChatOpenAI()
    pipeline = RetrievalAugmentedQAPipeline(
        llm=chat_openai,
        vector_db_retriever=vector_db
    )

    # Test the pipeline
    test_query = "What is the Michael Eisner Memorial Weak Executive Problem?"
    result = pipeline.run_pipeline(test_query)

    # Display the response
    print("\n" + "="*50)
    print("RESPONSE:")
    print("="*50)
    print(result["response"])
    print("="*50)

    # Display metadata for each chunk used in the response
    print("\nMETADATA FOR SOURCE CHUNKS:")
    print("-"*50)
    for i, item in enumerate(result["metadata"]):
        print(f"\nChunk {i+1}: {item['text']}")
        print(f"Relevance score: {item['similarity']:.4f}")
        print(f"Source: {item['metadata'].get('source')}")
        print(f"Author: {item['metadata'].get('author')}")
        print(f"Date: {item['metadata'].get('date')}")
        print(f"Chunk ID: {item['metadata'].get('chunk_id')}")
        print(f"Chunk size: {item['metadata'].get('chunk_size')} characters")

if __name__ == "__main__":
    main() 
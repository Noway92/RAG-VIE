import chromadb
from chromadb.config import Settings

def setup_chroma_db(chunks_with_embeddings, collection_name="vie_offers", persist_directory="./chroma_db"):
    """
    Initialise Chroma DB et ajoute les embeddings
    """
    # Configuration de Chroma DB avec persistance
    client = chromadb.PersistentClient(path=persist_directory)
    
    # Créer ou récupérer une collection
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}  # Similarité cosinus
    )
    
    # Préparer les données pour Chroma DB
    documents = []
    metadatas = []
    ids = []
    embeddings = []
    
    for i, chunk in enumerate(chunks_with_embeddings):
        documents.append(chunk["content"])
        metadatas.append(chunk["metadata"])
        ids.append(f"chunk_{i}")
        embeddings.append(chunk["embedding"])
    
    # Ajouter les données à la collection
    collection.add(
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    
    print(f"✅ {len(documents)} chunks ajoutés à Chroma DB")
    print(f"📁 Base de données sauvegardée dans: {persist_directory}")
    
    return collection, client

def query_chroma_db(question, collection, n_results=5):
    """
    Interroge Chroma DB avec une question
    """
    # Créer l'embedding de la question
    question_embedding = get_text_embedding(question)
    
    # Rechercher les chunks les plus similaires
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )
    
    return results


def load_chroma_collection(collection_name="vie_offers", persist_directory="./chroma_db"):
    """
    Charge une collection existante depuis Chroma DB
    """
    client = chromadb.PersistentClient(path=persist_directory)
    collection = client.get_collection(collection_name)
    print(f"✅ Collection '{collection_name}' chargée")
    return collection, client

def get_collection_stats(collection):
    """
    Affiche les statistiques d'une collection
    """
    count = collection.count()
    print(f"📊 Collection contient {count} documents")
    return count

def search_with_filters(collection, query_text, filters=None, n_results=5):
    """
    Recherche avec filtres
    """
    query_embedding = get_text_embedding(query_text)
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=filters,  # Filtres par métadonnées
        include=["documents", "metadatas", "distances"]
    )
    
    return results

# Exemple 4: Recherche par compétition faible
    filters = {"competition_level": "FAIBLE"}
    results = search_with_filters(
        collection,
        "offres avec peu de candidats",
        filters=filters
    )
import numpy as np
import os

def save_embeddings_numpy(chunks_with_embeddings, filename="vie_embeddings.npz"):
    """
    Sauvegarde les embeddings en format NumPy pour un chargement rapide
    """
    embeddings = np.array([chunk["embedding"] for chunk in chunks_with_embeddings])
    metadata = [chunk["metadata"] for chunk in chunks_with_embeddings]
    contents = [chunk["content"] for chunk in chunks_with_embeddings]
    
    np.savez_compressed(
        filename,
        embeddings=embeddings,
        metadata=metadata,
        contents=contents
    )
    print(f"✅ Embeddings sauvegardés dans {filename}")
    print(f"   Shape: {embeddings.shape}")


# Fonction pour charger les embeddings sauvegardés
def load_embeddings(filename="vie_embeddings.npz"):
    """Charge les embeddings depuis un fichier NumPy"""
    data = np.load(filename, allow_pickle=True)
    return {
        'embeddings': data['embeddings'],
        'metadata': data['metadata'],
        'contents': data['contents']
    }

def append_embeddings(new_chunks_with_embeddings, filename="vie_embeddings.npz"):
    """
    Ajoute de nouveaux embeddings aux fichiers existants
    """
    if not os.path.exists(filename):
        print(f"⚠️ Fichier {filename} non trouvé, création d'un nouveau fichier")
        save_embeddings_numpy(new_chunks_with_embeddings, filename)
        return
    
    # Charger les données existantes
    existing_data = load_embeddings(filename)
    
    # Fusionner avec les nouveaux chunks
    all_embeddings = np.vstack([
        existing_data['embeddings'], 
        np.array([chunk["embedding"] for chunk in new_chunks_with_embeddings])
    ])
    
    all_metadata = np.concatenate([
        existing_data['metadata'], 
        [chunk["metadata"] for chunk in new_chunks_with_embeddings]
    ])
    
    all_contents = np.concatenate([
        existing_data['contents'], 
        [chunk["content"] for chunk in new_chunks_with_embeddings]
    ])
    
    # Sauvegarder le tout
    np.savez_compressed(
        filename,
        embeddings=all_embeddings,
        metadata=all_metadata,
        contents=all_contents
    )

    print(f"✅ {len(new_chunks_with_embeddings)} nouveaux embeddings ajoutés à {filename}")
    print(f"📊 Total: {len(all_embeddings)} embeddings")
    print(f"   Nouvelle shape: {all_embeddings.shape}")
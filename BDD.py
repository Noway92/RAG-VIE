import numpy as np
import os
from datetime import datetime
from typing import Dict,List

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



def filter_embeddings_by_criteria(
    loaded_embeddings: Dict,
    start_date_min: str = None,
    start_date_max: str = None,
    countries: List[str] = None,
    sectors: List[str] = None,
    duration_min: int = None,
    competition_level: str = None
) -> tuple[np.ndarray, List[Dict], List[str]]:
    """
    Filtre les embeddings selon différents critères
    
    Args:
        loaded_embeddings: Données chargées depuis le fichier .npz
        start_date_min: Date minimale au format "YYYY-MM-DD"
        start_date_max: Date maximale au format "YYYY-MM-DD"
        countries: Liste de pays à inclure
        sectors: Liste de secteurs à inclure
        duration_min: Durée minimale en mois
        competition_level: Niveau de compétition ("FAIBLE", "MOYENNE", "ÉLEVÉE")
    
    Returns:
        Tuple (embeddings filtrés, métadonnées filtrées, contenus filtrés)
    """
    all_embeddings = loaded_embeddings['embeddings']
    all_metadata = loaded_embeddings['metadata']
    all_contents = loaded_embeddings['contents']
    
    # Indices des embeddings qui passent tous les filtres
    valid_indices = []
    
    for idx, metadata in enumerate(all_metadata):
        # Filtre sur la date de début
        if start_date_min or start_date_max:
            start_date_str = metadata.get("start_date")
            if start_date_str and start_date_str != 'Date non spécifiée':
                try:
                    start_date = datetime.fromisoformat(start_date_str.split('T')[0]).replace(tzinfo=None)
                    
                    if start_date_min:
                        date_min = datetime.fromisoformat(start_date_min).replace(tzinfo=None)
                        if start_date < date_min:
                            continue
                    
                    if start_date_max:
                        date_max = datetime.fromisoformat(start_date_max).replace(tzinfo=None)
                        if start_date > date_max:
                            continue
                except (ValueError, TypeError):
                    continue
            else:
                continue
        
        # Filtre sur le pays
        if countries:
            if metadata.get('country') not in countries:
                continue
        
        # Filtre sur le secteur
        if sectors:
            if metadata.get('sector') not in sectors:
                continue
        
        # Filtre sur la durée
        if duration_min:
            duration = metadata.get('duration_months')
            if not duration or duration == 'Durée non spécifiée':
                continue
            try:
                if int(duration) < duration_min:
                    continue
            except (ValueError, TypeError):
                continue
        
        # Filtre sur le niveau de compétition
        if competition_level:
            if metadata.get('competition_level') != competition_level:
                continue
        
        # Si tous les filtres sont passés
        valid_indices.append(idx)
    
    # Retourner les données filtrées
    filtered_embeddings = all_embeddings[valid_indices] if valid_indices else np.array([])
    filtered_metadata = [all_metadata[i] for i in valid_indices]
    filtered_contents = [all_contents[i] for i in valid_indices]
    
    print(f"🔍 Filtrage: {len(valid_indices)}/{len(all_metadata)} embeddings conservés")
    
    return filtered_embeddings, filtered_metadata, filtered_contents
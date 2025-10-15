"""
Filters.py - Module de filtrage pour le système RAG
"""

import numpy as np
from typing import List, Dict, Optional, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity



def filter_embeddings(
    embeddings: np.ndarray,
    metadata: np.ndarray,
    contents: np.ndarray,
    filters: Optional[Dict[str, Any]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Filtre les embeddings selon les critères spécifiés
    
    Args:
        embeddings: Array des embeddings
        metadata: Array des métadonnées
        contents: Array des contenus
        filters: Dictionnaire de filtres à appliquer
        
    Returns:
        Tuple (filtered_embeddings, filtered_metadata, filtered_contents, indices_kept)
    """
    if filters is None or len(filters) == 0:
        indices = np.arange(len(embeddings))
        return embeddings, metadata, contents, indices
    
    # Initialiser le masque
    mask = np.ones(len(metadata), dtype=bool)
    
    # Appliquer chaque filtre
    for key, value in filters.items():
        if key == "min_duration_months":
            mask &= np.array([
                meta.get("duration_months", 0) >= value 
                for meta in metadata
            ])
        
        elif key == "max_duration_months":
            mask &= np.array([
                meta.get("duration_months", 999) <= value 
                for meta in metadata
            ])
        
        elif key == "countries":
            mask &= np.array([
                meta.get("country") in value 
                for meta in metadata
            ])
        
        elif key == "cities":
            mask &= np.array([
                meta.get("city") in value 
                for meta in metadata
            ])
        
        elif key == "sectors":
            mask &= np.array([
                any(sector in meta.get("sector", "") for sector in value)
                for meta in metadata
            ])
        
        elif key == "companies":
            mask &= np.array([
                meta.get("company") in value 
                for meta in metadata
            ])
        
        elif key == "min_salary_eur":
            mask &= np.array([
                meta.get("salary_eur", 0) >= value 
                for meta in metadata
            ])
        
        elif key == "max_salary_eur":
            mask &= np.array([
                meta.get("salary_eur", 999999) <= value 
                for meta in metadata
            ])
        
        elif key == "start_date_after":
            mask &= np.array([
                meta.get("start_date", "1900-01-01") >= value 
                for meta in metadata
            ])
        
        elif key == "start_date_before":
            mask &= np.array([
                meta.get("start_date", "2100-01-01") <= value 
                for meta in metadata
            ])
        
        elif key == "chunk_type":
            # Filtrer par type de chunk (mission_description ou candidate_profile)
            mask &= np.array([
                meta.get("chunk_type") == value 
                for meta in metadata
            ])
        
        elif key == "max_competition_level":
            # Filtrer selon le niveau de compétition
            competition_order = {"FAIBLE": 1, "MOYENNE": 2, "ÉLEVÉE": 3}
            max_level = competition_order.get(value, 3)
            mask &= np.array([
                competition_order.get(meta.get("competition_level", "ÉLEVÉE"), 3) <= max_level
                for meta in metadata
            ])
        
        elif key == "max_application_rate":
            mask &= np.array([
                meta.get("application_rate", 100) <= value 
                for meta in metadata
            ])
        
        elif key == "min_application_rate":
            mask &= np.array([
                meta.get("application_rate", 0) >= value 
                for meta in metadata
            ])
    
    # Obtenir les indices des éléments conservés
    indices = np.where(mask)[0]
    
    # Appliquer le masque
    filtered_embeddings = embeddings[mask]
    filtered_metadata = metadata[mask]
    filtered_contents = contents[mask]
    
    return filtered_embeddings, filtered_metadata, filtered_contents, indices


def find_closest_embedding_with_filters(
    question: str,
    question_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    metadata: np.ndarray,
    contents: np.ndarray,
    filters: Optional[Dict[str, Any]] = None,
    top_n: int = 10,
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    Trouve les embeddings les plus proches avec filtrage préalable
    
    Args:
        question: La question posée
        question_embeddings: Embedding de la question
        text_embeddings: Embeddings de tous les documents
        metadata: Métadonnées de tous les documents
        contents: Contenus de tous les documents
        filters: Filtres manuels (optionnel)
        auto_detect_filters: Détecter automatiquement les filtres depuis la question
        top_n: Nombre de résultats à retourner
        
    Returns:
        Tuple (scores, indices_globaux, filtered_results)
    """
    # Combiner filtres automatiques et manuels
    
    # Filtrer les embeddings
    filtered_embeddings, filtered_metadata, filtered_contents, kept_indices = filter_embeddings(
        text_embeddings, metadata, contents, filters
    )
    
    print(f"📊 Résultats après filtrage: {len(filtered_embeddings)}/{len(text_embeddings)} embeddings")
    
    if len(filtered_embeddings) == 0:
        print("⚠️ Aucun résultat après filtrage. Essayez des critères moins restrictifs.")
        return np.array([]), np.array([]), []
    
    # Calculer les similarités uniquement sur les embeddings filtrés
    similarities = cosine_similarity(question_embeddings, filtered_embeddings)[0]
    
    # Récupérer les top_n
    n = min(top_n, len(similarities))
    top_indices_filtered = np.argsort(similarities)[-n:][::-1]
    top_scores = similarities[top_indices_filtered]
    
    # Convertir en indices globaux
    top_indices_global = kept_indices[top_indices_filtered]
    
    # Préparer les résultats avec métadonnées
    filtered_results = []
    for i, (score, global_idx) in enumerate(zip(top_scores, top_indices_global)):
        filtered_results.append({
            "rank": i + 1,
            "score": float(score),
            "content": contents[global_idx],
            "metadata": dict(metadata[global_idx]),
            "global_index": int(global_idx)
        })
    
    return top_scores, top_indices_global, filtered_results


def display_results(results: List[Dict], detailed: bool = True):
    """
    Affiche les résultats de manière formatée
    
    Args:
        results: Liste des résultats à afficher
        detailed: Si True, affiche les détails complets
    """
    print("\n" + "="*80)
    print(f"{'TOP RÉSULTATS FILTRÉS':^80}")
    print("="*80)
    
    for result in results:
        meta = result['metadata']
        
        print(f"\n🎯 Rang #{result['rank']} - Score de similarité: {result['score']:.4f}")
        print(f"📋 Référence: {meta['offer_reference']}")
        print(f"📝 Titre: {meta['title']}")
        print(f"🏢 Entreprise: {meta['company']}")
        print(f"📍 Localisation: {meta['city']}, {meta['country']}")
        print(f"🏭 Secteur: {meta['sector']}")
        print(f"⏱️ Durée: {meta['duration_months']} mois")
        print(f"💰 Indemnité: {meta['salary_eur']}€/mois")
        print(f"📅 Date de début: {meta['start_date'][:10]}")
        print(f"📊 Niveau de compétition: {meta['competition_level']} ({meta['application_rate']}%)")
        print(f"👥 Candidatures: {meta['candidates_count']} | 👀 Vues: {meta['views_count']}")
        print(f"✉️ Contact: {meta['contact_email']}")
        
        if detailed:
            print(f"\n📄 Type de chunk: {meta.get('chunk_type', 'N/A')}")
            print(f"🆔 Chunk ID: {meta.get('chunk_id', 'N/A')}")
        
        print("-" * 80)



def get_statistics(results: List[Dict]) -> Dict[str, Any]:
    """
    Calcule des statistiques sur les résultats
    
    Args:
        results: Liste des résultats
        
    Returns:
        Dictionnaire de statistiques
    """
    if not results:
        return {}
    
    countries = {}
    sectors = {}
    companies = {}
    durations = []
    salaries = []
    competition_levels = {"FAIBLE": 0, "MOYENNE": 0, "ÉLEVÉE": 0}
    
    for result in results:
        meta = result['metadata']
        
        # Pays
        country = meta.get('country', 'N/A')
        countries[country] = countries.get(country, 0) + 1
        
        # Secteurs
        sector = meta.get('sector', 'N/A')
        sectors[sector] = sectors.get(sector, 0) + 1
        
        # Entreprises
        company = meta.get('company', 'N/A')
        companies[company] = companies.get(company, 0) + 1
        
        # Durées
        durations.append(meta.get('duration_months', 0))
        
        # Salaires
        salaries.append(meta.get('salary_eur', 0))
        
        # Compétition
        comp_level = meta.get('competition_level', 'N/A')
        if comp_level in competition_levels:
            competition_levels[comp_level] += 1
    
    stats = {
        "total_results": len(results),
        "countries": dict(sorted(countries.items(), key=lambda x: x[1], reverse=True)),
        "sectors": dict(sorted(sectors.items(), key=lambda x: x[1], reverse=True)),
        "top_companies": dict(sorted(companies.items(), key=lambda x: x[1], reverse=True)[:5]),
        "duration": {
            "min": min(durations) if durations else 0,
            "max": max(durations) if durations else 0,
            "avg": sum(durations) / len(durations) if durations else 0
        },
        "salary": {
            "min": min(salaries) if salaries else 0,
            "max": max(salaries) if salaries else 0,
            "avg": sum(salaries) / len(salaries) if salaries else 0
        },
        "competition": competition_levels
    }
    
    return stats


def display_statistics(stats: Dict[str, Any]):
    """
    Affiche les statistiques de manière formatée
    
    Args:
        stats: Dictionnaire de statistiques
    """
    print("\n" + "="*80)
    print(f"{'STATISTIQUES DES RÉSULTATS':^80}")
    print("="*80)
    
    print(f"\n📈 Total de résultats: {stats['total_results']}")
    
    print("\n🌍 Répartition par pays:")
    for country, count in list(stats['countries'].items())[:5]:
        print(f"   • {country}: {count} offres")
    
    print("\n🏭 Répartition par secteur:")
    for sector, count in list(stats['sectors'].items())[:3]:
        sector_short = sector[:50] + "..." if len(sector) > 50 else sector
        print(f"   • {sector_short}: {count} offres")
    
    print("\n🏢 Top 5 entreprises:")
    for company, count in list(stats['top_companies'].items())[:5]:
        print(f"   • {company}: {count} offres")
    
    print(f"\n⏱️ Durée des missions:")
    print(f"   • Min: {stats['duration']['min']} mois")
    print(f"   • Max: {stats['duration']['max']} mois")
    print(f"   • Moyenne: {stats['duration']['avg']:.1f} mois")
    
    print(f"\n💰 Indemnités mensuelles:")
    print(f"   • Min: {stats['salary']['min']}€")
    print(f"   • Max: {stats['salary']['max']}€")
    print(f"   • Moyenne: {stats['salary']['avg']:.0f}€")
    
    print(f"\n📊 Niveau de compétition:")
    for level, count in stats['competition'].items():
        print(f"   • {level}: {count} offres")
    
    print("="*80)
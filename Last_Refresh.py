from datetime import datetime

def get_last_refresh_date():
    """
    Récupère la date du dernier rafraîchissement depuis le fichier Last_refresh
    """
    try:
        with open('Last_refresh', 'r') as f:
            last_refresh_str = f.read().strip()
            return datetime.fromisoformat(last_refresh_str)
    except FileNotFoundError:
        print("⚠️ Fichier Last_refresh non trouvé, utilisation de la date minimale")
        return datetime.min  # Date très ancienne pour tout récupérer
    except (ValueError, TypeError) as e:
        print(f"⚠️ Erreur lecture Last_refresh: {e}, utilisation de la date minimale")
        return datetime.min

def update_last_refresh_date():
    """
    Met à jour le fichier Last_refresh avec la date actuelle
    """
    current_date = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
    with open('Last_refresh', 'w') as f:
        f.write(current_date)
    print(f"✅ Last_refresh mis à jour: {current_date}")
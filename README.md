# Radar — tes signaux d'achat et de vente sur iPhone

Radar surveille les actifs que tu choisis (actions, ETF, indices, cryptos), calcule les principaux indicateurs techniques toutes les 30 minutes et t'envoie une notification push quand plusieurs d'entre eux convergent vers un point d'achat ou de vente. Pour chaque actif, tu décides : alertes à l'achat, à la vente, les deux, ou pause.

Tout est gratuit et tu n'as aucun serveur à gérer :

| Brique | Rôle | Où ça tourne |
|---|---|---|
| Appli Radar | Ta liste d'actifs, le tableau de bord, l'historique des alertes | Page web installée sur l'écran d'accueil de l'iPhone |
| Robot d'analyse | Récupère les cours, calcule les indicateurs, décide des alertes | GitHub Actions, toutes les 30 min |
| Notifications | Push sur l'écran verrouillé | Appli gratuite **ntfy** |

Installation : environ 15 minutes, une seule fois.

---

## 1. Créer le dépôt GitHub

1. Crée un compte gratuit sur [github.com](https://github.com) si tu n'en as pas.
2. En haut à droite : **+ → New repository**. Nom : `radar`. Coche **Public** (indispensable pour que l'hébergement de l'appli et l'analyse soient gratuites et illimitées). Clique **Create repository**.
3. Sur la page du dépôt vide, clique **uploading an existing file**, puis glisse **tout le contenu** du dossier `radar` décompressé (pas le dossier lui-même) : `index.html`, `watchlist.json`, `engine/`, `icons/`, `.github/`, etc. Clique **Commit changes**.

> Sur Mac, le dossier `.github` est masqué : dans le Finder, appuie sur **Cmd + Maj + .** pour l'afficher avant de le glisser. Si le glisser-déposer l'ignore, crée-le à la main : **Add file → Create new file**, tape `.github/workflows/radar.yml` comme nom, et colle le contenu du fichier.

Ce qui est public : ta liste d'actifs et les résultats d'analyse. Ce qui reste privé : ton jeton (uniquement sur ton téléphone) et ton canal de notification (secret GitHub).

## 2. Brancher les notifications (ntfy)

1. Installe **ntfy** depuis l'App Store et autorise les notifications.
2. Invente un nom de canal difficile à deviner, par exemple `radar-romain-k7q2x9` (n'importe qui connaissant ce nom pourrait lire tes alertes).
3. Dans ntfy : **+** → saisis ce nom → **Subscribe**.
4. Sur GitHub, dans ton dépôt : **Settings → Secrets and variables → Actions → New repository secret**.
   - Name : `NTFY_TOPIC`
   - Secret : le nom de ton canal (`radar-romain-k7q2x9`)

*Option Telegram* : ajoute aussi les secrets `TELEGRAM_TOKEN` (créé avec @BotFather) et `TELEGRAM_CHAT_ID`. Les deux canaux peuvent fonctionner ensemble.

## 3. Lancer le robot d'analyse

1. Onglet **Actions** du dépôt → si GitHub le demande, clique **I understand my workflows, go ahead and enable them**.
2. À gauche, **Radar — analyse et alertes → Run workflow**, coche **Envoyer une notification de test**, puis **Run workflow**.
3. Une à deux minutes plus tard : coche verte sur GitHub et notification « Radar est connecté » sur ton iPhone.

Ensuite, l'analyse tourne seule toutes les 30 minutes, et se relance immédiatement chaque fois que tu modifies ta liste depuis l'appli.

## 4. Mettre l'appli en ligne

1. **Settings → Pages**. Source : **Deploy from a branch**, branche **main**, dossier **/ (root)** → **Save**.
2. Une minute après, ton appli est à l'adresse `https://TON-PSEUDO.github.io/radar/`.

## 5. Créer le jeton d'accès de l'appli

L'appli en a besoin pour lire les résultats et enregistrer ta liste.

1. Photo de profil → **Settings → Developer settings → Personal access tokens → Fine-grained tokens → Generate new token**.
2. Nom : `Radar iPhone`. Expiration : 1 an (tu le renouvelleras).
3. **Repository access → Only select repositories → radar**.
4. **Permissions → Repository permissions** :
   - **Contents** : Read and write
   - **Actions** : Read and write
5. **Generate token** et copie-le (il commence par `github_pat_`).

## 6. Installer sur l'iPhone

1. Ouvre `https://TON-PSEUDO.github.io/radar/` dans **Safari**.
2. Saisis ton pseudo GitHub, `radar`, et colle le jeton → **Se connecter**.
3. Bouton **Partager → Sur l'écran d'accueil**. Radar s'ouvre désormais en plein écran comme une appli.

Tu peux tester l'interface avant tout ça avec **Essayer avec des données de démonstration**.

---

## Comment Radar décide d'une alerte

Aucun indicateur seul n'est fiable : la plupart donnent autant de faux signaux que de vrais. Radar utilise donc un **score de confluence** : chaque indicateur qui se déclenche ajoute des points, et l'alerte ne part que si le total atteint ton seuil **et** qu'au moins un vrai déclencheur (un croisement) vient d'apparaître.

| Indicateur | Côté achat | Côté vente | Points |
|---|---|---|---|
| RSI 14 | Sort de la survente (repasse au-dessus de 30) | Sort du surachat (repasse sous 70) | 2 |
| MACD 12/26/9 | Croise son signal à la hausse (+1 si sous zéro) | Croise son signal à la baisse (+1 si au-dessus de zéro) | 2 à 3 |
| Bollinger 20/2 | Réintègre la bande basse | Réintègre la bande haute | 2 |
| Stochastique 14/3/3 | Croisement haussier sous 20 | Croisement baissier au-dessus de 80 | 1 |
| MM50 / MM200 | Golden cross | Death cross | 3 |
| Cours / MM200 | Repasse au-dessus | Casse en dessous | 2 |
| Tendance de fond | +1 si au-dessus de la MM200, −1 sinon | +1 si en dessous, −1 sinon | ±1 |
| Volume | Bougie verte à plus de 1,5 × la moyenne | Bougie rouge à plus de 1,5 × la moyenne | 1 |

Seuils (achat / vente) : **Prudent** 6 / 6 · **Normal** 6 / 5 (recommandé) · **Réactif** 5 / 4. Ils ont été calibrés par backtest sur 36 actifs : en dessous, les signaux ne faisaient pas mieux que la tendance normale des actifs. Le module `engine/backtest.py` (workflow « Radar — backtest ») permet de refaire la mesure à tout moment.

Principes qui limitent les fausses alertes :

- **Bougies clôturées uniquement** pour les indicateurs : une alerte ne disparaît pas parce que le cours a bougé dans l'heure. En contrepartie, un signal « Jour » arrive après la clôture (le soir pour les actions, vers 2 h pour les cryptos, d'où les heures de silence).
- **Pas de répétition** : au moins 3 bougies entre deux alertes de même sens sur un même actif.
- **Tendance de fond** : un achat contre une tendance baissière perd un point, un achat sur repli dans une tendance haussière en gagne un.
- **Fiabilité mesurée** : chaque alerte indique ce qu'a donné ce même signal, avec ta sensibilité, sur l'historique de l'actif (pourcentage de réussite et variation moyenne à 20 séances), comparé à la variation moyenne sans signal. Si un signal ne bat pas cette moyenne sur un actif, tu le sais.

Chaque alerte d'achat propose aussi un **stop indicatif** (cours − 2 × ATR, la volatilité moyenne).

## Fonctionnalités en plus

- **Niveaux de prix personnels** par actif : zone d'achat, objectif de vente, stop. Surveillés en continu à chaque analyse, sans attendre la clôture.
- **Prix d'entrée** : affiche ta plus ou moins-value et un stop suggéré.
- **Horizon par actif** : 4 heures (court terme), Jour (recommandé), Semaine (long terme).
- **Heures de silence** : les alertes de la nuit sont gardées et envoyées au réveil.
- **Résumé quotidien** à l'heure de ton choix, avec la tendance et le score de chaque actif.
- **Historique** des 150 dernières alertes, filtrable achats / ventes.
- **Notification de test** et **analyse immédiate** depuis les Réglages.

## Trouver le bon symbole

Radar utilise les symboles de [Yahoo Finance](https://fr.finance.yahoo.com). Cherche l'actif sur le site et recopie le code affiché.

| Marché | Exemple |
|---|---|
| Euronext Paris | `MC.PA` (LVMH), `AIR.PA` (Airbus), `CW8.PA` (Amundi MSCI World) |
| Allemagne | `SAP.DE` |
| États-Unis | `AAPL`, `NVDA`, `SPY` |
| Cryptos | `BTC-EUR`, `ETH-EUR`, `SOL-USD` |
| Indices | `^FCHI` (CAC 40), `^GSPC` (S&P 500), `^IXIC` (Nasdaq) |

Un symbole introuvable apparaît en orange dans l'appli.

## Bon à savoir

- **Délai des cours** : Yahoo fournit les actions avec environ 15 minutes de retard, et GitHub peut décaler les analyses planifiées de quelques minutes aux heures chargées. Radar est fait pour repérer des zones, pas pour du scalping à la seconde.
- **Inactivité** : GitHub met en pause les tâches planifiées d'un dépôt public sans activité pendant 60 jours. Si tu reçois un e-mail à ce sujet, un clic sur **Enable workflow** dans l'onglet Actions suffit ; modifier ta liste depuis l'appli compte aussi comme activité.
- **Jeton expiré** : l'appli affiche « Jeton GitHub refusé ». Génère un nouveau jeton (étape 5) et colle-le dans Réglages → Connexion.
- **Une analyse en échec** : onglet **Actions** sur GitHub, clique sur l'exécution rouge pour lire l'erreur.

## Structure des fichiers

```
index.html              l'appli iPhone
manifest.webmanifest    installation sur l'écran d'accueil
sw.js                   ouverture hors connexion
icons/                  icônes
watchlist.json          ta liste d'actifs et tes réglages (modifiée par l'appli)
engine/screener.py      le moteur d'analyse
requirements.txt        dépendances Python
.github/workflows/radar.yml   planification toutes les 30 min
```

Les résultats sont écrits dans `status.json` sur la branche `data`, qui ne garde qu'une version pour ne pas encombrer le dépôt.

---

Radar est un outil d'aide à la décision fondé sur l'analyse technique. Ce n'est pas un conseil en investissement : aucun indicateur ne garantit l'évolution des cours, et les résultats passés ne préjugent pas des résultats futurs.

# Guide d'usage CropGuard-Q

Ce guide explique comment installer, lancer et **comprendre** chaque fonction du projet. Il ne contient volontairement aucun résultat chiffré : les résultats viennent des données réelles et se régénèrent avec le notebook (`notebooks/CropGuard_Q.ipynb`) ou avec `python -m challenge.report`.

**Sommaire**

1. [Le projet en une minute](#1-le-projet-en-une-minute)
2. [Installation](#2-installation)
3. [Les trois sources de réglages : aucune valeur forcée](#3-les-trois-sources-de-réglages--aucune-valeur-forcée)
4. [Parcours type](#4-parcours-type)
5. [La console (application tkinter)](#5-la-console-application-tkinter)
6. [Le notebook livrable](#6-le-notebook-livrable)
7. [Référence des fonctions, module par module](#7-référence-des-fonctions-module-par-module)
8. [Les fichiers de sortie](#8-les-fichiers-de-sortie)
9. [IBM Quantum pas à pas](#9-ibm-quantum-pas-à-pas)
10. [L'analyste IA (modèle de langage gratuit)](#10-lanalyste-ia-modèle-de-langage-gratuit)
11. [Sécurité](#11-sécurité)
12. [Dépannage](#12-dépannage)
13. [Glossaire](#13-glossaire)
14. [Questions probables du jury](#14-questions-probables-du-jury)

---

## 1. Le projet en une minute

**Tâche du hackathon** : reconnaître la culture d'un champ marocain (luzerne, maïs ou blé) à partir de mesures satellite Sentinel-2, avec un modèle quantique ou hybride, et le comparer honnêtement à des modèles classiques.

**Notre question** : *un modèle quantique ajouté à un modèle classique apporte-t-il quelque chose, surtout quand les données sont falsifiées ou incomplètes, par rapport à un autre modèle classique ajouté à sa place ?*

**Le système** (les trois volets du hackathon) :

| Volet | Pièce | Rôle |
|---|---|---|
| Quantique | SVM à noyau quantique (une porte ZZ par paire de qubits voisins, un qubit par nombre PCA) | un des trois « inspecteurs » qui votent |
| Cybersécurité | vote à trois + **alarme de falsification** + laboratoire d'attaques | un champ est signalé à un humain si les inspecteurs ne sont pas unanimes |
| IA | **correcteur d'erreurs** (régression linéaire) + analyste (modèle de langage, optionnel) | corrige les mesures bruitées d'un vrai ordinateur quantique ; rédige des notes d'incident |

Le **contrôle apparié** est le même système où le SVM quantique est remplacé par un SVM polynomial : c'est lui qui permet de répondre à la question.

---

## 2. Installation

```bash
git clone https://github.com/Knizmo/crop-guard-q.git
cd crop-guard-q
python3.12 -m venv .venv                                   # Python 3.12 épinglé ; 3.15 testé aussi
.venv/bin/python -m pip install -r requirements.txt       # indispensable
.venv/bin/python -m pip install -r requirements-noise.txt # optionnel : simulateur bruité (qiskit-aer)
.venv/bin/python -m pip install -r requirements-hardware.txt  # optionnel : IBM Quantum + calibrations réelles
```

Vérifier que tout fonctionne (≈ 10 s) :

```bash
.venv/bin/python -m tests.test_cropguard
```

Chaque ligne `ok test_...` est un contrôle réussi (formule quantique = Qiskit, règle de vote, budget d'attaque respecté, sécurité de l'analyste, etc.).

La console a besoin de **tkinter** (inclus avec Python sur Windows/macOS ; sous Fedora `sudo dnf install python3-tkinter`, sous Ubuntu `sudo apt install python3-tk`).

---

## 3. Les trois sources de réglages : aucune valeur forcée

Le code ne contient pas de chiffres « en dur » sur les données ou l'expérience. Tout vient de trois fichiers, lus par [`challenge/config.py`](../challenge/config.py) :

| Fichier | Contenu | Qui l'écrit |
|---|---|---|
| `data/manifest.json` | faits des **données réelles** : noms des cultures, noms des 13 bandes, dates, tailles des jeux, sommes de contrôle | les organisateurs (ne jamais modifier) |
| `configs/experiment.json` | **paramètres d'étude** : nombre de qubits, grilles de réglage, budgets d'attaque, étude de bruit, matériel IBM | vous (pour changer une expérience) |
| `configs/cropguard.json` | **réglages retenus** pour chaque inspecteur (gelés avant toute prédiction test) | `python -m challenge.tune` |

### `configs/experiment.json`, clé par clé

| Clé | Signification |
|---|---|
| `qubits` | nombre de qubits = nombre de composantes PCA (max 8, règle du concours) |
| `seeds` | graines pour la forêt et les attaques (0, 1, 2 exigées par le brief) |
| `forest_trees` | nombre d'arbres de la forêt aléatoire |
| `tuning.grids` | grilles de réglage, **même taille pour chaque inspecteur** (équité) |
| `tuning.forest_seed` | graine de la forêt pendant le réglage |
| `tuning.illustration_scales` | valeurs de `scale` hors grille, montrées seulement pour illustrer l'effondrement du noyau |
| `bootstrap.samples`, `bootstrap.seed` | nombre de rééchantillonnages par blocs spatiaux et leur graine |
| `attack.eps` | budgets de falsification ε (norme L2 en écarts-types d'entraînement) |
| `attack.iters` | pas de la recherche aléatoire de l'attaquant |
| `attack.poison_rates`, `attack.poison_draws` | fractions d'étiquettes empoisonnées et nombre de tirages |
| `attack.missing_bands`, `attack.missing_draws` | nombre de bandes perdues par champ et nombre de tirages |
| `noise.device` | QPU IBM dont on utilise les **calibrations réelles** pour le simulateur bruité |
| `noise.levels` | multiplicateurs des taux d'erreur (1 = l'appareil réel) |
| `noise.train_fields`, `validation_fields`, `calibration_pairs`, `shots`, `seed` | taille de l'étude de bruit |
| `hardware.shots`, `hardware.calibration_pairs` | tirs par circuit et paires de calibration envoyées sur le QPU |
| `hardware.tamper_eps` | budget du champ falsifié montré sur le matériel (et valeur par défaut du curseur de la console) |
| `hardware.min_vote_agreement` | seuil de la règle qui choisit le C « prêt pour le matériel » |
| `hardware.C_candidates`, `hardware.scan_draws` | valeurs de C testées hors ligne et nombre de tirages |

Changer une valeur ici suffit : le notebook, la console et les fonctions utilisent ces réglages comme valeurs par défaut.

---

## 4. Parcours type

```bash
.venv/bin/python -m tests.test_cropguard     # 1. tout est sain ?
.venv/bin/python -m challenge.tune           # 2. (optionnel) refaire le réglage -> configs/cropguard.json
.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/CropGuard_Q.ipynb   # 3. tous les résultats
.venv/bin/python -m challenge.report         # 4. résumé des résultats en Markdown
.venv/bin/python cropguard_app.py            # 5. démonstration visuelle
```

Le notebook complet prend une dizaine de minutes sur un portable 8 cœurs ; la partie principale (sections 1 à 4) environ une minute.

---

## 5. La console (application tkinter)

```bash
.venv/bin/python cropguard_app.py
```

Au démarrage, la console vérifie les sommes de contrôle, entraîne le comité quantique et le contrôle, réentraîne la référence officielle et lit les études sauvegardées dans `outputs/`. Six modules :

| Module | Ce qu'on y voit | Ce qu'on peut faire |
|---|---|---|
| **MISSION** | question, scores de validation (comité quantique, contrôle, référence recalculée), taux d'alarme, dernier job IBM, schéma, conclusions calculées | regarder les qubits du champ choisi tourner (aiguille = phase de la donnée) |
| **FIELD SCANNER** | les champs de validation (✓ vérifié, ⚠ signalé, ✗ faux non signalé), le vote de chaque inspecteur, les 13 mesures | choisir un champ ; **INJECT TAMPER** (ε, attaquant naïf ou adaptatif) ; **RESTORE** ; **AI ANALYST REPORT** |
| **QUANTUM CORE** | matrice du noyau quantique, journal de réglage | bouger `scale` puis **RECOMPUTE KERNEL** pour voir l'effondrement du noyau |
| **ATTACK LAB** | part des champs falsifiés qui passent sans alarme selon ε | **RUN QUICK STUDY** (refait l'étude naïve en direct) |
| **NOISE + AI FIXER** | erreur brute / ZNE / correcteur IA, nuage de points | — (lit `outputs/noise_study.json`) |
| **IBM QPU LINK** | les deux jobs réels, mesures vs exact, courbe « pourquoi C compte » | **SUBMIT NEW INFERENCE JOB** si `QISKIT_IBM_TOKEN` est défini (demande confirmation) |

---

## 6. Le notebook livrable

`notebooks/CropGuard_Q.ipynb` est le document rendu aux organisateurs. La première cellule de code contient quatre interrupteurs :

| Interrupteur | Effet |
|---|---|
| `RUN_TUNING` | refait la recherche de réglages (sans écraser `configs/cropguard.json`) et vérifie qu'elle redonne les mêmes réglages |
| `RUN_ATTACK_LAB` | refait les attaques ; sinon lit `outputs/attack_lab.json` |
| `RUN_NOISE_STUDY` | refait l'étude de bruit (si qiskit-aer est installé) ; sinon lit la sauvegarde |
| `RUN_HARDWARE` | **False par défaut** : envoie de vrais jobs IBM seulement si `True` et si le jeton est présent |

Sections : 1 entrées et PCA · 2 réglage à budget égal · 3 résultat principal (graines, métriques, ressources, prédictions test) · 4 bootstrap apparié · 5 laboratoire d'attaques · 6 bruit et correcteur IA · 7 IBM · 8 conclusions **générées** par `challenge/report.py`.

---

## 7. Référence des fonctions, module par module

Légende : ⭐ = à connaître absolument pour présenter le projet. Les valeurs par défaut « config » viennent de `configs/experiment.json`.

### 7.1 Données et mesures (code des organisateurs, réutilisé)

| Fonction | Rôle |
|---|---|
| ⭐ `challenge.data.load_data()` | charge train/validation/test **après** vérification des sommes de contrôle, des colonnes et des identifiants. Retourne `(tables, view, manifest, starter)`. À appeler avant tout. |
| ⭐ `challenge.metrics.score(y_true, y_pred, labels)` | macro-F1, accuracy, précision et rappel par culture, matrice de confusion |
| `challenge.metrics.probabilities(p, n, k)` | vérifie qu'un tableau de probabilités est valide (forme, somme à 1) |
| ⭐ `challenge.classical.reference()` | la référence officielle : StandardScaler + SVM RBF sur les 13 bandes |
| `challenge.coverage.metric_vector(y, pred, weights)` | macro-F1 et rappels pondérés ; réutilisé par notre bootstrap |
| `python -m challenge.coverage check / train / compare` | outils d'origine pour comparer des tailles d'entraînement |

### 7.2 Réglages : `challenge/config.py`

| Nom | Rôle |
|---|---|
| `MANIFEST`, `EXPERIMENT` | contenu de `data/manifest.json` et `configs/experiment.json` |
| `LABELS`, `FEATURES`, `QUBITS` | cultures, noms des bandes, nombre de qubits (vérifié ≤ 8) |
| `OUT` | dossier `outputs/` |
| `frozen()` | lit `configs/cropguard.json` (réglages retenus) |

### 7.3 Le noyau quantique : `challenge/kernel.py`

| Fonction | Rôle |
|---|---|
| ⭐ `QuantumKernelSVM(scale, C, reps=1)` | SVM dont le noyau est la **fidélité** entre deux états quantiques. `scale` = force de rotation (le bouton sensible), `C` = régularisation du SVM. |
| `.states(X)` | état quantique exact (vecteur de 2ⁿ amplitudes) de chaque champ ; formule fermée, identique à Qiskit et environ 100× plus rapide |
| `.qiskit_states(X)` | même calcul avec `Statevector` de Qiskit (sert aux tests ; utilisé si `reps > 1`) |
| `.fit(X, y)` | calcule les états d'entraînement, la matrice de noyau et entraîne le SVM |
| `.kernel(X)` | lignes de noyau ; seules les colonnes des **vecteurs de support** sont calculées (les autres ne servent pas à la décision) |
| `.predict(X)`, `.decision_function(X)` | culture prédite / scores par culture |
| `.pair_circuit(x, y)` | le circuit « calcul puis dé-calcul » qu'un vrai QPU exécute : P(000…0) = similarité |
| `.resources()` | qubits, profondeur, nombre de CX, vecteurs de support |
| `fidelity(A, B)` | matrice \|⟨a\|b⟩\|² entre deux piles d'états |
| `basis(n)` | bits et parités des 2ⁿ états de base (mis en cache) |

### 7.4 Le modèle soumis : `challenge/submission.py`

| Fonction | Rôle |
|---|---|
| ⭐ `Submission()` | **le modèle rendu** : comité [SVM quantique, SVM RBF, forêt] avec alarme |
| ⭐ `MatchedControl()` | le **contrôle apparié** : même comité, SVM polynomial à la place du quantique |
| `Committee(partner, config)` | classe commune aux deux |
| ⭐ `.fit(X_train, y_train, seed)` | apprend la PCA **sur l'entraînement seulement** puis entraîne les trois inspecteurs. `X` = les 13 bandes brutes. |
| ⭐ `.inspect(X)` | en un passage : `votes` (3 × n), `prediction` (vote majoritaire) et `flag` (True = pas unanime → humain) |
| `.predict(X)`, `.predict_proba(X)`, `.flag(X)` | interface standard du concours ; `predict_proba` = parts de vote pondérées |
| `.votes(X)`, `.transform(X)` | votes bruts ; les 13 bandes → nombres PCA |
| `.resources()` | coûts : valeurs ajustées, vecteurs de support, nœuds des arbres, temps |
| `Committee.vote_shares(V)` | règle de vote : majorité ; si les trois diffèrent, le SVM RBF tranche |
| `preprocessing()` | StandardScaler → PCA(qubits) → StandardScaler |
| `member(name, params, seed)` | fabrique un inspecteur : `quantum`, `rbf`, `poly` ou `forest` |
| `scores(model, Z)` | scores par culture (décision SVM ou probabilités de la forêt) |
| `raw_features(X)` | vérifie l'entrée : 13 colonnes finies, dans l'ordre du manifest (contrôle de sécurité) |

### 7.5 Le réglage : `challenge/tune.py`

| Fonction | Rôle |
|---|---|
| ⭐ `tune(write=True)` | essaie toutes les configurations de chaque grille sur la validation, garde la meilleure, écrit `configs/cropguard.json` et `outputs/tuning_log.json`. `write=False` ne touche pas aux réglages gelés. |
| `grid(name)` | liste des configurations d'un inspecteur |
| `GRIDS` | les grilles lues dans la configuration |

### 7.6 Cybersécurité : `challenge/guard.py`

| Fonction | Rôle |
|---|---|
| ⭐ `tamper(margin_fn, X, yi, eps, X_train, iters, seed)` | **l'attaquant** : cherche au hasard une modification des 13 bandes de norme ≤ ε (en écarts-types), dans les bornes plausibles, qui trompe le modèle visé |
| `member_margin(committee, slot)` | cible = un seul inspecteur (attaquant **naïf**) |
| `committee_margin(committee, X0, yi)` | cible = les trois à la fois (attaquant **adaptatif**, qui connaît la défense) |
| `margin(S, yi)` | marge = score de la vraie culture − meilleur autre score (< 0 = trompé) |
| ⭐ `slips(V, y)` | pour une alarme : champs **signalés** et champs **non détectés** (tous d'accord sur une mauvaise culture) |
| `alarm_views(sub, ctrl, X, y)` | les six alarmes comparées (RBF seul, + quantique, + poly, + forêt, comités) |
| ⭐ `tamper_study(X, y, Xv, yv, ...)` | étude complète de falsification (toutes graines, tous ε, naïf + adaptatif) |
| `poison_study(...)` | étiquettes d'entraînement empoisonnées, réentraînement, score sur validation propre |
| `missing_study(...)` | bandes perdues à la prédiction, remplacées par la moyenne d'entraînement |
| `committees(X, y, seed)` | entraîne le comité quantique et le contrôle sur les mêmes lignes |
| `index(y)` | cultures → indices |
| ⭐ `block_weights(sample_ids)` + `paired_bootstrap(a, b, y, W, stat)` | **la preuve statistique** : rééchantillonne les blocs spatiaux et donne l'écart b − a avec un intervalle à 95 %. Si l'intervalle contient 0, on ne peut pas départager. |

### 7.7 IA et bruit : `challenge/mitigation.py`

| Fonction | Rôle |
|---|---|
| ⭐ `fit_fixer(noisy, exact)` / `apply_fixer(fixer, noisy)` | **le correcteur IA** : une droite apprise sur des paires dont la valeur exacte est connue, puis appliquée aux autres paires |
| ⭐ `device_profile(backend)` | lit les **calibrations réelles** d'un QPU IBM (médianes : erreurs de portes, durées, erreur de lecture, T1, T2) |
| `save_profile(backend)` / `load_profile(device)` | sauvegarde le profil réel dans `outputs/device_profile.json` ; sinon utilise l'instantané hors ligne d'IBM du même appareil |
| `noise_model(profile, level)` | simulateur bruité construit à partir de ce profil ; `level` multiplie les erreurs |
| `circuit(x1, x2, fold)` | circuit de similarité ; `fold=3` triple chaque CX (pour la ZNE) |
| ⭐ `noise_study(quantum, Ztr, ytr, Zva, yva, profile)` | compare valeur exacte / bruitée / corrigée par l'IA / ZNE à plusieurs niveaux de bruit |
| `ideal_zeros(qc)`, `zeros_probability(...)`, `to_basis(...)`, `run_noisy(...)` | outils internes (probabilité exacte, comptage, transpilation, simulation) |

### 7.8 Matériel IBM : `challenge/hardware_kernel.py`

| Fonction | Rôle |
|---|---|
| `connect(backend_name=None)` | se connecte avec `QISKIT_IBM_TOKEN` (variable d'environnement) ; sans nom, choisit le QPU le moins chargé |
| ⭐ `pick_fields(sub, X_train, Xv, yv, ids)` | choisit les champs de la démo : un champ « typique » vérifié par culture + un champ falsifié que l'alarme détecte |
| `build(sub, X_fields)` | construit les circuits (champ × vecteur de support + paires de calibration) et leurs valeurs exactes |
| `submit(circuits, backend, shots, approved)` | transpile et envoie le job ; refuse sans `approved=True` |
| ⭐ `run(sub, fields, backend, approved=True)` | tout le job : envoi, correcteur IA calibré **dans le même job**, votes exact / brut / corrigé, sauvegarde JSON (avec la calibration de l'appareil au moment du job) |
| `zeros(job, shots)` | valeurs de noyau mesurées |
| `quantum_votes(sub, values, n_fields)` | décision du SVM à partir de valeurs de noyau mesurées |
| `residuals(hw)` | erreurs restantes après correction, sur un job sauvegardé |
| ⭐ `robustness_scan(Z, y, Zv, yv, errors, scale)` | hors ligne : pour chaque C, part des votes qui survivent au bruit de tirage et à l'erreur matérielle mesurée |
| ⭐ `choose_C(scan, threshold)` | la règle : le plus grand C qui garde au moins `threshold` des votes (calculé, pas imposé) |

### 7.9 Analyste IA : `challenge/analyst.py`

| Fonction | Rôle |
|---|---|
| `note(info)` | note d'incident pour un champ : modèle de langage si `CROPGUARD_LLM_URL` est défini, sinon gabarit hors ligne. **N'influence jamais la décision.** |
| `facts(info)` | les seules données envoyées : votes, cultures, plus grandes variations de bandes |
| `offline_note(info)`, `llm_note(info)`, `endpoint()` | gabarit, appel au modèle, vérification de l'URL (https obligatoire sauf localhost) |

### 7.10 Rapport : `challenge/report.py`

| Fonction | Rôle |
|---|---|
| ⭐ `findings(...)` / `python -m challenge.report` | écrit les conclusions à partir de `outputs/*.json` : chaque chiffre et chaque verdict (« différence détectable ou non ») suit les données |
| `clean_section`, `tamper_section`, `member_section`, `noise_section`, `hardware_section`, `resources_text` | les paragraphes du rapport |
| `detectable(b)`, `pair(...)`, `load(name)` | outils : intervalle sans zéro ?, retrouver une comparaison, lire un résultat |

### 7.11 Console : `cropguard_app.py`

Classe `Console` (tkinter). Méthodes à connaître : `load()` (données + modèles, dans un fil séparé), `select(i)` (choisir un champ), `inject()` (falsifier), `analyse()` (note IA), `recompute()` (noyau à une autre force de rotation), `quick_attack()` (étude naïve en direct), `submit_qpu()` (job IBM, avec confirmation). Les calculs longs passent par `bg()` pour ne jamais figer la fenêtre.

---

## 8. Les fichiers de sortie

| Fichier | Contenu |
|---|---|
| `outputs/main_results.json` | métriques par modèle et graine, bootstrap, ressources, réglages, variance PCA |
| `outputs/tuning_log.json` | chaque configuration essayée et son score |
| `outputs/attack_lab.json` | falsification, empoisonnement, bandes manquantes, bootstrap des alarmes |
| `outputs/noise_study.json` | étude de bruit + calibration réelle utilisée |
| `outputs/device_profile.json` | calibration réelle du QPU (lue en direct chez IBM) |
| `outputs/hardware_result.json`, `hardware_result_robust.json` | les deux jobs IBM (backend, identifiant, mesures, votes) |
| `outputs/hardware_robustness.json` | balayage de C et C choisi par la règle |
| `outputs/predictions/*_validation.csv` | prédictions de validation au format de la référence |
| `outputs/predictions/quantum_committee_0_test.csv` | **fichier de soumission** : prédiction, alarme et vote de chaque inspecteur pour les 221 champs test |
| `outputs/cg_*.png` | figures pour les slides |

---

## 9. IBM Quantum pas à pas

1. Créez un compte sur IBM Quantum Platform et copiez votre clé API.
2. Mettez-la **uniquement** dans l'environnement :
   ```bash
   export QISKIT_IBM_TOKEN='votre-clé'     # jamais dans un fichier, un notebook ou un commit
   ```
3. Installez `requirements-hardware.txt`.
4. Choisissez : console → *IBM QPU LINK* → *SUBMIT NEW INFERENCE JOB*, ou notebook avec `RUN_HARDWARE = True` (deux jobs : le modèle soumis, puis le modèle au C choisi par la règle).
5. Coût : environ 2 à 3 minutes de QPU par job avec les réglages actuels (le plan gratuit donne 10 minutes par mois glissant). Vérifiez votre solde avant.
6. Le résultat est sauvegardé ; les autres personnes peuvent relancer le notebook **sans jeton**.

Rappel du règlement : inférence seulement (aucun entraînement sur le matériel), avec nom du backend, identifiant de job et comparaison au simulateur exact.

---

## 10. L'analyste IA (modèle de langage gratuit)

Par défaut, l'analyste écrit une note à partir d'un gabarit, sans réseau. Pour qu'un modèle de langage la rédige :

```bash
# gratuit, sans clé (testé)
export CROPGUARD_LLM_URL=https://text.pollinations.ai/openai
# ou Groq (clé gratuite)
export CROPGUARD_LLM_URL=https://api.groq.com/openai/v1/chat/completions
export CROPGUARD_LLM_KEY='votre-clé' CROPGUARD_LLM_MODEL=llama-3.3-70b-versatile
# ou un modèle local Ollama
export CROPGUARD_LLM_URL=http://localhost:11434/v1/chat/completions CROPGUARD_LLM_MODEL=llama3.2
```

Seuls les votes, les cultures et les plus grandes variations de bandes sont envoyés. Le texte reçu est affiché tel quel (limité, jamais exécuté) et ne change jamais le verdict.

---

## 11. Sécurité

- **Secrets** : jeton IBM et clé de modèle de langage uniquement en variables d'environnement ; `.gitignore` exclut `.env`, clés et `.qiskit/`.
- **Entrées** : `load_data` vérifie les sommes de contrôle ; `raw_features` refuse toute entrée de mauvaise forme ou non finie.
- **Matériel** : `submit` refuse sans `approved=True` ; le code IBM n'est importé que quand on l'utilise.
- **Modèle de langage** : sortie traitée comme non fiable (texte brut, longueur limitée, jamais exécutée) ; https obligatoire sauf localhost.
- **Données** : les attaques travaillent sur des copies ; le modèle soumis s'entraîne sur les étiquettes propres ; les étiquettes test ne sont jamais utilisées.

---

## 12. Dépannage

| Problème | Solution |
|---|---|
| `ModuleNotFoundError: qiskit_aer` | `pip install -r requirements-noise.txt` (ou laissez le notebook afficher l'étude sauvegardée) |
| `ModuleNotFoundError: qiskit_ibm_runtime` | `pip install -r requirements-hardware.txt` |
| `KeyError: 'QISKIT_IBM_TOKEN'` | `export QISKIT_IBM_TOKEN=...` dans le terminal qui lance la console ou Jupyter |
| erreur `dill` / `co_lnotab` à la transpilation | Python 3.15 : le code passe déjà `num_processes=1` ; si vous ajoutez des transpilations, faites de même |
| `No module named tkinter` | installez `python3-tkinter` (Fedora) ou `python3-tk` (Ubuntu) |
| la console paraît serrée | elle s'adapte jusqu'à ~900 px de large ; agrandissez la fenêtre si votre gestionnaire la place en demi-écran |
| `Release checksum mismatch` | un fichier de `data/` a été modifié : restaurez-le (`git checkout data/`) |
| `Manifest labels must be in alphabetical order` | le code suppose l'ordre alphabétique des cultures (ordre de scikit-learn) |
| `The challenge allows at most 8 qubits` | `qubits` doit rester entre 1 et 8 dans `configs/experiment.json` |

---

## 13. Glossaire

| Terme | Explication simple |
|---|---|
| **Qubit** | un « cadran » : sa position fixe les chances de lire 0 ou 1 |
| **Carte de caractéristiques ZZ** | circuit qui transforme les nombres d'un champ en angles et lie les qubits voisins |
| **Noyau quantique / fidélité** | similarité entre deux champs = chance de revenir à 000…0 après le circuit de A puis l'inverse de B |
| **Vecteur de support** | champ d'entraînement qui compte vraiment pour la décision du SVM |
| **C** | réglage du SVM : grand C = suit les données de près mais amplifie le bruit |
| **scale** | force de rotation : trop grande → le noyau s'effondre (chaque champ ressemble à aucun autre) |
| **PCA** | compresse les 13 bandes en quelques nombres (un par qubit) |
| **macro-F1** | moyenne des F1 des trois cultures (métrique principale du concours) |
| **Bootstrap par blocs spatiaux** | on rééchantillonne des zones géographiques entières pour mesurer l'incertitude d'un écart |
| **ε (epsilon)** | budget de l'attaquant : taille totale de la modification des bandes |
| **Attaquant naïf / adaptatif** | vise un inspecteur / connaît les trois |
| **Alarme / non détecté** | inspecteurs pas d'accord → humain / tous d'accord sur une mauvaise culture |
| **Contrôle apparié** | même système avec un modèle classique à la place du quantique |
| **Tirs (shots)** | répétitions d'une mesure sur un vrai QPU |
| **ZNE** | extrapolation à bruit nul : relancer avec plus de bruit et extrapoler (méthode classique de comparaison) |
| **Correcteur IA** | droite apprise qui ramène une mesure bruitée vers la valeur exacte |
| **Calibration** | erreurs et durées publiées par IBM pour chaque qubit et chaque porte |

---

## 14. Questions probables du jury

| Question | Où trouver la réponse |
|---|---|
| Le quantique fait-il mieux ? | section 8 du notebook (`report.findings`) : verdict du bootstrap, généré à partir des données |
| La comparaison est-elle équitable ? | même entrée, même PCA, mêmes lignes, grilles de même taille, mêmes graines (sections 2–4) |
| Combien de qubits, quelle profondeur, combien de circuits ? | `Submission().fit(...).resources()` et le paragraphe « Resources » du notebook |
| Pourquoi l'alarme ? | section 5 : part des champs falsifiés qui passent selon l'alarme choisie |
| Le correcteur IA améliore-t-il la précision ? | non revendiqué : il améliore les valeurs de similarité (section 6) |
| Preuve du matériel IBM ? | backend, identifiant de job et comparaison au simulateur : section 7 et `outputs/hardware_result*.json` |
| Pourquoi deux jobs IBM ? | le premier montre qu'un grand C amplifie le bruit ; la règle `choose_C` choisit un C robuste, vérifié par le second |
| Ce que nous ne prétendons pas | aucun avantage quantique, aucune robustesse supérieure du quantique, l'alarme n'arrête pas un attaquant qui connaît la défense |

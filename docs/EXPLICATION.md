# Les modules CropGuard-Q expliqués simplement

Ce document explique les fichiers ajoutés dans `challenge/` pour CropGuard-Q, avec des exemples concrets.
Les chiffres viennent du README (instantané) ; `python -m challenge.report` donne les valeurs actuelles.
Les valeurs « blé/maïs 0,9 / 0,2 » et « 0,72 → 0,80 » sont des illustrations, pas des mesures du projet.

**L'image d'ensemble.** Une agence agricole vérifie les déclarations des agriculteurs (« mon champ, c'est du blé »)
à partir d'images satellite. Trois experts examinent chaque champ. S'ils ne sont pas d'accord, le dossier est bloqué.
Chaque fichier joue un rôle dans cette agence.

---

## 1. `kernel.py` : l'expert quantique (« mesurer la ressemblance »)

- Après préparation, chaque champ est décrit par **8 nombres**. Chaque nombre fait tourner **un qubit**,
  ce qui donne 8 qubits et un « état quantique » par champ.
- On compare deux champs avec un score de **ressemblance entre 0 et 1** (la formule est |⟨a|b⟩|²) :
  - un champ de blé comparé à un autre champ de blé donne environ 0,9 (très proches) ;
  - un champ de blé comparé à un champ de maïs donne environ 0,2 (différents).
- Le **SVM** reçoit ce tableau de ressemblances et trace une frontière entre les cultures.
- Le réglage **`scale`** (0,075) fixe la force des rotations. S'il est trop grand, tous les champs paraissent
  différents (scores proches de 0) et le modèle n'apprend plus rien.
- « Calculé exactement » : avec 8 qubits, il n'y a que 256 valeurs à calculer, donc un PC le fait directement.
  C'est environ 100 fois plus rapide que Qiskit, et les tests vérifient qu'on obtient le même résultat.

**À retenir : comme comparer des empreintes digitales, puis un juge (le SVM) qui trace la frontière.**

## 2. `submission.py` : le comité des 3 experts

- Préparation des données : 13 bandes brutes → normalisation → PCA à 8 dimensions → normalisation.
  Elle est apprise **uniquement sur les données d'entraînement**.
- Les trois experts : le **SVM quantique**, le **SVM RBF** (classique) et la **forêt aléatoire**.
- La règle de vote :

| Votes | Résultat | Alarme ? |
|---|---|---|
| blé, blé, blé | blé | non, dossier vérifié |
| blé, blé, maïs | blé (2 contre 1) | **oui** |
| blé, maïs, luzerne | c'est le **SVM RBF qui tranche** (son vote pèse 1,001 au lieu de 1) | **oui** |

- **`MatchedControl`** est le même comité, avec un SVM polynomial classique à la place du quantique.
  C'est comme un **placebo dans un essai de médicament** : sans lui, impossible de savoir si le gain vient
  du quantique ou simplement du fait d'avoir un troisième expert.
- Résultat : 0,921 contre 0,922, donc aucune différence.

**À retenir : 2 contre 1 l'emporte, et tout désaccord lève un drapeau.**

## 3. `tune.py` : régler les boutons, puis sceller

- Chaque expert a ses boutons. Pour le quantique : 6 valeurs de `scale` × 5 valeurs de `C` = **30 essais**.
  Les autres experts ont aussi droit à 30 essais chacun, pour que la comparaison soit **équitable**.
- On garde le réglage qui obtient le meilleur score sur la validation. Pour le quantique : `scale=0.075, C=100`.
- Ce réglage est écrit dans `configs/cropguard.json`, puis **gelé**. Le jeu de test ne sert jamais à choisir.

**À retenir : comme un concours de cuisine. Chaque chef a 30 essais pour ajuster sa recette,
puis la recette est mise sous enveloppe scellée.**

## 4. `guard.py` : le labo d'attaques (des crash-tests)

On teste quatre situations, toujours sur des **copies** des données :

1. **Falsification.** Un fraudeur retouche légèrement les 13 bandes pour faire passer du maïs pour du blé.
   - Le budget `eps = 0.1` signifie une modification totale de 0,1 écart-type seulement, et les valeurs restent plausibles.
   - **L'attaquant naïf** ne vise que le SVM RBF. Il trompe ce SVM, seul, sur **24,9 %** des champs.
     Face au comité complet, **12,7 %** passent sans alarme avec le quantique, contre 14,7 % avec le contrôle.
   - **L'attaquant malin** connaît les trois experts. Avec un budget plus grand (eps = 0,5), il fait encore
     passer 50 à 55 % des champs. L'alarme ne suffit donc pas contre quelqu'un qui connaît la défense.
2. **Empoisonnement.** On change au hasard 30 % des étiquettes d'entraînement (on déclare « maïs » des champs de blé).
   Le quantique, seul, tombe à 0,78 ; le SVM polynomial reste à 0,85.
3. **Bandes manquantes.** Un capteur tombe en panne et la valeur manquante est remplacée par la moyenne.
4. **Bootstrap par blocs spatiaux.** Il répond à la question : « la différence est-elle réelle ou due au hasard ? »
   - On retire au hasard des **groupes de champs voisins** (18 blocs) et on recalcule le score, 2 000 fois.
   - Si l'intervalle à 95 % **contient 0**, la différence n'est pas détectable.
   - Sur données propres : −0,0004 [−0,019 ; +0,024]. L'intervalle contient 0, donc c'est une égalité.
   - On travaille par blocs parce que deux champs voisins se ressemblent. Les tirer un par un donnerait
     une confiance trompeuse.

**À retenir : on joue le rôle du fraudeur pour mesurer combien de fraudes passent inaperçues.**

## 5. `mitigation.py` : le correcteur d'erreurs (IA)

- Un vrai ordinateur quantique est **bruité**. Une ressemblance dont la vraie valeur est 0,80 peut être mesurée à 0,72.
- L'astuce : pour des paires de champs d'entraînement, on **connaît la vraie valeur** grâce à la simulation exacte.
  On mesure ces paires de calibration :
  - mesuré 0,72, vrai 0,80 ;
  - mesuré 0,40, vrai 0,45.
- On trace **une droite** qui passe par ces points, puis on s'en sert pour corriger les nouvelles mesures.
- **Analogie : une balance qui affiche toujours un peu moins.** On pèse des poids connus (1 kg s'affiche 0,9 kg)
  et on en déduit la correction.
- Le bruit est simulé d'après la **vraie calibration de `ibm_fez`**.
- Résultat : l'erreur passe de 0,052 à **0,011**, et de 0,131 à 0,015 quand le bruit est triplé.
- Comparaison avec la **ZNE** (extrapolation à bruit nul) : on refait chaque circuit avec un bruit triplé et on
  prolonge la tendance jusqu'à un bruit nul. Elle fait moins bien (0,032) et demande deux fois plus de circuits.

**À retenir : une simple droite de correction bat une méthode plus coûteuse.
En revanche, il n'est pas prouvé qu'elle améliore la précision finale.**

## 6. `hardware_kernel.py` : le passage sur un vrai processeur IBM

- Le modèle est entraîné sur le PC, puis gelé. Le processeur quantique ne fait que **mesurer des ressemblances**,
  entre 4 champs (un par culture et un champ falsifié) et les « vecteurs de support », c'est-à-dire les
  champs-frontière que le SVM a retenus.
- Des paires de calibration sont envoyées dans le **même job**, si bien que le correcteur est réglé sur la machine du jour.
- **Le piège de `C`.** Avec C = 100, les poids du SVM sont énormes, et une petite erreur de mesure fait basculer
  le vote. C'est comme un **long levier** : un petit tremblement à une extrémité devient un grand mouvement à l'autre.
  Sur `ibm_fez`, seulement 1 vote sur 4 correspondait au simulateur exact.
- **`choose_C`** teste chaque valeur de C hors ligne, avec l'erreur matérielle mesurée. Il garde le plus grand C
  qui conserve au moins 95 % des votes, soit **C = 1**. Sur `ibm_marrakesh`, **4 votes sur 4** correspondaient,
  et l'alarme s'est déclenchée sur le champ falsifié.
- Côté sécurité, il faut `approved=True` pour lancer un job, et le jeton IBM reste dans une variable d'environnement.

**À retenir : sur une machine bruitée, un modèle moins « nerveux » (C petit) donne des votes plus fiables.**

## 7. `analyst.py` : le greffier

Il rédige le procès-verbal, mais ne rend jamais le jugement. Exemple de note hors ligne :

```
VERDICT  FLAGGED - inspectors disagree
VOTES    quantum: wheat, rbf: wheat, forest: maize
LABEL    wheat (majority; RBF SVM settles a three-way split)
ACTION   Hold the claim; request a fresh acquisition or a field visit before approval.
```

Avec un LLM (optionnel) :
- seuls les votes, les étiquettes et les changements de bandes sont envoyés ;
- la réponse est limitée à 2 500 caractères, affichée comme du simple texte et jamais exécutée ;
- seules les adresses en `https` sont acceptées.

## 8. `config.py` : le tableau de bord unique

Il lit trois sources :
- `data/manifest.json` : les faits sur les données (noms des cultures, noms des bandes) ;
- `configs/experiment.json` : les paramètres de l'étude (8 qubits, graines 0, 1, 2…) ;
- `configs/cropguard.json` : les réglages gelés.

Exemple : pour passer à 4 qubits, on change `"qubits": 8` en `4` dans le JSON, **sans toucher au code**.
Des garde-fous refusent plus de 8 qubits (règle du challenge).

## 9. `report.py` : le rédacteur des conclusions

- Il lit `outputs/*.json` et écrit lui-même les phrases avec les chiffres.
- Exemple : si on relance le notebook et que le score devient 0,925, le rapport se met à jour tout seul.
  Aucun chiffre n'est recopié à la main, donc pas d'erreur de recopie.
- Sa fonction `detectable` indique si l'intervalle exclut 0, autrement dit si la différence est réelle.

---

## Récapitulatif : la phrase à retenir

**« Mesurer, Voter, Régler, Attaquer, Corriger, Tester, Raconter, Configurer, Conclure. »**

| Verbe | Fichier |
|---|---|
| Mesurer la ressemblance | `kernel.py` |
| Voter à 3 | `submission.py` |
| Régler puis geler | `tune.py` |
| Attaquer | `guard.py` |
| Corriger le bruit | `mitigation.py` |
| Tester sur IBM | `hardware_kernel.py` |
| Raconter (note d'incident) | `analyst.py` |
| Configurer | `config.py` |
| Conclure | `report.py` |

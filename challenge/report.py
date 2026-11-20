"""Findings written from the saved results (outputs/*.json), so no number is typed by hand. The text is in French.

Run from the repository root:  .venv/bin/python -m challenge.report      (prints Markdown)
The notebook's last section calls findings(); rerunning the notebook rewrites every number.
"""
import json
import re

import pandas as pd

from .config import EXPERIMENT, OUT

pct = lambda x: f'{100 * x:.1f} %'
small_pct = lambda x: f'{100 * x:.2g} %'  # calibration error rates are fractions of a percent
interval = lambda b: f"[{b['ci95'][0]:+.3f} ; {b['ci95'][1]:+.3f}]"
decimal_comma = lambda text: re.sub(r'(?<=\d)\.(?=\d)', ',', text)  # French decimals: 0.921 -> 0,921

# French names of the models, inspectors and crops (the keys stay English in outputs/*.json)
NOMS = {'quantum committee': 'comité quantique', 'matched control': 'contrôle apparié', 'reference': 'référence',
        'quantum SVM': 'SVM quantique', 'poly SVM': 'SVM polynomial', 'RBF SVM': 'SVM RBF', 'forest': 'forêt',
        'quantum': 'SVM quantique', 'poly': 'SVM polynomial', 'rbf': 'SVM RBF',
        'alfalfa': 'luzerne', 'maize': 'maïs', 'wheat': 'blé'}
nom = lambda key: NOMS.get(key, key)


def load(name):
    path = OUT / name
    return json.loads(path.read_text()) if path.exists() else None


def detectable(b):
    """True when the 95% interval of a paired bootstrap excludes zero."""
    return not b['ci95'][0] <= 0 <= b['ci95'][1]


def pair(rows, first, second, **match):
    return next(r for r in rows if r['first'] == first and r['second'] == second and all(r.get(k) == v for k, v in match.items()))


def resources_text(res):
    """Plain-language resources of a fitted committee (Committee.resources())."""
    q = res['members']['quantum']
    circuits = f"{q['training_kernel_circuits']:,}".replace(',', ' ')  # French thousands separator
    return decimal_comma(
        f"**Ressources.** Le membre quantique utilise **{q['qubits']} qubits**, une feature map de profondeur "
        f"{q['feature_map_depth']} avec {q['feature_map_cx']} portes CX (un circuit de noyau est la feature map suivie de son "
        f"inverse : {q['kernel_circuit_cx']} CX avant compilation), **{q['trainable_quantum']} angle quantique entraînable** "
        f"et {q['support_vectors']} vecteurs de support. La simulation exacte calcule un vecteur d'état de "
        f"{2 ** q['qubits']} amplitudes par champ ({q['training_states_simulated']} pour l'entraînement) au lieu des "
        f"{circuits} circuits compute-uncompute qu'un QPU demanderait pour le noyau d'entraînement ; prédire un champ sur un "
        f"QPU demande {q['circuits_per_predicted_field']} circuits, un par vecteur de support. Noyau d'entraînement : "
        f"{q['kernel_storage_bytes'] / 1e6:.1f} Mo. Le prétraitement stocke {res['preprocessing_fitted_values']} valeurs "
        f"apprises. Comparer des nombres de paramètres n'a pas de sens pour des noyaux et des arbres : on rapporte donc les "
        f"vecteurs de support, les valeurs stockées, les nœuds des arbres et le temps de calcul.")


def clean_section(main):
    r = pd.DataFrame(main['records'])
    g = r.groupby('model')
    f1, sd = g.macro_f1.mean(), g.macro_f1.std(ddof=0)
    acc, flagged, unflagged = g.accuracy.mean(), g.flagged.mean(), g.accuracy_unflagged.mean()
    qc = pair(main['bootstrap'], 'matched control', 'quantum committee')
    ref = pair(main['bootstrap'], 'reference', 'quantum committee')
    rbf = pair(main['bootstrap'], 'RBF SVM', 'quantum committee')
    best = f1.drop(['quantum committee', 'matched control']).idxmax()
    verdict = ('aucune différence détectable' if not detectable(qc) else
               f"un {'gain' if qc['delta'] > 0 else 'recul'} détectable pour le comité quantique")
    return (f"**Comparaison sur données propres (résultat principal).** Comité quantique {f1['quantum committee']:.4f} ± "
            f"{sd['quantum committee']:.4f} en macro-F1 ({pct(acc['quantum committee'])} d'exactitude) contre "
            f"{f1['matched control']:.4f} ± {sd['matched control']:.4f} pour le contrôle apparié ; bootstrap apparié "
            f"{qc['delta']:+.4f}, intervalle à 95 % {interval(qc)} : {verdict}. Face à la référence fournie "
            f"({f1['reference']:.4f}), le comité quantique diffère de {ref['delta']:+.3f} {interval(ref)}"
            f"{' (détectable)' if detectable(ref) else ' (l’intervalle contient zéro)'}. Meilleur modèle seul : {nom(best)} "
            f"({f1[best]:.3f}) ; comité quantique moins SVM RBF : {rbf['delta']:+.3f} {interval(rbf)}. L'alarme signale "
            f"{pct(flagged['quantum committee'])} des champs propres, et {pct(unflagged['quantum committee'])} des champs non "
            f"signalés sont bien classés."), verdict


def tamper_section(lab):
    tr = pd.DataFrame(lab['tamper_rows'])
    naive = tr[tr.attack.isin(['none', 'naive'])]
    und = naive.pivot_table(index='alarm', columns='eps', values='undetected')
    flg = naive.pivot_table(index='alarm', columns='eps', values='flagged')
    eps = [e for e in und.columns if e > 0]
    e1 = eps[min(1, len(eps) - 1)]; e2 = eps[min(2, len(eps) - 1)]; top = eps[-1]
    boots = [b for b in lab['bootstrap'] if b['attack'] == 'naive' and b['first'] == 'matched control']
    sig = [b for b in boots if detectable(b)]
    adaptive = tr[tr.attack == 'adaptive'].pivot_table(index='alarm', columns='eps', values='undetected')
    partners = {p: und.loc[p, e2] for p in ['+ quantum SVM', '+ poly SVM', '+ forest']}
    best_partner = min(partners, key=partners.get)
    text = (f"**Falsification et alarme** (attaquant naïf qui vise le SVM RBF). À ε = {e1}, "
            f"{pct(und.loc['RBF SVM alone', e1])} des champs passent inaperçus face au SVM RBF seul, "
            f"{pct(und.loc['+ quantum SVM', e1])} avec un partenaire quantique, {pct(und.loc['+ poly SVM', e1])} avec un "
            f"partenaire polynomial, {pct(und.loc['+ forest', e1])} avec la forêt, {pct(und.loc['quantum committee', e1])} face "
            f"au comité quantique complet et {pct(und.loc['matched control', e1])} face au contrôle apparié ; le comité "
            f"quantique signale {pct(flg.loc['quantum committee', e1])} des champs contre {pct(flg.loc['matched control', e1])}. "
            f"Bootstrap apparié, quantique moins contrôle : " +
            ' ; '.join(f"ε = {b['eps']} : {b['delta']:+.3f} {interval(b)}" for b in boots) +
            f" ({len(sig)} intervalles sur {len(boots)} excluent zéro ; {len(lab['bootstrap'])} comparaisons au total, donc "
            f"chaque intervalle isolé se lit avec prudence). À ε = {e2}, le partenaire qui réduit le plus les passages inaperçus "
            f"est : {nom(best_partner.removeprefix('+ '))} ({pct(und.loc['RBF SVM alone', e2])} → {pct(partners[best_partner])} ; "
            f"SVM quantique {pct(partners['+ quantum SVM'])}, SVM polynomial {pct(partners['+ poly SVM'])}, forêt "
            f"{pct(partners['+ forest'])}). Un attaquant qui connaît les trois inspecteurs fait passer "
            f"{pct(adaptive.loc['quantum committee', top])} (comité quantique) et {pct(adaptive.loc['matched control', top])} "
            f"(contrôle) des champs à ε = {top}.")
    return text, best_partner


def member_section(lab):
    tr = pd.DataFrame(lab['tamper_rows'])
    transfer = tr[tr.attack == 'naive transfer'].pivot_table(index='alarm', columns='eps', values='accuracy')
    pr = pd.DataFrame(lab['poison_rows']).pivot_table(index='model', columns='rate', values='macro_f1')
    mr = pd.DataFrame(lab['missing_rows'])
    mf = mr[mr.macro_f1.notna()].pivot_table(index='model', columns='missing', values='macro_f1')
    ma = mr[mr.macro_f1.isna()].pivot_table(index='model', columns='missing', values='flagged')
    e, rate, k = transfer.columns[min(1, len(transfer.columns) - 1)], pr.columns[-1], mf.columns[-1]
    members = ['quantum SVM', 'poly SVM', 'RBF SVM', 'forest']
    worst_poison, worst_missing = pr.loc[members, rate].idxmin(), mf.loc[members, k].idxmin()
    return (f"**Inspecteurs seuls.** Sur les champs falsifiés contre le SVM RBF (ε = {e}), le SVM quantique garde "
            f"{transfer.loc['quantum SVM', e]:.3f} d'exactitude contre {transfer.loc['poly SVM', e]:.3f} pour le SVM polynomial. "
            f"Avec {pct(rate)} d'étiquettes empoisonnées : " + ' ; '.join(f'{nom(m)} {pr.loc[m, rate]:.3f}' for m in members) +
            f" (le plus bas : {nom(worst_poison)}). Avec {k} bandes manquantes : " +
            ' ; '.join(f'{nom(m)} {mf.loc[m, k]:.3f}' for m in members) +
            f" (le plus bas : {nom(worst_missing)}). Comités : {pr.loc['quantum committee', rate]:.3f} / "
            f"{pr.loc['matched control', rate]:.3f} avec empoisonnement et {mf.loc['quantum committee', k]:.3f} / "
            f"{mf.loc['matched control', k]:.3f} avec bandes manquantes (le comité quantique signale "
            f"{pct(ma.loc['quantum committee alarm', k])} des champs contre {pct(ma.loc['matched control alarm', k])}). "
            f"Description seulement : aucun bootstrap n'a été fait pour l'empoisonnement ni pour les bandes manquantes.")


def noise_section(noise, hw):
    rows = noise['rows']; p = noise['profile']
    text = (f"**Correcteur IA sur un processeur simulé à partir d'une vraie calibration** ({p['device']}, relevée le "
            f"{p['retrieved_utc']} : erreur à deux qubits {small_pct(p['two_qubit_error'])}, à un qubit "
            f"{small_pct(p['sx_error'])}, de lecture {small_pct(p['readout_error'])}, T1 {p['t1_seconds'] * 1e6:.0f} µs, "
            f"T2 {p['t2_seconds'] * 1e6:.0f} µs). Erreur de similarité par rapport à la valeur exacte, brute → corrigée : " +
            ' ; '.join(f"{r['level']}× {r['mae_raw']:.3f} → {r['mae_fixed']:.3f} (ZNE {r['mae_zne']:.3f})" for r in rows) +
            f". Une seule droite, ajustée sur {noise['n_cal']} paires de calibration ; la ZNE coûte deux fois plus de "
            f"circuits. Le test de classification sur {noise['n_val']} champs est trop petit pour conclure quoi que ce soit "
            f"sur l'exactitude.")
    if hw:
        text += (f" Vérification de réalisme : le modèle à calibration médiane prévoit une erreur brute de "
                 f"{rows[0]['mae_raw']:.3f} à 1× ; le vrai run sur {hw['backend']} a mesuré {hw['mae_raw']:.3f} (autres paires "
                 f"de champs, et un vrai processeur ajoute des écarts entre qubits, de la diaphonie et de la dérive).")
    return text


def hardware_section(hw, robust, scan, chosen):
    def votes(h):
        return sum(f['quantum_hardware_fixed'] == f['quantum_exact'] for f in h['fields']), len(h['fields'])
    a, n = votes(hw)
    sub_C = hw.get('quantum_params', {}).get('C', max(r['C'] for r in scan))
    row = {r['C']: r for r in scan}
    text = (f"**Matériel IBM (inférence seulement).** `{hw['backend']}` (job `{hw['job_id']}`, modèle soumis, C = {sub_C}) : "
            f"le correcteur, ajusté dans le même job, a fait passer l'erreur du noyau de {hw['mae_raw']:.3f} à "
            f"{hw['mae_fixed']:.3f} ; votes quantiques identiques au simulateur exact : {a} sur {n}. Analyse hors ligne : le "
            f"seul bruit de tir garde {pct(row[sub_C]['agree_shot_noise'])} des votes de ce modèle "
            f"({pct(row[sub_C]['agree_hardware_error'])} sous l'erreur matérielle mesurée), car un grand C donne de grands "
            f"poids duaux qui amplifient les erreurs du noyau. Règle : le plus grand C qui garde au moins "
            f"{pct(EXPERIMENT['hardware']['min_vote_agreement'])} des votes sous l'erreur mesurée → C = {chosen}"
            + (f" ({pct(row[chosen]['agree_hardware_error'])} ; macro-F1 exact {row[chosen]['exact_macro_f1']:.3f} au lieu de "
               f"{row[sub_C]['exact_macro_f1']:.3f})." if chosen is not None else " (aucun candidat ne convient)."))
    if robust:
        b, m = votes(robust)
        tampered = [f for f in robust['fields'] if f['name'].startswith('tampered')]
        text += (f" `{robust['backend']}` (job `{robust['job_id']}`, C = {robust.get('quantum_params', {}).get('C', chosen)}) : "
                 f"votes identiques au simulateur : {b} sur {m}.")
        for f in tampered:
            if f['quantum_hardware_raw'] != f['quantum_exact'] and f['quantum_hardware_fixed'] == f['quantum_exact']:
                text += (f" Sur le champ falsifié, le vote brut du processeur était faux ({nom(f['quantum_hardware_raw'])}) et "
                         f"le correcteur l'a rétabli ({nom(f['quantum_hardware_fixed'])}) : l'alarme s'est donc déclenchée sur "
                         f"un vrai processeur.")
    return text + " Un petit exemple d'inférence par modèle, pas une estimation de l'exactitude."


def findings(main=None, lab=None, noise=None, hw=None, robust=None, scan=None, chosen=None, runtime=None):
    main, lab, noise = main or load('main_results.json'), lab or load('attack_lab.json'), noise or load('noise_study.json')
    hw, robust = hw or load('hardware_result.json'), robust or load('hardware_result_robust.json')
    scan = scan or (load('hardware_robustness.json') or {}).get('rows')
    if scan and chosen is None:
        from .hardware_kernel import choose_C
        chosen = choose_C(scan)
    clean, verdict = clean_section(main)
    parts = [clean]
    answer = f"Sur données propres : {verdict} entre le comité quantique et un comité classique réglé avec le même effort."
    if lab:
        tamper, best = tamper_section(lab)
        parts += [tamper, member_section(lab)]
        answer += f" Face à la falsification, le partenaire qui protège le plus est : {nom(best.removeprefix('+ '))}."
    if noise: parts.append(noise_section(noise, hw))
    if hw and scan: parts.append(hardware_section(hw, robust, scan, chosen))
    tuning = load('tuning_log.json')
    if tuning:
        parts.append("**Effort de réglage.** " + ' ; '.join(f"{nom(k)} : {v['configurations']} configurations en {v['seconds']:.1f} s"
                                                           for k, v in tuning.items()) +
                     " (macro-F1 en validation ; `outputs/tuning_log.json`).")
    if runtime: parts.append(f"**Durée.** Partie principale {main['main_run_seconds']:.0f} s ; notebook complet {runtime:.0f} s.")
    return decimal_comma('\n\n'.join([f'**Réponse.** {answer}'] + [f'{i}. {p}' for i, p in enumerate(parts, 1)]))


if __name__ == '__main__':
    print(findings())

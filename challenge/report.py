"""Findings written from the saved results (outputs/*.json), so no number is typed by hand.

Run from the repository root:  .venv/bin/python -m challenge.report      (prints Markdown)
The notebook's last section calls findings(); rerunning the notebook rewrites every number.
"""
import json

import pandas as pd

from .config import EXPERIMENT, OUT

pct = lambda x: f'{100 * x:.1f}%'
small_pct = lambda x: f'{100 * x:.2g}%'  # calibration error rates are fractions of a percent
interval = lambda b: f"[{b['ci95'][0]:+.3f}, {b['ci95'][1]:+.3f}]"


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
    return (f"**Resources.** The quantum member uses **{q['qubits']} qubits**, a feature map of depth {q['feature_map_depth']} "
            f"with {q['feature_map_cx']} CX gates (a kernel circuit is the map followed by its inverse: {q['kernel_circuit_cx']} "
            f"CX before compilation), **{q['trainable_quantum']} trainable quantum angles** and {q['support_vectors']} support "
            f"vectors. Exact simulation computes one {2 ** q['qubits']}-amplitude statevector per field "
            f"({q['training_states_simulated']} for training) instead of the {q['training_kernel_circuits']:,} compute-uncompute "
            f"circuits a QPU would need for the training kernel; predicting one field on a QPU takes "
            f"{q['circuits_per_predicted_field']} circuits, one per support vector. Training kernel: "
            f"{q['kernel_storage_bytes'] / 1e6:.1f} MB. Preprocessing stores {res['preprocessing_fitted_values']} fitted values. "
            f"Parameter matching is not required for kernels and trees, so support vectors, stored values, tree nodes and "
            f"runtime are reported instead.")


def clean_section(main):
    r = pd.DataFrame(main['records'])
    g = r.groupby('model')
    f1, sd = g.macro_f1.mean(), g.macro_f1.std(ddof=0)
    acc, flagged, unflagged = g.accuracy.mean(), g.flagged.mean(), g.accuracy_unflagged.mean()
    qc = pair(main['bootstrap'], 'matched control', 'quantum committee')
    ref = pair(main['bootstrap'], 'reference', 'quantum committee')
    rbf = pair(main['bootstrap'], 'RBF SVM', 'quantum committee')
    best = f1.drop(['quantum committee', 'matched control']).idxmax()
    verdict = ('no detectable difference' if not detectable(qc) else
               f"a detectable {'gain' if qc['delta'] > 0 else 'loss'} for the quantum committee")
    return (f"**Clean comparison (main result).** Quantum committee {f1['quantum committee']:.4f} ± {sd['quantum committee']:.4f} "
            f"macro-F1 ({pct(acc['quantum committee'])} accuracy) vs matched control {f1['matched control']:.4f} ± "
            f"{sd['matched control']:.4f}; paired bootstrap {qc['delta']:+.4f}, 95% interval {interval(qc)}: {verdict}. "
            f"Against the supplied reference ({f1['reference']:.4f}) the quantum committee differs by {ref['delta']:+.3f} "
            f"{interval(ref)}{' (detectable)' if detectable(ref) else ' (interval includes zero)'}. The best single model is the "
            f"{best} ({f1[best]:.3f}); quantum committee minus RBF SVM {rbf['delta']:+.3f} {interval(rbf)}. "
            f"The alarm flags {pct(flagged['quantum committee'])} of clean fields, and {pct(unflagged['quantum committee'])} "
            f"of the unflagged ones are right."), verdict


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
    text = (f"**Tampering and the alarm** (naive attacker aiming at the RBF SVM). At ε = {e1}, "
            f"{pct(und.loc['RBF SVM alone', e1])} of fields slip through the RBF SVM alone, "
            f"{pct(und.loc['+ quantum SVM', e1])} with a quantum partner, {pct(und.loc['+ poly SVM', e1])} with a polynomial "
            f"partner, {pct(und.loc['+ forest', e1])} with the forest, {pct(und.loc['quantum committee', e1])} through the full "
            f"quantum committee and {pct(und.loc['matched control', e1])} through the matched control; the quantum committee "
            f"flags {pct(flg.loc['quantum committee', e1])} vs {pct(flg.loc['matched control', e1])}. Paired bootstrap, quantum "
            f"minus control: " + '; '.join(f"ε = {b['eps']}: {b['delta']:+.3f} {interval(b)}" for b in boots) +
            f" ({len(sig)} of {len(boots)} intervals exclude zero; {len(lab['bootstrap'])} comparisons in total, so read single "
            f"intervals with care). At ε = {e2} the partner that cuts slips most is the {best_partner.removeprefix('+ ')} "
            f"({pct(und.loc['RBF SVM alone', e2])} → {pct(partners[best_partner])}; quantum SVM {pct(partners['+ quantum SVM'])}, "
            f"poly SVM {pct(partners['+ poly SVM'])}, forest {pct(partners['+ forest'])}). An attacker who knows all three "
            f"inspectors gets {pct(adaptive.loc['quantum committee', top])} (quantum) and {pct(adaptive.loc['matched control', top])} "
            f"(control) of fields through at ε = {top}.")
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
    return (f"**Single inspectors.** On fields tampered against the RBF SVM (ε = {e}) the quantum SVM keeps "
            f"{transfer.loc['quantum SVM', e]:.3f} accuracy vs {transfer.loc['poly SVM', e]:.3f} for the poly SVM. With "
            f"{pct(rate)} poisoned labels: " + ', '.join(f'{m} {pr.loc[m, rate]:.3f}' for m in members) +
            f" (lowest: {worst_poison}). With {k} bands missing: " + ', '.join(f'{m} {mf.loc[m, k]:.3f}' for m in members) +
            f" (lowest: {worst_missing}). Committees: {pr.loc['quantum committee', rate]:.3f} / {pr.loc['matched control', rate]:.3f} "
            f"with poisoning and {mf.loc['quantum committee', k]:.3f} / {mf.loc['matched control', k]:.3f} with missing bands "
            f"(quantum committee flags {pct(ma.loc['quantum committee alarm', k])} vs {pct(ma.loc['matched control alarm', k])}). "
            f"Descriptive only: no bootstrap was run for poisoning or missing bands.")


def noise_section(noise, hw):
    rows = noise['rows']; p = noise['profile']
    text = (f"**AI error-fixer on a simulated device built from real calibration** ({p['device']}, retrieved {p['retrieved_utc']}: "
            f"two-qubit error {small_pct(p['two_qubit_error'])}, single-qubit {small_pct(p['sx_error'])}, "
            f"readout {small_pct(p['readout_error'])}, "
            f"T1 {p['t1_seconds'] * 1e6:.0f} µs, T2 {p['t2_seconds'] * 1e6:.0f} µs). Similarity error vs exact, raw → fixed: " +
            '; '.join(f"{r['level']}× {r['mae_raw']:.3f} → {r['mae_fixed']:.3f} (ZNE {r['mae_zne']:.3f})" for r in rows) +
            f". One straight line fitted on {noise['n_cal']} calibration pairs; ZNE costs twice the circuits. The "
            f"{noise['n_val']}-field classification check is too small to support any accuracy claim.")
    if hw:
        text += (f" Realism check: the median-calibration model predicts a raw error of {rows[0]['mae_raw']:.3f} at 1×; the real "
                 f"{hw['backend']} run measured {hw['mae_raw']:.3f} (different field pairs, and the real device adds per-qubit "
                 f"spread, crosstalk and drift).")
    return text


def hardware_section(hw, robust, scan, chosen):
    def votes(h):
        return sum(f['quantum_hardware_fixed'] == f['quantum_exact'] for f in h['fields']), len(h['fields'])
    a, n = votes(hw)
    sub_C = hw.get('quantum_params', {}).get('C', max(r['C'] for r in scan))
    row = {r['C']: r for r in scan}
    text = (f"**IBM hardware (inference only).** `{hw['backend']}` (job `{hw['job_id']}`, submitted model, C = {sub_C}): the in-job "
            f"error-fixer took the kernel error from {hw['mae_raw']:.3f} to {hw['mae_fixed']:.3f}; {a} of {n} quantum votes matched "
            f"the exact simulator. Offline scan: shot noise alone keeps {pct(row[sub_C]['agree_shot_noise'])} of this model's votes "
            f"({pct(row[sub_C]['agree_hardware_error'])} under the measured hardware error), because large C means large dual "
            f"weights that amplify kernel errors. Rule: the largest C keeping at least "
            f"{pct(EXPERIMENT['hardware']['min_vote_agreement'])} of votes under the measured error → C = {chosen}"
            + (f" ({pct(row[chosen]['agree_hardware_error'])}; exact macro-F1 {row[chosen]['exact_macro_f1']:.3f} instead of "
               f"{row[sub_C]['exact_macro_f1']:.3f})." if chosen is not None else " (no candidate qualifies)."))
    if robust:
        b, m = votes(robust)
        tampered = [f for f in robust['fields'] if f['name'].startswith('tampered')]
        text += (f" `{robust['backend']}` (job `{robust['job_id']}`, C = {robust.get('quantum_params', {}).get('C', chosen)}): {b} of {m} "
                 f"votes matched.")
        for f in tampered:
            if f['quantum_hardware_raw'] != f['quantum_exact'] and f['quantum_hardware_fixed'] == f['quantum_exact']:
                text += (f" On the tampered field the raw hardware vote was wrong ({f['quantum_hardware_raw']}) and the error-fixer "
                         f"restored it ({f['quantum_hardware_fixed']}), so the alarm fired on a real device.")
    return text + ' One small inference example per model, not an accuracy estimate.'


def findings(main=None, lab=None, noise=None, hw=None, robust=None, scan=None, chosen=None, runtime=None):
    main, lab, noise = main or load('main_results.json'), lab or load('attack_lab.json'), noise or load('noise_study.json')
    hw, robust = hw or load('hardware_result.json'), robust or load('hardware_result_robust.json')
    scan = scan or (load('hardware_robustness.json') or {}).get('rows')
    if scan and chosen is None:
        from .hardware_kernel import choose_C
        chosen = choose_C(scan)
    clean, verdict = clean_section(main)
    parts = [clean]
    answer = (f"On clean data: {verdict} between the quantum committee and an equally tuned classical one.")
    if lab:
        tamper, best = tamper_section(lab)
        parts += [tamper, member_section(lab)]
        answer += f" Under tampering, the partner that protects most is the {best.removeprefix('+ ')}."
    if noise: parts.append(noise_section(noise, hw))
    if hw and scan: parts.append(hardware_section(hw, robust, scan, chosen))
    tuning = load('tuning_log.json')
    if tuning:
        parts.append('**Tuning effort.** ' + ', '.join(f"{k}: {v['configurations']} configurations in {v['seconds']:.1f} s"
                                                       for k, v in tuning.items()) + ' (validation macro-F1; `outputs/tuning_log.json`).')
    if runtime: parts.append(f"**Runtime.** Main run {main['main_run_seconds']:.0f} s; whole notebook {runtime:.0f} s.")
    return '\n\n'.join([f'**Answer.** {answer}'] + [f'{i}. {p}' for i, p in enumerate(parts, 1)])


if __name__ == '__main__':
    print(findings())

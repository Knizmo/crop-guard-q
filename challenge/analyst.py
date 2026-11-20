"""AI analyst for the console app: writes a short incident note about one inspected field.
It explains a verdict; it never changes a prediction or the flag. Not part of the submitted workflow.

Offline by default (a fixed template). To let a language model write the note, point CROPGUARD_LLM_URL at any
OpenAI-compatible chat-completions endpoint, for example:
  free, no key:  CROPGUARD_LLM_URL=https://text.pollinations.ai/openai
  Groq:          CROPGUARD_LLM_URL=https://api.groq.com/openai/v1/chat/completions  CROPGUARD_LLM_MODEL=llama-3.3-70b-versatile
  local Ollama:  CROPGUARD_LLM_URL=http://localhost:11434/v1/chat/completions      CROPGUARD_LLM_MODEL=llama3.2
and put a key, if the service needs one, in CROPGUARD_LLM_KEY (environment only, never in a file).
Only the verdict and numbers below are sent: no credentials, no coordinates.
"""
import json
import os
import urllib.request
from urllib.parse import urlparse

BANDS = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12', 'NDVI']
MAX_CHARS = 2500  # model output is untrusted text: shown as plain text, capped, never executed

SYSTEM = ('You are a security analyst for a satellite crop-claim checker. Three independent inspectors (an 8-qubit '
          'quantum-kernel SVM, a classical RBF SVM and a random forest) classified one field. Write a short incident '
          'note (max 120 words): verdict, why, risk, recommended next step for a human reviewer. Be factual; do not '
          'claim the quantum model is more robust or accurate; never invent data.')


def facts(info):
    """The only data that leaves the machine: labels, votes and the largest band changes (in training SDs)."""
    out = {k: info[k] for k in ('sample_id', 'votes', 'final', 'flagged') if k in info}
    if info.get('true') is not None: out['surveyed_crop'] = info['true']
    if info.get('band_shift') is not None:
        top = sorted(zip(BANDS, info['band_shift']), key=lambda b: -abs(b[1]))[:3]
        out['largest_band_changes_sd'] = {b: round(float(v), 3) for b, v in top}
        out['tamper_budget_sd'] = info.get('eps')
    return out


def offline_note(info):
    f = facts(info)
    votes = ', '.join(f'{k}: {v}' for k, v in f['votes'].items())
    lines = [f"VERDICT  {'FLAGGED - inspectors disagree' if f['flagged'] else 'VERIFIED - unanimous'}",
             f'VOTES    {votes}', f"LABEL    {f['final']} (majority; RBF SVM settles a three-way split)"]
    if 'surveyed_crop' in f:
        lines.append(f"SURVEY   {f['surveyed_crop']} -> {'match' if f['surveyed_crop'] == f['final'] else 'MISMATCH'}")
    if 'largest_band_changes_sd' in f:
        lines.append('CHANGES  ' + ', '.join(f'{b} {v:+.3f} SD' for b, v in f['largest_band_changes_sd'].items()))
    lines.append('ACTION   ' + ('Hold the claim; request a fresh acquisition or a field visit before approval.'
                                if f['flagged'] else 'No alarm. A coordinated attack can still fool all three inspectors.'))
    return '\n'.join(lines)


def endpoint():
    url = os.environ.get('CROPGUARD_LLM_URL', '').strip()
    if not url: return None
    p = urlparse(url)
    if p.scheme != 'https' and p.hostname not in ('localhost', '127.0.0.1'):
        raise ValueError('CROPGUARD_LLM_URL must use https (http only for localhost)')
    return url


def llm_note(info, timeout=45):
    url = endpoint()
    body = {'model': os.environ.get('CROPGUARD_LLM_MODEL', 'openai'), 'temperature': 0.2,
            'messages': [{'role': 'system', 'content': SYSTEM},
                         {'role': 'user', 'content': json.dumps(facts(info))}]}
    headers = {'Content-Type': 'application/json', 'User-Agent': 'CropGuard-Q/1.0'}
    if os.environ.get('CROPGUARD_LLM_KEY'): headers['Authorization'] = 'Bearer ' + os.environ['CROPGUARD_LLM_KEY']
    req = urllib.request.Request(url, json.dumps(body).encode(), headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        text = json.loads(r.read())['choices'][0]['message']['content']
    text = str(text).replace('**', '').replace('__', '')  # plain-text display: drop markdown emphasis
    return ''.join(ch for ch in text if ch.isprintable() or ch in '\n\t')[:MAX_CHARS]


def note(info):
    """(text, source). Falls back to the offline template on any network or format error."""
    try:
        if endpoint(): return llm_note(info), 'llm:' + urlparse(endpoint()).hostname
    except Exception as e:
        return offline_note(info) + f'\n\n[language model unavailable: {type(e).__name__}]', 'offline'
    return offline_note(info), 'offline'

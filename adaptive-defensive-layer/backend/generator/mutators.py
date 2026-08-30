"""Strategy mutation + graph-structure construction.

Given a freshly sampled behaviour vector the mutator decides:

  * how the new account differs from its parent strategy (new device? VPN?
    shifted amounts? changed timing?) and records human-readable tags,
  * whether the account participates in a fraud ring / referral chain or is
    attached to familiar (previously attacked) victims.

The resulting fraud "strategy name" is ``parent + mutation-tags`` which lets
the whole evolution of attacker behaviour be traced round after round.
"""

from ..world import INTRINSIC_NAMES, _BINARY_INTRINSIC

BINARY_INTRINSIC = _BINARY_INTRINSIC


def row_to_attrs(row, rng):
    """Turn a numeric behaviour row (in INTRINSIC_NAMES order) into a dict."""
    attrs = {}
    for j, name in enumerate(INTRINSIC_NAMES):
        v = float(row[j])
        if name in BINARY_INTRINSIC:
            attrs[name] = 1.0 if v >= 0.5 else 0.0
        else:
            attrs[name] = max(0.0, min(1.0, v))
    return attrs


def _tag_for(name, source_val, new_val, threshold=0.12):
    if abs(new_val - source_val) < threshold:
        return None
    if new_val > source_val:
        return name
    return None


def mutate_spec(row, source, rng, diversity=1.0, ring=True, victims=()):
    """Build one new-fraud spec from a sampled row and its parent strategy.

    ``source`` is a dict describing the parent missed-fraud account:
        {'attrs': {...}, 'device_id': int, 'ip_id': int, 'base': str}
    """
    attrs = row_to_attrs(row, rng)
    tags = []
    src_attrs = source['attrs']
    base = source.get('base', 'evolved')

    # --- behaviour changes vs the parent -------------------------------
    if _tag_for('device_fresh', src_attrs.get('device_fresh', 0.0), attrs['device_fresh']):
        tags.append('new_device')
    if _tag_for('ip_proxy', src_attrs.get('ip_proxy', 0.0), attrs['ip_proxy']):
        tags.append('vpn')
    if _tag_for('email_disposable', src_attrs.get('email_disposable', 0.0),
                attrs['email_disposable']):
        tags.append('new_email')
    if _tag_for('amount_mean', src_attrs.get('amount_mean', 0.0), attrs['amount_mean']):
        tags.append('amount_shift')
    if _tag_for('login_night', src_attrs.get('login_night', 0.0), attrs['login_night']):
        tags.append('timing_shift')

    # --- environment reuse vs refresh ----------------------------------
    device = None
    if attrs['device_fresh'] >= 0.5 or rng.random() < 0.55:
        tags.append('new_device') if 'new_device' not in tags else None
    else:
        device = source.get('device_id')
        if device is not None:
            tags.append('same_device')

    ip = None
    if attrs['ip_proxy'] >= 0.5 or rng.random() < 0.45:
        if 'vpn' not in tags:
            tags.append('new_ip')
    else:
        ip = source.get('ip_id')
        if ip is not None:
            tags.append('same_ip')

    tags = sorted(set(tags))
    strategy = '+'.join([base] + tags) if tags else base

    return {
        'attrs': attrs,
        'base': base,
        'tags': tags,
        'strategy': strategy,
        'device': device,
        'ip': ip,
        'spray': source.get('spray', 0.3),
        'ip_reuse': source.get('ip_reuse', 0.4),
        'ring': ring,
        'victims': list(victims),
        'referrals': [],
    }


def build_structure(specs, rng, conn_coef=0.6, ring_ratio=0.5):
    """Connect a batch of new-fraud specs.

    * ring members -> referral chains / rings between the new accounts
    * others       -> attach to familiar victims

    Returns the list of ``(src, dst)`` referral edges.
    """
    edges = []
    ring_idx = [i for i, s in enumerate(specs) if s.get('ring')]
    victim_idx = [i for i, s in enumerate(specs) if not s.get('ring')]

    # chains / rings
    if len(ring_idx) > 1:
        for pos, i in enumerate(ring_idx):
            nxt = ring_idx[(pos + 1) % len(ring_idx)]
            specs[i]['referrals'].append(nxt)
            edges.append((i, nxt))
            if rng.random() < 0.4 and len(ring_idx) > 3:
                j = rng.choice([k for k in ring_idx if k != i and k != nxt])
                specs[i]['referrals'].append(j)
                edges.append((i, j))

    # victim attachments
    for i in victim_idx:
        victims = specs[i].get('victims') or []
        if not victims:
            continue
        k = max(1, min(int(round(len(victims) * conn_coef)), len(victims)))
        chosen = rng.choice(victims, size=k, replace=False).tolist()
        specs[i]['victim_referrals'] = list(chosen)
        for v in chosen:
            edges.append((i, v))

    return edges

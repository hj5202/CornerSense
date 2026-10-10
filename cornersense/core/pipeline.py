"""CornerSense direction and range estimation from CSI recordings.

Preloaded CSI has shape (packets, 3 antennas, 30 tones), with times in seconds.
motion_geometry contains path angles in radians and robot speed in m/s;
Dphys, tr and Ig are in metres.
"""
import copy
import numpy as np
from scipy.signal import butter, filtfilt

from .intel5300 import read_bf_file


class _Constants:
    C = 2.99792458e8
    DF = 625e3
    FC = 5.32e9
    V_M = 0.2
V = _Constants()

K = 30
C = V.C
DF = V.DF
FC = V.FC
KG = (np.arange(K) - (K - 1) / 2).astype(float)

BAND_LO, BAND_HI = 0.8, 8.0


def hp2d(P, fs, fc=0.4, order=2):
    b, a = butter(order, fc, btype='high', fs=fs)
    return filtfilt(b, a, P, axis=0)


def ma2d(P, fs, sec):
    w = max(3, int(sec * fs) | 1)
    ker = np.ones(w) / w
    pad = w // 2
    Pp = np.pad(P, ((pad, pad), (0, 0)), mode='reflect')
    return np.apply_along_axis(lambda v: np.convolve(v, ker, mode='valid'),
                               0, Pp)


def bandlimit_t(X, fs, flo, fhi, order=4):
    b, a = butter(order, [flo, fhi], btype='band', fs=fs)
    return filtfilt(b, a, X, axis=0)


def scan_windows(H, fs, win_s=1.5, hop_s=0.5, flo=BAND_LO):
    N = H.shape[0]
    W = int(win_s * fs)
    hop = int(hop_s * fs)
    out = []
    for st in range(0, N - W + 1, hop):
        best = None
        for a, b in [(0, 1), (1, 2), (0, 2)]:
            P = H[st:st + W, a, :] * np.conj(H[st:st + W, b, :])

            Php = hp2d(P, fs)
            Xf = np.fft.fft(Php * np.hanning(W)[:, None], axis=0)
            S = (np.abs(Xf) ** 2).sum(axis=1)

            bmin = max(1, int(np.ceil(flo * W / fs)))
            S[:bmin] = 0
            S[W - bmin:] = 0
            bstar = int(np.argmax(S))
            if S[bstar] <= 0:
                continue
            prom = S[bstar] / (np.median(S[S > 0]) + 1e-12)
            fbeat = bstar * fs / W
            if fbeat > fs / 2:
                fbeat = fs - fbeat
            if best is None or prom > best['prom']:
                best = dict(start=st, W=W, pair=(a, b), prom=prom,
                            fbeat=fbeat)
        if best:
            out.append(best)
    return out


def select_segments(wins, N, fs, k=1, min_len_s=2.5, prom_thr=4.0,
                    ftol=2.0):
    runs, cur = [], []
    for w in wins + [None]:
        if w is not None and w['prom'] >= prom_thr:
            cur.append(w)
        else:
            if cur:
                runs.append(cur)
            cur = []
    segs = []
    for run in sorted(runs, key=len, reverse=True):
        fmed = np.median([w['fbeat'] for w in run])
        run = [w for w in run if abs(w['fbeat'] - fmed) <= ftol]
        if not run:
            continue
        st = run[0]['start']
        en = min(N, run[-1]['start'] + run[-1]['W'])
        if (en - st) / fs < min_len_s:
            continue
        segs.append((st, en, run))
        if len(segs) >= k:
            break
    return segs or None


def beat_census(Hs, fs, nkeys=4, fmin=BAND_LO, fmax=BAND_HI, guard_hz=0.5):
    N = Hs.shape[0]
    fr = np.fft.fftfreq(N, 1 / fs)
    keys = []
    for a, b in [(0, 1), (1, 2), (0, 2)]:
        Php = hp2d(Hs[:, a, :] * np.conj(Hs[:, b, :]), fs)
        Xf = np.fft.fft(Php * np.hanning(N)[:, None], axis=0)
        S = (np.abs(Xf) ** 2).sum(axis=1)
        Sm = S.copy()
        Sm[(np.abs(fr) < fmin) | (np.abs(fr) > fmax)] = 0
        floor = np.median(Sm[Sm > 0]) + 1e-12
        g = int(guard_hz / (fr[1] - fr[0]))
        for _ in range(nkeys):
            bb = int(np.argmax(Sm))
            if Sm[bb] <= 0:
                break
            keys.append(dict(pair=(a, b), bin=bb, f=fr[bb],
                             prom=Sm[bb] / floor, Xf=Xf, Php=Php))
            Sm[max(0, bb - g):bb + g + 1] = 0
    keys.sort(key=lambda d: -d['prom'])
    return keys


def track_key(key, fs, bw=0.35, win_s=1.5, hop_s=0.25, ts=None):
    Xf, Php = key['Xf'], key['Php']
    N = Php.shape[0]
    if ts is None:
        ts = np.arange(N) / fs
    ts = np.asarray(ts, dtype=float)
    fr = np.fft.fftfreq(N, 1 / fs)
    b0 = key['bin']
    v = Xf[b0]
    sel = np.abs(fr - key['f']) <= bw
    Xb = np.zeros_like(Xf)
    Xb[sel] = Xf[sel]
    bt = np.fft.ifft(Xb, axis=0)
    bs = (bt * np.conj(v)[None, :]).sum(axis=1)
    Phi = np.unwrap(np.angle(bs))
    amp = np.abs(bs)

    W = int(win_s * fs)
    hop = int(hop_s * fs)
    han = np.hanning(W)[:, None]

    Ds, phw, tcw = [], [], []
    for s0 in range(0, N - W + 1, hop):
        tw = ts[s0:s0 + W] - ts[s0]
        if tw[-1] > 1.5 * win_s:
            continue
        f_use = key['f']
        E = np.exp(-2j * np.pi * f_use * tw)[:, None]
        vw = ((Php[s0:s0 + W] * han) * E).sum(axis=0)
        w = np.abs(vw) ** 2
        if w.max() < 1e-9 or w.max() < 3 * np.median(w):
            continue
        from .adjacent_metric import weighted_adjacent_slope
        fit = weighted_adjacent_slope(KG * DF, np.unwrap(np.angle(vw)), w)
        Ds.append(fit["slope"] * C / (2 * np.pi))

        t_mid = 0.5 * (ts[s0] + ts[s0 + W - 1])
        phw.append(np.angle(vw.sum()) + 2 * np.pi * f_use * (t_mid - ts[s0]))
        tcw.append(t_mid)
    aux = {}
    if len(Ds) >= 5:
        D = float(np.median(Ds))
        D_iqr = float(np.percentile(Ds, 75) - np.percentile(Ds, 25))
    else:
        D, D_iqr = float('nan'), float('nan')
    if len(phw) >= 8:
        z = np.exp(1j * np.array(phw))
        tt = np.array(tcw)
        dphi = np.angle(z[1:] * np.conj(z[:-1])
                        * np.exp(-2j * np.pi * key['f'] * np.diff(tt)))
        phi_abs = np.concatenate([[0.0], np.cumsum(dphi)]) \
            + 2 * np.pi * key['f'] * (tt - tt[0])
        A = np.vstack([tt, np.ones_like(tt)]).T
        slope = np.linalg.lstsq(A, phi_abs, rcond=None)[0][0]
        aux['trend_ts'] = float(slope * (tt[-1] - tt[0]) * C / (2 * np.pi * FC))
    else:
        aux['trend_ts'] = float('nan')
    return Phi, amp, D, D_iqr, aux


def stage_pca(P2):
    Pc = P2 - P2.mean(axis=0)
    U, s, Vt = np.linalg.svd(Pc, full_matrices=False)
    scores = U[:, :2] * s[:2]
    return scores[:, 0], Vt[0], s


def pca_beat_match(score, keys_tracked, fs, N, bw=0.35, ts=None):
    fr = np.fft.fftfreq(N, 1 / fs)
    Sf = np.fft.fft(score * np.hanning(N))
    t = (np.arange(N) / fs) if ts is None else np.asarray(ts, float)
    t = t - t[0]
    best = None
    for i, kt in enumerate(keys_tracked):
        f0 = abs(kt['f'])
        sel = (fr >= f0 - bw) & (fr <= f0 + bw)
        if not sel.any():
            continue
        Zb = np.zeros_like(Sf)
        Zb[sel] = Sf[sel]
        za = 2.0 * np.fft.ifft(Zb)
        phz = np.unwrap(np.angle(za))
        w = np.abs(za) * kt['amp']
        w = w / (w.sum() + 1e-12)
        for sgn in (1, -1):
            d = phz - sgn * kt['Phi']
            for df in np.arange(-0.08, 0.081, 0.02):
                lock = np.abs(np.sum(w * np.exp(1j * (d - 2 * np.pi * df * t))))
                if best is None or lock > best[1]:
                    chi = np.angle(np.sum(
                        w * np.exp(1j * (d - 2 * np.pi * df * t))))
                    best = (i, lock, chi, sgn)
    return best if best else (None, 0.0, 0.0, 1)

GAP_S = 0.05


def split_at_gaps(st, en, ts, min_len_s=2.5, gap_s=GAP_S):
    gaps = np.where(np.diff(ts) > gap_s)[0]
    cuts = [g + 1 for g in gaps if st <= g < en - 1]
    bounds = [st] + cuts + [en]
    pieces = [(a, b) for a, b in zip(bounds[:-1], bounds[1:])]
    pieces = [p for p in pieces if ts[p[1] - 1] - ts[p[0]] >= min_len_s]
    if not pieces:
        return st, en, len(cuts)
    a, b = max(pieces, key=lambda p: ts[p[1] - 1] - ts[p[0]])
    return a, b, len(cuts)


def score_phase(score, f, fs, bw=0.35):
    N = len(score)
    fr = np.fft.fftfreq(N, 1 / fs)
    Sf = np.fft.fft(score * np.hanning(N))
    sel = (fr >= abs(f) - bw) & (fr <= abs(f) + bw)
    Zb = np.zeros_like(Sf)
    Zb[sel] = Sf[sel]
    za = 2.0 * np.fft.ifft(Zb)
    return np.unwrap(np.angle(za)), np.abs(za)


def wfit_slope(phi, w, fs, ts=None):
    t = (np.arange(len(phi)) / fs) if ts is None else np.asarray(ts, float)
    t = t - t[0]
    w = w / (w.sum() + 1e-12)
    tm = (w * t).sum()
    pm = (w * phi).sum()
    return float((w * (t - tm) * (phi - pm)).sum()
                 / ((w * (t - tm) ** 2).sum() + 1e-12))


def deflate_beat_ls(X, score, f0, fs, bw=0.35):
    phz, az = score_phase(score, f0, fs, bw)
    B = np.column_stack([az * np.cos(phz), az * np.sin(phz)])
    coef, *_ = np.linalg.lstsq(B, X, rcond=None)
    return X - B @ coef


def rmpoc_correction(ts, robot_speed, cos_human, cos_reference, flip=1.0):
    ts = np.asarray(ts, dtype=float)
    rate = robot_speed * (cos_human - cos_reference)
    if ts.ndim != 1 or not np.all(np.isfinite(ts)):
        raise ValueError("timestamps must be a finite 1-D array")
    if np.any(np.diff(ts) < 0):
        raise ValueError("timestamps must be nondecreasing")
    if not len(ts):
        return np.empty(0, dtype=float)
    return flip * rate * (ts - ts[0])


def cross_pair_distance_check(first, second, tolerance_m=0.6):
    d1, d2 = first["D"], second["D"]
    if first["f"] * second["f"] < 0:
        d2 = -d2
    return "OK" if abs(d1 - d2) < tolerance_m else "MISMATCH"


def run_file(path, motion_geometry, max_seg_s=None, nseg=1, preloaded=None,
             return_intermediates=False):
    if preloaded is None:
        tr = read_bf_file(path)
        H = tr.csi[:, 0]
        fs = float(tr.sample_rate_hz or 922.0)
        ts_full = np.asarray(getattr(tr, 'timestamps_s',
                                     np.arange(H.shape[0]) / fs), dtype=float)
    else:
        H, ts_full = preloaded
        H = np.asarray(H)
        ts_full = np.asarray(ts_full, dtype=float)
        fs = float((len(ts_full) - 1) / max(ts_full[-1] - ts_full[0], 1e-9))
    Nfull = H.shape[0]
    name = path.split('/')[-1]
    ts_full = ts_full - ts_full[0]

    fs_ts = (Nfull - 1) / max(ts_full[-1], 1e-9)
    if abs(fs_ts - fs) / fs > 0.01:
        print('    fs header %.1f vs timestamps %.1f -> using timestamps'
              % (fs, fs_ts))
        fs = float(fs_ts)
    wins = scan_windows(H, fs)
    segs = select_segments(wins, Nfull, fs, k=1)
    if segs is None:
        print('%-24s fs=%4.0f %5.1fs | no qualifying segment'
              % (name, fs, Nfull / fs))
        return {'name': name, 'status': 'no-segment'}

    st0, en0, run = segs[0]
    min_sub = int(4.0 * fs)
    k = int(min(max(1, nseg), max(1, (en0 - st0) // min_sub)))
    if k > 1:
        bounds = np.linspace(st0, en0, k + 1).astype(int)
        seg_list = [(int(bounds[i]), int(bounds[i + 1]), run)
                    for i in range(k)]
    else:
        seg_list = [(st0, en0, run)]
    results = []
    for si, (st, en, run) in enumerate(seg_list):
        if si:
            print('    ---- sub-segment %d/%d ----' % (si + 1, len(seg_list)))
        r = _process_one(H, fs, ts_full, name, st, en, motion_geometry,
                         max_seg_s, return_intermediates=return_intermediates)
        if r:
            results.append(r)
    ok = [r for r in results if 'tr' in r]
    if not ok:
        for r in results:
            return r
        return {'name': name, 'status': 'no-candidate'}
    n_ap = sum(1 for r in ok if r['Ig'] > 0.10)
    if 2 * n_ap != len(ok):
        voted = 2 * n_ap > len(ok)
    else:
        voted = max(ok, key=lambda r: r['lock1'])['Ig'] > 0.10
    rep = dict(max((r for r in ok if (r['Ig'] > 0.10) == voted),
                   key=lambda r: r['lock1'], default=ok[0]))
    rep['approach_voted'] = bool(voted)
    n_ap_s = sum(1 for r in ok if r.get('Ig_signed', 0.0) > 0.10)
    rep['approach_voted_s'] = bool(n_ap_s * 2 > len(ok)) \
        if 2 * n_ap_s != len(ok) else voted
    rep['nseg_ok'] = len(ok)
    rep['seg_Igs'] = [round(float(r['Ig']), 3) for r in ok]
    if len(ok) > 1:
        print('    VOTE: %d/%d segments approach -> %s (rep Ig %+.3f, '
              'all %s)' % (n_ap, len(ok), voted, rep['Ig'], rep['seg_Igs']))
    return rep


def _process_one(H, fs, ts_full, name, st, en, motion_geometry, max_seg_s,
                 return_intermediates=False):
    """Track beat candidates and construct a motion-compensated path metric."""
    Nfull = H.shape[0]
    st, en, ngaps = split_at_gaps(st, en, ts_full)
    if max_seg_s is not None and (en - st) / fs > max_seg_s:
        en = st + int(max_seg_s * fs)
    Hs = H[st:en]
    ts = ts_full[st:en] - ts_full[st]
    N = Hs.shape[0]
    print('%-24s fs=%4.0f %5.1fs | seg %5.1f-%5.1fs (%4.1fs, %d gaps split)'
          % (name, fs, Nfull / fs, ts_full[st], ts_full[en - 1],
             ts[-1], ngaps))

    keys = beat_census(Hs, fs)
    # Each candidate carries phase Phi (rad), distance anchor D and its IQR (m),
    # and signed path drift over the segment, trend (m).
    tracked = []
    for kd in keys[:6]:
        Phi, amp, D, D_iqr, aux = track_key(kd, fs, ts=ts)

        trend = aux['trend_ts']
        if np.isnan(trend):
            trend = float((Phi[-1] - Phi[0]) * C / (2 * np.pi * FC))
        tracked.append(dict(f=kd['f'], pair=kd['pair'], prom=kd['prom'],
                            Phi=Phi, amp=amp, D=D, D_iqr=D_iqr, trend=trend))

    for i in range(len(tracked)):
        for j in range(i + 1, len(tracked)):
            if tracked[i]['pair'] != tracked[j]['pair'] and \
                    abs(abs(tracked[i]['f']) - abs(tracked[j]['f'])) < 0.3:
                agree = cross_pair_distance_check(
                    tracked[i], tracked[j])
                tracked[i].setdefault('xcheck', []).append(agree)
                tracked[j].setdefault('xcheck', []).append(agree)

    for i in range(len(tracked)):
        for j in range(i + 1, len(tracked)):
            if abs(tracked[i]['f'] + tracked[j]['f']) < 0.3:
                cagree = 'OK' if abs(abs(tracked[i]['D'])
                                     - abs(tracked[j]['D'])) < 0.6 \
                    else 'MISMATCH'
                tracked[i].setdefault('conj', []).append(cagree)
                tracked[j].setdefault('conj', []).append(cagree)
    for kt in tracked:
        xc = '/'.join(kt.get('xcheck', [])) or '-'
        print('    key %+.2f Hz pair%d%d prom %6.0fx  signed D ~ %+5.2f m '
              '(IQR %.1f)  trend %+.2f m/seg  xpair %s'
              % (kt['f'], *kt['pair'], kt['prom'], kt['D'], kt['D_iqr'],
                 kt['trend'], xc))

    H0 = Hs[:, 0, :]
    Ha = 2.0 * np.max(np.abs(H0)) * np.exp(1j * np.pi / 4)
    P2 = np.abs(H0 + Ha) ** 2
    P2p = bandlimit_t(ma2d(P2, fs, 0.2), fs, BAND_LO, BAND_HI).real

    z1, v1, sv = stage_pca(P2p)
    b1 = pca_beat_match(z1, tracked, fs, N, ts=ts)
    if b1[0] is None:
        print('    stage-1 beat not key-confirmed; skip')
        return {'name': name, 'status': 'stage1-unconfirmed'}
    print('    stage-1 PCA: sv=%s; z1 <-> key %+.2f Hz, phase-lock=%.2f'
          % (np.round(sv[:3] / sv[0], 2), tracked[b1[0]]['f'], b1[1]))
    if b1[1] < 0.5:
        print('    stage-1 beat not key-confirmed; skip')
        return {'name': name, 'status': 'stage1-unconfirmed'}

    P2r = deflate_beat_ls(P2p, z1, tracked[b1[0]]['f'], fs)
    z2, v2, sv2 = stage_pca(P2r)
    b2 = pca_beat_match(z2, tracked, fs, N, ts=ts)
    if b2[0] is None:
        print('    stage-2 beat not key-confirmed; using stage-1 candidates')
    else:
        print('    stage-2 PCA (LS): sv=%s; z2 <-> key %+.2f Hz, phase-lock=%.2f'
              % (np.round(sv2[:3] / sv2[0], 2), tracked[b2[0]]['f'], b2[1]))

    pca_idx = {b1[0]}
    if b2[0] is not None and b2[1] >= 0.5:
        pca_idx.add(b2[0])
    closure = None
    if b2[0] is not None and b2[1] >= 0.5 and b2[0] != b1[0]:
        f1, f2 = tracked[b1[0]]['f'], tracked[b2[0]]['f']
        cands3 = [abs(f1 + f2), abs(f1 - f2)]
        closure = any(
            any(abs(abs(kt['f']) - fc) <= 0.35 for kt in tracked)
            for fc in cands3)
        print('    closure: f1=%+.2f f2=%+.2f -> third beat at %.2f/%.2f Hz '
              '%s' % (f1, f2, cands3[0], cands3[1],
                      'FOUND' if closure else 'not found'))

    def admissible_anchor(candidate):
        if 0.4 <= abs(candidate['D']) <= 4.0:
            return True

        # A stable anchor permits the wider distance interval.
        return (not np.isnan(candidate['D_iqr']) and candidate['D_iqr'] <= 1.0
                and 0.2 <= abs(candidate['D']) <= 6.0)

    gate = 'trend-only'
    pool = [kt for kt in tracked if not np.isnan(kt['D'])]
    idx_of = {id(kt): i for i, kt in enumerate(tracked)}
    # Prefer candidates supported by a PCA component; retain a phase-only
    # fallback when none of those candidates has an admissible anchor.
    cands = [kt for kt in pool if idx_of[id(kt)] in pca_idx and admissible_anchor(kt)]
    confirmed = True
    if not cands:
        confirmed = False
        cands = [kt for kt in pool if admissible_anchor(kt)]
    if not cands:
        print('    no geometrically admissible beat; skip metric')
        return {'name': name, 'status': 'no-geom-candidate', 'gate': gate}
    def key_score(kt):
        checks = kt.get('xcheck', []) + kt.get('conj', [])
        bad = bool(checks) and all(c == 'MISMATCH' for c in checks)
        return abs(kt['trend']) * (0.5 if bad else 1.0)
    hu = max(cands, key=key_score)
    if not confirmed:
        print('    WARNING: chosen beat not PCA-confirmed (fallback)')

    hu_idx = next(i for i, kt in enumerate(tracked) if kt is hu)
    if hu_idx == b2[0] and b2[1] >= 0.5:
        src_score, src_lock, src_sgn, src_stage = z2, b2[1], b2[3], 2
    elif hu_idx == b1[0]:
        src_score, src_lock, src_sgn, src_stage = z1, b1[1], b1[3], 1
    else:
        src_score, src_lock, src_sgn, src_stage = None, 0.0, 1, 0
    Phi = hu['Phi']
    D0 = hu['D']

    # Conjugate beats share a positive-anchor convention for phase and motion
    # compensation, so flipping the anchor also flips the phase drift.
    flip = -1.0 if D0 < 0 else 1.0
    if D0 < 0:
        D0, Phi = -D0, -Phi
    trend_key = flip * hu['trend']
    if src_score is not None:
        phz, az = score_phase(src_score, hu['f'], fs)
        phz = flip * src_sgn * phz
        slope_z = wfit_slope(phz, az, fs, ts=ts)
        trend_pca = slope_z * ts[-1] * C / (2 * np.pi * FC)
        agree = abs(trend_pca - trend_key) < 0.30
        print('    PCA-carried beat: stage-%d lock=%.2f trend %+.2f m/seg '
              '(key %+.2f) %s'
              % (src_stage, src_lock, trend_pca, trend_key,
                 'agree' if agree else 'DISAGREE'))
        Phi_used = (phz - np.median(phz)) * C / (2 * np.pi * FC)
    else:
        trend_pca = float('nan')
        agree = False
        print('    PCA-carried beat: chosen key not stage-confirmed; '
              'distance falls back to key phase (flagged)')
        Phi_used = (Phi - np.median(Phi)) * C / (2 * np.pi * FC)

    # Subcarrier phase supplies the anchor; carrier phase supplies displacement.
    Dt = D0 + Phi_used

    th_r = motion_geometry['th_r']
    mu_r = np.cos(th_r)
    robot_speed = float(motion_geometry.get('robot_speed', V.V_M))
    corr = rmpoc_correction(
        ts, robot_speed, np.cos(motion_geometry['th_u']), mu_r, flip=flip)
    Dphys = Dt + corr
    tr = np.abs(Dphys)
    k = max(3, N // 5)
    Ig_end = float(tr[-1] - tr[0])
    Ig = float(np.median(tr[-k:]) - np.median(tr[:k]))

    Ig_signed = float(np.median(Dphys[-k:]) - np.median(Dphys[:k]))

    q = np.percentile(tr, [5, 50, 95])
    print('    human candidate: key %+.2f Hz pair%d%d, signed D0=%+.2f m, '
          'trend(key) %+.2f m/seg%s'
          % (hu['f'], *hu['pair'], D0, trend_key,
             '' if closure is None else (', closure=OK' if closure
                                         else ', closure=NO')))
    print('    tr(t): med %.2f [q5 %.2f q95 %.2f], min %.2f, Ig %+.3f '
          '(end %+.3f) m'
          % (q[1], q[0], q[2], tr.min(), Ig, Ig_end))
    result = dict(name=name, seg=(st, en), fs=fs, tr=tr, hu=hu,
                lock1=b1[1], lock2=b2[1], trend_pca=trend_pca,
                trend_key=trend_key, pca_agree=agree, src_stage=src_stage,
                gate=gate, Ig=Ig, Ig_end=Ig_end, Ig_signed=Ig_signed,
                Dphys=Dphys, metric_t=ts_full[st:en].copy(),
                metric_time_origin_s=float(ts_full[st]),
                sampling_max_gap_s=float(np.max(np.diff(ts))) if N > 1 else 0.0,
                sampling_gap_count=int(np.count_nonzero(np.diff(ts) > GAP_S)),
                selected_key_pair=tuple(hu['pair']),
                selected_key_frequency_hz=float(hu['f']),
                selected_distance_anchor_m=float(D0),
                selected_conjugate_flip=float(flip))
    if return_intermediates:
        result["intermediates"] = copy.deepcopy({
            "Hs": Hs, "ts": ts, "metric_t": ts_full[st:en],
            "P2p": P2p, "P2r": P2r, "Ha": Ha,
            "stage1": {"score": z1, "loading": v1, "singular_values": sv,
                       "match": b1},
            "stage2": {"score": z2, "loading": v2, "singular_values": sv2,
                       "match": b2},
            "tracked": tracked, "selected_key_index": hu_idx,
            "selected_key": hu, "robot_correction_m": corr,
            "carrier_displacement_m": Phi_used})
    return result

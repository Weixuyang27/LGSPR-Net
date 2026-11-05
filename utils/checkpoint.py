"""Deterministic LGSPR-Net checkpoint continuation utilities."""
import hashlib
import os
import random
import numpy as np
import torch


def add_arguments(p):
    p.add_argument('--paired_resume', default='')
    p.add_argument('--paired_snapshot', default='')
    p.add_argument('--paired_stop_at', type=int, default=0)
    p.add_argument('--paired_deterministic', action='store_true')


def setup(args):
    if not 0 <= args.paired_stop_at <= args.episode:
        raise ValueError('invalid paired_stop_at')
    if args.paired_stop_at and not args.paired_snapshot:
        raise ValueError('paired_stop_at requires paired_snapshot')
    if args.paired_snapshot and os.path.exists(args.paired_snapshot):
        raise FileExistsError(args.paired_snapshot)
    if args.paired_deterministic:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['cpu'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


def episode_seed(args, episode):
    if args.paired_deterministic:
        # Make episode RNG independent of evaluation calls.
        seed = (args.seed * 1000003 + episode * 9176) % (2**32)
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def digest(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def save(args, model, optimizer, episode, best_oa, best_episode, best_path):
    if not args.paired_snapshot:
        return
    state = rng_state()
    packet = dict(version=2, args=vars(args), model=model.state_dict(),
        optimizer=optimizer.state_dict(), episode=episode, best_oa=best_oa, best_episode=best_episode,
        best=torch.load(best_path, map_location='cpu', weights_only=False), rng=state,
        model_sha256=digest(model))
    os.makedirs(os.path.dirname(os.path.abspath(args.paired_snapshot)), exist_ok=True)
    # Do not overwrite a continuation snapshot.
    with open(args.paired_snapshot, 'xb') as f:
        torch.save(packet, f)
    print(f"[paired-save] episode={episode} model_sha256={packet['model_sha256']}", flush=True)


def resume(args, model, optimizer, best_path):
    if not args.paired_resume:
        return 0, float('-inf'), 0
    packet = torch.load(args.paired_resume, map_location='cpu', weights_only=False)
    allowed = {'paired_resume', 'paired_snapshot', 'paired_stop_at', 'exp_name', 'metrics_json',
               'save_dir', 'train_log_interval'}
    differences = [key for key, value in packet['args'].items()
                   if key not in allowed and vars(args).get(key) != value]
    if differences:
        raise ValueError(f'incompatible continuation arguments: {differences}')
    start = packet['episode']
    if start >= args.episode:
        raise ValueError('snapshot episode must precede the requested final episode')
    if os.path.exists(best_path):
        raise FileExistsError(best_path)
    model.load_state_dict(packet['model'])
    optimizer.load_state_dict(packet['optimizer'])
    if digest(model) != packet['model_sha256']:
        raise RuntimeError('snapshot parameter checksum mismatch')
    torch.save(packet['best'], best_path)
    restore_rng(packet['rng'])
    print(f"[paired-resume] episode={start} model_sha256={packet['model_sha256']} "
          'optimizer/RNG restored; shared-prefix best retained', flush=True)
    return start, packet['best_oa'], packet['best_episode']


def metadata(args):
    return dict(paired_resume=args.paired_resume, paired_deterministic=args.paired_deterministic,
                paired_episode_seed_rule='seed*1000003+episode*9176 mod 2**32')

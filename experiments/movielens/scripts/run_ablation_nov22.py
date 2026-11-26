"""Metadata-only / recency-only ablation of IKGR_dyn_nointent (TO split, MF backbone, 3 seeds).

The two specs flip one switch of IKGR_dyn_nointent each (MF + metadata KG + recency, no LLM intent):
  IKGR_meta_only_nov22     use_dynamic=False  (metadata KG only)
  IKGR_recency_only_nov22  use_meta_kg=False  (recency only)
Everything else (code, data, protocol, 40-epoch budget) is the one used for the existing TO results.

This is the runner that produced the 2026-11-24 runs (TODO appendix version), moved into the repository:
  * repository = parent of scripts/ (the appendix expected <runner dir>/<dataset>/)
  * the git-commit check is replaced by a check of the training code bytes (line endings normalized)
    against the code that produced the published runs; --skip-code-check turns it off
  * the metadata-KG readiness check works for both datasets (Goodreads has no `meta_rels`)
  * `run SPEC SEED` runs one job through the same child-process environment as `all`
The computation is unchanged: seed 2020 runs of this logic reproduced the published per-user results and
checkpoint tensors byte for byte on a different GPU (see the MovieLens repository, analysis/activity_TO/repro/).

usage (repository root, CUDA GPU):
  python scripts/run_ablation_nov22.py all              # 6 runs: 2 specs x seeds 2020-2022, one at a time
  python scripts/run_ablation_nov22.py run IKGR_recency_only_nov22 2020
Outputs: run_k30_TO/ablation_nov22/ (result JSON, per_user_TO/, recbole_slice/ checkpoints, logs/),
ablation_nov22.yaml and ablation_nov22_manifest.json in the repository root. Finished spec/seed pairs are
skipped by eval_slices (same run signature).
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

DATASET = 'movielens'
REPO = Path(__file__).resolve().parents[1]
WORKDIR = 'run_k30_TO/ablation_nov22'
SPEC_NAMES = ('IKGR_meta_only_nov22', 'IKGR_recency_only_nov22')
SEEDS = (2020, 2021, 2022)
SETTINGS = {
    'movielens': {
        'source': ('https://github.com/HK1DS/movielens_november.git', '7b08e8924398dafed60c3dfc6d142cb8670814a1', 'main'),
        'base_config': 'run_k30_TO/config.yaml',
        'n_users': 133684,
        'meta_kg_hash': 'eb64bb38a9656e81',
        # sha256 of the training code with CRLF -> LF, as used for the published runs
        'code': {
            'eval_slices.py': 'ce040ba0152bc21d0c21dd49ef2b608a0d07fb8066307b9e3eec6056e0b57319',
            'ikgr_core/model_ikgr.py': 'fe14700a19a4ec34640472a0c936f8928e67a3b45f55840393266bd6ef1337bd',
            'ikgr_core/corona_retriever.py': '3c5de7ed95c9cc957f0b7832bac8beb0d36dd60edcc455cb48a666e3e8ece2cc',
        },
    },
    'goodreads': {
        'source': ('https://github.com/HK1DS/goodreads_november.git', '14e8c4d4c9f77ef52dd77c524f0102161e520520', 'architect'),
        'base_config': 'configs/todo_5_8.yaml',
        'n_users': 62142,
        'meta_kg_hash': None,
        'code': {
            'eval_slices.py': '36102e2992edd134f51796fdb00063d8cc118bc08a73cf12b67b9f145df1ccd5',
            'ikgr_core/model_ikgr.py': '9377986a23917d0e95bb5724b4546eb1caf1d4f04ef612b81c7721bfb8794977',
            'ikgr_core/corona_retriever.py': '7e13603c36fdb02a274d79c20d85580e3ab0d9dcd37a9be52681c029ca64bdcd',
            'ikgr_core/intent_filter.py': 'ce9a83faa2681aa51366da559889781eb64b25e5da36b2cd208b9ed92620ec13',
            'ikgr_core/pause_resume.py': 'b6399c03f6496a4ab11c73ab6a45965fe5fb4843c76a5f061c9fe586bee3a55f',
            'scripts/run_todo_5_8.py': '72de703318d859a9559983f04a3a276c9fcf00c96058e5f18cb2152f911f78cd',
        },
    },
}
S = SETTINGS[DATASET]
JOBS = [(s, 2020) for s in SPEC_NAMES] + [(s, seed) for s in SPEC_NAMES for seed in SEEDS[1:]]


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    tmp.replace(path)


def normalized_sha256(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def code_mismatches(repo=REPO):
    """Training-code files whose bytes differ from the code of the published runs."""
    return {p: normalized_sha256(repo / p) for p, want in S['code'].items() if normalized_sha256(repo / p) != want}


def derive_specs(available):
    """Add the two ablation specs to eval_slices.build_specs() output (each flips one switch)."""
    model, extra = available['IKGR_dyn_nointent']
    assert extra['use_kg'] is False and extra['use_meta_kg'] and extra['use_dynamic']
    available[SPEC_NAMES[0]] = (model, dict(extra, use_dynamic=False))
    available[SPEC_NAMES[1]] = (model, dict(extra, use_meta_kg=False))
    return available


def configure():
    import yaml
    original = REPO / S['base_config']
    config = yaml.safe_load(original.read_text(encoding='utf-8'))
    config['paths']['workdir'] = WORKDIR + '/'
    config['paths']['kg_pack'] = 'run_k30_TO/kg_pack.pt'
    config['paths']['meta_kg_pack'] = 'run_k30_TO/meta_kg_pack.pt'
    config['recbole']['worker'] = 0
    config['recbole']['gpu_id'] = '0'
    config['recbole']['require_cuda'] = True
    path = REPO / 'ablation_nov22.yaml'
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    atomic(REPO / 'ablation_nov22_manifest.json', {'source_sha': S['source'][1],
        'base_config': S['base_config'], 'config_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'jobs': [[DATASET, spec, seed] for spec, seed in JOBS], 'llm_calls': False, 'split': 'TO'})


def load_evaluation():
    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    import eval_slices as ev
    original = ev.build_specs
    ev.build_specs = lambda paths: derive_specs(original(paths))
    original_trainer = ev.get_trainer

    def named_trainer(*args, **kwargs):
        parent = original_trainer(*args, **kwargs)

        class NamedCheckpointTrainer(parent):
            def __init__(self, config, model):
                assert str(config['device']).startswith('cuda')
                assert not model.use_kg and model.lgcn_layers == 0
                expected = (True, False) if os.environ['IKGR_SPECS'] == SPEC_NAMES[0] else (False, True)
                assert (model.use_meta_kg, model.use_dynamic) == expected
                if model.use_meta_kg:   # MovieLens: meta_rels; Goodreads: author/publisher/shelf tables
                    assert model._meta_ready and (getattr(model, 'meta_rels', None)
                                                  or getattr(model, '_has_author', False)
                                                  or getattr(model, '_has_shelf', False))
                if model.use_dynamic:
                    assert model._recency_ready
                super().__init__(config, model)
                spec = os.environ.get('IKGR_SPECS', '')
                seed = os.environ.get('IKGR_SEEDS', '')
                if spec in SPEC_NAMES and seed.isdigit():
                    self.saved_model_file = str(Path(config['checkpoint_dir']) / f'{spec}__s{seed}_best.pth')

            def fit(self, *args, **kwargs):
                print('TRAINING_STARTED', os.environ['IKGR_SPECS'], os.environ['IKGR_SEEDS'], 'PID', os.getpid(), flush=True)
                return super().fit(*args, **kwargs)

        return NamedCheckpointTrainer

    ev.get_trainer = named_trainer
    return ev


def environment(spec, seed):
    env = {k: v for k, v in os.environ.items() if not k.startswith('IKGR_')}
    env.update(IKGR_CONFIG=str(REPO / 'ablation_nov22.yaml'), IKGR_SPLIT='TO',
        IKGR_DUMP_VALID='1', IKGR_REQUIRE_PROVENANCE='1', IKGR_SEEDS=str(seed),
        IKGR_SPECS=spec, PYTHONHASHSEED=str(seed),
        CUBLAS_WORKSPACE_CONFIG=':4096:8', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        WANDB_MODE='disabled', PYTHONIOENCODING='utf-8', PYTHONUTF8='1', OMP_NUM_THREADS='8')
    return env


def validate_artifacts(spec, seed):
    import gzip
    import csv
    wd = REPO / WORKDIR
    report = json.loads((wd / 'slice_eval_TO_result.json').read_text())[spec]['seeds'][str(seed)]
    assert report['fit']['best_epoch'] is not None
    expected = S['n_users']
    identities = []
    for split in ('valid', 'test'):
        path = wd / f'per_user_TO/{spec}__s{seed}__{split}.csv.gz'
        with gzip.open(path, 'rt') as file:
            rows = list(csv.DictReader(file))
        assert len(rows) == expected and 'user_id' in rows[0]
        users = [r['user_id'] for r in rows]
        assert len(set(users)) == expected
        identities.append(set(users))
    assert identities[0] == identities[1]
    assert (wd / 'recbole_slice' / f'{spec}__s{seed}_best.pth').exists()
    return report


def train(spec, seed, skip_code_check=False):
    import torch
    assert torch.cuda.is_available(), 'CUDA GPU required'
    bad = code_mismatches()
    if bad and not skip_code_check:
        raise SystemExit(f'training code differs from the published runs: {sorted(bad)} (--skip-code-check to run anyway)')
    configure()
    clean = environment(spec, seed)
    os.environ.clear()
    os.environ.update(clean)
    ev = load_evaluation()
    if S['meta_kg_hash']:
        assert ev._file_sha256('run_k30_TO/meta_kg_pack.pt') == S['meta_kg_hash']
    atomic(REPO / f'{WORKDIR}/launch_{spec}_s{seed}.json', {
        'source_sha': S['source'][1], 'code_matches_published': not bad, 'code_mismatches': bad,
        'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
        'cuda': torch.version.cuda, 'seed': seed, 'spec': spec, 'time': time.time(), 'pid': os.getpid()})
    sys.argv = [sys.argv[0]]
    ev.main()
    report = validate_artifacts(spec, seed)
    atomic(REPO / f'{WORKDIR}/complete_{spec}_s{seed}.json', {
        'time': time.time(), 'best_epoch': report['fit']['best_epoch'], 'train_sec': report['train_sec']})
    print('RUN_VERIFIED', DATASET, spec, seed, flush=True)


def queue(jobs, skip_code_check=False):
    """Each job runs in a child process so PYTHONHASHSEED and the other variables apply from interpreter start."""
    logs = REPO / WORKDIR / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    completed = []
    for spec, seed in jobs:
        log_path = logs / f'{DATASET}_{spec}_s{seed}.log'
        print('STARTING_JOB', DATASET, spec, seed, str(log_path), flush=True)
        cmd = [sys.executable, '-u', str(Path(__file__).resolve()), '_child', spec, str(seed)]
        if skip_code_check:
            cmd.append('--skip-code-check')
        with log_path.open('a', encoding='utf8') as log:
            proc = subprocess.Popen(cmd, cwd=REPO, env=environment(spec, seed), stdout=log, stderr=subprocess.STDOUT)
            atomic(logs / 'queue_status.json', {'phase': 'running', 'job': [DATASET, spec, seed],
                'child_pid': proc.pid, 'completed': completed, 'time': time.time()})
            code = proc.wait()
        if code:
            atomic(logs / 'queue_status.json', {'phase': 'failed', 'job': [DATASET, spec, seed],
                'exit_code': code, 'completed': completed, 'log': str(log_path)})
            raise SystemExit(code)
        completed.append([DATASET, spec, seed])
    atomic(logs / 'queue_status.json', {'phase': 'completed', 'completed': completed})
    print('ALL_RUNS_VERIFIED', len(completed), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('action', choices=['all', 'run', '_child'])
    parser.add_argument('spec', nargs='?', choices=SPEC_NAMES)
    parser.add_argument('seed', nargs='?', type=int, choices=SEEDS)
    parser.add_argument('--skip-code-check', action='store_true')
    args = parser.parse_args()
    if args.action == 'all':
        if args.spec is not None or args.seed is not None:
            parser.error('all takes no spec or seed')
        queue(JOBS, args.skip_code_check)
    else:
        if args.spec is None or args.seed is None:
            parser.error(f'{args.action} requires spec and seed')
        if args.action == 'run':
            queue([(args.spec, args.seed)], args.skip_code_check)
        else:
            train(args.spec, args.seed, args.skip_code_check)

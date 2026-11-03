"""Portable command-line interface. Run from any working directory."""
from __future__ import annotations
import argparse, csv, hashlib, importlib.metadata, json, os, platform, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def dump(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str)+'\n')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def environment():
    versions={}
    for name in ['torch','rdkit','numpy','pandas','jax','jaxlib','flax','optax','scipy']:
        try: versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: pass
    return {'python':platform.python_version(),'platform':platform.platform(),'packages':versions}

def child_env():
    env=dict(os.environ)
    env['PYTHONPATH']=str(ROOT)+os.pathsep+env.get('PYTHONPATH','')
    env.setdefault('OMP_NUM_THREADS','1'); env.setdefault('MKL_NUM_THREADS','1')
    env.setdefault('PYTHONHASHSEED','2026')
    env.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
    return env

def verify(args):
    root=args.data_root.resolve()
    manifest=json.loads((root/'manifest.json').read_text())
    failures=[]
    for item in manifest['files']:
        path=(root/item['path']).resolve()
        if not path.is_relative_to(root) or not path.is_file(): failures.append(item['path']+': missing or unsafe')
        elif path.stat().st_size!=item['bytes'] or sha(path)!=item['sha256']: failures.append(item['path']+': checksum mismatch')
    if failures: raise ValueError('\n'.join(failures))
    print(f"Verified {len(manifest['files'])} files; dataset version {manifest['version']}")

def train(args):
    key=f'{args.task}/{args.model}'
    experiments=json.loads((ROOT/'configs/experiments.json').read_text())
    if key not in experiments: raise ValueError(f'Unknown model {key}; available: '+', '.join(experiments))
    spec=experiments[key]
    config=json.loads((ROOT/'configs/qsar'/args.task/(args.model+'.json')).read_text())
    source=(args.input or args.data_root/spec['data']).resolve()
    if not source.is_file(): raise FileNotFoundError(source)
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    config.update(data_path=str(source),output_path=str(output/'model.pt'),device=args.device,seed=args.seed,num_workers=0)
    config['output_dir']=str(output)
    if args.epochs is not None: config['epochs']=args.epochs
    if args.smoke:
        import pandas as pd
        frame=pd.read_csv(source)
        target=config.get('target_column') or config['target_columns'][0]
        frame=frame.dropna(subset=[config['smiles_column'],target])
        if args.task=='dynamic_viscosity':
            ids=sorted(frame.compound_id.unique())[:8]
            frame=frame[frame.compound_id.isin(ids)].groupby('compound_id',sort=True).head(3)
        else: frame=frame.head(32)
        source_smoke=output/'smoke_input.csv'; frame.to_csv(source_smoke,index=False)
        config.update(data_path=str(source_smoke),epochs=1,hidden_dim=16,num_layers=1,readout_dim=16,batch_size=8)
        if args.model=='schnet': config.update(optimize_steps=0,no_add_hydrogens=True,num_gaussians=10)
        if args.task=='dynamic_viscosity': config['condition_dim']=8
        if args.model.startswith('conditional'): config['molecule_dim']=16
    resolved=output/'config.json'; dump(resolved,config)
    command=[sys.executable,'-m',spec['module'],'--config',str(resolved)]
    provenance={'experiment':key,'source_sha256':sha(source),'effective_input_sha256':sha(config['data_path']),
                'smoke_test':args.smoke,'environment':environment(),'command':command,
                'seed':args.seed,'split':'compound-grouped' if args.task=='dynamic_viscosity' else 'seeded row split after valid-label filtering',
                'status':'running'}
    dump(output/'run_manifest.json',provenance)
    with (output/'train.log').open('w') as log:
        result=subprocess.run(command,cwd=ROOT,env=child_env(),stdout=log,stderr=subprocess.STDOUT)
    provenance['status']='passed' if result.returncode==0 else 'failed'
    dump(output/'run_manifest.json',provenance)
    if result.returncode:
        print((output/'train.log').read_text()[-6000:],file=sys.stderr)
        raise RuntimeError(f'Training failed; see {output / "train.log"}')
    print(f'Trained {key}: {output / "model.pt"}')

def predict(args):
    import numpy as np
    from cases.ester_oil_multi_prop.qsar_predictors import QsarObjective
    if args.input:
        with args.input.open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
        smiles=[r[args.smiles_column] for r in rows]
    else: smiles=args.smiles; rows=[{'smiles':s} for s in smiles]
    if not rows: raise ValueError('Input contains no molecules')
    spec={'name':args.task,'checkpoint':str(args.checkpoint.resolve()),'kind':'viscosity' if args.task=='dynamic_viscosity' else 'scalar',
          'target_key': 'flash_point_K' if args.task=='flash_point' else 'property_value_si',
          'output_transform':'exp' if args.task=='dynamic_viscosity' else 'identity',
          'temperature_K':args.temperature,'pressure_kPa':args.pressure}
    objective=QsarObjective(spec,device=args.device)
    values=objective.predict(smiles,batch_size=args.batch_size)
    name={'flash_point':'flash_point_K','dynamic_viscosity':'dynamic_viscosity_Pa_s','dielectric_constant':'dielectric_constant'}[args.task]
    for row,value in zip(rows,values):
        row[name]=float(value) if np.isfinite(value) else ''
        row['prediction_status']='ok' if np.isfinite(value) else 'invalid_structure_or_prediction'
        if args.task=='dynamic_viscosity': row.update(temperature_K=args.temperature,pressure_kPa=args.pressure)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(f'Wrote {len(rows)} predictions: {args.output}')

def design(args):
    config=json.loads(args.config.read_text())
    config_base=args.config.resolve().parent
    for key in ['dit_config_path','dit_params_path','vae_config_path','vae_params_path','alphabet_path']:
        p=Path(config[key]); p=p if p.is_absolute() else config_base/p
        if not p.is_file(): raise FileNotFoundError(f'{key}: {p}; first run the generation training recipe')
        config[key]=str(p.resolve())
    for obj in config['qsar_objectives']:
        p=Path(obj['checkpoint']); p=p if p.is_absolute() else config_base/p
        if not p.is_file(): raise FileNotFoundError(p)
        obj['checkpoint']=str(p.resolve())
    args.output.mkdir(parents=True,exist_ok=False)
    config['qsar_device']=args.device
    config['selection_algorithm']='nsga2' # fixed across environments
    p=args.output.resolve()/'config.json'; dump(p,config)
    cmd=[sys.executable,'-m','cases.ester_oil_multi_prop.diff_evo','--config_path',str(p),'--save_path',str(args.output.resolve())]
    subprocess.run(cmd,cwd=ROOT,env=child_env(),check=True)

def main():
    parser=argparse.ArgumentParser(description='Ester-oil dataset reproduction tools')
    commands=parser.add_subparsers(dest='command',required=True)
    v=commands.add_parser('verify-data'); v.add_argument('--data-root',type=Path,required=True); v.set_defaults(func=verify)
    t=commands.add_parser('train'); t.add_argument('--task',required=True,choices=['flash_point','dynamic_viscosity','dielectric_constant'])
    t.add_argument('--model',required=True); t.add_argument('--data-root',type=Path,default=Path('data'))
    t.add_argument('--input',type=Path,help='Optional replacement CSV with the documented schema')
    t.add_argument('--output',type=Path,required=True); t.add_argument('--device',default='cpu'); t.add_argument('--epochs',type=int)
    t.add_argument('--seed',type=int,default=2026); t.add_argument('--smoke',action='store_true'); t.set_defaults(func=train)
    p=commands.add_parser('predict'); p.add_argument('--task',required=True,choices=['flash_point','dynamic_viscosity','dielectric_constant'])
    p.add_argument('--checkpoint',type=Path,required=True); group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--input',type=Path); group.add_argument('--smiles',nargs='+'); p.add_argument('--smiles-column',default='canonical_smiles')
    p.add_argument('--output',type=Path,required=True); p.add_argument('--device',default='cpu'); p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--temperature',type=float,default=298.15); p.add_argument('--pressure',type=float,default=101.325); p.set_defaults(func=predict)
    d=commands.add_parser('design'); d.add_argument('--config',type=Path,required=True); d.add_argument('--output',type=Path,required=True)
    d.add_argument('--device',default='cpu'); d.set_defaults(func=design)
    args=parser.parse_args(); args.func(args)

if __name__=='__main__': main()

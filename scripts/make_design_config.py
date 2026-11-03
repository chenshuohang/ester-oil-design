"""Bind a design configuration to trained AE/DiT and the baseline model package."""
import argparse,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--generation-run',type=Path,required=True)
p.add_argument('--data-root',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--device',default='cpu')
args=p.parse_args()
c=json.loads((ROOT/'configs/design.json').read_text())
for key,path in [('vae_config_path','ae/config.pkl'),('vae_params_path','ae/params_final.pkl'),('dit_config_path','dit/config.pkl'),('dit_params_path','dit/params_final.pkl'),('alphabet_path','alphabet.pkl')]:
    target=(args.generation_run/path).resolve()
    if not target.is_file():raise FileNotFoundError(target)
    c[key]=str(target)
for objective,task,model in zip(c['qsar_objectives'],['flash_point','dynamic_viscosity','dielectric_constant'],['schnet','conditional_communicative_mpnn','schnet']):
    target=(args.data_root/'pretrained/qsar'/task/(model+'.pt')).resolve()
    if not target.is_file():raise FileNotFoundError(target)
    objective['checkpoint']=str(target)
c['qsar_device']=args.device
args.output.parent.mkdir(parents=True,exist_ok=True)
with args.output.open('x') as f:json.dump(c,f,indent=2)
print(args.output)

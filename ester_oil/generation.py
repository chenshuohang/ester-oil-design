"""Rebuild the AE -> latent -> conditional DiT chain from explicit allowed CSVs."""
from __future__ import annotations
import argparse, csv, json, os, pickle, random, re, subprocess, sys
from pathlib import Path
from .__main__ import ROOT, child_env, dump, environment, sha

def pkl(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('wb') as f: pickle.dump(value,f,protocol=4)

def config(name):
    import numpy as np
    c=json.loads((ROOT/'configs/generation'/f'{name}.json').read_text())
    if name=='ae': c['data_config']['dtype']=np.int16
    return c

def tiny(ae,dit):
    ae['global_config']['bf16_flag']=False
    n=ae['net_config']; n['projector']['output_sizes']=[16,8]
    m=n['encoder']['molct']; m['n_layers']=1
    m['feat_transformer'].update(atom_act_dim=32,pair_act_dim=16)
    g=m['graph_transformer']; g['node_update']['attention_embedding'].update(dim_feature=32,n_head=2)
    g['node_update']['post_attention']['out_dim']=32
    g['edge_update']['outer_product'].update(outerproduct_dim=8,output_dim=16)
    d=n['decoder']; d['dim_feature']=32; d['num_groups']=1
    t=d['transformer']; t['n_layers']=1; t['adaLN']['hidden_size']=32
    t['attention']['attention_embedding'].update(dim_feature=32,n_head=2)
    t['attention']['post_attention']['out_dim']=32
    ae['train_config']['learning_rate'].update(warmup_steps=1,decay_steps=20)
    dit['global_config']['bf16_flag']=False
    n=dit['net_config']; n.update(hidden_size=32,n_iterations=1)
    n['dit_output']['hidden_size']=32
    n['label_embedding'].update(hidden_size=32,num_basis=16,emb_sigma=1/16)
    n['time_embedding'].update(hidden_size=32,frequency_embedding_size=32)
    b=n['dit_block']; b['adaLN']['hidden_size']=32
    b['attention']['attention_embedding'].update(dim_feature=32,n_head=2)
    b['attention']['post_attention']['out_dim']=32
    dit['train_config']['learning_rate'].update(warmup_steps=1,decay_steps=20)

TOKEN=re.compile(r'(\%\([0-9]{3}\)|\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\||\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])')

def prepare(args):
    import numpy as np
    from rdkit import Chem
    from molsculptor.train.inference import smi2graph_features
    work=args.work.resolve(); work.mkdir(parents=True,exist_ok=True)
    prepared=work/'prepared'; prepared.mkdir(exist_ok=False)
    alphabet=json.loads((ROOT/'configs/generation/alphabet.json').read_text())
    pkl(work/'alphabet.pkl',alphabet)
    ae,dit=config('ae'),config('dit')
    if args.smoke: tiny(ae,dit)
    pkl(work/'ae_config.pkl',ae); pkl(work/'dit_config.pkl',dit)
    sources=args.input or [args.data_root/'generation/pubchem_filtered_smiles.csv',args.data_root/'generation/brics500_filtered_smiles.csv']
    accepted={}; rejected=[]; examined=0
    for source in sources:
        with source.open(encoding='utf-8-sig',newline='') as f:
            reader=csv.DictReader(f)
            col=next((x for x in ['canonical_smiles','smiles','SMILES'] if x in reader.fieldnames),None)
            if not col: raise ValueError(f'No SMILES column in {source}')
            for index,row in enumerate(reader,1):
                examined+=1; s=row[col]
                try:
                    mol=Chem.MolFromSmiles(s)
                    if mol is None: raise ValueError('invalid SMILES')
                    if len(Chem.GetMolFrags(mol))!=1: raise ValueError('multiple fragments')
                    if any(a.GetAtomicNum() not in (1,6,8) or a.GetFormalCharge()!=0 for a in mol.GetAtoms()):
                        raise ValueError('generation recipe requires neutral C/H/O molecules')
                    if not 3<mol.GetNumAtoms()<=128: raise ValueError('atom count outside 4..128')
                    s=Chem.MolToSmiles(mol,isomericSmiles=True)
                    tokens=TOKEN.findall(s)
                    if ''.join(tokens)!=s or len(tokens)+1+16>256: raise ValueError('tokenization or length')
                    ids=[alphabet['symbol_to_idx'][x] for x in tokens]
                    if s not in accepted: accepted[s]=(ids,source.name,index)
                except (ValueError,KeyError) as exc: rejected.append({'source':source.name,'row':index,'reason':str(exc)})
                if args.limit and len(accepted)>=args.limit: break
        if args.limit and len(accepted)>=args.limit: break
    if len(accepted)<4: raise ValueError('At least four valid distinct molecules are required')
    smiles=sorted(accepted); random.Random(args.seed).shuffle(smiles)
    n_test=max(1,round(len(smiles)*0.1)); rows=[]; train=[]; test=[]
    for i,s in enumerate(smiles):
        ids,source,row=accepted[s]
        a=np.asarray([0]+ids,np.int16); b=np.asarray(ids+[1],np.int16)
        length=len(a); pad=256-length
        features={'smiles':np.array(s,object),
                  'tokens':{'input_tokens':np.pad(a,(0,pad),constant_values=1)[None,:],
                            'label':np.pad(b,(0,pad),constant_values=1)[None,:],
                            'input_mask':np.ones((1,256),np.int16),'label_mask':np.ones((1,256),np.int16)},
                  'graph_features':smi2graph_features(s,n_padding_atom=Chem.MolFromSmiles(s).GetNumAtoms())}
        path=prepared/'graphs'/f'{i//1000:04d}'/f'{i:08d}.pkl'; pkl(path,features)
        split='test' if i<n_test else 'train'
        (test if split=='test' else train).append(str(path))
        rows.append({'id':i,'canonical_smiles':s,'split':split,'source':source,'source_row':row,'graph':str(path.relative_to(work))})
    for name,paths in [('train',train),('test',test)]: pkl(prepared/f'{name}_index.pkl',{'path':np.asarray(paths,object)})
    with (prepared/'structures.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    dump(prepared/'manifest.json',{'seed':args.seed,'examined':examined,'train':len(train),'test':len(test),'smoke':args.smoke,
          'input_files':[{'name':p.name,'sha256':sha(p)} for p in sources], 'rejected':rejected,'environment':environment(),
          'recipe':'canonical unique structures, 90/10 shuffled split; no historical merged indices or latent arrays'})
    print(f'Prepared {len(train)} train / {len(test)} test structures',flush=True)

def run_training(args,stage):
    work=args.work.resolve(); out=work/stage; out.mkdir(exist_ok=False)
    common=['--num_processes','1','--device_batch_size',str(args.batch_size),'--pre_load_steps','1',
            '--random_seed',str(args.seed),'--np_random_seed',str(args.seed),'--callback_steps','1' if args.smoke else '100',
            '--save_steps','1000','--logger_path',str(out/'train.log'),'--save_ckpt_path',str(out)]
    if stage=='ae':
        cmd=[sys.executable,str(ROOT/'main_train_ae.py'),'--config_path',str(work/'ae_config.pkl'),
             '--name_list_path',str(work/'prepared/train_index.pkl'),'--num_total_epochs',str(1 if args.smoke else args.epochs),*common]
    else:
        cmd=[sys.executable,str(ROOT/'main_train_dit.py'),'--config_path',str(work/'dit_config.pkl'),
             '--cond_name_list_path',str(work/'latents/index.pkl'),'--uncond_name_list_path',str(work/'latents/index.pkl'),
             '--num_total_steps',str(2 if args.smoke else args.steps),*common]
    with (out/'console.log').open('w') as f:
        p=subprocess.run(cmd,cwd=ROOT,env=child_env(),stdout=f,stderr=subprocess.STDOUT)
    if p.returncode:
        print((out/'console.log').read_text()[-7000:],file=sys.stderr)
        raise RuntimeError(f'{stage} training failed')
    print(f'{stage}: {out / "params_final.pkl"}',flush=True)

def encode(args):
    import numpy as np
    import jax
    import jax.numpy as jnp
    from ml_collections import ConfigDict
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    from molsculptor.train.inference import InferEncoder,smi2graph_features
    work=args.work.resolve(); output=work/'latents';output.mkdir(exist_ok=False)
    with (work/'ae/config.pkl').open('rb') as f: c=pickle.load(f)
    with (work/'ae/params_final.pkl').open('rb') as f: ck=pickle.load(f)
    c['global_config']['dropout_flag']=False
    net=InferEncoder(ConfigDict(c['net_config']),ConfigDict(c['global_config']))
    params={k:ck['params']['generator'][k] for k in ['Encoder_0','Dense_0']}
    apply=jax.jit(net.apply)
    with (work/'prepared/structures.csv').open() as f: rows=[r for r in csv.DictReader(f) if r['split']=='train']
    shard=[]; files=[]
    def save(items,number):
        d={k:np.asarray([x[k] for x in items]) for k in items[0]}
        path=output/f'shard_{number:05d}.pkl';pkl(path,d);files.append(str(path))
    for offset in range(0,len(rows),args.batch_size):
        batch=rows[offset:offset+args.batch_size]; smi=[r['canonical_smiles'] for r in batch]
        graphs=[smi2graph_features(s,n_padding_atom=128) for s in smi]
        graph=jax.tree_util.tree_map(lambda *a:jnp.asarray(np.stack(a)),*graphs)
        latent=np.asarray(apply({'params':params},**graph))
        for r,s,z in zip(batch,smi,latent):
            mol=Chem.MolFromSmiles(s); hmol=Chem.AddHs(mol)
            carbon=sum(a.GetAtomicNum()==6 for a in hmol.GetAtoms()); hydrogen=sum(a.GetAtomicNum()==1 for a in hmol.GetAtoms())
            shard.append({'data_id':r['id'],'smiles':s,'latent':z,
                'n_double_bonds':float(sum(b.GetBondType()==Chem.BondType.DOUBLE and b.GetBeginAtom().GetAtomicNum()==6 and b.GetEndAtom().GetAtomicNum()==6 for b in mol.GetBonds())),
                'n_unsaturation':float((2*carbon+2-hydrogen)//2),'mw':float(rdMolDescriptors.CalcExactMolWt(mol)),
                'n_rings':float(rdMolDescriptors.CalcNumRings(mol)),'logp':float(rdMolDescriptors.CalcCrippenDescriptors(mol)[0]),
                'n_rot_bonds':float(rdMolDescriptors.CalcNumRotatableBonds(mol))})
        if len(shard)>=args.shard_size:save(shard,len(files));shard=[]
    if shard: save(shard,len(files))
    pkl(output/'index.pkl',files)
    dump(output/'manifest.json',{'count':len(rows),'ae_sha256':sha(work/'ae/params_final.pkl'),
         'train_structures_sha256':sha(work/'prepared/structures.csv'),'recipe':'encode only training split; all encodable structures, no reconstruction-hit filtering'})
    print(f'Encoded {len(rows)} training structures into {len(files)} shards',flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','train-ae','encode','train-dit','all'])
    p.add_argument('--data-root',type=Path,default=Path('data')); p.add_argument('--input',type=Path,nargs='+')
    p.add_argument('--work',type=Path,required=True);p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--batch-size',type=int,default=2);p.add_argument('--limit',type=int)
    p.add_argument('--epochs',type=int,default=10);p.add_argument('--steps',type=int,default=2000000)
    p.add_argument('--shard-size',type=int,default=10000);p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    if args.smoke and not args.limit: args.limit=8
    stages=['prepare','train-ae','encode','train-dit'] if args.stage=='all' else [args.stage]
    for stage in stages:
        if stage=='prepare':prepare(args)
        elif stage=='encode':encode(args)
        else:run_training(args,'ae' if stage=='train-ae' else 'dit')

if __name__=='__main__':main()

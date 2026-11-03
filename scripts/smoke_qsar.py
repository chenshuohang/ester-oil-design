"""Run all 13 small train/reload/predict checks without changing source datasets."""
import argparse, json, os, subprocess, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(); out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    registry=json.loads((ROOT/'configs/experiments.json').read_text())
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONHASHSEED='2026')
    result=[]
    for key in registry:
        task,model=key.split('/'); run=out/task/model
        cmd=[sys.executable,'-m','ester_oil','train','--task',task,'--model',model,'--data-root',str(args.data_root.resolve()),'--output',str(run),'--smoke']
        a=subprocess.run(cmd,cwd=ROOT,env=env,capture_output=True,text=True)
        b=None
        if a.returncode==0:
            b=subprocess.run([sys.executable,'-m','ester_oil','predict','--task',task,'--checkpoint',str(run/'model.pt'),
                '--smiles','CCCCCCCC(=O)OCC','CCOC(=O)CCCCC','--output',str(run/'predictions.csv')],cwd=ROOT,env=env,capture_output=True,text=True)
        item={'experiment':key,'train_returncode':a.returncode,'predict_returncode':None if b is None else b.returncode,
              'split_recorded':(run/'split.json').is_file(),'error':a.stderr[-3000:]+('' if b is None else b.stderr[-3000:])}
        result.append(item); print(json.dumps(item),flush=True)
        (out/'report.json').write_text(json.dumps(result,indent=2))
    if any(x['train_returncode'] or x['predict_returncode'] or not x['split_recorded'] for x in result): sys.exit(1)

if __name__=='__main__':main()

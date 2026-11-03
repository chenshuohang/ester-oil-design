"""Record the actual post-featurization split used by every training entrypoint."""
from pathlib import Path
from .__main__ import dump, sha

def save_split(args, train, validation, dataset):
    output=getattr(args,'output_path',None)
    folder=Path(output).parent if output else Path(args.output_dir)
    train=list(map(int,train)); validation=list(map(int,validation))
    assert set(train).isdisjoint(validation)
    assert sorted(train+validation)==list(range(len(dataset)))
    record={'input_sha256':sha(args.data_path),'seed':args.seed,'val_size':args.val_size,
            'index_space':'valid featurized observations in original input order',
            'valid_count':len(dataset),'skipped_rows':dataset.skipped_rows,
            'train_indices':train,'validation_indices':validation}
    items=getattr(dataset,'items',[])
    if items and hasattr(items[0],'compound_id'):
        left=sorted({items[i].compound_id for i in train});right=sorted({items[i].compound_id for i in validation})
        assert set(left).isdisjoint(right),'Compound leakage across split'
        record.update(train_compounds=left,validation_compounds=right)
    dump(folder/'split.json',record)

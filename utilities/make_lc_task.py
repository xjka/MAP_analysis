import os
from pathlib import Path
import click

def group_by_prefix(fns: list[Path]) -> dict[str, list[Path]]:
    groups = {}
    for fn in fns:
        prefix = fn.name.split('.')[0]
        if prefix not in groups:
            groups[prefix] = []
        groups[prefix] += [fn]
    return groups


@click.command
@click.argument('lcdir')
#@click.option('-t', '--taskfile', default='lc_to_asdf.sh', help='name of output task-file', metavar='FN')
def main(lcdir):

    TASK = 'python -m abacus.post.lc '
    kinds = ['heal', 'pid', 'rv']
    lcdir = Path(lcdir)
    outroot = lcdir.parent.resolve(strict=True) / 'lightcone-post'
    with open(os.environ.get('HOME')+ '/abacus/prod/lc_to_asdf.sh', 'w') as tf:
        tf.write('#!/bin/bash\n\n')
        tf.write('mkdir -p '+str(outroot)+'\n\n')
        for stepdir in sorted(lcdir.glob('Step*')):
            tf.write('cd '+str(stepdir)+' \n')
            fns = stepdir.glob('LightCone*')
            fns = [f.relative_to(stepdir) for f in fns]
            fns = group_by_prefix(fns)
            for prefix in fns:
                kind = prefix.split('_')[1].split('.')[0]
                if kind not in kinds:
                    raise ValueError(f'Unknown file type {kind} in {prefix}')
                outfn = outroot / kind / f'{prefix}_{stepdir.name}.asdf'
        
                tf.write(TASK + ' '.join(['-o', str(outfn)] + [str(f) for f in fns[prefix]])+'\n')
            tf.write('\n')

if __name__=='__main__':
    main()

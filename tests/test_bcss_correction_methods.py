import copy

import numpy as np
import pytest

from experiments.bcss_wcsp.multiscale.correction_methods import (
    edit_counts, presence_problem, region_pass, region_masks, relax_state,
    appearance_features, region_appearance_unary, solve_with_regions)
from experiments.bcss_wcsp.multiscale.fullres_corrections import treatments
from experiments.bcss_wcsp.multiscale.convergence_solver import (
    MeasuredBlocks, MeasuredHierarchy, MeasuredICM)
from wcsp.config import ProblemConfig
from wcsp.model import build_problem
from wcsp.model.problem import GridProblem
from wcsp.solve.energy import energy


def hierarchy(mu=1):
    return MeasuredHierarchy(MeasuredBlocks(MeasuredICM(), block=5),
                             levels=2, size=2, mu=mu)


def test_treatment_grid_has_matched_controls_and_uncapped_solvers():
    base=dict(problem=ProblemConfig(lam=8).to_dict(),solver=dict(name='hierarchy',levels=3,
        size=4,mu=1,block_alpha=1,block_lam=8,block_types=[1,2],max_rounds=4,
        pixel=dict(name='blocks',block=100,max_rounds=4,inner=dict(name='icm',max_sweeps=50))))
    original=copy.deepcopy(base)
    rows=treatments(base)
    assert len(rows)==40 and len({r['name'] for r in rows})==40
    assert all(r['control'] in {t['name'] for t in rows} for r in rows)
    for r in rows:
        s=r['config']['solver']
        assert s['max_rounds'] is s['block_max_sweeps'] is None
        assert s['pixel']['max_rounds'] is s['pixel']['inner']['max_sweeps'] is None
    assert base==original


def test_zero_presence_strength_is_exact_identity_and_strong_prior_changes_set():
    probs=np.tile(np.array([.55,.4,.02,.02,.01])[:,None,None],(1,4,4))
    p=build_problem(probs,ProblemConfig(lam=1))
    prior=np.full(32,10.)
    prior[2]=0
    same,info=presence_problem(p,prior,0)
    assert same is p
    changed,info=presence_problem(p,prior,100)
    assert info['chosen']==2
    assert np.all(changed.unary.argmin(-1)==1)


def test_coordinated_region_move_escapes_a_single_pixel_local_minimum():
    # Label 1 has a unary advantage everywhere, but a single 0->1 change
    # incurs a larger neighbor cost. Flipping the whole region is beneficial.
    u=np.zeros((4,4,2));u[...,0]=1
    p=GridProblem(u,np.array([[0,1],[1,0]]),np.array([4.,4.]),np.array([[0.,1.],[1.,0.]]))
    solver=hierarchy(mu=0)
    solver.pixel=MeasuredICM()
    solver.block_alpha=0;solver.block_lam=0
    initial=np.zeros((4,4),int)
    result=solver.solve(p,initial)
    assert np.all(result.labels==0) and result.stats['audit']['improving_variables']==0
    labels,levels,stats=region_pass(solver,p,result.labels,result.levels,[np.ones((4,4),bool)])
    assert stats['accepted']==1 and np.all(labels==1)
    assert stats['moves'][0]['after']<stats['moves'][0]['before']
    assert energy(p,labels)==0


def test_region_proposals_and_warm_cleanup_never_raise_the_full_objective():
    rng=np.random.default_rng(49)
    probs=rng.dirichlet(np.ones(5),size=(16,16)).transpose(2,0,1)
    p=build_problem(probs,ProblemConfig(lam=3))
    solver=hierarchy()
    initial=p.unary.argmin(-1)
    local=solver.solve(p,initial)
    masks=[np.indices((16,16))[0]<8,np.ones((16,16),bool)]
    labels,levels,stats=region_pass(solver,p,local.labels,local.levels,masks)
    blocks=solver.block_problems(p)
    before=solver.total_energy(p,blocks,labels,levels)
    assert before<=local.energy+1e-9
    labels,levels,value,_,_,audit=relax_state(solver,p,labels,levels)
    assert value<=before+1e-9 and audit['improving_variables']==0
    assert value==pytest.approx(solver.total_energy(p,blocks,labels,levels))
    result=solve_with_regions(solver,p,initial)
    assert result.energy<=local.energy+1e-9 and result.stats['audit']['improving_variables']==0


def test_region_budget_uses_no_annotations_and_handles_segment_ids():
    labels=np.zeros((16,16),int)
    segments=np.where(np.indices((16,16))[0]<8,100,999)
    masks=region_masks(labels,segments,minimum=64,maximum=1)
    assert len(masks)==1 and masks[0].sum()==128
    assert len(region_masks(labels,minimum=64))==1


def test_edit_audit_separates_fixes_breaks_and_wrong_to_wrong():
    initial=np.array([[0,1],[2,3]])
    target=np.array([[1,1],[4,255]])
    labels=np.array([[1,2],[0,0]])
    counts=edit_counts(initial,labels,target)['all']
    assert counts==dict(changed=3,fixes=1,breaks=1,wrong_to_wrong=1)


def test_appearance_unary_is_finite_and_constant_within_each_image_region():
    rng=np.random.default_rng(4)
    image=rng.integers(50,230,(16,16,3),dtype=np.uint8)
    features=appearance_features(image)
    assert features.shape==(16,16,9) and np.isfinite(features).all()
    model=dict(means=np.zeros((5,9)),variances=np.ones((5,9)),temperature=2)
    model['means'][1]=.2
    segments=np.where(np.indices((16,16))[0]<8,1,2)
    u=region_appearance_unary(image,segments,model)
    assert u.shape==(16,16,5) and np.isfinite(u).all() and (u>=0).all()
    np.testing.assert_array_equal(u[0,0],u[7,15])
    np.testing.assert_array_equal(u[8,0],u[15,15])


@pytest.mark.parametrize('family', ['temperature','presence','region_moves','regional_evidence','image_boundary','appearance_control'])
def test_runner_saves_audited_outputs_for_each_method_family(tmp_path, family):
    import json
    from experiments.bcss_wcsp.multiscale.fullres_corrections import solve, digest
    source=tmp_path/'source'
    (source/'inputs').mkdir(parents=True)
    (source/'protocol.json').write_text('{}')
    rng=np.random.default_rng(48)
    probs=rng.dirichlet(np.ones(5),size=(16,16)).transpose(2,0,1)
    target=rng.integers(0,5,(16,16),dtype=np.uint8)
    sample=source/'inputs/0000.npz'
    np.savez(sample,probs=probs.astype(np.float16),target=target,id='tile')
    root=tmp_path/'output'
    (root/'assets').mkdir(parents=True)
    image=rng.integers(0,255,(16,16,3),dtype=np.uint8)
    asset=root/'assets/0000.npz'
    np.savez(asset,id='tile',image=image,segments=np.ones((16,16),np.int32),
             appearance_unary=-np.log(probs.transpose(1,2,0)))
    prior=np.full(32,5.);prior[31]=0
    (root/'presence_prior.json').write_text(json.dumps(dict(cost=prior.tolist())))
    config=dict(problem=ProblemConfig(lam=3,temperature=2,contrast_beta=1 if family=='image_boundary' else 0).to_dict(),
        solver=dict(name='hierarchy',levels=2,size=2,mu=1,block_alpha=1,block_lam=3,
        block_types=[1,2],max_rounds=None,block_max_sweeps=None,
        pixel=dict(name='blocks',block=5,max_rounds=None,inner=dict(name='icm',max_sweeps=None))))
    row=dict(name='test',family=family,mu=1,config=config,gamma=4000,regions='slic',appearance_weight=.3)
    record=dict(index=0,id='tile',roi='roi',sha256=digest(sample))
    protocol=dict(source_root=str(source),source_protocol_sha256=digest(source/'protocol.json'),
        files={},samples=[record],assets=[dict(index=0,id='tile',sha256=digest(asset))],treatments=[row])
    (root/'protocol.json').write_text(json.dumps(protocol))
    solve(root,0)
    out=json.loads((root/'runs/test/dev/samples/0000.json').read_text())
    assert out['converged'] and out['audit']['improving_variables']==0
    assert out['edits']['all']['changed']==out['edits']['all']['fixes']+out['edits']['all']['breaks']+out['edits']['all']['wrong_to_wrong']
    with np.load(root/'runs/test/dev/samples/0000.npz') as z:
        np.testing.assert_array_equal(z['argmax'],probs.astype(np.float16).argmax(0))
    solve(root,0)  # Completed audited outputs resume without re-solving.


def test_partial_population_report_withholds_accuracy_and_paired_comparisons(tmp_path):
    import json
    from experiments.bcss_wcsp.multiscale.fullres_corrections import report, digest
    source=tmp_path/'source'
    source.mkdir()
    (source/'protocol.json').write_text('{}')
    root=tmp_path/'output'
    directory=root/'runs/test/dev/samples'
    directory.mkdir(parents=True)
    (directory/'0000.json').write_text(json.dumps(dict(index=0,id='a')))
    (directory/'0000.npz').touch()
    protocol=dict(source_root=str(source),source_protocol_sha256=digest(source/'protocol.json'),
        files={},samples=[dict(index=0,id='a',roi='r'),dict(index=1,id='b',roi='r')],
        baseline=dict(mIoU=.5),treatments=[dict(name='test',family='temperature',control='test')])
    (root/'protocol.json').write_text(json.dumps(protocol))
    report(root)
    row=json.loads((root/'summary.json').read_text())['results'][0]
    assert row['status']=='Incomplete' and row['samples']==1 and row['expected']==2
    assert 'metrics' not in row and 'vs_control' not in row and 'vs_unet' not in row

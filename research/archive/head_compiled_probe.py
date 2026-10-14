"""Measure actual W8A16 screening cost on frozen captured target activations.

Engineering prototype using existing Marlin quantization, not a novel-method
claim. No model generations are changed. Refinement and fallback are timed.
"""
import os
os.environ['VLLM_WORKER_MULTIPROC_METHOD']='spawn'
os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN']='1'
import json,time,traceback,statistics
from pathlib import Path
from datetime import datetime,timezone
BASE=Path('/content/qwen38_research')
FOLDER=BASE/'head-prefix-compiled-v1'
DEADLINE=datetime.fromisoformat('2026-10-14T07:11:31.211985+00:00')

def run_probe(model):
    import torch
    from vllm import _custom_ops as ops
    from vllm.scalar_type import scalar_types
    from vllm.model_executor.layers.quantization.utils.marlin_utils_test import marlin_quantize
    from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_make_workspace_new
    calls=torch.load(BASE/'head-prefix-feasibility-v2/captured_head_calls.pt',weights_only=True)
    heads=[m for n,m in model.named_modules() if n.endswith('lm_head')]
    assert len(heads)==1
    w=heads[0].weight
    vocab,dim=w.shape
    assert (vocab,dim)==(248320,5120)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    packs=[];scales=[];residuals=[];wnorms=[];qnorms=[]
    for lo in range(0,vocab,4096):
        if datetime.now(timezone.utc)>=DEADLINE:raise TimeoutError('Deadline')
        original=w[lo:min(lo+4096,vocab)].T.contiguous()
        qref,packed,scale=marlin_quantize(original,scalar_types.uint8b128,-1)
        residuals.append(torch.linalg.vector_norm((original.float()-qref.float()).double(),dim=0).float()*(1+1e-6))
        wnorms.append(torch.linalg.vector_norm(original.double(),dim=0).float()*(1+1e-6))
        qnorms.append(torch.linalg.vector_norm(qref.double(),dim=0).float()*(1+1e-6))
        packs.append(packed);scales.append(scale)
        del original,qref
    packed=torch.cat(packs,dim=1).contiguous();scale=torch.cat(scales,dim=1).contiguous()
    del packs,scales
    residual=torch.cat(residuals);wn=torch.cat(wnorms);qn=torch.cat(qnorms)
    workspace=marlin_make_workspace_new(w.device)
    gamma=(2*dim*2**-24)/(1-2*dim*2**-24)
    gamma_ref=(dim*2**-24)/(1-dim*2**-24)
    def coarse(h):
        out=torch.empty((len(h),vocab),dtype=h.dtype,device=h.device)
        return ops.marlin_gemm(h,out,packed,None,scale,None,None,None,workspace,
            scalar_types.uint8b128,len(h),vocab,dim,use_atomic_add=False,use_fp32_reduce=True,is_zp_float=False)
    def dense(h):return torch.nn.functional.linear(h,w).argmax(dim=1)
    def fast(h,details=False,meta=None):
        scores=coarse(h).float()
        hn=torch.linalg.vector_norm(h.float(),dim=1)*(1+1e-5)
        bound=hn[:,None]*(residual+gamma*(wn+qn))[None,:]
        # Both original and coarse results are rounded to BF16.
        bound=(bound+2*(2**-8)*scores.abs())/(1-2**-8)+1e-4
        lower=scores-bound;upper=scores+bound
        best_lower=lower.amax(dim=1)
        count=(upper>=best_lower[:,None]).sum(dim=1)
        upper64,idx=upper.topk(64,dim=1)
        ww=w[idx].float()
        exact=(ww*h[:,None,:].float()).sum(dim=2)
        guard=gamma_ref*hn[:,None]*wn[idx]+1e-4
        guard=guard+(exact.abs()+guard)*(2**-8)
        lows=exact-guard;highs=exact+guard
        val,k=lows.max(dim=1)
        other=highs.scatter(1,k[:,None],float('-inf')).amax(dim=1)
        cert=(count<=64)&(val>other)
        predicted=idx.gather(1,k[:,None]).flatten()
        if details:return predicted,cert,count,scores,bound
        # The host decision and dense fallback cost belong in wall time.
        if meta is None:
            complete=cert.all()
        else:
            target=meta['target_indices'].cuda();draft=meta['draft_ids'].cuda();bonus=meta['bonus_indices'].cuda()
            matches=cert[target]&(predicted[target]==draft)
            reached=torch.cat([torch.ones(1,device=h.device,dtype=torch.bool),matches.to(torch.int32).cumprod(0).bool()])
            complete=~((reached[:-1]&~cert[target]).any() | (reached[-1]&~cert[bonus]).any())
        if not bool(complete):return dense(h)
        return predicted
    factor=residual+gamma*(wn+qn)
    def postprocess(h,coarse_scores,target,draft,bonus):
        scores=coarse_scores.float()
        hn=torch.linalg.vector_norm(h.float(),dim=1)*(1+1e-5)
        bound=(hn[:,None]*factor[None,:]+2*(2**-8)*scores.abs())/(1-2**-8)+1e-4
        lower=scores-bound;upper=scores+bound
        best_lower=lower.amax(dim=1)
        count=(upper>=best_lower[:,None]).sum(dim=1)
        _,idx=upper.topk(64,dim=1)
        exact=(w[idx].float()*h[:,None,:].float()).sum(dim=2)
        guard=gamma_ref*hn[:,None]*wn[idx]+1e-4
        guard=guard+(exact.abs()+guard)*(2**-8)
        lows=exact-guard;highs=exact+guard
        val,k=lows.max(dim=1)
        other=highs.scatter(1,k[:,None],float('-inf')).amax(dim=1)
        cert=(count<=64)&(val>other)
        predicted=idx.gather(1,k[:,None]).flatten()
        matches=cert[target]&(predicted[target]==draft)
        reached=torch.cat([torch.ones(1,device=h.device,dtype=torch.bool),matches.to(torch.int32).cumprod(0).bool()])
        complete=~((reached[:-1]&~cert[target]).any() | (reached[-1]&~cert[bonus]).any())
        return predicted,cert,complete,cert.all()
    compiled_post=torch.compile(postprocess,fullgraph=True,mode='reduce-overhead')
    compiled_validation=[]
    def fast_compiled(h,meta,all_rows=False):
        if len(h)==1:return dense(h)  # Prefill excluded from the optimization.
        assert meta is not None
        torch.compiler.cudagraph_mark_step_begin()
        pred,cert,prefix_ok,full_ok=compiled_post(h,coarse(h),meta['target_indices'],meta['draft_ids'],meta['bonus_indices'])
        if not bool(full_ok if all_rows else prefix_ok):return dense(h)
        return pred
    outcomes=[]
    for i,call in enumerate(calls):
        h=call['hidden'].cuda();ref=dense(h);pred,cert,count,scores,bound=fast(h,True)
        original=call['argmax_ids'].cuda()
        orig_value=call['top2_values'][:,0].cuda()
        winner=scores.gather(1,original[:,None]).flatten()
        winner_bound=bound.gather(1,original[:,None]).flatten()
        rows={'call':i,'case':call['case'],'rows':len(h),'dense_vs_captured_mismatches':int((ref!=original).sum()),'coarse_argmax_mismatches':int((scores.argmax(1)!=original).sum()),'winner_interval_violations':int(((winner-orig_value).abs()>winner_bound).sum()),'survivors':count.cpu().tolist(),'certified_rows':int(cert.sum()),'wrong_certificates':int(((pred!=original)&cert).sum()),'refined_argmax_mismatches':int((pred!=original).sum()),'full_call_fallback':not bool(cert.all()),'pipeline_mismatches':int((fast(h)!=original).sum())}
        rows['topk_tie_alternative_count']=int((call['top2_ids'][:,0].cuda()!=original).sum())
        rows['row_certificates']=cert.cpu().tolist()
        meta=call.get('meta')
        def emitted(ids):
            if meta is None:return ids.cpu().tolist()
            result=[]
            for pos,token in zip(meta['target_indices'].tolist(),meta['draft_ids'].tolist()):
                result.append(int(ids[pos]))
                if int(ids[pos])!=token:return result
            return result+ids[meta['bonus_indices'].cuda()].cpu().tolist()
        baseline_emitted=emitted(ref)
        candidate_emitted=emitted(fast(h,meta=meta))
        rows['prefix_output_mismatch']=candidate_emitted!=baseline_emitted
        rows['replay_vs_sampler_mismatch']=meta is not None and baseline_emitted!=meta['sampled_ids']
        rows['emitted_count']=len(baseline_emitted)
        if meta is not None:
            required=meta['target_indices'].tolist()[:len(baseline_emitted)]
            if len(baseline_emitted)>len(meta['draft_ids']):required+=meta['bonus_indices'].tolist()
        else:required=list(range(len(h)))
        rows['prefix_fallback']=not bool(cert[required].all())
        if len(h)==8:
            gpu_meta={k:v.cuda() if torch.is_tensor(v) else v for k,v in meta.items()}
            comp=fast_compiled(h,gpu_meta).clone()
            comp_full=fast_compiled(h,gpu_meta,True)
            rows['compiled_prefix_output_mismatch']=emitted(comp)!=baseline_emitted
            rows['compiled_full_mismatches']=int((comp_full!=original).sum())
        else:
            rows['compiled_prefix_output_mismatch']=False;rows['compiled_full_mismatches']=0
        outcomes.append(rows)
        (FOLDER/'outcomes.json').write_text(json.dumps(outcomes))
    print('QWEN38_HEAD_REAL_KERNEL_CHECK',json.dumps({'calls':len(outcomes),'rows':sum(r['rows'] for r in outcomes),'certified_rows':sum(r['certified_rows'] for r in outcomes),'wrong_certificates':sum(r['wrong_certificates'] for r in outcomes),'call_fallbacks':sum(r['full_call_fallback'] for r in outcomes),'pipeline_mismatches':sum(r['pipeline_mismatches'] for r in outcomes)}),flush=True)
    timings=[]
    # All captured calls, including slow fallback cases; never select fast cases.
    for case,record in enumerate(calls):
        h=record['hidden'].cuda();meta=record.get('meta')
        if meta is not None:meta={k:v.cuda() if torch.is_tensor(v) else v for k,v in meta.items()}
        variants=[('dense',lambda:dense(h)),('screen_refine_fallback',lambda:fast(h)),('prefix_screen_refine_fallback',lambda:fast(h,meta=meta)),('compiled_full',lambda:fast_compiled(h,meta,True)),('compiled_prefix',lambda:fast_compiled(h,meta))]
        for _ in range(2):
            for _,fn in variants:fn()
        torch.cuda.synchronize()
        for repeat in range(10):
            for name,fn in (variants if repeat%2==0 else variants[::-1]):
                a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
                start=time.perf_counter();a.record();result=fn();b.record();b.synchronize()
                timings.append({'m':len(h),'case':case,'prompt_case':record['case'],'repeat':repeat,'variant':name,'gpu_ms':a.elapsed_time(b),'wall_ms':1000*(time.perf_counter()-start)})
        (FOLDER/'timings.json').write_text(json.dumps(timings))
    summary={'outcomes':{k:sum(r[k] for r in outcomes) for k in ['rows','dense_vs_captured_mismatches','coarse_argmax_mismatches','winner_interval_violations','certified_rows','wrong_certificates','refined_argmax_mismatches','full_call_fallback','pipeline_mismatches','prefix_output_mismatch','replay_vs_sampler_mismatch','prefix_fallback','topk_tie_alternative_count','compiled_prefix_output_mismatch','compiled_full_mismatches']},'timings':[],'limitations':['Isolated head only; no end-to-end speed claim.','Existing Marlin W8A16 kernel; this screening/refinement approach has prior art.','Interval arithmetic assumptions are conservative but not a formal proof about every vendor kernel.','Original whole batch BF16 head is the fallback.','Frozen engineering activations, no held-out task results consulted.']}
    for m in [1,8]:
        for name in ['dense','screen_refine_fallback','prefix_screen_refine_fallback','compiled_full','compiled_prefix']:
            selected=[r for r in timings if r['m']==m and r['variant']==name]
            summary['timings'].append({'m':m,'variant':name,'n':len(selected),'gpu_ms_median':statistics.median(r['gpu_ms'] for r in selected),'wall_ms_median':statistics.median(r['wall_ms'] for r in selected),'wall_ms_mean':statistics.mean(r['wall_ms'] for r in selected),'gpu_ms_mean':statistics.mean(r['gpu_ms'] for r in selected)})
    (FOLDER/'summary.json').write_text(json.dumps(summary,indent=2))
    print('QWEN38_HEAD_REAL_KERNEL_SUMMARY',json.dumps(summary),flush=True)
    return summary

class HeadSpeedWorker:
    def research_head_speed(self):return run_probe(self.get_model())

def main():
    assert (DEADLINE-datetime.now(timezone.utc)).total_seconds()>3600
    FOLDER.mkdir(exist_ok=True);assert not (FOLDER/'completion.json').exists()
    kwargs=json.loads((BASE/'int4_dflash7/protocol.json').read_text())['kwargs']
    kwargs.update(worker_extension_cls='head_compiled_probe.HeadSpeedWorker',gpu_memory_utilization=0.60)
    (FOLDER/'protocol.json').write_text(json.dumps({'kwargs':kwargs,'source_activations':'Frozen head-prefix-feasibility-v2 activations and exact argmax/acceptance records','change':'Compile only pointwise screening/refinement; unchanged bounds,64candidates,MarlinW8,whole-batch fallback','screen_bits':8,'refine_top':64,'shape':[248320,5120],'timed_repetitions':10,'timed_calls':'all captured calls, no fast-case selection'},indent=2))
    status='running';start=time.monotonic()
    try:
        from vllm import LLM
        print('QWEN38_HEAD_SPEED_START',flush=True);llm=LLM(**kwargs)
        llm.collective_rpc('research_head_speed');status='complete' 
    except BaseException:
        status='failed';(FOLDER/'error.json').write_text(json.dumps({'traceback':traceback.format_exc()}));raise
    finally:
        (FOLDER/'completion.json').write_text(json.dumps({'status':status,'seconds':time.monotonic()-start}))
        print('QWEN38_HEAD_SPEED_EXIT',status,flush=True)
if __name__=='__main__':main()

"""Experimental batch-one greedy target-head verification.

Existing Marlin W8A16 + conservative intervals + prefix-conditioned fallback.
No random sampling, penalties, logprobs, or general-serving guarantee.
Full BF16 head retained for fallback; extra memory is required.
"""
from pathlib import Path
from datetime import datetime,timezone
import json
BASE=Path('/content/qwen38_research')
DEADLINE=datetime.fromisoformat('2026-10-14T07:11:31.211985+00:00')
def install(worker):
    model=worker.get_model()
    import torch
    from vllm import _custom_ops as ops
    from vllm.scalar_type import scalar_types
    from vllm.model_executor.layers.quantization.utils.marlin_utils_test import marlin_quantize
    from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_make_workspace_new
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
    gamma_ref=(2*dim*2**-24)/(1-2*dim*2**-24)
    def coarse(h):
        out=torch.empty((len(h),vocab),dtype=h.dtype,device=h.device)
        return ops.marlin_gemm(h,out,packed,None,scale,None,None,None,workspace,
            scalar_types.uint8b128,len(h),vocab,dim,use_atomic_add=False,use_fp32_reduce=True,is_zp_float=False)
    factor=residual+gamma*(wn+qn)
    def postprocess(h,coarse_scores,target,draft,bonus,forbidden):
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
        cert=cert&~(predicted[:,None]==forbidden[None,:]).any(dim=1)
        matches=cert[target]&(predicted[target]==draft)
        reached=torch.cat([torch.ones(1,device=h.device,dtype=torch.bool),matches.to(torch.int32).cumprod(0).bool()])
        complete=~((reached[:-1]&~cert[target]).any() | (reached[-1]&~cert[bonus]).any())
        return predicted,cert,complete,cert.all()
    compiled_post=torch.compile(postprocess,fullgraph=True,options={'triton.cudagraphs':False})

    from vllm.v1.worker.gpu.spec_decode.rejection_sampler import gather_draft_sampled
    processors=[m for n,m in model.named_modules() if n.endswith('logits_processor')]
    assert len(processors)==1
    processor=processors[0]
    assert processor.scale==1.0 and processor.soft_cap is None and heads[0].tp_size==1
    original_apply=processor._apply_head
    original_sampler=worker.model_runner.rejection_sampler
    target=torch.arange(7,device=w.device);bonus=torch.tensor([7],device=w.device)
    static_h=torch.zeros((8,dim),device=w.device,dtype=w.dtype)
    static_scores=torch.zeros((8,vocab),device=w.device,dtype=w.dtype)
    static_draft=torch.zeros(7,device=w.device,dtype=torch.int64)
    static_forbidden=torch.full((128,),-1,device=w.device,dtype=torch.int64)
    capture_stream=torch.cuda.Stream();capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        for _ in range(3):compiled_post(static_h,static_scores,target,static_draft,bonus,static_forbidden)
    torch.cuda.current_stream().wait_stream(capture_stream)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=capture_stream):
        graph_outputs=compiled_post(static_h,static_scores,target,static_draft,bonus,static_forbidden)
    torch.cuda.current_stream().wait_stream(capture_stream)
    torch.cuda.synchronize()
    state={'enabled':False,'pending':None,'safe_requests':{},'unsupported_reason':{},'counts':{'coarse_calls':0,'certified_prefix_calls':0,'dense_uncertain_calls':0,'dense_unsupported_calls':0}}
    # CUDA graphs retain device addresses, not ownership of external tensors.
    # Keep every capture input and closure constant alive for all future replays.
    state['_graph_keepalive']=(compiled_post,postprocess,target,bonus,w,factor,wn,qn,residual,static_h,static_scores,static_draft,static_forbidden,graph_outputs,graph,capture_stream)
    sampler=original_sampler.sampler
    assert [type(p).__name__ for p in sampler.logits_processors]==['LogitBiasState','PenaltiesState','BadWordsState']
    original_add=sampler.add_request
    def add_request(req_idx,params):
        original_add(req_idx,params)
        conditions={
            'greedy':params.temperature==0,
            'no_logprobs':params.logprobs is None and params.prompt_logprobs is None,
            'no_penalties':params.frequency_penalty==0 and params.presence_penalty==0 and params.repetition_penalty==1,
            'no_bad_words':not params.bad_words_token_ids,
            'no_allowed_tokens':not params.allowed_token_ids,
            'no_bias':not params.logit_bias,
            'no_structured_output':params.structured_outputs is None,
            'no_extra_args':not params.extra_args,
            'no_reasoning_parser':worker.vllm_config.reasoning_config is None,
        }
        state['safe_requests'][req_idx]=all(conditions.values())
        state['unsupported_reason'][req_idx]=[k for k,v in conditions.items() if not v]
        stop=sorted(params.all_stop_token_ids)
        assert len(stop)<=128
        # Conservative: always fall back for a stop-token winner, even after
        # the minimum-length constraint expires. No change to sampling policy.
        static_forbidden.copy_(torch.tensor(stop+[-1]*(128-len(stop)),device=w.device,dtype=torch.int64))
    sampler.add_request=add_request
    def apply_head(lm_head,h,embedding_bias):
        state['pending']=None
        if not state['enabled'] or h.ndim!=2 or h.shape!=(8,dim) or h.dtype!=torch.bfloat16 or embedding_bias is not None:
            return original_apply(lm_head,h,embedding_bias)
        state['pending']=(lm_head,h,embedding_bias)
        state['counts']['coarse_calls']+=1
        return coarse(h)
    class GreedyPrefixSampler:
        def __getattr__(self,name):return getattr(original_sampler,name)
        def __call__(self,logits,input_batch,draft_logits=None):
            pending=state['pending'];state['pending']=None
            if pending is None:return original_sampler(logits,input_batch,draft_logits)
            lm_head,h,bias=pending
            sampler=original_sampler.sampler;mapping=input_batch.idx_mapping_np
            supported=(len(mapping)==1 and len(logits)==8
                and bool((sampler.sampling_states.temperature.np[mapping]==0).all())
                and state['safe_requests'].get(int(mapping[0]),False)
                and sampler.get_logprobs_dims(mapping) is None
                and sampler.trace_replay_state is None
                and not sampler.return_sampling_mask
                and original_sampler.watermark_key is None)
            if not supported:
                state['counts']['dense_unsupported_calls']+=1
                logits=original_apply(lm_head,h,bias)
            else:
                drafts,_=gather_draft_sampled(input_batch.input_ids,input_batch.positions,input_batch.logits_indices,input_batch.expanded_idx_mapping,input_batch.expanded_local_pos,sampler.req_states.prefill_len.gpu)
                static_h.copy_(h);static_scores.copy_(logits);static_draft.copy_(drafts[1:])
                graph.replay()
                predicted,_,prefix_ok,_=graph_outputs
                if bool(prefix_ok):
                    state['counts']['certified_prefix_calls']+=1
                    # Greedy-only surrogate scores. Unsupported sampling always
                    # receives the original full logits instead.
                    logits=torch.full_like(logits,float('-inf')).scatter_(1,predicted[:,None],0)
                else:
                    state['counts']['dense_uncertain_calls']+=1
                    logits=original_apply(lm_head,h,bias)
            return original_sampler(logits,input_batch,draft_logits)
    processor._apply_head=apply_head
    worker.model_runner.rejection_sampler=GreedyPrefixSampler()
    worker._research_prefix_state=state
    return {'installed':True,'restricted_to':'Batch1 greedy,8verificationrows,optional min-token stop mask; dense fallback for stop winners and all unsupported processors','extra_packed_weight_bytes':packed.numel()*packed.element_size()}

class PrefixHeadWorker:
    def research_prefix_install(self):
        import torch
        with torch.inference_mode():return install(self)
    def research_prefix_enable(self,enabled):
        self._research_prefix_state['enabled']=bool(enabled)
        self._research_prefix_state['pending']=None
        return {'enabled':bool(enabled)}
    def research_prefix_stats(self):return dict(self._research_prefix_state['counts'],unsupported_reason=self._research_prefix_state['unsupported_reason'])

"""Pinned Qwen3.8-27B batch-one engine baseline / speculative-decoding study.

Run as a separate process in /content/qwen38_env after qwen38_setup.py.
One GPU, one engine at a time; no network listener, no CPU offloading.
"""
import os
os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN']='1'
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
os.environ['VLLM_WORKER_MULTIPROC_METHOD']='spawn'
import argparse,json,time,statistics,hashlib,traceback,dataclasses,subprocess
from pathlib import Path
from datetime import datetime,timezone

DEADLINE=datetime.fromisoformat('2026-10-14T07:11:31.211985+00:00')
BASE=Path('/content/qwen38_research')
REVISION='1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
TEXTS=[
 'A library stores books on shelves and records every loan in a catalogue. Each book has an identifier, a title, an author and a publication year. Readers can borrow a book, return it, or join a waiting list. The librarian checks for duplicate records and missing return dates. A nightly report counts active loans and overdue books. ',
 'An engineering team measures the response time of a service. The experiment uses one request at a time, fixes the input, warms the implementation and records repeated observations. The report separates startup time from steady-state latency. The team also verifies output correctness and documents hardware, software versions and uncertainty. '
]
TASKS=[
 'Using the material above, explain a careful plan for improving the library system. Include concrete steps and examples.',
 'Using the material above, explain how to run a fair performance experiment and avoid misleading conclusions.'
]
SMOKES=[
 'Compute 17 times 23. Give the numeric answer and one brief explanation.',
 'Ø§ÙƒØªØ¨ Ø«Ù„Ø§Ø« Ø®Ø·ÙˆØ§Øª ÙˆØ§Ø¶Ø­Ø© Ù„ØªÙ†Ø¸ÙŠÙ… ØªØ¬Ø±Ø¨Ø© Ø¹Ù„Ù…ÙŠØ© Ø¹Ø§Ø¯Ù„Ø© Ù„Ù…Ù‚Ø§Ø±Ù†Ø© Ø³Ø±Ø¹Ø© Ù†Ù…ÙˆØ°Ø¬ÙŠÙ†.',
 'Write a Python function that returns the first non-repeating character in a string. Explain its time complexity.',
 'Explain why a faster program still needs correctness tests. Use a small example.'
]
def save(folder,name,obj):
    p=folder/(name+'.tmp');p.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=str));p.replace(folder/name)
def check_budget():
    if datetime.now(timezone.utc)>=DEADLINE:raise TimeoutError('Authorized research deadline reached')
def serial(obj):
    if dataclasses.is_dataclass(obj):return dataclasses.asdict(obj)
    if isinstance(obj,dict):return {str(k):serial(v) for k,v in obj.items()}
    if isinstance(obj,(list,tuple)):return [serial(v) for v in obj]
    if hasattr(obj,'__dict__'):return serial(vars(obj))
    return obj

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--spec-tokens',type=int,default=7,choices=[7]);args=parser.parse_args()
    check_budget();assert (DEADLINE-datetime.now(timezone.utc)).total_seconds()>3600
    variant='int4_dflash7'
    folder=BASE/variant;folder.mkdir(exist_ok=True)
    assert not (folder/'completion.json').exists(), 'Keep completed trials immutable; choose a new explicit run for replication.'
    import torch,transformers,vllm
    from transformers import AutoTokenizer
    from vllm import LLM,SamplingParams
    assert 'A100' in torch.cuda.get_device_name(0)
    assert torch.cuda.device_count()==1
    snapshot=(BASE/'snapshot_path.txt').read_text().strip()
    quant_snapshot=(BASE/'int4_snapshot_path.txt').read_text().strip()
    tokenizer=AutoTokenizer.from_pretrained(snapshot,trust_remote_code=False)
    # Inputs are frozen before the first model output, shared by every variant.
    ipath=BASE/'benchmark_inputs.json'
    if ipath.exists():inputs=json.loads(ipath.read_text())
    else:
        inputs=[]
        for length in [128,512,2048]:
            for case in range(2):
                prefix=tokenizer.encode(TEXTS[case]*200,add_special_tokens=False)
                suffix=tokenizer.encode('\n\n'+TASKS[case],add_special_tokens=False)
                ids=prefix[:length-len(suffix)]+suffix
                assert len(ids)==length
                inputs.append({'length':length,'case':case,'token_ids':ids,'scope':'Document-continuation engineering workload; not a capability benchmark or chat-template comparison.'})
        save(BASE,'benchmark_inputs.json',inputs)
    input_sha=hashlib.sha256(ipath.read_bytes()).hexdigest()
    kwargs=dict(model=quant_snapshot,tokenizer=snapshot,dtype='bfloat16',kv_cache_dtype='bfloat16',tensor_parallel_size=1,seed=39001,
        max_model_len=4096,max_num_seqs=1,max_num_batched_tokens=4096,
        gpu_memory_utilization=.88,language_model_only=True,enable_prefix_caching=False,
        enforce_eager=False,disable_log_stats=False,trust_remote_code=False,cpu_offload_gb=0,
        compilation_config={'cudagraph_capture_sizes':[1+args.spec_tokens],'max_cudagraph_capture_size':1+args.spec_tokens})
    kwargs.update(spec_method='dflash',spec_model=(BASE/'dflash_snapshot_path.txt').read_text().strip(),spec_tokens=args.spec_tokens)
    protocol={'model':'RedHatAI/Qwen3.8-27B-INT4','revision':'91bd022d5b49442a868bc35008f6c21e1860edfa','original_bf16_revision':REVISION,'variant':variant,'kwargs':kwargs,
        'drafter':{'model':'incoai/Qwen3.8-27B-DFlash2','revision':'015e795645c74b1a0eeef3b570031fb62e769bc5'},
        'quantization':'Publisher AWQ/GPTQ W4A16 group128; explicit BF16 KV and original tokenizer; no local quality acceptance yet.',
        'sequence_reference':'int4' if args.spec_tokens else 'bf16',
        'fixed_output_tokens':128,'repetitions':5,'warmups_per_input':2,'batch':1,'input_sha256':input_sha,
        'primary':'host wall seconds for pretokenized single-request generation; includes prefill and engine overhead, excludes engine startup, warmup, tokenizer and detokenization',
        'secondary':'decode tokens/s from first/last engine token timestamps when available; 127 tokens divided by last-first',
        'fidelity':'Every repeated greedy fixed-length sequence must match its own first sequence. Speculative variants additionally compare to frozen BF16 token sequences.',
        'quality':'Four diagnostic prompts only; no quality acceptance. Quantized weights differ from BF16; DFlash2 retains the INT4 target; no claim of numerical sequence identity without verification.',
        'deadline_utc':DEADLINE.isoformat(),'hardware':torch.cuda.get_device_name(0),
        'versions':{'torch':torch.__version__,'transformers':transformers.__version__,'vllm':vllm.__version__}}
    save(folder,'protocol.json',protocol)
    rows=[];diagnostics=[];status='running';started=time.monotonic()
    def telemetry(name):
        p=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.used,memory.total,temperature.gpu,power.draw,clocks.sm,clocks.mem','--format=csv'],capture_output=True,text=True)
        save(folder,name,{'utc':datetime.now(timezone.utc).isoformat(),'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr})
    try:
        telemetry('gpu_before.json')
        print('QWEN38_ENGINE_START',variant,flush=True)
        engine_start=time.monotonic();llm=LLM(**kwargs)
        save(folder,'startup.json',{'seconds':time.monotonic()-engine_start})
        print('QWEN38_ENGINE_READY',variant,round(time.monotonic()-engine_start,1),flush=True)
        params=SamplingParams(temperature=0,max_tokens=128,min_tokens=128,ignore_eos=True,detokenize=False,seed=39001)
        prior=json.loads((BASE/('int4' if args.spec_tokens else 'bf16')/'timing.json').read_text())
        reference={(r['length'],r['case']):r['token_ids'] for r in prior}
        for item in inputs:
            prompt={'prompt_token_ids':item['token_ids']}
            for _ in range(2):check_budget();llm.generate([prompt],params,use_tqdm=False)
            own=None
            for repeat in range(5):
                check_budget();t=time.perf_counter();out=llm.generate([prompt],params,use_tqdm=False)[0];elapsed=time.perf_counter()-t
                tokens=list(out.outputs[0].token_ids);assert len(tokens)==128
                if own is None:own=tokens
                metrics=serial(out.metrics)
                if isinstance(metrics,dict):assert not metrics.get('is_corrupted',False), 'Engine reported nonfinite logits.'
                row={'length':item['length'],'case':item['case'],'repeat':repeat,'wall_seconds':elapsed,'output_tokens_per_second':128/elapsed,
                    'token_ids':tokens,'own_sequence_equal':tokens==own,'engine_metrics':metrics}
                row['baseline_sequence_equal']=tokens==reference[(item['length'],item['case'])]
                if isinstance(metrics,dict):
                    first,last=metrics.get('first_token_ts',0),metrics.get('last_token_ts',0)
                    if last>first>0:row['engine_decode_tokens_per_second']=127/(last-first)
                rows.append(row);save(folder,'timing.json',rows)
                assert row['own_sequence_equal'], 'Non-deterministic repeated greedy outputs; retain but do not admit speed claim.'
            group=rows[-5:]
            print('QWEN38_SPEED',variant,item['length'],item['case'],round(statistics.median(r['output_tokens_per_second'] for r in group),3),'request tokens/s',flush=True)
        for i,prompt in enumerate(SMOKES):
            check_budget();text=tokenizer.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
            out=llm.generate([text],SamplingParams(temperature=0,max_tokens=256,seed=39001),use_tqdm=False)[0]
            diagnostics.append({'index':i,'prompt':prompt,'text':out.outputs[0].text,'token_ids':list(out.outputs[0].token_ids),'finish_reason':out.outputs[0].finish_reason})
            save(folder,'diagnostics.json',diagnostics)
            print('QWEN38_DIAGNOSTIC',variant,i,repr(out.outputs[0].text[:500]),flush=True)
        save(folder,'engine_aggregate_metrics.json',serial(llm.get_metrics()))
        status='complete'
    except BaseException:
        status='failed';save(folder,'error.json',{'traceback':traceback.format_exc()});raise
    finally:
        telemetry('gpu_after.json')
        save(folder,'completion.json',{'status':status,'timings':len(rows),'diagnostics':len(diagnostics),'seconds':time.monotonic()-started})
        print('QWEN38_BENCHMARK',variant,status,flush=True)

if __name__=='__main__':main()

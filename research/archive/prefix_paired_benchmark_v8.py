"""Same-engine paired end-to-end validation of prefix-conditioned head fallback."""
import os
os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN']='1'
os.environ['VLLM_WORKER_MULTIPROC_METHOD']='spawn'
import json,time,hashlib,traceback,dataclasses,statistics
from pathlib import Path
from datetime import datetime,timezone
BASE=Path('/content/qwen38_research');FOLDER=BASE/'int4_dflash7_prefix_v8'
DEADLINE=datetime.fromisoformat('2026-10-14T07:11:31.211985+00:00')
VALIDATION=[
 'Explain the difference between a median and a mean using five numbers.',
 'Write Python code to merge two already sorted integer lists and discuss its complexity.',
 'A train travels 90 km in 75 minutes. Calculate its average speed with units.',
 'Summarize the water cycle in four short sentences for a ten-year-old.',
 'اكتب فقرة قصيرة تشرح لماذا نحتاج مجموعة اختبار مستقلة في التعلم الآلي.',
 'عندي خمسة صناديق، بكل صندوق اثنا عشر كتاباً. تبرعت بسبعة كتب؛ كم كتاباً بقي؟',
 '请用三句话解释为什么实验需要对照组。',
 'Explique en français la différence entre une hypothèse et une observation.',
 'Give a counterexample to the claim that every increasing sequence is bounded.',
 'Create a small JSON object describing a book with title, year and two authors.',
 'Design a simple test that would reveal an off-by-one error in binary search.',
 'Why can two floating-point reductions of the same numbers give different answers?'
]
def save(name,value):
    p=FOLDER/(name+'.tmp');p.write_text(json.dumps(value,indent=2,ensure_ascii=False,default=str),encoding='utf-8');p.replace(FOLDER/name)
def budget():
    if datetime.now(timezone.utc)>=DEADLINE:raise TimeoutError('Authorized deadline')
def main():
    assert (DEADLINE-datetime.now(timezone.utc)).total_seconds()>3600
    FOLDER.mkdir(exist_ok=True);assert not (FOLDER/'completion.json').exists()
    from vllm import LLM,SamplingParams
    from transformers import AutoTokenizer
    import torch
    kwargs=json.loads((BASE/'int4_dflash7/protocol.json').read_text())['kwargs']
    kwargs.update(worker_extension_cls='prefix_extension_v8.PrefixHeadWorker',gpu_memory_utilization=.60)
    protocol={'variant':'int4_dflash7_prefix_v8','kwargs':kwargs,'input_sha256':hashlib.sha256((BASE/'benchmark_inputs.json').read_bytes()).hexdigest(),'runtime_extension':{'install_rpc':'research_prefix_install','enable_rpc':'research_prefix_enable','file':'prefix_extension_v8.py','sha256':hashlib.sha256((BASE/'prefix_extension_v8.py').read_bytes()).hexdigest()},'sampling_policy':'Unchanged128min/max ignoreEOS; whitelist actual request parameters and conservatively fall back if a needed winner is a stop token.','repair':'Postprocess is a child of the same conditional parent, removing one serial host graph launch. This is a fixed implementation change on development prompts, with unchanged gate. Native CUDA IF/ELSE selects retained private PyTorch child graphs; removes host predicate read and avoids global Inductor graph-tree capture. Capability probe passed eight alternating branches with exactly four executions per branch. Frozen endpoints and numerical guards unchanged. Retain all CUDA graph external inputs and closure tensors for the lifetime of the worker; private manual CUDA graph; no global Inductor step reset. Refinement FP32 guard includes both candidate and reference accumulation via gamma(2d).','comparison':'Same engine/weights/allocated caches; alternate stock and candidate order across five paired repetitions; two warmups per mode and input.','reference':'Original BF16 head on the same INT4+DFlash2 engine','eligibility_gate':'All paired and twelve new diagnostic token sequences identical, repeated sequences stable, certified fast path actually used. Paired six-input geometric-mean latency improvement >1%; exploratory six-prompt bootstrap upper95% ratio <1.','diagnostics':VALIDATION,'hardware':torch.cuda.get_device_name(0),'deadline_utc':DEADLINE.isoformat(),'scope':'Greedy batch1 verification only; no end-to-end claim before measurements. Includes host sync, checking and dense fallback. Extra head weights consume memory.'}
    save('protocol.json',protocol)
    status='running';start=time.monotonic();rows=[];checks=[]
    try:
        print('QWEN38_PREFIX_ENGINE_START',flush=True);llm=LLM(**kwargs)
        health_prompt={'prompt_token_ids':json.loads((BASE/'benchmark_inputs.json').read_text())[0]['token_ids']}
        health_params=SamplingParams(temperature=0,max_tokens=32,min_tokens=32,ignore_eos=True,detokenize=False,seed=39001)
        health_before=list(llm.generate([health_prompt],health_params,use_tqdm=False)[0].outputs[0].token_ids)
        print('QWEN38_PREFIX_PREINSTALL_HEALTH_OK',flush=True)
        installed=llm.collective_rpc('research_prefix_install');save('installation.json',installed)
        health_after=list(llm.generate([health_prompt],health_params,use_tqdm=False)[0].outputs[0].token_ids)
        save('installation_health.json',{'before':health_before,'after':health_after,'equal':health_before==health_after})
        assert health_before==health_after,'Installation changed disabled-control output'
        print('QWEN38_PREFIX_POSTINSTALL_HEALTH_OK',flush=True)
        print('QWEN38_PREFIX_INSTALLED',json.dumps(installed),flush=True)
        params=SamplingParams(temperature=0,max_tokens=128,min_tokens=128,ignore_eos=True,detokenize=False,seed=39001)
        inputs=json.loads((BASE/'benchmark_inputs.json').read_text())
        for item in inputs:
            prompt={'prompt_token_ids':item['token_ids']};reference=None
            for mode in [False,True]:
                llm.collective_rpc('research_prefix_enable',args=(mode,))
                for _ in range(2):budget();llm.generate([prompt],params,use_tqdm=False)
            for repeat in range(5):
                for mode in ([False,True] if repeat%2==0 else [True,False]):
                    budget();llm.collective_rpc('research_prefix_enable',args=(mode,))
                    before=time.perf_counter();result=llm.generate([prompt],params,use_tqdm=False)[0];wall=time.perf_counter()-before
                    ids=list(result.outputs[0].token_ids);assert len(ids)==128
                    if reference is None:reference=ids
                    metrics=dataclasses.asdict(result.metrics) if dataclasses.is_dataclass(result.metrics) else vars(result.metrics)
                    row={'length':item['length'],'case':item['case'],'repeat':repeat,'mode':'prefix' if mode else 'stock','wall_seconds':wall,'output_tokens_per_second':128/wall,'token_ids':ids,'baseline_sequence_equal':ids==reference,'own_sequence_equal':ids==reference,'engine_metrics':metrics}
                    first,last=metrics.get('first_token_ts',0),metrics.get('last_token_ts',0)
                    if last>first>0:row['engine_decode_tokens_per_second']=127/(last-first)
                    rows.append(row);save('paired_timing.json',rows);save('timing.json',[r for r in rows if r['mode']=='prefix']);save('baseline_timing.json',[r for r in rows if r['mode']=='stock'])
                    assert not metrics.get('is_corrupted',False) and row['baseline_sequence_equal'],'Preserve divergent result; no speed admission.'
            stats=llm.collective_rpc('research_prefix_stats');save('extension_stats.json',stats)
            assert stats[0]['certified_prefix_calls']>0,'Fast path never used; no optimized speed claim.'
            print('QWEN38_PREFIX_PAIRED',item['length'],item['case'],json.dumps({mode:statistics.median(r['output_tokens_per_second'] for r in rows[-10:] if r['mode']==mode) for mode in ['stock','prefix']}),flush=True)
        tok=AutoTokenizer.from_pretrained((BASE/'snapshot_path.txt').read_text().strip(),trust_remote_code=False)
        for index,text in enumerate(VALIDATION):
            rendered=tok.apply_chat_template([{'role':'user','content':text}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
            inputs=tok.encode(rendered,add_special_tokens=False);pair={}
            for mode in [False,True]:
                budget();llm.collective_rpc('research_prefix_enable',args=(mode,))
                out=llm.generate([{'prompt_token_ids':inputs}],SamplingParams(temperature=0,max_tokens=256,detokenize=False,seed=39001),use_tqdm=False)[0]
                pair['prefix' if mode else 'stock']=list(out.outputs[0].token_ids)
            pair.update(index=index,prompt=text,equal=pair['prefix']==pair['stock']);checks.append(pair);save('fidelity_checks.json',checks)
            assert pair['equal'],'New-prompt fidelity failure; no quality/performance admission.'
            print('QWEN38_PREFIX_FIDELITY',index,True,flush=True)
        save('extension_stats.json',llm.collective_rpc('research_prefix_stats'));status='complete'
    except BaseException:
        status='failed';save('error.json',{'traceback':traceback.format_exc()});raise
    finally:
        save('completion.json',{'status':status,'timings':len(rows)//2,'paired_records':len(rows),'fidelity_checks':len(checks),'seconds':time.monotonic()-start})
        print('QWEN38_PREFIX_PAIRED_EXIT',status,len(rows),len(checks),flush=True)
if __name__=='__main__':main()

"""Attached workers, at most one per GPU; no sessions, descendants or autonomous retries."""
import subprocess,time
def run_all(tasks,gpus,heartbeat,interval=60.):
 if not gpus or len(set(gpus))!=len(gpus):raise ValueError('distinct GPU devices required')
 pending=list(enumerate(tasks));active={};codes=[None]*len(tasks);handles=[];last=0.
 try:
  while pending or active:
   for gpu in gpus:
    if gpu in active or not pending:continue
    i,t=pending.pop(0);handle=open(t['log'],'xb');handles.append(handle)
    active[gpu]=(i,subprocess.Popen(t['argv'],env=dict(t['env'],CUDA_VISIBLE_DEVICES=gpu),cwd=t['cwd'],start_new_session=False,stdout=handle,stderr=subprocess.STDOUT))
   for gpu,(i,c) in list(active.items()):
    if c.poll() is None:continue
    codes[i]=c.wait();del active[gpu]
    if codes[i]!=0:raise RuntimeError('worker failed: '+repr(codes))
   now=time.monotonic()
   if now-last>=interval:heartbeat(codes);last=now
   if pending or active:time.sleep(.25)
  return codes
 finally:
  # Workers contain no subprocess edges. Always reap every child, even after a
  # partial spawn/failure/interrupt.
  for _,c in active.values():
   if c.poll() is None:c.terminate()
  until=time.monotonic()+10
  for _,c in active.values():
   try:c.wait(timeout=max(.01,until-time.monotonic()))
   except subprocess.TimeoutExpired:c.kill();c.wait()
  for h in handles:h.close()

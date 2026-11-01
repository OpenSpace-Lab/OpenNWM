#!/usr/bin/env python3
"""Apply reviewed, time-local face masks without changing Go2 clip selection/timing.

Plan schema: {clips: [{id, boxes: [{time,panel,box}], tracks:
[{start,end,panel,keyframes:[[time,x,y,width,height],...]}]}]}.
Boxes are in the 1920x1080 comparison canvas. The plan is an internal review
artifact; this tool never changes metrics, trajectories or model predictions.
Run with an explicit runtime, e.g. conda run -n nwm-preprocess --no-capture-output.
"""
from __future__ import annotations
import argparse, concurrent.futures, hashlib, json, os, pathlib, subprocess, time
import cv2
import numpy as np
from PIL import Image
from publish_go2_gallery import FFMPEG, FFPROBE, probe, check_media_info, sha256

PANELS = {'main': (24,48,1200,675), 'gt': (24,774,384,288),
          'opennwm': (432,774,384,288), 'nwm': (840,774,384,288),
          'goal': (1632,576,240,180)}

def synchronized_plan(plan, interval):
    """Use the union of reviewed face locations equally in all three model views.

    The visual goal is held for a local-plan interval. Cover that interval and
    its adjacent boundary when transferring a reviewed face to/from the goal.
    Main-camera masks remain independent of the first-person camera.
    """
    output=dict(plan, boxes=[], tracks=[])
    for event in plan.get('boxes', []):
        source=event['panel']
        targets=['main'] if source=='main' else ['gt','opennwm','nwm','goal']
        sx,sy,sw,sh=PANELS[source]
        for target in targets:
            tx,ty,tw,th=PANELS[target]
            x,y,w,h=event['box']
            box=[tx+(x-sx)*tw/sw,ty+(y-sy)*th/sh,w*tw/sw,h*th/sh]
            times=[event['time']]
            if target=='goal' and source!='goal':
                times=[event['time'],event['time']-interval]
            elif source=='goal' and target!='goal':
                times=[event['time'],event['time']+interval]
            for at in times:
                output['boxes'].append(dict(event,time=at,panel=target,box=box))
    for track in plan.get('tracks', []):
        if track['panel']=='main':
            output['tracks'].append(track)
            continue
        # Sampling padded manual tracks also covers static goal frames between
        # context resets. No second padding is applied to these rectangles.
        points=track['keyframes']
        for at in np.arange(track['start'],track['end']+.001,.125):
            box=[float(np.interp(at,[p[0] for p in points],[p[j] for p in points])) for j in range(1,5)]
            sample={'boxes':[dict(time=float(at),panel=track['panel'],box=box,padded=True)]}
            output['boxes'].extend(synchronized_plan(sample,interval)['boxes'])
    return output

def rectangles(plan, at):
    result=[]
    for event in plan.get('boxes', []):
        if abs(event['time']-at) <= .5:
            x,y,w,h=event['box']
            if event.get('padded'):
                result.append((event['panel'],(x,y,w,h)))
            else:
                result.append((event['panel'],(x-w*.45,y-h*.5,w*1.9,h*2.0)))
    for track in plan.get('tracks', []):
        if track['start'] <= at <= track['end']:
            points=track['keyframes']
            box=[float(np.interp(at,[p[0] for p in points],[p[j] for p in points])) for j in range(1,5)]
            result.append((track['panel'],box))
    bounded=[]
    for panel,(x,y,w,h) in result:
        px,py,pw,ph=PANELS[panel]
        x0=max(px,int(np.floor(x))); y0=max(py,int(np.floor(y)))
        x1=min(px+pw,int(np.ceil(x+w))); y1=min(py+ph,int(np.ceil(y+h)))
        if x1>x0 and y1>y0: bounded.append((x0,y0,x1,y1))
    return bounded

def redact(frame, rects):
    for x0,y0,x1,y1 in rects:
        roi=frame[y0:y1,x0:x1]
        # Four cells across the head erase facial detail even in full-resolution downloads.
        small=cv2.resize(roi,(4,4),interpolation=cv2.INTER_AREA)
        frame[y0:y1,x0:x1]=cv2.resize(small,(x1-x0,y1-y0),interpolation=cv2.INTER_NEAREST)
    return frame

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--website',type=pathlib.Path,default=pathlib.Path('website'))
    p.add_argument('--plan',type=pathlib.Path,required=True)
    p.add_argument('--audit-dir',type=pathlib.Path,required=True)
    p.add_argument('--workers',type=int,default=4)
    args=p.parse_args(); cv2.setNumThreads(1)
    manifest=json.loads((args.website/'content/go2.json').read_text())
    clips={c['id']:c for c in manifest['clips']}
    plans=[plan for plan in json.loads(args.plan.read_text())['clips'] if plan.get('boxes') or plan.get('tracks')]
    args.audit_dir.mkdir(parents=True,exist_ok=True)
    print(json.dumps({'pid':os.getpid(),'cwd':str(pathlib.Path.cwd()),'plan':str(args.plan),'clips':len(plans)}),flush=True)
    def encode(plan):
        clip=clips[plan['id']]
        original=args.website/clip.get('privacy',{}).get('original_video',clip['video'])
        info=probe(original,FFPROBE)
        dest=original.with_name('comparison-privacy-v1.mp4'); poster=dest.with_name('poster-privacy-v1.webp')
        receipt=args.audit_dir/(clip['id']+'.json')
        plan_hash=hashlib.sha256(json.dumps({'version':2,'plan':plan},sort_keys=True).encode()).hexdigest()
        plan=synchronized_plan(plan,clip['goal_interval_s'])
        if receipt.exists():
            old=json.loads(receipt.read_text())
            if (old.get('plan_sha256')==plan_hash and dest.exists() and poster.exists()
                    and old['video_sha256']==sha256(dest) and old['poster_sha256']==sha256(poster)):
                print('CACHED '+clip['id'],flush=True); return old
        cap=cv2.VideoCapture(str(original)); fps=cap.get(cv2.CAP_PROP_FPS)
        tmp=dest.with_name('.comparison-privacy-v1.tmp.mp4')
        command=[str(FFMPEG),'-v','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s','1920x1080','-r',str(fps),'-i','pipe:0','-an','-c:v','libx264','-preset','medium','-crf','18','-threads','4','-pix_fmt','yuv420p','-movflags','+faststart',str(tmp)]
        stderr=(args.audit_dir/(clip['id']+'.ffmpeg.log')).open('w')
        proc=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=stderr)
        print(json.dumps({'start':clip['id'],'ffmpeg_pid':proc.pid,'command':command}),flush=True)
        count=0; masked_frames=0; masked_pixels=0; samples={}
        try:
            while True:
                ok,frame=cap.read()
                if not ok: break
                rects=rectangles(plan,count/fps)
                if rects:
                    masked_frames+=1
                    mask=np.zeros((1080,1920),np.uint8)
                    for x0,y0,x1,y1 in rects: mask[y0:y1,x0:x1]=1
                    masked_pixels+=int(mask.sum())
                    redact(frame,rects)
                if count==0: Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)).save(poster,quality=95)
                if count%24==0: samples[count]=frame.copy()
                proc.stdin.write(frame.tobytes()); count+=1
            proc.stdin.close(); code=proc.wait()
            if code: raise RuntimeError(f'ffmpeg {clip["id"]}: exit {code}')
        finally:
            cap.release(); stderr.close()
        os.replace(tmp,dest)
        output=probe(dest,FFPROBE); check_media_info(output,info,clip['duration_s'],dest)
        assert count==info['frames']
        decoded=cv2.VideoCapture(str(dest)); at=0; errors=[]; outside=[]
        while decoded.grab():
            if at in samples:
                ok,frame=decoded.retrieve(); assert ok
                diff=np.abs(frame.astype(np.float32)-samples[at].astype(np.float32))
                errors.append(float(diff.mean()))
                mask=np.ones((1080,1920),bool)
                for x0,y0,x1,y1 in rectangles(plan,at/fps): mask[max(0,y0-8):min(1080,y1+8),max(0,x0-8):min(1920,x1+8)]=False
                outside.append(float(diff[mask].mean()))
            at+=1
        decoded.release(); assert at==count
        assert max(errors)<5, (clip['id'],errors)
        result={'id':clip['id'],'video':str(dest.relative_to(args.website)),'poster':str(poster.relative_to(args.website)),
            'video_sha256':sha256(dest),'poster_sha256':sha256(poster),'original_video_sha256':sha256(original),
            'original_video':clip['video'],'original_poster':clip['poster'],'original_poster_sha256':sha256(args.website/clip['poster']),
            'redacted':True,'frame_count':count,'fps':fps,'duration_seconds':output['duration_s'],
            'codec':'h264','pixel_format':'yuv420p','resolution':[1920,1080],'faststart':True,
            'masked_frames':masked_frames,'masked_fraction':masked_pixels/(count*1920*1080),
            'encoding_crf':18,'encoding_sample_max_mae':max(errors),'unmasked_sample_max_mae':max(outside),
            'plan_sha256':plan_hash,'boxes':len(plan.get('boxes',[])),'tracks':len(plan.get('tracks',[]))}
        receipt.write_text(json.dumps(result,indent=2)+'\n')
        print('COMPLETE '+clip['id']+' '+json.dumps(result),flush=True); return result
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        results=list(pool.map(encode,plans))
    (args.audit_dir/'render-results.json').write_text(json.dumps(results,indent=2)+'\n')
    print('ALL COMPLETE',flush=True)
if __name__=='__main__': main()

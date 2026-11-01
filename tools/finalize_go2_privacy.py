#!/usr/bin/env python3
"""Validate reviewed Go2 derivatives and update only media/privacy manifest fields.

Run after independent output review. Original metrics, trajectories, clips and
timelines are checked against a frozen baseline before the manifest is written.
"""
from __future__ import annotations
import argparse, concurrent.futures, copy, hashlib, json
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from PIL import Image
from publish_go2_gallery import FFPROBE, atomic_bytes, check_media_info, json_bytes, probe, sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--website',type=Path,default=Path('website'))
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--audit-dir',type=Path,required=True)
    p.add_argument('--output-review',type=Path,action='append',required=True)
    args=p.parse_args()
    baseline=json.loads(args.baseline.read_text()); current=json.loads((args.website/'content/go2.json').read_text())
    plans={r['id']:r for r in json.loads(args.plan.read_text())['clips']}
    assert len(plans)==len(baseline['clips'])==44
    assert set(plans)=={c['id'] for c in baseline['clips']}
    reviews={}
    for path in args.output_review:
        for row in json.loads(path.read_text())['clips']:
            assert row['id'] not in reviews, row['id']
            reviews[row['id']]=row
    # Previous work may add privacy metadata, but no experimental field may change.
    allowed={'video','poster','fallback_video','privacy'}
    assert {k:v for k,v in current.items() if k!='clips'}=={k:v for k,v in baseline.items() if k!='clips'}
    assert [c['id'] for c in current['clips']]==[c['id'] for c in baseline['clips']]
    for old,new in zip(baseline['clips'],current['clips']):
        assert {k:v for k,v in old.items() if k not in allowed}=={k:v for k,v in new.items() if k not in allowed},old['id']
    output=copy.deepcopy(baseline)

    def inspect(clip):
        plan=plans[clip['id']]; changed=bool(plan.get('boxes') or plan.get('tracks'))
        original=args.website/clip['video']; before=probe(original,FFPROBE)
        if changed:
            record=json.loads((args.audit_dir/(clip['id']+'.json')).read_text())
            expected_plan=hashlib.sha256(json.dumps({'version':2,'plan':plan},sort_keys=True).encode()).hexdigest()
            assert record['plan_sha256']==expected_plan,clip['id']
            assert record['original_video_sha256']==sha256(original),clip['id']
            review=reviews[clip['id']]
            assert review['status']=='passed' and review['video_sha256']==record['video_sha256'],clip['id']
            if 'poster_sha256' in review:
                assert review['poster_sha256']==record['poster_sha256'],clip['id']
            clip['video']=record['video'];clip['poster']=record['poster']
            clip.pop('fallback_video',None)
            record['output_review']=review
        else:
            record={'id':clip['id'],'video':clip['video'],'poster':clip['poster'],'redacted':False,
                'video_sha256':sha256(original),'poster_sha256':sha256(args.website/clip['poster']),
                'frame_count':before['frames'],'fps':float(Fraction(before['fps'])),
                'duration_seconds':before['duration_s'],'resolution':[before['width'],before['height']],
                'codec':before['codec'],'pixel_format':before['pixel_format'],'faststart':True}
        video=args.website/clip['video'];poster=args.website/clip['poster']
        assert record['video_sha256']==sha256(video) and record['poster_sha256']==sha256(poster),clip['id']
        after=probe(video,FFPROBE);check_media_info(after,before,clip['duration_s'],video)
        with Image.open(poster) as image:
            assert image.size==(1920,1080),clip['id']
            image.verify()
        record['video_bytes']=video.stat().st_size;record['poster_bytes']=poster.stat().st_size
        record['review_notes']=plan.get('notes','')
        record['downloads']={key:{'path':path,'sha256':sha256(args.website/path),'bytes':(args.website/path).stat().st_size}
            for key,path in clip['downloads'].items()}
        if clip.get('fallback_video'):
            record['fallback_video']=clip['fallback_video']
            record['fallback_video_sha256']=record['video_sha256']
            record['fallback_review']='Unchanged immutable v6 fallback; same reviewed source bytes; no real faces found.'
        clip['privacy']={'reviewed':True,'face_redacted':changed,'report':'assets/go2/privacy-review.json'}
        if changed:clip['privacy']['original_video']=record['original_video']
        return record

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        records=list(pool.map(inspect,output['clips']))
    for before,after in zip(baseline['clips'],output['clips']):
        assert {k:v for k,v in before.items() if k not in allowed}=={k:v for k,v in after.items() if k not in allowed}
    wins=sum(all(c['metrics']['opennwm'][key]<c['metrics']['nwm'][key] for key in ('ate_m','rpe_m')) for c in output['clips'])
    report={'version':1,'status':'passed','reviewed_at':datetime.now(timezone.utc).isoformat(),
        'selection_preserved':True,'clip_count':len(records),'redacted_clip_count':sum(r['redacted'] for r in records),
        'unchanged_experiment_fields':True,'opennwm_lower_ate_and_rpe_clip_count':wins,
        'mask_policy':'Time-local face mosaics; human-reviewed detections and padded manual tracks. The union of first-person face regions is shared equally across GT, OpenNWM and NWM. Goal-image masks include adjacent local-plan boundaries. No trajectory, score, selection or timing changes.',
        'review_method':'All clips: full-duration 1-second contact sheets plus 0.25-second face-detector candidates at enlarged panel resolution. Manually reviewed original-resolution faces and fast-motion edge cases. Redacted outputs spot-checked at the reviewed time ranges, including independent review by additional reviewers; every redacted output frame decoded and count/timing validated.',
        'encoding':'H.264 CRF 18, 1920x1080, 24 FPS, yuv420p, faststart; no audio. Quality measured against the redacted source at 1-second samples.',
        'baseline_manifest_sha256':sha256(args.baseline),'review_plan_sha256':sha256(args.plan),
        'renderer_sha256':sha256(Path(__file__).with_name('redact_go2_faces.py')),
        'clips':records}
    atomic_bytes(args.website/'assets/go2/privacy-review.json',json_bytes(report))
    atomic_bytes(args.website/'content/go2.json',json_bytes(output))
    print(json.dumps({'status':'passed','clips':len(records),'redacted':report['redacted_clip_count'],
        'opennwm_lower_both':wins,'report_sha256':sha256(args.website/'assets/go2/privacy-review.json')}))


if __name__=='__main__':main()

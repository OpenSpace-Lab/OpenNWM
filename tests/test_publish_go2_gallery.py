import argparse
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('publisher', Path(__file__).resolve().parents[1] / 'tools/publish_go2_gallery.py')
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)

class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.source = self.base / 'nas/gallery'
        self.clip = self.source / 'clip_one'
        self.clip.mkdir(parents=True)
        for filename in ('comparison.mp4', 'poster.webp', 'trajectories.csv', *p.STANDALONES):
            (self.clip / filename).write_bytes(('MASTER:' + filename).encode())
        (self.clip / 'metrics.json').write_text('{"opennwm":{"ate_m":0.12}}')
        (self.source / 'report.json').write_text('{"protocol":"offline","clips":[]}')
        manifest = {'version':2,'status':'ready','clips':[{'id':'clip_one','duration_s':24,
          'video':'assets/go2/clip_one/comparison.mp4','poster':'assets/go2/clip_one/poster.webp',
          'downloads':{'ego':'assets/go2/clip_one/gt_ego.mp4'}, 'metrics':{'ate_m':0.12}}]}
        (self.source / 'website_manifest.json').write_text(json.dumps(manifest))
        self.sites = [self.base / 'site_a', self.base / 'site_b']
        for w in self.sites:
            (w / 'assets/go2/clip_one').mkdir(parents=True)
            (w / 'index.html').write_text('unchanged site')
            (w / 'content').mkdir()
            (w / 'content/go2.json').write_text(json.dumps(manifest))
            for name in p.STANDALONES:
                (w / 'assets/go2/clip_one' / name).write_bytes((self.clip / name).read_bytes())
        self.args = argparse.Namespace(source=self.source,website=self.sites,cache=None,validation=None,
            ffmpeg=Path('/usr/bin/true'),ffprobe=Path('/usr/bin/true'),jobs=2,crf=22,preset='medium',max_site_mb=1,retained_source=[],prepare_only=False)
    def fake_encode(self, clip, source, cache, settings, args):
        path = cache / clip['id'] / 'comparison.mp4'
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(b'small web video')
        return {'passed':True,'output':p.identity(path),'master':p.identity(self.clip / 'comparison.mp4'), 'cache_hit':False}
    def snapshots(self, root):
        return {str(f.relative_to(root)):p.identity(f) for f in root.rglob('*') if f.is_file()}
    def v3_fixture(self):
        manifest=p.read(self.source/'website_manifest.json')
        manifest['version']=3
        manifest['routes']=[{'id':'route_05','episode':'new_episode'}]
        manifest['selection']={'display_clip_count':1}
        clip=manifest['clips'][0]
        metrics={m:{'ate_m':.2,'rpe_m':.02} for m in ('opennwm','nwm')}
        aggregation={'method':'arithmetic_mean_of_complete_local_plans','plan_count':2,
                     'display':'constant_for_entire_clip','metrics':['ate_m','rpe_m'],'units':'meters'}
        clip.update(episode='new_episode',route_id='route_05',start_time_s=0,goal_interval_s=12,
                    metrics=metrics,evaluated_goal_ids=[0,1],metric_aggregation=aggregation)
        report={'id':'clip_one','episode':'new_episode','start_s':0,'end_s':24,
                'goal_interval_s':12,'metrics':metrics,'metric_aggregation':aggregation,
                'plans':[{'id':i,'start_time_s':i*12,'end_time_s':(i+1)*12,
                          'models':{m:{'metrics':{'ate_m':ate,'rpe_m':.02}} for m in ('opennwm','nwm')}}
                         for i,ate in enumerate((.1,.3))]}
        self.write_v3(manifest,report)
        return manifest,report
    def write_v3(self,manifest,report):
        (self.source/'website_manifest.json').write_text(json.dumps(manifest))
        (self.clip/'metrics.json').write_text(json.dumps(report))
        (self.source/'report.json').write_text(json.dumps({'schema_version':3,
            'episodes':['new_episode'],'clips':[report]}))
    def test_v3_publication_preserves_routes_and_clip_mean_contract(self):
        manifest,report=self.v3_fixture()
        masters=self.snapshots(self.source)
        with patch.object(p,'encode_clip',side_effect=self.fake_encode):
            p.publish(self.args)
        self.assertEqual(masters,self.snapshots(self.source))
        for site in self.sites:
            published=p.read(site/'content/go2.json')
            self.assertEqual(published['routes'],manifest['routes'])
            self.assertEqual(published['clips'][0]['metric_aggregation'],report['metric_aggregation'])
            self.assertEqual(p.read(site/'assets/go2/clip_one/metrics.json'),report)
    def test_v3_rejects_nonconstant_or_incomplete_mean_contract(self):
        manifest,report=self.v3_fixture()
        for field,value in [('method','weighted_mean'),('display','current_local_plan'),
                            ('plan_count',1),('units','waypoint_units')]:
            with self.subTest(field=field):
                changed=copy.deepcopy(report)
                changed['metric_aggregation'][field]=value
                self.write_v3(manifest,changed)
                with self.assertRaisesRegex(ValueError,'aggregation contract'):
                    p.load_manifest(self.source)
        missing=copy.deepcopy(manifest)
        missing['clips'][0].pop('metric_aggregation')
        self.write_v3(missing,report)
        with self.assertRaisesRegex(ValueError,'aggregation contract'):
            p.load_manifest(self.source)
    def test_v3_recomputes_mean_and_checks_plan_coverage_before_encoding(self):
        original_manifest,original_report=self.v3_fixture()
        for mutation in ('wrong_mean','missing_plan','time_gap','nonfinite','report_copy'):
            with self.subTest(mutation=mutation):
                manifest=copy.deepcopy(original_manifest);report=copy.deepcopy(original_report)
                if mutation=='wrong_mean':
                    report['metrics']['nwm']['ate_m']=.9
                    manifest['clips'][0]['metrics']['nwm']['ate_m']=.9
                elif mutation=='missing_plan':
                    report['plans'].pop()
                elif mutation=='time_gap':
                    report['plans'][1]['start_time_s']=13
                elif mutation=='nonfinite':
                    report['plans'][0]['models']['nwm']['metrics']['ate_m']=float('nan')
                self.write_v3(manifest,report)
                if mutation=='report_copy':
                    (self.clip/'metrics.json').write_text(json.dumps(original_report|{'start_s':1}))
                with patch.object(p,'encode_clip') as encode:
                    with self.assertRaises(ValueError):
                        p.publish(self.args)
                    encode.assert_not_called()
    def test_two_sites_preserve_masters_and_remove_verified_redundancy(self):
        masters = self.snapshots(self.source)
        with patch.object(p,'encode_clip',side_effect=self.fake_encode):
            p.publish(self.args)
        self.assertEqual(masters,self.snapshots(self.source))
        for site in self.sites:
            manifest=p.read(site/'content/go2.json')
            self.assertEqual(set(manifest['clips'][0]['downloads']),{'metrics','trajectories'})
            self.assertEqual(manifest['clips'][0]['metrics'],{'ate_m':0.12})
            for name in p.STANDALONES:
                self.assertFalse((site/'assets/go2/clip_one'/name).exists())
            self.assertEqual((site/'index.html').read_text(),'unchanged site')
            self.assertEqual((site/'assets/go2/report.json').read_bytes(),(self.source/'report.json').read_bytes())
        self.assertTrue(p.read(self.source.parent/'webpublish_validation.json')['passed'])
    def test_checksum_mismatch_aborts_before_encoding_or_site_change(self):
        (self.sites[1]/'assets/go2/clip_one/nwm.mp4').write_bytes(b'unmatched user data')
        snapshots=[self.snapshots(w) for w in self.sites]
        with patch.object(p,'encode_clip') as encode:
            with self.assertRaisesRegex(ValueError,'refusing removal'):
                p.publish(self.args)
            encode.assert_not_called()
        self.assertEqual(snapshots,[self.snapshots(w) for w in self.sites])
    def test_budget_includes_unrelated_assets_and_aborts_all_sites(self):
        (self.sites[1]/'assets/unrelated.bin').write_bytes(b'x'*1_000_000)
        snapshots=[self.snapshots(w) for w in self.sites]
        with patch.object(p,'encode_clip',side_effect=self.fake_encode):
            with self.assertRaisesRegex(ValueError,'budget exceeded'):
                p.publish(self.args)
        self.assertEqual(snapshots,[self.snapshots(w) for w in self.sites])
    def test_prepare_only_does_not_mutate_either_site(self):
        self.args.prepare_only=True
        snapshots=[self.snapshots(w) for w in self.sites]
        with patch.object(p,'encode_clip',side_effect=self.fake_encode):
            p.publish(self.args)
        self.assertEqual(snapshots,[self.snapshots(w) for w in self.sites])
        report=p.read(self.source.parent/'webpublish_validation.json')
        self.assertTrue(report['passed']);self.assertTrue(report['prepare_only'])
        self.assertFalse(report['published'])
    def test_explicit_retained_original_proves_safe_removal(self):
        retained=self.base/'nas/retained'
        (retained/'clip_one').mkdir(parents=True)
        original=(self.clip/'nwm.mp4').read_bytes()
        (retained/'clip_one/nwm.mp4').write_bytes(original)
        (self.clip/'nwm.mp4').write_bytes(b'regenerated but different master')
        rows=p.deletion_plan(self.sites[0],self.source,self.sites,[retained])
        proof=next(r for r in rows if r['path'].endswith('/nwm.mp4'))
        self.assertEqual(proof['original'],str(retained/'clip_one/nwm.mp4'))
    def test_destination_symlink_rejected(self):
        poster=self.sites[0]/'assets/go2/clip_one/poster.webp'
        poster.symlink_to(self.clip/'poster.webp')
        with self.assertRaisesRegex(ValueError,'symlink'):
            p.checked_destination(self.sites[0],'assets/go2/clip_one/poster.webp')
    def test_nas_original_must_not_depend_on_public_copy(self):
        (self.clip/'nwm.mp4').unlink()
        (self.clip/'nwm.mp4').symlink_to(self.sites[0]/'assets/go2/clip_one/nwm.mp4')
        with self.assertRaisesRegex(ValueError,'points into a website'):
            p.deletion_plan(self.sites[0],self.source,self.sites)
    def test_only_hash_validated_cache_reused(self):
        cache=self.base/'cache'; output=cache/'clip_one/comparison.mp4'
        output.parent.mkdir(parents=True);output.write_bytes(b'\x00\x00\x00\x08moovvalidated web bytes')
        settings={'version':1}
        receipt={'passed':True,'settings':settings,'master':p.identity(self.clip/'comparison.mp4'),
                 'output':p.identity(output),'fully_decoded':True, 'media':self.media_info()}
        (output.parent/'validation.json').write_text(json.dumps(receipt))
        clip=p.load_manifest(self.source)['clips'][0]
        media=self.media_info()
        with patch.object(p,'probe',return_value=media),patch.object(p.subprocess,'Popen') as process:
            result=p.encode_clip(clip,self.source,cache,settings,self.args)
            self.assertTrue(result['cache_hit']);process.assert_not_called()
        output.write_bytes(b'corrupt cache')
        with patch.object(p,'probe',return_value=media),patch.object(p.subprocess,'Popen',side_effect=RuntimeError('encoding required')):
            with self.assertRaisesRegex(RuntimeError,'encoding required'):
                p.encode_clip(clip,self.source,cache,settings,self.args)

    @staticmethod
    def media_info():
        return {'width':1920,'height':1080,'duration_s':24,'codec':'h264',
                'pixel_format':'yuv420p','stream_count':1,'frames':576,'fps':'24/1'}

    def test_incomplete_or_invalid_receipt_requires_full_revalidation(self):
        cache=self.base/'cache';output=cache/'clip_one/comparison.mp4'
        output.parent.mkdir(parents=True);output.write_bytes(b'\x00\x00\x00\x08moovvalidated web bytes')
        settings={'version':1};clip=p.load_manifest(self.source)['clips'][0]
        valid={'passed':True,'settings':settings,'master':p.identity(self.clip/'comparison.mp4'),
               'output':p.identity(output),'fully_decoded':True,'media':self.media_info()}
        mutations=[{'fully_decoded':False},{'passed':False},{'fully_decoded':None},
                   {'media':{}},{'media':{**self.media_info(),'width':1280}},
                   {'media':{**self.media_info(),'frames':575}},
                   {'media':{**self.media_info(),'fps':'25/1'}},
                   {'media':{**self.media_info(),'duration_s':25}},
                   {'media':{**self.media_info(),'pixel_format':'yuv444p'}}]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                (output.parent/'validation.json').write_text(json.dumps({**valid,**mutation}))
                with patch.object(p,'probe',return_value=self.media_info()), \
                     patch.object(p,'validate_video',return_value=self.media_info()) as validate, \
                     patch.object(p.subprocess,'Popen') as encode:
                    result=p.encode_clip(clip,self.source,cache,settings,self.args)
                    validate.assert_called_once_with(output,self.media_info(),24,self.args.ffmpeg,self.args.ffprobe)
                    encode.assert_not_called()
                    self.assertTrue(result['passed']);self.assertTrue(result['fully_decoded'])
                    self.assertEqual(result['media'],self.media_info())
        (output.parent/'validation.json').write_text(json.dumps({**valid,'fully_decoded':False}))
        with patch.object(p,'probe',return_value=self.media_info()), \
             patch.object(p,'validate_video',side_effect=ValueError('bad decode')):
            with self.assertRaisesRegex(ValueError,'bad decode'):
                p.encode_clip(clip,self.source,cache,settings,self.args)
        self.assertFalse(p.read(output.parent/'validation.json')['fully_decoded'])


if __name__ == '__main__':
    unittest.main(verbosity=2)

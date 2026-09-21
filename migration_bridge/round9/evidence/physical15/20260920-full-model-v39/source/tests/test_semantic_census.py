import copy
import importlib.util
import math
import pathlib
import unittest

TOOL=pathlib.Path(__file__).parents[1]/'tools'/'semantic_census.py'
PREFIX='model.language_model.layers.'


class SemanticCensusTests(unittest.TestCase):
    def fixture(self):
        config={'text_config':dict(num_hidden_layers=2,num_experts=4,
                    hidden_size=8,moe_intermediate_size=6,ple_layer_ids=[2],
                    split_ngram_parts=2,ple_embed_dim=8,heads_per_ngram=2,
                    ngram_size=3)}
        tensors={}
        def add(name,shape):
            tensors[name]={'shape':shape,'dtype':'BF16','bytes':math.prod(shape)*2}
        for i in range(2):
            add(f'{PREFIX}{i}.mlp.experts.gate_up_proj',[4,12,8])
            add(f'{PREFIX}{i}.mlp.experts.down_proj',[4,8,6])
        for i in range(2):
            add(f'{PREFIX}1.ple.ple_embedding.ngram_embedding.shard_{i}.weight',[10,2])
        add(f'{PREFIX}1.ple.key_proj.weight',[32,8])
        add('mtp.layers.0.mlp.experts.down_proj',[4,8,6])
        add('model.visual.example.weight',[8,8])
        return config,{'tensor_count':len(tensors),
                       'payload_bytes':sum(t['bytes'] for t in tensors.values()),
                       'tensors':tensors}

    def run_census(self,config,census):
        self.assertTrue(TOOL.exists(),'semantic census implementation is missing')
        spec=importlib.util.spec_from_file_location('semantic_census',TOOL)
        module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        return module.classify(census,config)

    def test_exact_categories_and_ple_layer_offset(self):
        c,t=self.fixture(); r=self.run_census(c,t)
        self.assertEqual(r['groups']['routed']['bytes'],2*4*3*8*6*2)
        self.assertEqual(r['groups']['ple_tables']['bytes'],80)
        self.assertEqual(r['ple_checkpoint_layers'],[1])
        self.assertEqual(sum(v['bytes'] for v in r['groups'].values()),t['payload_bytes'])

    def test_missing_expert_fails(self):
        c,t=self.fixture(); del t['tensors'][f'{PREFIX}1.mlp.experts.down_proj']
        with self.assertRaises(ValueError): self.run_census(c,t)

    def test_wrong_axis_or_dtype_fails(self):
        for change in ({'shape':[4,8,12]},{'dtype':'F32'}):
            c,t=self.fixture(); t['tensors'][f'{PREFIX}0.mlp.experts.gate_up_proj'].update(change)
            with self.assertRaises(ValueError): self.run_census(c,t)

    def test_ple_wrong_layer_missing_shard_and_width(self):
        for mode in ('layer','missing','width'):
            c,t=self.fixture()
            key=f'{PREFIX}1.ple.ple_embedding.ngram_embedding.shard_1.weight'
            if mode=='layer': c['text_config']['ple_layer_ids']=[1]
            elif mode=='missing': del t['tensors'][key]
            else: t['tensors'][key]['shape']=[10,4]
            with self.assertRaises(ValueError): self.run_census(c,t)

    def test_payload_census_mismatch(self):
        c,t=self.fixture(); t['payload_bytes']+=1
        with self.assertRaises(ValueError): self.run_census(c,t)

    def test_noncanonical_alias_cannot_duplicate_logical_tensor(self):
        for key in (f'{PREFIX}0.mlp.experts.gate_up_proj',
                    f'{PREFIX}1.ple.ple_embedding.ngram_embedding.shard_0.weight'):
            c,t=self.fixture()
            alias=key.replace('layers.0.','layers.00.').replace('shard_0.','shard_00.')
            t['tensors'][alias]=copy.deepcopy(t['tensors'][key])
            t['tensor_count']+=1; t['payload_bytes']+=t['tensors'][alias]['bytes']
            with self.assertRaises(ValueError): self.run_census(c,t)

    def test_no_input_mutation(self):
        c,t=self.fixture(); before=copy.deepcopy(t); self.run_census(c,t)
        self.assertEqual(t,before)


if __name__=='__main__': unittest.main()

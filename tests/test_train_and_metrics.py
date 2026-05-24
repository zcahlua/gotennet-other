from gotennet_other.train import TrainerConfig, train

class DS:
    def __len__(self): return 8
    def __getitem__(self,i):
        import torch
        pos=torch.randn(3,3)
        return {"atomic_numbers":[1,6,8],"positions":pos,"formation_energies":pos.pow(2).sum().reshape(1),"forces":-2*pos}

def test_train_runs():
    cfg=TrainerConfig(dataset_name='transition1x',device='cpu',epochs=1,batch_size=2,output_dir='outputs/test',num_workers=0)
    m=train(cfg,dataset=DS())
    assert 'best_metric' in m

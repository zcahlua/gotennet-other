from gotennet_other.data import OpenQDCLoader, collate_molecules

class D:
    def __len__(self): return 2
    def __getitem__(self,i):
        return {"z":[1,8] if i==0 else [6],"pos":[[0,0,0],[1,0,0]] if i==0 else [[0,0,1]],"energy":[0.1],"force":None if i else [[0,0,0],[0,0,0]],"name":f"n{i}"}

def test_collate_handles_missing_force():
    ds=OpenQDCLoader('transition1x',dataset=D(),split='all')
    b=collate_molecules([ds[0],ds[1]])
    assert b.z.shape[0]==3 and b.force is not None

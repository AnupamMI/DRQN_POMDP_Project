import numpy as np
import os
files=['dqn.npy','dqn_rewards_seed0.npy','drqn_4.npy','drqn_16.npy','drqn_rewards.npy','drqn_rewards_seed0.npy']
for f in files:
    if os.path.exists(f):
        a=np.load(f, allow_pickle=True)
        print(f, 'exists shape=', getattr(a,'shape',None), 'dtype=', getattr(a,'dtype',None))
        try:
            if a.ndim>1:
                m=a.mean(axis=0)[-50:].mean()
            else:
                m=a[-50:].mean()
            print(' mean last50=',m)
        except Exception as e:
            print(' could not compute mean:', e)
    else:
        print(f,'MISSING')

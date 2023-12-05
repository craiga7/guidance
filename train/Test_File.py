# Test_File
import numpy as np
from gymnasium import spaces

observation_space = spaces.Discrete(5, start=1) # {0, 1}

print(observation_space.sample())

x = spaces.Box(
                low=np.array([1]),
                high=np.array([5]),
                shape=(1,),
                dtype=np.int0
            )
print(x.sample())

y = spaces.Discrete(n=5,start=1)
print(y.sample())
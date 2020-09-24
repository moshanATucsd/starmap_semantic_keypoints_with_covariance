## Readme 

This repo demonstrates how to compute the covariances using MC dropout for the semantic keypoints in [OrcVIO](http://me-llamo-sean.cf/orcvio_githubpage/). 

### how to train 

- use [this repo](https://github.com/moshanATucsd/starmap_train_with_dropout)
- pretrained weights are [here](https://github.com/moshanATucsd/orcvio_pytorch_models)

### how to run 

* modify the path 

```
    root = '/home/erl/moshan/other_stuff/star_map_semantic_keypoints/'
    model_path = '/home/erl/moshan/orcvio_gamma/orcvio_gamma/pytorch_models/starmap/trained_models/with_dropout/model_cpu.pth'
    img_path = root + 'images/car2.png'
    det_name = root + 'det/car2.png'
```

* run the main 

```
python src/main.py
```

* sample output 

![img](/assets/kp_cov.png)

### modifications to make it work with python 3

* modify pytorch

run

```
atom /lib/python3.6/site-packages/torch/serialization.py
```

and modify the code in `_load`

```
unpickler = pickle_module.Unpickler(f)
unpickler.persistent_load = persistent_load
result = unpickler.load()
```

change to

```
try:
    unpickler = pickle_module.Unpickler(f)
    unpickler.persistent_load = persistent_load
    result = unpickler.load()
except:
    unpickler = pickle_module.Unpickler(f, encoding='latin1')
    unpickler.persistent_load = persistent_load
    result = unpickler.load()
```

this is because the model is saved in python 2 but we use python 3

* modify mhParser

In src/mhParser.py, change the original part to this

```
for i in range(size // 2, det.shape[0] - size // 2):
  for j in range(size // 2, det.shape[1] - size // 2):
    pool[i, j] = (max(det[i - 1, j - 1], det[i - 1, j], det[i - 1, j + 1],
                     det[i, j - 1], det[i, j], det[i, j + 1],
                     det[i + 1, j - 1], det[i + 1, j], det[i + 1, j + 1]))
```

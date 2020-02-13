## how to run 

* modify the path 

```
    root = '/home/erl/moshan/other_stuff/star_map_semantic_keypoints/'
    model_path = '/home/erl/moshan/orcvio_gamma/orcvio_gamma/pytorch_models/starmap/trained_models/no_dropout/model_cpu.pth'
    img_path = root + 'images/car2.png'
    det_name = root + 'det/car2.png'
```

* run the main 

```
python src/main.py
```

* sample output 

![img](/assets/sample_output.png)
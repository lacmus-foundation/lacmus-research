|        Model                               |    Full   | Winter   |   Summer |   Spring |
|--------------------------------------------|-----------|----------|----------|----------|               
|resnet50_base_best.h5                       |   0.0023  |  0.0054  |  0.0005  |  0.0000  |
|resnet50_liza_alert_v1.h5                   |   0.6253  |  0.9565  |  0.3242  |  0.2544  |
|laddv4_summer_epoch2.h5                     |   0.5721  |  0.6705  |  0.7805  |  0.2005  |
|laddv4_summer_epoch10.h5                    |   0.5744  |  0.5988  |  0.8977  |  0.1632  |
|finetuning_800_1333_e7.h5                   |   0.9221  |  0.9671  |  0.8915  |  0.8526  |
|resnet50_2000_1500_inference.h5 (2000x1500) |   0.9536  |  0.9792  |  0.9228  |  0.9147  |
|mobilenet_v3_small_pascal_07.h5 (1635x981)  |   0.6923  |  0.6302  |  0.8079  |  0.7018  |



|        Model                             | size      |    Full   | Winter   | Spring   |  SummerM | SummerT  | comments  |
|------------------------------------------|-----------|-----------|----------|----------|----------|----------|-----------|
|resnet50_liza_alert_v1.h5                 |           |   0.5605  | 0.9565   | 0.2526   | 0.2734   | 0.5126   | |
|resnet50_liza_alert_prod.h5               |           |   0.7573  | 0.9614   | 0.8408   | 0.8902   | 0.5951   | |
|resnet50_pascal_20_ladd.h5                |           |   0.8443  | 0.9426   | 0.6823   | 0.7080   | 0.8345   | |
|resnet50_pascal_15_1500_2000_ladd.h5      |           |   0.9356  | 0.9825   | 0.8956   | 0.8839   | 0.9315   | |
|resnet50_LADD.pth (basic torch)		   | 1500,2000 |   0.8767  | 0.9595   | 0.6556   | 0.8053   | 0.8987   | *1  |
|resnet50_LADD_epoch_8.pth (2 phase torch) | 1500,2000 |   0.8985  | 0.9731   | 0.7455   | 0.8067   | 0.9146   | *2  |

---
*1 Not tuned ancors, not train on empty images?, trainable_backbone_layers default = None
*2 Not tuned ancors, not train on empty images?, train head, then head+backbone


SDD ds metric

|  Model & params                         |    DS     | eval DS | size      | config | mAp    | Epoch |
| ----------------------------------------|-----------|---------|-----------|--------|--------| ------|
| keras, from CoCo	                      | peds only | val     | default   |   +    | 0.3391 |   8   |
| keras, from CoCo                        | all       | val     | defalut   |   +    | 0.3928 |   11  |
| keras, from CoCo, no-random-transform   | peds only | val     | 1500x2000 |   +    | 0.4197 |   3   |
| keras, from CoCo, no-random-transform   | all       | val     | 1088x1424 |   +    | 0.3708 |   7   |
| pretrained SDD                          | peds      | val     | default   |   -    | 0.6030 |   -   |
| keras, from CoCo                        | peds      | val     | default   |   -    | 0.4286 |   7   |
| keras, from CoCo                        | peds      | test    | default   |   -    | 0.3200 |   7   |
| keras, from CoCo,  freeze bb +/         | peds      | val     | default   |   -    | 0.4065 |   19  |
| keras, from OID, freeze bb +/           | peds      | val     | default   |   -    | 0.4094 |   24  |
| torch, from CoCo                        | peds      | test    | no_resize |   -    | 0.2238 |   4   |
| torch, from CoCo                        | peds      | val     | no_resize |   -    | 0.2914 |   8   |
| torch, from CoCo                        | peds      | val     | 800,1333  |   -    | 0.3601 |   5   | 
| torch, from CoCo, lr 0.005              | peds      | val     | 800,1333  |   -    | 0.3774 |   5   |

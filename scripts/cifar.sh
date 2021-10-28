python anneal.py --net ResNet18 --dataset CIFAR10 --data_path /vulcanscratch/psando/cifar-10/ \
--recipe targeted --eps 8 --budget 1.0 --save poison_dataset --poison_path /vulcanscratch/psando/untrainable_datasets/adv_poisons \
--attackoptim PGD
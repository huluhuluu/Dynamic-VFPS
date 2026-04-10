# python -u test_gpu.py --encryption tenseal --mi-mode static --epochs 100 --local-epochs 3 | tee tenseal_static_ep100_lr0.001_bs256_lep3.log
# python -u test_gpu.py --encryption paillier --mi-mode static --epochs 100 --local-epochs 3 | tee paillier_static_ep100_lr0.001_bs256_lep3.log

# python -u test_gpu.py --encryption tenseal --mi-mode static --epochs 100 | tee tenseal_static_ep100_lr0.001_bs256_lep1.log
# python -u test_gpu.py --encryption paillier --mi-mode static --epochs 100 | tee paillier_static_ep100_lr0.001_bs256_lep1.log


# python -u test_gpu.py --encryption tenseal --mi-mode static --epochs 100 --lr 0.0005 | tee tenseal_static_ep100_lr0.0005_bs256_lep1.log
# python -u test_gpu.py --encryption paillier --mi-mode static --epochs 100 --lr 0.0005 | tee paillier_static_ep100_lr0.0005_bs256_lep1.log

# selected 8
# python -u test_gpu.py --encryption tenseal --mi-mode static --epochs 100 --selected 8 | tee tenseal_static_ep100_lr0.001_bs256_sc8.log
# python -u test_gpu.py --encryption paillier --mi-mode static --epochs 100 --selected 8 | tee paillier_static_ep100_lr0.001_bs256_sc8.log


# python -u test_gpu.py --encryption tenseal --mi-mode static --epochs 100 --dataset cifar-10 --lr 0.003 | tee cifar10_tenseal_static_ep100_lr0.003_bs256_lep1.log
# python -u test_gpu.py --encryption paillier --mi-mode static --epochs 100 --dataset cifar-10 --lr 0.003 | tee cifar10_paillier_static_ep100_lr0.003_bs256_lep1.log


python -u test_gpu.py --encryption tenseal --mi-mode static --epochs 300 --dataset cifar-10 --lr 0.01 | tee cifar10_tenseal_static_ep200_lr0.01_bs256_lep1.log
python -u test_gpu.py --encryption paillier --mi-mode static --epochs 300 --dataset cifar-10 --lr 0.01 | tee cifar10_paillier_static_ep200_lr0.01_bs256_lep1.log

# python -u test_gpu.py --encryption tenseal --mi-mode static --epochs 200 --dataset cifar-10 --lr 0.01 --local-epochs 3 | tee cifar10_tenseal_static_ep200_lr0.01_bs256_lep3.log
# python -u test_gpu.py --encryption paillier --mi-mode static --epochs 200 --dataset cifar-10 --lr 0.01 --local-epochs 3 | tee cifar10_paillier_static_ep200_lr0.01_bs256_lep3.log
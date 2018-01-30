#########################
# To evaluate our modules
#########################

# To run a pretrained cifar10 module
wget https://www.dropbox.com/s/kjxh1oaxkhtpbmw/anytime_cifar_10.pth.tar
python3 main.py --model msdnet -b 20  -j 2 cifar10 --msd-blocks 10 --msd-base 4 \
--msd-step 2 --msd-stepmode even --growth 6-12-24 --gpu 0 --resume \
--evaluate-from anytime_cifar_10.pth.tar

# To run cifar10, all shared GCN, kernel 5, base 2, step 1
wget https://www.dropbox.com/s/ifu6efuuhzn9mwi/all_shared_gcn_cifar10_b2_k5_s1.tar
python3 main.py --model msdnet -b 64  -j 2 cifar10 --msd-blocks 10 --msd-base 2 \
--msd-step 1 --msd-stepmode even --growth 6-12-24 --gpu 0  --msd-gcn --msd-gcn-kernel 5 \
--msd-share-weights --msd-all-gcn --resume --evaluate-from all_shared_gcn_cifar10_b2_k5_s1.tar

# To run cifar100, all shared gcn, kernel 5, base 3, step 1
wget https://www.dropbox.com/s/6rt2qhfajoq3bte/all_shared_gcn_cifar100_b3_k5_s1.tar
python3 main.py --model msdnet -b 64  -j 2 cifar100 --msd-blocks 10 --msd-base 3 \
--msd-step 1 --msd-stepmode even --growth 6-12-24 --gpu 0 --msd-gcn --msd-gcn-kernel 5 \
--msd-share-weights --msd-all-gcn --resume --evaluate-from all_shared_gcn_cifar100_b3_k5_s1.tar

#############################
# To reproduce CIFAR results
#############################

# Anytime setup, cifar10
python3 main.py --model msdnet -b 64  -j 2 cifar10 --msd-blocks 10 --msd-base 4 \
--msd-step 2 --msd-stepmode even --growth 6-12-24 --gpu 0 --savedir anytime_cifar10/

# Anytime setup, cifar100
python3 main.py --model msdnet -b 64  -j 2 cifar100 --msd-blocks 10 --msd-base 4 \
--msd-step 2 --msd-stepmode even --growth 6-12-24 --gpu 0 --savedir anytime_cifar100/

#############################
# To apply GCN on First layer
#############################

# cifar10, first layer GCN, kernel 7, base 2
python3 main.py --model msdnet -b 64  -j 2 cifar10 --msd-blocks 10 --msd-base 2 \
--msd-step 2 --msd-stepmode even --growth 6-12-24 --gpu 0 --msd-gcn --msd-gcn-kernel 7

# cifar10, first layer GCN, kernel 5, base 3
python3 main.py --model msdnet -b 64  -j 2 cifar10 --msd-blocks 10 --msd-base 3 \
--msd-step 2 --msd-stepmode even --growth 6-12-24 --gpu 0  --msd-gcn --msd-gcn-kernel 5

# cifar100, first layer GCN, kernel 7, base 2
python3 main.py --model msdnet -b 64  -j 2 cifar100 --msd-blocks 10 --msd-base 2 \
--msd-step 2 --msd-stepmode even --growth 6-12-24 --gpu 0  --msd-gcn --msd-gcn-kernel 7

# cifar100, first layer GCN, kernel 5, base 3
python3 main.py --model msdnet -b 64  -j 2 cifar100 --msd-blocks 10 --msd-base 3 \
--msd-step 2 --msd-stepmode even --growth 6-12-24 --gpu 0 --msd-gcn --msd-gcn-kernel 5

#############################
# To apply GCN on all layers
#############################

# cifar10, all shared GCN, kernel 5, base 2, step 1
python3 main.py --model msdnet -b 64  -j 2 cifar10 --msd-blocks 10 --msd-base 2 \
--msd-step 1 --msd-stepmode even --growth 6-12-24 --gpu 0  --msd-gcn --msd-gcn-kernel 5 \
--msd-share-weights --msd-all-gcn

# cifar100, all shared gcn, kernel 5, base 3, step 1
python3 main.py --model msdnet -b 64  -j 2 cifar100 --msd-blocks 10 --msd-base 3 \
--msd-step 1 --msd-stepmode even --growth 6-12-24 --gpu 0  --msd-gcn --msd-gcn-kernel 5 \
--msd-share-weights --msd-all-gcn

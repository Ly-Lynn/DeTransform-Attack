python task1.py --model efficientnet_v2_m --data_dir data/ --output_dir output/ --attack_type FGSM --epsilon 0.01 --seed 22520766
python task1.py --model efficientnet_v2_m --data_dir data/ --output_dir output/ --attack_type PGD --epsilon 0.01 --alpha 0.01 --num_steps 10 --seed 22520766
python task1.py --model efficientnet_v2_m --data_dir data/ --output_dir output/ --attack_type DDN --epsilon 0.01 --num_steps 10 --seed 22520766


python task1.py --model vgg16 --data_dir data/ --output_dir output/ --attack_type FGSM --epsilon 0.01 --seed 22520766
python task1.py --model vgg16 --data_dir data/ --output_dir output/ --attack_type PGD --epsilon 0.01 --alpha 0.01 --num_steps 10 --seed 22520766
python task1.py --model vgg16 --data_dir data/ --output_dir output/ --attack_type DDN --epsilon 0.01 --num_steps 10 --seed 22520766

python task1.py --model resnet50 --data_dir data/ --output_dir output/ --attack_type FGSM --epsilon 0.01 --seed 22520766
python task1.py --model resnet50 --data_dir data/ --output_dir output/ --attack_type PGD --epsilon 0.01 --alpha 0.01 --num_steps 10 --seed 22520766
python task1.py --model resnet50 --data_dir data/ --output_dir output/ --attack_type DDN --epsilon 0.01 --num_steps 10 --seed 22520766

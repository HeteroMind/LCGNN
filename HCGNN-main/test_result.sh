
#HCGNN
python3 DP_AC_HAGNN_search_retrain_noHGNN.py --dataset=DBLP  --valid-attributed-type=1 --hidden-dim 64 --num-heads 8 --attn-vec-dim 128 --num_layers 2 --beta_1 0.3 --droupout 0.5 --slope 0.05 --grad_clip 5

#python3 DP_AC_HAGNN_search_retrain_noHGNN.py --dataset=IMDB  --valid-attributed-type=0 --hidden-dim 64 --num-heads 8 --attn-vec-dim 128 --num_layers 2 --beta_1 1 --droupout 0.5 --slope 0.1 --grad_clip 5

#python3 DP_AC_HAGNN_search_retrain_noHGNN.py --dataset=ACM  --valid-attributed-type=0 --hidden-dim 64 --num-heads 8 --attn-vec-dim 96 --num_layers 2 --beta_1 0.3 --droupout 0.5 --slope 0.05 --grad_clip 5


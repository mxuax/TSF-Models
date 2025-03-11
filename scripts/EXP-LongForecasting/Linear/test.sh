
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=192
model_name=DLinear

for model_name in NLinear DLinear
do
for pred_len in 96 336 720
do 
for seq_len in 48 96 192 336
do

 python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path sin_add_noise.csv \
  --model_id sin_add_noise_$seq_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 1 \
  --des 'Exp' \
  --itr 1 --batch_size 8 --learning_rate 0.0005 >logs/LongForecasting/$model_name'_'sin_add_noise_$seq_len'_'$pred_len.log 

 python -u run_longExp.py \
   --is_training 1 \
   --root_path ./dataset/ \
   --data_path sin_multi_noise.csv \
   --model_id sin_multi_noise_$seq_len'_'$pred_len \
   --model $model_name \
   --data custom \
   --features M \
   --seq_len $seq_len \
   --pred_len $pred_len \
   --enc_in 1 \
   --des 'Exp' \
   --itr 1 --batch_size 8 --learning_rate 0.0005 >logs/LongForecasting/$model_name'_'sin_multi_noise_$seq_len'_'$pred_len.log 

 python -u run_longExp.py \
   --is_training 1 \
   --root_path ./dataset/ \
   --data_path sin_linear_noise1.csv \
   --model_id sin_linear_noise1_$seq_len'_'$pred_len \
   --model $model_name \
   --data custom \
   --features M \
   --seq_len $seq_len \
   --pred_len $pred_len \
   --enc_in 1 \
   --des 'Exp' \
   --itr 1 --batch_size 8 --learning_rate 0.0005 >logs/LongForecasting/$model_name'_'sin_linear_noise1_$seq_len'_'$pred_len.log 

done
done
done


# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

python run_ner_via_tp.py \
  --model_name_or_path roberta-base \
  --dataset_name data \
  --output_dir /data/test-ner \
  --do_train --do_eval \
  --evaluation_strategy steps \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --learning_rate 1e-4 \
  --num_train_epochs 10 \
  --max_seq_length 128 \
  --stride_window 32 \
  --data_seed 421 \
  --eval_delay 1000 \
  --save_steps 1000 \
  --logging_steps 1000 \
  --labels /data/label_list.txt \
  --weight_decay 0.01 \
  --save_total_limit 2 \
  --metric_for_best_model eval_f1-score \
  --greater_is_better True \
  --dataload_num_workers 4 \
  --use_cuda False

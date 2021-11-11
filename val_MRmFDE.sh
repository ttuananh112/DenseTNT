DATA_DIR=/home/anhtt163/dataset/prediction/argoverse/forecasting_train_v1.1/val/data/; \
OUTPUT_DIR=outputs/models.densetnt.1; \
python src/run.py --argoverse --future_frame_num 30 \
--do_train --data_dir ${DATA_DIR} --output_dir ${OUTPUT_DIR} \
--hidden_size 128 --train_batch_size 64 --sub_graph_batch_size 4096 --use_map \
--core_num 16 --use_centerline \
--distributed_training 1 \
--other_params semantic_lane direction l1_loss \
goals_2D enhance_global_graph subdivide lazy_points new laneGCN point_sub_graph \
stage_one stage_one_dynamic=0.95 laneGCN-4 point_level point_level-4 \
point_level-4-3 complete_traj complete_traj-3 \
--do_eval --eval_params optimization MRminFDE=0.0 cnt_sample=9 opti_time=0.1
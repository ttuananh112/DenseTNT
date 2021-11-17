DATA_DIR=/raid/anhtt163/dataset/prediction/carla_simulation/v4/data/batch01/dynamic_by_ts/; \
OUTPUT_DIR=outputs/carla.models.densetnt.4; \
python src/run_carla.py --argoverse --future_frame_num 30 \
--num_train_epochs 30 \
--do_train --data_dir ${DATA_DIR} --output_dir ${OUTPUT_DIR} \
--hidden_size 128 --train_batch_size 8 --sub_graph_batch_size 320 --use_map \
--core_num 10 --use_centerline \
--distributed_training 1 \
--other_params semantic_lane direction l1_loss \
goals_2D enhance_global_graph subdivide lazy_points new laneGCN point_sub_graph \
stage_one stage_one_dynamic=0.95 laneGCN-4 point_level point_level-4 \
point_level-4-3 complete_traj complete_traj-3 \

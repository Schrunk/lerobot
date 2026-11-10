# Gripper config:
# so101_parallel
# my_follower_arm

lerobot-calibrate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=my_follower_arm

lerobot-calibrate \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=my_leader_arm

lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=so101_parallel \
    --robot.cameras="{ top: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30} }" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=my_leader_arm \
    --display_data=true

# record data for training
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0   \
    --robot.id=my_follower_arm  \
    --robot.cameras="{ top: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30} }" \
    --teleop.type=so101_leader  \
    --teleop.port=/dev/ttyACM1  \
    --teleop.id=my_leader_arm   \
    --display_data=true         \
    --dataset.repo_id=Schrunk/so101_FMB_green_cylinder_in_assembly_board \
    --dataset.num_episodes=50    \
    --dataset.single_task="pick up green cylinder and place it in the assembly board" \
    --dataset.streaming_encoding=true  

# training
## on HF Hardware: --job.target=XXX ## hf jobs hardware
# smolVLA
## policy.path=lerobot/smolVLA_base
## rename map needed:     --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' \
# pi0
## --policy.type=pi0 \
## keine rename map weil Modell neu trainiert
## effizienteres/reduziertes Training
# --policy.pretrained_path=lerobot/pi0_base \
# --policy.compile_model=true \
# --policy.gradient_checkpointing=true \
# --policy.dtype=bfloat16 \
# --policy.freeze_vision_encoder=false \
# --policy.train_expert_only=false \
## relative actions
# --policy.use_relative_actions=true \
# --policy.relative_exclude_joints='["gripper"]' \

lerobot-train \
    --dataset.repo_id=/home/student/.cache/huggingface/lerobot/Schrunk/so101_FMB_merged \
    --job_name=FastWAM \
    --policy.device=cuda \
    --policy.repo_id=Schrunk/so101_smolVLA_FMB_18000 \
    --wandb.enable=true \
    --steps=18000 \
    --batch_size=32 \
    --policy.path=lerobot/smolVLA_base \
    --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' \
    --num_workers=24 

lerobot-train \
  --config_path=/home/student/lerobot_franz/outputs/train/2026-11-05/16-26-37_pi05_libero_expert/checkpoints/010000/pretrained_model \
  --resume=true \
  --steps=67000 \
  --batch_size=4 \
  --policy.repo_id=Schrunk/so101_Pi05_FMB_18000 

## --policy.type=pi05
lerobot-train \
    --dataset.repo_id=/home/student/.cache/huggingface/lerobot/Schrunk/so101_FMB_merged \
    --policy.type=pi05 \
    --policy.pretrained_path=/home/student/lerobot_franz/outputs/train/2026-11-05/16-26-37_pi05_libero_expert/checkpoints/010000/pretrained_model \
    --policy.normalization_mapping='{"ACTION": "MEAN_STD", "STATE": "MEAN_STD", "VISUAL": "IDENTITY"}' \
    --policy.n_action_steps=10 \
    --policy.empty_cameras=1 \
    --policy.freeze_vision_encoder=true \
    --policy.train_expert_only=true \
    --policy.gradient_checkpointing=true \
    --policy.dtype=bfloat16 \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --job_name=pi05_libero_expert \
    --batch_size=4 \
    --num_workers=24 \
    --steps=72000 \
    --save_freq=5000

## --polcy.type=fastwam
lerobot-train \
  --dataset.repo_id=/home/student/.cache/huggingface/lerobot/Schrunk/so101_FMB_merged \
  --policy.type=fastwam \
  --policy.action_dim=6 \
  --policy.proprio_dim=6 \
  --policy.action_horizon=32 \
  --policy.n_action_steps=10 \
  --policy.image_size='[480,1280]' \
  --policy.repo_id=Schrunk/so101_fastwam_FMB_18000 \
  --policy.freeze_language_encoder=true \
  --job_name=fastwam_training \
  --steps=18000 \
  --batch_size=1 \
  --policy.device=cuda

## --policy.typed=multi_task_dit
lerobot-train \
  --dataset.repo_id=/home/student/.cache/huggingface/lerobot/Schrunk/so101_FMB_merged \
  --batch_size=32 \
  --steps=18000 \
  --save_freq=2000 \
  --log_freq=100 \
  --policy.type=multi_task_dit \
  --policy.device=cuda \
  --policy.repo_id="Schrunk/so101_multi_task_dit_FMB_merged" \
  --wandb.enable=true

## --policy.type=vla_jepa
lerobot-train \
  --policy.path=lerobot/VLA-JEPA-LIBERO \
  --policy.repo_id="Schrunk/so101_vla_jepa_FMB_merged" \
  --dataset.repo_id=/home/student/.cache/huggingface/lerobot/Schrunk/so101_FMB_merged \
  --policy.freeze_qwen=true \
  --policy.reinit_modules='["model.action_model.action_encoder", "model.action_model.action_decoder", "model.action_model.state_encoder"]' \
  --rename_map='{"observation.images.top": "observation.images.image", "observation.images.wrist": "observation.images.image2"}' \
  --batch_size=16 \
  --steps=36000 


watch -n 0.5 nvidia-smi

uv run python /home/student/Desktop/MA_Zuarbeit/02_Code/estimate_training_time.py

# edit dataset
lerobot-edit-dataset \
    --new_repo_id=Schrunk/so101_FMB_merged \
    --operation.type=merge \
    --operation.repo_ids="['Schrunk/so101_FMB_green_cylinder_in_assembly_board_20261022_162416', 'Schrunk/so101_FMB_yellow_cuboid_left_from_assembly_board_20261021_160253_old']"

lerobot-edit-dataset \
    --repo_id=/home/student/.cache/huggingface/lerobot/Schrunk/so101_FMB_yellow_cuboid_left_from_assembly_board_20261021_160253_old \
    --operation.type=delete_episodes \
    --operation.episode_indices="[31]"

# recompute stats for dataset e.g. absolute in relative actions
lerobot-edit-dataset \
    --repo_id=/home/student/lerobot_franz/outputs/train/2026-11-05/16-26-37_pi05_libero_expert/checkpoints/005000/pretrained_model \
    --operation.type recompute_stats \
    --operation.relative_action true \
    --operation.chunk_size 50 \
    --operation.relative_exclude_joints "['gripper']" \
    --push_to_hub true

python record_gui.py \
  --repo-id user/pick_place \
  --task "Pick up the cube and place it in the box" \
  --robot-port /dev/ttyACM0 --robot-id my_follower \
  --teleop-port /dev/ttyACM1 --teleop-id my_leader \
  --camera front=0 --camera wrist=2 --camera side=/dev/video4 \
  --fps 30

# inference
## RTC:
# --inference.type=rtc \
# --inference.queue_threshold=5
# --inference.rtc.execution_horizon=20 \
# --inference.rtc.max_guidance_weight=5.0 \

lerobot-rollout \
    --strategy.type=base \
    --policy.path=lerobot/pi05_libero_base \
    --robot.type=so101_follower \
    --robot.id=so101_parallel \
    --robot.port=/dev/ttyACM1 \
    --robot.cameras="{ image: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, image2: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30} }" \
    --task="pick up the yellow block" \
    --duration=180 \
    --inference.type=rtc \
    --inference.queue_threshold=5 \
    --inference.rtc.execution_horizon=20 \
    --inference.rtc.max_guidance_weight=5.0 \
    --policy.dtype=bfloat16 \
    --device=cuda


# async inference server and client example
python -m lerobot.async_inference.policy_server \
     --host=127.0.0.1 \
     --port=8080

# aggregate_fn_name: average, weighted_average, conservative, new
python -m lerobot.async_inference.robot_client \
    --server_address=127.0.0.1:8080 \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=my_follower_arm \
    --robot.cameras="{ camera1: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, camera2: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30} }" \
    --task="Pick up the green cylinder and place it in the asssembly board" \
    --policy_type=smolvla \
    --pretrained_name_or_path=/home/student/lerobot_franz/outputs/train/13-38-37_smolVLA_FMB/checkpoints/018000/pretrained_model \
    --policy_device=cuda \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=weighted_average \
    --debug_visualize_queue_size=True





    [-h] [--config_path str] [--dataset str] [--dataset.repo_id str]
                     [--dataset.repo_type str] [--dataset.root str]
                     [--dataset.episodes str] [--dataset.exclude_episodes str]
                     [--image_transforms str] [--dataset.image_transforms str]
                     [--dataset.image_transforms.enable str]
                     [--dataset.image_transforms.max_num_transforms str]
                     [--dataset.image_transforms.random_order str]
                     [--dataset.image_transforms.tfs str] [--dataset.revision str]
                     [--dataset.use_imagenet_stats str] [--dataset.video_backend str]
                     [--dataset.return_uint8 str] [--dataset.depth_output_unit str]
                     [--dataset.streaming str] [--dataset.eval_split str] [--env str]
                     [--env.type {aloha,pusht,gym_manipulator,libero,metaworld,robocasa,vlabench,isaaclab_arena,unitree_g1_mujoco,libero_plus,robotwin,robomme}]
                     [--robot str] [--env.robot.type str] [--teleop str]
                     [--env.teleop str] [--env.teleop.type str] [--processor str]
                     [--env.processor str] [--env.processor.control_mode str]
                     [--observation str] [--env.processor.observation str]
                     [--env.processor.observation.add_joint_velocity_to_observation str]
                     [--env.processor.observation.add_current_to_observation str]
                     [--env.processor.observation.add_ee_pose_to_observation str]
                     [--env.processor.observation.display_cameras str]
                     [--image_preprocessing str]
                     [--env.processor.image_preprocessing str]
                     [--env.processor.image_preprocessing.crop_params_dict str]
                     [--env.processor.image_preprocessing.resize_size str]
                     [--gripper str] [--env.processor.gripper str]
                     [--env.processor.gripper.use_gripper str]
                     [--env.processor.gripper.gripper_penalty str] [--reset str]
                     [--env.processor.reset str]
                     [--env.processor.reset.fixed_reset_joint_positions str]
                     [--env.processor.reset.reset_time_s str]
                     [--env.processor.reset.control_time_s str]
                     [--env.processor.reset.terminate_on_success str]
                     [--inverse_kinematics str] [--env.processor.inverse_kinematics str]
                     [--env.processor.inverse_kinematics.urdf_path str]
                     [--env.processor.inverse_kinematics.target_frame_name str]
                     [--env.processor.inverse_kinematics.end_effector_bounds str]
                     [--env.processor.inverse_kinematics.end_effector_step_sizes str]
                     [--reward_classifier str] [--env.processor.reward_classifier str]
                     [--env.processor.reward_classifier.pretrained_path str]
                     [--env.processor.reward_classifier.success_threshold str]
                     [--env.processor.reward_classifier.success_reward str]
                     [--env.processor.max_gripper_pos str] [--env.name str]
                     [--env.multitask_eval str] [--env.visualization_height str]
                     [--env.visualization_width str] [--env.split str]
                     [--env.obj_registries str] [--env.render_resolution str]
                     [--env.robot str] [--env.num_envs str] [--env.embodiment str]
                     [--env.object str] [--env.mimic str] [--env.teleop_device str]
                     [--env.seed str] [--env.device str] [--env.disable_fabric str]
                     [--env.enable_cameras str] [--env.headless str]
                     [--env.enable_pinocchio str] [--env.environment str]
                     [--env.state_dim str] [--env.action_dim str]
                     [--env.camera_height str] [--env.camera_width str] [--env.video str]
                     [--env.video_length str] [--env.video_interval str]
                     [--env.state_keys str] [--env.camera_keys str] [--env.kwargs str]
                     [--env.hub_path str] [--env.end_effector str]
                     [--env.publish_images str] [--env.camera_port str]
                     [--env.onscreen str] [--env.camera_name str] [--env.init_states str]
                     [--env.hard_reset str] [--env.camera_name_mapping str]
                     [--env.is_libero_plus str] [--env.control_mode str]
                     [--env.obs_type str] [--env.render_mode str]
                     [--env.camera_names str] [--env.observation_height str]
                     [--env.observation_width str] [--env.action_mode str]
                     [--env.task str] [--env.fps str] [--env.features str]
                     [--env.features_map str] [--env.max_parallel_tasks str]
                     [--env.disable_env_checker str] [--env.episode_length str]
                     [--env.action_space str] [--env.dataset_split str]
                     [--env.task_ids str] [--env.front_camera_name str]
                     [--env.wrist_camera_name str] [--policy str]
                     [--policy.type {act,diffusion,eo1,evo1,fastwam,flux3,gaussian_actor,groot,lawam,lingbot_va,molmoact2,multi_task_dit,pi0,pi0_fast,pi05,smolvla,tdmpc,vla_jepa,vqbet,wall_x,xvla}]
                     [--policy.replace_final_stride_with_dilation str]
                     [--policy.pre_norm str] [--policy.dim_model str]
                     [--policy.n_heads str] [--policy.dim_feedforward str]
                     [--policy.feedforward_activation str]
                     [--policy.n_encoder_layers str] [--policy.n_decoder_layers str]
                     [--policy.use_vae str] [--policy.n_vae_encoder_layers str]
                     [--policy.temporal_ensemble_coeff str] [--policy.kl_weight str]
                     [--policy.optimizer_lr_backbone str] [--policy.resize_shape str]
                     [--policy.crop_ratio str] [--policy.down_dims str]
                     [--policy.kernel_size str] [--policy.n_groups str]
                     [--policy.diffusion_step_embed_dim str]
                     [--policy.use_film_scale_modulation str] [--policy.vlm_base str]
                     [--policy.vlm_config str] [--policy.image_min_pixels str]
                     [--policy.image_max_pixels str] [--policy.use_fast_processor str]
                     [--policy.num_denoise_steps str] [--policy.num_action_layers str]
                     [--policy.action_act str]
                     [--policy.supervise_padding_action_dims str]
                     [--policy.supervise_padding_actions str]
                     [--policy.force_fp32_autocast str] [--policy.training_stage str]
                     [--policy.max_views str] [--policy.postprocess_action_dim str]
                     [--policy.binarize_gripper str] [--policy.gripper_index str]
                     [--policy.gripper_below_threshold_value str]
                     [--policy.gripper_above_threshold_value str]
                     [--policy.vlm_num_layers str] [--policy.vlm_dtype str]
                     [--policy.max_text_length str] [--policy.use_flash_attn str]
                     [--policy.action_head str] [--policy.embed_dim str]
                     [--policy.state_hidden_dim str] [--policy.num_categories str]
                     [--policy.return_cls_only str]
                     [--policy.enable_gradient_checkpointing str]
                     [--policy.gradient_checkpointing_use_reentrant str]
                     [--policy.finetune_vlm str] [--policy.finetune_language_model str]
                     [--policy.finetune_vision_model str]
                     [--policy.finetune_action_head str]
                     [--policy.apply_training_stage_defaults str]
                     [--policy.task_field str] [--policy.embodiment_id_field str]
                     [--policy.default_embodiment_id str] [--policy.proprio_dim str]
                     [--policy.action_video_freq_ratio str] [--policy.context_len str]
                     [--policy.model_id str] [--policy.tokenizer_model_id str]
                     [--policy.text_encoder_model_id str] [--policy.base_model_id str]
                     [--policy.tokenizer_max_len str] [--policy.load_text_encoder str]
                     [--policy.mot_checkpoint_mixed_attn str] [--policy.rand_device str]
                     [--policy.text_cfg_scale str] [--policy.negative_prompt str]
                     [--policy.sigma_shift str] [--policy.tiled str]
                     [--policy.fp32_attention str] [--policy.compile_action_infer str]
                     [--policy.use_gradient_checkpointing str]
                     [--policy.freeze_video_expert str]
                     [--policy.toggle_action_dimensions str]
                     [--policy.video_scheduler str] [--policy.action_scheduler str]
                     [--policy.loss str] [--policy.video_dit_config str]
                     [--policy.action_dit_config str] [--policy.conditioning str]
                     [--policy.packer str] [--policy.condition_on_past_actions str]
                     [--policy.action_representation str]
                     [--policy.delta_absolute_dims str] [--policy.text_fixed_length str]
                     [--policy.video_position_fps str] [--policy.history_snapshots str]
                     [--policy.normalization_stats str] [--policy.normalization_clip str]
                     [--policy.separate_timesteps str] [--policy.video_logit_mean str]
                     [--policy.video_logit_std str] [--policy.conditioning_noise_max str]
                     [--policy.loss_reduction str] [--policy.action_channel_weights str]
                     [--policy.fps str] [--policy.camera_keys str]
                     [--policy.canvas_hw str] [--policy.action_modality str]
                     [--policy.action_scale str] [--policy.gripper_flip_dims str]
                     [--policy.trunk_weights str] [--policy.video_vae_id str]
                     [--policy.text_encoder_id str] [--policy.dit_config str]
                     [--policy.sampler str] [--policy.guidance_scale_action str]
                     [--policy.sampler_shift str] [--policy.inference_seed str]
                     [--policy.head_init_seed str] [--policy.train_timestep_width str]
                     [--policy.train_timestep_shift str]
                     [--policy.action_loss_weight str] [--policy.video_loss_weight str]
                     [--policy.caption_dropout str] [--policy.augment str]
                     [--policy.optimizer_lr_heads_multiplier str]
                     [--policy.scheduler_freeze_backbone_steps str]
                     [--policy.scheduler_warmup_steps_heads str]
                     [--policy.scheduler_cooldown_steps str] [--policy.dataset_stats str]
                     [--policy.storage_device str] [--policy.shared_encoder str]
                     [--policy.num_discrete_actions str]
                     [--policy.image_embedding_pooling_dim str]
                     [--policy.online_steps str] [--policy.online_buffer_capacity str]
                     [--policy.offline_buffer_capacity str] [--policy.async_prefetch str]
                     [--policy.online_step_before_learning str]
                     [--actor_learner_config str] [--policy.actor_learner_config str]
                     [--policy.actor_learner_config.learner_host str]
                     [--policy.actor_learner_config.learner_port str]
                     [--policy.actor_learner_config.policy_parameters_push_frequency str]
                     [--policy.actor_learner_config.queue_get_timeout str]
                     [--concurrency str] [--policy.concurrency str]
                     [--policy.concurrency.actor str] [--policy.concurrency.learner str]
                     [--policy.concurrency.multiprocessing_context str]
                     [--actor_network_kwargs str] [--policy.actor_network_kwargs str]
                     [--policy.actor_network_kwargs.hidden_dims str]
                     [--policy.actor_network_kwargs.activate_final str]
                     [--policy_kwargs str] [--policy.policy_kwargs str]
                     [--policy.policy_kwargs.use_tanh_squash str]
                     [--policy.policy_kwargs.std_min str]
                     [--policy.policy_kwargs.std_max str]
                     [--policy.policy_kwargs.init_final str]
                     [--discrete_critic_network_kwargs str]
                     [--policy.discrete_critic_network_kwargs str]
                     [--policy.discrete_critic_network_kwargs.hidden_dims str]
                     [--policy.discrete_critic_network_kwargs.activate_final str]
                     [--policy.discrete_critic_network_kwargs.final_activation str]
                     [--policy.base_model_path str]
                     [--policy.action_decode_transform str] [--policy.embodiment_tag str]
                     [--policy.tune_llm str] [--policy.tune_visual str]
                     [--policy.tune_projector str] [--policy.tune_diffusion_model str]
                     [--policy.tune_vlln str] [--policy.tune_top_llm_layers str]
                     [--policy.rtc_ramp_rate str] [--policy.use_flash_attention str]
                     [--policy.warmup_ratio str] [--policy.use_bf16 str]
                     [--policy.model_params_fp32 str] [--policy.image_size str]
                     [--policy.tokenizer_assets_repo str] [--policy.lora_full_model str]
                     [--policy.video_backend str] [--policy.balance_dataset_weights str]
                     [--policy.balance_trajectory_weights str]
                     [--policy.dataset_paths str] [--policy.output_dir str]
                     [--policy.save_steps str] [--policy.max_steps str]
                     [--policy.batch_size str] [--policy.dataloader_num_workers str]
                     [--policy.report_to str] [--policy.resume str]
                     [--policy.action_horizon str] [--policy.base_vlm str]
                     [--policy.lam_dim str] [--policy.lam_num_heads str]
                     [--policy.lam_ffn_expansion_factor str]
                     [--policy.lam_enc_layers str] [--policy.lam_code_dim str]
                     [--policy.lam_max_state_dim str] [--policy.lam_num_queries str]
                     [--policy.lam_dec_layers str] [--policy.lam_dropout str]
                     [--policy.lam_vq_layer_norm str] [--policy.lam_norm_latents str]
                     [--policy.lam_norm_latents_type str]
                     [--policy.lam_enc_modal_mask str]
                     [--policy.lam_latent_layer_to_use str]
                     [--policy.lam_num_embodiments str] [--policy.lam_image_hw str]
                     [--policy.lam_patch_size str] [--policy.lam_decoder_last_ln str]
                     [--policy.dinov3_hidden_size str]
                     [--policy.dinov3_intermediate_size str]
                     [--policy.dinov3_num_hidden_layers str]
                     [--policy.dinov3_num_attention_heads str]
                     [--policy.dinov3_num_register_tokens str]
                     [--policy.primary_image_features str]
                     [--policy.wrist_image_features str] [--policy.lam_image_feature str]
                     [--policy.default_task str] [--policy.embodiment_id str]
                     [--policy.latent_action_placeholder_token str]
                     [--policy.num_action_queries str]
                     [--policy.flow_action_num_queries str]
                     [--policy.perceptual_weight str]
                     [--policy.lam_encoder_distill_weight str]
                     [--policy.enable_loss_distill str] [--policy.future_prediction str]
                     [--policy.detach_future_feature str] [--policy.flow_action_dim str]
                     [--policy.flow_state_dim str] [--policy.flow_hidden_dim str]
                     [--policy.flow_num_layers str] [--policy.flow_attention_heads str]
                     [--policy.flow_vlm_dim str] [--policy.flow_vision_dim str]
                     [--policy.flow_num_vision_tokens str]
                     [--policy.flow_num_target_vision_tokens str]
                     [--policy.flow_use_state str] [--policy.flow_num_embodiments str]
                     [--policy.flow_cfg_drop_prob str]
                     [--policy.flow_cfg_guidance_scale str]
                     [--policy.flow_num_inference_steps str]
                     [--policy.flow_num_timestep_buckets str]
                     [--policy.flow_interleave_self_attention str]
                     [--policy.flow_use_alternate_vldit str]
                     [--policy.flow_attend_text_every_n_blocks str]
                     [--policy.flow_noise_beta_alpha str]
                     [--policy.flow_noise_beta_beta str] [--policy.flow_noise_s str]
                     [--policy.flow_token_independent_noise str]
                     [--policy.flow_use_action_positional_embeddings str]
                     [--policy.freeze_vision_backbone str]
                     [--policy.freeze_llm_backbone str]
                     [--policy.unfreeze_vision_merger str]
                     [--policy.unfreeze_lam_decoder str]
                     [--policy.keep_llm_first_n_layers str]
                     [--policy.unfreeze_llm_last_n_layers str] [--policy.patch_size str]
                     [--policy.num_attention_heads str] [--policy.attention_head_dim str]
                     [--policy.in_channels str] [--policy.out_channels str]
                     [--policy.text_dim str] [--policy.freq_dim str]
                     [--policy.ffn_dim str] [--policy.cross_attn_norm str]
                     [--policy.eps str] [--policy.rope_max_seq_len str]
                     [--policy.attn_mode str] [--policy.wan_pretrained_path str]
                     [--policy.text_encoder_device str] [--policy.obs_cam_keys str]
                     [--policy.image_hflip str] [--policy.camera_layout str]
                     [--policy.height str] [--policy.width str]
                     [--policy.action_per_frame str] [--policy.frame_chunk_size str]
                     [--policy.attn_window str] [--policy.video_exec_step str]
                     [--policy.action_num_inference_steps str]
                     [--policy.guidance_scale str] [--policy.action_guidance_scale str]
                     [--policy.snr_shift str] [--policy.action_snr_shift str]
                     [--policy.used_action_channel_ids str]
                     [--policy.save_predicted_video str] [--policy.checkpoint_path str]
                     [--policy.checkpoint_revision str]
                     [--policy.checkpoint_force_download str]
                     [--policy.inference_action_mode str]
                     [--policy.discrete_action_tokenizer str]
                     [--policy.discrete_generation_max_steps str] [--policy.norm_tag str]
                     [--policy.norm_stats_path str] [--policy.setup_type str]
                     [--policy.control_mode str] [--policy.image_keys str]
                     [--policy.normalize_language str] [--policy.add_setup_tokens str]
                     [--policy.add_control_tokens str] [--policy.normalize_gripper str]
                     [--policy.num_state_tokens str] [--policy.max_sequence_length str]
                     [--policy.expected_max_action_dim str]
                     [--policy.num_flow_timesteps str]
                     [--policy.flow_matching_cutoff str]
                     [--policy.flow_matching_time_offset str]
                     [--policy.flow_matching_time_scale str]
                     [--policy.flow_matching_beta_alpha str]
                     [--policy.flow_matching_beta_beta str]
                     [--policy.mask_action_dim_padding str]
                     [--policy.enable_inference_cuda_graph str]
                     [--policy.per_episode_seed str] [--policy.eval_seed str]
                     [--policy.joint_signs str] [--policy.joint_offsets str]
                     [--policy.train_mode_vlm str] [--policy.lora_rank str]
                     [--policy.lora_alpha str] [--policy.lora_dropout str]
                     [--policy.lora_bias str] [--policy.enable_knowledge_insulation str]
                     [--policy.freeze_embedding str] [--policy.llm_residual_dropout str]
                     [--policy.softmax_auxiliary_loss str]
                     [--policy.softmax_auxiliary_loss_scale str]
                     [--policy.discrete_loss_token_weighting str]
                     [--policy.optimizer_vit_lr str]
                     [--policy.optimizer_connector_lr str]
                     [--policy.optimizer_action_expert_lr str]
                     [--policy.dataset_feature_names str] [--policy.objective str]
                     [--policy.noise_scheduler_type str]
                     [--policy.num_train_timesteps str] [--policy.beta_schedule str]
                     [--policy.beta_start str] [--policy.beta_end str]
                     [--policy.prediction_type str] [--policy.clip_sample str]
                     [--policy.clip_sample_range str] [--policy.sigma_min str]
                     [--policy.num_integration_steps str]
                     [--policy.integration_method str]
                     [--policy.timestep_sampling_strategy str]
                     [--policy.timestep_sampling_s str]
                     [--policy.timestep_sampling_alpha str]
                     [--policy.timestep_sampling_beta str] [--policy.hidden_dim str]
                     [--policy.num_layers str] [--policy.use_positional_encoding str]
                     [--policy.timestep_embed_dim str] [--policy.use_rope str]
                     [--policy.rope_base str] [--policy.vision_encoder_name str]
                     [--policy.use_separate_rgb_encoder_per_camera str]
                     [--policy.vision_encoder_lr_multiplier str]
                     [--policy.image_resize_shape str] [--policy.image_crop_shape str]
                     [--policy.image_crop_is_random str] [--policy.text_encoder_name str]
                     [--policy.tokenizer_padding str] [--policy.tokenizer_truncation str]
                     [--policy.scheduler_name str]
                     [--policy.do_mask_loss_for_padding str]
                     [--policy.drop_n_last_frames str] [--policy.max_action_tokens str]
                     [--policy.action_tokenizer_name str] [--policy.temperature str]
                     [--policy.max_decoding_steps str] [--policy.fast_skip_tokens str]
                     [--policy.validate_action_token_prefix str]
                     [--policy.use_kv_cache str] [--policy.paligemma_variant str]
                     [--policy.action_expert_variant str]
                     [--policy.use_visual_memory str]
                     [--policy.use_proprioceptive_memory str]
                     [--policy.memory_frames str] [--policy.memory_stride str]
                     [--policy.memory_temporal_attention_every str]
                     [--policy.num_inference_steps str]
                     [--policy.time_sampling_beta_alpha str]
                     [--policy.time_sampling_beta_beta str]
                     [--policy.time_sampling_scale str]
                     [--policy.time_sampling_offset str]
                     [--policy.rtc_training_max_delay str]
                     [--policy.image_resolution str] [--policy.text_tokenizer_name str]
                     [--policy.gradient_checkpointing str]
                     [--policy.adapt_to_pi_aloha str]
                     [--policy.use_delta_joint_actions_aloha str]
                     [--policy.num_steps str] [--policy.use_cache str]
                     [--policy.train_expert_only str] [--policy.train_state_proj str]
                     [--policy.vlm_model_name str] [--policy.load_vlm_weights str]
                     [--policy.add_image_special_tokens str]
                     [--policy.attention_mode str] [--policy.prefix_length str]
                     [--policy.num_expert_layers str] [--policy.num_vlm_layers str]
                     [--policy.self_attn_every_n_layers str]
                     [--policy.expert_width_multiplier str] [--policy.min_period str]
                     [--policy.max_period str] [--rtc_config str]
                     [--policy.rtc_config str] [--policy.rtc_config.enabled str]
                     [--policy.rtc_config.mode str]
                     [--policy.rtc_config.prefix_attention_schedule str]
                     [--policy.rtc_config.max_guidance_weight str]
                     [--policy.rtc_config.execution_horizon str]
                     [--policy.rtc_config.debug str]
                     [--policy.rtc_config.debug_maxlen str] [--policy.compile_model str]
                     [--policy.compile_mode str] [--policy.n_action_repeats str]
                     [--policy.horizon str] [--policy.image_encoder_hidden_dim str]
                     [--policy.state_encoder_hidden_dim str] [--policy.latent_dim str]
                     [--policy.q_ensemble_size str] [--policy.mlp_dim str]
                     [--policy.discount str] [--policy.use_mpc str]
                     [--policy.cem_iterations str] [--policy.max_std str]
                     [--policy.min_std str] [--policy.n_gaussian_samples str]
                     [--policy.n_pi_samples str]
                     [--policy.uncertainty_regularizer_coeff str] [--policy.n_elites str]
                     [--policy.elite_weighting_temperature str]
                     [--policy.gaussian_mean_momentum str]
                     [--policy.max_random_shift_ratio str] [--policy.reward_coeff str]
                     [--policy.expectile_weight str] [--policy.value_coeff str]
                     [--policy.consistency_coeff str] [--policy.advantage_scaling str]
                     [--policy.pi_coeff str] [--policy.temporal_decay_coeff str]
                     [--policy.target_model_momentum str] [--policy.qwen_model_name str]
                     [--policy.jepa_encoder_name str] [--policy.freeze_qwen str]
                     [--policy.enable_world_model str] [--policy.reinit_modules str]
                     [--policy.prompt_template str] [--policy.special_action_token str]
                     [--policy.embodied_action_token str] [--policy.action_dim str]
                     [--policy.state_dim str] [--policy.use_relative_actions str]
                     [--policy.relative_exclude_joints str]
                     [--policy.action_feature_names str]
                     [--policy.num_action_tokens_per_timestep str]
                     [--policy.num_embodied_action_tokens_per_instruction str]
                     [--policy.num_inference_timesteps str]
                     [--policy.action_hidden_size str] [--policy.action_model_type str]
                     [--policy.action_num_layers str] [--policy.action_num_heads str]
                     [--policy.action_attention_head_dim str]
                     [--policy.action_dropout str]
                     [--policy.action_num_timestep_buckets str]
                     [--policy.action_noise_beta_alpha str]
                     [--policy.action_noise_beta_beta str] [--policy.action_noise_s str]
                     [--policy.action_max_seq_len str]
                     [--policy.num_target_vision_tokens str]
                     [--policy.num_video_frames str] [--policy.predictor_depth str]
                     [--policy.predictor_num_heads str]
                     [--policy.predictor_mlp_ratio str] [--policy.predictor_dropout str]
                     [--policy.world_model_loss_weight str]
                     [--policy.jepa_tubelet_size str]
                     [--policy.world_model_num_views str]
                     [--policy.repeated_diffusion_steps str]
                     [--policy.causal_world_model_context str]
                     [--policy.resize_images_to str]
                     [--policy.binarize_gripper_action str]
                     [--policy.pre_snap_gripper_action str]
                     [--policy.clip_normalized_actions str] [--policy.gripper_dim str]
                     [--policy.gripper_threshold str] [--policy.gripper_joint_names str]
                     [--policy.torch_dtype str] [--policy.n_action_pred_token str]
                     [--policy.action_chunk_size str] [--policy.vision_backbone str]
                     [--policy.crop_shape str] [--policy.crop_is_random str]
                     [--policy.pretrained_backbone_weights str]
                     [--policy.use_group_norm str]
                     [--policy.spatial_softmax_num_keypoints str]
                     [--policy.n_vqvae_training_steps str] [--policy.vqvae_n_embed str]
                     [--policy.vqvae_embedding_dim str]
                     [--policy.vqvae_enc_hidden_dim str] [--policy.gpt_block_size str]
                     [--policy.gpt_input_dim str] [--policy.gpt_output_dim str]
                     [--policy.gpt_n_layer str] [--policy.gpt_n_head str]
                     [--policy.gpt_hidden_dim str] [--policy.dropout str]
                     [--policy.offset_loss_weight str]
                     [--policy.primary_code_loss_weight str]
                     [--policy.secondary_code_loss_weight str]
                     [--policy.bet_softmax_temperature str]
                     [--policy.sequentially_select str] [--policy.optimizer_vqvae_lr str]
                     [--policy.optimizer_vqvae_weight_decay str]
                     [--policy.pretrained_name_or_path str]
                     [--policy.action_tokenizer_path str] [--policy.prediction_mode str]
                     [--policy.attn_implementation str]
                     [--policy.vision_attn_implementation str] [--policy.recipe_path str]
                     [--policy.recipe str] [--policy.text_temperature str]
                     [--policy.text_top_p str] [--policy.flow_loss_weight str]
                     [--policy.text_loss_weight str] [--policy.n_obs_steps str]
                     [--policy.input_features str] [--policy.output_features str]
                     [--policy.device str] [--policy.use_amp str] [--policy.use_peft str]
                     [--policy.push_to_hub str] [--policy.repo_id str]
                     [--policy.private str] [--policy.tags str] [--policy.license str]
                     [--policy.pretrained_path str] [--policy.pretrained_revision str]
                     [--policy.chunk_size str] [--policy.n_action_steps str]
                     [--policy.dtype str] [--policy.normalization_mapping str]
                     [--policy.florence_config str] [--policy.tokenizer_name str]
                     [--policy.tokenizer_max_length str]
                     [--policy.tokenizer_padding_side str] [--policy.pad_language_to str]
                     [--policy.hidden_size str] [--policy.depth str]
                     [--policy.num_heads str] [--policy.mlp_ratio str]
                     [--policy.num_domains str] [--policy.len_soft_prompts str]
                     [--policy.dim_time str] [--policy.max_len_seq str]
                     [--policy.use_hetero_proj str] [--policy.action_mode str]
                     [--policy.num_denoising_steps str] [--policy.use_proprio str]
                     [--policy.max_state_dim str] [--policy.max_action_dim str]
                     [--policy.domain_feature_key str]
                     [--policy.resize_imgs_with_padding str]
                     [--policy.num_image_views str] [--policy.empty_cameras str]
                     [--policy.freeze_vision_encoder str]
                     [--policy.freeze_language_encoder str]
                     [--policy.train_policy_transformer str]
                     [--policy.train_soft_prompts str] [--policy.optimizer_lr str]
                     [--policy.optimizer_betas str] [--policy.optimizer_eps str]
                     [--policy.optimizer_weight_decay str]
                     [--policy.optimizer_grad_clip_norm str]
                     [--policy.optimizer_soft_prompt_lr_scale str]
                     [--policy.optimizer_soft_prompt_warmup_lr_scale str]
                     [--policy.scheduler_warmup_steps str]
                     [--policy.scheduler_decay_steps str]
                     [--policy.scheduler_decay_lr str] [--reward_model str]
                     [--reward_model.type {reward_classifier,robometer,sarm,topreward}]
                     [--reward_model.name str] [--reward_model.num_classes str]
                     [--reward_model.latent_dim str]
                     [--reward_model.image_embedding_pooling_dim str]
                     [--reward_model.dropout_rate str] [--reward_model.model_name str]
                     [--reward_model.model_type str] [--reward_model.num_cameras str]
                     [--reward_model.learning_rate str] [--reward_model.weight_decay str]
                     [--reward_model.grad_clip_norm str]
                     [--reward_model.reward_output str]
                     [--reward_model.base_model_id str]
                     [--reward_model.use_multi_image str]
                     [--reward_model.use_per_frame_progress_token str]
                     [--reward_model.average_temporal_patches str]
                     [--reward_model.frame_pooling str]
                     [--reward_model.frame_pooling_attn_temperature str]
                     [--reward_model.progress_loss_type str]
                     [--reward_model.progress_discrete_bins str]
                     [--reward_model.vlm_config str] [--reward_model.annotation_mode str]
                     [--reward_model.n_obs_steps str] [--reward_model.frame_gap str]
                     [--reward_model.max_rewind_steps str] [--reward_model.image_dim str]
                     [--reward_model.text_dim str] [--reward_model.hidden_dim str]
                     [--reward_model.num_heads str] [--reward_model.num_layers str]
                     [--reward_model.max_state_dim str]
                     [--reward_model.drop_n_last_frames str]
                     [--reward_model.batch_size str] [--reward_model.clip_batch_size str]
                     [--reward_model.dropout str] [--reward_model.stage_loss_weight str]
                     [--reward_model.rewind_probability str]
                     [--reward_model.language_perturbation_probability str]
                     [--reward_model.num_sparse_stages str]
                     [--reward_model.sparse_subtask_names str]
                     [--reward_model.sparse_temporal_proportions str]
                     [--reward_model.num_dense_stages str]
                     [--reward_model.dense_subtask_names str]
                     [--reward_model.dense_temporal_proportions str]
                     [--reward_model.pretrained_model_path str]
                     [--reward_model.state_key str] [--reward_model.input_features str]
                     [--reward_model.output_features str] [--reward_model.device str]
                     [--reward_model.pretrained_path str]
                     [--reward_model.pretrained_revision str]
                     [--reward_model.push_to_hub str] [--reward_model.repo_id str]
                     [--reward_model.license str] [--reward_model.tags str]
                     [--reward_model.private str] [--reward_model.vlm_name str]
                     [--reward_model.torch_dtype str]
                     [--reward_model.attn_implementation str]
                     [--reward_model.image_key str] [--reward_model.task_key str]
                     [--reward_model.default_task str] [--reward_model.max_frames str]
                     [--reward_model.fps str] [--reward_model.prompt_prefix str]
                     [--reward_model.prompt_suffix_template str]
                     [--reward_model.add_chat_template str]
                     [--reward_model.success_threshold str]
                     [--reward_model.max_input_length str]
                     [--reward_model.normalization_mapping str] [--output_dir str]
                     [--job_name str] [--resume str] [--seed str]
                     [--cudnn_deterministic str] [--num_workers str] [--batch_size str]
                     [--prefetch_factor str] [--persistent_workers str]
                     [--dataloader_multiprocessing_context str] [--steps str]
                     [--env_eval_freq str] [--log_freq str] [--eval_steps str]
                     [--max_eval_samples str] [--tolerance_s str] [--save_checkpoint str]
                     [--save_freq str] [--checkpoint_format str]
                     [--use_policy_training_preset str] [--optimizer str]
                     [--optimizer.type {adam,adamw,sgd,xvla-adamw,multi_adam,molmoact2_adamw}]
                     [--optimizer.momentum str] [--optimizer.dampening str]
                     [--optimizer.nesterov str] [--optimizer.soft_prompt_lr_scale str]
                     [--optimizer.soft_prompt_warmup_lr_scale str]
                     [--optimizer.optimizer_groups str] [--optimizer.lr str]
                     [--optimizer.weight_decay str] [--optimizer.grad_clip_norm str]
                     [--optimizer.betas str] [--optimizer.eps str]
                     [--optimizer.group_grad_clip_norm str] [--scheduler str]
                     [--scheduler.type {diffuser,vqbet,constant_with_warmup,cosine_annealing_with_warmup,cosine_decay_with_warmup,frozen_warmup_constant,lawam_cosine_with_min_lr,molmoact2_cosine_with_warmup}]
                     [--scheduler.name str] [--scheduler.num_vqvae_training_steps str]
                     [--scheduler.num_cycles str] [--scheduler.freeze_steps str]
                     [--scheduler.warmup_steps_heads str] [--scheduler.decay_steps str]
                     [--scheduler.cooldown_steps str] [--scheduler.num_warmup_steps str]
                     [--scheduler.num_decay_steps str] [--scheduler.peak_lr str]
                     [--scheduler.decay_lr str] [--parallelism str]
                     [--parallelism.dp_replicate str] [--parallelism.dp_shard str]
                     [--context_parallel str] [--parallelism.context_parallel str]
                     [--parallelism.context_parallel.ring_degree str]
                     [--parallelism.context_parallel.ulysses_degree str]
                     [--parallelism.cfg_parallel str] [--accelerator str]
                     [--accelerator.mixed_precision str] [--gradient_accumulation str]
                     [--accelerator.gradient_accumulation str]
                     [--accelerator.gradient_accumulation.steps str] [--grad_scaler str]
                     [--accelerator.grad_scaler str]
                     [--accelerator.grad_scaler.init_scale str]
                     [--accelerator.grad_scaler.growth_factor str]
                     [--accelerator.grad_scaler.backoff_factor str]
                     [--accelerator.grad_scaler.growth_interval str] [--fsdp str]
                     [--accelerator.fsdp str]
                     [--accelerator.fsdp.reshard_after_forward str]
                     [--accelerator.fsdp.wrap_modules str]
                     [--accelerator.fsdp.min_num_params str]
                     [--accelerator.fsdp.cpu_offload str]
                     [--accelerator.fsdp.ignored_modules str] [--ddp str]
                     [--accelerator.ddp str]
                     [--accelerator.ddp.find_unused_parameters str]
                     [--accelerator.ddp.gradient_as_bucket_view str]
                     [--accelerator.ddp.static_graph str] [--compile str]
                     [--accelerator.compile str] [--accelerator.compile.enabled str]
                     [--accelerator.compile.backend str] [--accelerator.compile.mode str]
                     [--accelerator.compile.regional str]
                     [--activation_checkpointing str]
                     [--accelerator.activation_checkpointing str]
                     [--accelerator.activation_checkpointing.mode str] [--eval str]
                     [--eval.n_episodes str] [--eval.batch_size str]
                     [--eval.use_async_envs str] [--eval.recording str]
                     [--eval.recording_repo_id str] [--eval.recording_private str]
                     [--ema str] [--ema.enable str] [--ema.decay str]
                     [--ema.update_after_step str] [--ema.inv_gamma str]
                     [--ema.power str] [--ema.min_decay str] [--ema.max_decay str]
                     [--ema.use_for_eval str] [--wandb str] [--wandb.enable str]
                     [--wandb.disable_artifact str] [--wandb.project str]
                     [--wandb.entity str] [--wandb.notes str] [--wandb.run_id str]
                     [--wandb.resume str] [--wandb.mode str] [--wandb.console str]
                     [--wandb.console_multipart str]
                     [--wandb.console_chunk_max_seconds str] [--wandb.add_tags str]
                     [--peft str] [--peft.target_modules str]
                     [--peft.full_training_modules str] [--peft.method_type str]
                     [--peft.init_type str] [--peft.r str] [--peft.lora_alpha str]
                     [--job str] [--job.target str] [--job.image str] [--job.timeout str]
                     [--job.detach str] [--job.tags str] [--save_checkpoint_to_hub str]
                     [--sample_weighting str] [--sample_weighting.type str]
                     [--sample_weighting.progress_path str]
                     [--sample_weighting.head_mode str] [--sample_weighting.kappa str]
                     [--sample_weighting.epsilon str]
                     [--sample_weighting.extra_params str] [--rename_map str]


until lerobot-train \
  --config_path=/home/student/lerobot_franz/outputs/train/2026-11-05/16-26-37_pi05_libero_expert/checkpoints/015000/pretrained_model \
  --resume=true \
  --steps=67000 \
  --batch_size=4 \
  --policy.repo_id=Schrunk/so101_Pi05_FMB_18000 ; do echo "Abbruch (Exit-Code $?), Neustart in 10s..."; sleep 600; done
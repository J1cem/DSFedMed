'''

Foundation Model -> The initial teacher model is pretrained on the synthetic dataset using SAM before Mutual KD.
Lightweight Model -> The initial student model is the global model trained via FedAvg on the original dataset with TinySAM. See train_fed_sam_base_od.py for FL details.
The baseline model is the global model trained via FedAvg on the original dataset with SAM, which reflects the upper-bound performance.

'''
python ../train_fed_sam_kd_sam_sample_2.0.py \
--data Nuclei_od \
--gpu 1 \
--batch_size 4 \
--max_epoch 20 \
--exp nuclei-3-od-cn-kd-sam_2_sample_2.0 \
--base_lr 1e-4 \
--display_freq 200 \
--unseen_site 2 \
--tea_ckpt "/.../teacher_2.pth" \
--stu_ckpt "/.../student_2.pth" \
--baseline_ckpt "/.../baseline_2.pth"

python ../train_fed_sam_kd_sam_sample_2.0.py \
--data Nuclei_od \
--gpu 1 \
--batch_size 4 \
--max_epoch 20 \
--exp nuclei-3-od-cn-kd-sam_0_sample_2.0 \
--base_lr 1e-4 \
--display_freq 200 \
--unseen_site 0 \
--tea_ckpt "/.../teacher_0.pth" \
--stu_ckpt "/.../student_0.pth" \
--baseline_ckpt "/.../baseline_0.pth"

python ../train_fed_sam_kd_sam_sample_2.0.py \
--data Nuclei_od \
--gpu 1 \
--batch_size 4 \
--max_epoch 20 \
--exp nuclei-3-od-cn-kd-sam_1_sample_2.0 \
--base_lr 1e-4 \
--display_freq 200 \
--unseen_site 1 \
--tea_ckpt "/.../teacher_1.pth" \
--stu_ckpt "/.../student_1.pth" \
--baseline_ckpt "/.../baseline_1.pth"

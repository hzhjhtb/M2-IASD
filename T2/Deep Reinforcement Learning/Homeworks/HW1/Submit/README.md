**Question 3.1.1 Ant**

python cs285/scripts/run_hw1.py `
    --expert_policy_file cs285/policies/experts/Ant.pkl `
    --env_name Ant-v4 `
    --exp_name bc_ant `
    --n_iter 1 `
    --eval_batch_size 5000 ` 
    --expert_data cs285/expert_data/expert_data_Ant-v4.pkl `
    --video_log_freq -1

**Question 3.1.2 : Hopper (Try 1)**
$exp_name = "Hopper"
python cs285/scripts/run_hw1.py `
    --expert_policy_file "cs285/policies/experts/$exp_name.pkl" `
    --env_name "$exp_name-v4" `
    --exp_name "bc_$exp_name" `
    --eval_batch_size 5000 `
    --expert_data "cs285/expert_data/expert_data_$exp_name-v4.pkl" `
    --video_log_freq -1

**Question 3.1.2 : Hopper (Try 2)**
$exp_name = "Hopper"
python cs285/scripts/run_hw1.py `
    --expert_policy_file "cs285/policies/experts/$exp_name.pkl" `
    --env_name "$exp_name-v4" `
    --exp_name "bc_$exp_name" `
    --eval_batch_size 5000 `
    --num_agent_train_steps_per_iter 10000 `
    -lr 1e-3 `
    --expert_data "cs285/expert_data/expert_data_$exp_name-v4.pkl" `
    --video_log_freq -1

**Question 3.1.2 : HalfCheetah**
$exp_name = "HalfCheetah"
python cs285/scripts/run_hw1.py `
    --expert_policy_file "cs285/policies/experts/$exp_name.pkl" `
    --env_name "$exp_name-v4" `
    --exp_name "bc_$exp_name" `
    --eval_batch_size 5000 `
    --expert_data "cs285/expert_data/expert_data_$exp_name-v4.pkl" `
    --video_log_freq -1

**Question 3.1.2 : Walker2d (Try 1)**
$exp_name = "Walker2d"
python cs285/scripts/run_hw1.py `
    --expert_policy_file "cs285/policies/experts/$exp_name.pkl" `
    --env_name "$exp_name-v4" `
    --exp_name "bc_$exp_name" `
    --eval_batch_size 5000 `
    --expert_data "cs285/expert_data/expert_data_$exp_name-v4.pkl" `
    --video_log_freq -1

**Question 3.1.2 : Walker2d  (Try 2)**
$exp_name = "Walker2d"
python cs285/scripts/run_hw1.py `
    --expert_policy_file "cs285/policies/experts/$exp_name.pkl" `
    --env_name "$exp_name-v4" `
    --exp_name "bc_$exp_name" `
    --eval_batch_size 5000 `
    --num_agent_train_steps_per_iter 10000 `
    -lr 3e-3 `
    --expert_data "cs285/expert_data/expert_data_$exp_name-v4.pkl" `
    --video_log_freq -1

**Question 3.2**
$exp_name = "Hopper"

$nb_layers = @(2, 3, 4)
$learning_rates = @(1e-3, 2e-3, 3e-3, 4e-3)

foreach ($layers in $nb_layers) {
    foreach ($learning_rate in $learning_rates) {
        python cs285/scripts/run_hw1.py `
            --expert_policy_file cs285/policies/experts/$exp_name.pkl `
            --env_name $exp_name-v4 `
            --exp_name bc_$exp_name`_$layers`_$learning_rate `
            --eval_batch_size 5000 `
            --num_agent_train_steps_per_iter 10000 `
            -lr $learning_rate `
            --n_layers $layers `
            --expert_data cs285/expert_data/expert_data_$exp_name-v4.pkl `
            --video_log_freq -1
    }
}


**Question 4.1**
python cs285/scripts/run_hw1.py `
    --expert_policy_file cs285/policies/experts/Ant.pkl `
    --env_name Ant-v4 `
    --exp_name dagger_ant `
    --eval_batch_size 5000 `
    --n_iter 10 `
    --do_dagger `
    --expert_data cs285/expert_data/expert_data_Ant-v4.pkl `
    --video_log_freq -1

**Question 4.1 : Hopper (Try 1)**
$exp_name = "Hopper"
python cs285/scripts/run_hw1.py `
    --expert_policy_file "cs285/policies/experts/$exp_name.pkl" `
    --env_name "$exp_name-v4" `
    --exp_name "dagger$exp_name" `
    --eval_batch_size 100000 `
    --n_iter 10 `
    --do_dagger `
    --expert_data "cs285/expert_data/expert_data_$exp_name-v4.pkl" `
    --video_log_freq -1

**Question 4.1 : Hopper (Try 2)**
$exp_name = "Hopper"
python cs285/scripts/run_hw1.py `
    --expert_policy_file "cs285/policies/experts/$exp_name.pkl" `
    --env_name "$exp_name-v4" `
    --exp_name "dagger$exp_name" `
    --eval_batch_size 100000 `
    --n_iter 10 `
    --do_dagger `
    --num_agent_train_steps_per_iter 10000 `
    -lr 2e-3 `
    --expert_data "cs285/expert_data/expert_data_$exp_name-v4.pkl" `
    --video_log_freq -1

**Question 4.1 : HalfCheetah**
$exp_name = "HalfCheetah"
python cs285/scripts/run_hw1.py `
    --expert_policy_file "cs285/policies/experts/$exp_name.pkl" `
    --env_name "$exp_name-v4" `
    --exp_name "dagger_$exp_name" `
    --eval_batch_size 100000 `
    --n_iter 10 `
    --do_dagger `
    --expert_data "cs285/expert_data/expert_data_$exp_name-v4.pkl" `
    --video_log_freq -1

**Question 4.1 : Walker2d (Try 1)**
$exp_name = "Walker2d"
python cs285/scripts/run_hw1.py `
    --expert_policy_file "cs285/policies/experts/$exp_name.pkl" `
    --env_name "$exp_name-v4" `
    --exp_name "dagger_$exp_name" `
    --eval_batch_size 100000 `
    --n_iter 10 `
    --do_dagger `
    --expert_data "cs285/expert_data/expert_data_$exp_name-v4.pkl" `
    --video_log_freq -1

**Question 4.1 : Walker2d (Try 2)**
$exp_name = "Walker2d"
python cs285/scripts/run_hw1.py `
    --expert_policy_file "cs285/policies/experts/$exp_name.pkl" `
    --env_name "$exp_name-v4" `
    --exp_name "dagger_$exp_name" `
    --eval_batch_size 100000 `
    --n_iter 10 `
    --do_dagger `
    --num_agent_train_steps_per_iter 10000 `
    -lr 1e-3 `
    --expert_data "cs285/expert_data/expert_data_$exp_name-v4.pkl" `
    --video_log_freq -1

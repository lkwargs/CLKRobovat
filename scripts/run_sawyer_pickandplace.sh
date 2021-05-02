
python tools/run_env.py --env RepeatGraspEnv \
                        --policy RepeatedRandomGraspPolicy \
                        --env_config configs/envs/push_env.yaml \
                        --policy_config configs/policies/heuristic_push_policy.yaml
                        --debug 1 
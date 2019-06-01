import argparse
import pickle
import sys
sys.path.append('.')
import rlkit.torch.pytorch_util as ptu
ptu.set_gpu_mode(True)
from rlkit.core import logger
from rlkit.samplers.rollout_functions import multitask_rollout
from rlkit.torch import pytorch_util as ptu
from rlkit.core.flex_wrappers import FetchReach


def simulate_policy(args):
    data = pickle.load(open(args.file, "rb"))
    policy = data['evaluation/policy']
    env = FetchReach(1)
    print("Policy and environment loaded")
    if args.gpu:
        ptu.set_gpu_mode(True)
        policy.to(ptu.device)
    paths = []
    while True:
        paths.append(multitask_rollout(
            env,
            policy,
            max_path_length=args.H,
            observation_key='observation',
            desired_goal_key='desired_goal',
        ))
        if hasattr(env, "log_diagnostics"):
            env.log_diagnostics(paths)
        if hasattr(env, "get_diagnostics"):
            for k, v in env.get_diagnostics(paths).items():
                logger.record_tabular(k, v)
        logger.dump_tabular()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--H', type=int, default=300,
                        help='Max length of rollout')
    parser.add_argument('--speedup', type=float, default=10,
                        help='Speedup')
    parser.add_argument('--mode', default='video_env', type=str,
                        help='env mode')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--enable_render', action='store_true')
    parser.add_argument('--hide', action='store_true')
    args = parser.parse_args()

    simulate_policy(args)

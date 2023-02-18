from config import get_config
from envs.fdran_env import FdranEnv

if __name__ == '__main__':
    sce_idx = 1
    device = "cpu"
    parser = get_config()
    cfgs = parser.parse_known_args()[0]

    fd = FdranEnv(cfgs, sce_idx, device)

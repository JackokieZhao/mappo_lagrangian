from config import get_config
from envs.fd_ran.fdran import FDRAN

if __name__ == '__main__':
    sce_idx = 1
    device = "cpu"
    parser = get_config()
    cfgs = parser.parse_known_args()[0]

    fd = FDRAN(sce_idx, device, cfgs)

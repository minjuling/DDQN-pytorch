class Config(object):
    env = 'PongNoFrameskip-v4'
    cuda_num = 'cuda:2' # no cuda -> None
    lr = 1e-4
    discount_factor = 0.99
    memory_size = 100000
    frames = 2000000
    update_target_frequency = 1000
    epsilon = 1
    epsilon_min = 0.01
    eps_decay = 30000
    use_cuda = True
    save_logs_frequency = 5000
    show_loss_frequency = 5000
    batch_size = 32
    initial_observe_episode = 100
    logs_path = './logs'
    tensorboard_path = './tensorboard/'
    win_reward = 18
    win_break = True
    
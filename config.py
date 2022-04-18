class Config(object):
    lr = 1e-4
    discount_factor = 0.99
    memory_size = 100000
    total_episode = 2000000
    explore = 3000000
    update_target_frequency = 1000
    # epsilon_start = 1.0
    epsilon_start = 1
    epsilon_final = 0.01
    epsilon_decay = 30000
    # save_video_frequency = 500
    save_logs_frequency = 5000
    show_loss_frequency = 5000
    batch_size = 32
    initial_observe_episode = 100
    maximum_model = 5
    screen_width = 84
    screen_height = 84
    # Hyper Parameters
    memory_capacity = 10000
    logs_path = './logs'
    tensorboard_path = './tensorboard/'
    win_reward = 18
    win_break = True
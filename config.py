class Config(object):
    lr = 0.00025
    discount_factor = 0.99
    memory_size = 1000000
    total_episode = 1000000
    update_target_frequency = 10000
    # epsilon_start = 1.0
    epsilon_start = 0.95
    epsilon_final = 0.01
    epsilon_decay = 30000
    # save_video_frequency = 500
    save_logs_frequency = 10000
    show_loss_frequency = 1
    batch_size = 32
    initial_observe_episode = 100
    maximum_model = 5
    screen_width = 84
    screen_height = 84
    # Hyper Parameters
    memory_capacity = 50000
    logs_path = './logs'
    tensorboard_path = './tensorboard/'
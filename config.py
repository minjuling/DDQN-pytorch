class Config(object):
    lr = 0.0002
    discount_factor = 0.99
    memory_size = 1000000
    total_episode = 1000000
    explore = 3000000
    update_target_frequency = 5
    # epsilon_start = 1.0
    epsilon_start = 0.1
    epsilon_final = 0.05
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
class Config(object):
    lr = 0.01
    discount_factor = 0.99
    memory_size = 50000
    total_episode = 100000
    update_target_frequency = 100
    epsilon = 0.9
    gamma = 0.9                 # reward discount
    epsilon_discount_rate = 1e-7
    # save_video_frequency = 500
    save_logs_frequency = 10
    show_loss_frequency = 10
    batch_size = 32
    initial_observe_episode = 100
    maximum_model = 5
    screen_width = 84
    screen_height = 84
    # Hyper Parameters
    memory_capacity = 33
    logs_path = './logs'
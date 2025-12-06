Config= {

    "cartpole":{
        "env_id": "CartPole-v1",
        "human": False,
        "replay_buffer_size": 100000,
        "num_episode_train": 500,
        "num_episode_validation": 15,
        "hidden_size": 128,
        "batch_size": 64,
        "target_update_freq" : 700,
        "lr":0.0001,
        "epsilon_start":1.0,
        "epsilon_end":0.05,
        "epsilon_decay": 10000


    },
    "lunars":{
        "env_id": "LunarLander-v3",
        "human": False,
        "replay_buffer_size": 1000000,
        "num_episode_train":1000,
        "num_episode_validation": 10,
        "hidden_size": 512,
        "batch_size": 128,
        "target_update_freq" : 1700,
        "lr":0.0001,
        "epsilon_start":1.0,
        "epsilon_end":0.05,
        "epsilon_decay": 50000
    }


}
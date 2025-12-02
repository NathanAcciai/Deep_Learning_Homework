Config= {

    "cartpole":{
        "env_id": "CartPole-v1",
        "human": False,
        "replay_buffer_size": "50000",
        "num_episode_train": 500,
        "num_episode_validation": 10,
        "hidden_size": 128,


    },
    "lunars":{
        "env_id": "LunarLander-v3",
        "human": False,
        "replay_buffer_size": "100000",
        "num_episode_train":1000,
        "num_episode_validation": 10,
        "hidden_size": 256
    }


}
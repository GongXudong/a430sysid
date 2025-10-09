from a430py.env.a430_gym import A430Gym


class TestA430Gym:
    def test_init_1(self):
        print("In test init 1: ")
        env = A430Gym()
        obs, info = env.reset()
        print(obs)

    def test_step_1(self):
        print("In test step 1: ")
        env = A430Gym()
        obs, info = env.reset()

        # 用于配平的动作
        action = [0.0, -1.998228, 0.0, 0.689030]

        for i in range(60):
            next_obs, reward, terminated, truncated, info = env.step(action)
            print(f"next_obs = {next_obs}")

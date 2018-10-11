import numpy as np
from collections import deque
import gym
from gym import spaces
import cv2
cv2.ocl.setUseOpenCL(False)


class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        gym.ObservationWrapper.__init__(self, env)
        self.width = 84
        self.height = 84
        self.observation_space = spaces.Box(low=0, high=255,
            shape=(self.height, self.width, 1), dtype=np.uint8)

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame[:, :, None]


class SkipAndStackEnv(gym.Wrapper):
    def __init__(self, env, skip, stack):
        """Return only every `skip`-th frame"""
        assert skip >= stack
        super().__init__(env)
        self._obs_buffer = deque(maxlen=stack)
        self._skip = skip
        self._stack = stack

    def reset(self):
        """Repeat action, sum reward, and stack over last observations."""
        obs = self.env.reset()
        return self._skip_frame(0)[0]

    def step(self, action):
        """Repeat action, sum reward, and stack over last observations."""
        return self._skip_frame(action)

    def _skip_frame(self, action):
        total_reward = 0.0
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            self._obs_buffer.append(obs)
            if done:
                break
        return np.concatenate(self._obs_buffer, axis=2), total_reward, done, info


def make_atari(env_id, skip=4, stack=2):
    env = gym.make(env_id)
    assert 'NoFrameskip' in env.spec.id
    env = FireResetEnv(env)
    env = WarpFrame(env)
    env = SkipAndStackEnv(env, skip, stack)
    return env


class ImageToPyTorch(gym.ObservationWrapper):
    """
    Image shape to num_channels x weight x height
    """
    def __init__(self, env):
        super().__init__(env)
        stack = env._stack
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=255.0, shape=(stack, old_shape[0], old_shape[1]), dtype=np.uint8)

    def observation(self, observation):
        return observation.transpose(2,0,1)


def wrap_pytorch(env):
    return ImageToPyTorch(env)



if __name__=="__main__":
    # env = gym.make('PongNoFrameskip-v4')
    # env.reset()
    # for i in range(100):
    #     env.render()
    #     env.step(0)
    # env.close()

    env = make_atari('PongNoFrameskip-v4', stack=4)
    env = wrap_pytorch(env)
    state = env.reset()
    print(state.shape)
    for i in range(50):
        env.render()
        state, _, _, _ = env.step(3)
    import matplotlib.pyplot as plt
    ax1 = plt.subplot(221)
    ax1.imshow(state[0,:,:])
    ax2 = plt.subplot(222)
    ax2.imshow(state[1, :, :])
    ax3 = plt.subplot(223)
    ax3.imshow(state[2, :, :])
    ax4 = plt.subplot(224)
    ax4.imshow(state[3, :, :])
    plt.show()
    env.close()


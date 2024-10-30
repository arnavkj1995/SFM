import numpy as np
import typing as tp


class ReplayBuffer(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_size=1e6,
        batch_size=256,
        max_action=1,
        normalize_actions=True,
        ep_len=1000,
        future=.98,
    ):
        max_size = int(max_size)
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.ep_id = 0
        self._future = future

        self.batch_size = batch_size
        self.ep_len = ep_len

        self.num_ep = max_size // ep_len + 1

        self.state = np.zeros((self.num_ep, ep_len, state_dim))
        self.action = np.zeros((self.num_ep, ep_len, action_dim))
        self.next_state = np.zeros((self.num_ep, ep_len, state_dim))
        self.reward = np.zeros((self.num_ep, ep_len, 1))
        self.not_done = np.zeros((self.num_ep, ep_len, 1))

        self.normalize_actions = max_action if normalize_actions else 1

    def add(self, state, action, next_state, reward, done, term_trunc):
        self.state[self.ep_id, self.ptr] = state
        self.action[self.ep_id, self.ptr] = action / self.normalize_actions
        self.next_state[self.ep_id, self.ptr] = next_state
        self.reward[self.ep_id, self.ptr] = reward
        self.not_done[self.ep_id, self.ptr] = 1.0 - done

        self.ptr +=1
        if term_trunc:
            self.ep_id = (self.ep_id + 1) % self.num_ep
            self.ptr = 0

        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size=None):
        if not batch_size:
            batch_size = self.batch_size
        self.ind = np.random.randint(0, self.size, size=batch_size)

        ep_ind = self.ind // self.ep_len
        step_ind = self.ind % self.ep_len

        if self._future < 1:
            future_ind = step_ind + np.random.geometric(p=(1 - self._future), size=batch_size)
            future_ind = np.clip(future_ind, 0, self.ep_len - 1)

        return (
            np.array(self.state[ep_ind, step_ind], dtype=np.float32),
            np.array(self.action[ep_ind, step_ind], dtype=np.float32),
            np.array(self.next_state[ep_ind, step_ind], dtype=np.float32),
            np.array(self.reward[ep_ind, step_ind], dtype=np.float32),
            np.array(self.not_done[ep_ind, step_ind], dtype=np.float32),
            np.array(self.state[ep_ind, future_ind], dtype=np.float32)
        )
    
    def reset(self):
        self.ptr = 0
        self.ep_id = 0
        self.size = 0

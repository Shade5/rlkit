from flex_gym.flex_vec_env import set_flex_bin_path, FlexVecEnv
from autolab_core import YamlConfig
import numpy as np
from gym import spaces


class FetchReach:
	def __init__(self):
		self.cfg = YamlConfig('/home/georgejo/rlkit/examples/her/cfg/fetch_cube.yaml')
		self.numAgents = self.cfg['scene']['NumAgents'] = 1
		self.cfg['scene']['NumPerRow'] = np.sqrt(np.floor(self.numAgents))
		self.cfg['scene']['SampleInitStates'] = True
		self.cfg['scene']['InitialGrasp'] = False
		self.cfg['scene']['RelativeTarget'] = False
		self.cfg['scene']['DoDeltaPlanarControl'] = True
		self.cfg['scene']['DoGripperControl'] = True
		self.cfg['scene']['InitialGraspProbability'] = 1
		self.cfg['scene']['DoWristRollControl'] = False

		set_flex_bin_path('/home/georgejo/FlexRobotics/bin')
		self.env = FlexVecEnv(self.cfg)

		self.action_space = self.env.action_space
		self.observation_space = spaces.Dict(
			{"achieved_goal": spaces.Box(low=-np.inf, high=np.inf, shape=(3, 1), dtype=np.float32),
			 "desired_goal": spaces.Box(low=-np.inf, high=np.inf, shape=(3, 1), dtype=np.float32),
			 "observation": spaces.Box(low=-np.inf, high=np.inf, shape=(7, 1), dtype=np.float32)})

	def reset(self):
		self.env.reset()
		for i in range(20):
			obs, reward, done, info = self.step(np.array([0, 0, 0, -1]))
		return obs

	def step(self, action):
		obs, reward, done, info = self.env.step(action)
		return {'observation': obs[0], 'achieved_goal': obs[0][:3],
		        'desired_goal': obs[0][4:8]}, reward[0], done[0], {'is_success': reward[0] + 1}

	def compute_reward(self, achieved_goal, desired_goal, info):
		return self.env.compute_reward(achieved_goal[None, :], desired_goal[None, :], info)

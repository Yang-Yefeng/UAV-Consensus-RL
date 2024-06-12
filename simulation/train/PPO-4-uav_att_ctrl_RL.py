import os, sys, datetime, time
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../../")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from environment.envs.uav_att_ctrl.uav_att_ctrl_RL import uav_att_ctrl_RL, uav_param
from environment.envs.UAV.FNTSMC import fntsmc_param
from environment.envs.UAV.ref_cmd import *

from algorithm.policy_base.Proximal_Policy_Optimization2 import Proximal_Policy_Optimization2 as PPO2

from utils.classes import *
from utils.functions import *

timestep = 0
ENV = 'uav_att_ctrl_RL'
ALGORITHM = 'PPO2'

'''Parameter of the UAV'''
DT = 0.02
uav_param = uav_param()
uav_param.m = 0.8
uav_param.g = 9.8
uav_param.J = np.array([4.212e-3, 4.212e-3, 8.255e-3])
uav_param.d = 0.12
uav_param.CT = 2.168e-6
uav_param.CM = 2.136e-8
uav_param.J0 = 1.01e-5
uav_param.kr = 1e-3
uav_param.kt = 1e-3
uav_param.pos0 = np.array([0, 0, 0])
uav_param.vel0 = np.array([0, 0, 0])
uav_param.angle0 = np.array([0, 0, 0])
uav_param.pqr0 = np.array([0, 0, 0])
uav_param.dt = DT
uav_param.time_max = 10
'''Parameter of the UAV'''

'''Parameter list of the attitude controller'''
att_ctrl_param = fntsmc_param(
    k1=np.array([4., 4., 15.]).astype(float),
    k2=np.array([1., 1., 1.5]).astype(float),
    k3=np.array([0.05, 0.05, 0.05]).astype(float),
    k4=np.array([5, 4, 5]).astype(float),  # 要大
    alpha1=np.array([1.01, 1.01, 1.01]).astype(float),
    alpha2=np.array([1.01, 1.01, 1.01]).astype(float),
    dim=3,
    dt=DT
    # k1 控制滑模中 e 的占比，k3 控制滑模中 sig(e)^alpha1 的占比
    # k2-alpha1 的组合不要太大
    # k3 是观测器补偿用的，实际上观测的都很好，所以 k4 要大于0，但是非常小
    # k4-alpha2 的组合要比k3-alpha1 大，但是也别太大
)
'''Parameter list of the attitude controller'''

test_episode = []
test_reward = []
sumr_list = []


def reset_att_ctrl_param(flag: str):
    if flag == 'zero':
        att_ctrl_param.k1 = 0.01 * np.ones(3)
        att_ctrl_param.k2 = 0.01 * np.ones(3)
        att_ctrl_param.k4 = 0.01 * np.ones(3)
    elif flag == 'random':
        att_ctrl_param.k1 = np.random.random(3)
        att_ctrl_param.k2 = np.random.random(3)
        att_ctrl_param.k4 = np.random.random(3)
    else:  # optimal 手调的
        att_ctrl_param.k1 = np.array([4., 4., 15.])
        att_ctrl_param.k2 = np.array([1., 1., 1.5])
        att_ctrl_param.k4 = np.array([5, 4, 5])


class PPOActor_Gaussian(nn.Module):
    def __init__(self,
                 state_dim: int = 3,
                 action_dim: int = 3,
                 a_min: np.ndarray = 0.1 * np.ones(3),
                 a_max: np.ndarray = 100 * np.ones(3),
                 init_std: float = 0.7,
                 use_orthogonal_init: bool = True):
        super(PPOActor_Gaussian, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 32)
        self.mean_layer = nn.Linear(32, action_dim)
        # self.log_std = nn.Parameter(torch.zeros(1, action_dim))  # We use 'nn.Parameter' to train log_std automatically
        # self.log_std = nn.Parameter(np.log(init_std) * torch.ones(action_dim))  # We use 'nn.Parameter' to train log_std automatically
        self.activate_func = nn.Tanh()
        self.a_min = torch.tensor(a_min, dtype=torch.float)
        self.a_max = torch.tensor(a_max, dtype=torch.float)
        self.off = (self.a_min + self.a_max) / 2.0
        self.gain = self.a_max - self.off
        self.action_dim = action_dim
        self.std = torch.tensor(init_std, dtype=torch.float)

        if use_orthogonal_init:
            self.orthogonal_init_all()

    @staticmethod
    def orthogonal_init(layer, gain=1.0):
        nn.init.orthogonal_(layer.weight, gain=gain)
        nn.init.constant_(layer.bias, 0)

    def orthogonal_init_all(self):
        self.orthogonal_init(self.fc1)
        self.orthogonal_init(self.fc2)
        self.orthogonal_init(self.fc3)
        self.orthogonal_init(self.mean_layer, gain=0.01)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        s = self.activate_func(self.fc3(s))
        # mean = torch.tanh(self.mean_layer(s)) * self.gain + self.off
        mean = torch.relu(self.mean_layer(s))
        return mean

    def get_dist(self, s):
        mean = self.forward(s)
        # mean = torch.tensor(mean, dtype=torch.float)
        # log_std = self.log_std.expand_as(mean)
        # std = torch.exp(log_std)
        std = self.std.expand_as(mean)
        dist = Normal(mean, std)  # Get the Gaussian distribution
        # std = self.std.expand_as(mean)
        # dist = Normal(mean, std)
        return dist

    def evaluate(self, state):
        with torch.no_grad():
            t_state = torch.unsqueeze(torch.tensor(state, dtype=torch.float), 0)
            action_mean = self.forward(t_state)
        return action_mean.detach().cpu().numpy().flatten()


class PPOCritic(nn.Module):
    def __init__(self, state_dim=3, use_orthogonal_init: bool = True):
        super(PPOCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.activate_func = nn.Tanh()

        if use_orthogonal_init:
            self.orthogonal_init_all()

    @staticmethod
    def orthogonal_init(layer, gain=1.0):
        nn.init.orthogonal_(layer.weight, gain=gain)
        nn.init.constant_(layer.bias, 0)

    def orthogonal_init_all(self):
        self.orthogonal_init(self.fc1)
        self.orthogonal_init(self.fc2)
        self.orthogonal_init(self.fc3)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        v_s = self.fc3(s)
        return v_s

    def init(self, use_orthogonal_init):
        if use_orthogonal_init:
            self.orthogonal_init_all()
        else:
            self.fc1.reset_parameters()
            self.fc2.reset_parameters()
            self.fc3.reset_parameters()


if __name__ == '__main__':
    log_dir = os.path.dirname(os.path.abspath(__file__)) + '/../../datasave/log/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    simulationPath = log_dir + datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d-%H-%M-%S') + '-' + ALGORITHM + '-' + ENV + '/'
    os.mkdir(simulationPath)
    
    RETRAIN = True  # 基于之前的训练结果重新训练
    HEHE_FLAG = True
    
    env = uav_att_ctrl_RL(uav_param, att_ctrl_param)
    reset_att_ctrl_param('zero')
    env.reset_env(random_att_trajectory=False, yaw_fixed=False, new_att_ctrl_param=att_ctrl_param)
    
    env_test = uav_att_ctrl_RL(uav_param, att_ctrl_param)
    reset_att_ctrl_param('zero')
    env_test.reset_env(random_att_trajectory=False, yaw_fixed=False, new_att_ctrl_param=att_ctrl_param)
    
    reward_norm = Normalization(shape=1)
    env_msg = {'state_dim': env.state_dim, 'action_dim': env.action_dim, 'name': env.name, 'action_range': env.action_range}
    ppo_msg = {'gamma': 0.99,
               'K_epochs': 10,
               'eps_clip': 0.2,
               'buffer_size': int(env.time_max / env.dt) * 2,
               'state_dim': env_test.state_dim,
               'action_dim': env_test.action_dim,
               'a_lr': 1e-5,
               'c_lr': 1e-4,
               'set_adam_eps': True,
               'lmd': 0.95,
               'use_adv_norm': True,
               'mini_batch_size': 64,
               'entropy_coef': 0.01,
               'use_grad_clip': True,
               'use_lr_decay': True,
               'max_train_steps': int(5e6),
               'using_mini_batch': False}

    # action_std_init = 0.4  # 初始探索方差
    # min_action_std = 0.05  # 最小探索方差
    timestep = 0
    t_epoch = 0  # 当前训练次数
    test_num = 0
    
    agent = PPO2(env_msg=env_msg,
                 ppo_msg=ppo_msg,
                 actor=PPOActor_Gaussian(state_dim=env.state_dim,
                                         action_dim=env.action_dim,
                                         a_min=np.array(env.action_range)[:, 0],
                                         a_max=np.array(env.action_range)[:, 1],
                                         init_std=0.5,  # 第2次学是 0.3
                                         use_orthogonal_init=True),
                 critic=PPOCritic(state_dim=env.state_dim, use_orthogonal_init=True))
    agent.PPO2_info()

    if RETRAIN:
        print('RELOADING......')
        '''如果两次奖励函数不一样，那么必须重新初始化 critic'''
        # optPath = os.path.dirname(os.path.abspath(__file__)) + '/../../datasave/nets/att_good_1/'
        optPath = os.path.dirname(os.path.abspath(__file__)) + '/../../datasave/nets/att_good_2/'
        agent.actor.load_state_dict(torch.load(optPath + 'actor'))  # 测试时，填入测试actor网络
        agent.critic.load_state_dict(torch.load(optPath + 'critic'))
        # agent.critic.init(True)
        '''如果两次奖励函数不一样，那么必须重新初始化 critic'''
    
    while True:
        '''1. 初始化 buffer 索引和累计奖励记录'''
        buffer_index = 0
        '''1. 初始化 buffer 索引和累计奖励记录'''
        
        '''2. 收集数据'''
        while buffer_index < agent.buffer.batch_size:
            if env.is_terminal:  # 如果某一个回合结束
                reset_att_ctrl_param('zero')
                # if t_epoch % 10 == 0 and t_epoch > 0:
                print('Sumr:  ', env.sum_reward)
                sumr_list.append(env.sum_reward)
                env.reset_env(random_att_trajectory=True, yaw_fixed=False, new_att_ctrl_param=att_ctrl_param)
            else:
                env.current_state = env.next_state.copy()  # 此时相当于时间已经来到了下一拍，所以 current 和 next 都得更新
                s = env.current_state_norm(env.current_state, update=True)
                a, a_log_prob = agent.choose_action(s)
                env.get_param_from_actor(a, hehe_flag=HEHE_FLAG)
                
                rhod = env.rho_d_all[env.n]
                dot_rhod = env.dot_rho_d_all[env.n]
                dot2_rhod = env.dot2_rho_d_all[env.n]
            
                torque = env.att_control(ref=rhod, dot_ref=dot_rhod, dot2_ref=dot2_rhod, att_only=True)
                env.step_update([torque[0], torque[1], torque[2]])
                if env.is_terminal and (env.terminal_flag != 1):
                    success = 1.0
                else:
                    success = 0.
                agent.buffer.append(s=s,
                                    a=a,  # a
                                    log_prob=a_log_prob,  # a_lp
                                    # r=env.reward,							# r
                                    r=reward_norm(env.reward),  # 这里使用了奖励归一化
                                    s_=env.next_state_norm(env.next_state, update=True),
                                    done=1.0 if env.is_terminal else 0.0,  # done
                                    success=success,  # 固定时间内，不出界，就是 success
                                    index=buffer_index  # index
                                    )
                buffer_index += 1
        '''2. 收集数据'''
        
        '''3. 开始学习'''
        print('~~~~~~~~~~ Training Start~~~~~~~~~~')
        print('Train Epoch: {}'.format(t_epoch))
        timestep += ppo_msg['buffer_size']
        agent.learn(timestep, buf_num=1)
        agent.cnt += 1
        print('~~~~~~~~~~  Training End ~~~~~~~~~~')
        '''3. 开始学习'''
        
        '''4. 每学习 10 次，测试一下'''
        if t_epoch % 10 == 0 and t_epoch > 0:
            n = 1
            print('    Training pause......')
            print('    Testing...')
            for i in range(n):
                reset_att_ctrl_param('zero')
                env_test.reset_env(random_att_trajectory=False, yaw_fixed=False, new_att_ctrl_param=att_ctrl_param)
                while not env_test.is_terminal:
                    _a = agent.evaluate(env.current_state_norm(env_test.current_state, update=False))
                    env_test.get_param_from_actor(_a, hehe_flag=HEHE_FLAG)  # 将控制器参数更新
                    _rhod = env_test.rho_d_all[env_test.n]
                    _dot_rhod = env_test.dot_rho_d_all[env_test.n]
                    _dot2_rhod = env_test.dot2_rho_d_all[env_test.n]
                    _torque = env_test.att_control(_rhod, _dot_rhod, _dot2_rhod, True)
                    env_test.step_update([_torque[0], _torque[1], _torque[2]])
                    
                    env_test.visualization()
                test_num += 1
                test_reward.append(env_test.sum_reward)
                print('    Evaluating %.0f | Reward: %.2f ' % (i, env_test.sum_reward))
            pd.DataFrame({'reward': test_reward}).to_csv(simulationPath + 'test_record.csv')
            pd.DataFrame({'sumr_list': sumr_list}).to_csv(simulationPath + 'sumr_list.csv')
            print('    Testing finished...')
            print('    Go back to training...')
        '''4. 每学习 10 次，测试一下'''
        
        '''5. 每学习 250 次，减小一次探索概率'''
        if t_epoch % 250 == 0 and t_epoch > 0:
            if agent.actor.std > 0.1:
                agent.actor.std -= 0.05
        '''5. 每学习 250 次，减小一次探索概率'''
        
        '''6. 每学习 50 次，保存一下 policy'''
        if t_epoch % 50 == 0 and t_epoch > 0:
            # 	average_test_r = agent.agent_evaluate(5)
            test_num += 1
            print('...check point save...')
            temp = simulationPath + 'trainNum_{}/'.format(t_epoch)
            os.mkdir(temp)
            time.sleep(0.01)
            agent.save_ac(msg=''.format(t_epoch), path=temp)
            env.save_state_norm(temp)
        '''6. 每学习 50 次，保存一下 policy'''
        
        t_epoch += 1

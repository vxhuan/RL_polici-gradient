import gymnasium as gym
import matplotlib.pyplot as plt
import pickle
import numpy as np

def train(episode,render=False):
    env=gym.make("FrozenLake-v1",map_name="8x8",is_slippery=False,render_mode="human" if render else None )
    reward_per_episode=[]

    lr=0.99
    discounted=0.99
    epsilon=1
    min_epsilon=0.001
    epsilon_decay=0.0001

    q=np.zeros((env.observation_space.n,env.action_space.n))
    
    for i in range(1,episode+1):
        reward_current=0
        done=False
        state=env.reset()[0]
        while not done :
            if np.random.uniform(0,1)<epsilon :
                action=env.action_space.sample()
            else :
                action=np.argmax(q[state,:])
            
            new_state , reward , terminated , truncated , infor=env.step(action)
            q[state,action]=q[state,action]+lr*(reward+discounted*np.max(q[new_state,:])-q[state,action])
            reward_current+=reward
            state=new_state
            
            done=terminated or truncated
        reward_per_episode.append(reward_current)
        epsilon=epsilon-epsilon_decay
        
    
    with open("frozen_lake8x8.pkl","wb") as f:
        pickle.dump(q,f)
    
    smooth_average_reward=np.convolve(np.array(reward_per_episode),np.ones(1000)/1000,mode="valid")  

    plt.plot(reward_per_episode,label="raw_reward")
    plt.plot(range(1000-1,len(smooth_average_reward)+1000-1),smooth_average_reward,label="smoothed_reward")
    plt.xlabel("episode")
    plt.ylabel("reward")
    plt.grid(True)
    plt.legend()
    plt.show()
def test(episode):
    env=gym.make("FrozenLake-v1",map_name="8x8",is_slippery=False,render_mode="human"  )
    with open("frozen_lake8x8.pkl","rb") as f:
        q=pickle.load(f)
    for i in range(1,episode+1):
        reward_current=0
        done=False
        state=env.reset()[0]
        while not done :
           
          
            action=np.argmax(q[state,:])
            
            new_state , reward , terminated , truncated , infor=env.step(action)
            reward_current+=reward
            state=new_state
            
            done=terminated or truncated



#train(10000)
test(3)


























        

        

        




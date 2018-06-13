import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        # Computing reward based on difference betwen simulated heigh and target height
        reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()

        # Add a reward for vertical velocity
        
        vx = self.sim.v[0]
        vy = self.sim.v[1]
        vz = self.sim.v[2] 
        
        if vz > 0:
            reward += 1
        
        if vz > vx or vz > vy:
            reward += 1
  
        # Penalize if horizontal velocities are higher than that of vectical velocity
        if vx > vz or vy > vz:
            reward -= 1
            
        # Add a reward for vertical position
        
        x = self.sim.pose[0]
        y = self.sim.pose[1]
        z = self.sim.pose[2]  
        
        if z > 0:
            reward += 1
        
        if z > x or z > y:
            reward += 1
 
        # Penalize if horizontal position is more than that of vectical position
        if x > z or y > z:
            reward -= 1

        # Clip the rewards between -1 and 1
        reward = np.clip(reward, -1, 1)
        
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state
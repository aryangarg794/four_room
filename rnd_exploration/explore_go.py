from hashlib import sha1
import numpy as np
from collections import defaultdict

class Dataset:
    """Basic dataset class. Makes it easier to pickle the file. 
    """
    
    def __init__(self):
        self.X = []
        self.Y = []
        self.states = []
        
    def add(self, x, y, state):
        """Add a data sample to the dataset"""
        self.X.append(x)
        self.Y.append(y)
        self.states.append(state)
        
    def sort(self):
        """Returns a dictionary which for each q-vector returns the  
        corresponding states. 
        """             
        binned_dict = defaultdict(list)
        
        X = np.array(self.X)
        Y = np.array(self.Y)
        
        for q in np.unique(Y, axis=0):
            mask = np.all(Y == q, axis=1)
            binned_dict[tuple(q)] = X[mask]
        return binned_dict
    
    def __getitem__(self, index):
        return (self.X[index], self.Y[index])
    
    def __len__(self):
        return len(self.X)
    

class Transition:
    
    def __init__(
        self, 
        state: np.ndarray, 
        q_value: np.ndarray
    ):  
        self.state = state
        self.q_value = q_value
        
    def __eq__(self, other):
        return bool(np.all(self.state == other.state) and np.all(self.q_value == other.q_value))
    
    def __hash__(self):
        state_hash = int(sha1(self.state.flatten()).hexdigest(), 16)
        q_value_hash = int(sha1(self.q_value.flatten()).hexdigest(), 16)
        return hash((state_hash, q_value_hash))
    
class Trajectory: 
    
    def __init__(
        self
    ):
        self.transitions = []
        self.unique_transitions = set([])
    
    def add(self, transition: Transition):
        self.transitions.append(transition)
        self.unique_transitions.add(transition)
    
    def __eq__(self, other):
        if len(self.transitions) != len(other.transitions):
            return False
        return all(t1 == t2 for t1, t2 in zip(self.transitions, other.transitions))
    
    def __hash__(self):
        t_matrix_state = np.array([t.state for t in self.transitions]).flatten() 
        t_matrix_q = np.array([t.state for t in self.transitions]).flatten() 
        state_hash = int(sha1(t_matrix_state).hexdigest(), 16)
        q_value_hash = int(sha1(t_matrix_q).hexdigest(), 16)
        return hash((state_hash, q_value_hash))

    def uniqueness(self, other):
        intersection = self.unique_transitions.intersection(other.unique_transitions)
        return len(intersection) / len(self.transitions) # compares how unique is our trajectory compared to other
    
    def __iter__(self):
        return iter([(t.state, t.q_value) for t in self.transitions])
    
class ExploreGoDataset(Dataset):
    
    def __init__(self):
        super().__init__()
        
        self.trajectories = []
        self.unique_trajectories = set([])
        self.trans = []
        self.unique_trans = set([])
        self.current_traj = Trajectory()
        
    def wrap_trajectory(self):
        self.trajectories.append(self.current_traj)
        self.unique_trajectories.add(self.current_traj)
        for state, q in self.current_traj:
            self.add(state, q, None) # not storing the state
        self.current_traj = Trajectory()
        
    def reset(self): 
        self.current_traj = Trajectory()
    
    def add_traj(self, obs: np.ndarray, q_value: np.ndarray, state: np.ndarray):
        transition = Transition(obs, q_value)
        self.current_traj.add(transition)
        self.unique_trans.add(transition)
        
    def add_trans(self, obs: np.ndarray, q_value: np.ndarray):
        transition = Transition(obs, q_value)
        self.trans.append(transition)
        self.unique_trans.add(transition)
        
    def traj_uniqueness(self, trajectory: Trajectory): 
        unique = 0
        for traj in self.unique_trajectories:
            unique = max(unique, trajectory.uniqueness(traj))
        return unique  
    
    @property
    def ratio_unique_trans(self):
        return len(self.unique_trans) / len(self.trans)
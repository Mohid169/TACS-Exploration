import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class CartPoleParams:
    """Morphological parameters for cart-pole system."""
    m_cart: float  # Cart mass (kg)
    m_pole: float  # Pole mass (kg)
    length: float  # Pole length (m)
    I: float       # Pole moment of inertia (kg⋅m²)
    F_max: float   # Maximum actuator force (N)
    L: float       # Track length (m)
    g: float = 9.81  # Gravitational acceleration (m/s²)
    
    def validate(self) -> bool:
        """Check if parameters are physically valid."""
        # All parameters must be positive
        if any(v <= 0 for v in [self.m_cart, self.m_pole, self.length, 
                                self.I, self.F_max, self.L]):
            return False
        
        # Inertia must be reasonable for a rod-like object
        I_rod = (1/3) * self.m_pole * self.length**2
        if not (0.5 * I_rod <= self.I <= 2.0 * I_rod):
            return False
        return True
    
    def to_array(self) -> np.ndarray:
        """Convert to parameter array for evolutionary optimization."""
        return np.array([
            self.m_cart, self.m_pole, self.length,
            self.I, self.F_max, self.L
        ])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'CartPoleParams':
        """Create CartPoleParams from array."""
        return cls(
            m_cart=float(arr[0]),
            m_pole=float(arr[1]),
            length=float(arr[2]),
            I=float(arr[3]),
            F_max=float(arr[4]),
            L=float(arr[5])
        )

class CartPoleDynamics:
    """Nonlinear cart-pole dynamics with constraint checking."""
    
    def __init__(self, params: CartPoleParams, dt: float = 0.01):
        """
        Initialize dynamics simulator.
        
        Args:
            params: Cart-pole morphological parameters
            dt: Integration time step (seconds)
        """
        if not params.validate():
            raise ValueError("Invalid cart-pole parameters")
        
        self.params = params
        self.dt = dt
        self.theta_max = np.pi / 6  # 30 degrees stability bound
        
    def dynamics(self, state: np.ndarray, u: float) -> np.ndarray:
        """
        Compute continuous-time dynamics: ẋ = f(x, u)
        
        Uses nonlinear equations derived from Lagrangian mechanics.
        
        Args:
            state: [p, p_dot, theta, theta_dot]
            u: control input (horizontal force on cart)
            
        Returns:
            state_dot: time derivative of state
        """
        p, p_dot, theta, theta_dot = state
        
        # Apply actuation limits
        u = np.clip(u, -self.params.F_max, self.params.F_max)
        
        # Precompute trigonometric functions
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        
        # Mass and length terms
        m_total = self.params.m_cart + self.params.m_pole
        ml = self.params.m_pole * self.params.length
        
        #Standard cart-pole dynamics equations from Lagrangian mechanics
        # See: https://sharpneat.sourceforge.io/research/cart-pole/cart-pole-equations.html
        # Also: Underactuated Robotics (Tedrake)
        
        # Common denominator term
        denom = m_total - self.params.m_pole * cos_theta**2
        
        # Cart acceleration
        p_ddot = (u + ml * theta_dot**2 * sin_theta - 
                  self.params.m_pole * self.params.g * sin_theta * cos_theta) / denom
        
        # Pole angular acceleration  
        theta_ddot = (-u * cos_theta - ml * theta_dot**2 * sin_theta * cos_theta +
                      m_total * self.params.g * sin_theta) / (self.params.length * denom)
        
        return np.array([p_dot, p_ddot, theta_dot, theta_ddot])
        
    
    def step(self, state: np.ndarray, u: float) -> Tuple[np.ndarray, bool]:
        """
        Integrate dynamics forward one time step using RK4.
        
        Args:
            state: current state
            u: control input
            
        Returns:
            next_state: state after dt
            done: True if constraint violated
        """
        # Fourth-order Runge-Kutta integration
        k1 = self.dynamics(state, u)
        k2 = self.dynamics(state + 0.5 * self.dt * k1, u)
        k3 = self.dynamics(state + 0.5 * self.dt * k2, u)
        k4 = self.dynamics(state + self.dt * k3, u)
        
        next_state = state + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        # Check for constraint violations
        done = self._check_constraints(next_state)
        
        return next_state, done
    
    def _check_constraints(self, state: np.ndarray) -> bool:
        """
        Check if state violates any constraints.
        
        Args:
            state: state to check
            
        Returns:
            True if constraints violated
        """
        p, _, theta, _ = state
        
        # Track position limits
        if abs(p) > self.params.L / 2:
            return True
        
        # Pole angle stability bounds
        if abs(theta) > self.theta_max:
            return True
        
        # Numerical divergence check
        if not np.all(np.isfinite(state)):
            return True
        
        return False
    
    def compute_jacobians(self, state: np.ndarray, u: float,
                         eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute linearization Jacobians A = ∂f/∂x, B = ∂f/∂u.
        
        Uses central finite differences for numerical differentiation.
        
        Args:
            state: operating point state
            u: operating point control
            eps: finite difference step size
            
        Returns:
            A: 4x4 state Jacobian matrix
            B: 4x1 input Jacobian matrix
        """
        n_states = 4
        
        # Compute A = ∂f/∂x using central differences
        A = np.zeros((n_states, n_states))
        for i in range(n_states):
            state_plus = state.copy()
            state_minus = state.copy()
            state_plus[i] += eps
            state_minus[i] -= eps
            
            A[:, i] = (self.dynamics(state_plus, u) - 
                      self.dynamics(state_minus, u)) / (2 * eps)
        
        # Compute B = ∂f/∂u using central differences
        f_plus = self.dynamics(state, u + eps)
        f_minus = self.dynamics(state, u - eps)
        B = ((f_plus - f_minus) / (2 * eps)).reshape(-1, 1)
        
        return A, B
    
    def discretize_jacobians(self, A_cont: np.ndarray, B_cont: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Discretize continuous-time Jacobians using first-order approximation.
        
        Args:
            A_cont: continuous-time A matrix
            B_cont: continuous-time B matrix
            
        Returns:
            A_disc: discrete-time A matrix (A_d ≈ I + A_c * dt)
            B_disc: discrete-time B matrix (B_d ≈ B_c * dt)
        """
        A_disc = np.eye(4) + A_cont * self.dt
        B_disc = B_cont * self.dt
        
        return A_disc, B_disc
use pyo3::prelude::*;
use rand::Rng;

const GRAVITY: f32 = 9.8;
const MASS_CART: f32 = 1.0;
const MASS_POLE: f32 = 0.1;
const TOTAL_MASS: f32 = MASS_CART + MASS_POLE;
const LENGTH: f32 = 0.5;
const POLE_MASS_LENGTH: f32 = MASS_POLE * LENGTH;
const FORCE_MAG: f32 = 10.0;
const TAU: f32 = 0.02;
const THETA_THRESHOLD_RADIANS: f32 = 12.0 * 2.0 * std::f32::consts::PI / 360.0;
const X_THRESHOLD: f32 = 2.4;

/// Cart position, cart velocity, pole angle, pole angular velocity.
type State = (f32, f32, f32, f32);

/// Rust implementation of the cartpole environment.
/// Based on the OpenAI gym implementation.
#[pyclass]
pub struct CartpoleEnv {
    pub state: State,
}

#[pymethods]
impl CartpoleEnv {
    #[new]
    pub fn new() -> CartpoleEnv {
        let mut env = CartpoleEnv {
            state: (0.0, 0.0, 0.0, 0.0),
        };
        env.reset();
        env
    }

    pub fn step(&mut self, action: u32) -> (State, f32, bool) {
        let (x, x_dot, theta, theta_dot) = self.state;
        let force = if action == 1 { FORCE_MAG } else { -FORCE_MAG };
        let cos_theta = theta.cos();
        let sin_theta = theta.sin();

        let temp = (force + POLE_MASS_LENGTH * theta_dot.powi(2) * sin_theta) / TOTAL_MASS;
        let theta_acc = (GRAVITY * sin_theta - cos_theta * temp)
            / (LENGTH * (4.0 / 3.0 - MASS_POLE * cos_theta.powi(2) / TOTAL_MASS));
        let xacc = temp - POLE_MASS_LENGTH * theta_acc * cos_theta / TOTAL_MASS;

        let x = x + TAU * x_dot;
        let x_dot = x_dot + TAU * xacc;
        let theta = theta + TAU * theta_dot;
        let theta_dot = theta_dot + TAU * theta_acc;

        self.state = (x, x_dot, theta, theta_dot);
        let terminated = !(-X_THRESHOLD..=X_THRESHOLD).contains(&x)
            || !(-THETA_THRESHOLD_RADIANS..=THETA_THRESHOLD_RADIANS).contains(&theta);
        let reward = 1.0;

        (self.state, reward, terminated)
    }

    pub fn reset(&mut self) -> State {
        let low = -0.05;
        let high = 0.05;
        let mut rng = rand::thread_rng();
        self.state = (
            rng.gen_range(low..high),
            rng.gen_range(low..high),
            rng.gen_range(low..high),
            rng.gen_range(low..high),
        );
        self.state
    }
}

impl Default for CartpoleEnv {
    fn default() -> Self {
        Self::new()
    }
}

#[pymodule]
fn rl_template_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<CartpoleEnv>()?;
    Ok(())
}
